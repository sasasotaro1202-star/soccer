"""Concrete bridge for running the existing V9 engine under Research Engine control.

No V9 feature/model implementation is copied here. This adapter calls the
existing functions in backtest.py and separates pre-match prediction from
post-match observation so the current match cannot enter its own features.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np
import backtest as v9


class V9ResearchAdapter:
    source_version = "V9"

    def __init__(self, random_state: int | None = None):
        if random_state is not None:
            v9.RANDOM_STATE = int(random_state)
        v9.make_feature_names()
        self.leagues = {code: v9.League() for code in v9.LEAGUES}
        self.histories = defaultdict(list)
        self.bundles = {}
        self.blends = {}
        self.players = defaultdict(v9.Player)
        self.last_dates = {}
        self.h2h = {}
        self.global_elo = {}
        self.active_seasons = {}
        self.code_counts = defaultdict(int)
        self._pending_features = {}

    def _ensure_competition(self, code: str, year: int):
        if code not in self.leagues:
            self.leagues[code] = v9.League()
        if self.active_seasons.get(code) != year:
            if code in self.active_seasons:
                hist = self.histories[code]
                if len(hist) >= 360:
                    self.blends[code] = v9.optimize_blend(hist)
            self.leagues[code].new_season()
            self.active_seasons[code] = year
            hist = self.histories[code]
            bundle = v9.fit_ensemble(hist) if len(hist) >= v9.MIN_TRAIN else None
            if bundle is not None:
                self.bundles[code] = bundle

    @staticmethod
    def _result_code(home_goals: int, away_goals: int) -> int:
        return 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2

    def predict(self, row: Mapping[str, Any], pit_verified: bool, understat: Mapping[str, float] | None = None) -> dict[str, Any]:
        if pit_verified is not True:
            raise ValueError("V9 baseline requires PIT-verified input")
        code = str(row["League"])
        year = int(float(row["SeasonStart"]))
        self._ensure_competition(code, year)
        league = self.leagues[code]
        home = league.team(row["HomeTeam"])
        away = league.team(row["AwayTeam"])

        # This is the exact V9 feature function and is evaluated before any
        # observation of the current match.
        x = v9.make_features(row, home, away, league, self.last_dates, self.h2h, self.players, self.global_elo)
        match_id = str(row.get("match_id", f"{code}|{row['Date']}|{row['HomeTeam']}|{row['AwayTeam']}"))
        self._pending_features[match_id] = np.asarray(x, dtype=float)

        market = v9.closing_market(row)
        score_p, top_scores, lh, la = v9.score_model(home, away, league)
        hist = self.histories[code]
        bundle = self.bundles.get(code)
        if len(hist) >= v9.MIN_TRAIN:
            if bundle is None or self.code_counts[code] % v9.RETRAIN_EVERY == 0:
                bundle = v9.fit_ensemble(hist)
                if bundle is not None:
                    self.bundles[code] = bundle
            ml = v9.pred_ml(bundle, x)
        else:
            ml = v9.norm3(.60 * market[:3] + .40 * score_p)
        lw, mw, sw = self.blends.get(code, (.56, .28, .16))
        final = v9.norm3(lw * ml + mw * market[:3] + sw * score_p)
        return {
            "match_id": match_id,
            "probabilities": {"H": float(final[0]), "D": float(final[1]), "A": float(final[2])},
            "score_candidates": [{"home": int(hg), "away": int(ag), "probability": float(p)} for hg, ag, p in top_scores],
            "metadata": {
                "legacy_source": "backtest.py",
                "source_version": self.source_version,
                "ml": [float(z) for z in ml],
                "market": [float(z) for z in market[:3]],
                "poisson": [float(z) for z in score_p],
                "lambda_home": float(lh), "lambda_away": float(la),
                "blend_weights": [float(lw), float(mw), float(sw)],
                "validation_logloss": dict(bundle.get("validation_logloss", {})) if bundle else {},
                "understat_used": understat is not None,
            },
        }

    def observe_match(self, row: Mapping[str, Any], *, understat: Mapping[str, float] | None = None, player_updates: list[Mapping[str, Any]] | None = None) -> None:
        """Apply realized data only after the current prediction is complete."""
        code = str(row["League"])
        year = int(float(row["SeasonStart"]))
        self._ensure_competition(code, year)
        league = self.leagues[code]
        home = league.team(row["HomeTeam"])
        away = league.team(row["AwayTeam"])
        ah, aa = int(row["FTHG"]), int(row["FTAG"])
        actual = self._result_code(ah, aa)

        hstats = {"shots": v9.sf(row.get("HS")), "sot": v9.sf(row.get("HST")), "corners": v9.sf(row.get("HC"))}
        astats = {"shots": v9.sf(row.get("AS")), "sot": v9.sf(row.get("AST")), "corners": v9.sf(row.get("AC"))}
        if understat is not None:
            home.update(ah, aa, 3 if actual == 0 else 1 if actual == 1 else 0, "H", hstats, understat.get("hxg", np.nan), understat.get("axg", np.nan))
            away.update(aa, ah, 0 if actual == 0 else 1 if actual == 1 else 3, "A", astats, understat.get("axg", np.nan), understat.get("hxg", np.nan))
        else:
            home.update(ah, aa, 3 if actual == 0 else 1 if actual == 1 else 0, "H", hstats)
            away.update(aa, ah, 0 if actual == 0 else 1 if actual == 1 else 3, "A", astats)

        expected = v9.sigmoid((home.elo + 55 - away.elo) / 400)
        actual_h = 1 if actual == 0 else .5 if actual == 1 else 0
        delta = 18 * (1 + np.log1p(max(1, abs(ah - aa)))) * (actual_h - expected)
        home.elo += delta; away.elo -= delta; league.update(ah, aa)

        ge_h = self.global_elo.get(row["HomeTeam"], 1500.0); ge_a = self.global_elo.get(row["AwayTeam"], 1500.0)
        ge_expected = v9.sigmoid((ge_h + 45 - ge_a) / 400)
        ge_delta = 14 * (1 + .25 * np.log1p(max(1, abs(ah - aa)))) * (actual_h - ge_expected)
        self.global_elo[row["HomeTeam"]] = ge_h + ge_delta; self.global_elo[row["AwayTeam"]] = ge_a - ge_delta

        pair = tuple(sorted([row["HomeTeam"], row["AwayTeam"]]))
        pair_actual = actual if row["HomeTeam"] == pair[0] else (2 - actual if actual in (0, 2) else 1)
        self.h2h.setdefault((code, pair), []).append(pair_actual)
        self.last_dates[row["HomeTeam"]] = row["DateParsed"]; self.last_dates[row["AwayTeam"]] = row["DateParsed"]

        if player_updates:
            for p in player_updates:
                team = row["HomeTeam"] if p.get("side") == "H" else row["AwayTeam"]
                self.players[(team, p["name"])].update(p, row["DateParsed"].date().isoformat())

        match_id = str(row.get("match_id", f"{code}|{row['Date']}|{row['HomeTeam']}|{row['AwayTeam']}"))
        x = self._pending_features.pop(match_id, None)
        if x is None:
            raise ValueError("Missing pre-match feature snapshot for observed match")
        self.histories[code].append({"x": x, "y": actual})
        self.code_counts[code] += 1
