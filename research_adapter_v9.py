"""Research Engine bridge for the legacy V9 soccer engine.

This file does not reimplement V9 feature/model functions.  It orchestrates
calls to the existing functions in backtest.py so the Research Engine can run
the legacy engine as a named baseline without changing its implementation.

Safety boundary:
- the caller must provide PIT-verified rows;
- no source-availability timestamp is invented here;
- the adapter only predicts from state accumulated before the current row;
- post-match state updates happen only after the prediction is emitted.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

import backtest as v9


class V9ResearchAdapter:
    """Stateful wrapper around the concrete V9 functions in ``backtest.py``."""

    source_version = "V9"

    def __init__(self, random_state: int | None = None):
        if random_state is not None:
            v9.RANDOM_STATE = int(random_state)
        v9.make_feature_names()
        self.leagues = {code: v9.League() for code in v9.LEAGUES}
        self.histories = defaultdict(list)
        self.bundles: dict[str, Any] = {}
        self.blends: dict[str, tuple[float, float, float]] = {}
        self.players = defaultdict(v9.Player)
        self.last_dates: dict[str, Any] = {}
        self.h2h: dict[Any, list[int]] = {}
        self.global_elo: dict[str, float] = {}
        self.active_seasons: dict[str, int] = {}
        self.code_counts = defaultdict(int)

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
        """Predict one match using V9 pre-match state.

        ``row`` must contain the same fields expected by the legacy V9 helpers.
        The method intentionally does not update team/player state; call
        ``observe_match`` only after the prediction has been evaluated.
        """
        if pit_verified is not True:
            raise ValueError("V9 baseline requires PIT-verified input")
        code = str(row["League"])
        year = int(float(row["SeasonStart"]))
        self._ensure_competition(code, year)
        L = self.leagues[code]
        h = L.team(row["HomeTeam"])
        a = L.team(row["AwayTeam"])
        x = v9.make_features(row, h, a, L, self.last_dates, self.h2h, self.players, self.global_elo)
        market = v9.closing_market(row)
        score_p, top_scores, lh, la = v9.score_model(h, a, L)

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
            "match_id": str(row.get("match_id", f"{code}|{row['Date']}|{row['HomeTeam']}|{row['AwayTeam']}")),
            "probabilities": {"H": float(final[0]), "D": float(final[1]), "A": float(final[2])},
            "score_candidates": [
                {"home": int(hg), "away": int(ag), "probability": float(p)}
                for hg, ag, p in top_scores
            ],
            "metadata": {
                "legacy_source": "backtest.py",
                "source_version": self.source_version,
                "ml": [float(z) for z in ml],
                "market": [float(z) for z in market[:3]],
                "poisson": [float(z) for z in score_p],
                "lambda_home": float(lh),
                "lambda_away": float(la),
                "blend_weights": [float(lw), float(mw), float(sw)],
                "validation_logloss": dict(bundle.get("validation_logloss", {})) if bundle else {},
                "understat_used": understat is not None,
            },
        }

    def observe_match(
        self,
        row: Mapping[str, Any],
        *,
        understat: Mapping[str, float] | None = None,
        player_updates: list[Mapping[str, Any]] | None = None,
    ) -> None:
        """Apply post-match state after a prediction has been emitted/evaluated."""
        code = str(row["League"])
        year = int(float(row["SeasonStart"]))
        self._ensure_competition(code, year)
        L = self.leagues[code]
        h = L.team(row["HomeTeam"])
        a = L.team(row["AwayTeam"])
        ah, aa = int(row["FTHG"]), int(row["FTAG"])
        actual = self._result_code(ah, aa)

        hs = v9.sf(row.get("HS")); hst = v9.sf(row.get("HST")); hc = v9.sf(row.get("HC"))
        ass = v9.sf(row.get("AS")); ast = v9.sf(row.get("AST")); ac = v9.sf(row.get("AC"))
        hstats = {"shots": hs, "sot": hst, "corners": hc}
        astats = {"shots": ass, "sot": ast, "corners": ac}
        if understat is not None:
            h.update(ah, aa, 3 if actual == 0 else 1 if actual == 1 else 0, "H", hstats, understat.get("hxg", np.nan), understat.get("axg", np.nan))
            a.update(aa, ah, 0 if actual == 0 else 1 if actual == 1 else 3, "A", astats, understat.get("axg", np.nan), understat.get("hxg", np.nan))
        else:
            h.update(ah, aa, 3 if actual == 0 else 1 if actual == 1 else 0, "H", hstats)
            a.update(aa, ah, 0 if actual == 0 else 1 if actual == 1 else 3, "A", astats)

        expected = v9.sigmoid((h.elo + 55 - a.elo) / 400)
        actual_h = 1 if actual == 0 else .5 if actual == 1 else 0
        delta = 18 * (1 + np.log1p(max(1, abs(ah - aa)))) * (actual_h - expected)
        h.elo += delta; a.elo -= delta; L.update(ah, aa)

        ge_h = self.global_elo.get(row["HomeTeam"], 1500.0)
        ge_a = self.global_elo.get(row["AwayTeam"], 1500.0)
        ge_expected = v9.sigmoid((ge_h + 45 - ge_a) / 400)
        ge_delta = 14 * (1 + .25 * np.log1p(max(1, abs(ah - aa)))) * (actual_h - ge_expected)
        self.global_elo[row["HomeTeam"]] = ge_h + ge_delta
        self.global_elo[row["AwayTeam"]] = ge_a - ge_delta

        pair = tuple(sorted([row["HomeTeam"], row["AwayTeam"]]))
        pair_actual = actual if row["HomeTeam"] == pair[0] else (2 - actual if actual in (0, 2) else 1)
        self.h2h.setdefault((code, pair), []).append(pair_actual)
        self.last_dates[row["HomeTeam"]] = row["DateParsed"]
        self.last_dates[row["AwayTeam"]] = row["DateParsed"]

        if player_updates:
            for p in player_updates:
                team = row["HomeTeam"] if p.get("side") == "H" else row["AwayTeam"]
                self.players[(team, p["name"])].update(p, row["DateParsed"].date().isoformat())

        self.histories[code].append({"x": v9.make_features(row, h, a, L, self.last_dates, self.h2h, self.players, self.global_elo), "y": actual})
        self.code_counts[code] += 1
