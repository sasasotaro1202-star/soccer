#!/usr/bin/env python3
from pathlib import Path

P=Path('backtest.py')
MARKER='SOCCER_UNDERSTAT_FAIL_CLOSED_V1'
if not P.exists(): raise SystemExit('backtest.py missing')
s=P.read_text(encoding='utf-8')
if MARKER in s: raise SystemExit(0)
old='''def build_understat_index(matches):
    idx = {}
    cov = []
    for (code, year), _ in matches.groupby(["League", "SeasonStart"]):
        if left() < 25:
            break
        rows = load_understat_season(code, int(year))
        matched = 0
        for z in rows:
            try:
                h = z.get("h", {}) if isinstance(z.get("h"), dict) else {}
                a = z.get("a", {}) if isinstance(z.get("a"), dict) else {}
                home = team_name(h.get("title") or z.get("home_team"))
                away = team_name(a.get("title") or z.get("away_team"))
                dt = pd.to_datetime(z.get("datetime") or z.get("date"), errors="coerce")
                hxg = sf(h.get("xG") or h.get("xg"))
                axg = sf(a.get("xG") or a.get("xg"))
                if pd.isna(dt) or not (np.isfinite(hxg) and np.isfinite(axg)):
                    continue
                idx[(code, dt.date().isoformat(), home, away)] = {"hxg": hxg, "axg": axg}
                matched += 1
            except Exception:
                continue
        cov.append({"League": code, "SeasonStart": int(year), "Source": "Understat", "Available": len(rows), "Matched": matched})
    return idx, cov
'''
new='''def build_understat_index(matches):
    idx = {}
    cov = []
    for (code, year), group in matches.groupby(["League", "SeasonStart"]):
        if code not in UNDERSTAT_LEAGUE:
            continue
        if left() < 25:
            raise RuntimeError("UNDERSTAT_FAIL_CLOSED: runtime budget exhausted before required Understat coverage")
        rows = load_understat_season(code, int(year))
        if not rows:
            raise RuntimeError(f"UNDERSTAT_FAIL_CLOSED: no Understat payload for {code} {int(year)}")
        matched = 0
        for z in rows:
            try:
                h = z.get("h", {}) if isinstance(z.get("h"), dict) else {}
                a = z.get("a", {}) if isinstance(z.get("a"), dict) else {}
                home = team_name(h.get("title") or z.get("home_team"))
                away = team_name(a.get("title") or z.get("away_team"))
                dt = pd.to_datetime(z.get("datetime") or z.get("date"), errors="coerce")
                hxg = sf(h.get("xG") or h.get("xg"))
                axg = sf(a.get("xG") or a.get("xg"))
                if pd.isna(dt) or not (np.isfinite(hxg) and np.isfinite(axg)):
                    continue
                idx[(code, dt.date().isoformat(), home, away)] = {"hxg": hxg, "axg": axg}
                matched += 1
            except Exception:
                continue
        expected = len(group)
        coverage = matched / max(1, len(rows))
        print(f"[UNDERSTAT] {code} {int(year)} available={len(rows)} matched={matched} coverage={coverage:.3f} expected_matches={expected}")
        if matched == 0:
            raise RuntimeError(f"UNDERSTAT_FAIL_CLOSED: zero matched xG rows for {code} {int(year)}")
        cov.append({"League": code, "SeasonStart": int(year), "Source": "Understat", "Available": len(rows), "Matched": matched, "Coverage": coverage})
    return idx, cov

# SOCCER_UNDERSTAT_FAIL_CLOSED_V1
'''
if old not in s: raise SystemExit('expected build_understat_index() block not found; refusing unsafe patch')
P.write_text(s.replace(old,new,1),encoding='utf-8')
print('[PATCH] Understat fail-closed runtime hardening applied')
