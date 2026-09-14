#!/usr/bin/env python3
from pathlib import Path

P=Path('backtest.py')
MARKER='SOCCER_UNDERSTAT_GRACEFUL_V2'
if not P.exists(): raise SystemExit('backtest.py missing')
s=P.read_text(encoding='utf-8')
if MARKER in s: raise SystemExit(0)
start=s.find('def build_understat_index(matches):')
if start < 0: raise SystemExit('build_understat_index() not found; refusing unsafe patch')
# Support both the original fail-closed source and the current unmarked baseline.
end=s.find('\n\n# SOCCER_UNDERSTAT_FAIL_CLOSED_V1', start)
if end < 0:
    end=s.find('\n\n# =========================\n# TEAM STATE', start)
if end < 0: raise SystemExit('Understat function boundary not found; refusing unsafe patch')
new='''def build_understat_index(matches):
    idx = {}
    cov = []
    for (code, year), group in matches.groupby(["League", "SeasonStart"]):
        if code not in UNDERSTAT_LEAGUE:
            continue
        if left() < 25:
            cov.append({"League": code, "SeasonStart": int(year), "Source": "Understat", "Available": 0, "Matched": 0, "Status": "TIME_BUDGET"})
            continue
        rows = load_understat_season(code, int(year))
        if not rows:
            # Understat is an optional enrichment source. Missing historical coverage
            # must degrade to CORE_ONLY rather than aborting the entire backtest.
            print(f"[UNDERSTAT] {code} {int(year)} unavailable -> CORE_ONLY fallback")
            cov.append({"League": code, "SeasonStart": int(year), "Source": "Understat", "Available": 0, "Matched": 0, "Status": "UNAVAILABLE"})
            continue
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
        coverage = matched / max(1, expected)
        status = "OK" if matched > 0 else "NO_MATCH"
        print(f"[UNDERSTAT] {code} {int(year)} available={len(rows)} matched={matched} coverage={coverage:.3f} expected_matches={expected} status={status}")
        cov.append({"League": code, "SeasonStart": int(year), "Source": "Understat", "Available": len(rows), "Matched": matched, "Coverage": coverage, "Status": status})
    return idx, cov

# SOCCER_UNDERSTAT_GRACEFUL_V2'''
P.write_text(s[:start]+new+s[end:],encoding='utf-8')
print('[PATCH] Understat graceful degradation V2 applied')
