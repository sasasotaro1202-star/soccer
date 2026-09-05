#!/usr/bin/env python3
"""Leak-safe failure analysis for Soccer V12.

This module diagnoses *past* predictions only. It never feeds the current
match result back into a current prediction.
"""
from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from pathlib import Path


def f(row, *keys):
    for k in keys:
        try:
            if row.get(k) not in (None, ""):
                return float(row[k])
        except (TypeError, ValueError):
            pass
    return None


def actual(row):
    a = row.get("Actual") or row.get("actual")
    if a in ("H", "D", "A"):
        return a
    h, aw = f(row,"FTHG"), f(row,"FTAG")
    return "H" if h is not None and aw is not None and h>aw else "A" if h is not None and aw is not None and h<aw else "D" if h is not None and aw is not None else None


def pred(row):
    p = row.get("PredictedV12") or row.get("Predicted")
    return p if p in ("H","D","A") else None


def analyze(rows):
    errors=[]; by_league=defaultdict(Counter); by_season=defaultdict(Counter)
    for r in rows:
        p,a=pred(r),actual(r)
        if not p or not a or p==a: continue
        reason="prediction_mismatch"
        hp,dp,ap=[f(r,k) for k in ("HomeProbV12","DrawProbV12","AwayProbV12")]
        probs=[x for x in (hp,dp,ap) if x is not None]
        if a=="D": reason="draw_missed"
        elif p=="H" and a=="A": reason="home_overconfidence"
        elif p=="A" and a=="H": reason="away_overconfidence"
        elif probs and max(probs)>=0.70: reason="high_confidence_miss"
        ph,pa,ah,aa=f(r,"PredHomeGoals"),f(r,"PredAwayGoals"),f(r,"FTHG"),f(r,"FTAG")
        if ah is not None and aa is not None and ph is not None and pa is not None:
            goal_error=abs(ph-ah)+abs(pa-aa)
            if goal_error>=2.5: reason += ":large_goal_error"
        rec={"league":r.get("League"),"season":r.get("Season"),"reason":reason,"predicted":p,"actual":a}
        errors.append(rec); by_league[str(r.get("League"))][reason]+=1; by_season[str(r.get("Season"))][reason]+=1
    return {"version":"V12-failure-analysis-v1","error_count":len(errors),"reason_counts":dict(Counter(x["reason"] for x in errors)),"by_league":{k:dict(v) for k,v in by_league.items()},"by_season":{k:dict(v) for k,v in by_season.items()},"examples":errors[-100:]}


def main():
    paths=[Path("backtest_results_v12.csv"),Path("backtest_results_v9.csv"),Path("backtest_results.csv")]
    src=next((p for p in paths if p.exists()),None)
    rows=list(csv.DictReader(src.open(encoding="utf-8-sig"))) if src else []
    out=analyze(rows); Path("failure_analysis.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(out,ensure_ascii=False))

if __name__=="__main__": main()
