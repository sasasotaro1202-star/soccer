#!/usr/bin/env python3
"""Conservative model promotion gate for Soccer V12.

A candidate is promotable only when an externally produced walk-forward/OOS
result demonstrates improvement. No current outcome is used to choose its
current prediction.
"""
from __future__ import annotations
import json
from pathlib import Path


def compare(baseline, candidates, min_accuracy_delta=0.0, max_logloss_delta=0.0):
    rows=[]
    bacc=baseline.get("accuracy"); bll=baseline.get("logloss")
    for c in candidates:
        acc=c.get("accuracy"); ll=c.get("logloss")
        valid=acc is not None and ll is not None and bacc is not None and bll is not None
        improved=bool(valid and acc >= bacc+min_accuracy_delta and ll <= bll+max_logloss_delta and (acc>bacc or ll<bll))
        rows.append({"id":c.get("id"),"accuracy":acc,"logloss":ll,"improved":improved,"adopt":improved,"reason":"OOS improvement" if improved else "No demonstrated OOS improvement"})
    best=next((r for r in rows if r["adopt"]),None)
    return {"version":"V12-model-compare-v1","baseline":baseline,"candidates":rows,"recommended":best["id"] if best else None,"fail_closed":True}


def main():
    baseline=json.loads(Path("evaluation.json").read_text(encoding="utf-8")) if Path("evaluation.json").exists() else {}
    candidates=[]
    p=Path("candidate_evaluations.json")
    if p.exists(): candidates=json.loads(p.read_text(encoding="utf-8")).get("candidates",[])
    out=compare(baseline,candidates); Path("model_comparison_v12.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(out,ensure_ascii=False))

if __name__=="__main__": main()
