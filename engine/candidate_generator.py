#!/usr/bin/env python3
"""Generate bounded, reproducible V12 research candidates from diagnostics.

Candidate generation is deliberately declarative: it proposes experiments;
it does not promote a candidate without out-of-sample validation.
"""
from __future__ import annotations
import json
from pathlib import Path

BASE = {
    "temperature_grid": [0.80,0.90,1.00,1.10,1.20,1.35],
    "recency_half_life": [90,180,360,720],
    "expert_weight_floor": [0.00,0.02,0.05],
}


def generate(evaluation=None, failures=None):
    evaluation=evaluation or {}; failures=failures or {}
    reasons=failures.get("reason_counts",{})
    candidates=[]
    def add(cid, kind, params, rationale): candidates.append({"id":cid,"kind":kind,"params":params,"rationale":rationale,"validation":"walk_forward_required"})
    add("v12_calibration_grid","calibration",{"temperature_grid":BASE["temperature_grid"]},"Re-test probability calibration without changing labels.")
    add("v12_recency_grid","stacking",{"half_life":BASE["recency_half_life"]},"Test recency adaptation of prior expert performance.")
    add("v12_weight_floor","stacking",{"floor":BASE["expert_weight_floor"]},"Prevent unstable expert starvation while retaining continuous blending.")
    if reasons.get("draw_missed",0)>0:
        add("v12_draw_specialist","1x2",{"draw_prior_strength":[0.02,0.05,0.10]},"Historical diagnostics show draw misses; test conservative draw prior adjustment.")
    if reasons.get("high_confidence_miss",0)>0:
        add("v12_confidence_shrink","calibration",{"temperature_floor":[1.05,1.15,1.25]},"High-confidence errors justify stronger probability shrinkage.")
    return {"version":"V12-candidate-generator-v1","candidates":candidates}


def main():
    ev=json.loads(Path("evaluation.json").read_text(encoding="utf-8")) if Path("evaluation.json").exists() else {}
    fa=json.loads(Path("failure_analysis.json").read_text(encoding="utf-8")) if Path("failure_analysis.json").exists() else {}
    out=generate(ev,fa); Path("candidate_models.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(out,ensure_ascii=False))

if __name__=="__main__": main()
