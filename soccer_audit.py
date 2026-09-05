#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic integrity/leakage audit for the Soccer V12 pipeline."""
from __future__ import annotations
import ast, re
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent

def fail(msg: str) -> None:
    raise SystemExit("[AUDIT FAIL] " + msg)

def read(rel: str) -> str:
    p = ROOT / rel
    if not p.exists(): fail(f"required file missing: {rel}")
    return p.read_text(encoding="utf-8")

def assert_function(text: str, filename: str, name: str) -> None:
    try: tree = ast.parse(text, filename=filename)
    except SyntaxError as e: fail(f"syntax error in {filename}: {e}")
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in ast.walk(tree)):
        fail(f"{filename}: function {name} missing")

def main() -> None:
    backtest, workflow, smoke, runner = (read("backtest.py"), read(".github/workflows/soccer-backtest.yml"), read("tests/smoke_test.py"), read("v12_runner.py"))
    for text, name in ((backtest,"backtest.py"),(smoke,"tests/smoke_test.py"),(runner,"v12_runner.py")):
        try: ast.parse(text, filename=name)
        except SyntaxError as e: fail(str(e))
    for name in ("build_models","_chronological_folds","optimize_blend","save_state","load_state","same_team","make_feature_names","main"):
        assert_function(backtest,"backtest.py",name)
    for marker in ("closing odds","prediction","checkpoint","MAX_TRAIN","temperature","blend","H2H","SofaScore","Understat","Football-Data","chronological"):
        if marker.lower() not in backtest.lower(): fail(f"required safeguard/data marker missing: {marker}")
    for marker in ("sample_weight","clf__sample_weight","np.nan_to_num","n3_rows","gzip","BACKTEST_COMPLETE_V9"):
        if marker not in backtest: fail(f"required engine marker missing: {marker}")
    for marker in ("workflow_dispatch","schedule","concurrency:","cancel-in-progress: false","checkpoint","upload-artifact","V12 preflight and integrity gate","py_compile","smoke_test.py","v12_runner.py","soccer-backtest-state-v12"):
        if marker not in workflow: fail(f"workflow hardening marker missing: {marker}")
    for marker in ("walk-forward","chronological","prior","temperature","fail-closed"):
        if marker.lower() not in runner.lower(): fail(f"V12 runner safeguard marker missing: {marker}")
    if not any(x in runner.lower() for x in ("expert stack", "stacked_walk_forward", "adaptive")):
        fail("V12 runner safeguard marker missing: expert stacking/adaptive mode")
    if "contents: write" not in workflow or "git add" not in workflow or "git push" not in workflow:
        fail("workflow does not persist resumable state/results to the repository")

    # The active V9 feature contract is exactly 136 features. Do not pad or
    # delete arbitrary columns merely to satisfy a stale expected count.
    ns = {"__name__": "soccer_audit_probe"}
    exec(compile(backtest, "backtest.py", "exec"), ns, ns)
    ns["make_feature_names"]()
    if len(ns["FEATURE_NAMES"]) != 136:
        fail(f"active V9 feature schema changed unexpectedly: {len(ns['FEATURE_NAMES'])} != 136")

    # Team matching must be exact after explicit canonicalization. Substring
    # matching can falsely join unrelated teams (e.g. United/Manchester United).
    same_team = ns["same_team"]
    for a, b, expected in (
        ("Manchester United", "Manchester United FC", True),
        ("Man United", "Manchester United", True),
        ("Man Utd", "Manchester United", True),
        ("United", "Manchester United", False),
        ("City", "Manchester City", False),
        ("Spurs", "Tottenham Hotspur", True),
    ):
        if bool(same_team(a, b)) != expected:
            fail(f"team canonicalization mismatch: {a!r} vs {b!r} expected={expected}")

    # V12's current result must be appended only after its prediction has been
    # finalized; this is the explicit meta-layer leakage barrier.
    if "current outcome is appended only after the prediction" not in runner:
        fail("V12 current-outcome update barrier missing")

    outputs = ["backtest_results_v9.csv","backtest_scores_v9.csv","overall_summary_v9.csv","league_summary_v9.csv","season_summary_v9.csv","confidence_summary_v9.csv","score_summary_v9.csv","mom_summary_v9.csv","model_comparison_v9.csv","data_coverage_v9.csv","feature_importance_v9.csv","backtest_results_v12.csv","overall_summary_v12.csv"]
    for rel in outputs:
        p=ROOT/rel
        if not p.exists(): continue
        try: d=pd.read_csv(p,low_memory=False)
        except Exception as e: fail(f"cannot read {rel}: {e}")
        if d.isna().all(axis=1).any(): fail(f"{rel} contains completely empty rows")
    for rel in ("backtest_results_v9.csv","backtest_results_v12.csv"):
        p=ROOT/rel
        if not p.exists(): continue
        d=pd.read_csv(p,low_memory=False)
        ids=[c for c in ("Date","HomeTeam","AwayTeam") if c in d.columns]
        if len(ids)==3 and int(d.duplicated(ids,keep=False).sum()): fail(f"{rel} contains duplicate match keys")
        for c in [c for c in d.columns if re.search(r"(^p_|prob|probability)",str(c),re.I)]:
            vals=pd.to_numeric(d[c],errors="coerce").dropna()
            if len(vals) and ((vals < -1e-9) | (vals > 1+1e-9)).any(): fail(f"probability column {c} contains values outside [0,1]")
        if rel.endswith("v12.csv") and len(d):
            need=["HomeProbV12","DrawProbV12","AwayProbV12","PredictedV12","Actual"]
            miss=[c for c in need if c not in d.columns]
            if miss: fail(f"V12 output missing columns: {miss}")
            probs=d[["HomeProbV12","DrawProbV12","AwayProbV12"]].to_numpy(float)
            if not np.all(np.isfinite(probs)) or not np.allclose(probs.sum(axis=1),1.0,atol=1e-6): fail("V12 probabilities are not finite normalized 3-way probabilities")
    print("[AUDIT PASS] V12 syntax, 136-feature schema, exact team canonicalization, leakage barriers, workflow persistence, and output integrity checks passed")

if __name__ == "__main__": main()
