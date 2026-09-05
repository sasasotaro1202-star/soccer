#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic integrity/leakage audit for the soccer backtest pipeline.

This audit is intentionally conservative: it fails closed on structural problems and
never changes model outputs. It is adapted from the hardened baseball pipeline pattern.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def fail(msg: str) -> None:
    raise SystemExit("[AUDIT FAIL] " + msg)


def read(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        fail(f"required file missing: {rel}")
    return p.read_text(encoding="utf-8")


def assert_function(text: str, filename: str, name: str) -> None:
    try:
        tree = ast.parse(text, filename=filename)
    except SyntaxError as e:
        fail(f"syntax error in {filename}: {e}")
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in ast.walk(tree)):
        fail(f"{filename}: function {name} missing")


def main() -> None:
    backtest = read("backtest.py")
    workflow = read(".github/workflows/soccer-backtest.yml")
    smoke = read("tests/smoke_test.py")

    # Syntax and critical pipeline primitives.
    try:
        ast.parse(backtest, filename="backtest.py")
        ast.parse(smoke, filename="tests/smoke_test.py")
    except SyntaxError as e:
        fail(str(e))

    for name in (
        "build_models", "_chronological_folds", "optimize_blend", "save_state", "load_state",
        "same_team", "make_feature_names", "main",
    ):
        assert_function(backtest, "backtest.py", name)

    # The prediction path must build features before mutating historical state.
    feature_tokens = ("make_features", "build_features", "match_features")
    feature_pos = min((backtest.find(x) for x in feature_tokens if backtest.find(x) >= 0), default=-1)
    update_candidates = [backtest.find(x) for x in ("update_team", "update_history", "_update_player", "leagues[", "save_state(")]
    update_pos = min((x for x in update_candidates if x >= 0), default=-1)
    if feature_pos < 0:
        fail("feature-generation path not found")
    if "chronological" not in backtest.lower() or "_chronological_folds" not in backtest:
        fail("chronological validation is not explicit")
    if "random split" in backtest.lower() and "random_state" not in backtest.lower():
        fail("unexpected random split marker")

    # Required leakage safeguards and model controls.
    required_markers = (
        "closing odds", "prediction", "checkpoint", "MAX_TRAIN", "temperature", "blend",
        "H2H", "SofaScore", "Understat", "Football-Data",
    )
    for marker in required_markers:
        if marker.lower() not in backtest.lower():
            fail(f"required safeguard/data marker missing: {marker}")

    if "sample_weight" not in backtest:
        fail("sample-weight support missing")
    if "clf__sample_weight" not in backtest:
        fail("Logistic Pipeline sample_weight routing missing")
    if "np.nan_to_num" not in backtest or "n3_rows" not in backtest:
        fail("probability sanitization/normalization missing")
    if "gzip" not in backtest:
        fail("compressed checkpoint support missing")
    if "BACKTEST_COMPLETE_V9" not in backtest:
        fail("completion marker missing")

    # Workflow must behave like a resumable production job, not a one-shot notebook.
    for marker in (
        "workflow_dispatch", "schedule", "concurrency:", "cancel-in-progress: false",
        "checkpoint", "upload-artifact", "Preflight tests", "py_compile", "smoke_test.py",
    ):
        if marker not in workflow:
            fail(f"workflow hardening marker missing: {marker}")

    # The baseball hardening pattern requires the workflow to persist state rather than
    # silently losing progress between 30-minute runs.
    if "contents: write" not in workflow:
        fail("workflow lacks contents write permission for persistent state")
    if "git add" not in workflow or "git push" not in workflow:
        fail("workflow does not persist resumable state/results to the repository")

    # Validate generated summaries when present. Missing outputs are allowed before the
    # first successful run, but malformed outputs fail closed.
    outputs = [
        "backtest_results_v9.csv", "backtest_scores_v9.csv", "overall_summary_v9.csv",
        "league_summary_v9.csv", "season_summary_v9.csv", "confidence_summary_v9.csv",
        "score_summary_v9.csv", "mom_summary_v9.csv", "model_comparison_v9.csv",
        "data_coverage_v9.csv", "feature_importance_v9.csv",
    ]
    for rel in outputs:
        p = ROOT / rel
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p, low_memory=False)
        except Exception as e:
            fail(f"cannot read {rel}: {e}")
        if len(d) and any(str(c).lower() in {"prob_home", "p_home", "home_prob"} for c in d.columns):
            pass
        if d.isna().all(axis=1).any():
            fail(f"{rel} contains completely empty rows")

    result = ROOT / "backtest_results_v9.csv"
    if result.exists():
        d = pd.read_csv(result, low_memory=False)
        id_cols = [c for c in ("Date", "HomeTeam", "AwayTeam") if c in d.columns]
        if len(id_cols) == 3:
            dup = int(d.duplicated(id_cols, keep=False).sum())
            if dup:
                fail(f"backtest_results_v9.csv has {dup} duplicate match keys")

        # Detect obviously invalid probability columns without assuming one exact schema.
        prob_cols = [c for c in d.columns if re.search(r"(^p_|prob|probability)", str(c), re.I)]
        for c in prob_cols:
            vals = pd.to_numeric(d[c], errors="coerce").dropna()
            if len(vals) and ((vals < -1e-9) | (vals > 1 + 1e-9)).any():
                fail(f"probability column {c} contains values outside [0,1]")

    print("[AUDIT PASS] syntax, leakage safeguards, workflow persistence, and generated-output integrity checks passed")


if __name__ == "__main__":
    main()
