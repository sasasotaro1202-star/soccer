#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed production gate for the V9 score and pre-match MOM artifacts.

This gate does not claim score/MOM improve 1X2 accuracy. It verifies that the
signals produced by the chronological V9 engine are actually present, aligned,
finite where numeric, and based on pre-match candidate fields. It also emits a
small manifest so later OOS ablation work can compare these signals without
silently dropping them from production artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
V9 = ROOT / "backtest_results_v9.csv"
SCORES = ROOT / "backtest_scores_v9.csv"
MOMS = ROOT / "backtest_mom_v9.csv"
OUT = ROOT / "production_score_mom_gate.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    raise SystemExit(f"SCORE_MOM_FAIL_CLOSED: {msg}")


def main() -> None:
    for p in (V9, SCORES, MOMS):
        if not p.exists() or p.stat().st_size == 0:
            fail(f"missing or empty artifact: {p}")

    v9 = pd.read_csv(V9, low_memory=False)
    scores = pd.read_csv(SCORES, low_memory=False)
    moms = pd.read_csv(MOMS, low_memory=False)
    if v9.empty:
        fail("V9 result is empty")

    key = ["League", "Season", "Date", "HomeTeam", "AwayTeam"]
    score_required = key + ["Rank", "PredScore", "Probability", "ActualScore", "Hit"]
    mom_required = key + ["Rank", "Player", "Team", "PreMatchMOMScore", "ActualMOM", "Hit", "ActualMOMMethod"]
    for name, df, required in (("score", scores, score_required), ("mom", moms, mom_required)):
        missing = [c for c in required if c not in df.columns]
        if missing:
            fail(f"{name} artifact missing columns: {missing}")
        if df.duplicated().any():
            fail(f"{name} artifact contains exact duplicate rows")

    if not np.isfinite(scores["Probability"].to_numpy(float)).all():
        fail("score probabilities contain non-finite values")
    if ((scores["Probability"] < 0) | (scores["Probability"] > 1)).any():
        fail("score probabilities outside [0,1]")
    if not np.isfinite(moms["PreMatchMOMScore"].to_numpy(float)).all():
        fail("pre-match MOM scores contain non-finite values")

    # Each V9 match must have three exact-score rows. MOM may have 0 rows for a
    # match when no historical player candidate exists; that is a coverage issue,
    # not permission to invent candidates.
    v9_keys = v9[key].astype(str).drop_duplicates()
    score_keys = scores[key].astype(str).drop_duplicates()
    if len(score_keys) != len(v9_keys):
        fail(f"score match coverage mismatch: v9={len(v9_keys)} score={len(score_keys)}")
    score_counts = scores.groupby(key, dropna=False).size()
    if (score_counts != 3).any():
        fail("score artifact does not contain exactly Top-3 rows per match")
    if set(map(tuple, score_keys.to_numpy())) != set(map(tuple, v9_keys.to_numpy())):
        fail("score match keys do not exactly align with V9 results")

    mom_keys = moms[key].astype(str).drop_duplicates()
    mom_coverage = float(len(mom_keys) / max(len(v9_keys), 1))
    if not moms.empty:
        if (moms["Rank"] < 1).any() or (moms["Rank"] > 4).any():
            fail("MOM rank outside Top-4")
        mom_counts = moms.groupby(key, dropna=False).size()
        if (mom_counts > 4).any():
            fail("more than four MOM candidates for a match")
        # The source engine names this field explicitly to distinguish it from
        # post-match ActualMOM. This is a structural PIT guard, not proof of
        # timestamp completeness for every external source.
        if "ActualMOM" not in moms.columns or "PreMatchMOMScore" not in moms.columns:
            fail("pre-match/post-match MOM separation missing")

    report = {
        "status": "PASS",
        "matches_v9": int(len(v9)),
        "score_rows": int(len(scores)),
        "score_rows_per_match": 3,
        "mom_rows": int(len(moms)),
        "mom_match_coverage": mom_coverage,
        "score_sha256": sha256(SCORES),
        "mom_sha256": sha256(MOMS),
        "v9_sha256": sha256(V9),
        "note": "Score/MOM artifacts are preserved and validated; predictive lift requires separate chronological OOS ablation and is not assumed.",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
