#!/usr/bin/env python3
"""Strict equality gate for the four exact-parallel Soccer workers."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"exact_parallel"/"baseline"/"backtest_results_v12.csv"
OUT=ROOT/"exact_parallel"/"merged_backtest_results_v12.csv"
workers=[ROOT/"exact_parallel"/"out"/f"soccer_worker_{i}.csv" for i in range(4)]
if not BASE.exists(): raise SystemExit("baseline output missing")
if not all(p.exists() for p in workers): raise SystemExit("one or more worker outputs missing")

b=pd.read_csv(BASE,low_memory=False)
parts=[pd.read_csv(p,low_memory=False) for p in workers]
all_rows=pd.concat(parts,ignore_index=True)
if "__parallel_row" not in all_rows: raise SystemExit("worker row identity missing")
if len(all_rows)!=len(b): raise SystemExit(f"row count mismatch baseline={len(b)} merged={len(all_rows)}")
if all_rows["__parallel_row"].duplicated().any(): raise SystemExit("duplicate worker rows")
if set(all_rows["__parallel_row"]) != set(range(len(b))): raise SystemExit("worker shard gap/overlap")
all_rows=all_rows.sort_values("__parallel_row",kind="mergesort").reset_index(drop=True)
all_rows=all_rows.drop(columns=["__parallel_row"])

# Exact categorical equality plus tight floating-point equality. A mismatch is
# a hard failure; there is no automatic tolerance that could hide a changed
# model/OOS rule.
if list(all_rows.columns)!=list(b.columns): raise SystemExit("column schema mismatch")
num=[c for c in b.columns if pd.api.types.is_numeric_dtype(b[c])]
for c in b.columns:
    if c in num:
        x=all_rows[c].to_numpy(float); y=b[c].to_numpy(float)
        if not np.array_equal(np.nan_to_num(x,nan=np.nan),np.nan_to_num(y,nan=np.nan),equal_nan=True):
            diff=np.nanmax(np.abs(x-y)); raise SystemExit(f"NUMERIC MISMATCH {c} max_abs={diff}")
    else:
        x=all_rows[c].astype(object); y=b[c].astype(object)
        if not x.equals(y): raise SystemExit(f"COLUMN MISMATCH {c}")

OUT.parent.mkdir(parents=True,exist_ok=True); all_rows.to_csv(OUT,index=False)
summary={"status":"PASS","rows":len(b),"workers":4,"message":"Four-worker output is exactly identical to single-run baseline."}
(OUT.parent/"equality_gate.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False))
