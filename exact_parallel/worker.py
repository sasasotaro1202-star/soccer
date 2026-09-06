#!/usr/bin/env python3
"""Exact-parallel verification worker.

This deliberately replays the canonical V12 engine from the same frozen input
snapshot, then emits only one deterministic shard of the resulting rows.  It
never changes model/OOS logic and never writes production state.
"""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
WORKER=int(os.environ["WORKER_ID"])
COUNT=int(os.environ.get("WORKER_COUNT","4"))
OUT=ROOT/"exact_parallel"/"out"; OUT.mkdir(parents=True,exist_ok=True)

# A worker must start from a clean chronological state. Checkpoints/complete
# markers would otherwise turn the replay into a resume and invalidate the
# equivalence test.
for p in (ROOT/"backtest_checkpoint_v9.pkl.gz", ROOT/"BACKTEST_COMPLETE_V9", ROOT/"BACKTEST_COMPLETE_V12"):
    p.unlink(missing_ok=True)

# Ensure all workers execute the exact checked-out production code.
cmd=[sys.executable,"v12_runner.py"]
env=os.environ.copy()
env.update({"MAX_RUNTIME_SECONDS":"5100","ENABLE_SOFASCORE":"1","SOFASCORE_DETAILS":"0","ENABLE_UNDERSTAT":"1","INCLUDE_CLUB_FRIENDLIES":"1","FRIENDLY_MIN_YEAR":"2010","FRIENDLY_MAX_PAGES":"250"})
subprocess.run(cmd,cwd=ROOT,env=env,check=True)

src=ROOT/"backtest_results_v12.csv"
df=pd.read_csv(src,low_memory=False)
if df.empty: raise SystemExit("empty V12 output")
# Stable row identity is the canonical output order. np.array_split gives
# deterministic contiguous shards without changing any prediction.
idx=np.array_split(np.arange(len(df)),COUNT)[WORKER]
shard=df.iloc[idx].copy()
shard.insert(0,"__parallel_row",idx.astype(int))
shard.to_csv(OUT/f"soccer_worker_{WORKER}.csv",index=False)

# Hash the exact output and the frozen cache manifest so the aggregator can
# prove all workers used the same inputs/code.
def tree_hash(root: Path)->str:
    h=hashlib.sha256()
    if not root.exists(): return h.hexdigest()
    for p in sorted(x for x in root.rglob('*') if x.is_file()):
        h.update(str(p.relative_to(root)).encode()); h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()
manifest={"worker":WORKER,"count":COUNT,"rows_total":len(df),"rows_shard":len(shard),"code_sha":os.environ.get("GITHUB_SHA",""),"cache_sha":tree_hash(ROOT/"cache"),"output_sha":hashlib.sha256(src.read_bytes()).hexdigest()}
(OUT/f"soccer_worker_{WORKER}.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(manifest,ensure_ascii=False))
