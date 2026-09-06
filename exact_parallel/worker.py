#!/usr/bin/env python3
"""True exact Soccer shard worker.

Each worker starts from a canonical V9 state snapshot produced by one
sequential baseline replay. It then executes only its assigned chronological
interval. V12 is rebuilt over the resulting prefix+shard and only the shard is
emitted. No worker writes production state.
"""
from __future__ import annotations
import gzip,json,os,pickle,shutil,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; W=int(os.environ["WORKER_ID"]); N=int(os.environ.get("WORKER_COUNT","4")); OUT=ROOT/"exact_parallel"/"out"; OUT.mkdir(parents=True,exist_ok=True); BASE=ROOT/"exact_parallel"/"baseline"
manifest=json.loads((BASE/"boundaries.json").read_text(encoding="utf-8")); bounds=[0]+[int(x["cursor"]) for x in manifest["boundaries"]]+[int(manifest["input_rows"])]
if len(bounds)!=5: raise SystemExit("invalid boundary manifest")
start,end=bounds[W],bounds[W+1]
if W>0: shutil.copy2(BASE/f"checkpoint_boundary_{W}.pkl.gz",ROOT/"backtest_checkpoint_v9.pkl.gz")
else: (ROOT/"backtest_checkpoint_v9.pkl.gz").unlink(missing_ok=True)
for p in (ROOT/"BACKTEST_COMPLETE_V9",ROOT/"BACKTEST_COMPLETE_V12",ROOT/"backtest_results_v9.csv",ROOT/"backtest_results_v12.csv"): p.unlink(missing_ok=True)
os.environ.update({"MAX_RUNTIME_SECONDS":"5100","ENABLE_SOFASCORE":"1","SOFASCORE_DETAILS":"0","ENABLE_UNDERSTAT":"1","INCLUDE_CLUB_FRIENDLIES":"1","FRIENDLY_MIN_YEAR":"2010","FRIENDLY_MAX_PAGES":"250"})
import backtest
backtest.CHECKPOINT=ROOT/"backtest_checkpoint_v9.pkl.gz"; original_save=backtest.save_state
class ShardStop(Exception): pass
def save_and_stop(*args,**kwargs):
    original_save(*args,**kwargs); cursor=int(args[14] if len(args)>14 else kwargs.get("cursor",0))
    if W<3 and cursor>=end: raise ShardStop(f"worker={W} reached end cursor={cursor}")
backtest.save_state=save_and_stop
try: backtest.main()
except ShardStop as e: print("[SHARD STOP]",e)
if not backtest.CHECKPOINT.exists(): raise SystemExit("worker checkpoint missing")
with gzip.open(backtest.CHECKPOINT,"rb") as f: ck=pickle.load(f)
rows=pd.DataFrame(ck.get("results",[])); cursor=int(ck.get("cursor",-1))
if rows.empty: raise SystemExit("worker produced no V9 rows")
if len(rows)!=cursor: raise SystemExit(f"cursor/results mismatch {cursor}!={len(rows)}")
if cursor<end: raise SystemExit(f"worker prefix shorter than shard end: cursor={cursor} end={end}")
if W==3 and cursor!=end: raise SystemExit(f"final worker cardinality mismatch cursor={cursor} end={end}")
rows.to_csv(ROOT/"backtest_results_v9.csv",index=False,encoding="utf-8-sig")
import v12_runner
v12_runner.build_v12()
full=pd.read_csv(ROOT/"backtest_results_v12.csv",low_memory=False)
if len(full)!=cursor: raise SystemExit(f"V12 cardinality mismatch {len(full)}!={cursor}")
shard=full.iloc[start:end].copy(); shard.insert(0,"__parallel_row",np.arange(start,end,dtype=int)); shard.to_csv(OUT/f"soccer_worker_{W}.csv",index=False)
meta={"worker":W,"count":N,"start":start,"end":end,"rows_total":len(full),"rows_shard":len(shard),"code_sha":os.environ.get("GITHUB_SHA","")}
(OUT/f"soccer_worker_{W}.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(meta,ensure_ascii=False))
