#!/usr/bin/env python3
"""Create a frozen canonical baseline plus deterministic V9 state-boundary snapshots."""
from __future__ import annotations
import gzip,json,pickle,shutil,time
from pathlib import Path
import backtest
import v12_runner
ROOT=Path(__file__).resolve().parents[1]; SNAP=ROOT/"exact_parallel"/"baseline"; SNAP.mkdir(parents=True,exist_ok=True)
for p in (ROOT/"backtest_checkpoint_v9.pkl.gz",ROOT/"BACKTEST_COMPLETE_V9",ROOT/"BACKTEST_COMPLETE_V12"): p.unlink(missing_ok=True)
backtest.START=time.time(); backtest.make_feature_names(); total=len(backtest.load_matches())
if total<1: raise SystemExit("empty canonical input")
targets=[int(total*0.25),int(total*0.50),int(total*0.75)]; seen=set(); original=backtest.save_state

def capture(*args,**kwargs):
    original(*args,**kwargs); cursor=int(args[14] if len(args)>14 else kwargs.get("cursor",0))
    for n,t in enumerate(targets,1):
        if n not in seen and cursor>=t:
            src=backtest.CHECKPOINT; dst=SNAP/f"checkpoint_boundary_{n}.pkl.gz"
            if src.exists(): shutil.copy2(src,dst); seen.add(n); print(f"[SNAPSHOT] boundary={n} target={t} cursor={cursor}")
backtest.save_state=capture
v12_runner.main()
import pandas as pd
v9=pd.read_csv(ROOT/"backtest_results_v9.csv",low_memory=False); v12=pd.read_csv(ROOT/"backtest_results_v12.csv",low_memory=False)
if len(v9)!=total: raise SystemExit(f"baseline cardinality mismatch input={total} v9={len(v9)}")
if len(v12)!=len(v9): raise SystemExit("V12 cardinality mismatch")
if len(seen)!=3: raise SystemExit(f"missing boundary snapshots: captured={sorted(seen)}")
rows=[]
for n in range(1,4):
    with gzip.open(SNAP/f"checkpoint_boundary_{n}.pkl.gz","rb") as f: ck=pickle.load(f)
    cursor=int(ck["cursor"]); result_rows=len(ck.get("results",[]))
    if cursor!=result_rows: raise SystemExit(f"boundary {n}: cursor/results mismatch {cursor}!={result_rows}")
    rows.append({"boundary":n,"cursor":cursor,"result_rows":result_rows})
if not (0<rows[0]["cursor"]<rows[1]["cursor"]<rows[2]["cursor"]<total): raise SystemExit("invalid boundary ordering")
manifest={"input_rows":total,"v9_rows":len(v9),"v12_rows":len(v12),"boundaries":rows}
(SNAP/"boundaries.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
shutil.copy2(ROOT/"backtest_results_v12.csv",SNAP/"backtest_results_v12.csv")
print(json.dumps(manifest,ensure_ascii=False))
