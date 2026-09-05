#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Soccer Backtest V12 runner.

V12 keeps the proven V9 chronological engine intact and adds a strictly
walk-forward meta-layer. Only rows preceding each evaluated match can affect
its V12 probability. No future/current outcome is used in the meta decision.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import backtest

SRC=Path("backtest_results_v9.csv")
OUT=Path("backtest_results_v12.csv")
V9_DONE=Path("BACKTEST_COMPLETE_V9")
V12_DONE=Path("BACKTEST_COMPLETE_V12")
WINDOW=240
TEMP_WINDOW=360
MIN_HISTORY=80
EPS=1e-7

def _p(row,prefix): return np.array([row[f"{prefix}Home"],row[f"{prefix}Draw"],row[f"{prefix}Away"]],dtype=float)

def _clean(p):
    p=np.nan_to_num(np.asarray(p,dtype=float),nan=1/3,posinf=1/3,neginf=1/3); p=np.clip(p,EPS,1.0); return p/p.sum()

def _ll(p,y): return -math.log(float(np.clip(p[int(y)],EPS,1.0)))

def _softmax_inverse(losses):
    a=-np.asarray(losses,dtype=float); a-=np.nanmax(a); w=np.exp(np.clip(a,-20,20)); w[~np.isfinite(w)]=0.0; s=w.sum(); return w/s if s>0 else np.ones(len(losses))/len(losses)

def _temp(p,t): return _clean(np.power(np.clip(p,EPS,1.0),1.0/float(t)))

def _best_temperature(hist_p,hist_y):
    if len(hist_y)<MIN_HISTORY: return 1.0
    best_t,best_ll=1.0,float("inf")
    for t in np.linspace(.70,1.60,19):
        loss=float(np.mean([_ll(_temp(p,t),y) for p,y in zip(hist_p,hist_y)]))
        if loss<best_ll: best_ll,best_t=loss,float(t)
    return best_t

def build_v12():
    if not SRC.exists(): raise FileNotFoundError(f"required V9 output missing: {SRC}")
    df=pd.read_csv(SRC,low_memory=False)
    required=["Predicted","Actual","HomeProb","DrawProb","AwayProb","MarketHome","MarketDraw","MarketAway","MLHome","MLDraw","MLAway","PoissonHome","PoissonDraw","PoissonAway"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise RuntimeError(f"V12 fail-closed: missing columns: {missing}")
    df["__v12_order"]=np.arange(len(df))
    if "Date" in df.columns:
        df["__v12_date"]=pd.to_datetime(df["Date"],errors="coerce",dayfirst=True)
        df=df.sort_values(["__v12_date","__v12_order"],kind="mergesort").reset_index(drop=True)
    out_probs=[]; temps=[]; modes=[]
    hist={k:[] for k in ("market","ml","poisson","base","adaptive")}; hist_y=[]
    for i,row in df.iterrows():
        base=_clean([row["HomeProb"],row["DrawProb"],row["AwayProb"]])
        comps={"market":_clean(_p(row,"Market")),"ml":_clean(_p(row,"ML")),"poisson":_clean(_p(row,"Poisson"))}
        start=max(0,i-WINDOW); hy=hist_y[start:]
        losses=[float(np.mean([_ll(p,y) for p,y in zip(hist[name][start:],hy)])) if hy else 1.10 for name in ("market","ml","poisson")]
        w=_softmax_inverse(losses)
        adaptive=_clean(sum(w[j]*comps[name] for j,name in enumerate(("market","ml","poisson"))))
        hb,ha=hist["base"][start:],hist["adaptive"][start:]
        if len(hy)>=MIN_HISTORY:
            base_ll=float(np.mean([_ll(p,y) for p,y in zip(hb,hy)])); adap_ll=float(np.mean([_ll(p,y) for p,y in zip(ha,hy)]))
            raw=adaptive if adap_ll+1e-5<base_ll else base; mode="adaptive" if adap_ll+1e-5<base_ll else "v9_base"
        else: raw,mode=base,"v9_base_warmup"
        hs=hist["adaptive" if mode=="adaptive" else "base"][max(0,i-TEMP_WINDOW):]; hy_t=hist_y[max(0,i-TEMP_WINDOW):]
        t=_best_temperature(hs,hy_t); out_probs.append(_temp(raw,t)); temps.append(t); modes.append(mode)
        # Outcome enters state only after this row's prediction is finalized.
        hist["market"].append(comps["market"]); hist["ml"].append(comps["ml"]); hist["poisson"].append(comps["poisson"]); hist["base"].append(base); hist["adaptive"].append(adaptive); hist_y.append(int(row["Actual"]))
    P=np.asarray(out_probs); df["HomeProbV12"]=P[:,0]; df["DrawProbV12"]=P[:,1]; df["AwayProbV12"]=P[:,2]; df["ConfidenceV12"]=P.max(axis=1); df["PredictedV12"]=P.argmax(axis=1); df["CorrectV12"]=(df["PredictedV12"].astype(int)==df["Actual"].astype(int)).astype(int); df["V12Temperature"]=temps; df["V12Mode"]=modes
    df=df.sort_values("__v12_order",kind="mergesort").drop(columns=[c for c in ("__v12_order","__v12_date") if c in df.columns]); df.to_csv(OUT,index=False)
    y=df["Actual"].to_numpy(int); p=df[["HomeProbV12","DrawProbV12","AwayProbV12"]].to_numpy(float); pred=p.argmax(1); onehot=np.eye(3)[y]
    acc=float(np.mean(pred==y)); logloss=float(np.mean([-math.log(np.clip(p[j,y[j]],EPS,1.0)) for j in range(len(y))])); brier=float(np.mean(np.sum((p-onehot)**2,axis=1)))
    pd.DataFrame([{"Version":"V12","Matches":len(df),"Accuracy":acc,"LogLoss":logloss,"Brier":brier,"MeanConfidence":float(p.max(1).mean()),"AdaptiveShare":float(np.mean(np.asarray(modes)=="adaptive")),"MeanTemperature":float(np.mean(temps))}]).to_csv("overall_summary_v12.csv",index=False)
    print(f"[V12] wrote {OUT} rows={len(df)} accuracy={acc:.6f} logloss={logloss:.6f} brier={brier:.6f}")

def main():
    backtest.main()
    if not V9_DONE.exists():
        raise RuntimeError("V12 not finalized: chronological V9 engine did not reach true completion")
    build_v12()
    V12_DONE.write_text("BACKTEST_COMPLETE_V12\n",encoding="utf-8")
    print("[V12] BACKTEST_COMPLETE_V12")

if __name__=="__main__": main()
