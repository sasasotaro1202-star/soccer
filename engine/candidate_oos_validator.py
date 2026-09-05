#!/usr/bin/env python3
"""Real walk-forward candidate training/OOS validation for Soccer V12.

Candidates are trained only on rows strictly preceding each prediction row.
The current row's Actual is never available to the learner until after scoring.
This module deliberately works from the V9/V12 component probability columns so
experiments can be reproduced without re-downloading external data.
"""
from __future__ import annotations
import json, math, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

V9=Path("backtest_results_v9.csv")
V12=Path("backtest_results_v12.csv")
CAND=Path("candidate_models.json")
OUT=Path("candidate_evaluations.json")
EPS=1e-7
MIN_TRAIN=int(os.environ.get("CANDIDATE_MIN_TRAIN","260"))
RETRAIN_EVERY=int(os.environ.get("CANDIDATE_RETRAIN_EVERY","100"))
MAX_ROWS=int(os.environ.get("CANDIDATE_MAX_ROWS","0"))
BUDGET=float(os.environ.get("MAX_RUNTIME_SECONDS","3300"))


def clean(p):
    p=np.nan_to_num(np.asarray(p,float),nan=1/3,posinf=1/3,neginf=1/3)
    p=np.clip(p,EPS,1.0); s=p.sum(); return p/s if s>0 else np.ones(3)/3


def feature_frame(df):
    cols=[]
    for prefix in ("HomeProb","MarketHome","MLHome","PoissonHome"):
        if prefix in df: pass
    # Use probability logits, expert disagreement, market-vs-model gaps and
    # contextual categorical fields that are known before kickoff.
    X=pd.DataFrame(index=df.index)
    for name in ("HomeProb","Market","ML","Poisson"):
        for c in ("Home","Draw","Away"):
            col=f"{name}{c}"
            if col in df.columns: X[f"{name}_{c}"]=pd.to_numeric(df[col],errors="coerce").fillna(1/3)
    if "League" in df: X["League"]=df["League"].fillna("__UNKNOWN__").astype(str)
    else: X["League"]="__UNKNOWN__"
    if "Season" in df: X["Season"]=df["Season"].fillna("__UNKNOWN__").astype(str)
    else: X["Season"]="__UNKNOWN__"
    if "Date" in df:
        d=pd.to_datetime(df["Date"],errors="coerce",dayfirst=True)
        X["month"] = d.dt.month.fillna(0).astype(int).astype(str)
    return X


def make_pipeline(X):
    cat=[c for c in ("League","Season","month") if c in X.columns]
    num=[c for c in X.columns if c not in cat]
    pre=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
    clf=LogisticRegression(max_iter=500,C=0.5,multi_class="multinomial",random_state=42)
    return Pipeline([("pre",pre),("clf",clf)])


def ll(P,y):
    return float(np.mean([-math.log(np.clip(P[i,int(y[i])],EPS,1.0)) for i in range(len(y))]))


def brier(P,y):
    oh=np.eye(3)[y]
    return float(np.mean(np.sum((P-oh)**2,axis=1)))


def run_candidate(cid, kind, params, df, baseline):
    X=feature_frame(df); y=pd.to_numeric(df["Actual"],errors="coerce").astype(int).to_numpy()
    n=len(df); pred=np.zeros((n,3),float); last_model=None; trained=0
    start=time.monotonic()
    for i in range(n):
        if time.monotonic()-start > max(30.0,BUDGET*0.72):
            return {"id":cid,"status":"timeout","params":params}
        if i<MIN_TRAIN:
            pred[i]=clean(df.loc[df.index[i],["HomeProb","DrawProb","AwayProb"]].to_numpy(float))
            continue
        if last_model is None or (i-MIN_TRAIN)%RETRAIN_EVERY==0:
            train_idx=np.arange(i)
            # bounded rolling train window keeps Actions runtime predictable
            if len(train_idx)>1800: train_idx=train_idx[-1800:]
            model=make_pipeline(X.iloc[train_idx])
            model.fit(X.iloc[train_idx],y[train_idx])
            last_model=model; trained+=1
        pred[i]=clean(last_model.predict_proba(X.iloc[[i]])[0])
    eval_idx=np.arange(MIN_TRAIN,n)
    if not len(eval_idx): return {"id":cid,"status":"insufficient_data","params":params}
    yp=y[eval_idx]; pp=pred[eval_idx]
    acc=float(np.mean(pp.argmax(1)==yp)); loss=ll(pp,yp); br=brier(pp,yp)
    bidx=eval_idx
    bp=baseline[bidx]
    bacc=float(np.mean(bp.argmax(1)==yp)); bll=ll(bp,yp); bb=brier(bp,yp)
    return {"id":cid,"status":"validated","kind":kind,"params":params,"oos_rows":int(len(eval_idx)),"trained_models":trained,
            "candidate":{"accuracy":acc,"logloss":loss,"brier":br},
            "baseline":{"accuracy":bacc,"logloss":bll,"brier":bb},
            "delta":{"accuracy":acc-bacc,"logloss":loss-bll,"brier":br-bb},
            "adoptable":bool(acc>=bacc and loss<=bll and (acc>bacc or loss<bll))}


def main():
    if not V9.exists() or not V12.exists() or not CAND.exists(): raise SystemExit("candidate OOS prerequisites missing")
    df=pd.read_csv(V9,low_memory=False)
    v12=pd.read_csv(V12,low_memory=False)
    if len(df)!=len(v12): raise SystemExit("V9/V12 row count mismatch")
    required=["Actual","HomeProb","DrawProb","AwayProb","MarketHome","MarketDraw","MarketAway","MLHome","MLDraw","MLAway","PoissonHome","PoissonDraw","PoissonAway"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise SystemExit(f"missing candidate features: {missing}")
    baseline=v12[["HomeProbV12","DrawProbV12","AwayProbV12"]].to_numpy(float)
    candidates=json.loads(CAND.read_text(encoding="utf-8")).get("candidates",[])
    if MAX_ROWS>0: df=df.iloc[:MAX_ROWS].copy(); baseline=baseline[:MAX_ROWS]
    results=[]
    for c in candidates:
        r=run_candidate(c["id"],c.get("kind","unknown"),c.get("params",{}),df,baseline)
        results.append(r)
        if r.get("status")=="timeout": break
    payload={"version":"V12-candidate-oos-v1","leak_policy":"prior_rows_only","min_train":MIN_TRAIN,"retrain_every":RETRAIN_EVERY,"candidates":results}
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False))

if __name__=="__main__": main()
