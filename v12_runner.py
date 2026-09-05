#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Soccer Backtest V12 runner.

V12 preserves the V9 chronological engine and adds a strictly walk-forward
meta-layer. The meta-layer uses only information from rows that precede the
current match. It combines the V9 base forecast, market/ML/Poisson experts,
and a smoothed historical class-prior expert. Expert weights are learned from
recent out-of-sample log loss with league-specific shrinkage toward the global
history. Final probabilities receive a walk-forward temperature calibration.
No current/future outcome is available to any prediction decision.
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
LEAGUE_SHRINK=80.0
PRIOR_ALPHA=1.0
EPS=1e-7
EXPERTS=("base","market","ml","poisson","prior")


def _p(row,prefix):
    return np.array([row[f"{prefix}Home"],row[f"{prefix}Draw"],row[f"{prefix}Away"]],dtype=float)


def _clean(p):
    p=np.nan_to_num(np.asarray(p,dtype=float),nan=1/3,posinf=1/3,neginf=1/3)
    p=np.clip(p,EPS,1.0)
    s=float(p.sum())
    return p/s if s>0 and math.isfinite(s) else np.ones(3)/3


def _ll(p,y):
    return -math.log(float(np.clip(p[int(y)],EPS,1.0)))


def _softmax_inverse(losses):
    a=-np.asarray(losses,dtype=float)
    a-=np.nanmax(a)
    w=np.exp(np.clip(a,-20,20))
    w[~np.isfinite(w)]=0.0
    s=float(w.sum())
    return w/s if s>0 else np.ones(len(losses))/len(losses)


def _temp(p,t):
    return _clean(np.power(np.clip(p,EPS,1.0),1.0/float(t)))


def _weighted_loss(probs,ys,half_life=90.0):
    if not ys:
        return 1.10
    P=np.asarray(probs,dtype=float)
    Y=np.asarray(ys,dtype=int)
    if len(Y)==0:
        return 1.10
    age=np.arange(len(Y)-1,-1,-1,dtype=float)
    w=np.exp(-math.log(2.0)*age/max(half_life,1.0))
    vals=-np.log(np.clip(P[np.arange(len(Y)),Y],EPS,1.0))
    return float(np.sum(w*vals)/np.sum(w))


def _best_temperature(hist_p,hist_y):
    if len(hist_y)<MIN_HISTORY:
        return 1.0
    P=np.asarray(hist_p,dtype=float)
    Y=np.asarray(hist_y,dtype=int)
    best_t,best_ll=1.0,float("inf")
    # Vectorized grid search: materially faster than a Python loop per row.
    for t in np.linspace(.72,1.55,18):
        Q=np.power(np.clip(P,EPS,1.0),1.0/float(t))
        Q/=Q.sum(axis=1,keepdims=True)
        loss=float(np.mean(-np.log(np.clip(Q[np.arange(len(Y)),Y],EPS,1.0))))
        if loss<best_ll:
            best_ll,best_t=loss,float(t)
    return best_t


def _league_key(value):
    if pd.isna(value):
        return "__UNKNOWN__"
    return str(value)


def _smoothed_prior(global_counts,league_counts,league_n):
    # Shrink a league's empirical class frequencies toward the global prior.
    g=np.asarray(global_counts,dtype=float)+PRIOR_ALPHA
    g=g/g.sum()
    if league_n<=0:
        return g
    l=np.asarray(league_counts,dtype=float)+PRIOR_ALPHA
    l=l/l.sum()
    a=float(league_n/(league_n+LEAGUE_SHRINK))
    return _clean(a*l+(1.0-a)*g)


def build_v12():
    if not SRC.exists():
        raise FileNotFoundError(f"required V9 output missing: {SRC}")
    df=pd.read_csv(SRC,low_memory=False)
    required=[
        "Predicted","Actual","HomeProb","DrawProb","AwayProb",
        "MarketHome","MarketDraw","MarketAway",
        "MLHome","MLDraw","MLAway",
        "PoissonHome","PoissonDraw","PoissonAway",
    ]
    missing=[c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"V12 fail-closed: missing columns: {missing}")
    if df.empty:
        raise RuntimeError("V12 fail-closed: V9 output is empty")
    if df["Actual"].isna().any():
        raise RuntimeError("V12 fail-closed: Actual contains missing values")

    df["__v12_order"]=np.arange(len(df))
    if "Date" in df.columns:
        df["__v12_date"]=pd.to_datetime(df["Date"],errors="coerce",dayfirst=True)
        # Stable chronological ordering; original row order breaks exact date ties.
        df=df.sort_values(["__v12_date","__v12_order"],kind="mergesort").reset_index(drop=True)

    out_probs=[]
    temps=[]
    weights_out=[]
    modes=[]

    hist_probs={k:[] for k in EXPERTS}
    hist_y=[]
    hist_league=[]
    league_probs={k:{} for k in EXPERTS}
    league_y={}
    global_counts=np.zeros(3,dtype=float)
    league_counts={}

    for i,row in df.iterrows():
        league=_league_key(row["League"] if "League" in df.columns else "__UNKNOWN__")
        base=_clean([row["HomeProb"],row["DrawProb"],row["AwayProb"]])
        comps={
            "base":base,
            "market":_clean(_p(row,"Market")),
            "ml":_clean(_p(row,"ML")),
            "poisson":_clean(_p(row,"Poisson")),
        }

        gc=global_counts.copy()
        lc=league_counts.get(league,np.zeros(3,dtype=float))
        prior=_smoothed_prior(gc,lc,float(sum(lc)))
        comps["prior"]=prior

        start=max(0,i-WINDOW)
        hy=hist_y[start:]
        gl_losses=[_weighted_loss(hist_probs[k][start:],hy) for k in EXPERTS]

        # League-specific expert skill is blended with global skill. Small
        # leagues therefore cannot dominate the meta decision.
        losses=[]
        for k,g_loss in zip(EXPERTS,gl_losses):
            lp=league_probs[k].get(league,[])
            ly=league_y.get(league,[])
            if len(ly)>=20:
                l_loss=_weighted_loss(lp[-WINDOW:],ly[-WINDOW:])
                n=float(len(ly[-WINDOW:]))
                a=n/(n+LEAGUE_SHRINK)
                losses.append((1.0-a)*g_loss+a*l_loss)
            else:
                losses.append(g_loss)
        weights=_softmax_inverse(losses)
        raw=_clean(sum(weights[j]*comps[k] for j,k in enumerate(EXPERTS)))

        # Temperature is learned only from earlier final V12 probabilities.
        hs=hist_probs["__final__"][max(0,i-TEMP_WINDOW):] if "__final__" in hist_probs else []
        hy_t=hist_y[max(0,i-TEMP_WINDOW):]
        t=_best_temperature(hs,hy_t)
        final=_temp(raw,t)

        out_probs.append(final)
        temps.append(t)
        weights_out.append(weights)
        modes.append("stacked_walk_forward")

        # IMPORTANT: current outcome is appended only after the prediction is
        # fully finalized. This is the core leakage barrier.
        for k in EXPERTS:
            hist_probs[k].append(comps[k])
            league_probs[k].setdefault(league,[]).append(comps[k])
        hist_probs.setdefault("__final__",[]).append(final)
        y=int(row["Actual"])
        hist_y.append(y)
        league_y.setdefault(league,[]).append(y)
        global_counts[y]+=1.0
        league_counts.setdefault(league,np.zeros(3,dtype=float))[y]+=1.0

    P=np.asarray(out_probs,dtype=float)
    W=np.asarray(weights_out,dtype=float)
    df["HomeProbV12"]=P[:,0]
    df["DrawProbV12"]=P[:,1]
    df["AwayProbV12"]=P[:,2]
    df["ConfidenceV12"]=P.max(axis=1)
    df["PredictedV12"]=P.argmax(axis=1)
    df["CorrectV12"]=(df["PredictedV12"].astype(int)==df["Actual"].astype(int)).astype(int)
    df["V12Temperature"]=temps
    df["V12Mode"]=modes
    for j,k in enumerate(EXPERTS):
        df[f"V12Weight_{k}"]=W[:,j]

    # Restore original file order after chronological prediction generation.
    df=df.sort_values("__v12_order",kind="mergesort").drop(
        columns=[c for c in ("__v12_order","__v12_date") if c in df.columns]
    )
    df.to_csv(OUT,index=False)

    y=df["Actual"].to_numpy(int)
    p=df[["HomeProbV12","DrawProbV12","AwayProbV12"]].to_numpy(float)
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1.0,atol=1e-8):
        raise RuntimeError("V12 fail-closed: invalid probability vector")
    pred=p.argmax(1)
    onehot=np.eye(3)[y]
    acc=float(np.mean(pred==y))
    logloss=float(np.mean([-math.log(np.clip(p[j,y[j]],EPS,1.0)) for j in range(len(y))]))
    brier=float(np.mean(np.sum((p-onehot)**2,axis=1)))
    summary={
        "Version":"V12",
        "Matches":len(df),
        "Accuracy":acc,
        "LogLoss":logloss,
        "Brier":brier,
        "MeanConfidence":float(p.max(1).mean()),
        "MeanTemperature":float(np.mean(temps)),
    }
    pd.DataFrame([summary]).to_csv("overall_summary_v12.csv",index=False)

    # Additional chronological diagnostics by competition/season.
    rows=[]
    for (league,season),g in df.groupby(["League","Season"],dropna=False,sort=False):
        yy=g["Actual"].to_numpy(int)
        pp=g[["HomeProbV12","DrawProbV12","AwayProbV12"]].to_numpy(float)
        oo=pp.argmax(1)
        rows.append({
            "League":league,"Season":season,"Matches":len(g),
            "Accuracy":float(np.mean(oo==yy)),
            "LogLoss":float(np.mean([-math.log(np.clip(pp[j,yy[j]],EPS,1.0)) for j in range(len(g))])),
            "Brier":float(np.mean(np.sum((pp-np.eye(3)[yy])**2,axis=1))),
        })
    pd.DataFrame(rows).to_csv("league_season_summary_v12.csv",index=False)
    print(f"[V12] wrote {OUT} rows={len(df)} accuracy={acc:.6f} logloss={logloss:.6f} brier={brier:.6f}")


def main():
    V12_DONE.unlink(missing_ok=True)
    backtest.main()
    if not V9_DONE.exists():
        raise RuntimeError("V12 not finalized: chronological V9 engine did not reach true completion")
    build_v12()
    V12_DONE.write_text("BACKTEST_COMPLETE_V12\n",encoding="utf-8")
    print("[V12] BACKTEST_COMPLETE_V12")


if __name__=="__main__":
    main()
