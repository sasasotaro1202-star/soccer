#!/usr/bin/env python3
"""Soccer condition analysis inspired by the Baseball evaluation module.

All conditions are derived from pre-match prediction fields already present in
V9/V12 output. No post-match stat is used to construct a condition.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd

SRC=Path("backtest_results_v12.csv"); OUT=Path("condition_analysis_v12.json")

def acc(g): return float(g["CorrectV12"].mean()) if len(g) else 0.0

def summarize(df,key):
    out={}
    for value,g in df.groupby(key,dropna=False,sort=False):
        label="unknown" if pd.isna(value) else str(value)
        out[label]={"samples":int(len(g)),"accuracy":acc(g),"mean_confidence":float(g["ConfidenceV12"].mean())}
    return dict(sorted(out.items(),key=lambda x:(x[1]["accuracy"],-x[1]["samples"])))

def main():
    if not SRC.exists(): raise SystemExit("V12 result missing")
    df=pd.read_csv(SRC,low_memory=False)
    pcols=["HomeProbV12","DrawProbV12","AwayProbV12"]
    df["favored_class"]=df[pcols].idxmax(axis=1).str.replace("ProbV12","")
    df["confidence_bin"]=pd.cut(df["ConfidenceV12"],[-1,.45,.50,.55,.60,.65,.70,.75,.80,1.01],labels=["<.45",".45-.50",".50-.55",".55-.60",".60-.65",".65-.70",".70-.75",".75-.80",".80+"])
    # Market disagreement is a useful pre-match tactical proxy: it identifies
    # matches where the learned football model and market imply different game
    # states without using the eventual result.
    m=df[["MarketHome","MarketDraw","MarketAway"]].copy(); v=df[pcols]
    df["market_disagreement"]=(v.to_numpy()-m.to_numpy())**2
    df["market_disagreement"]=df["market_disagreement"].sum(axis=1).pow(0.5)
    df["disagreement_bin"]=pd.cut(df["market_disagreement"],[ -1,.03,.06,.10,.15,10],labels=["very_low","low","medium","high","very_high"])
    # Pre-match model shape: draw probability and home/away imbalance.
    df["draw_band"]=pd.cut(df["DrawProbV12"],[ -1,.20,.25,.30,.35,.40,1.01],labels=["<.20",".20-.25",".25-.30",".30-.35",".35-.40",".40+"])
    df["home_edge_band"]=pd.cut(df["HomeProbV12"]-df["AwayProbV12"],[-1,-.30,-.15,-.05,.05,.15,.30,1],labels=["<-30","-30--15","-15--5","-5-5","5-15","15-30",">30"])
    result={"version":"V12-soccer-condition-analysis-v1","policy":"pre_match_fields_only","dimensions":{}}
    keys=["League","Season","favored_class","confidence_bin","disagreement_bin","draw_band","home_edge_band"]
    for k in keys:
        if k in df.columns: result["dimensions"][k]=summarize(df,k)
    # Cross-condition slices: league x confidence and league x market disagreement.
    for k1,k2,name in [("League","confidence_bin","league_confidence"),("League","disagreement_bin","league_disagreement"),("League","favored_class","league_favored_class")]:
        if k1 in df.columns and k2 in df.columns:
            result["dimensions"][name]={}
            for (a,b),g in df.groupby([k1,k2],dropna=False,sort=False):
                result["dimensions"][name][f"{a}|{b}"]={"samples":int(len(g)),"accuracy":acc(g),"mean_confidence":float(g["ConfidenceV12"].mean())}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__": main()
