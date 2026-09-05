#!/usr/bin/env python3
"""Leak-safe evaluation utilities for Soccer V12/V9 outputs."""
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
EPS=1e-15
ROOT=Path('.')

def _f(r,k,d=None):
    try:
        v=r.get(k,d)
        return d if v in (None,'') else float(v)
    except (TypeError,ValueError): return d

def _idx(v):
    if isinstance(v,str):
        s=v.strip().upper()
        if s in ('H','D','A'): return ('H','D','A').index(s)
        if s in ('0','1','2'): return int(s)
    try: return int(v)
    except (TypeError,ValueError): return None

def _actual(r):
    a=r.get('Actual') or r.get('actual') or r.get('Result') or r.get('result')
    i=_idx(a)
    if i is not None: return ('H','D','A')[i]
    h,aa=_f(r,'FTHG'),_f(r,'FTAG')
    if h is not None and aa is not None: return 'H' if h>aa else 'A' if h<aa else 'D'
    return None

def _pred(r):
    for k in ('PredictedV12','Predicted','prediction'):
        if k in r:
            i=_idx(r.get(k))
            if i is not None: return ('H','D','A')[i]
    return None

def _probs(r):
    for trio in [('HomeProbV12','DrawProbV12','AwayProbV12'),('HomeProb','DrawProb','AwayProb')]:
        vals=[_f(r,k) for k in trio]
        if all(v is not None and math.isfinite(v) for v in vals):
            s=sum(vals)
            if s>0: return [max(EPS,v/s) for v in vals]
    return None

def _brier(rows):
    vals=[]
    for r in rows:
        p,a=_probs(r),_actual(r)
        if p is None or a is None: continue
        y=[float(a==x) for x in ('H','D','A')]; vals.append(sum((p[i]-y[i])**2 for i in range(3)))
    return sum(vals)/len(vals) if vals else None

def _logloss(rows):
    vals=[]
    for r in rows:
        p,a=_probs(r),_actual(r)
        if p is not None and a is not None: vals.append(-math.log(max(EPS,p[('H','D','A').index(a)])))
    return sum(vals)/len(vals) if vals else None

def _calibration(rows,bins=10):
    bucket=defaultdict(lambda:[0,0.0,0.0])
    for r in rows:
        p,a=_probs(r),_actual(r)
        if p is None or a is None: continue
        c=max(p); ok=int(_pred(r)==a); b=min(bins-1,int(c*bins)); bucket[b][0]+=1; bucket[b][1]+=c; bucket[b][2]+=ok
    return {'bins':[{'bin':b,'n':n,'mean_confidence':ps/n,'accuracy':ys/n} for b,(n,ps,ys) in sorted(bucket.items())]}

def _score_metrics(rows):
    errors=[]; exact=0;n=0
    for r in rows:
        ph,pa,ah,aa=[_f(r,k) for k in ('PredHomeGoals','PredAwayGoals','FTHG','FTAG')]
        if None in (ph,pa,ah,aa): continue
        n+=1; errors.append((abs(ph-ah)+abs(pa-aa))/2.0); exact+=int(round(ph)==round(ah) and round(pa)==round(aa))
    return {'sample_size':n,'score_mae':sum(errors)/n if n else None,'exact_score_accuracy':exact/n if n else None}

def _group_accuracy(rows,key):
    groups=defaultdict(list)
    for r in rows:
        if r.get(key) not in (None,''): groups[str(r[key])].append(r)
    out={}
    for g,rs in groups.items():
        valid=[(_pred(r),_actual(r)) for r in rs]; valid=[x for x in valid if x[0] and x[1]]
        out[g]={'matches':len(valid),'accuracy':sum(p==a for p,a in valid)/len(valid) if valid else None,'logloss':_logloss(rs),'brier':_brier(rs)}
    return out

def evaluate(data):
    valid=[(r,_pred(r),_actual(r)) for r in data]; valid=[x for x in valid if x[1] and x[2]]
    accuracy=sum(p==a for _,p,a in valid)/len(valid) if valid else None
    return {'version':'V12-evaluator-v3','sample_size':len(data),'valid_1x2':len(valid),'accuracy':accuracy,'class_metrics':{c:{'actual_n':sum(a==c for _,_,a in valid),'recall':sum(p==a for _,p,a in valid if a==c)/sum(a==c for _,_,a in valid) if sum(a==c for _,_,a in valid) else None} for c in ('H','D','A')},'logloss':_logloss(data),'brier':_brier(data),'calibration':_calibration(data),'score':_score_metrics(data),'by_league':_group_accuracy(data,'League'),'by_season':_group_accuracy(data,'Season')}

def load_csv(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    source=next((p for p in (ROOT/'backtest_results_v12.csv',ROOT/'backtest_results_v9.csv',ROOT/'backtest_results.csv') if p.exists()),None)
    if source is None: raise SystemExit('no prediction output available')
    result=evaluate(load_csv(source)); (ROOT/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
