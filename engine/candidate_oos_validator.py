#!/usr/bin/env python3
"""Real leak-safe walk-forward candidate training and OOS validation.

Each candidate predicts a row using only strictly earlier rows. The current
Actual is appended to training history only after that row has been scored.
Candidate parameters materially change the model, feature set, or calibration.
"""
from __future__ import annotations
import hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

V9=Path('backtest_results_v9.csv'); V12=Path('backtest_results_v12.csv'); CAND=Path('candidate_models.json'); OUT=Path('candidate_evaluations.json')
EPS=1e-7; MIN_TRAIN=int(os.environ.get('CANDIDATE_MIN_TRAIN','260')); RETRAIN_EVERY=int(os.environ.get('CANDIDATE_RETRAIN_EVERY','150')); MAX_ROWS=int(os.environ.get('CANDIDATE_MAX_ROWS','0')); BUDGET=float(os.environ.get('MAX_RUNTIME_SECONDS','3300'))
CLASSES=np.array([0,1,2])

def idx(v):
    if isinstance(v,str):
        s=v.strip().upper()
        if s in ('H','D','A'): return ('H','D','A').index(s)
        if s in ('0','1','2'): return int(s)
    return int(v)

def clean(p):
    p=np.nan_to_num(np.asarray(p,float),nan=1/3,posinf=1/3,neginf=1/3); p=np.clip(p,EPS,1.0); s=p.sum(); return p/s if s>0 else np.ones(3)/3

def _hash(s):
    return (int(hashlib.md5(str(s).encode()).hexdigest()[:8],16)%1000000)/1000000.0

def feature_frame(df,kind):
    def arr(prefix): return np.column_stack([pd.to_numeric(df[f'{prefix}Home'],errors='coerce').fillna(1/3),pd.to_numeric(df[f'{prefix}Draw'],errors='coerce').fillna(1/3),pd.to_numeric(df[f'{prefix}Away'],errors='coerce').fillna(1/3)])
    base=arr('') if all(f'{x}' in df.columns for x in ('HomeProb','DrawProb','AwayProb')) else None
    market=np.column_stack([df['MarketHome'],df['MarketDraw'],df['MarketAway']]).astype(float)
    ml=np.column_stack([df['MLHome'],df['MLDraw'],df['MLAway']]).astype(float)
    poi=np.column_stack([df['PoissonHome'],df['PoissonDraw'],df['PoissonAway']]).astype(float)
    X=[base,market,ml,poi]
    X.extend([base-market,base-ml,base-poi])
    X.append(np.column_stack([base.max(1),np.abs(base[:,0]-base[:,2]),base[:,1]]))
    if kind in ('context','draw_specialist'):
        league=np.array([_hash(x) for x in df.get('League','__UNKNOWN__')])[:,None]
        month=np.zeros((len(df),2))
        if 'Date' in df:
            d=pd.to_datetime(df['Date'],errors='coerce',dayfirst=True); m=d.dt.month.fillna(0).to_numpy(float); month[:,0]=np.sin(2*np.pi*m/12); month[:,1]=np.cos(2*np.pi*m/12)
        X.extend([league,month])
    if kind=='market_gap': X=[base,market,base-market,base-ml,base-poi]
    return np.nan_to_num(np.column_stack(X),nan=1/3,posinf=1/3,neginf=1/3)

def metrics(P,y):
    P=np.asarray(P); y=np.asarray(y,int); acc=float(np.mean(P.argmax(1)==y)); ll=float(np.mean(-np.log(np.clip(P[np.arange(len(y)),y],EPS,1.0))); br=float(np.mean(np.sum((P-np.eye(3)[y])**2,axis=1))); return acc,ll,br

def run_candidate(cid,kind,params,df,baseline,deadline):
    X=feature_frame(df,kind); y=np.array([idx(v) for v in df['Actual']],dtype=int); n=len(df); pred=np.zeros((n,3)); last=None; trained=0; start=time.monotonic(); alpha=float(params.get('alpha',0.0005)); class_weight=None
    if kind=='draw_specialist': class_weight={0:1.0,1:float(params.get('draw_weight',1.15)),2:1.0}
    for i in range(n):
        if time.monotonic()>deadline: return {'id':cid,'kind':kind,'params':params,'status':'timeout'}
        if i<MIN_TRAIN:
            pred[i]=clean(df.iloc[i][['HomeProb','DrawProb','AwayProb']].to_numpy(float)); continue
        if last is None or (i-MIN_TRAIN)%RETRAIN_EVERY==0:
            lo=max(0,i-1800); model=SGDClassifier(loss='log_loss',alpha=alpha,max_iter=250,tol=1e-4,random_state=42,class_weight=class_weight)
            sw=None
            if 'half_life' in params:
                age=np.arange(i-lo-1,-1,-1,dtype=float); sw=np.exp(-math.log(2)*age/max(float(params['half_life']),1.0))
            model.fit(X[lo:i],y[lo:i],sample_weight=sw); last=model; trained+=1
        q=clean(last.predict_proba(X[i:i+1])[0])
        if 'temperature' in params: q=clean(np.power(np.clip(q,EPS,1.0),1.0/float(params['temperature'])))
        pred[i]=q
    e=np.arange(MIN_TRAIN,n); ca,cl,cb=metrics(pred[e],y[e]); ba,bl,bb=metrics(baseline[e],y[e])
    return {'id':cid,'kind':kind,'params':params,'status':'validated','oos_rows':int(len(e)),'trained_models':trained,'candidate':{'accuracy':ca,'logloss':cl,'brier':cb},'baseline':{'accuracy':ba,'logloss':bl,'brier':bb},'delta':{'accuracy':ca-ba,'logloss':cl-bl,'brier':cb-bb},'adoptable':bool(ca>=ba and cl<=bl and (ca>ba or cl<bl))}

def main():
    for p in (V9,V12,CAND):
        if not p.exists(): raise SystemExit(f'candidate OOS prerequisite missing: {p}')
    df=pd.read_csv(V9,low_memory=False); v12=pd.read_csv(V12,low_memory=False)
    if len(df)!=len(v12): raise SystemExit('V9/V12 row count mismatch')
    req=['Actual','HomeProb','DrawProb','AwayProb','MarketHome','MarketDraw','MarketAway','MLHome','MLDraw','MLAway','PoissonHome','PoissonDraw','PoissonAway']
    miss=[c for c in req if c not in df.columns]
    if miss: raise SystemExit(f'missing candidate features: {miss}')
    baseline=v12[['HomeProbV12','DrawProbV12','AwayProbV12']].to_numpy(float)
    if MAX_ROWS>0: df=df.iloc[:MAX_ROWS].copy(); baseline=baseline[:MAX_ROWS]
    candidates=json.loads(CAND.read_text(encoding='utf-8')).get('candidates',[])
    deadline=time.monotonic()+BUDGET*0.90; results=[]
    for c in candidates[:6]:
        r=run_candidate(c['id'],c.get('kind','stacking'),c.get('params',{}),df,baseline,deadline); results.append(r)
        if r.get('status')=='timeout': break
    OUT.write_text(json.dumps({'version':'V12-candidate-oos-v2','leak_policy':'prior_rows_only','min_train':MIN_TRAIN,'retrain_every':RETRAIN_EVERY,'candidates':results},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(json.loads(OUT.read_text(encoding='utf-8')),ensure_ascii=False))

if __name__=='__main__': main()
