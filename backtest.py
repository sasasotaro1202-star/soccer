#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Soccer Backtest V7 - leak-safe walk-forward production version.
1X2: chronological ML ensemble + closing market + dynamic Elo/team form.
Score: dynamic Poisson + Dixon-Coles.
MOM: optional, cache-first FotMob enrichment; never uses current-match data before prediction.
"""
from __future__ import annotations
import json, math, pickle, time, warnings
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
LEAGUES={"E0":"Premier League","E1":"Championship","D1":"Bundesliga","I1":"Serie A","SP1":"La Liga","F1":"Ligue 1"}
SEASONS=[2020,2021,2022,2023,2024,2025]
BASE_URL="https://www.football-data.co.uk/mmz4281/{folder}/{league}.csv"
OUT=Path("."); CACHE=OUT/"cache"; FM_CACHE=CACHE/"fotmob"; CHECKPOINT=OUT/"backtest_checkpoint_v7.pkl"
MAX_RUNTIME=27*60
MIN_TRAIN=260; MAX_TRAIN=2200; RETRAIN_EVERY=45; VALID_FRAC=.20; RANDOM_STATE=42
WALK_FOLDS=4; CAL_GRID=np.arange(.70,1.46,.05); MIN_CAL=45
TIME_DECAY_HALFLIFE=650.0; MARKET_BLEND_DEFAULT=(.55,.25,.20)
ENABLE_MOM=True; FOTMOB_CACHE_ONLY=True; REQUEST_TIMEOUT=10
START=time.time()

# ---------------- HELPERS ----------------
def left(): return MAX_RUNTIME-(time.time()-START)
def sf(x,default=np.nan):
    try:
        if x is None or pd.isna(x): return default
        s=str(x).strip().replace(",",".")
        if s in ("","-","nan","NaN","None"): return default
        return float(s)
    except Exception: return default

def n3(x):
    a=np.nan_to_num(np.asarray(x,dtype=float),nan=1/3,posinf=1/3,neginf=1/3); a=np.maximum(a,1e-10); return a/a.sum()
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
def team_name(x):
    if pd.isna(x): return ""
    return {"Man United":"Manchester United","Man Utd":"Manchester United","Man City":"Manchester City","Spurs":"Tottenham Hotspur","Tottenham":"Tottenham Hotspur","Nott'm Forest":"Nottingham Forest","Newcastle":"Newcastle United","West Ham":"West Ham United","Wolves":"Wolverhampton Wanderers","Leicester":"Leicester City","Leeds":"Leeds United","Brighton":"Brighton & Hove Albion","Sheffield Utd":"Sheffield United","Norwich":"Norwich City","QPR":"Queens Park Rangers"}.get(str(x).strip(),str(x).strip())
def folder(y): return f"{str(y)[-2:]}{str(y+1)[-2:]}"

# ---------------- DATA ----------------
def load_one(code,y):
    try:
        d=pd.read_csv(BASE_URL.format(folder=folder(y),league=code)); d["League"]=code; d["SeasonStart"]=y; d["Season"]=f"{y}/{str(y+1)[-2:]}"; return d
    except Exception as e:
        print(f"[WARN] {code} {y}: {e}"); return pd.DataFrame()

def load_data():
    fs=[]
    for code in LEAGUES:
        for y in SEASONS:
            if left()<90: break
            d=load_one(code,y)
            if not d.empty: fs.append(d)
    if not fs: raise RuntimeError("No Football-Data data loaded")
    d=pd.concat(fs,ignore_index=True)
    d["DateParsed"]=pd.to_datetime(d["Date"],dayfirst=True,errors="coerce")
    d["HomeTeam"]=d["HomeTeam"].map(team_name); d["AwayTeam"]=d["AwayTeam"].map(team_name)
    d["FTHG"]=pd.to_numeric(d["FTHG"],errors="coerce"); d["FTAG"]=pd.to_numeric(d["FTAG"],errors="coerce")
    d=d.dropna(subset=["DateParsed","HomeTeam","AwayTeam","FTHG","FTAG"]).copy()
    d["FTHG"]=d["FTHG"].astype(int); d["FTAG"]=d["FTAG"].astype(int)
    d["Result"]=np.where(d.FTHG>d.FTAG,0,np.where(d.FTHG==d.FTAG,1,2))
    return d.sort_values(["League","SeasonStart","DateParsed"],kind="stable").reset_index(drop=True)

# ---------------- ODDS ----------------
ODDS=[("B365C",("B365CH","B365CD","B365CA")),("BWC",("BWCH","BWCD","BWCA")),("IWC",("IWCH","IWCD","IWCA")),("PSC",("PSCH","PSCD","PSCA")),("WHC",("WHCH","WHCD","WHCA")),("VCC",("VCCH","VCCD","VCCA")),("MaxC",("MaxCH","MaxCD","MaxCA")),("AvgC",("AvgCH","AvgCD","AvgCA")),("B365",("B365H","B365D","B365A")),("BW",("BWH","BWD","BWA")),("IW",("IWH","IWD","IWA")),("PS",("PSH","PSD","PSA")),("WH",("WHH","WHD","WHA")),("VC",("VCH","VCD","VCA")),("Max",("MaxH","MaxD","MaxA")),("Avg",("AvgH","AvgD","AvgA"))]
def odds(row):
    # Prefer closing odds; use non-closing only when no valid closing market exists.
    closing, fallback = [], []
    for name,cs in ODDS:
        v=[sf(row.get(c,np.nan)) for c in cs]
        if all(np.isfinite(z) and z>1.01 for z in v):
            q=n3(1/np.asarray(v,float))
            (closing if name.endswith("C") else fallback).append(q)
    a=np.vstack(closing if closing else fallback) if (closing or fallback) else None
    if a is None:
        p=np.ones(3)/3; return np.array([*p,p.max(),1.,0.,0.],float)
    # Median reduces bookmaker outliers; dispersion becomes a reliability feature.
    p=n3(np.median(a,axis=0))
    ent=-np.sum(p*np.log(np.maximum(p,1e-12)))/np.log(3)
    disp=float(np.mean(np.std(a,axis=0))) if len(a)>1 else 0.
    return np.array([p[0],p[1],p[2],p.max(),ent,disp,1.],float)

# ---------------- TEAM / LEAGUE STATE ----------------
class Team:
    def __init__(self,elo=1500.): self.__init_state(elo)
    def __init_state(self,elo):
        self.elo=float(elo); self.matches=self.gf=self.ga=self.points=0.; self.hm=self.hgf=self.hga=self.hp=0.; self.am=self.agf=self.aga=self.ap=0.; self.lp=deque(maxlen=10); self.lgf=deque(maxlen=10); self.lga=deque(maxlen=10); self.egf=self.ega=self.ep=None; self.hegf=self.hega=self.aegf=self.aega=None; self.stats={k:None for k in ("shots","sot","corners","fouls","yellow","red")}
    def reset(self): self.__init_state(1500+.72*(self.elo-1500))
    @staticmethod
    def ema(old,v,a=.25): return v if old is None else a*v+(1-a)*old
    def update(self,gf,ga,pts,venue,stats):
        self.matches+=1; self.gf+=gf; self.ga+=ga; self.points+=pts; self.lp.append(pts); self.lgf.append(gf); self.lga.append(ga); self.egf=self.ema(self.egf,gf); self.ega=self.ema(self.ega,ga); self.ep=self.ema(self.ep,pts)
        if venue=="H": self.hm+=1; self.hgf+=gf; self.hga+=ga; self.hp+=pts; self.hegf=self.ema(self.hegf,gf); self.hega=self.ema(self.hega,ga)
        else: self.am+=1; self.agf+=gf; self.aga+=ga; self.ap+=pts; self.aegf=self.ema(self.aegf,gf); self.aega=self.ema(self.aega,ga)
        for k,v in stats.items():
            if np.isfinite(v): self.stats[k]=self.ema(self.stats[k],v,.20)
    def ppg(self): return self.points/max(1,self.matches)
    def gfpg(self): return self.gf/max(1,self.matches)
    def gapg(self): return self.ga/max(1,self.matches)

class League:
    def __init__(self): self.teams={}; self.hg=deque(maxlen=500); self.ag=deque(maxlen=500); self.hp=deque(maxlen=500)
    def team(self,n):
        if n not in self.teams: self.teams[n]=Team()
        return self.teams[n]
    def reset(self):
        # New season: regress Elo toward league mean, reset short-term form/stats.
        for t in self.teams.values(): t.reset()
    def av(self): return (max(.75,np.mean(self.hg) if self.hg else 1.45),max(.60,np.mean(self.ag) if self.ag else 1.15),np.mean(self.hp) if self.hp else 1.55)
    def update(self,hg,ag,hp): self.hg.append(hg); self.ag.append(ag); self.hp.append(hp)

def tf(t,venue,lh,la):
    if not t.matches: return [1500,0,0,0,0,1.3,1.3,1.1,1.1,lh if venue=="H" else la,la if venue=="H" else lh,0,0,12,4,5,11]
    if venue=="H": m,gf,ga,pp,egf,ega=t.hm,t.hgf,t.hga,t.hp/max(1,t.hm),t.hegf or lh,t.hega or la
    else: m,gf,ga,pp,egf,ega=t.am,t.agf,t.aga,t.ap/max(1,t.am),t.aegf or la,t.aega or lh
    return [t.elo,t.elo-1500,t.ppg(),t.gfpg(),t.gapg(),np.mean(t.lp) if t.lp else 1.3,np.mean(t.lgf) if t.lgf else 1.3,np.mean(t.lga) if t.lga else 1.1,pp,gf/max(1,m),ga/max(1,m),egf,ega,t.ep or 1.3,t.stats["shots"] or 12,t.stats["sot"] or 4,t.stats["corners"] or 5]

def features(row,h,a,L):
    lh,la,_=L.av(); o=odds(row); he=h.hegf or lh; ae=a.aegf or la; hd=h.hega or la; ad=a.aega or lh; ed=h.elo-a.elo; pd=h.ppg()-a.ppg()
    z=[ed,pd,h.gfpg()-a.gfpg(),a.gapg()-h.gapg(),he-ae,hd-ad,sigmoid(ed/180),sigmoid(pd),he/max(lh,.5),ae/max(la,.5),hd/max(la,.5),ad/max(lh,.5)]
    return np.asarray(tf(h,"H",lh,la)+tf(a,"A",lh,la)+z+o.tolist(),dtype=float)

def elo_update(h,a,result,hg,ag):
    e=sigmoid((h.elo+55-a.elo)/400); actual=1 if result==0 else .5 if result==1 else 0; k=18*(1+math.log1p(max(1,abs(hg-ag)))); d=k*(actual-e); h.elo+=d; a.elo-=d

# ---------------- SCORE ----------------
def pois(k,l): return 0 if l<=0 else math.exp(-l)*l**k/math.factorial(k)
def tau(h,a,lh,la,r=-.08):
    if h==0 and a==0:return 1-lh*la*r
    if h==0 and a==1:return 1+lh*r
    if h==1 and a==0:return 1+la*r
    if h==1 and a==1:return 1-r
    return 1
def score_model(h,a,L):
    lh0,la0,_=L.av()
    ha=h.hegf if h.hegf is not None else lh0
    hd=h.hega if h.hega is not None else la0
    aa=a.aegf if a.aegf is not None else la0
    ad=a.aega if a.aega is not None else lh0
    # Multiplicative attack/defence strengths, heavily shrunk early in a season.
    hm=max(1,h.hm); am=max(1,a.am)
    h_att=.55*(ha/lh0)+.45*((h.hgf/max(1,h.hm))/lh0)
    h_def=.55*(hd/la0)+.45*((h.hga/max(1,h.hm))/la0)
    a_att=.55*(aa/la0)+.45*((a.agf/max(1,a.am))/la0)
    a_def=.55*(ad/lh0)+.45*((a.aga/max(1,a.am))/lh0)
    ed=(h.elo+55-a.elo)/400
    form=(h.ppg()-a.ppg())/2
    lh=lh0*np.clip(h_att,.55,1.75)*np.clip(a_def,.55,1.75)
    la=la0*np.clip(a_att,.55,1.75)*np.clip(h_def,.55,1.75)
    lh*=np.clip(1+.085*ed+.025*form,.82,1.20)
    la*=np.clip(1-.060*ed-.018*form,.84,1.18)
    lh=float(np.clip(lh,.20,3.80)); la=float(np.clip(la,.15,3.40))
    m=np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            m[i,j]=pois(i,lh)*pois(j,la)*tau(i,j,lh,la,r=-.065)
    m/=max(m.sum(),1e-12)
    p=[sum(m[i,j] for i in range(8) for j in range(8) if i>j),
       sum(m[i,j] for i in range(8) for j in range(8) if i==j),
       sum(m[i,j] for i in range(8) for j in range(8) if i<j)]
    top=sorted([(i,j,float(m[i,j])) for i in range(8) for j in range(8)],
               key=lambda x:x[2],reverse=True)[:3]
    return n3(p),top,lh,la

# ---------------- ML ----------------
def models():
    return {
        "Logistic":Pipeline([
            ("scale",StandardScaler()),
            ("clf",LogisticRegression(C=.18,max_iter=900,random_state=RANDOM_STATE))
        ]),
        "ExtraTrees":ExtraTreesClassifier(
            n_estimators=360,min_samples_leaf=8,max_features=.70,
            class_weight="balanced_subsample",random_state=RANDOM_STATE,n_jobs=-1
        ),
        "HistGB":HistGradientBoostingClassifier(
            max_iter=220,learning_rate=.035,max_leaf_nodes=15,
            min_samples_leaf=18,l2_regularization=2.0,random_state=RANDOM_STATE
        )
    }

def align(model,p):
    q=np.zeros(3)
    for i,c in enumerate(model.classes_):
        if int(c) in (0,1,2): q[int(c)]=p[i]
    return n3(q)

def temp_scale(p,t):
    z=np.log(np.maximum(n3(p),1e-12))/float(t)
    z-=np.max(z); q=np.exp(z); return n3(q)

def _walk_splits(n):
    # Multiple chronological OOS tails, never random/shuffled.
    if n < 180: return []
    tail=max(35,int(n*.12)); splits=[]
    for k in range(WALK_FOLDS,0,-1):
        end=n-(k-1)*tail
        val_start=max(100,end-tail)
        if val_start>=end or val_start<100: continue
        splits.append((val_start,end))
    return splits

def fit_ensemble(hist):
    if len(hist)<MIN_TRAIN:return None
    d=hist[-MAX_TRAIN:]
    X=np.vstack([z["x"] for z in d]); y=np.array([z["y"] for z in d],int)
    splits=_walk_splits(len(y))
    if not splits:return None
    losses=defaultdict(list); oos_pred=defaultdict(list); oos_y=[]
    for vs,ve in splits:
        Xtr,ytr=X[:vs],y[:vs]
        for name in models():
            try:
                m=models()[name]
                # Exponential recency weighting; older matches matter less.
                ages=np.arange(len(ytr))[::-1]
                w=np.exp(-np.log(2)*ages/TIME_DECAY_HALFLIFE)
                m.fit(Xtr,ytr,**({"clf__sample_weight":w} if name=="Logistic" else {}))
                pp=np.vstack([align(m,q) for q in m.predict_proba(X[vs:ve])])
                losses[name].append(log_loss(y[vs:ve],pp,labels=[0,1,2]))
                oos_pred[name].append(pp)
            except Exception as e:
                print(f"[WARN] OOS {name}: {e}")
        oos_y.extend(y[vs:ve].tolist())
    if not oos_y:return None
    agg_losses={n:float(np.mean(v)) for n,v in losses.items() if v}
    names=list(agg_losses)
    if not names:return None
    # OOS LogLoss weights, clipped to prevent one lucky fold dominating.
    raw=np.exp(-(np.array([agg_losses[n] for n in names])-min(agg_losses.values()))/.10)
    raw=np.clip(raw,.12,.65); raw/=raw.sum()
    # Temperature calibrated per model from pooled OOS predictions.
    temps={}
    for n in names:
        pp=np.vstack(oos_pred[n]); yy=np.asarray(oos_y[:len(pp)])
        best=(1.,1e9)
        for t in CAL_GRID:
            ll=log_loss(yy,np.vstack([temp_scale(q,t) for q in pp]),labels=[0,1,2])
            if ll<best[1]:best=(float(t),float(ll))
        temps[n]=best[0]
    fm={}
    for n in names:
        try:
            m=models()[n]
            ages=np.arange(len(y))[::-1]
            w=np.exp(-np.log(2)*ages/TIME_DECAY_HALFLIFE)
            m.fit(X,y,**({"clf__sample_weight":w} if n=="Logistic" else {}))
            fm[n]=m
        except Exception as e: print(f"[WARN] final fit {n}: {e}")
    return {"models":fm,"weights":dict(zip(names,raw)),"validation_logloss":agg_losses,"temperatures":temps}

def predict_ml(bundle,x):
    if not bundle:return np.ones(3)/3
    ps=[]; ws=[]
    for n,m in bundle["models"].items():
        try:
            q=align(m,m.predict_proba(x.reshape(1,-1))[0])
            q=temp_scale(q,bundle.get("temperatures",{}).get(n,1.))
            ps.append(q); ws.append(bundle["weights"].get(n,0))
        except Exception: pass
    if not ps:return np.ones(3)/3
    w=np.asarray(ws,float); w/=max(w.sum(),1e-12)
    return n3(np.average(np.vstack(ps),axis=0,weights=w))

def optimize_blend(hist):
    # True chronological OOS optimization across three independent experts.
    if len(hist)<300:return MARKET_BLEND_DEFAULT
    d=hist[-MAX_TRAIN:]
    y=np.array([z["y"] for z in d],int)
    X=np.vstack([z["x"] for z in d])
    splits=_walk_splits(len(y)); rows=[]
    for vs,ve in splits:
        m=Pipeline([("scale",StandardScaler()),
                    ("clf",LogisticRegression(C=.18,max_iter=800,random_state=RANDOM_STATE))])
        try:
            ages=np.arange(vs)[::-1]
            w=np.exp(-np.log(2)*ages/TIME_DECAY_HALFLIFE)
            m.fit(X[:vs],y[:vs],clf__sample_weight=w)
            ml=np.vstack([align(m,q) for q in m.predict_proba(X[vs:ve])])
            market=np.vstack([n3(z["market"]) for z in d[vs:ve]])
            pois_p=np.vstack([n3(z["poisson"]) for z in d[vs:ve]])
            rows.append((y[vs:ve],ml,market,pois_p))
        except Exception as e: print(f"[WARN] blend OOS: {e}")
    if not rows:return MARKET_BLEND_DEFAULT
    yy=np.concatenate([r[0] for r in rows])
    ml=np.vstack([r[1] for r in rows])
    mk=np.vstack([r[2] for r in rows])
    pp=np.vstack([r[3] for r in rows])
    best=(*MARKET_BLEND_DEFAULT,1e9)
    # Coarse simplex grid, then select the lowest OOS LogLoss.
    for lw in np.arange(.20,.81,.05):
        for mw in np.arange(.10,.71,.05):
            sw=1-lw-mw
            if sw<.05: continue
            pred=n3_rows(lw*ml+mw*mk+sw*pp)
            ll=log_loss(yy,pred,labels=[0,1,2])
            if ll<best[3]:best=(float(lw),float(mw),float(sw),float(ll))
    return best[:3]

# ---------------- MOM / FOTMOB ----------------
SESSION=requests.Session(); SESSION.headers.update({"User-Agent":"Mozilla/5.0 SoccerBacktest/Production"})
def fm_path(mid): FM_CACHE.mkdir(parents=True,exist_ok=True); return FM_CACHE/f"{mid}.json"
def fetch_fm(mid):
    p=fm_path(mid)
    if p.exists():
        try:return json.loads(p.read_text(encoding="utf-8"))
        except Exception:pass
    if FOTMOB_CACHE_ONLY:return None
    try:
        r=SESSION.get(f"https://www.fotmob.com/api/data/matchDetails?matchId={mid}",timeout=REQUEST_TIMEOUT)
        if r.status_code!=200:return None
        x=r.json(); p.write_text(json.dumps(x,ensure_ascii=False),encoding="utf-8"); return x
    except Exception:return None
def find_mom(x):
    if isinstance(x,dict):
        for k,v in x.items():
            if "playerofthematch" in str(k).lower() or str(k).lower() in ("mom","motm"):
                if isinstance(v,str):return v
                if isinstance(v,dict):
                    for z in ("name","playerName","title"):
                        if v.get(z):return str(v[z])
            q=find_mom(v)
            if q:return q
    elif isinstance(x,list):
        for v in x:
            q=find_mom(v)
            if q:return q
    return None
def players_from_fm(x):
    out={}
    def add(o,side):
        if not isinstance(o,dict) or side not in ("H","A"):return
        p=o.get("player") if isinstance(o.get("player"),dict) else o; name=p.get("name") or p.get("playerName")
        if not name:return
        s=o.get("stats") or o.get("statistics") or o
        def val(*keys):
            for k in keys:
                v=s.get(k) if isinstance(s,dict) else None
                if isinstance(v,dict):v=v.get("value",v.get("displayValue"))
                v=sf(v)
                if np.isfinite(v):return v
            return np.nan
        out[(side,str(name))]={"name":str(name),"side":side,"rating":val("rating","FotMobRating"),"goals":val("goals","goal"),"assists":val("assists","assist"),"xg":val("expectedGoals","xG"),"xa":val("expectedAssists","xA"),"minutes":val("minutesPlayed","minutes"),"key_passes":val("keyPasses","key_passes")}
    def walk(o,side=None):
        if isinstance(o,dict):
            for k,v in o.items():
                s=side; kl=str(k).lower()
                if kl in ("home","hometeam","homeplayers"):s="H"
                if kl in ("away","awayteam","awayplayers"):s="A"
                if isinstance(v,list) and kl in ("players","lineups","starters","substitutes","bench"):
                    for z in v:add(z,s)
                walk(v,s)
        elif isinstance(o,list):
            for v in o:walk(v,side)
    walk(x); return list(out.values())
class Player:
    def __init__(self):self.matches=self.starts=self.minutes=self.goals=self.assists=self.xg=self.xa=self.key=0.; self.rating=6.5
    def update(self,p):
        self.matches+=1; self.minutes+=p.get("minutes",0) if np.isfinite(p.get("minutes",np.nan)) else 0
        for k in ("goals","assists","xg","xa","key_passes"):
            v=p.get(k,np.nan)
            if np.isfinite(v):setattr(self,k,getattr(self,k)+v)
        r=p.get("rating",np.nan)
        if np.isfinite(r):self.rating=.3*r+.7*self.rating
def mom_value(p,elo,opp,winp):
    g=max(1,p.matches); rating=np.clip((p.rating-6.2)/1.2,0,1.5); mins=np.clip((p.minutes/g)/80,0,1.25); prod=np.clip(.38*p.goals/g+.28*p.assists/g+.22*p.xg/g+.16*p.xa/g+.06*p.key/g,0,1.5); return float(.42*rating+.16*mins+.22*prod+.14*winp+.06*sigmoid((elo-opp)/180))

def load_map():
    p=OUT/"fotmob_match_map.csv"
    if not p.exists():return {}
    try:
        d=pd.read_csv(p); need={"Date","HomeTeam","AwayTeam","match_id"}
        if not need.issubset(d.columns):return {}
        return {(str(r.Date),team_name(r.HomeTeam),team_name(r.AwayTeam)):str(r.match_id) for r in d.itertuples()}
    except Exception:return {}

# ---------------- CHECKPOINT ----------------
def save_ck(done,results,scores,moms,models):
    tmp=CHECKPOINT.with_suffix(".tmp")
    with tmp.open("wb") as f:pickle.dump({"version":"V7","done":list(done),"results":results,"scores":scores,"moms":moms,"models":models},f,pickle.HIGHEST_PROTOCOL)
    tmp.replace(CHECKPOINT)
def load_ck():
    if not CHECKPOINT.exists():return None
    try:
        with CHECKPOINT.open("rb") as f:return pickle.load(f)
    except Exception:return None

# ---------------- MAIN ----------------
def main():
    global START
    START=time.time(); print("=== Soccer Backtest Production ===")
    data=load_data(); print(f"Loaded {len(data):,} matches")
    groups=list(data.groupby(["League","SeasonStart","Season"],sort=True))
    leagues={k:League() for k in LEAGUES}; histories=defaultdict(list); bundles={}; blends=defaultdict(lambda:(.62,.23,.15)); players=defaultdict(Player)
    results=[]; scores=[]; moms=[]; model_rows=[]; done=set(); ck=load_ck()
    if ck:
        done=set(ck.get("done",[])); results=ck.get("results",[]); scores=ck.get("scores",[]); moms=ck.get("moms",[]); model_rows=ck.get("models",[]); print(f"Resuming {len(done)} completed groups")
    fmap=load_map()
    for (code,sy,season),g in groups:
        key=f"{code}|{sy}|{season}"
        if key in done:continue
        if left()<100:save_ck(done,results,scores,moms,model_rows); print("[SAFE STOP] checkpoint saved");break
        L=leagues[code]; L.reset(); hist=histories[code]; bundle=bundles.get(code)
        if len(hist)>=MIN_TRAIN:bundle=fit_ensemble(hist); bundles[code]=bundle
        print(f"[{code}] {season}: {len(g)}")
        for idx,(_,r) in enumerate(g.iterrows()):
            if left()<70:save_ck(done,results,scores,moms,model_rows); print("[SAFE STOP] checkpoint saved");return
            h=L.team(r.HomeTeam); a=L.team(r.AwayTeam); x=features(r,h,a,L); mp=odds(r)[:3]; sp,top,lh,la=score_model(h,a,L)
            if len(hist)>=MIN_TRAIN:
                if bundle is None or idx%RETRAIN_EVERY==0:
                    bundle=fit_ensemble(hist); bundles[code]=bundle
                    if bundle:
                        for n,v in bundle["validation_logloss"].items():model_rows.append({"League":code,"Season":season,"Model":n,"ValidationLogLoss":v})
                ml=predict_ml(bundle,x)
            else:ml=n3(.55*mp+.45*sp)
            lw,mw,sw=blends[code]
            # Reliability gate: when market is highly decisive, shrink model disagreement.
            disagreement=float(np.abs(ml-mp).mean())
            market_strength=float(mp.max())
            gate=np.clip(1.0-0.55*disagreement,0.55,1.0)
            final=n3(lw*gate*ml + mw*mp + sw*(gate*sp+(1-gate)*mp)); pred=int(np.argmax(final)); actual=int(r.Result)
            results.append({"League":code,"Season":season,"Date":r.Date,"HomeTeam":r.HomeTeam,"AwayTeam":r.AwayTeam,"Predicted":"HDA"[pred],"Actual":"HDA"[actual],"HomeProb":final[0],"DrawProb":final[1],"AwayProb":final[2],"Confidence":final.max(),"MarketHome":mp[0],"MarketDraw":mp[1],"MarketAway":mp[2],"MLHome":ml[0],"MLDraw":ml[1],"MLAway":ml[2],"PoissonHome":sp[0],"PoissonDraw":sp[1],"PoissonAway":sp[2],"LambdaHome":lh,"LambdaAway":la})
            ah,aa=int(r.FTHG),int(r.FTAG)
            for rank,(sh,sa,p) in enumerate(top,1):scores.append({"League":code,"Season":season,"Date":r.Date,"HomeTeam":r.HomeTeam,"AwayTeam":r.AwayTeam,"Rank":rank,"PredScore":f"{sh}-{sa}","Probability":float(p),"ActualScore":f"{ah}-{aa}","Hit":int(sh==ah and sa==aa),"AbsGoalError":abs(sh-ah)+abs(sa-aa)})
            # MOM: prediction uses only player history accumulated before this match.
            if ENABLE_MOM and fmap:
                mid=fmap.get((str(r.Date),r.HomeTeam,r.AwayTeam))
                if mid:
                    detail=fetch_fm(mid)
                    if detail:
                        actual_mom=find_mom(detail); ps=players_from_fm(detail); cand=[]
                        for p in ps:
                            tn=r.HomeTeam if p["side"]=="H" else r.AwayTeam; elo=h.elo if p["side"]=="H" else a.elo; opp=a.elo if p["side"]=="H" else h.elo; wp=final[0] if p["side"]=="H" else final[2]; st=players[(code,tn,p["name"])]
                            cand.append((p["name"],tn,mom_value(st,elo,opp,wp),p))
                        cand.sort(key=lambda z:z[2],reverse=True)
                        for rank,(pn,tn,ms,_) in enumerate(cand[:4],1):moms.append({"League":code,"Season":season,"Date":r.Date,"HomeTeam":r.HomeTeam,"AwayTeam":r.AwayTeam,"Rank":rank,"Player":pn,"Team":tn,"PreMatchMOMScore":ms,"ActualMOM":actual_mom,"Hit":int(actual_mom is not None and pn.lower().strip()==str(actual_mom).lower().strip())})
                        for p in ps:players[(code,r.HomeTeam if p["side"]=="H" else r.AwayTeam,p["name"])].update(p)
            hg,ag=ah,aa; hp=3 if hg>ag else 1 if hg==ag else 0; ap=0 if hg>ag else 1 if hg==ag else 3
            def st(c):return sf(r.get(c,np.nan))
            h.update(hg,ag,hp,"H",{"shots":st("HS"),"sot":st("HST"),"corners":st("HC"),"fouls":st("HF"),"yellow":st("HY"),"red":st("HR")}); a.update(ag,hg,ap,"A",{"shots":st("AS"),"sot":st("AST"),"corners":st("AC"),"fouls":st("AF"),"yellow":st("AY"),"red":st("AR")}); elo_update(h,a,int(r.Result),hg,ag); L.update(hg,ag,hp); hist.append({"x":x.copy(),"y":int(r.Result),"market":mp.copy(),"poisson":sp.copy()})
        done.add(key); bundles[code]=bundle
        if len(hist)>=300:blends[code]=optimize_blend(hist)
        save_ck(done,results,scores,moms,model_rows)
    rdf=pd.DataFrame(results); sdf=pd.DataFrame(scores); mdf=pd.DataFrame(moms); modf=pd.DataFrame(model_rows)
    if not rdf.empty:
        rdf["Correct"]=(rdf.Predicted==rdf.Actual).astype(int); rdf.to_csv(OUT/"backtest_results_v7.csv",index=False,encoding="utf-8-sig"); y=rdf.Actual.map({"H":0,"D":1,"A":2}).values; p=rdf[["HomeProb","DrawProb","AwayProb"]].values; br=np.mean([brier_score_loss((y==c).astype(int),p[:,c]) for c in range(3)]); pd.DataFrame([{"Matches":len(rdf),"Accuracy":accuracy_score(y,np.argmax(p,1)),"LogLoss":log_loss(y,p,labels=[0,1,2]),"Brier":br,"MeanConfidence":rdf.Confidence.mean()}]).to_csv(OUT/"overall_summary_v7.csv",index=False); rows=[]
        for lg,g in rdf.groupby("League"):
            gy=g.Actual.map({"H":0,"D":1,"A":2}); gp=g[["HomeProb","DrawProb","AwayProb"]]; rows.append({"League":lg,"Matches":len(g),"Accuracy":(g.Correct.mean()),"LogLoss":log_loss(gy,gp,labels=[0,1,2]),"MeanConfidence":g.Confidence.mean()})
        pd.DataFrame(rows).to_csv(OUT/"backtest_summary_v7.csv",index=False); pd.DataFrame([{"Threshold":t,"Matches":int((rdf.Confidence>=t).sum()),"Accuracy":rdf.loc[rdf.Confidence>=t,"Correct"].mean()} for t in [.5,.55,.6,.65,.7,.75,.8]]).to_csv(OUT/"confidence_summary_v7.csv",index=False)
    if not sdf.empty:
        sdf.to_csv(OUT/"backtest_scores_v7.csv",index=False); top1=sdf[sdf.Rank==1].Hit.mean(); top3=sdf.groupby(["League","Season","Date","HomeTeam","AwayTeam"]).Hit.max().mean(); pd.DataFrame([{"ScoreTop1HitRate":top1,"ScoreTop3HitRate":top3,"MeanAbsoluteGoalError":sdf.AbsGoalError.mean()}]).to_csv(OUT/"score_summary_v7.csv",index=False)
    if not mdf.empty:
        mdf.to_csv(OUT/"backtest_mom_v7.csv",index=False); t1=mdf[mdf.Rank==1].Hit.mean(); t4=mdf.groupby(["League","Season","Date","HomeTeam","AwayTeam"]).Hit.max().mean(); pd.DataFrame([{"MOMTop1HitRate":t1,"MOMTop4HitRate":t4,"EvaluatedRows":len(mdf)}]).to_csv(OUT/"mom_summary_v7.csv",index=False)
    if not modf.empty:modf.to_csv(OUT/"model_comparison_v7.csv",index=False)
    if len(done)==len(groups):
        try:CHECKPOINT.unlink()
        except Exception:pass
    print("=========================================="); print("BACKTEST FINISHED"); print(f"Runtime: {(time.time()-START)/60:.2f} minutes")
    if not rdf.empty:print(f"1X2 Accuracy: {rdf.Correct.mean()*100:.2f}%")
    if not sdf.empty:print(f"Score Top-1: {sdf[sdf.Rank==1].Hit.mean()*100:.2f}%")
    if not mdf.empty:print(f"MOM Top-1: {mdf[mdf.Rank==1].Hit.mean()*100:.2f}%")
    else:print("MOM: no mapped/cached FotMob data")
    print("==========================================")
if __name__=="__main__":main()
