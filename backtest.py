#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Soccer Backtest - production version.
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
OUT=Path("."); CACHE=OUT/"cache"; FM_CACHE=CACHE/"fotmob"; CHECKPOINT=OUT/"backtest_checkpoint.pkl"
MAX_RUNTIME=27*60; MIN_TRAIN=220; MAX_TRAIN=1800; RETRAIN_EVERY=30; VALID_FRAC=.20; RANDOM_STATE=42
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
    ps=[]
    for _,cs in ODDS:
        v=[sf(row.get(c,np.nan)) for c in cs]
        if all(np.isfinite(z) and z>1 for z in v): ps.append(n3(1/np.asarray(v)))
    if not ps: return np.array([1/3,1/3,1/3,0,1,0,0.],float)
    a=np.vstack(ps); p=n3(a.mean(0)); return np.array([p[0],p[1],p[2],p.max(),-np.sum(p*np.log(np.maximum(p,1e-12)))/np.log(3),a.std(0).mean(),1.])

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
    lh0,la0,_=L.av(); ha=h.hegf or lh0; hd=h.hega or la0; aa=a.aegf or la0; ad=a.aega or lh0
    lh=lh0*np.clip((ha/lh0),.55,1.8)*np.clip((ad/lh0),.55,1.8); la=la0*np.clip((aa/la0),.55,1.8)*np.clip((hd/la0),.55,1.8)
    ed=(h.elo+55-a.elo)/400; lh*=np.clip(1+.10*ed,.88,1.12); la*=np.clip(1-.07*ed,.90,1.10); lh=float(np.clip(lh,.2,3.8)); la=float(np.clip(la,.15,3.4))
    m=np.zeros((8,8))
    for i in range(8):
        for j in range(8): m[i,j]=pois(i,lh)*pois(j,la)*tau(i,j,lh,la)
    m/=max(m.sum(),1e-12); p=[sum(m[i,j] for i in range(8) for j in range(8) if i>j),sum(m[i,j] for i in range(8) for j in range(8) if i==j),sum(m[i,j] for i in range(8) for j in range(8) if i<j)]
    top=sorted([(i,j,float(m[i,j])) for i in range(8) for j in range(8)],key=lambda x:x[2],reverse=True)[:3]
    return n3(p),top,lh,la

# ---------------- ML ----------------
def models():
    return {"Logistic":Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(C=.25,max_iter=700,random_state=RANDOM_STATE))]),"ExtraTrees":ExtraTreesClassifier(n_estimators=280,min_samples_leaf=7,max_features=.75,class_weight="balanced_subsample",random_state=RANDOM_STATE,n_jobs=-1),"HistGB":HistGradientBoostingClassifier(max_iter=180,learning_rate=.045,max_leaf_nodes=15,l2_regularization=1.5,random_state=RANDOM_STATE)}
def align(model,p):
    q=np.zeros(3)
    for i,c in enumerate(model.classes_):
        if int(c) in (0,1,2): q[int(c)]=p[i]
    return n3(q)
def fit_ensemble(hist):
    if len(hist)<MIN_TRAIN:return None
    d=hist[-MAX_TRAIN:]; X=np.vstack([z["x"] for z in d]); y=np.array([z["y"] for z in d],int); cut=max(1,min(len(y)-1,int(len(y)*(1-VALID_FRAC))))
    losses={}; good={}
    for name,m in models().items():
        try: m.fit(X[:cut],y[:cut]); losses[name]=log_loss(y[cut:],m.predict_proba(X[cut:]),labels=[0,1,2]); good[name]=True
        except Exception as e: print(f"[WARN] validation {name}: {e}")
    if not good:return None
    names=list(good); raw=np.exp(-(np.array([losses[n] for n in names])-min(losses.values()))/.12); raw=np.clip(raw,.10,.80); raw/=raw.sum(); fm={}
    for n in names:
        try:
            m=models()[n]; m.fit(X,y); fm[n]=m
        except Exception: pass
    return {"models":fm,"weights":dict(zip(names,raw)),"validation_logloss":losses}
def predict_ml(bundle,x):
    if not bundle:return np.ones(3)/3
    ps=[]; ws=[]
    for n,m in bundle["models"].items():
        try: ps.append(align(m,m.predict_proba(x.reshape(1,-1))[0])); ws.append(bundle["weights"].get(n,0))
        except Exception: pass
    if not ps:return np.ones(3)/3
    w=np.asarray(ws); w/=max(w.sum(),1e-12); return n3(np.average(np.vstack(ps),axis=0,weights=w))

def optimize_blend(hist):
    if len(hist)<300:return (.62,.23,.15)
    d=hist[-MAX_TRAIN:]; X=np.vstack([z["x"] for z in d]); y=np.array([z["y"] for z in d]); cut=int(len(y)*.8)
    if len(y)-cut<30:return (.62,.23,.15)
    m=Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(C=.25,max_iter=600,random_state=RANDOM_STATE))])
    try:
        m.fit(X[:cut],y[:cut]); ml=np.vstack([align(m,p) for p in m.predict_proba(X[cut:])]); market=np.vstack([n3(x[-7:-4]) for x in X[cut:]]); best=(.62,.23,.15,1e9)
        for mw in np.arange(.05,.46,.05):
            for sw in np.arange(0,.31,.05):
                lw=1-mw-sw
                if lw<.4:continue
                pred=(lw*ml+mw*market+sw*n3((.55*ml+.45*market).T).T) # row normalization below
                pred=pred/np.maximum(pred.sum(1,keepdims=True),1e-12); ll=log_loss(y[cut:],pred,labels=[0,1,2])
                if ll<best[3]:best=(float(lw),float(mw),float(sw),float(ll))
        return best[:3]
    except Exception:return (.62,.23,.15)

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
    with tmp.open("wb") as f:pickle.dump({"done":list(done),"results":results,"scores":scores,"moms":moms,"models":models},f,pickle.HIGHEST_PROTOCOL)
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
            lw,mw,sw=blends[code]; final=n3(lw*ml+mw*mp+sw*sp); pred=int(np.argmax(final)); actual=int(r.Result)
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
            h.update(hg,ag,hp,"H",{"shots":st("HS"),"sot":st("HST"),"corners":st("HC"),"fouls":st("HF"),"yellow":st("HY"),"red":st("HR")}); a.update(ag,hg,ap,"A",{"shots":st("AS"),"sot":st("AST"),"corners":st("AC"),"fouls":st("AF"),"yellow":st("AY"),"red":st("AR")}); elo_update(h,a,int(r.Result),hg,ag); L.update(hg,ag,hp); hist.append({"x":x.copy(),"y":int(r.Result)})
        done.add(key); bundles[code]=bundle
        if len(hist)>=300:blends[code]=optimize_blend(hist)
        save_ck(done,results,scores,moms,model_rows)
    rdf=pd.DataFrame(results); sdf=pd.DataFrame(scores); mdf=pd.DataFrame(moms); modf=pd.DataFrame(model_rows)
    if not rdf.empty:
        rdf["Correct"]=(rdf.Predicted==rdf.Actual).astype(int); rdf.to_csv(OUT/"backtest_results.csv",index=False,encoding="utf-8-sig"); y=rdf.Actual.map({"H":0,"D":1,"A":2}).values; p=rdf[["HomeProb","DrawProb","AwayProb"]].values; br=np.mean([brier_score_loss((y==c).astype(int),p[:,c]) for c in range(3)]); pd.DataFrame([{"Matches":len(rdf),"Accuracy":accuracy_score(y,np.argmax(p,1)),"LogLoss":log_loss(y,p,labels=[0,1,2]),"Brier":br,"MeanConfidence":rdf.Confidence.mean()}]).to_csv(OUT/"overall_summary.csv",index=False); rows=[]
        for lg,g in rdf.groupby("League"):
            gy=g.Actual.map({"H":0,"D":1,"A":2}); gp=g[["HomeProb","DrawProb","AwayProb"]]; rows.append({"League":lg,"Matches":len(g),"Accuracy":(g.Correct.mean()),"LogLoss":log_loss(gy,gp,labels=[0,1,2]),"MeanConfidence":g.Confidence.mean()})
        pd.DataFrame(rows).to_csv(OUT/"backtest_summary.csv",index=False); pd.DataFrame([{"Threshold":t,"Matches":int((rdf.Confidence>=t).sum()),"Accuracy":rdf.loc[rdf.Confidence>=t,"Correct"].mean()} for t in [.5,.55,.6,.65,.7,.75,.8]]).to_csv(OUT/"confidence_summary.csv",index=False)
    if not sdf.empty:
        sdf.to_csv(OUT/"backtest_scores.csv",index=False); top1=sdf[sdf.Rank==1].Hit.mean(); top3=sdf.groupby(["League","Season","Date","HomeTeam","AwayTeam"]).Hit.max().mean(); pd.DataFrame([{"ScoreTop1HitRate":top1,"ScoreTop3HitRate":top3,"MeanAbsoluteGoalError":sdf.AbsGoalError.mean()}]).to_csv(OUT/"score_summary.csv",index=False)
    if not mdf.empty:
        mdf.to_csv(OUT/"backtest_mom.csv",index=False); t1=mdf[mdf.Rank==1].Hit.mean(); t4=mdf.groupby(["League","Season","Date","HomeTeam","AwayTeam"]).Hit.max().mean(); pd.DataFrame([{"MOMTop1HitRate":t1,"MOMTop4HitRate":t4,"EvaluatedRows":len(mdf)}]).to_csv(OUT/"mom_summary.csv",index=False)
    if not modf.empty:modf.to_csv(OUT/"model_comparison.csv",index=False)
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
