#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soccer Backtest FINAL / Multi-Source / Leak-Safe

用途ごとに最適なソースを分離して検証する。

SOURCE MAP
----------
Football-Data.co.uk : FT結果 / 日程 / closing odds / 基本match stats
Understat             : xG / xGA
SofaScore             : 過去選手rating / player stats / 実績MOM
自前計算              : Elo / form / home-away / rest / H2H / season transition

重要なリーク対策
----------------
* 当該試合の結果・stats・xGは、予測後にstateへ反映。
* 当該試合のSofaScore lineupはMOM候補選定に使用しない。
  MOM候補は「その試合より前に蓄積した選手履歴」だけから選ぶ。
* 当該試合のSofaScore ratingは、予測後にplayer stateへ反映。
* closing oddsは試合前情報なので使用可能。
* chronological validationのみ。ランダムsplitは使用しない。
* blend weightも過去validation区間だけで決定。

30分GitHub Actions対策
----------------------
* 安全停止
* checkpoint/resume
* 外部API失敗時のgraceful fallback
* キャッシュ優先

OUTPUT
------
backtest_results.csv       1X2全試合
backtest_scores.csv        exact score Top-3
backtest_mom.csv           MOM候補 Top-4
overall_summary.csv        総合1X2
league_summary.csv         リーグ別
season_summary.csv         シーズン別
confidence_summary.csv     信頼度別
score_summary.csv          スコア精度
mom_summary.csv            MOM Top1/Top4
model_comparison.csv       validation model比較
data_coverage.csv         ソース取得率
feature_importance.csv     ExtraTrees重要度
"""
from __future__ import annotations

import json
import math
import os
import pickle
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "SP1": "La Liga",
    "F1": "Ligue 1",
}
SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
FD_URL = "https://www.football-data.co.uk/mmz4281/{folder}/{league}.csv"

ROOT = Path(".")
CACHE = ROOT / "cache"
FD_CACHE = CACHE / "football_data"
UNDERSTAT_CACHE = CACHE / "understat"
SOFA_CACHE = CACHE / "sofascore"
CHECKPOINT = ROOT / "backtest_checkpoint.pkl"

MAX_RUNTIME = int(os.getenv("MAX_RUNTIME_SECONDS", "1680"))
MIN_TRAIN = 260
MAX_TRAIN = 2200
RETRAIN_EVERY = 35
VALID_FRAC = 0.20
SAVE_EVERY = 20
RANDOM_STATE = 42
TIMEOUT = 12

ENABLE_UNDERSTAT = os.getenv("ENABLE_UNDERSTAT", "1") == "1"
ENABLE_SOFASCORE = os.getenv("ENABLE_SOFASCORE", "1") == "1"

START = time.time()
HTTP = requests.Session()
HTTP.headers.update({
    "User-Agent": "Mozilla/5.0 SoccerBacktestFinal/1.0",
    "Accept": "application/json,text/plain,*/*",
})


def left():
    return MAX_RUNTIME - (time.time() - START)


def sf(x, default=np.nan):
    try:
        if x is None or pd.isna(x):
            return default
        s = str(x).strip().replace(",", ".")
        if s in ("", "-", "nan", "NaN", "None"):
            return default
        return float(s)
    except Exception:
        return default


def norm3(x):
    a = np.asarray(x, dtype=float)
    a = np.nan_to_num(a, nan=1/3, posinf=1/3, neginf=1/3)
    a = np.maximum(a, 1e-12)
    return a / a.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def season_folder(y):
    return f"{str(y)[-2:]}{str(y + 1)[-2:]}"


def team_name(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    aliases = {
        "Man United": "Manchester United", "Man Utd": "Manchester United",
        "Man City": "Manchester City", "Spurs": "Tottenham Hotspur",
        "Tottenham": "Tottenham Hotspur", "Nott'm Forest": "Nottingham Forest",
        "Nottm Forest": "Nottingham Forest", "Newcastle": "Newcastle United",
        "West Ham": "West Ham United", "Wolves": "Wolverhampton Wanderers",
        "Leicester": "Leicester City", "Leeds": "Leeds United",
        "Brighton": "Brighton & Hove Albion", "Sheffield Utd": "Sheffield United",
        "Norwich": "Norwich City", "QPR": "Queens Park Rangers",
        "West Brom": "West Bromwich Albion", "Huddersfield": "Huddersfield Town",
        "Ath Bilbao": "Athletic Bilbao", "Ath Madrid": "Atletico Madrid",
        "Betis": "Real Betis", "Sociedad": "Real Sociedad",
        "M'gladbach": "Borussia Monchengladbach", "Leverkusen": "Bayer Leverkusen",
        "Ein Frankfurt": "Eintracht Frankfurt", "FC Koln": "FC Cologne",
        "Hertha": "Hertha Berlin", "Inter": "Inter Milan", "Milan": "AC Milan",
        "Roma": "AS Roma", "PSG": "Paris Saint-Germain", "Paris SG": "Paris Saint-Germain",
    }
    return aliases.get(s, s)


def get_json(url, cache_path=None):
    if cache_path and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if left() < 8:
        return None
    try:
        r = HTTP.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data
    except Exception:
        return None


# =========================
# FOOTBALL-DATA
# =========================
def load_fd(code, year):
    path = FD_CACHE / f"{code}_{year}.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    if left() < 10:
        return pd.DataFrame()
    url = FD_URL.format(folder=season_folder(year), league=code)
    try:
        r = HTTP.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Football-Data {code} {year}: {e}")
        return pd.DataFrame()


def load_matches():
    frames = []
    for code in LEAGUES:
        for year in SEASONS:
            if left() < 20:
                break
            d = load_fd(code, year)
            if d.empty:
                continue
            d["League"] = code
            d["SeasonStart"] = year
            d["Season"] = f"{year}/{str(year + 1)[-2:]}"
            frames.append(d)
    if not frames:
        raise RuntimeError("Football-Data.co.ukから試合データを取得できませんでした。")
    d = pd.concat(frames, ignore_index=True)
    d["DateParsed"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
    d["HomeTeam"] = d["HomeTeam"].map(team_name)
    d["AwayTeam"] = d["AwayTeam"].map(team_name)
    d["FTHG"] = pd.to_numeric(d["FTHG"], errors="coerce")
    d["FTAG"] = pd.to_numeric(d["FTAG"], errors="coerce")
    d = d.dropna(subset=["DateParsed", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    d["FTHG"] = d["FTHG"].astype(int)
    d["FTAG"] = d["FTAG"].astype(int)
    d["Result"] = np.where(d.FTHG > d.FTAG, 0, np.where(d.FTHG == d.FTAG, 1, 2))
    return d.sort_values(["League", "SeasonStart", "DateParsed"], kind="stable").reset_index(drop=True)


ODDS_GROUPS = [
    ("B365C", ("B365CH", "B365CD", "B365CA")),
    ("BWC", ("BWCH", "BWCD", "BWCA")),
    ("IWC", ("IWCH", "IWCD", "IWCA")),
    ("PSC", ("PSCH", "PSCD", "PSCA")),
    ("WHC", ("WHCH", "WHCD", "WHCA")),
    ("VCC", ("VCCH", "VCCD", "VCCA")),
    ("MaxC", ("MaxCH", "MaxCD", "MaxCA")),
    ("AvgC", ("AvgCH", "AvgCD", "AvgCA")),
]


def closing_market(row):
    rows = []
    for _, cols in ODDS_GROUPS:
        vals = [sf(row.get(c, np.nan)) for c in cols]
        if all(np.isfinite(v) and v > 1 for v in vals):
            rows.append(norm3(1 / np.asarray(vals)))
    if not rows:
        return np.array([1/3, 1/3, 1/3, 1/3, 1.0, 0.0], dtype=float)
    a = np.vstack(rows)
    p = norm3(np.median(a, axis=0))
    ent = -np.sum(p * np.log(np.maximum(p, 1e-12))) / np.log(3)
    disp = float(np.mean(np.std(a, axis=0)))
    return np.array([p[0], p[1], p[2], p.max(), ent, disp], dtype=float)


# =========================
# UNDERSTAT: xG source
# =========================
UNDERSTAT_LEAGUE = {
    "E0": "EPL", "D1": "Bundesliga", "I1": "Serie_A",
    "SP1": "La_liga", "F1": "Ligue_1",
}


def load_understat_season(code, year):
    if not ENABLE_UNDERSTAT or code not in UNDERSTAT_LEAGUE:
        return []
    cache = UNDERSTAT_CACHE / f"{code}_{year}.json"
    url = f"https://understat.com/getLeagueData/{UNDERSTAT_LEAGUE[code]}/{year}"
    data = get_json(url, cache)
    if isinstance(data, dict):
        data = data.get("datesData") or data.get("matches") or data.get("games") or []
    return data if isinstance(data, list) else []


def build_understat_index(matches):
    idx = {}
    cov = []
    for (code, year), _ in matches.groupby(["League", "SeasonStart"]):
        if left() < 25:
            break
        rows = load_understat_season(code, int(year))
        matched = 0
        for z in rows:
            try:
                h = z.get("h", {}) if isinstance(z.get("h"), dict) else {}
                a = z.get("a", {}) if isinstance(z.get("a"), dict) else {}
                home = team_name(h.get("title") or z.get("home_team"))
                away = team_name(a.get("title") or z.get("away_team"))
                dt = pd.to_datetime(z.get("datetime") or z.get("date"), errors="coerce")
                hxg = sf(h.get("xG") or h.get("xg"))
                axg = sf(a.get("xG") or a.get("xg"))
                if pd.isna(dt) or not (np.isfinite(hxg) and np.isfinite(axg)):
                    continue
                idx[(code, dt.date().isoformat(), home, away)] = {"hxg": hxg, "axg": axg}
                matched += 1
            except Exception:
                continue
        cov.append({"League": code, "SeasonStart": int(year), "Source": "Understat", "Available": len(rows), "Matched": matched})
    return idx, cov


# =========================
# TEAM STATE
# =========================
class Team:
    def __init__(self, elo=1500.0):
        self.elo = elo
        self.matches = 0
        self.gf = self.ga = self.points = 0.0
        self.hm = self.am = 0
        self.hgf = self.hga = self.hp = 0.0
        self.agf = self.aga = self.ap = 0.0
        self.form = deque(maxlen=8)
        self.gfr = deque(maxlen=8)
        self.gar = deque(maxlen=8)
        self.xgf = deque(maxlen=8)
        self.xga = deque(maxlen=8)
        self.shots = deque(maxlen=8)
        self.sot = deque(maxlen=8)
        self.corners = deque(maxlen=8)
        self.egf = self.ega = self.ep = None
        self.hegf = self.hega = self.aegf = self.aega = None

    @staticmethod
    def ema(old, value, alpha=.22):
        return value if old is None else alpha * value + (1-alpha) * old

    def update(self, gf, ga, pts, venue, stats, xg_for=np.nan, xg_against=np.nan):
        self.matches += 1
        self.gf += gf; self.ga += ga; self.points += pts
        self.form.append(pts); self.gfr.append(gf); self.gar.append(ga)
        self.egf = self.ema(self.egf, gf); self.ega = self.ema(self.ega, ga); self.ep = self.ema(self.ep, pts)
        if venue == "H":
            self.hm += 1; self.hgf += gf; self.hga += ga; self.hp += pts
            self.hegf = self.ema(self.hegf, gf); self.hega = self.ema(self.hega, ga)
        else:
            self.am += 1; self.agf += gf; self.aga += ga; self.ap += pts
            self.aegf = self.ema(self.aegf, gf); self.aega = self.ema(self.aega, ga)
        if np.isfinite(xg_for): self.xgf.append(xg_for)
        if np.isfinite(xg_against): self.xga.append(xg_against)
        for k, v in stats.items():
            if np.isfinite(v):
                getattr(self, k).append(v)

    def ppg(self): return self.points / max(1, self.matches)
    def gfpg(self): return self.gf / max(1, self.matches)
    def gapg(self): return self.ga / max(1, self.matches)


class League:
    def __init__(self):
        self.teams = {}
        self.hg = deque(maxlen=500); self.ag = deque(maxlen=500)

    def team(self, name):
        if name not in self.teams: self.teams[name] = Team()
        return self.teams[name]

    def new_season(self):
        # strengthは部分的に持ち越し、form/statsはリセット。
        for t in self.teams.values():
            t.elo = 1500 + .72 * (t.elo - 1500)
            t.matches = t.hm = t.am = 0
            t.gf = t.ga = t.points = 0
            t.hgf = t.hga = t.hp = t.agf = t.aga = t.ap = 0
            t.form.clear(); t.gfr.clear(); t.gar.clear(); t.xgf.clear(); t.xga.clear()
            t.shots.clear(); t.sot.clear(); t.corners.clear()
            t.egf = t.ega = t.ep = t.hegf = t.hega = t.aegf = t.aega = None
        self.hg.clear(); self.ag.clear()

    def av(self):
        return (max(.75, np.mean(self.hg) if self.hg else 1.45), max(.60, np.mean(self.ag) if self.ag else 1.15))

    def update(self, hg, ag):
        self.hg.append(hg); self.ag.append(ag)


def avg_or(x, default):
    return float(np.mean(x)) if x else default


def team_features(t, venue, lh, la):
    if venue == "H":
        m, gf, ga, p = t.hm, t.hgf, t.hga, t.hp/max(1,t.hm)
        egf, ega = t.hegf if t.hegf is not None else lh, t.hega if t.hega is not None else la
    else:
        m, gf, ga, p = t.am, t.agf, t.aga, t.ap/max(1,t.am)
        egf, ega = t.aegf if t.aegf is not None else la, t.aega if t.aega is not None else lh
    return [
        t.elo, t.ppg(), t.gfpg(), t.gapg(), avg_or(t.form,1.25), avg_or(t.gfr,1.25), avg_or(t.gar,1.15),
        p, gf/max(1,m), ga/max(1,m), egf, ega, avg_or(t.xgf,egf), avg_or(t.xga,ega),
        avg_or(t.shots,12), avg_or(t.sot,4), avg_or(t.corners,5)
    ]


FEATURE_NAMES = []

def make_feature_names():
    base = ["elo","ppg","gfpg","gapg","form","gf_recent","ga_recent","venue_ppg","venue_gfpg","venue_gapg","ema_gf","ema_ga","xg_for","xg_against","shots","sot","corners"]
    global FEATURE_NAMES
    FEATURE_NAMES = [f"H_{x}" for x in base] + [f"A_{x}" for x in base] + [
        "elo_diff","ppg_diff","gfpg_diff","def_diff","attack_diff","def_attack_diff",
        "xg_for_diff","xg_against_diff","rest_h","rest_a","rest_diff","elo_sigmoid","ppg_sigmoid",
        "market_h","market_d","market_a","market_favorite","market_entropy","market_dispersion",
        "h2h_home_rate","h2h_draw_rate","h2h_away_rate","h2h_count"
    ]


def make_features(row, h, a, L, last_dates, h2h):
    lh, la = L.av()
    market = closing_market(row)
    rest_h = 7.0 if row.HomeTeam not in last_dates else np.clip((row.DateParsed-last_dates[row.HomeTeam]).total_seconds()/86400, .5, 21)
    rest_a = 7.0 if row.AwayTeam not in last_dates else np.clip((row.DateParsed-last_dates[row.AwayTeam]).total_seconds()/86400, .5, 21)
    pair = tuple(sorted([row.HomeTeam,row.AwayTeam]))
    hh = h2h.get((row.League,pair), [])[-8:]
    hc = max(1,len(hh))
    # stored result is from perspective of the team that was home in each old match; for
    # H2H feature we keep neutral rates based on raw outcome history only.
    h2h_home = sum(v==0 for v in hh)/hc if hh else 1/3
    h2h_draw = sum(v==1 for v in hh)/hc if hh else 1/3
    h2h_away = sum(v==2 for v in hh)/hc if hh else 1/3
    x = (
        team_features(h,"H",lh,la) + team_features(a,"A",lh,la) + [
            h.elo-a.elo,
            h.ppg()-a.ppg(),
            h.gfpg()-a.gfpg(),
            a.gapg()-h.gapg(),
            (h.hegf or lh)-(a.aegf or la),
            (a.aega or lh)-(h.hega or la),
            avg_or(h.xgf,h.hegf or lh)-avg_or(a.xgf,a.aegf or la),
            avg_or(h.xga,h.hega or la)-avg_or(a.xga,a.aega or lh),
            rest_h,rest_a,rest_h-rest_a,
            sigmoid((h.elo-a.elo)/180),sigmoid(h.ppg()-a.ppg()),
            market[0],market[1],market[2],market[3],market[4],market[5],
            h2h_home,h2h_draw,h2h_away,len(hh)
        ]
    )
    return np.nan_to_num(np.asarray(x,dtype=float),nan=0.0,posinf=0.0,neginf=0.0)


# =========================
# SCORE MODEL
# =========================
def pois(k, lam):
    return 0.0 if lam <= 0 else math.exp(-lam)*lam**k/math.factorial(k)


def dc_tau(i,j,lh,la,rho=-.075):
    if i==0 and j==0: return 1-lh*la*rho
    if i==0 and j==1: return 1+lh*rho
    if i==1 and j==0: return 1+la*rho
    if i==1 and j==1: return 1-rho
    return 1.0


def score_model(h,a,L):
    league_h,league_a=L.av()
    ha=h.hegf if h.hegf is not None else league_h
    hd=h.hega if h.hega is not None else league_a
    aa=a.aegf if a.aegf is not None else league_a
    ad=a.aega if a.aega is not None else league_h
    lh=league_h*np.clip(ha/league_h,.55,1.85)*np.clip(ad/league_h,.55,1.85)
    la=league_a*np.clip(aa/league_a,.55,1.85)*np.clip(hd/league_a,.55,1.85)
    ed=(h.elo+55-a.elo)/400
    lh*=np.clip(1+.10*ed,.88,1.14); la*=np.clip(1-.07*ed,.90,1.10)
    lh=float(np.clip(lh,.2,3.8)); la=float(np.clip(la,.15,3.4))
    m=np.zeros((9,9))
    for i in range(9):
        for j in range(9): m[i,j]=pois(i,lh)*pois(j,la)*dc_tau(i,j,lh,la)
    m/=max(m.sum(),1e-12)
    p=norm3([m[np.triu_indices(9,1)].sum(),np.trace(m),m[np.tril_indices(9,-1)].sum()])
    top=sorted([(i,j,float(m[i,j])) for i in range(9) for j in range(9)],key=lambda z:z[2],reverse=True)[:3]
    return p,top,lh,la


# =========================
# ML
# =========================
def build_models():
    return {
        "Logistic": Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(C=.18,max_iter=900,random_state=RANDOM_STATE))]),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=320,min_samples_leaf=8,max_features=.72,class_weight="balanced_subsample",random_state=RANDOM_STATE,n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=260,min_samples_leaf=8,max_features=.70,class_weight="balanced_subsample",random_state=RANDOM_STATE,n_jobs=-1),
        "HistGB": HistGradientBoostingClassifier(max_iter=210,learning_rate=.035,max_leaf_nodes=15,l2_regularization=2.0,random_state=RANDOM_STATE),
    }


def fit_ensemble(hist):
    if len(hist)<MIN_TRAIN: return None
    d=hist[-MAX_TRAIN:]
    X=np.vstack([z["x"] for z in d]); y=np.asarray([z["y"] for z in d],dtype=int)
    cut=max(1,min(len(y)-1,int(len(y)*(1-VALID_FRAC))))
    val={}; fitted={}
    for name,model in build_models().items():
        try:
            model.fit(X[:cut],y[:cut])
            val[name]=float(log_loss(y[cut:],model.predict_proba(X[cut:]),labels=[0,1,2]))
        except Exception as e:
            print(f"[WARN] validation {name}: {e}")
    if not val: return None
    names=list(val)
    w=np.exp(-(np.asarray([val[n] for n in names])-min(val.values()))/.10)
    w=np.clip(w,.08,.72); w/=w.sum()
    for name in names:
        try:
            model=build_models()[name]; model.fit(X,y); fitted[name]=model
        except Exception: pass
    return {"models":fitted,"weights":dict(zip(names,w)),"validation_logloss":val}


def pred_ml(bundle,x):
    if not bundle: return np.ones(3)/3
    ps=[]; ws=[]
    for name,model in bundle["models"].items():
        try:
            raw=model.predict_proba(x.reshape(1,-1))[0]
            p=np.zeros(3)
            for i,c in enumerate(model.classes_):
                if int(c) in (0,1,2): p[int(c)]=raw[i]
            ps.append(norm3(p)); ws.append(bundle["weights"].get(name,0))
        except Exception: pass
    if not ps: return np.ones(3)/3
    w=np.asarray(ws); w/=max(w.sum(),1e-12)
    return norm3(np.average(np.vstack(ps),axis=0,weights=w))


def optimize_blend(hist):
    if len(hist)<360: return (.56,.28,.16)
    d=hist[-MAX_TRAIN:]; cut=int(len(d)*.80); val=d[cut:]
    y=np.asarray([z["y"] for z in val],dtype=int)
    ml=np.vstack([z["ml"] for z in val]); mk=np.vstack([z["market"] for z in val]); sc=np.vstack([z["score"] for z in val])
    best=(.56,.28,.16,999.)
    for mw in np.arange(.10,.46,.05):
        for sw in np.arange(.05,.36,.05):
            lw=1-mw-sw
            if lw<.35: continue
            p=np.asarray([norm3(lw*ml[i]+mw*mk[i]+sw*sc[i]) for i in range(len(y))])
            ll=log_loss(y,p,labels=[0,1,2])
            if ll<best[3]: best=(float(lw),float(mw),float(sw),float(ll))
    return best[:3]


# =========================
# SOFASCORE PLAYER / MOM
# =========================
class Player:
    def __init__(self):
        self.matches=0; self.minutes=0.; self.rating=6.5; self.goals=0.; self.assists=0.; self.xg=0.; self.key=0.; self.position=""
    def update(self,p):
        self.matches+=1
        mins=sf(p.get("minutes")); self.minutes += mins if np.isfinite(mins) else 0
        rating=sf(p.get("rating"));
        if np.isfinite(rating): self.rating=.25*rating+.75*self.rating
        for k in ("goals","assists","xg","key"):
            v=sf(p.get(k),0)
            if np.isfinite(v): setattr(self,k,getattr(self,k)+v)
        if p.get("position"): self.position=str(p["position"])


def sofascore_event_ids(date,home,away):
    if not ENABLE_SOFASCORE or left()<12: return []
    try:
        dt=pd.to_datetime(date)
        for dd in (dt,dt-pd.Timedelta(days=1),dt+pd.Timedelta(days=1)):
            cache=SOFA_CACHE/"scheduled"/f"{dd.strftime('%Y-%m-%d')}.json"
            url=f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{dd.strftime('%Y-%m-%d')}"
            data=get_json(url,cache)
            if not data: continue
            for ev in data.get("events",[]):
                if team_name(ev.get("homeTeam",{}).get("name"))==home and team_name(ev.get("awayTeam",{}).get("name"))==away:
                    return [str(ev.get("id"))]
    except Exception: pass
    return []


def sofa_event(event_id):
    lc=SOFA_CACHE/"events"/f"{event_id}_lineups.json"
    dc=SOFA_CACHE/"events"/f"{event_id}_details.json"
    lineups=get_json(f"https://www.sofascore.com/api/v1/event/{event_id}/lineups",lc)
    details=get_json(f"https://www.sofascore.com/api/v1/event/{event_id}",dc)
    return lineups,details


def extract_players(lineups):
    out=[]
    if not isinstance(lineups,dict): return out
    for side_key,side in (("home","H"),("away","A")):
        block=lineups.get(side_key,{})
        for item in (block.get("players",[]) if isinstance(block,dict) else []):
            if not isinstance(item,dict): continue
            p=item.get("player",{})
            name=p.get("name")
            if not name: continue
            st=item.get("statistics",{}) or {}
            out.append({
                "side":side,"name":str(name),"player_id":p.get("id"),"position":p.get("position"),
                "rating":sf(st.get("rating")),"minutes":sf(st.get("minutesPlayed") or st.get("minutes")),
                "goals":sf(st.get("goals"),0),"assists":sf(st.get("assists"),0),
                "xg":sf(st.get("expectedGoals"),0),"key":sf(st.get("keyPasses"),0)
            })
    return out


def actual_mom(details,lineups):
    if isinstance(details,dict):
        for key in ("bestPlayer","manOfTheMatch","playerOfTheMatch","motm"):
            v=details.get(key)
            if isinstance(v,dict):
                n=v.get("name") or v.get("player",{}).get("name")
                if n: return str(n),"explicit"
            if isinstance(v,str) and v: return v,"explicit"
        found=[]
        def walk(o):
            if isinstance(o,dict):
                for k,v in o.items():
                    kl=str(k).lower()
                    if "manofthematch" in kl or kl in ("motm","playerofthematch","bestplayer"):
                        if isinstance(v,dict):
                            n=v.get("name") or v.get("player",{}).get("name")
                            if n: found.append(str(n))
                        elif isinstance(v,str): found.append(v)
                    walk(v)
            elif isinstance(o,list):
                for v in o: walk(v)
        walk(details)
        if found: return found[0],"explicit"
    return None,"unavailable"


def mom_score(p, winp, elo, opp):
    if p.matches<=0: return .15
    rating=np.clip((p.rating-6.2)/1.2,0,1.5)
    minutes=np.clip((p.minutes/max(1,p.matches))/80,0,1.25)
    production=np.clip(.34*p.goals/max(1,p.matches)+.26*p.assists/max(1,p.matches)+.24*p.xg/max(1,p.matches)+.10*p.key/max(1,p.matches),0,1.5)
    strength=sigmoid((elo-opp)/180)
    return float(.42*rating+.14*minutes+.24*production+.14*winp+.06*strength)


# =========================
# CHECKPOINT
# =========================
def save_state(done,results,scores,moms,model_rows,coverage,histories,bundles,blends,leagues,players,last_dates,h2h):
    tmp=CHECKPOINT.with_suffix(".tmp")
    payload={
        "done":list(done),"results":results,"scores":scores,"moms":moms,"model_rows":model_rows,"coverage":coverage,
        "histories":dict(histories),"bundles":bundles,"blends":dict(blends),"leagues":leagues,
        "players":{k:v.__dict__ for k,v in players.items()},"last_dates":last_dates,"h2h":h2h,
    }
    with tmp.open("wb") as f: pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL)
    tmp.replace(CHECKPOINT)


def load_state():
    if not CHECKPOINT.exists(): return None
    try:
        with CHECKPOINT.open("rb") as f: return pickle.load(f)
    except Exception: return None


# =========================
# MAIN
# =========================
def main():
    global START
    START=time.time(); make_feature_names()
    print("=== SOCCER BACKTEST FINAL / MULTI-SOURCE ===")
    print("Football-Data: results+closing odds | Understat: xG | SofaScore: players/MOM")
    matches=load_matches(); print(f"Matches loaded: {len(matches):,}")
    under_idx,under_cov=build_understat_index(matches); print(f"Understat matched: {len(under_idx):,}")

    groups=list(matches.groupby(["League","SeasonStart","Season"],sort=True))
    ck=load_state()
    if ck:
        done=set(ck.get("done",[])); results=ck.get("results",[]); scores=ck.get("scores",[]); moms=ck.get("moms",[])
        model_rows=ck.get("model_rows",[]); coverage=ck.get("coverage",[]); histories=defaultdict(list,ck.get("histories",{}))
        bundles=ck.get("bundles",{}); blends=defaultdict(lambda:(.56,.28,.16),ck.get("blends",{})); leagues=ck.get("leagues",{})
        players=defaultdict(Player)
        for k,v in ck.get("players",{}).items():
            p=Player(); p.__dict__.update(v); players[k]=p
        last_dates=ck.get("last_dates",{}); h2h=ck.get("h2h",{})
        print(f"Checkpoint resume: {len(done)} groups")
    else:
        done=set(); results=[]; scores=[]; moms=[]; model_rows=[]; coverage=[]; histories=defaultdict(list); bundles={}
        blends=defaultdict(lambda:(.56,.28,.16)); leagues={c:League() for c in LEAGUES}; players=defaultdict(Player); last_dates={}; h2h={}
    for c in LEAGUES:
        if c not in leagues: leagues[c]=League()

    counter=0
    for (code,year,season),group in groups:
        gkey=f"{code}|{year}|{season}"
        if gkey in done: continue
        if left()<35:
            save_state(done,results,scores,moms,model_rows,coverage,histories,bundles,blends,leagues,players,last_dates,h2h); print("[SAFE STOP] checkpoint saved"); break

        L=leagues[code]; L.new_season(); hist=histories[code]
        bundle=fit_ensemble(hist) if len(hist)>=MIN_TRAIN else None
        if bundle: bundles[code]=bundle
        source_counts=defaultdict(int)
        print(f"[{code}] {season}: {len(group)} matches")

        for i,(_,row) in enumerate(group.iterrows()):
            if left()<25:
                save_state(done,results,scores,moms,model_rows,coverage,histories,bundles,blends,leagues,players,last_dates,h2h); print("[SAFE STOP] checkpoint saved"); return
            h=L.team(row.HomeTeam); a=L.team(row.AwayTeam)

            # ---------- PRE-MATCH ----------
            x=make_features(row,h,a,L,last_dates,h2h)
            market=closing_market(row)
            score_p,top_scores,lh,la=score_model(h,a,L)
            if len(hist)>=MIN_TRAIN:
                if bundle is None or i%RETRAIN_EVERY==0:
                    bundle=fit_ensemble(hist); bundles[code]=bundle
                ml=pred_ml(bundle,x)
            else:
                ml=norm3(.60*market[:3]+.40*score_p)
            lw,mw,sw=blends[code]
            final=norm3(lw*ml+mw*market[:3]+sw*score_p)
            pred=int(np.argmax(final)); actual=int(row.Result)

            results.append({
                "League":code,"Season":season,"Date":row.Date,"HomeTeam":row.HomeTeam,"AwayTeam":row.AwayTeam,
                "Predicted":"HDA"[pred],"Actual":"HDA"[actual],"Correct":int(pred==actual),
                "HomeProb":final[0],"DrawProb":final[1],"AwayProb":final[2],"Confidence":float(final.max()),
                "MarketHome":market[0],"MarketDraw":market[1],"MarketAway":market[2],
                "MLHome":ml[0],"MLDraw":ml[1],"MLAway":ml[2],"PoissonHome":score_p[0],"PoissonDraw":score_p[1],"PoissonAway":score_p[2],
                "LambdaHome":lh,"LambdaAway":la,
            })

            ah,aa=int(row.FTHG),int(row.FTAG)
            for rank,(sh,sa,prob) in enumerate(top_scores,1):
                scores.append({"League":code,"Season":season,"Date":row.Date,"HomeTeam":row.HomeTeam,"AwayTeam":row.AwayTeam,
                               "Rank":rank,"PredScore":f"{sh}-{sa}","Probability":prob,"ActualScore":f"{ah}-{aa}",
                               "Hit":int(sh==ah and sa==aa),"AbsGoalError":abs(sh-ah)+abs(sa-aa)})

            # ---------- MOM: candidate prediction uses ONLY PRE-MATCH PLAYER HISTORY ----------
            # We intentionally do not use this match's lineup to select candidates.
            home_candidates=[]; away_candidates=[]
            for (lg,team,name),ps in list(players.items()):
                if lg!=code: continue
                if team==row.HomeTeam: home_candidates.append((name,ps,h.elo,a.elo,final[0]))
                elif team==row.AwayTeam: away_candidates.append((name,ps,a.elo,h.elo,final[2]))
            cand=[]
            for name,ps,elo,opp,wp in home_candidates+away_candidates:
                if ps.matches>0:
                    cand.append((name,ps,wp,elo,opp))
            cand.sort(key=lambda z:mom_score(z[1],z[2],z[3],z[4]),reverse=True)

            # ---------- POST-MATCH EXTERNAL DATA ----------
            # Only now may same-match SofaScore rating/lineup/MOM be read into state.
            if ENABLE_SOFASCORE:
                ids=sofascore_event_ids(row.DateParsed.strftime("%Y-%m-%d"),row.HomeTeam,row.AwayTeam)
                if ids:
                    lineups,details=sofa_event(ids[0])
                    if lineups:
                        source_counts["SofaScore"]+=1
                        actual_name,method=actual_mom(details,lineups)
                        # Top-4 prediction was calculated before accessing the lineup.
                        for rank,(name,ps,wp,elo,opp) in enumerate(cand[:4],1):
                            moms.append({"League":code,"Season":season,"Date":row.Date,"HomeTeam":row.HomeTeam,"AwayTeam":row.AwayTeam,
                                         "Rank":rank,"Player":name,"Team":row.HomeTeam if wp==final[0] else row.AwayTeam,
                                         "Position":ps.position,"PreMatchMOMScore":mom_score(ps,wp,elo,opp),"ActualMOM":actual_name,
                                         "Hit":int(actual_name is not None and name.strip().lower()==str(actual_name).strip().lower()),
                                         "ActualMOMMethod":method})
                        # Update player state after prediction.
                        for p in extract_players(lineups):
                            team=row.HomeTeam if p["side"]=="H" else row.AwayTeam
                            players[(code,team,p["name"])].update(p)
                    else:
                        source_counts["SofaScore_missing"]+=1
                else:
                    source_counts["SofaScore_no_event"]+=1

            # Understat xG is also post-match information for this historical match.
            uk=(code,row.DateParsed.date().isoformat(),row.HomeTeam,row.AwayTeam)
            u=under_idx.get(uk)
            if u:
                h.update(ah,aa,3 if ah>aa else 1 if ah==aa else 0,"H",
                         {"shots":sf(row.get("HS")),"sot":sf(row.get("HST")),"corners":sf(row.get("HC"))},u["hxg"],u["axg"])
                a.update(aa,ah,0 if ah>aa else 1 if ah==aa else 3,"A",
                         {"shots":sf(row.get("AS")),"sot":sf(row.get("AST")),"corners":sf(row.get("AC"))},u["axg"],u["hxg"])
                source_counts["Understat"]+=1
            else:
                h.update(ah,aa,3 if ah>aa else 1 if ah==aa else 0,"H",
                         {"shots":sf(row.get("HS")),"sot":sf(row.get("HST")),"corners":sf(row.get("HC"))})
                a.update(aa,ah,0 if ah>aa else 1 if ah==aa else 3,"A",
                         {"shots":sf(row.get("AS")),"sot":sf(row.get("AST")),"corners":sf(row.get("AC"))})

            # Elo after result
            expected=sigmoid((h.elo+55-a.elo)/400); actual_h=1 if actual==0 else .5 if actual==1 else 0
            delta=18*(1+math.log1p(max(1,abs(ah-aa))))*(actual_h-expected)
            h.elo+=delta; a.elo-=delta; L.update(ah,aa)

            pair=tuple(sorted([row.HomeTeam,row.AwayTeam])); h2h.setdefault((code,pair),[]).append(actual)
            last_dates[row.HomeTeam]=row.DateParsed; last_dates[row.AwayTeam]=row.DateParsed

            # This record is now eligible for future training; it contains predictions made pre-match.
            hist.append({"x":x,"y":actual,"ml":ml,"market":market[:3],"score":score_p})
            counter+=1
            if counter>=SAVE_EVERY:
                save_state(done,results,scores,moms,model_rows,coverage,histories,bundles,blends,leagues,players,last_dates,h2h); counter=0

        done.add(gkey)
        if len(hist)>=360: blends[code]=optimize_blend(hist)
        if bundle:
            for name,ll in bundle["validation_logloss"].items(): model_rows.append({"League":code,"Season":season,"Model":name,"ValidationLogLoss":ll})
        coverage.append({"League":code,"Season":season,**dict(source_counts)})
        save_state(done,results,scores,moms,model_rows,coverage,histories,bundles,blends,leagues,players,last_dates,h2h)

    # =========================
    # REPORTS
    # =========================
    rdf=pd.DataFrame(results); sdf=pd.DataFrame(scores); mdf=pd.DataFrame(moms); modf=pd.DataFrame(model_rows)
    if not rdf.empty:
        y=rdf.Actual.map({"H":0,"D":1,"A":2}).to_numpy(); p=rdf[["HomeProb","DrawProb","AwayProb"]].to_numpy()
        br=np.mean([brier_score_loss((y==c).astype(int),p[:,c]) for c in range(3)])
        pd.DataFrame([{"Matches":len(rdf),"Accuracy":accuracy_score(y,np.argmax(p,1)),"LogLoss":log_loss(y,p,labels=[0,1,2]),"Brier":br,"MeanConfidence":rdf.Confidence.mean()}]).to_csv(ROOT/"overall_summary.csv",index=False)
        rdf.to_csv(ROOT/"backtest_results.csv",index=False,encoding="utf-8-sig")
        rows=[]
        for lg,g in rdf.groupby("League"):
            gy=g.Actual.map({"H":0,"D":1,"A":2}).to_numpy(); gp=g[["HomeProb","DrawProb","AwayProb"]].to_numpy()
            rows.append({"League":lg,"Matches":len(g),"Accuracy":g.Correct.mean(),"LogLoss":log_loss(gy,gp,labels=[0,1,2]),"MeanConfidence":g.Confidence.mean()})
        pd.DataFrame(rows).to_csv(ROOT/"league_summary.csv",index=False)
        rows=[]
        for s,g in rdf.groupby("Season"):
            gy=g.Actual.map({"H":0,"D":1,"A":2}).to_numpy(); gp=g[["HomeProb","DrawProb","AwayProb"]].to_numpy()
            rows.append({"Season":s,"Matches":len(g),"Accuracy":g.Correct.mean(),"LogLoss":log_loss(gy,gp,labels=[0,1,2]),"MeanConfidence":g.Confidence.mean()})
        pd.DataFrame(rows).to_csv(ROOT/"season_summary.csv",index=False)
        pd.DataFrame([{"Threshold":t,"Matches":int((rdf.Confidence>=t).sum()),"Accuracy":rdf.loc[rdf.Confidence>=t,"Correct"].mean()} for t in [.50,.55,.60,.65,.70,.75,.80]]).to_csv(ROOT/"confidence_summary.csv",index=False)
    if not sdf.empty:
        sdf.to_csv(ROOT/"backtest_scores.csv",index=False,encoding="utf-8-sig")
        grp=sdf.groupby(["League","Season","Date","HomeTeam","AwayTeam"])
        pd.DataFrame([{"ScoreTop1HitRate":sdf[sdf.Rank==1].Hit.mean(),"ScoreTop3HitRate":grp.Hit.max().mean(),"MeanAbsoluteGoalError":sdf.AbsGoalError.mean()}]).to_csv(ROOT/"score_summary.csv",index=False)
    if not mdf.empty:
        mdf.to_csv(ROOT/"backtest_mom.csv",index=False,encoding="utf-8-sig")
        grp=mdf.groupby(["League","Season","Date","HomeTeam","AwayTeam"])
        pd.DataFrame([{"MOMTop1HitRate":mdf[mdf.Rank==1].Hit.mean(),"MOMTop4HitRate":grp.Hit.max().mean(),"EvaluatedMatches":grp.ngroups,"Rows":len(mdf)}]).to_csv(ROOT/"mom_summary.csv",index=False)
    if not modf.empty: modf.to_csv(ROOT/"model_comparison.csv",index=False)
    cov=under_cov+coverage
    if cov: pd.DataFrame(cov).to_csv(ROOT/"data_coverage.csv",index=False)

    # ExtraTrees feature importance from latest fitted bundle, when available.
    for code,b in bundles.items():
        m=b.get("models",{}).get("ExtraTrees") if isinstance(b,dict) else None
        if m is not None and hasattr(m,"feature_importances_"):
            fi=pd.DataFrame({"League":code,"Feature":FEATURE_NAMES,"Importance":m.feature_importances_}).sort_values("Importance",ascending=False)
            fi.to_csv(ROOT/"feature_importance.csv",index=False); break

    if len(done)==len(groups):
        try: CHECKPOINT.unlink()
        except Exception: pass

    print("="*62)
    print("BACKTEST FINISHED")
    print(f"Runtime: {(time.time()-START)/60:.2f} min")
    if not rdf.empty: print(f"1X2 Accuracy: {rdf.Correct.mean()*100:.2f}% | LogLoss: {log_loss(y,p,labels=[0,1,2]):.5f}")
    if not sdf.empty:
        print(f"Score Top-1: {sdf[sdf.Rank==1].Hit.mean()*100:.2f}% | Top-3: {sdf.groupby(['League','Season','Date','HomeTeam','AwayTeam']).Hit.max().mean()*100:.2f}%")
    if not mdf.empty:
        print(f"MOM Top-1: {mdf[mdf.Rank==1].Hit.mean()*100:.2f}% | Top-4: {mdf.groupby(['League','Season','Date','HomeTeam','AwayTeam']).Hit.max().mean()*100:.2f}%")
    else: print("MOM: SofaScore historical event/player data could not be matched.")
    print("="*62)


if __name__=="__main__":
    main()
