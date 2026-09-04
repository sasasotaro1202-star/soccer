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
import re
import time
import warnings
from datetime import datetime, timezone
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
    # User-specified core: European Big 5 + Netherlands
    "E0": "Premier League",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "SP1": "La Liga",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    # Domestic Japan
    "J1": "J1 League",
    "J2": "J2 League",
    "J3": "J3 League",
    # UEFA / cup competitions
    "UCL": "UEFA Champions League",
    "UEL": "UEFA Europa League",
    "DFBP": "DFB-Pokal",
}

# 2015-16 through 2025-26: substantially larger chronological sample.
SEASONS = list(range(2010, 2026))
FD_URL = "https://www.football-data.co.uk/mmz4281/{folder}/{league}.csv"

ROOT = Path(".")
CACHE = ROOT / "cache"
FD_CACHE = CACHE / "football_data"
UNDERSTAT_CACHE = CACHE / "understat"
SOFA_CACHE = CACHE / "sofascore"
OPENFOOTBALL_CACHE = Path(os.getenv("OPENFOOTBALL_CACHE", str(CACHE / "openfootball")))
CHECKPOINT = ROOT / "backtest_checkpoint_v5.pkl"
COMPLETE_MARKER = ROOT / "BACKTEST_COMPLETE_V5"

# Optimization target: maximize out-of-sample quality as far as the data allows.
# "100%" is an optimization target, not a promise of perfect real-world accuracy.
OPTIMIZATION_TARGET = 1.00
INCLUDE_CLUB_FRIENDLIES = os.getenv("INCLUDE_CLUB_FRIENDLIES", "1") == "1"
FRIENDLY_TOURNAMENT_ID = 853  # SofaScore: Club Friendly Games
FRIENDLY_MIN_YEAR = int(os.getenv("FRIENDLY_MIN_YEAR", "2010"))
FRIENDLY_MAX_PAGES = int(os.getenv("FRIENDLY_MAX_PAGES", "250"))
FRIENDLY_COVERAGE = []

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
SOFASCORE_DETAILS = os.getenv("SOFASCORE_DETAILS", "0") == "1"

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
        "FC Bayern München": "Bayern Munich", "Bayern München": "Bayern Munich", "Borussia Dortmund": "Dortmund",
        "Bayer 04 Leverkusen": "Leverkusen", "Eintracht Frankfurt": "Ein Frankfurt", "1. FSV Mainz 05": "Mainz",
        "VfB Stuttgart": "Stuttgart", "VfL Wolfsburg": "Wolfsburg", "SC Freiburg": "Freiburg",
        "TSG 1899 Hoffenheim": "Hoffenheim", "Hertha BSC": "Hertha", "FC Schalke 04": "Schalke 04",
        "Athletic Club": "Ath Bilbao", "Athletic Club (ESP)": "Ath Bilbao", "Real Madrid CF": "Real Madrid",
        "FC Barcelona": "Barcelona", "Club Atlético de Madrid": "Ath Madrid", "Atletico Madrid": "Ath Madrid",
        "Paris Saint-Germain FC": "Paris SG", "Paris Saint-Germain": "Paris SG", "FC Internazionale Milano": "Inter Milan",
        "SSC Napoli": "Napoli", "Sport Lisboa e Benfica": "Benfica", "AFC Ajax": "Ajax", "PSV Eindhoven": "PSV",
        "Manchester City FC": "Man City", "Manchester United FC": "Man United", "Liverpool FC": "Liverpool",
        "Tottenham Hotspur FC": "Tottenham", "Chelsea FC": "Chelsea", "Arsenal FC": "Arsenal",
    }
    return aliases.get(s, s)



def team_key(x):
    import unicodedata
    s=unicodedata.normalize("NFKD",str(x or "")).lower()
    aliases={
        "fc bayern munchen":"bayern","bayern munich":"bayern","bayern munchen":"bayern",
        "borussia dortmund":"dortmund","bayer 04 leverkusen":"leverkusen","bayer leverkusen":"leverkusen",
        "eintracht frankfurt":"frankfurt","ein frankfurt":"frankfurt","borussia monchengladbach":"monchengladbach",
        "borussia mönchengladbach":"monchengladbach","fc koln":"cologne","fc cologne":"cologne",
        "1 fsv mainz 05":"mainz","mainz 05":"mainz","vfb stuttgart":"stuttgart","vfl wolfsburg":"wolfsburg",
        "sv werder bremen":"bremen","werder bremen":"bremen","sc freiburg":"freiburg","tsg 1899 hoffenheim":"hoffenheim",
        "hertha bsc":"hertha berlin","hertha berlin":"hertha berlin","1 fc union berlin":"union berlin","fc schalke 04":"schalke",
    }
    s=aliases.get(s,s)
    for token in (" football club"," fc"," cf"," afc"," sc"," sk"," fk"," sv"," kv"," ac"," as"," ss"," sfc"):
        s=s.replace(token," ")
    s="".join(ch if ch.isalnum() else " " for ch in s)
    return " ".join(s.split())


def same_team(a,b):
    ka,kb=team_key(a),team_key(b)
    return ka==kb or ka in kb or kb in ka

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


def parse_openfootball_text(text, code, year, season):
    """Parse openfootball Football.TXT match lines with date context."""
    rows=[]; current_date=None
    for raw in text.splitlines():
        line=raw.strip()
        if not line: continue
        # Full date e.g. Tue Sep 16 2025 / short date e.g. Fri Aug 22
        for fmt in ("%a %b %d %Y", "%a %b %d"):
            try:
                dt=pd.to_datetime(line,format=fmt)
                if fmt=="%a %b %d":
                    y=year if dt.month>=7 else year+1
                    dt=dt.replace(year=y)
                current_date=dt
                break
            except Exception: pass
        if current_date is None: continue
        # Match line: [time] Home v Away score [extra text]
        m=re.match(r'^\s*(?:\d{1,2}:\d{2}\s+)?(.+?)\s+v\s+(.+?)\s+(\d+)\s*-\s*(\d+)(?:\s|$)',line)
        if not m: continue
        home=team_name(m.group(1).strip()); away=team_name(m.group(2).strip())
        hg,ag=int(m.group(3)),int(m.group(4))
        rows.append({"League":code,"SeasonStart":year,"Season":season,"DateParsed":current_date,
                     "Date":current_date.strftime("%d/%m/%Y"),"HomeTeam":home,"AwayTeam":away,
                     "FTHG":hg,"FTAG":ag})
    return rows


def load_openfootball_repo(repo_dir, code, year, filename):
    path=OPENFOOTBALL_CACHE/repo_dir/f"{year}-{str(year+1)[-2:]}"/filename
    if not path.exists(): return []
    try:
        return parse_openfootball_text(path.read_text(encoding="utf-8",errors="ignore"),code,year,f"{year}/{str(year+1)[-2:]}")
    except Exception: return []


def load_football_json_matches(year, code, filename):
    path=OPENFOOTBALL_CACHE/"football.json"/str(year)/filename
    if not path.exists(): return []
    try:
        obj=json.loads(path.read_text(encoding="utf-8")); rows=[]
        for m in obj.get("matches",[]):
            sc=m.get("score",{})
            ft=sc.get("ft") if isinstance(sc,dict) else None
            if not isinstance(ft,list) or len(ft)<2: continue
            dt=pd.to_datetime(m.get("date"),errors="coerce")
            if pd.isna(dt): continue
            rows.append({"League":code,"SeasonStart":year,"Season":f"{year}/{str(year+1)[-2:]}","DateParsed":dt,
                         "Date":dt.strftime("%d/%m/%Y"),"HomeTeam":team_name(m.get("team1")),"AwayTeam":team_name(m.get("team2")),
                         "FTHG":int(ft[0]),"FTAG":int(ft[1])})
        return rows
    except Exception: return []


def load_japan_fallback():
    """Football-Data Japan aggregate file; schema may contain Div/J1/J2/J3."""
    path=FD_CACHE/"JPN_all.csv"
    if path.exists():
        try: d=pd.read_csv(path)
        except Exception: return []
    else:
        if left()<12: return []
        try:
            r=HTTP.get("https://www.football-data.co.uk/new/JPN.csv",timeout=TIMEOUT); r.raise_for_status()
            path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(r.content); d=pd.read_csv(path)
        except Exception: return []
    rows=[]
    divcol=next((c for c in d.columns if str(c).lower() in ("div","division","league")),None)
    mapping={"J1":"J1","J2":"J2","J3":"J3","1":"J1","2":"J2","3":"J3"}
    for _,r in d.iterrows():
        dt=pd.to_datetime(r.get("Date"),dayfirst=True,errors="coerce")
        if pd.isna(dt): continue
        div=mapping.get(str(r.get(divcol)).strip(),"J1") if divcol else "J1"
        y=int(dt.year if dt.month>=2 else dt.year-1)
        if y not in SEASONS: continue
        rows.append({"League":div,"SeasonStart":y,"Season":f"{y}/{str(y+1)[-2:]}","DateParsed":dt,
                     "Date":r.get("Date"),"HomeTeam":team_name(r.get("HomeTeam")),"AwayTeam":team_name(r.get("AwayTeam")),
                     "FTHG":sf(r.get("FTHG")),"FTAG":sf(r.get("FTAG"))})
    return rows


def _friendly_season_start(name):
    s=str(name or "")
    m=re.search(r"(20\d{2})",s)
    if m: return int(m.group(1))
    m=re.search(r"(\d{2})/(\d{2})",s)
    if m:
        y=int(m.group(1)); return 2000+y if y>=10 else 2000+y
    return None


def load_sofascore_friendlies(target_teams):
    """Load historical Club Friendly Games involving clubs from the selected ecosystem.

    We intentionally do NOT ingest every friendly worldwide. Only a match with at least
    one team already observed in the selected Big-5/Eredivisie/UCL/UEL/DFB/J-League
    ecosystem is retained. This adds preseason/midseason context without exploding the
    dataset with unrelated clubs. SofaScore exposes Club Friendly Games as tournament 853.
    """
    if not INCLUDE_CLUB_FRIENDLIES or not ENABLE_SOFASCORE or left()<30:
        return [], []
    base="https://www.sofascore.com/api/v1"
    seasons_cache=SOFA_CACHE/"friendlies"/"seasons.json"
    data=get_json(f"{base}/unique-tournament/{FRIENDLY_TOURNAMENT_ID}/seasons",seasons_cache)
    seasons=data.get("seasons",[]) if isinstance(data,dict) else []
    target={team_key(x) for x in target_teams if x}
    if not target:
        return [], []
    rows=[]; cov=[]
    seen=set()
    for season_obj in seasons:
        sy=_friendly_season_start(season_obj.get("name") or season_obj.get("year"))
        if sy is None or sy<FRIENDLY_MIN_YEAR or sy>=2026: continue
        sid=season_obj.get("id")
        if not sid: continue
        page=0; pages=0; season_events=0; kept=0
        while pages<FRIENDLY_MAX_PAGES and left()>35:
            cp=SOFA_CACHE/"friendlies"/f"{sid}_last_{page}.json"
            obj=get_json(f"{base}/unique-tournament/{FRIENDLY_TOURNAMENT_ID}/season/{sid}/events/last/{page}",cp)
            if not isinstance(obj,dict): break
            evs=obj.get("events",[]) or []
            if not evs: break
            season_events+=len(evs)
            for ev in evs:
                status=ev.get("status",{}) or {}
                if status.get("type")!="finished" and status.get("code") not in (100,): continue
                ht=team_name((ev.get("homeTeam") or {}).get("name")); at=team_name((ev.get("awayTeam") or {}).get("name"))
                if not ht or not at: continue
                if team_key(ht) not in target and team_key(at) not in target: continue
                ts=ev.get("startTimestamp")
                dt=pd.to_datetime(ts,unit="s",errors="coerce") if ts else pd.NaT
                if pd.isna(dt): continue
                season_start=int(dt.year if dt.month>=7 else dt.year-1)
                if season_start<FRIENDLY_MIN_YEAR or season_start>=2026: continue
                hs=sf((ev.get("homeScore") or {}).get("current")); aas=sf((ev.get("awayScore") or {}).get("current"))
                if not (np.isfinite(hs) and np.isfinite(aas)): continue
                key=("FRI",season_start,dt.date(),team_key(ht),team_key(at))
                if key in seen: continue
                seen.add(key); kept+=1
                rows.append({"League":"FRI","SeasonStart":season_start,"Season":f"{season_start}/{str(season_start+1)[-2:]}","DateParsed":dt,
                             "Date":dt.strftime("%d/%m/%Y"),"HomeTeam":ht,"AwayTeam":at,"FTHG":int(hs),"FTAG":int(aas),
                             "FriendlyEventId":str(ev.get("id")),"FriendlyTournament":"Club Friendly Games"})
            if not obj.get("hasNextPage"): break
            page+=1; pages+=1
        cov.append({"League":"FRI","SeasonStart":sy,"Source":"SofaScore_ClubFriendlyGames","Available":season_events,"Matched":kept})
    return rows,cov


def load_matches():
    frames=[]
    # Core domestic leagues: Big 5 + Eredivisie, season-by-season Football-Data files.
    for code in ("E0","D1","I1","SP1","F1","N1"):
        for year in SEASONS:
            if left()<20: break
            d=load_fd(code,year)
            if d.empty: continue
            d["League"]=code; d["SeasonStart"]=year; d["Season"]=f"{year}/{str(year+1)[-2:]}"
            d["DateParsed"]=pd.to_datetime(d["Date"],dayfirst=True,errors="coerce")
            frames.append(d)

    # Japan J1/J2/J3. Prefer openfootball's season JSON, then Football-Data aggregate fallback.
    for year in SEASONS:
        for code,fn in (("J1","jp.1.json"),("J2","jp.2.json"),("J3","jp.3.json")):
            rows=load_football_json_matches(year,code,fn)
            if rows: frames.append(pd.DataFrame(rows))
    jp_fallback=load_japan_fallback()
    if jp_fallback:
        existing={(r["League"],r["SeasonStart"],r["DateParsed"].date(),r["HomeTeam"],r["AwayTeam"]) for f in frames if "League" in f.columns for _,r in f.iterrows() if r.get("League") in ("J1","J2","J3")}
        extra=[r for r in jp_fallback if (r["League"],r["SeasonStart"],r["DateParsed"].date(),r["HomeTeam"],r["AwayTeam"]) not in existing]
        if extra: frames.append(pd.DataFrame(extra))

    # UEFA competitions: openfootball provides season files for CL and, where available, EL.
    for year in SEASONS:
        cl=load_openfootball_repo("champions-league","UCL",year,"cl.txt")
        el=load_openfootball_repo("champions-league","UEL",year,"el.txt")
        if cl: frames.append(pd.DataFrame(cl))
        if el: frames.append(pd.DataFrame(el))

    # DFB-Pokal: retain ties involving clubs observed in the Bundesliga ecosystem.
    # The source may not provide 2.Bundesliga as a separate feed, so we use the
    # observed German top-flight club universe as the conservative partial scope.
    german_teams=set()
    for f in frames:
        if "League" in f.columns and f["League"].isin(["D1"]).any():
            german_teams.update(f.loc[f.League=="D1","HomeTeam"].dropna().tolist())
            german_teams.update(f.loc[f.League=="D1","AwayTeam"].dropna().tolist())
    for year in SEASONS:
        cup=load_openfootball_repo("deutschland","DFBP",year,"cup.txt")
        if cup:
            if german_teams:
                gkeys={team_key(x) for x in german_teams}
                cup=[r for r in cup if team_key(r["HomeTeam"]) in gkeys or team_key(r["AwayTeam"]) in gkeys]
            if cup: frames.append(pd.DataFrame(cup))

    # Club friendlies: retain only friendlies involving clubs already observed in the
    # selected ecosystem. These are useful as squad/fitness/player-form context.
    if INCLUDE_CLUB_FRIENDLIES and ENABLE_SOFASCORE:
        target_teams=set()
        for f in frames:
            if "HomeTeam" in f.columns:
                target_teams.update(f["HomeTeam"].dropna().astype(str).tolist())
                target_teams.update(f["AwayTeam"].dropna().astype(str).tolist())
        fr,fc=load_sofascore_friendlies(target_teams)
        if fr:
            frames.append(pd.DataFrame(fr))
            print(f"Club friendlies loaded: {len(fr):,}")
        # Save coverage in a module-global cache consumed later by reports.
        global FRIENDLY_COVERAGE
        FRIENDLY_COVERAGE=fc

    if not frames: raise RuntimeError("No match data available.")
    d=pd.concat(frames,ignore_index=True,sort=False)
    d["DateParsed"]=pd.to_datetime(d["DateParsed"] if "DateParsed" in d.columns else d["Date"],errors="coerce")
    d["HomeTeam"]=d["HomeTeam"].map(team_name); d["AwayTeam"]=d["AwayTeam"].map(team_name)
    d["FTHG"]=pd.to_numeric(d["FTHG"],errors="coerce"); d["FTAG"]=pd.to_numeric(d["FTAG"],errors="coerce")
    d=d.dropna(subset=["DateParsed","HomeTeam","AwayTeam","FTHG","FTAG"]).copy()
    d["FTHG"]=d["FTHG"].astype(int); d["FTAG"]=d["FTAG"].astype(int)
    d["Result"]=np.where(d.FTHG>d.FTAG,0,np.where(d.FTHG==d.FTAG,1,2))
    # Deduplicate source overlaps while preserving competition identity.
    d=d.drop_duplicates(subset=["League","SeasonStart","DateParsed","HomeTeam","AwayTeam"],keep="first")
    return d.sort_values(["DateParsed","League","SeasonStart"],kind="stable").reset_index(drop=True)


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

PLAYER_FEATURE_BASE = [
    "attack","creation","progression","passing","defending","aerial","goalkeeping",
    "discipline","security","availability","squad_depth","top_attack","top_creation","top_defending","finishing_efficiency","shot_quality","chance_creation","progressive_carrying","progressive_passing","crossing","long_passing","pressing","defensive_actions","duel_strength","ball_retention","directness","set_piece_threat"
]

def team_player_features(code, team, players):
    rows=[]
    for (tm,name),ps in players.items():
        if tm!=team or ps.matches<=0: continue
        v=ps.style_vector(); rel=ps.reliability()
        recent=np.mean(ps.recent_minutes) if ps.recent_minutes else 0.
        avail=float(np.clip(recent/70.,0,1.25))
        rows.append((ps,rel,v,avail))
    if not rows: return [0.0]*len(PLAYER_FEATURE_BASE)
    w=np.asarray([max(.05,r[1]) for r in rows],dtype=float); w/=w.sum()
    def avg(k): return float(np.average([r[2].get(k,0.) for r in rows],weights=w))
    def top(k): return float(np.mean(sorted([r[2].get(k,0.) for r in rows],reverse=True)[:min(5,len(rows))]))
    attack=avg("finishing")+.35*avg("shot_volume")
    creation=avg("chance_creation")
    progression=avg("ball_progression")
    passing=avg("passing")
    defending=avg("defending")
    aerial=avg("aerial")
    goalkeeping=avg("goalkeeping")
    discipline=avg("discipline")
    security=avg("ball_security")
    availability=float(np.average([r[3] for r in rows],weights=w))
    depth=float(min(len(rows),25)/11.)
    vals=[attack,creation,progression,passing,defending,aerial,goalkeeping,discipline,security,availability,depth,
          top("finishing"),top("chance_creation"),top("defending"),avg("finishing_efficiency"),avg("shot_quality"),
          avg("chance_creation"),avg("progressive_carrying"),avg("progressive_passing"),avg("crossing"),avg("long_passing"),
          avg("pressing"),avg("defensive_actions"),avg("duel_strength"),avg("ball_retention"),avg("directness"),avg("set_piece_threat")]
    return [float(np.nan_to_num(v)) for v in vals]

def make_feature_names():
    base = ["elo","ppg","gfpg","gapg","form","gf_recent","ga_recent","venue_ppg","venue_gfpg","venue_gapg","ema_gf","ema_ga","xg_for","xg_against","shots","sot","corners"]
    global FEATURE_NAMES
    FEATURE_NAMES = [f"H_{x}" for x in base] + [f"A_{x}" for x in base] + [
        "elo_diff","ppg_diff","gfpg_diff","def_diff","attack_diff","def_attack_diff",
        "xg_for_diff","xg_against_diff","rest_h","rest_a","rest_diff","elo_sigmoid","ppg_sigmoid",
        "market_h","market_d","market_a","market_favorite","market_entropy","market_dispersion",
        "h2h_home_rate","h2h_draw_rate","h2h_away_rate","h2h_count"
    ] + [f"H_player_{x}" for x in PLAYER_FEATURE_BASE] + [f"A_player_{x}" for x in PLAYER_FEATURE_BASE] + [
        "player_attack_diff","player_creation_diff","player_progression_diff","player_passing_diff","player_defending_diff",
        "player_aerial_diff","player_goalkeeping_diff","player_availability_diff","player_depth_diff",
        "player_finishing_efficiency_diff","player_shot_quality_diff","player_progressive_carrying_diff","player_progressive_passing_diff",
        "player_crossing_diff","player_long_passing_diff","player_pressing_diff","player_defensive_actions_diff",
        "player_duel_strength_diff","player_ball_retention_diff","player_directness_diff","player_set_piece_threat_diff",
        "global_elo_diff","competition_elo_diff","is_friendly","friendly_low_information"
    ]


def make_features(row, h, a, L, last_dates, h2h, players=None, global_elo=None):
    lh, la = L.av()
    market = closing_market(row)
    rest_h = 7.0 if row.HomeTeam not in last_dates else np.clip((row.DateParsed-last_dates[row.HomeTeam]).total_seconds()/86400, .5, 21)
    rest_a = 7.0 if row.AwayTeam not in last_dates else np.clip((row.DateParsed-last_dates[row.AwayTeam]).total_seconds()/86400, .5, 21)
    pair = tuple(sorted([row.HomeTeam,row.AwayTeam]))
    hh = h2h.get((row.League,pair), [])[-8:]
    hc = max(1,len(hh))
    # stored result is from perspective of the team that was home in each old match; for
    # H2H feature we keep neutral rates based on raw outcome history only.
    raw_h=sum(v==0 for v in hh)/hc if hh else 1/3
    raw_d=sum(v==1 for v in hh)/hc if hh else 1/3
    raw_a=sum(v==2 for v in hh)/hc if hh else 1/3
    if pair and row.HomeTeam==pair[0]: h2h_home,h2h_draw,h2h_away=raw_h,raw_d,raw_a
    else: h2h_home,h2h_draw,h2h_away=raw_a,raw_d,raw_h
    hp = team_player_features(row.League,row.HomeTeam,players) if players is not None else [0.0]*len(PLAYER_FEATURE_BASE)
    ap = team_player_features(row.League,row.AwayTeam,players) if players is not None else [0.0]*len(PLAYER_FEATURE_BASE)
    ge_h=(global_elo or {}).get(row.HomeTeam,1500.0); ge_a=(global_elo or {}).get(row.AwayTeam,1500.0)
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
            h2h_home,h2h_draw,h2h_away,len(hh),
        ] + hp + ap + [
            hp[0]-ap[0],hp[1]-ap[1],hp[2]-ap[2],hp[3]-ap[3],hp[4]-ap[4],hp[5]-ap[5],hp[6]-ap[6],hp[9]-ap[9],hp[10]-ap[10],
            hp[14]-ap[14],hp[15]-ap[15],hp[17]-ap[17],hp[18]-ap[18],hp[19]-ap[19],hp[20]-ap[20],hp[21]-ap[21],hp[22]-ap[22],
            hp[23]-ap[23],hp[24]-ap[24],hp[25]-ap[25],hp[26]-ap[26],
            ge_h-ge_a,h.elo-a.elo,
            1.0 if str(row.League)=="FRI" else 0.0, 1.0 if str(row.League)=="FRI" and (not np.isfinite(market[0]) or np.allclose(market[:3],[1/3,1/3,1/3])) else 0.0
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
    """Pre-match player state. All values are accumulated only from matches already played."""
    STAT_KEYS = [
        "goals","assists","xg","xa","npxg","shots","shots_on_target","key_passes",
        "passes","accurate_passes","long_balls","accurate_long_balls","crosses","accurate_crosses",
        "dribbles","successful_dribbles","tackles","interceptions","clearances","blocks",
        "aerial_won","aerial_lost","ground_won","ground_lost","possession_lost","dispossessed",
        "fouls","was_fouled","yellow","red","big_chances_created","big_chances_missed",
        "through_balls","final_third_passes","progressive_passes","saves","goals_prevented"
    ]
    def __init__(self):
        self.matches=0; self.starts=0; self.minutes=0.; self.rating=6.5
        self.position=""; self.preferred_foot=""; self.height=0.; self.age=0.
        self.last_date=None; self.recent_minutes=deque(maxlen=10); self.recent_ratings=deque(maxlen=10)
        for k in self.STAT_KEYS: setattr(self,k,0.)

    def update(self,p,date=None):
        self.matches += 1
        mins=sf(p.get("minutes"),0); self.minutes += max(0,mins)
        if p.get("started"): self.starts += 1
        rating=sf(p.get("rating"))
        if np.isfinite(rating):
            self.rating=.22*rating+.78*self.rating; self.recent_ratings.append(rating)
        self.recent_minutes.append(mins)
        if p.get("position"): self.position=str(p["position"])
        if p.get("preferred_foot"): self.preferred_foot=str(p["preferred_foot"])
        if np.isfinite(sf(p.get("height"))): self.height=sf(p.get("height"))
        if np.isfinite(sf(p.get("age"))): self.age=sf(p.get("age"))
        for k in self.STAT_KEYS:
            v=sf(p.get(k),0)
            if np.isfinite(v): setattr(self,k,getattr(self,k)+v)
        self.last_date=str(date) if date is not None else self.last_date

    def per90(self,key):
        mins=max(90.,self.minutes)
        return float(getattr(self,key,0.)/mins*90.)

    def reliability(self):
        return float(np.clip((self.minutes/900.)*(0.5+0.5*self.starts/max(1,self.matches)),0,1.5))

    def style_vector(self):
        # Data-derived micro-style vector. Every component is calculated only from
        # matches already completed before the prediction being made.
        shots=max(0.25,self.per90("shots")); xg=max(0.0,self.per90("xg"))
        return {
            "finishing": self.per90("goals") + .35*xg,
            "shot_volume": shots,
            "chance_creation": self.per90("key_passes") + .65*self.per90("xa") + .15*self.per90("big_chances_created"),
            "ball_progression": self.per90("successful_dribbles") + .06*self.per90("progressive_passes") + .08*self.per90("final_third_passes"),
            "passing": self.per90("accurate_passes") + .12*self.per90("accurate_long_balls"),
            "defending": self.per90("tackles") + self.per90("interceptions") + .35*self.per90("blocks") + .25*self.per90("clearances"),
            "aerial": self.per90("aerial_won"),
            "duel_intensity": self.per90("ground_won") + self.per90("aerial_won") + .2*self.per90("fouls"),
            "discipline": self.per90("yellow") + 2*self.per90("red"),
            "ball_security": -(self.per90("possession_lost") + .4*self.per90("dispossessed")),
            "goalkeeping": self.per90("saves") + self.per90("goals_prevented"),
            "finishing_efficiency": self.per90("goals")/shots,
            "shot_quality": xg/shots,
            "progressive_carrying": self.per90("successful_dribbles") + .05*self.per90("progressive_passes"),
            "progressive_passing": self.per90("progressive_passes") + .05*self.per90("final_third_passes"),
            "crossing": self.per90("accurate_crosses"),
            "long_passing": self.per90("accurate_long_balls"),
            "pressing": self.per90("tackles") + .55*self.per90("interceptions") + .25*self.per90("fouls"),
            "defensive_actions": self.per90("tackles") + self.per90("interceptions") + self.per90("clearances") + self.per90("blocks"),
            "duel_strength": self.per90("ground_won") + self.per90("aerial_won"),
            "ball_retention": self.per90("accurate_passes") - self.per90("possession_lost") - .4*self.per90("dispossessed"),
            "directness": self.per90("shots") + .5*self.per90("successful_dribbles") + .15*self.per90("through_balls"),
            "set_piece_threat": self.per90("crosses") + .5*self.per90("accurate_crosses") + .4*self.per90("key_passes"),
        }

    def style_label(self):
        pos=(self.position or "").upper()
        v=self.style_vector()
        if pos in ("G","GK"): return "sweeper/distributor goalkeeper" if self.per90("accurate_long_balls")>=4 else "shot-stopper goalkeeper"
        if pos in ("D","DF"): 
            if v["aerial"]>3.5: return "aerial defender"
            if v["ball_progression"]>2.5 and v["passing"]>35: return "progressive defender"
            if v["defending"]>7: return "ball-winning defender"
            return "defensive defender"
        if pos in ("M","MF"): 
            if v["chance_creation"]>2.0: return "creative midfielder"
            if v["ball_progression"]>3.0: return "progressive midfielder"
            if v["defending"]>6: return "ball-winning midfielder"
            return "two-way midfielder"
        if pos in ("F","FW","ST","A"): 
            if v["chance_creation"]>2.0: return "creative forward"
            if v["finishing"]>0.65 and v["shot_volume"]>2.0: return "finisher"
            if v["ball_progression"]>2.5: return "dribbling forward"
            return "forward"
        return "outfield player"

    def profile(self,league,team,name):
        v=self.style_vector()
        d={"League":league,"Team":team,"Player":name,"Position":self.position,"PreferredFoot":self.preferred_foot,
           "Height":self.height,"Age":self.age,"Matches":self.matches,"Starts":self.starts,"Minutes":self.minutes,
           "Rating":self.rating,"Reliability":self.reliability(),"Style":self.style_label()}
        for k in self.STAT_KEYS: d[k]=getattr(self,k)
        for k,val in v.items(): d[f"Style_{k}"]=val
        for k in self.STAT_KEYS: d[f"Per90_{k}"]=self.per90(k)
        return d


def stat_value(st, *keys, default=0):
    for k in keys:
        if k in st:
            v=sf(st.get(k))
            if np.isfinite(v): return v
    return default


def extract_players(lineups):
    out=[]
    if not isinstance(lineups,dict): return out
    for side_key,side in (("home","H"),("away","A")):
        block=lineups.get(side_key,{})
        for item in (block.get("players",[]) if isinstance(block,dict) else []):
            if not isinstance(item,dict): continue
            p=item.get("player",{}) or {}; name=p.get("name")
            if not name: continue
            st=item.get("statistics",{}) or {}
            out.append({
                "side":side,"name":str(name),"player_id":p.get("id"),"position":p.get("position"),
                "preferred_foot":p.get("preferredFoot") or p.get("preferredFootType"),"height":sf(p.get("height")),"age":sf(p.get("age")),
                "started":not bool(item.get("substitute",False)),"rating":sf(st.get("rating")),
                "minutes":stat_value(st,"minutesPlayed","minutes"),
                "goals":stat_value(st,"goals"),"assists":stat_value(st,"assists"),"xg":stat_value(st,"expectedGoals","xg"),
                "xa":stat_value(st,"expectedAssists","xA","xa"),"npxg":stat_value(st,"expectedGoalsNonPenalty","npxG"),
                "shots":stat_value(st,"totalShots","shots"),"shots_on_target":stat_value(st,"shotsOnTarget"),
                "key_passes":stat_value(st,"keyPasses"),"passes":stat_value(st,"totalPasses"),"accurate_passes":stat_value(st,"accuratePasses"),
                "long_balls":stat_value(st,"totalLongBalls","longBalls"),"accurate_long_balls":stat_value(st,"accurateLongBalls"),
                "crosses":stat_value(st,"totalCrosses","crosses"),"accurate_crosses":stat_value(st,"accurateCrosses"),
                "dribbles":stat_value(st,"dribbles","totalDribbles"),"successful_dribbles":stat_value(st,"successfulDribbles"),
                "tackles":stat_value(st,"tackles"),"interceptions":stat_value(st,"interceptions"),"clearances":stat_value(st,"clearances"),
                "blocks":stat_value(st,"blockedShots","blocks"),"aerial_won":stat_value(st,"aerialDuelsWon","aerialDuelsWon"),
                "aerial_lost":stat_value(st,"aerialDuelsLost","aerialDuelsLost"),"ground_won":stat_value(st,"groundDuelsWon","groundDuelsWon"),
                "ground_lost":stat_value(st,"groundDuelsLost","groundDuelsLost"),"possession_lost":stat_value(st,"possessionLostCtrl","possessionLost"),
                "dispossessed":stat_value(st,"dispossessed"),"fouls":stat_value(st,"fouls"),"was_fouled":stat_value(st,"wasFouled"),
                "yellow":stat_value(st,"yellowCards"),"red":stat_value(st,"redCards"),"big_chances_created":stat_value(st,"bigChanceCreated","bigChancesCreated"),
                "big_chances_missed":stat_value(st,"bigChanceMissed","bigChancesMissed"),"through_balls":stat_value(st,"accurateThroughBalls","throughBalls"),
                "final_third_passes":stat_value(st,"passesToFinalThird","finalThirdPasses"),"progressive_passes":stat_value(st,"progressivePasses"),
                "saves":stat_value(st,"saves"),"goals_prevented":stat_value(st,"goalsPrevented"),
            })
    return out

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
                if same_team(ev.get("homeTeam",{}).get("name"),home) and same_team(ev.get("awayTeam",{}).get("name"),away):
                    return [str(ev.get("id"))]
    except Exception: pass
    return []


def sofa_event(event_id):
    lc=SOFA_CACHE/"events"/f"{event_id}_lineups.json"
    lineups=get_json(f"https://www.sofascore.com/api/v1/event/{event_id}/lineups",lc)
    details=None
    if SOFASCORE_DETAILS:
        dc=SOFA_CACHE/"events"/f"{event_id}_details.json"
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
    # Fallback: highest-rated player with meaningful minutes from the same-match lineup.
    # This is used only as the realized label after prediction; it is never a pre-match feature.
    best=None
    if isinstance(lineups,dict):
        for side in ("home","away"):
            block=lineups.get(side,{})
            for item in (block.get("players",[]) if isinstance(block,dict) else []):
                st=item.get("statistics",{}) or {}; p=item.get("player",{}) or {}
                n=p.get("name"); r=sf(st.get("rating")); mins=stat_value(st,"minutesPlayed","minutes")
                if n and np.isfinite(r) and mins>=30:
                    if best is None or r>best[0]: best=(r,str(n))
    if best is not None: return best[1],"highest_rating_fallback"
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
def save_state(done,results,scores,moms,model_rows,coverage,histories,bundles,blends,leagues,players,last_dates,h2h,global_elo=None,cursor=0,active_seasons=None):
    tmp=CHECKPOINT.with_suffix(".tmp")
    payload={
        "version":5,"done":list(done),"results":results,"scores":scores,"moms":moms,"model_rows":model_rows,"coverage":coverage,
        "histories":dict(histories),"bundles":bundles,"blends":dict(blends),"leagues":leagues,
        "players":{k:v.__dict__ for k,v in players.items()},"last_dates":last_dates,"h2h":h2h,
        "global_elo":global_elo or {},"cursor":int(cursor),"active_seasons":active_seasons or {}
    }
    with tmp.open("wb") as f: pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL)
    tmp.replace(CHECKPOINT)


def load_state():
    if not CHECKPOINT.exists(): return None
    try:
        with CHECKPOINT.open("rb") as f:
            ck=pickle.load(f)
        return ck if ck.get("version",0)>=5 else None
    except Exception: return None


# =========================
# MAIN
# =========================
def main():
    global START
    START=time.time(); make_feature_names()
    if COMPLETE_MARKER.exists():
        print("=== BACKTEST ALREADY COMPLETE ===")
        return
    print("=== SOCCER BACKTEST V5 / DEEP PLAYER / CLUB FRIENDLIES ===")
    print("Football-Data: results+closing odds | Understat: xG | SofaScore: players/MOM/friendlies")
    print(f"Optimization target: {OPTIMIZATION_TARGET*100:.0f}% (target only; no accuracy guarantee) | Club friendlies: {INCLUDE_CLUB_FRIENDLIES}")
    matches=load_matches(); print(f"Matches loaded: {len(matches):,}")
    under_idx,under_cov=build_understat_index(matches); print(f"Understat matched: {len(under_idx):,}")

    # Process every competition globally by kickoff date. This prevents cross-competition
    # player/team information from leaking from the future into an earlier match.
    matches=matches.sort_values(["DateParsed","League","SeasonStart"],kind="stable").reset_index(drop=True)
    group_sizes=matches.groupby(["League","SeasonStart","Season"],sort=False).size().to_dict()
    group_seen=defaultdict(int)
    group_source_counts=defaultdict(lambda: defaultdict(int))

    ck=load_state()
    if ck:
        done=set(ck.get("done",[])); results=ck.get("results",[]); scores=ck.get("scores",[]); moms=ck.get("moms",[])
        model_rows=ck.get("model_rows",[]); coverage=ck.get("coverage",[]); histories=defaultdict(list,ck.get("histories",{}))
        bundles=ck.get("bundles",{}); blends=defaultdict(lambda:(.56,.28,.16),ck.get("blends",{})); leagues=ck.get("leagues",{})
        players=defaultdict(Player)
        for k,v in ck.get("players",{}).items():
            p=Player(); p.__dict__.update(v); players[k]=p
        last_dates=ck.get("last_dates",{}); h2h=ck.get("h2h",{})
        global_elo=defaultdict(lambda:1500.0,ck.get("global_elo",{})); cursor=int(ck.get("cursor",0))
        active_seasons=dict(ck.get("active_seasons",{}))
        print(f"Checkpoint resume: cursor={cursor:,} / {len(matches):,} | completed groups={len(done)}")
    else:
        done=set(); results=[]; scores=[]; moms=[]; model_rows=[]; coverage=[]; histories=defaultdict(list); bundles={}
        blends=defaultdict(lambda:(.56,.28,.16)); leagues={c:League() for c in LEAGUES}; players=defaultdict(Player); last_dates={}; h2h={}
        global_elo=defaultdict(lambda:1500.0); cursor=0; active_seasons={}
    for c in LEAGUES:
        if c not in leagues: leagues[c]=League()

    code_counts=defaultdict(int)
    counter=0
    while cursor < len(matches):
        if left()<28:
            save_state(done,results,scores,moms,model_rows,coverage,histories,bundles,blends,leagues,players,last_dates,h2h,global_elo,cursor,active_seasons)
            print("[SAFE STOP] checkpoint saved"); break
        row=matches.iloc[cursor]
        code=str(row.League); year=int(row.SeasonStart); season=str(row.Season)
        gkey=f"{code}|{year}|{season}"
        if code not in leagues: leagues[code]=League()
        L=leagues[code]; hist=histories[code]
        # Reset only this competition's domestic/competition state at its season boundary.
        if active_seasons.get(code)!=year:
            if code in active_seasons:
                if len(hist)>=360: blends[code]=optimize_blend(hist)
            L.new_season(); active_seasons[code]=year
            bundle=fit_ensemble(hist) if len(hist)>=MIN_TRAIN else None
            if bundle: bundles[code]=bundle
            print(f"[{code}] {season} started")
        bundle=bundles.get(code)
        h=L.team(row.HomeTeam); a=L.team(row.AwayTeam)
        x=make_features(row,h,a,L,last_dates,h2h,players,global_elo)
        market=closing_market(row)
        score_p,top_scores,lh,la=score_model(h,a,L)
        if len(hist)>=MIN_TRAIN:
            if bundle is None or code_counts[code]%RETRAIN_EVERY==0:
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

        # Pre-match MOM candidates from accumulated player history only.
        cand=[]
        for (team,name),ps in list(players.items()):
            if team==row.HomeTeam: cand.append((name,ps,final[0],h.elo,a.elo))
            elif team==row.AwayTeam: cand.append((name,ps,final[2],a.elo,h.elo))
        cand=[z for z in cand if z[1].matches>0]
        cand.sort(key=lambda z:mom_score(z[1],z[2],z[3],z[4]),reverse=True)

        # Post-match player/event data. Never fed into the current prediction.
        if ENABLE_SOFASCORE:
            ids=sofascore_event_ids(row.DateParsed.strftime("%Y-%m-%d"),row.HomeTeam,row.AwayTeam)
            if ids:
                lineups,details=sofa_event(ids[0])
                if lineups:
                    group_source_counts[gkey]["SofaScore"]+=1
                    actual_name,method=actual_mom(details,lineups)
                    for rank,(name,ps,wp,elo,opp) in enumerate(cand[:4],1):
                        moms.append({"League":code,"Season":season,"Date":row.Date,"HomeTeam":row.HomeTeam,"AwayTeam":row.AwayTeam,
                                     "Rank":rank,"Player":name,"Team":row.HomeTeam if wp==final[0] else row.AwayTeam,
                                     "Position":ps.position,"PreMatchMOMScore":mom_score(ps,wp,elo,opp),"ActualMOM":actual_name,
                                     "Hit":int(actual_name is not None and name.strip().lower()==str(actual_name).strip().lower()),"ActualMOMMethod":method})
                    for pp in extract_players(lineups):
                        team=row.HomeTeam if pp["side"]=="H" else row.AwayTeam
                        players[(team,pp["name"])].update(pp,row.DateParsed.date().isoformat())
                else: group_source_counts[gkey]["SofaScore_missing"]+=1
            else: group_source_counts[gkey]["SofaScore_no_event"]+=1

        # Understat match xG, available for Big 5, is incorporated only after prediction.
        uk=(code,row.DateParsed.date().isoformat(),row.HomeTeam,row.AwayTeam); u=under_idx.get(uk)
        hs,as_,hc,ac=[sf(row.get(k)) for k in ("HS","AS","HC","AC")]
        hst,ast=[sf(row.get(k)) for k in ("HST","AST")]
        h.update(ah,aa,3 if ah>aa else 1 if ah==aa else 0,"H",{"shots":hs,"sot":hst,"corners":hc},u["hxg"],u["axg"]) if u else h.update(ah,aa,3 if ah>aa else 1 if ah==aa else 0,"H",{"shots":hs,"sot":hst,"corners":hc})
        a.update(aa,ah,0 if ah>aa else 1 if ah==aa else 3,"A",{"shots":as_,"sot":ast,"corners":ac},u["axg"],u["hxg"]) if u else a.update(aa,ah,0 if ah>aa else 1 if ah==aa else 3,"A",{"shots":as_,"sot":ast,"corners":ac})
        if u: group_source_counts[gkey]["Understat"]+=1

        expected=sigmoid((h.elo+55-a.elo)/400); actual_h=1 if actual==0 else .5 if actual==1 else 0
        delta=18*(1+math.log1p(max(1,abs(ah-aa))))*(actual_h-expected)
        h.elo+=delta; a.elo-=delta; L.update(ah,aa)
        # Global cross-competition Elo update.
        ge_h=global_elo[row.HomeTeam]; ge_a=global_elo[row.AwayTeam]
        ge_expected=sigmoid((ge_h+45-ge_a)/400)
        ge_delta=14*(1+0.25*math.log1p(max(1,abs(ah-aa))))*(actual_h-ge_expected)
        global_elo[row.HomeTeam]=ge_h+ge_delta; global_elo[row.AwayTeam]=ge_a-ge_delta

        pair=tuple(sorted([row.HomeTeam,row.AwayTeam]))
        # Store H2H outcome from the perspective of the alphabetically first team.
        pair_actual=actual if row.HomeTeam==pair[0] else (2-actual if actual in (0,2) else 1)
        h2h.setdefault((code,pair),[]).append(pair_actual)
        last_dates[row.HomeTeam]=row.DateParsed; last_dates[row.AwayTeam]=row.DateParsed
        hist.append({"x":x,"y":actual,"ml":ml,"market":market[:3],"score":score_p})
        code_counts[code]+=1; group_seen[gkey]+=1; counter+=1; cursor+=1

        if group_seen[gkey]>=group_sizes[gkey]:
            done.add(gkey)
            if len(hist)>=360: blends[code]=optimize_blend(hist)
            if bundle:
                # Store latest validation snapshot; actual fitting remains chronological.
                for name,ll in bundle["validation_logloss"].items(): model_rows.append({"League":code,"Season":season,"Model":name,"ValidationLogLoss":ll})
            coverage.append({"League":code,"Season":season,"Matches":group_sizes[gkey],**dict(group_source_counts[gkey])})

        if counter>=SAVE_EVERY:
            save_state(done,results,scores,moms,model_rows,coverage,histories,bundles,blends,leagues,players,last_dates,h2h,global_elo,cursor,active_seasons); counter=0

    if cursor>=len(matches):
        save_state(done,results,scores,moms,model_rows,coverage,histories,bundles,blends,leagues,players,last_dates,h2h,global_elo,cursor,active_seasons)
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
        rows=[]
        for (lg,sn),g in rdf.groupby(["League","Season"]):
            gy=g.Actual.map({"H":0,"D":1,"A":2}).to_numpy(); gp=g[["HomeProb","DrawProb","AwayProb"]].to_numpy()
            rows.append({"League":lg,"Season":sn,"Matches":len(g),"Accuracy":g.Correct.mean(),"LogLoss":log_loss(gy,gp,labels=[0,1,2]),"MeanConfidence":g.Confidence.mean()})
        pd.DataFrame(rows).to_csv(ROOT/"competition_season_summary.csv",index=False)
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
    cov=under_cov+coverage+FRIENDLY_COVERAGE
    if cov: pd.DataFrame(cov).to_csv(ROOT/"data_coverage.csv",index=False)

    # Detailed player-by-player profile and data-derived playing style.
    profiles=[]
    for (tm,name),ps in players.items():
        if ps.matches>0:
            profiles.append(ps.profile("ALL",tm,name))
    if profiles:
        pdf=pd.DataFrame(profiles).sort_values(["League","Team","Minutes"],ascending=[True,True,False])
        pdf.to_csv(ROOT/"player_profiles.csv",index=False,encoding="utf-8-sig")
        # Compact team-style snapshot built only from pre-match accumulated player history.
        ts=[]
        for code in LEAGUES:
            teams=sorted({tm for (tm,_),ps in players.items() if ps.matches>0})
            for tm in teams:
                vals=team_player_features(code,tm,players)
                ts.append({"League":code,"Team":tm,**{f"PlayerStyle_{k}":v for k,v in zip(PLAYER_FEATURE_BASE,vals)}})
        if ts: pd.DataFrame(ts).to_csv(ROOT/"team_player_style.csv",index=False,encoding="utf-8-sig")

    # ExtraTrees feature importance from latest fitted bundle, when available.
    for code,b in bundles.items():
        m=b.get("models",{}).get("ExtraTrees") if isinstance(b,dict) else None
        if m is not None and hasattr(m,"feature_importances_"):
            fi=pd.DataFrame({"League":code,"Feature":FEATURE_NAMES,"Importance":m.feature_importances_}).sort_values("Importance",ascending=False)
            fi.to_csv(ROOT/"feature_importance.csv",index=False); break

    if cursor>=len(matches):
        # Durable completion marker: scheduled runners exit instead of restarting from scratch.
        Path("BACKTEST_COMPLETE_V5").write_text(
            f"completed_at={datetime.now(timezone.utc).isoformat()}\n"
            f"groups={len(done)}\n"
            f"matches={len(results)}\n", encoding="utf-8"
        )
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