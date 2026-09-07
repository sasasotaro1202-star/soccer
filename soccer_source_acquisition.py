#!/usr/bin/env python3
from __future__ import annotations
import json, os, time
from pathlib import Path
import requests

ROOT=Path('.')
CACHE=ROOT/'cache'
(CACHE/'football_data').mkdir(parents=True,exist_ok=True)
(CACHE/'understat').mkdir(parents=True,exist_ok=True)
(CACHE/'sofascore').mkdir(parents=True,exist_ok=True)
TIMEOUT=int(os.getenv('ACQ_TIMEOUT','20'))
RETRIES=int(os.getenv('ACQ_RETRIES','5'))
CACHE_TTL_SEC=int(os.getenv('ACQ_CACHE_TTL_SEC','43200'))
START_SEASON=int(os.getenv('ACQ_START_SEASON','2010'))
UNDERSTAT_START_SEASON=max(2014, START_SEASON)
END_SEASON=int(os.getenv('ACQ_END_SEASON',str(time.gmtime().tm_year)))
S=requests.Session()
S.headers.update({'User-Agent':'Mozilla/5.0 SoccerSourceAcquisition/1.3','Accept':'application/json,text/plain,*/*'})
LEAGUES=['E0','D1','I1','SP1','F1','N1']
UNDERSTAT=['epl','bundesliga','serie_a','la_liga','ligue_1','eredivisie']
SOFA='https://api.sofascore.com/api/v1'

def fresh(path: Path) -> bool:
    try: return path.exists() and path.stat().st_size > 500
    except Exception: return False

def get(url, *, binary=False, understat=False):
    last=''; headers={}
    if understat:
        headers.update({'Referer':'https://understat.com/','Origin':'https://understat.com','X-Requested-With':'XMLHttpRequest','Accept':'application/json, text/plain, */*'})
    for attempt in range(RETRIES):
        try:
            r=S.get(url,headers=headers,timeout=TIMEOUT)
            if r.status_code==200: return r.content if binary else r.json()
            last=f'HTTP {r.status_code}'
            if r.status_code not in (408,425,429) and r.status_code<500: break
        except Exception as e: last=repr(e)
        time.sleep(min(1.5*(attempt+1),8))
    raise RuntimeError(f'{url}: {last}')

def season_folder(y): return f'{str(y)[-2:]}{str(y+1)[-2:]}'
health={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sources':{},'cache_ttl_sec':CACHE_TTL_SEC,'scope':{'start_season':START_SEASON,'end_season':END_SEASON,'understat_start_season':UNDERSTAT_START_SEASON}}

for code in LEAGUES:
    ok=0; cached=0; failed=[]
    for y in range(START_SEASON,END_SEASON+1):
        out=CACHE/'football_data'/f'{code}_{y}.csv'
        if fresh(out): cached += 1; ok += 1; continue
        url=f'https://www.football-data.co.uk/mmz4281/{season_folder(y)}/{code}.csv'
        try:
            raw=get(url,binary=True)
            if len(raw)>500: out.write_bytes(raw); ok+=1
            else: failed.append(y)
        except Exception as e:
            failed.append(y); health['sources'][f'football-data:{code}:{y}']=str(e)
    health['sources'][f'football-data:{code}']={'usable_seasons':ok,'cache_hits':cached,'failed_seasons':failed}

for league in UNDERSTAT:
    ok=0; cached=0; failed=[]
    for y in range(UNDERSTAT_START_SEASON,END_SEASON+1):
        out=CACHE/'understat'/f'{league}_{y}.json'
        if fresh(out): cached += 1; ok += 1; continue
        url=f'https://understat.com/getLeagueData/{league}/{y}'
        try:
            data=get(url,understat=True)
            if isinstance(data,dict) and data:
                out.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8'); ok+=1
            else: failed.append(y)
        except Exception as e:
            failed.append(y); health['sources'][f'understat:{league}:{y}']=str(e)
    health['sources'][f'understat:{league}']={'usable_seasons':ok,'cache_hits':cached,'failed_seasons':failed}

try:
    dates=[time.strftime('%Y-%m-%d',time.gmtime(t)) for t in (time.time(), time.time()+86400)]
    all_events=[]; seen=set()
    for day in dates:
        data=get(f'{SOFA}/sport/football/scheduled-events/{day}')
        events=data.get('events',[]) if isinstance(data,dict) else []
        for ev in events:
            eid=ev.get('id') if isinstance(ev,dict) else None
            if eid not in seen: seen.add(eid); all_events.append(ev)
    out=CACHE/'sofascore'/'scheduled_events_latest.json'
    out.write_text(json.dumps({'events':all_events,'probe_dates':dates},ensure_ascii=False),encoding='utf-8')
    health['sources']['sofascore_scheduled_events']={'events':len(all_events),'probe_dates':dates,'status':'ok' if all_events else 'empty'}
except Exception as e:
    health['sources']['sofascore_scheduled_events']=str(e)

Path('data_source_health.json').write_text(json.dumps(health,ensure_ascii=False,indent=2),encoding='utf-8')
fd=sum(v.get('usable_seasons',0) for k,v in health['sources'].items() if k.startswith('football-data:') and isinstance(v,dict))
us=sum(v.get('usable_seasons',0) for k,v in health['sources'].items() if k.startswith('understat:') and isinstance(v,dict))
fd_expected=len(LEAGUES)*(END_SEASON-START_SEASON+1)
us_expected=len(UNDERSTAT)*(END_SEASON-UNDERSTAT_START_SEASON+1)
ss=health['sources'].get('sofascore_scheduled_events',{})
sofa_n=ss.get('events',0) if isinstance(ss,dict) else 0
print(f'[ACQ] Football-Data usable={fd}/{fd_expected}; Understat usable={us}/{us_expected}; SofaScore events={sofa_n}')
if fd==0: raise SystemExit('[ACQ] FAIL: Football-Data acquisition returned zero usable files')
if us==0: raise SystemExit('[ACQ] FAIL: Understat acquisition returned zero usable seasons')
if sofa_n==0: raise SystemExit('[ACQ] FAIL: SofaScore acquisition returned zero scheduled events')
print('[ACQ] PASS: full-scope soccer source acquisition complete')
