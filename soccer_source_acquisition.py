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
TIMEOUT=int(os.getenv('ACQ_TIMEOUT','30'))
RETRIES=int(os.getenv('ACQ_RETRIES','6'))
S=requests.Session()
S.headers.update({'User-Agent':'Mozilla/5.0 SoccerSourceAcquisition/1.0','Accept':'application/json,text/plain,*/*'})

LEAGUES=['E0','D1','I1','SP1','F1','N1']
SOFA='https://api.sofascore.com/api/v1'


def get(url, *, binary=False):
    last=''
    for attempt in range(RETRIES):
        try:
            r=S.get(url,timeout=TIMEOUT)
            if r.status_code==200:
                return r.content if binary else r.json()
            last=f'HTTP {r.status_code}'
            if r.status_code not in (408,425,429) and r.status_code<500:
                break
        except Exception as e:
            last=repr(e)
        time.sleep(min(2**attempt,20))
    raise RuntimeError(f'{url}: {last}')


def season_folder(y): return f'{str(y)[-2:]}{str(y+1)[-2:]}'

health={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sources':{}}

# Football-Data is currently publishing historical CSV seasons and current fixtures.
for code in LEAGUES:
    ok=0
    for y in (2024,2025):
        url=f'https://www.football-data.co.uk/mmz4281/{season_folder(y)}/{code}.csv'
        try:
            raw=get(url,binary=True)
            if len(raw)>500:
                (CACHE/'football_data'/f'{code}_{y}.csv').write_bytes(raw); ok+=1
        except Exception as e:
            health['sources'][f'football-data:{code}:{y}']=str(e)
    health['sources'][f'football-data:{code}']= {'nonempty_recent_seasons':ok}

# Understat: keep raw JSON so the backtest can consume it without another network dependency.
for league in ['epl','bundesliga','serie_a','la_liga','ligue_1','eredivisie']:
    ok=0
    for y in (2024,2025):
        url=f'https://understat.com/getLeagueData/{league}/{y}'
        try:
            data=get(url)
            if isinstance(data,dict) and data:
                (CACHE/'understat'/f'{league}_{y}.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8'); ok+=1
        except Exception as e:
            health['sources'][f'understat:{league}:{y}']=str(e)
    health['sources'][f'understat:{league}']={'nonempty_recent_seasons':ok}

# SofaScore: use the documented API host, not the web host. Probe current scheduled events and one tournament season list.
try:
    data=get(f'{SOFA}/sport/football/scheduled-events/{time.strftime("%Y-%m-%d",time.gmtime())}')
    events=data.get('events',[]) if isinstance(data,dict) else []
    (CACHE/'sofascore'/'scheduled_events_latest.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    health['sources']['sofascore_scheduled_events']={'events':len(events),'status':'ok'}
except Exception as e:
    health['sources']['sofascore_scheduled_events']=str(e)

Path('data_source_health.json').write_text(json.dumps(health,ensure_ascii=False,indent=2),encoding='utf-8')

fd=sum(v.get('nonempty_recent_seasons',0) for k,v in health['sources'].items() if k.startswith('football-data:') and isinstance(v,dict))
us=sum(v.get('nonempty_recent_seasons',0) for k,v in health['sources'].items() if k.startswith('understat:') and isinstance(v,dict))
ss=health['sources'].get('sofascore_scheduled_events',{})
sofa_n=ss.get('events',0) if isinstance(ss,dict) else 0
print(f'[ACQ] Football-Data nonempty={fd}; Understat nonempty={us}; SofaScore events={sofa_n}')
if fd==0: raise SystemExit('[ACQ] FAIL: Football-Data acquisition returned zero usable files')
if us==0: raise SystemExit('[ACQ] FAIL: Understat acquisition returned zero usable seasons')
if sofa_n==0: raise SystemExit('[ACQ] FAIL: SofaScore acquisition returned zero scheduled events')
print('[ACQ] PASS: all critical soccer sources returned nonzero data')
