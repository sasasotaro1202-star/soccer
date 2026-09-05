#!/usr/bin/env python3
"""Soccer V12 autonomous research runner.

Baseball's repository separates evaluation, condition/failure analysis and
research workflows. This football version keeps that modular pattern while
making promotion fail-closed and requiring OOS candidate evidence.
"""
from __future__ import annotations
import json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

BUDGET=float(os.environ.get('MAX_RUNTIME_SECONDS','3300'))
STATE=Path('research_state.json'); HISTORY=Path('research_history.json')

def load(p,d):
    try:return json.loads(p.read_text(encoding='utf-8')) if p.exists() else d
    except Exception:return d

def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    started=time.monotonic(); state=load(STATE,{'version':'V12-auto-research-v1','iteration':0,'best_model':'V12'})
    history=load(HISTORY,[]); stages=[]
    commands=[
      ('evaluation','python -m engine.soccer_evaluator'),
      ('failure_analysis','python -m engine.failure_analysis'),
      ('candidate_generation','python -m engine.candidate_generator'),
    ]
    for name,cmd in commands:
        remaining=BUDGET-(time.monotonic()-started)-30
        if remaining<=0: break
        try:
            p=subprocess.run(cmd,shell=True,text=True,capture_output=True,timeout=min(900,int(remaining)))
            rec={'stage':name,'success':p.returncode==0,'returncode':p.returncode,'stdout_tail':p.stdout[-2500:],'stderr_tail':p.stderr[-2500:]}
        except Exception as e: rec={'stage':name,'success':False,'error':str(e)}
        stages.append(rec)
        if not rec.get('success'): break
    # Promotion is deliberately fail-closed. Candidate scores must be generated
    # by a separate chronological/OOS validator before comparison can adopt one.
    if Path('candidate_evaluations.json').exists():
        try:
            subprocess.run('python -m engine.model_compare',shell=True,check=False,timeout=max(1,int(BUDGET-(time.monotonic()-started)-10)))
            subprocess.run('python -m engine.model_registry',shell=True,check=False,timeout=60)
        except Exception: pass
    state['iteration']=int(state.get('iteration',0))+1; state['updated']=datetime.now(timezone.utc).isoformat()
    state['next_task']='candidate_oos_validation' if not Path('candidate_evaluations.json').exists() else 'next_research'
    rec={'time':state['updated'],'runner':'v12_auto_research','stages':stages,'oos_candidate_evaluations_present':Path('candidate_evaluations.json').exists()}
    history.append(rec); save(HISTORY,history[-500:]); save(STATE,state)
    print(json.dumps(rec,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
