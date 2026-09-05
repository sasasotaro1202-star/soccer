#!/usr/bin/env python3
"""Soccer V12 autonomous research runner.

Evaluation -> failure analysis -> condition analysis -> candidate generation
-> real walk-forward OOS -> fail-closed comparison -> registry.
"""
from __future__ import annotations
import json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

BUDGET=float(os.environ.get('MAX_RUNTIME_SECONDS','3300'))
STATE=Path('research_state.json')
HISTORY=Path('research_history.json')
MANIFEST=Path('research_manifest.json')


def load(p,d):
    try:return json.loads(p.read_text(encoding='utf-8')) if p.exists() else d
    except Exception:return d


def save(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def run_stage(name,cmd,timeout):
    try:
        p=subprocess.run(cmd,shell=True,text=True,capture_output=True,timeout=max(1,int(timeout)))
        return {'stage':name,'success':p.returncode==0,'returncode':p.returncode,'stdout_tail':p.stdout[-2500:],'stderr_tail':p.stderr[-2500:]}
    except Exception as e:
        return {'stage':name,'success':False,'error':str(e)}


def main():
    started=time.monotonic()
    now=datetime.now(timezone.utc).isoformat()
    state=load(STATE,{'version':'V12-auto-research-v4','iteration':0,'best_model':'V12'})
    history=load(HISTORY,[])
    trigger_sha=os.environ.get('RESEARCH_TRIGGER_SHA','')
    trigger_run=os.environ.get('RESEARCH_TRIGGER_RUN_ID','')
    runner=os.environ.get('RESEARCH_RUNNER',os.environ.get('RUNNER_OS','unknown'))
    stages=[]
    manifest={'version':'soccer-research-manifest-v1','started':now,'trigger_sha':trigger_sha,'trigger_run_id':trigger_run,'runner':runner,'policy':'prior_rows_only'}
    save(MANIFEST,manifest)

    commands=[
        ('evaluation','python -m engine.soccer_evaluator'),
        ('failure_analysis','python -m engine.failure_analysis'),
        ('soccer_condition_analysis','python -m engine.condition_analysis_soccer'),
        ('candidate_generation','python -m engine.candidate_generator'),
        ('candidate_oos_validation','python -m engine.candidate_oos_validator'),
    ]
    for name,cmd in commands:
        remaining=BUDGET-(time.monotonic()-started)-60
        if remaining<=0: break
        rec=run_stage(name,cmd,min(1200,remaining)); stages.append(rec)
        if not rec.get('success'): break

    oos_ok=any(s['stage']=='candidate_oos_validation' and s.get('success') for s in stages) and Path('candidate_evaluations.json').exists()
    if oos_ok:
        remaining=BUDGET-(time.monotonic()-started)-30
        if remaining>10:
            cmp_timeout=min(180,max(10,int(remaining-10)))
            stages.append(run_stage('model_compare','python -m engine.model_compare',cmp_timeout))
            remaining2=BUDGET-(time.monotonic()-started)-5
            if remaining2>5 and Path('model_comparison_v12.json').exists():
                stages.append(run_stage('model_registry','python -m engine.model_registry',min(60,int(remaining2))))

    state['version']='V12-auto-research-v4'
    state['iteration']=int(state.get('iteration',0))+1
    state['updated']=datetime.now(timezone.utc).isoformat()
    state['next_task']='next_research' if oos_ok else 'repair_or_retry'
    state['candidate_oos_validated']=oos_ok
    state['last_trigger_sha']=trigger_sha
    state['last_trigger_run_id']=trigger_run
    state['last_runner']=runner
    rec={'time':state['updated'],'runner':'v12_auto_research_v4','trigger_sha':trigger_sha,'trigger_run_id':trigger_run,'research_runner':runner,'stages':stages,'candidate_oos_evaluations_present':Path('candidate_evaluations.json').exists(),'promotion_evidence_present':Path('model_comparison_v12.json').exists(),'leak_policy':'prior_rows_only'}
    history.append(rec)
    save(HISTORY,history[-500:])
    save(STATE,state)
    manifest['completed']=state['updated']; manifest['success']=all(x.get('success',False) for x in stages) and oos_ok; manifest['candidate_oos_validated']=oos_ok
    save(MANIFEST,manifest)
    print(json.dumps(rec,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
