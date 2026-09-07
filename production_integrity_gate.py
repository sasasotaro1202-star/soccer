#!/usr/bin/env python3
"""Fail-closed production output validation and provenance manifest."""
from __future__ import annotations
import csv,hashlib,json,math,os,platform,socket,subprocess
from datetime import datetime,timezone
from pathlib import Path
MANIFEST=Path('production_run_manifest.json')
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def git_value(args):
 try:return subprocess.check_output(['git',*args],text=True,stderr=subprocess.DEVNULL).strip()
 except Exception:return 'unknown'
def finite(v):
 try:return math.isfinite(float(v))
 except Exception:return False
def main():
 targets=[Path('backtest_results_v12.csv'),Path('overall_summary_v12.csv')]
 for p in targets:
  if not p.exists() or p.stat().st_size==0: raise SystemExit(f'INTEGRITY_FAIL: missing/empty required output: {p}')
 dups=set(); total_rows=0; probability_columns=[]; checks={'required_outputs':True,'probabilities_valid':True,'duplicate_rows':True}
 for path in targets:
  with path.open('r',encoding='utf-8',errors='replace',newline='') as f:
   reader=csv.DictReader(f); fields=reader.fieldnames or []; rows=list(reader)
  if not fields or not rows: raise SystemExit(f'INTEGRITY_FAIL: empty CSV: {path}')
  total_rows+=len(rows)
  for c in [c for c in fields if c in {'HomeProbV12','DrawProbV12','AwayProbV12'} or c.lower().endswith('prob')]:
   probability_columns.append(f'{path}:{c}')
   for i,row in enumerate(rows,2):
    if row.get(c,'')=='' or not finite(row[c]): raise SystemExit(f'INTEGRITY_FAIL: invalid probability {path}:{c} row={i}')
  seen=set()
  for row in rows:
   key=tuple(row.get(c,'') for c in fields)
   if key in seen: raise SystemExit(f'INTEGRITY_FAIL: duplicate result row in {path}')
   seen.add(key)
  if path.name=='backtest_results_v12.csv':
   required={'HomeProbV12','DrawProbV12','AwayProbV12','PredictedV12','Actual'}
   if not required.issubset(fields): raise SystemExit('INTEGRITY_FAIL: required V12 prediction columns missing')
   for i,row in enumerate(rows,2):
    vals=[float(row[c]) for c in ('HomeProbV12','DrawProbV12','AwayProbV12')]
    if not math.isclose(sum(vals),1.0,abs_tol=1e-6): raise SystemExit(f'INTEGRITY_FAIL: probability sum != 1 row={i}')
 if total_rows<102: raise SystemExit(f'INTEGRITY_FAIL: insufficient output rows across required CSVs: {total_rows}')
 now=datetime.now(timezone.utc).isoformat()
 manifest={'schema':'production-integrity-v1','status':'PASS','validation_timestamp_utc':now,'prediction_timestamp_utc':os.environ.get('PREDICTION_TIMESTAMP_UTC','unknown'),'repository':os.environ.get('GITHUB_REPOSITORY','unknown'),'run_id':os.environ.get('GITHUB_RUN_ID','unknown'),'run_attempt':os.environ.get('GITHUB_RUN_ATTEMPT','unknown'),'commit_sha':os.environ.get('GITHUB_SHA',git_value(['rev-parse','HEAD'])),'branch':os.environ.get('GITHUB_REF_NAME',git_value(['branch','--show-current'])),'runner_name':os.environ.get('RUNNER_NAME',socket.gethostname()),'runner_os':platform.platform(),'python_version':platform.python_version(),'requirements_sha256':sha256(Path('requirements.txt')) if Path('requirements.txt').exists() else 'missing','checks':checks,'total_result_rows':total_rows,'probability_columns':probability_columns,'result_files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in targets]}
 MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
