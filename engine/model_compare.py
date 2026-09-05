#!/usr/bin/env python3
"""Fail-closed promotion gate using real candidate OOS metrics."""
from __future__ import annotations
import json
from pathlib import Path

def compare(candidates):
    rows=[]
    for c in candidates:
        cand=c.get('candidate') or {}; base=c.get('baseline') or {}
        acc,ll,br=cand.get('accuracy'),cand.get('logloss'),cand.get('brier')
        bacc,bll,bbr=base.get('accuracy'),base.get('logloss'),base.get('brier')
        valid=c.get('status')=='validated' and c.get('leak_policy','prior_rows_only')=='prior_rows_only' and None not in (acc,ll,br,bacc,bll,bbr)
        # Require no degradation in any primary metric and a strict gain in at least one.
        improved=bool(valid and acc>=bacc and ll<=bll and br<=bbr and (acc>bacc or ll<bll or br<bbr))
        rows.append({'id':c.get('id'),'status':c.get('status'),'oos_rows':c.get('oos_rows'),'candidate':cand,'baseline':base,'improved':improved,'adopt':improved,'reason':'OOS improvement across primary metrics' if improved else 'No demonstrated OOS improvement','params':c.get('params',{})})
    valid=[r for r in rows if r['adopt']]
    best=min(valid,key=lambda r:(r['candidate']['logloss'],-r['candidate']['accuracy'],r['candidate']['brier'])) if valid else None
    return {'version':'V12-model-compare-v2','candidates':rows,'recommended':best['id'] if best else None,'fail_closed':True,'policy':'accuracy_non_decreasing_logloss_non_increasing_brier_non_increasing'}

def main():
    p=Path('candidate_evaluations.json')
    data=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    out=compare(data.get('candidates',[])); Path('model_comparison_v12.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__': main()
