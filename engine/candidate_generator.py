#!/usr/bin/env python3
"""Generate a small, reproducible, materially different soccer candidate set."""
from __future__ import annotations
import json
from pathlib import Path

def generate(evaluation=None,failures=None):
    failures=failures or {}; reasons=failures.get('reason_counts',{}); out=[]
    def add(cid,kind,params,rationale): out.append({'id':cid,'kind':kind,'params':params,'rationale':rationale,'validation':'walk_forward_required'})
    add('v12_stack_alpha_0001','stacking',{'alpha':0.0001},'Less regularization for component stacking.')
    add('v12_stack_alpha_0005','stacking',{'alpha':0.0005},'Baseline regularization candidate.')
    add('v12_stack_alpha_001','stacking',{'alpha':0.001},'More regularization for stability.')
    add('v12_market_gap','market_gap',{'alpha':0.0005},'Focus on disagreement between market and model experts.')
    add('v12_context','context',{'alpha':0.0005,'half_life':180},'Add league/seasonal context with recency weighting.')
    if reasons.get('draw_missed',0)>0:
        add('v12_draw_specialist','draw_specialist',{'alpha':0.0005,'draw_weight':1.15},'Increase draw sensitivity only after historical draw misses are demonstrated.')
    return {'version':'V12-candidate-generator-v2','candidates':out[:6]}

def main():
    ev=json.loads(Path('evaluation.json').read_text(encoding='utf-8')) if Path('evaluation.json').exists() else {}
    fa=json.loads(Path('failure_analysis.json').read_text(encoding='utf-8')) if Path('failure_analysis.json').exists() else {}
    out=generate(ev,fa); Path('candidate_models.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__': main()
