#!/usr/bin/env python3
"""Persistent registry for Soccer V12 research candidates and promotions."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

PATH=Path("model_registry.json")

def load():
    if PATH.exists():
        try:return json.loads(PATH.read_text(encoding="utf-8"))
        except Exception:pass
    return {"version":1,"current":"V12","history":[]}

def record(comparison):
    state=load(); now=datetime.now(timezone.utc).isoformat(); rec={"time":now,"recommended":comparison.get("recommended"),"candidates":comparison.get("candidates",[])}
    state.setdefault("history",[]).append(rec); state["history"]=state["history"][-500:]
    if comparison.get("recommended"): state["current"]=comparison["recommended"]
    PATH.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8"); return state

if __name__=="__main__":
    p=Path("model_comparison_v12.json")
    print(json.dumps(record(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}),ensure_ascii=False,indent=2))
