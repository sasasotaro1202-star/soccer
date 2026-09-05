#!/usr/bin/env python3
"""Soccer autonomous research loop foundation.

Runs repeatable research stages while keeping a history of experiments.
Designed for GitHub Actions execution.
"""
from pathlib import Path
import json
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path('.')
STATE = ROOT / 'research_state.json'
HISTORY = ROOT / 'research_history.json'


def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            pass
    return default


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def run_stage(name, command):
    result = {
        'stage': name,
        'started': datetime.now(timezone.utc).isoformat(),
        'command': command,
    }
    try:
        p = subprocess.run(command, shell=True, timeout=300, capture_output=True, text=True)
        result['returncode'] = p.returncode
        result['success'] = p.returncode == 0
        result['stdout_tail'] = p.stdout[-1000:]
        result['stderr_tail'] = p.stderr[-1000:]
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
    return result


def main():
    state = load_json(STATE, {'completed': [], 'updated': None})
    history = load_json(HISTORY, [])

    stages = [
        ('quality_check', 'python -c "print(\\"data quality stage\\")"'),
        ('backtest', 'python backtest.py'),
    ]

    run = {'time': datetime.now(timezone.utc).isoformat(), 'results': []}
    for name, command in stages:
        run['results'].append(run_stage(name, command))

    history.append(run)
    save_json(HISTORY, history[-100:])
    state['updated'] = run['time']
    state['completed'] = [r['stage'] for r in run['results'] if r.get('success')]
    save_json(STATE, state)


if __name__ == '__main__':
    main()
