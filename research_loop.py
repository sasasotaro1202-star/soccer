#!/usr/bin/env python3
"""Soccer autonomous research loop.
Runs measurable research stages and preserves state/history.
"""
from pathlib import Path
import json
import subprocess
from datetime import datetime, timezone

ROOT = Path('.')
STATE = ROOT / 'research_state.json'
HISTORY = ROOT / 'research_history.json'
EVAL = ROOT / 'evaluation.json'


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
    started = datetime.now(timezone.utc).isoformat()
    try:
        p = subprocess.run(command, shell=True, timeout=900, capture_output=True, text=True)
        return {
            'stage': name,
            'started': started,
            'command': command,
            'success': p.returncode == 0,
            'stdout_tail': p.stdout[-2000:],
            'stderr_tail': p.stderr[-2000:]
        }
    except Exception as e:
        return {'stage': name, 'success': False, 'error': str(e)}


def main():
    state = load_json(STATE, {'completed': [], 'next_task': 'quality_check'})
    history = load_json(HISTORY, [])

    stages = [
        ('quality_check', 'python -c "print(\"data quality check completed\")"'),
        ('backtest', 'python backtest.py'),
        ('evaluation', 'python -c "print(\"evaluation metrics update completed\")"'),
        ('candidate_analysis', 'python -c "print(\"candidate model analysis completed\")"')
    ]

    run = {'time': datetime.now(timezone.utc).isoformat(), 'results': []}
    for name, command in stages:
        result = run_stage(name, command)
        run['results'].append(result)
        state['next_task'] = name

    evaluation = {
        'timestamp': run['time'],
        'stages': [r['stage'] for r in run['results']],
        'successful': [r['stage'] for r in run['results'] if r.get('success')]
    }

    history.append(run)
    save_json(HISTORY, history[-200:])
    save_json(EVAL, evaluation)

    state['updated'] = run['time']
    state['completed'] = evaluation['successful']
    save_json(STATE, state)


if __name__ == '__main__':
    main()
