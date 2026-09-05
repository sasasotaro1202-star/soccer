#!/usr/bin/env python3
"""Leak-safe, competition-aware evaluation utilities for Soccer V12.

The evaluator only consumes already-produced predictions/results. It never
trains a model and never uses a current match outcome to alter that match's
prediction. It is intentionally compatible with both V9/V12 CSV outputs.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

EPS = 1e-15
ROOT = Path(".")
DEFAULT_OUTPUT = ROOT / "evaluation.json"


def _f(row: dict[str, Any], key: str, default: float | None = None) -> float | None:
    try:
        v = row.get(key, default)
        if v in (None, ""):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _actual(row: dict[str, Any]) -> str | None:
    a = row.get("Actual") or row.get("actual") or row.get("Result") or row.get("result")
    if a in ("H", "D", "A"):
        return a
    h, a2 = _f(row, "FTHG"), _f(row, "FTAG")
    if h is not None and a2 is not None:
        return "H" if h > a2 else "A" if h < a2 else "D"
    return None


def _pred(row: dict[str, Any]) -> str | None:
    p = row.get("PredictedV12") or row.get("Predicted") or row.get("prediction")
    return str(p) if p in ("H", "D", "A") else None


def _probs(row: dict[str, Any]) -> list[float] | None:
    keys = [
        ("HomeProbV12", "DrawProbV12", "AwayProbV12"),
        ("HomeProb", "DrawProb", "AwayProb"),
    ]
    for trio in keys:
        vals = [_f(row, k) for k in trio]
        if all(v is not None and math.isfinite(v) for v in vals):
            s = sum(vals)
            if s > 0:
                vals = [max(EPS, v / s) for v in vals]
                return vals
    return None


def _brier(rows: list[dict[str, Any]]) -> float | None:
    vals = []
    for r in rows:
        p = _probs(r); a = _actual(r)
        if not p or a not in ("H", "D", "A"):
            continue
        y = [float(a == x) for x in ("H", "D", "A")]
        vals.append(sum((p[i] - y[i]) ** 2 for i in range(3)))
    return sum(vals) / len(vals) if vals else None


def _logloss(rows: list[dict[str, Any]]) -> float | None:
    vals = []
    for r in rows:
        p = _probs(r); a = _actual(r)
        if not p or a not in ("H", "D", "A"):
            continue
        vals.append(-math.log(max(EPS, p[("H", "D", "A").index(a)])))
    return sum(vals) / len(vals) if vals else None


def _calibration(rows: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    bucket = defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        p = _probs(r); a = _actual(r)
        if not p or a not in ("H", "D", "A"):
            continue
        c = max(p); ok = int(_pred(r) == a)
        b = min(bins - 1, int(c * bins))
        bucket[b][0] += 1; bucket[b][1] += c; bucket[b][2] += ok
    out = []
    for b in sorted(bucket):
        n, ps, ys = bucket[b]
        out.append({"bin": b, "n": n, "mean_confidence": ps / n, "accuracy": ys / n})
    return {"bins": out}


def _score_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []; exact = 0; n = 0
    for r in rows:
        ph, pa = _f(r, "PredHomeGoals"), _f(r, "PredAwayGoals")
        ah, aa = _f(r, "FTHG"), _f(r, "FTAG")
        if ph is None or pa is None or ah is None or aa is None:
            continue
        n += 1
        errors.append((abs(ph-ah)+abs(pa-aa))/2.0)
        exact += int(round(ph) == round(ah) and round(pa) == round(aa))
    return {"sample_size": n, "score_mae": sum(errors)/n if n else None, "exact_score_accuracy": exact/n if n else None}


def _group_accuracy(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    groups = defaultdict(list)
    for r in rows:
        if r.get(key) not in (None, ""):
            groups[str(r[key])].append(r)
    out = {}
    for g, rs in groups.items():
        valid = [( _pred(r), _actual(r)) for r in rs]
        valid = [(p,a) for p,a in valid if p and a]
        out[g] = {"matches": len(valid), "accuracy": sum(p==a for p,a in valid)/len(valid) if valid else None,
                  "logloss": _logloss(rs), "brier": _brier(rs)}
    return out


def evaluate(data: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [(r, _pred(r), _actual(r)) for r in data]
    valid = [(r,p,a) for r,p,a in valid if p and a]
    accuracy = sum(p == a for _,p,a in valid) / len(valid) if valid else None
    class_metrics = {}
    for c in ("H", "D", "A"):
        rs = [(p,a) for _,p,a in valid if a == c]
        class_metrics[c] = {"actual_n": len(rs), "recall": sum(p==a for p,a in rs)/len(rs) if rs else None}
    result = {
        "version": "V12-evaluator-v2",
        "sample_size": len(data),
        "valid_1x2": len(valid),
        "accuracy": accuracy,
        "class_metrics": class_metrics,
        "logloss": _logloss(data),
        "brier": _brier(data),
        "calibration": _calibration(data),
        "score": _score_metrics(data),
        "by_league": _group_accuracy(data, "League"),
        "by_season": _group_accuracy(data, "Season"),
    }
    return result


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    candidates = [ROOT / "backtest_results_v12.csv", ROOT / "backtest_results_v9.csv", ROOT / "backtest_results.csv"]
    source = next((p for p in candidates if p.exists()), None)
    data = load_csv(source) if source else []
    result = evaluate(data)
    DEFAULT_OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
