#!/usr/bin/env python3
"""Soccer V12 evaluation engine.
Calculates prediction quality metrics without changing the prediction model.
"""
from pathlib import Path
import json
import math


OUTPUT = Path("evaluation.json")


def accuracy(predictions):
    if not predictions:
        return 0.0
    return sum(p.get("prediction") == p.get("result") for p in predictions) / len(predictions)


def mae(values):
    if not values:
        return 0.0
    return sum(abs(v.get("pred", 0) - v.get("actual", 0)) for v in values) / len(values)


def evaluate(data):
    return {
        "sample_size": len(data),
        "accuracy": accuracy(data),
        "score_mae": mae(data),
    }


def main():
    source = Path("predictions.json")
    data = []
    if source.exists():
        data = json.loads(source.read_text(encoding="utf-8"))
    result = evaluate(data)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
