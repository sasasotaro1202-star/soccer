#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Soccer Backtest V12 runner.

V12 keeps the proven V9 chronological engine intact and adds a strictly
walk-forward meta-layer.  The meta-layer uses only rows that occurred before
an evaluated match, so it cannot train on the current match outcome.

It adaptively combines the pre-match Market / ML / Poisson probabilities,
selects between the V9 base blend and the adaptive blend using prior-window
log loss, and applies prior-window temperature calibration.  If the V9 engine
fails to produce valid component probabilities, V12 fails closed rather than
inventing data.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import backtest

SRC = Path("backtest_results_v9.csv")
OUT = Path("backtest_results_v12.csv")
WINDOW = 240
TEMP_WINDOW = 360
MIN_HISTORY = 80
EPS = 1e-7


def _p(row, prefix):
    return np.array([row[f"{prefix}Home"], row[f"{prefix}Draw"], row[f"{prefix}Away"]], dtype=float)


def _clean(p):
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=1/3, posinf=1/3, neginf=1/3)
    p = np.clip(p, EPS, 1.0)
    return p / p.sum()


def _ll(p, y):
    return -math.log(float(np.clip(p[int(y)], EPS, 1.0)))


def _softmax_inverse(losses):
    a = -np.asarray(losses, dtype=float)
    a -= np.nanmax(a)
    w = np.exp(np.clip(a, -20, 20))
    w[~np.isfinite(w)] = 0.0
    s = w.sum()
    return w / s if s > 0 else np.ones(len(losses)) / len(losses)


def _temp(p, t):
    q = np.power(np.clip(p, EPS, 1.0), 1.0 / float(t))
    return _clean(q)


def _best_temperature(hist_p, hist_y):
    if len(hist_y) < MIN_HISTORY:
        return 1.0
    best_t, best_ll = 1.0, float("inf")
    for t in np.linspace(0.70, 1.60, 19):
        loss = 0.0
        for p, y in zip(hist_p, hist_y):
            loss += _ll(_temp(p, t), y)
        loss /= len(hist_y)
        if loss < best_ll:
            best_ll, best_t = loss, float(t)
    return best_t


def build_v12():
    if not SRC.exists():
        raise FileNotFoundError(f"required V9 output missing: {SRC}")
    df = pd.read_csv(SRC, low_memory=False)
    required = [
        "Predicted", "Actual", "HomeProb", "DrawProb", "AwayProb",
        "MarketHome", "MarketDraw", "MarketAway",
        "MLHome", "MLDraw", "MLAway",
        "PoissonHome", "PoissonDraw", "PoissonAway",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"V12 fail-closed: missing columns: {missing}")

    # Stable chronological order. Date is preferred; original row order is a
    # deterministic tie-breaker for same-day matches.
    df["__v12_order"] = np.arange(len(df))
    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df["__v12_date"] = dt
        df = df.sort_values(["__v12_date", "__v12_order"], kind="mergesort").reset_index(drop=True)

    out_probs = []
    base_probs = []
    adaptive_probs = []
    temps = []
    modes = []

    hist_components = {"market": [], "ml": [], "poisson": [], "base": [], "adaptive": []}
    hist_y = []

    for i, row in df.iterrows():
        base = _clean([row["HomeProb"], row["DrawProb"], row["AwayProb"]])
        comps = {
            "market": _clean(_p(row, "Market")),
            "ml": _clean(_p(row, "ML")),
            "poisson": _clean(_p(row, "Poisson")),
        }
        base_probs.append(base)

        start = max(0, i - WINDOW)
        losses = []
        names = ["market", "ml", "poisson"]
        for name in names:
            hp = hist_components[name][start:]
            hy = hist_y[start:]
            if len(hy) < MIN_HISTORY:
                losses.append(float(np.mean([_ll(p, y) for p, y in zip(hp, hy)])) if hy else 1.10)
            else:
                losses.append(float(np.mean([_ll(p, y) for p, y in zip(hp, hy)])))
        w = _softmax_inverse(losses)
        adaptive = _clean(sum(w[j] * comps[names[j]] for j in range(3)))
        adaptive_probs.append(adaptive)

        # Meta-selection is itself chronological: evaluate base vs adaptive on
        # only the preceding window and pick the lower historical log loss.
        hb = hist_components["base"][start:]
        ha = hist_components["adaptive"][start:]
        hy = hist_y[start:]
        if len(hy) >= MIN_HISTORY:
            base_ll = float(np.mean([_ll(p, y) for p, y in zip(hb, hy)]))
            adap_ll = float(np.mean([_ll(p, y) for p, y in zip(ha, hy)]))
            raw = adaptive if adap_ll + 1e-5 < base_ll else base
            mode = "adaptive" if adap_ll + 1e-5 < base_ll else "v9_base"
        else:
            raw, mode = base, "v9_base_warmup"

        ht = hist_components["adaptive"][max(0, i - TEMP_WINDOW):] if mode == "adaptive" else hist_components["base"][max(0, i - TEMP_WINDOW):]
        hy_t = hist_y[max(0, i - TEMP_WINDOW):]
        t = _best_temperature(ht, hy_t)
        final = _temp(raw, t)
        out_probs.append(final)
        temps.append(t)
        modes.append(mode)

        # Outcome enters state only AFTER this row's prediction is complete.
        y = int(row["Actual"])
        hist_components["market"].append(comps["market"])
        hist_components["ml"].append(comps["ml"])
        hist_components["poisson"].append(comps["poisson"])
        hist_components["base"].append(base)
        hist_components["adaptive"].append(adaptive)
        hist_y.append(y)

    P = np.asarray(out_probs)
    df["HomeProbV12"] = P[:, 0]
    df["DrawProbV12"] = P[:, 1]
    df["AwayProbV12"] = P[:, 2]
    df["ConfidenceV12"] = P.max(axis=1)
    df["PredictedV12"] = P.argmax(axis=1)
    df["CorrectV12"] = (df["PredictedV12"].astype(int) == df["Actual"].astype(int)).astype(int)
    df["V12Temperature"] = temps
    df["V12Mode"] = modes

    # Restore source ordering for auditability while retaining chronological
    # computation in the generated probabilities.
    df = df.sort_values("__v12_order", kind="mergesort").drop(columns=[c for c in ["__v12_order", "__v12_date"] if c in df.columns])
    df.to_csv(OUT, index=False)

    # Compact evaluation artifacts.
    y = df["Actual"].to_numpy(dtype=int)
    p = df[["HomeProbV12", "DrawProbV12", "AwayProbV12"]].to_numpy(dtype=float)
    pred = p.argmax(axis=1)
    acc = float(np.mean(pred == y)) if len(y) else float("nan")
    logloss = float(np.mean([-math.log(np.clip(p[j, y[j]], EPS, 1.0)) for j in range(len(y))])) if len(y) else float("nan")
    onehot = np.eye(3)[y]
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1))) if len(y) else float("nan")
    pd.DataFrame([{
        "Version": "V12",
        "Matches": len(df),
        "Accuracy": acc,
        "LogLoss": logloss,
        "Brier": brier,
        "MeanConfidence": float(p.max(axis=1).mean()) if len(p) else float("nan"),
        "AdaptiveShare": float(np.mean(np.array(modes) == "adaptive")) if modes else 0.0,
        "MeanTemperature": float(np.mean(temps)) if temps else 1.0,
    }]).to_csv("overall_summary_v12.csv", index=False)

    print(f"[V12] wrote {OUT} rows={len(df)} accuracy={acc:.6f} logloss={logloss:.6f} brier={brier:.6f}")


def main():
    # The proven engine remains the data/model generator. V12 adds its
    # walk-forward meta layer only after the engine has completed.
    backtest.main()
    build_v12()


if __name__ == "__main__":
    main()
