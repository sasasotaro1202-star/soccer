#!/usr/bin/env python3
"""Fail-closed production output validation and provenance manifest.

Stdlib-only. Intended to run after a production backtest and before any
verified state is persisted.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, platform, socket, subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(".")
MANIFEST = Path("production_run_manifest.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def main() -> None:
    targets = [Path("backtest_results_v12.csv"), Path("overall_summary_v12.csv")]
    for p in targets:
        if not p.exists() or p.stat().st_size == 0:
            raise SystemExit(f"INTEGRITY_FAIL: missing/empty required output: {p}")

    dups = set()
    total_rows = 0
    probability_columns = []
    checks = {"required_outputs": True, "probabilities_valid": True, "duplicate_rows": True}
    for path in targets:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            rows = list(reader)
        if not fields or not rows:
            raise SystemExit(f"INTEGRITY_FAIL: empty CSV: {path}")
        total_rows += len(rows)
        prob = [c for c in fields if c in {"HomeProbV12", "DrawProbV12", "AwayProbV12"} or c.lower().endswith("prob")]
        for c in prob:
            probability_columns.append(f"{path}:{c}")
            for i, row in enumerate(rows, 2):
                if row.get(c, "") == "" or not finite(row[c]):
                    checks["probabilities_valid"] = False
                    raise SystemExit(f"INTEGRITY_FAIL: invalid probability {path}:{c} row={i}")
        seen = set()
        for row in rows:
            key = tuple(row.get(c, "") for c in fields)
            if key in seen:
                checks["duplicate_rows"] = False
                raise SystemExit(f"INTEGRITY_FAIL: duplicate result row in {path}")
            seen.add(key)
        if path.name == "backtest_results_v12.csv":
            if not {"HomeProbV12", "DrawProbV12", "AwayProbV12", "PredictedV12", "Actual"}.issubset(fields):
                raise SystemExit("INTEGRITY_FAIL: required V12 prediction columns missing")
            for i, row in enumerate(rows, 2):
                vals = [float(row[c]) for c in ("HomeProbV12", "DrawProbV12", "AwayProbV12")]
                if not math.isclose(sum(vals), 1.0, abs_tol=1e-6):
                    raise SystemExit(f"INTEGRITY_FAIL: probability sum != 1 row={i}")

    if total_rows < 102:
        raise SystemExit(f"INTEGRITY_FAIL: insufficient output rows across required CSVs: {total_rows}")

    manifest = {
        "schema": "production-integrity-v1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository": os.environ.get("GITHUB_REPOSITORY", "unknown"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "unknown"),
        "commit_sha": os.environ.get("GITHUB_SHA", git_value(["rev-parse", "HEAD"])),
        "branch": os.environ.get("GITHUB_REF_NAME", git_value(["branch", "--show-current"])),
        "runner_name": os.environ.get("RUNNER_NAME", socket.gethostname()),
        "runner_os": platform.platform(),
        "python_version": platform.python_version(),
        "requirements_sha256": sha256(Path("requirements.txt")) if Path("requirements.txt").exists() else "missing",
        "checks": checks,
        "total_result_rows": total_rows,
        "probability_columns": probability_columns,
        "result_files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in targets],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
