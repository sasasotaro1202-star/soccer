import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import backtest as b

b.START = time.time()
b.make_feature_names()
assert len(b.FEATURE_NAMES) == 136, len(b.FEATURE_NAMES)

rng = np.random.default_rng(7)
X = rng.normal(size=(180, len(b.FEATURE_NAMES)))
y = np.array([i % 3 for i in range(180)])
w = np.linspace(0.2, 1.0, 180)
for name, model in b.build_models().items():
    b._fit_sample_weight(model, X, y, w)
    p = model.predict_proba(X[:4])
    assert p.shape == (4, 3), (name, p.shape)
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-8)

folds = b._chronological_folds(500, min_train=140, n_folds=3)
assert folds == [(320, 380), (380, 440), (440, 500)]
assert all(folds[i][1] <= folds[i + 1][0] for i in range(len(folds) - 1))

# Team matching must be conservative: aliases are explicit, never substring-based.
assert b.same_team("Manchester United", "Manchester United FC")
assert b.same_team("Man United", "Manchester United")
assert not b.same_team("United", "Manchester United")
assert not b.same_team("City", "Manchester City")

with tempfile.TemporaryDirectory() as td:
    old = b.CHECKPOINT
    b.CHECKPOINT = Path(td) / "ck.pkl.gz"
    histories = b.defaultdict(list)
    for i in range(3000):
        histories["E0"].append({
            "x": np.zeros(len(b.FEATURE_NAMES)),
            "y": i % 3,
            "ml": np.ones(3) / 3,
            "market": np.ones(3) / 3,
            "score": np.ones(3) / 3,
        })
    b.save_state(
        {"E0|2010|2010/11"}, [{"x": 1}] * 1000,
        [{"x": 2}] * 1000, [], [], [], histories, {}, {},
        {"E0": b.League()}, b.defaultdict(b.Player), {}, {},
        {"A": 1500}, 123, {"E0": 2010}
    )
    ck = b.load_state()
    assert ck is not None
    assert ck["version"] >= 9
    assert ck["cursor"] == 123
    assert len(ck["histories"]["E0"]) == b.MAX_TRAIN
    assert b.CHECKPOINT.stat().st_size < 5_000_000
    b.CHECKPOINT = old

print("SOCCER SMOKE TEST: PASS")
