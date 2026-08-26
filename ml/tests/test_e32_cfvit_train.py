from __future__ import annotations

import numpy as np

from experiments import e32_cfvit_train as cf
from experiments import e32_r0_train as r0


def test_fit_head_selects_a_valid_preregistered_candidate() -> None:
    rng = np.random.default_rng(7)
    train_x = np.r_[rng.normal(-2, 0.2, (40, 4)), rng.normal(2, 0.2, (40, 4))]
    train_y = np.r_[np.zeros(40, dtype=int), np.ones(40, dtype=int)]
    train_role = np.array(["TRAIN"] * 80)
    train_source = np.array(["vision-base-native"] * 40 + ["qwen-image-2512"] * 40)

    real_sources = ["vision-base-native", "forchheim-fodb", "csafe-mcsidb-s21"]
    cal_x = []
    cal_y = []
    cal_source = []
    for source in real_sources:
        cal_x.extend(rng.normal(-2, 0.2, (3, 4)))
        cal_y.extend([0] * 3)
        cal_source.extend([source] * 3)
    for source in r0.CURRENT_AI_SOURCES:
        cal_x.extend(rng.normal(2, 0.2, (3, 4)))
        cal_y.extend([1] * 3)
        cal_source.extend([source] * 3)
    features = np.r_[train_x, np.asarray(cal_x)]
    labels = np.r_[train_y, np.asarray(cal_y)]
    roles = np.r_[train_role, np.array(["CALIBRATION"] * len(cal_y))]
    sources = np.r_[train_source, np.asarray(cal_source)]

    _, best_c, threshold, metrics, grid = cf.fit_head(features, labels, roles, sources)
    assert best_c in r0.C_GRID
    assert set(grid) == {str(value) for value in r0.C_GRID}
    assert 0.0 <= threshold <= 1.0
    assert metrics["auc"] == 1.0
