from __future__ import annotations

import numpy as np

from experiments.e31_ensemble import (
    assign_meta_folds,
    group_bootstrap_gain,
    metrics_from_predictions,
)


def test_meta_folds_keep_groups_together() -> None:
    sources = np.array(["a", "a", "a", "a", "b", "b"])
    groups = np.array(["g1", "g1", "g2", "g3", "g1", "g2"])
    folds = assign_meta_folds(sources, groups, n_folds=2)
    assert len(set(folds[groups == "g1"][sources[groups == "g1"] == "a"])) == 1
    assert set(folds.tolist()) <= {0, 1}


def test_metrics_are_source_macro_not_pooled_only() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.linspace(0.1, 0.9, 6)
    predictions = np.array([False, True, False, True, True, False])
    sources = np.array(["real-a", "real-a", "real-b", "flux-1-dev", "nano-banana", "nano-banana-pro"])
    metrics = metrics_from_predictions(labels, scores, predictions, sources)
    assert metrics["macro_real_fp"] == 0.25
    assert metrics["worst_real_fp"] == 0.5
    assert metrics["current_ai_macro_recall"] == 2 / 3


def test_group_bootstrap_detects_uniform_gain() -> None:
    labels = np.ones(6, dtype=int)
    sources = np.array(["flux-1-dev"] * 2 + ["nano-banana"] * 2 + ["nano-banana-pro"] * 2)
    groups = np.array(["a", "b", "c", "d", "e", "f"])
    baseline = np.zeros(6, dtype=bool)
    candidate = np.ones(6, dtype=bool)
    interval = group_bootstrap_gain(labels, sources, groups, baseline, candidate, replicates=100)
    assert interval["lower_95"] == 1.0
    assert interval["upper_95"] == 1.0
