from __future__ import annotations

from collections import Counter

import numpy as np

from experiments.e40_adaptation import (
    C_VALUE,
    CLUSTERS,
    FOLDS,
    MODES,
    fold_ids,
    replay_indices,
    sample_weights,
)


def test_head_ladder_hyperparameters_are_preregistered() -> None:
    assert C_VALUE == 0.01
    assert CLUSTERS == 16
    assert MODES == ("uniform", "source_balanced", "source_content_balanced")


def test_fold_contract_assigns_all_e39_sources_once() -> None:
    real = sorted(source for fold in FOLDS for source in fold["real"])
    ai = sorted(source for fold in FOLDS for source in fold["ai"])
    assigned = fold_ids(
        np.asarray([0] * len(real) + [1] * len(ai)),
        np.asarray(real + ai),
    )
    assert len(real) == 4
    assert len(ai) == 7
    assert Counter(assigned.tolist()) == Counter({0: 2, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1})


def test_fold_contract_rejects_unknown_source() -> None:
    try:
        fold_ids(np.asarray([0]), np.asarray(["unknown-device"]))
    except ValueError as error:
        assert "0 fold assignments" in str(error)
    else:
        raise AssertionError("unknown source should fail closed")


def test_replay_is_deterministic_and_stratified() -> None:
    record_ids = np.asarray([f"row-{index}" for index in range(80)])
    labels = np.asarray([0] * 40 + [1] * 40)
    roles = np.asarray(["TRAIN"] * 60 + ["CAL"] * 20)
    sources = np.asarray(["real-a"] * 20 + ["real-b"] * 20 + ["ai-a"] * 20 + ["ai-b"] * 20)
    first = replay_indices(record_ids, labels, roles, sources)
    second = replay_indices(record_ids, labels, roles, sources)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 3
    assert set(roles[first]) == {"TRAIN"}


def test_weight_modes_equalize_class_mass() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    sources = np.asarray(["r1", "r1", "r1", "r2", "a1", "a1", "a2", "a2"])
    clusters = np.asarray([0, 0, 1, 0, 0, 1, 0, 0])
    for mode in MODES:
        weights = sample_weights(labels, sources, mode, clusters)
        assert np.isclose(weights[labels == 0].sum(), weights[labels == 1].sum())
        assert np.isclose(weights.mean(), 1.0)


def test_content_weights_equalize_sources_and_occupied_cells() -> None:
    labels = np.asarray([0, 0, 0, 0, 0, 1, 1])
    sources = np.asarray(["r1", "r1", "r1", "r1", "r2", "a1", "a1"])
    clusters = np.asarray([0, 0, 0, 1, 0, 0, 1])
    weights = sample_weights(labels, sources, "source_content_balanced", clusters)
    assert np.isclose(weights[(labels == 0) & (sources == "r1")].sum(), weights[(labels == 0) & (sources == "r2")].sum())
    r1_c0 = weights[(sources == "r1") & (clusters == 0)].sum()
    r1_c1 = weights[(sources == "r1") & (clusters == 1)].sum()
    assert np.isclose(r1_c0, r1_c1)
