import numpy as np

from experiments.e43_train import parent_source_weights


def test_e43_weights_balance_labels_sources_parents_and_views():
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    sources = np.array(["r1", "r1", "r1", "r2", "a1", "a1", "a1"])
    parents = np.array(["p1", "p1", "p2", "p3", "p4", "p5", "p5"])
    weights = parent_source_weights(labels, sources, parents)

    assert np.isclose(weights[labels == 0].sum(), weights[labels == 1].sum())
    real_source_totals = [weights[(labels == 0) & (sources == source)].sum() for source in ("r1", "r2")]
    assert np.isclose(real_source_totals[0], real_source_totals[1])
    assert np.isclose(weights[parents == "p1"].sum(), weights[parents == "p2"].sum())
    assert np.isclose(weights[parents == "p4"].sum(), weights[parents == "p5"].sum())


def test_e43_weights_are_finite_positive_and_normalized():
    labels = np.array([0, 0, 1, 1])
    sources = np.array(["real", "real", "ai", "ai"])
    parents = np.array(["r1", "r2", "a1", "a2"])
    weights = parent_source_weights(labels, sources, parents)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0)
    assert np.isclose(weights.sum(), len(weights))
