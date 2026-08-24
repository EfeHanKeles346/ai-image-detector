import numpy as np
import pytest
import torch

from pixelproof.models import create_model
from pixelproof.stay_positive import (
    create_zero_positive_head,
    extract_nonnegative_features,
    install_head,
    source_stratified_holdout,
    stratified_sample_indices,
    train_positive_head,
)


def synthetic_problem():
    rng = np.random.RandomState(7)
    labels = np.tile(np.array([0, 1], dtype=np.int64), 40)
    sources = np.repeat(np.array(["a", "b", "c", "d"]), 20)
    features = rng.uniform(0.0, 0.2, size=(80, 6)).astype(np.float32)
    features[:, 0] += labels * 0.8
    features[:, 1] += labels * 0.4
    return features, labels, sources


def test_zero_head_and_training_projection_are_nonnegative():
    features, labels, sources = synthetic_problem()
    initial = create_zero_positive_head(features.shape[1])
    assert torch.count_nonzero(initial.weight) == 0
    assert torch.count_nonzero(initial.bias) == 0

    head, metadata = train_positive_head(
        features, labels, sources, seed=2024, epochs=4, batch_size=16
    )

    assert torch.all(head.weight >= 0)
    assert metadata["negative_feature_weights"] == 0
    assert metadata["minimum_feature_weight"] >= 0
    assert metadata["validation_auc"] > 0.9
    assert metadata["best_epoch"] in range(1, 5)


def test_source_stratified_split_is_deterministic_and_disjoint():
    _, labels, sources = synthetic_problem()
    first = source_stratified_holdout(labels, sources, seed=42)
    second = source_stratified_holdout(labels, sources, seed=42)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert not set(first[0]).intersection(first[1])
    assert set(labels[first[1]]) == {0, 1}
    assert set(sources[first[1]]) == {"a", "b", "c", "d"}


def test_smoke_sampling_is_balanced_and_deterministic():
    _, labels, sources = synthetic_problem()
    first = stratified_sample_indices(labels, sources, limit=24, seed=9)
    second = stratified_sample_indices(labels, sources, limit=24, seed=9)

    assert np.array_equal(first, second)
    assert len(first) == 24
    assert set(labels[first]) == {0, 1}
    assert set(sources[first]) == {"a", "b", "c", "d"}


def test_negative_features_are_rejected():
    features, labels, sources = synthetic_problem()
    features[0, 0] = -0.01
    with pytest.raises(ValueError, match="non-negative"):
        train_positive_head(features, labels, sources, seed=1, epochs=1)


def test_frozen_resnet_features_and_installed_head_are_compatible():
    model = create_model("resnet18", pretrained=False, dropout=0.0)
    tiles = np.zeros((2, 128, 128, 3), dtype=np.uint8)
    features = extract_nonnegative_features(
        model, tiles, device=torch.device("cpu"), batch_size=1
    )
    head = create_zero_positive_head(features.shape[1])
    install_head(model, head)

    assert features.shape == (2, 512)
    assert float(features.min()) >= 0
    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert torch.count_nonzero(model.classifier[-1].weight) == 0
