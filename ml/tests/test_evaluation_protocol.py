import numpy as np
import pytest

from pixelproof.evaluation_protocol import (
    AGGREGATION_RULES,
    aggregate_tile_scores,
    evaluate_image_score_records,
    operating_point,
    stable_calibration_split,
    threshold_at_fpr,
    union_operating_point,
    union_threshold_at_fpr,
)


def test_every_declared_aggregation_returns_a_finite_image_score():
    tiles = np.linspace(0.0, 1.0, 32)
    for rule in AGGREGATION_RULES:
        assert np.isfinite(aggregate_tile_scores(tiles, rule))


def test_fixed16_control_does_not_give_large_images_unlimited_high_score_chances():
    tiles = np.zeros(64)
    tiles[1] = 1.0  # not one of the sixteen deterministic spatial samples
    assert aggregate_tile_scores(tiles, "top3") > 0.0
    assert aggregate_tile_scores(tiles, "fixed16_top3") == 0.0


def test_stable_split_is_disjoint_repeatable_and_keeps_both_halves_nonempty():
    records = [{"path": f"image-{index}.png"} for index in range(10)]
    first_cal, first_eval = stable_calibration_split(records, 0.4, seed=2026)
    second_cal, second_eval = stable_calibration_split(records, 0.4, seed=2026)

    assert first_cal == second_cal
    assert first_eval == second_eval
    assert len(first_cal) == 4
    assert len(first_eval) == 6
    assert {r["path"] for r in first_cal}.isdisjoint(r["path"] for r in first_eval)


def test_threshold_respects_budget_even_when_boundary_scores_are_tied():
    real = [0.9, 0.8, 0.8, 0.8, 0.1]
    threshold = threshold_at_fpr(real, budget=0.4)
    assert np.mean(np.asarray(real) >= threshold) <= 0.4


def test_operating_point_never_uses_evaluation_real_to_fit_threshold():
    calibration = [0.1, 0.2, 0.3, 0.4, 0.5]
    original = operating_point(calibration, [0.2, 0.6], [0.7, 0.8], budget=0.2)
    shifted = operating_point(calibration, [0.95, 0.99], [0.7, 0.8], budget=0.2)

    assert original["threshold"] == shifted["threshold"]
    assert original["evaluation_fp"] != shifted["evaluation_fp"]


def test_union_threshold_uses_only_remaining_calibration_capacity():
    baseline = {
        "clean": [True, False, False, False, False],
        "shifted": [True, True, False, False, False],
    }
    arm = {
        "clean": [100.0, 0.9, 0.8, 0.1, 0.0],
        "shifted": [100.0, 99.0, 0.7, 0.6, 0.5],
    }
    threshold = union_threshold_at_fpr(baseline, arm, budget=0.2)

    # clean has one baseline hit (its full budget); shifted is already above the
    # budget. Neither source may gain a new calibration hit.
    assert threshold > 0.9
    for source in baseline:
        hits = np.asarray(baseline[source]) | (np.asarray(arm[source]) >= threshold)
        assert hits.mean() <= max(0.2, np.mean(baseline[source]))


def test_union_evaluation_scores_cannot_change_the_fitted_threshold():
    calibration_baseline = {"camera": [False] * 10}
    calibration_arm = {"camera": np.linspace(0.0, 0.9, 10)}
    evaluation_baseline = {"camera": [False] * 5}

    original = union_operating_point(
        calibration_baseline,
        calibration_arm,
        evaluation_baseline,
        {"camera": [0.0] * 5},
        budget=0.1,
    )
    shifted = union_operating_point(
        calibration_baseline,
        calibration_arm,
        evaluation_baseline,
        {"camera": [100.0] * 5},
        budget=0.1,
    )

    assert original["threshold"] == shifted["threshold"]
    assert original["evaluation_union_fp"] != shifted["evaluation_union_fp"]


def test_invalid_protocol_parameters_fail_loudly():
    with pytest.raises(ValueError):
        stable_calibration_split([], 1.0, seed=0)
    with pytest.raises(ValueError):
        threshold_at_fpr([0.1], 1.0)
    with pytest.raises(ValueError):
        aggregate_tile_scores([0.1], "invented")


def test_whole_image_protocol_transfers_one_calibration_threshold_to_all_sources():
    def records(prefix, values):
        return [
            {"path": f"{prefix}-{index}.png", "image_score": value}
            for index, value in enumerate(values)
        ]

    result = evaluate_image_score_records(
        real_records=records("real", [0.1, 0.2, 0.3, 0.4]),
        generator_records={"generator": records("ai", [0.6, 0.7, 0.8, 0.9])},
        genimage_records=(
            records("gen-real", [0.1, 0.2]), records("gen-ai", [0.8, 0.9])
        ),
        forensic_records={
            "clean": records("clean", [0.0, 0.1]),
            "shifted": records("shifted", [0.95, 0.99]),
        },
        calibration_fraction=0.5,
        split_seed=2026,
        budget=0.5,
    )

    assert result["defactify_calibration_fp"] <= 0.5
    assert result["forensics_source_fp"]["clean"] == 0.0
    assert result["forensics_source_fp"]["shifted"] == 1.0
    assert result["forensics_worst_fp"] == 1.0
