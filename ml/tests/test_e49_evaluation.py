from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from experiments.e49_evaluation import (
    CONDITIONS,
    SOURCE_COUNTS,
    bootstrap_primary,
    evaluate_condition,
    selective_metrics,
    source_rates,
    validate_paired_final,
)


def _small_rows():
    return [
        {"parent_id": "r1", "label": 0, "source": "camera-a", "score": 0.001, "status": "ok"},
        {"parent_id": "r2", "label": 0, "source": "camera-b", "score": 0.002, "status": "ok"},
        {"parent_id": "a1", "label": 1, "source": "model-a", "score": 0.8, "status": "ok"},
        {"parent_id": "a2", "label": 1, "source": "model-b", "score": 0.9, "status": "ok"},
    ]


def _paired_rows():
    parents = []
    for source, count in SOURCE_COUNTS.items():
        label = 0 if count == 100 else 1
        for index in range(count):
            parents.append({"parent_id": f"{source}:{index}", "label": label, "source": source})
    return [{**row, "condition": condition, "status": "ok",
             "score": 0.001 if row["label"] == 0 else 0.9}
            for condition in CONDITIONS for row in parents]


def test_perfect_condition_passes_all_ten_frozen_gates():
    found = evaluate_condition(_small_rows())
    assert found["gate"] == {"passed": True, "passed_checks": 10, "total_checks": 10,
                             "checks": found["gate"]["checks"]}


def test_failures_are_uncertain_and_pessimistic_for_source_rates():
    rows = _small_rows()
    rows[0] = {**rows[0], "status": "error", "score": None}
    rows[2] = {**rows[2], "status": "error", "score": None}
    rates = source_rates(rows)
    selective = selective_metrics(rows)
    assert rates["real_false_ai_by_source"]["camera-a"] == 1.0
    assert rates["ai_recall_by_source"]["model-a"] == 0.0
    assert selective["automatic_coverage"] == 0.5
    assert selective["uncertain_rate"] == 0.5


def test_paired_validator_binds_exact_parent_and_source_quotas():
    rows = _paired_rows()
    validate_paired_final(rows)
    broken = deepcopy(rows)
    broken[-1]["parent_id"] = "different-parent"
    with pytest.raises(ValueError, match="exact parent"):
        validate_paired_final(broken)


def test_final_ai_sources_match_frozen_openfake_successor():
    assert {source for source, count in SOURCE_COUNTS.items() if count != 100} == {
        "GPT Image 2", "Z-Image Turbo", "Seedream v5.0", "FLUX.2 Klein 9B",
        "Midjourney 7", "StyleGAN2",
    }


def test_bootstrap_is_seeded_parent_level_and_finite():
    first = bootstrap_primary(_small_rows(), samples=100, seed=7)
    second = bootstrap_primary(_small_rows(), samples=100, seed=7)
    assert first == second
    assert all(np.all(np.isfinite(bounds)) for bounds in first.values())


def test_bootstrap_rejects_duplicate_parents():
    rows = _small_rows()
    rows[1]["parent_id"] = rows[0]["parent_id"]
    with pytest.raises(ValueError, match="unique parent"):
        bootstrap_primary(rows, samples=10)
