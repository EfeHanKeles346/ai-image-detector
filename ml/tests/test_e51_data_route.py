from __future__ import annotations

from collections import Counter

import pytest

from experiments.e51_data_route import (
    DATAPOINT_RESERVE_PER_MODEL,
    DATAPOINT_MODELS,
    DATAPOINT_PROMPT_CATEGORIES,
    DATAPOINT_RESERVE_PER_CATEGORY,
    select_datapoint_paired_rows,
    select_datapoint_rows,
    select_scmi30_rows,
    summarize_ieee,
)


def test_scmi30_selection_is_device_and_branch_balanced():
    files = []
    for device in range(1, 31):
        for branch in ("Random", "Similar"):
            for index in range(25):
                files.append({
                    "name": f"SCMI30-IITRPR/{branch}/D{device:02d}_Phone/image_{index}.jpg",
                    "total_bytes": 100 + index,
                })
    selected = select_scmi30_rows(files)
    assert len(selected) == 1_200
    assert set(Counter((row["device_id"], row["branch"]) for row in selected).values()) == {20}
    assert all(row["role"] == "CAL" and row["label"] == 0 for row in selected)


def test_scmi30_selection_rejects_missing_branch():
    files = [
        {"name": f"SCMI30-IITRPR/Random/D{device:02d}_Phone/a.jpg", "total_bytes": 1}
        for device in range(1, 31)
    ]
    with pytest.raises(ValueError, match="device/branch"):
        select_scmi30_rows(files, per_branch_device=1)


def test_ieee_inventory_keeps_all_real_test_cells():
    files = [{"name": "sample_submission.csv", "total_bytes": 5}]
    files.extend({"name": f"train/train/D01/a_{i}.jpg", "total_bytes": 10} for i in range(2_750))
    files.extend(
        {"name": f"test/test/img_{i:07x}_{suffix}.tif", "total_bytes": 20}
        for suffix in ("manip", "unalt") for i in range(1_320)
    )
    found = summarize_ieee(files)
    assert found["test_files"] == 2_640
    assert found["transport_cells"] == {"postprocessed": 1_320, "unaltered": 1_320}
    assert all(row["label"] == 0 and row["role"] == "DEVELOPMENT" for row in found["rows"])


def test_ieee_inventory_rejects_cell_drift():
    files = [{"name": "sample_submission.csv", "total_bytes": 5}]
    files.extend({"name": f"train/train/D01/a_{i}.jpg", "total_bytes": 10} for i in range(2_750))
    files.extend(
        {"name": f"test/test/img_{i:07x}_manip.tif", "total_bytes": 20}
        for i in range(2_640)
    )
    with pytest.raises(ValueError, match="test inventory"):
        summarize_ieee(files)


def test_datapoint_selection_is_hash_ranked_and_deduplicated():
    rows = []
    for index in range(DATAPOINT_RESERVE_PER_MODEL + 20):
        rows.append({
            "model_id": "flux-2-max",
            "image_key": f"flux-2-max/{index + 1:03d}-x.png",
            "prompt_ordinal": index + 1,
            "prompt_category": "category",
            "width": 1_024,
            "height": 1_024,
            "source_format": "jpeg",
            "byte_size": 100,
            "sha256": f"{index:064x}",
            "source_shard": "data/images/images-0001.parquet",
        })
    rows.append(dict(rows[0]))
    first = select_datapoint_rows(rows, "flux-2-max")
    second = select_datapoint_rows(list(reversed(rows)), "flux-2-max")
    assert first == second
    assert len(first) == DATAPOINT_RESERVE_PER_MODEL
    assert len({row["expected_sha256"] for row in first}) == len(first)
    assert all(row["label"] == 1 and row["role"] == "DEVELOPMENT" for row in first)


def test_datapoint_paired_selection_uses_identical_prompt_ids():
    rows_by_model = {}
    for model_id in DATAPOINT_MODELS:
        rows = []
        population = DATAPOINT_PROMPT_CATEGORIES * (DATAPOINT_RESERVE_PER_CATEGORY + 1)
        for ordinal in range(1, population + 1):
            rows.append({
                "model_id": model_id,
                "image_key": f"{model_id}/{ordinal:03d}-x.png",
                "prompt_ordinal": ordinal,
                "prompt_category": f"category-{(ordinal - 1) // (DATAPOINT_RESERVE_PER_CATEGORY + 1)}",
                "width": 1_024,
                "height": 1_024,
                "source_format": "jpeg",
                "byte_size": 100,
                "sha256": __import__("hashlib").sha256(f"{model_id}:{ordinal}".encode()).hexdigest(),
                "source_shard": "data/images/images-0001.parquet",
            })
        rows_by_model[model_id] = rows
    selected, prompts = select_datapoint_paired_rows(rows_by_model)
    assert len(prompts) == DATAPOINT_RESERVE_PER_MODEL
    assert len(selected) == len(DATAPOINT_MODELS) * DATAPOINT_RESERVE_PER_MODEL
    by_model = {
        model_id: [row["prompt_ordinal"] for row in selected if row["model_id"] == model_id]
        for model_id in DATAPOINT_MODELS
    }
    assert all(value == prompts for value in by_model.values())
