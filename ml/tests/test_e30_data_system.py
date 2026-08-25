from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "experiments/e30_data_system.py"
SPEC = importlib.util.spec_from_file_location("e30_data_system", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e30 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e30
SPEC.loader.exec_module(e30)


def fake_tree(repo_id, revision, directory):
    if repo_id == e30.MLLM_ID:
        return [
            {"path": f"{directory}/{index}.jpg", "size": 20_000 + index}
            for index in range(1, 31)
        ]
    return [
        {"path": f"{directory}/{index:06d}_hash.png", "size": 1_000_000 + index}
        for index in range(1, 11)
    ]


def test_mllm_selection_is_nine_balanced_cells_under_the_cap():
    assets = e30.freeze_mllm_assets(fake_tree)
    assert len(assets) == 180
    assert sum(asset.label == "real" for asset in assets) == 60
    assert sum(asset.generator == "GPT Image 2" for asset in assets) == 60
    assert sum(asset.generator == "Nano Banana 2" for asset in assets) == 60
    assert sum(int(asset.expected_bytes or 0) for asset in assets) < e30.MLLM_TOTAL_CEILING


def test_qwen_selection_is_generator_balanced_and_capped():
    assets = e30.freeze_qwen_assets(fake_tree)
    assert len(assets) == len(e30.QWEN_GENERATORS) * e30.QWEN_PER_GENERATOR
    for generator in e30.QWEN_GENERATORS:
        assert sum(asset.generator == generator for asset in assets) == e30.QWEN_PER_GENERATOR
    assert all(asset.role == e30.DataRole.LOCKED_FINAL_TEST for asset in assets)


class FakeResponse:
    def __init__(self, chunks, status_code=200):
        self._chunks = chunks
        self.status_code = status_code

    def iter_content(self, chunk_size):
        yield from self._chunks


def test_interrupted_asset_resumes_with_range_and_atomic_replace(tmp_path):
    destination = tmp_path / "image.bin"
    partial = destination.with_suffix(".bin.part")
    partial.write_bytes(b"abc")
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, kwargs["headers"]))
        return FakeResponse([b"def"], status_code=206)

    raw = e30.download_resumable(
        "https://example.invalid/image",
        destination,
        expected_bytes=6,
        hard_ceiling=6,
        requester=request,
    )
    assert raw == b"abcdef"
    assert calls == [("GET", {"Range": "bytes=3-"})]
    assert destination.read_bytes() == b"abcdef"
    assert not partial.exists()


def test_asset_download_refuses_stream_above_declared_size(tmp_path):
    def request(method, url, **kwargs):
        return FakeResponse([b"1234"], status_code=200)

    with pytest.raises(RuntimeError, match="exceeded"):
        e30.download_resumable(
            "https://example.invalid/image",
            tmp_path / "image.bin",
            expected_bytes=3,
            hard_ceiling=10,
            requester=request,
        )


def test_laion_candidates_require_every_frozen_pipeline():
    rows = []
    image_id = 0
    for make, model in e30.LAION_PIPELINES:
        for _ in range(10):
            rows.append(
                {
                    "image_id": str(image_id),
                    "make": make,
                    "model": model,
                    "url": f"https://example.invalid/{image_id}.jpg",
                    "content_sha256": f"{image_id:064x}",
                }
            )
            image_id += 1
    grouped = e30.laion_candidates(rows)
    assert set(grouped) == set(e30.LAION_PIPELINES)
    assert all(len(values) == 10 for values in grouped.values())

    rows.pop()
    with pytest.raises(RuntimeError, match="only 9"):
        e30.laion_candidates(rows)
