from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image


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


class FakeHttpxResponse:
    def __init__(self, chunks, status_code=200):
        self._chunks = chunks
        self.status_code = status_code
        self.closed = False

    def iter_bytes(self, chunk_size):
        yield from self._chunks

    def close(self):
        self.closed = True


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


def test_httpx_style_stream_is_consumed_and_closed(tmp_path):
    response = FakeHttpxResponse([b"abc"])

    def request(method, url, **kwargs):
        return response

    raw = e30.download_resumable(
        "https://example.invalid/image",
        tmp_path / "image.bin",
        expected_bytes=3,
        hard_ceiling=3,
        requester=request,
    )
    assert raw == b"abc"
    assert response.closed


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


def test_derived_variants_inherit_parent_role_content_and_split(tmp_path):
    output = tmp_path / "mllm_development"
    images = output / "images"
    images.mkdir(parents=True)
    records = []
    for index in range(10):
        label = "real" if index < 5 else "ai"
        image = Image.new(
            "RGB",
            (320 + index, 240 + index),
            color=(index * 20, 100, 200 - index * 10),
        )
        path = images / f"{index}.jpg"
        image.save(path, format="JPEG", quality=95)
        raw = path.read_bytes()
        records.append(
            {
                "record_id": f"parent-{index}",
                "role": "development_test",
                "source_id": e30.MLLM_ID,
                "source_revision": "1" * 40,
                "source_key": f"source/{index}.jpg",
                "label": label,
                "group": "matched",
                "transport": "standardized_jpeg",
                "path": f"images/{index}.jpg",
                "generator": "generator" if label == "ai" else None,
                "camera_pipeline": "camera" if label == "real" else None,
                "content_id": None,
                "parent_id": None,
                "sha256": e30.sha256_bytes(raw),
                "dhash": None,
                "bytes": len(raw),
                "width": image.width,
                "height": image.height,
                "image_format": "JPEG",
            }
        )
    (output / "manifest.json").write_text(
        __import__("json").dumps(
            {
                "schema_version": 1,
                "content_set_sha256": "a" * 64,
                "records": records,
            }
        )
    )
    result = e30.derive_mllm_variants(output)
    assert result["parent_count"] == 10
    assert result["derived_count"] == 40
    assert len(result["records"]) == 50
    assert all(record["role"] == "development_test" for record in result["records"])
    derived = [record for record in result["records"] if record["parent_id"]]
    parents = {record["record_id"]: record for record in result["records"] if not record["parent_id"]}
    for record in derived:
        parent = parents[record["parent_id"]]
        assert record["content_id"] == parent["content_id"]
        assert record["label"] == parent["label"]
