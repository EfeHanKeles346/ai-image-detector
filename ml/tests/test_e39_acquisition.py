from __future__ import annotations

from experiments.e39_acquisition import (
    AI_ARCHIVE,
    AI_ARCHIVE_BYTES,
    AI_ARCHIVE_SHA256,
    AI_XET_HASH,
    HF_REPO,
    HF_REVISION,
    REAL_DEVICES,
    select_ai_members,
    select_real_urls,
    validate_huggingface,
)


def test_real_selection_is_device_balanced_and_location_diverse(monkeypatch) -> None:
    lines = []
    for device in REAL_DEVICES:
        for location in range(1, 8):
            for subject in range(1, 4):
                for capture in range(1, 4):
                    lines.append(
                        f"https://lesc.dinfo.unifi.it/FloreView/Dataset/{device}/Nat/jpeg-h264/"
                        f"L{location}/S{subject}/{device[:3]}_L{location}S{subject}C{capture}.jpg"
                    )
    raw = ("\n".join(lines) + "\n").encode()
    monkeypatch.setattr("experiments.e39_acquisition.FLOREVIEW_CATALOG_SHA256", __import__("hashlib").sha256(raw).hexdigest())
    selected = select_real_urls(raw)
    assert {device: len(rows) for device, rows in selected.items()} == {device: 40 for device in REAL_DEVICES}
    assert all(len({row["remote_path"].split("/")[-3] for row in rows}) == 7 for rows in selected.values())


def test_huggingface_contract_requires_pinned_archive() -> None:
    metadata = {"id": HF_REPO, "sha": HF_REVISION, "cardData": {"license": "cc-by-4.0"}}
    tree = [{
        "path": AI_ARCHIVE,
        "size": AI_ARCHIVE_BYTES,
        "xetHash": AI_XET_HASH,
        "lfs": {"oid": AI_ARCHIVE_SHA256},
    }]
    result = validate_huggingface(metadata, tree)
    assert result["sha256"] == AI_ARCHIVE_SHA256
    tree[0]["size"] -= 1
    try:
        validate_huggingface(metadata, tree)
    except ValueError as error:
        assert "archive contract changed" in str(error)
    else:
        raise AssertionError("changed archive must fail closed")


def test_ai_selection_is_generator_balanced_and_score_blind() -> None:
    prefixes = {
        "Reve Image 1.0": "fal-ai_reve_text-to-image_",
        "HiDream I1 Dev": "fal-ai_hidream-i1-dev_",
        "Ideogram 3": "fal-ai_ideogram_v3_",
        "Midjourney v7": "image_midjourneyv7_",
        "Adobe Firefly Image 5": "Firefly_",
        "Z Image Turbo": "fal-ai_z-image_turbo_",
        "Gemini 3 Pro Image": "fal-ai_gemini-3-pro-image-preview_",
    }
    members = [
        {"member": f"root/train/1_fake/{prefix}{index:03d}.png", "bytes": 100 + index}
        for prefix in prefixes.values()
        for index in range(45)
    ]
    selected, eligible = select_ai_members(members)
    assert eligible == {generator: 45 for generator in prefixes}
    assert {generator: len(rows) for generator, rows in selected.items()} == {generator: 40 for generator in prefixes}
    assert all("score" not in row for rows in selected.values() for row in rows)
