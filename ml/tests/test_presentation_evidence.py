import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_model_card_and_demo_evidence_are_bound_to_the_canonical_artifact():
    manifest = json.loads((ROOT / "ml/artifacts.manifest.json").read_text())
    entry = next(
        item for item in manifest["artifacts"]
        if item["id"] == "e20-tile-resnet18-seed2024"
    )
    bfree = next(item for item in manifest["artifacts"] if item["id"] == "bfree-dino2reg4")
    evidence = json.loads((ROOT / "evidence/demo_disagreement.json").read_text())
    model_card = (ROOT / "MODEL_CARD.md").read_text()

    assert evidence["project_model"]["artifact_sha256"] == entry["sha256"]
    assert evidence["input"]["source_revision"] == bfree["revision"]
    assert entry["sha256"] in model_card
    assert evidence["input"]["upstream_label"] == 0
    assert evidence["project_model"]["triggered"] is True
    assert evidence["project_model"]["research_only"] is True
    assert evidence["external_comparison"]["decision"] == "insufficient"
    assert "86.2% ± 3.1%" in model_card

    optional_input = ROOT / evidence["input"]["path"]
    if optional_input.is_file():
        assert hashlib.sha256(optional_input.read_bytes()).hexdigest() == evidence["input"]["sha256"]


def test_presentation_ledger_names_every_completed_model_first_commit():
    presentation = (ROOT / "PRESENTATION_EVIDENCE.md").read_text()
    for commit in (
        "774520b",
        "71d6435",
        "fa106e2",
        "3f87d72",
        "f158f0e",
        "590c3ae",
        "95fe2b2",
    ):
        assert commit in presentation
    assert "evidence/demo_disagreement.json" in presentation
