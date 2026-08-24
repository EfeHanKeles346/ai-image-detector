import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_rejected_e28_candidate_cannot_appear_in_runtime_manifest():
    evidence = json.loads((ROOT / "evidence/e28_seed2024_rejection.json").read_text())
    manifest = json.loads((ROOT / "ml/artifacts.manifest.json").read_text())

    assert evidence["decision"] == "rejected_after_single_seed_gate"
    assert evidence["single_seed_gate"]["all_passed"] is False
    assert evidence["three_seed_run_allowed"] is False
    assert evidence["serving_changed"] is False
    assert evidence["candidate_selected"]["forensics_macro_false_positive_rate"] > 0.35
    assert evidence["candidate_selected"]["forensics_worst_false_positive_rate"] > 0.70
    assert all("e28" not in entry["id"].lower() for entry in manifest["artifacts"])
    assert all(
        entry.get("sha256") != evidence["provenance"]["candidate_sha256"]
        for entry in manifest["artifacts"]
    )
