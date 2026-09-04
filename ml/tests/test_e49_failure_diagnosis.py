from __future__ import annotations

from experiments.e49_failure_diagnosis import diagnose_rows


def test_failure_diagnosis_reports_pair_consensus_without_new_cut():
    rows = []
    for parent_id, label, original, q75 in (("r", 0, 0.01, 0.2), ("a", 1, 0.8, 0.9)):
        common = {"parent_id": parent_id, "label": label, "source": "source",
                  "format": "JPEG", "width": 1000, "height": 1000}
        rows.extend([
            {**common, "condition": "publisher_original", "score": original},
            {**common, "condition": "social_q75", "score": q75},
        ])
    found = diagnose_rows(rows)
    assert found["paired_existing_cut"]["0"]["both_at_or_above_binary_cut"] == 0.0
    assert found["paired_existing_cut"]["0"]["either_at_or_above_binary_cut"] == 1.0
    assert found["paired_existing_cut"]["1"]["both_at_or_above_binary_cut"] == 1.0
