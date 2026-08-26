"""Leave one complete E32 source out of R0 fitting and threshold selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import e32_r0_train as r0
except ModuleNotFoundError:  # imported as experiments.e32_r0_loco in tests
    from experiments import e32_r0_train as r0
from pixelproof.project_paths import ML_ROOT


OUTPUT = ML_ROOT.parent / "evidence" / "e32_r0_loco.json"
EXPECTED_FEATURE_SHA256 = "716df956fc40b1cf557b30c34e1adb216d2abeb2314c20b2edf10671a387be3b"
EXPECTED_INPUT_SHA256 = "2255b123a24b2cd7cb371878a134f39f7ef352ed437dd57494c55280e01f5199"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def source_metric(label: int, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted_ai = scores >= threshold
    result = {
        "count": int(len(scores)),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "p90_score": float(np.quantile(scores, 0.90)),
        "positive_rate": float(np.mean(predicted_ai)),
    }
    result["ai_recall" if label == 1 else "real_false_positive_rate"] = result["positive_rate"]
    return result


def run() -> dict[str, Any]:
    input_evidence = json.loads(r0.INPUT_EVIDENCE.read_text())
    if input_evidence["detailed_report_sha256"] != EXPECTED_INPUT_SHA256:
        raise ValueError("R0 input evidence changed")
    if r0._sha256_file(r0.FEATURE_PATH) != EXPECTED_FEATURE_SHA256:
        raise ValueError("R0 feature archive changed")
    with np.load(r0.FEATURE_PATH, allow_pickle=False) as stored:
        features = stored["features"]
        labels = stored["labels"].astype(np.int64)
        roles = stored["roles"].astype(str)
        sources = stored["sources"].astype(str)
    results = []
    for heldout_source in sorted(set(sources.tolist())):
        heldout = sources == heldout_source
        train = (roles == "TRAIN") & ~heldout
        calibration = (roles == "CALIBRATION") & ~heldout
        if len(set(labels[train].tolist())) != 2 or len(set(labels[calibration].tolist())) != 2:
            raise ValueError(f"LOCO {heldout_source} loses a class")
        head = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=3000,
                random_state=r0.SEED,
                solver="lbfgs",
            ),
        )
        head.fit(features[train], labels[train])
        calibration_scores = head.predict_proba(features[calibration])[:, 1]
        threshold = r0.threshold_at_fp_budget(
            labels[calibration], calibration_scores, sources[calibration]
        )
        heldout_scores = head.predict_proba(features[heldout])[:, 1]
        unique_label = set(labels[heldout].tolist())
        if len(unique_label) != 1:
            raise ValueError(f"source {heldout_source} has mixed labels")
        result = {
            "source_id": heldout_source,
            "label": "ai" if next(iter(unique_label)) == 1 else "real",
            "fit_rows": int(train.sum()),
            "threshold_rows": int(calibration.sum()),
            "threshold": threshold,
            **source_metric(next(iter(unique_label)), heldout_scores, threshold),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    real_rates = [row["real_false_positive_rate"] for row in results if row["label"] == "real"]
    ai_rates = [row["ai_recall"] for row in results if row["label"] == "ai"]
    report = {
        "schema_version": 1,
        "experiment": "E32/C4-R0-leave-one-collection-out",
        "state": "loco_diagnostic_complete_no_artifact_change",
        "feature_archive_sha256": EXPECTED_FEATURE_SHA256,
        "input_receipt_sha256": EXPECTED_INPUT_SHA256,
        "head_c": 0.1,
        "source_results": results,
        "summary": {
            "real_source_macro_fp": float(np.mean(real_rates)),
            "real_source_worst_fp": float(max(real_rates)),
            "ai_source_macro_recall": float(np.mean(ai_rates)),
            "ai_source_worst_recall": float(min(ai_rates)),
        },
        "boundary": "Diagnostic source holdouts only; owner gallery is not opened and the accepted artifact is unchanged.",
    }
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".part")
    temporary.write_bytes(_json_bytes(report))
    temporary.replace(OUTPUT)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    print(json.dumps(run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
