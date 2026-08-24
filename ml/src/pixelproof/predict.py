"""Command-line entry point for the project model and optional comparison methods."""

from __future__ import annotations

import argparse
from pathlib import Path

from pixelproof.image_input import ImagePolicyError, decode_image, enough_evidence
from pixelproof.serve import METHODS, ModelRuntime, RuntimeUnavailable


def official_label(result: dict) -> str:
    decision = result.get("decision")
    if not result.get("enough_evidence") or not decision:
        return "insufficient evidence"
    return "AI detected" if decision.get("label") == "ai" else "insufficient evidence"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument(
        "--artifacts",
        type=Path,
        help="artifact directory override; defaults to the package's ml/artifacts directory",
    )
    parser.add_argument("--method", choices=METHODS, default="project_model")
    args = parser.parse_args()

    runtime = ModelRuntime(args.artifacts)
    if not runtime.ensure_loaded():
        parser.error(f"model runtime unavailable: {runtime.health()['load_errors']}")

    for path in args.images:
        try:
            raw = path.read_bytes()
            picture = decode_image(raw)
            result = runtime.predict(picture, len(raw), args.method)
            result["enough_evidence"] = enough_evidence(picture)
            if not result["enough_evidence"]:
                result["decision"] = None
        except (OSError, ImagePolicyError, RuntimeUnavailable) as error:
            print(f"{path}: rejected ({error})")
            continue
        project = result.get("project_model")
        if project:
            label = "AI-oriented signal" if project["triggered"] else "below experimental threshold"
            print(
                f"{path}: {label} "
                f"(project_score={project['score']:.3f}, threshold={project['threshold']:.3f}, "
                f"model={project['revision']}, research_only=true)"
            )
        else:
            print(
                f"{path}: {official_label(result)} "
                f"(research_score={result['p_ai']:.3f}, method={result['method']})"
            )


if __name__ == "__main__":
    main()
