"""E21 — evaluate an official external detector on PixelProof's protocol.

This script intentionally does not vendor third-party source or weights.  Each
supported detector is loaded exactly as its authors ship it, and its image
scores are evaluated with the same disjoint calibration/source-transfer rules
as E20-v2, so every external baseline answers the question our own models
failed: does the threshold survive unseen real sources?

Two arms:

- ``bfree`` — B-Free (GRIP-UNINA, CVPR 2025).  Loads an explicitly supplied
  checkout of the official repository and runs the authors' native inference
  contract.  The licence permits informational and nonprofit use only; the
  explicit CLI acknowledgement prevents this research adapter from quietly
  becoming a commercial runtime dependency later.
- ``community-forensics`` — the Community-Forensics ViT-S (Park & Owens,
  CVPR 2025; MIT licence), the strongest out-of-the-box detector in the
  23-model benchmark of arXiv 2602.07814.  Loaded from a local snapshot, or
  from the HuggingFace hub only when ``--allow-download`` is passed — the
  script never downloads weights silently.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import Compose

from pixelproof.evaluation_protocol import evaluate_image_score_records


HOME = Path.home() / "Desktop"
GENERATORS = ("dalle3", "midjourney", "sd21", "sd3", "sdxl")
FP_BUDGET = 0.10
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
BFree_URL = "https://github.com/grip-unina/B-Free"
CF_HUB_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
CF_URL = f"https://huggingface.co/{CF_HUB_ID}"


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


class BFreeDetector:
    """Thin loader around an untouched official B-Free checkout."""

    def __init__(
        self,
        repo: Path,
        weights_dir: Path,
        model_name: str,
        device: torch.device,
    ) -> None:
        code_dir = repo / "code"
        license_path = repo / "LICENSE.txt"
        config_path = weights_dir / model_name / "config.yaml"
        if not code_dir.is_dir() or not license_path.is_file():
            raise FileNotFoundError(
                f"{repo} is not an official B-Free checkout (code/ or LICENSE.txt missing)"
            )
        if not config_path.is_file():
            raise FileNotFoundError(
                f"{config_path} missing; download BFREE_dino2reg4.zip as documented upstream"
            )

        config = yaml.safe_load(config_path.read_text())
        weights_path = config_path.parent / config["weights_file"]
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)

        # Upstream's networks package uses relative imports, so load it as the
        # package the authors ship.  No source is copied or modified here.
        sys.path.insert(0, str(code_dir))
        try:
            networks = importlib.import_module("networks")
            normalization = importlib.import_module("utils.normalization")
        finally:
            sys.path.pop(0)

        began = time.perf_counter()
        model = networks.get_network(config["arch"])
        self.model = networks.load_weights(model, weights_path).to(device).eval()
        self.transform = Compose(normalization.get_list_norm(config["norm_type"]))
        self.device = device
        self.repo = repo.resolve()
        self.revision = git_revision(repo)
        self.weights_sha256 = sha256_file(weights_path)
        self.detector_id = (
            f"bfree:{model_name}:{self.revision}:{self.weights_sha256[:16]}"
        )
        self.metadata = {
            "name": "B-Free",
            "model_name": model_name,
            "source": BFree_URL,
            "checkout": str(self.repo),
            "upstream_commit": self.revision,
            "weights": str(weights_path.resolve()),
            "weights_sha256": self.weights_sha256,
            "architecture": config["arch"],
            "normalization": config["norm_type"],
            "input_contract": "official native-resolution five-crop mean at 504px; no resize",
            "license_scope": "informational and nonprofit only",
            "device": str(device),
            "load_seconds": time.perf_counter() - began,
        }

    @torch.inference_mode()
    def score(self, path: Path) -> tuple[float, int, int]:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(tensor)
        if output.ndim != 2 or output.shape[0] != 1:
            raise RuntimeError(f"unexpected B-Free output shape {tuple(output.shape)}")
        if output.shape[1] == 1:
            score = output[0, 0]
        elif output.shape[1] == 2:
            score = output[0, 1] - output[0, 0]
        else:
            raise RuntimeError(f"unexpected B-Free class count {output.shape[1]}")
        return float(score.cpu()), width, height


class CommunityForensicsDetector:
    """The Community-Forensics ViT-S, loaded exactly as published (MIT)."""

    def __init__(
        self,
        model_dir: Path | None,
        device: torch.device,
        allow_download: bool,
    ) -> None:
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
        except ImportError as error:
            raise RuntimeError(
                "transformers is required for --detector community-forensics "
                "(.venv/bin/pip install transformers)"
            ) from error

        if model_dir is not None:
            source = str(model_dir)
            weights_path = model_dir / "model.safetensors"
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"{weights_path} missing; expected a local snapshot of {CF_HUB_ID}"
                )
            local_only = True
        elif allow_download:
            # ~83 MB from the hub, cached under ~/.cache/huggingface afterwards.
            from huggingface_hub import hf_hub_download

            source = CF_HUB_ID
            weights_path = Path(hf_hub_download(CF_HUB_ID, "model.safetensors"))
            local_only = False
        else:
            raise FileNotFoundError(
                "no --model-dir given and downloads are not allowed; pass a local "
                f"snapshot of {CF_HUB_ID}, or --allow-download to fetch it (~83 MB, MIT)"
            )

        began = time.perf_counter()
        model = ViTForImageClassification.from_pretrained(
            source, local_files_only=local_only
        )
        self.processor = ViTImageProcessor.from_pretrained(
            source, local_files_only=local_only
        )
        self.model = model.to(device).eval()
        self.device = device
        self.weights_sha256 = sha256_file(weights_path)
        self.detector_id = f"cf-vit:{self.weights_sha256[:16]}"
        self.metadata = {
            "name": "Community-Forensics ViT-S",
            "source": CF_URL,
            "weights": str(weights_path.resolve()),
            "weights_sha256": self.weights_sha256,
            "architecture": model.config.model_type,
            "num_labels": model.config.num_labels,
            "preprocessing": {
                "size": dict(self.processor.size),
                "crop": dict(getattr(self.processor, "crop_size", {}) or {}),
                "image_mean": list(self.processor.image_mean),
                "image_std": list(self.processor.image_std),
            },
            "input_contract": (
                "authors' processor: shortest edge to 440, centre-crop 384, CLIP "
                "normalisation — whole image, no tiling"
            ),
            "license_scope": "MIT",
            "device": str(device),
            "load_seconds": time.perf_counter() - began,
        }

    @torch.inference_mode()
    def score(self, path: Path) -> tuple[float, int, int]:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            inputs = self.processor(images=image, return_tensors="pt")
        logits = self.model(pixel_values=inputs["pixel_values"].to(self.device)).logits
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError(f"unexpected CF-ViT output shape {tuple(logits.shape)}")
        if logits.shape[1] == 1:  # single logit; sigmoid(logit) = P(fake)
            score = logits[0, 0]
        elif logits.shape[1] == 2:
            score = logits[0, 1] - logits[0, 0]
        else:
            raise RuntimeError(f"unexpected CF-ViT class count {logits.shape[1]}")
        return float(score.cpu()), width, height


def image_files(folder: Path, limit: int) -> list[Path]:
    files = sorted(
        path
        for path in folder.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
        and not path.name.startswith("._")
        and ".mask" not in path.name
    )
    if not files:
        return []
    return files[:: max(1, len(files) // limit)][:limit]


def load_cache(path: Path, detector_id: str) -> dict[str, dict]:
    if not path.is_file():
        return {}
    records: dict[str, dict] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
            if record.get("detector_id") == detector_id and "image_score" in record:
                records[record["path"]] = record
    return records


def score_folder(
    folder: Path,
    detector: BFreeDetector | CommunityForensicsDetector,
    cache: dict[str, dict],
    raw_path: Path,
    dataset: str,
    source: str,
    label: int,
    limit: int,
) -> tuple[list[dict], list[dict]]:
    paths = image_files(folder, limit)
    records: list[dict] = []
    failures: list[dict] = []
    new_count = 0
    began = time.perf_counter()
    print(f"  {dataset}/{source}: {len(paths)} images", flush=True)
    with raw_path.open("a") as raw:
        for index, path in enumerate(paths, 1):
            key = str(path)
            cached = cache.get(key)
            if cached is not None:
                records.append(cached)
                continue
            try:
                image_score, width, height = detector.score(path)
                record = {
                    "detector_id": detector.detector_id,
                    "path": key,
                    "dataset": dataset,
                    "source": source,
                    "label": label,
                    "width": width,
                    "height": height,
                    "bytes_per_pixel": path.stat().st_size / max(width * height, 1),
                    "image_score": image_score,
                }
                raw.write(json.dumps(record, separators=(",", ":")) + "\n")
                raw.flush()
                cache[key] = record
                records.append(record)
                new_count += 1
            except Exception as error:  # keep a corrupt file from erasing a long run
                failure = {"path": key, "error": f"{type(error).__name__}: {error}"}
                failures.append(failure)
                print(f"    FAILED {key}: {failure['error']}", flush=True)
            if new_count and new_count % 25 == 0:
                elapsed = time.perf_counter() - began
                print(
                    f"    {index}/{len(paths)} ({new_count} newly scored, {elapsed:.0f}s)",
                    flush=True,
                )
    return records, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector", choices=["bfree", "community-forensics"], default="bfree"
    )
    parser.add_argument("--repo", type=Path, help="official B-Free checkout (bfree only)")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help="directory containing BFREE_dino2reg4 (default: REPO/weights)",
    )
    parser.add_argument("--model", default="BFREE_dino2reg4")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help=f"local snapshot of {CF_HUB_ID} (community-forensics only)",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="permit fetching the community-forensics checkpoint (~83 MB, MIT) from "
        "HuggingFace when no --model-dir is given",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--raw", type=Path, help="per-image score JSONL (default per detector)")
    parser.add_argument("--results", type=Path, help="results JSON (default per detector)")
    parser.add_argument(
        "--acknowledge-noncommercial-license",
        action="store_true",
        help="confirm this run is informational/nonprofit as required by B-Free's licence",
    )
    args = parser.parse_args()
    if args.limit < 2:
        parser.error("--limit must be at least 2 for disjoint calibration/evaluation")
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("--calibration-fraction must be strictly between 0 and 1")

    device = select_device(args.device)
    if args.detector == "bfree":
        if args.repo is None:
            parser.error("--detector bfree requires --repo")
        if not args.acknowledge_noncommercial_license:
            parser.error("B-Free requires --acknowledge-noncommercial-license")
        weights_dir = args.weights_dir or args.repo / "weights"
        detector = BFreeDetector(args.repo, weights_dir, args.model, device)
        stem = "bfree"
    else:
        detector = CommunityForensicsDetector(
            args.model_dir, device, args.allow_download
        )
        stem = "cf_vit"
    args.raw = args.raw or Path(f"artifacts/e21/{stem}_scores.jsonl")
    args.results = args.results or Path(f"artifacts/e21/{stem}_results.json")
    print(
        f"E21 {detector.detector_id}\n"
        f"  device: {device}  load: {detector.metadata['load_seconds']:.1f}s\n"
        f"  contract: {detector.metadata['input_contract']}\n",
        flush=True,
    )

    args.raw.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.raw, detector.detector_id)
    if cache:
        print(f"  resume cache: {len(cache)} image scores\n", flush=True)

    failures: list[dict] = []

    def run(folder: Path, dataset: str, source: str, label: int) -> list[dict]:
        records, failed = score_folder(
            folder, detector, cache, args.raw, dataset, source, label, args.limit
        )
        failures.extend(failed)
        return records

    began = time.perf_counter()
    defactify_real = run(HOME / "defactify_test/real", "defactify", "real", 0)
    generator_records = {
        generator: run(
            HOME / "defactify_test/ai" / generator,
            "defactify",
            generator,
            1,
        )
        for generator in GENERATORS
    }
    genimage_records = (
        run(HOME / "genimage_split/test/REAL", "genimage", "real", 0),
        run(HOME / "genimage_split/test/FAKE", "genimage", "ai", 1),
    )
    forensic_records = {
        folder.parent.name: run(folder, "forensics", folder.parent.name, 0)
        for folder in sorted((HOME / "manipulation_test").glob("*/auth"))
    }

    metrics = evaluate_image_score_records(
        defactify_real,
        generator_records,
        genimage_records,
        forensic_records,
        args.calibration_fraction,
        args.split_seed,
        FP_BUDGET,
    )
    elapsed = time.perf_counter() - began
    payload = {
        "protocol": {
            "name": "E20-v2 whole-image external baseline",
            "fp_budget": FP_BUDGET,
            "calibration_fraction": args.calibration_fraction,
            "split_seed": args.split_seed,
            "limit_per_set": args.limit,
            "threshold_fit": "Defactify real calibration half only",
            "evaluation": "untouched Defactify halves plus unchanged forensic sources",
        },
        "model": detector.metadata,
        "raw_scores": str(args.raw),
        "elapsed_seconds": elapsed,
        "failures": failures,
        "metrics": metrics,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w") as handle:
        json.dump(json_compatible(payload), handle, indent=2, allow_nan=False)

    separator = "=" * 92
    detector_name = detector.metadata["name"].upper()
    print(f"\n{separator}\nE21 {detector_name} — untouched evaluation results\n{separator}")
    print(
        f"Defactify: AUC {metrics['defactify_evaluation_auc']:.3f} · "
        f"recall {100 * metrics['defactify_evaluation_recall']:.1f}% · "
        f"FP {100 * metrics['defactify_evaluation_fp']:.1f}%"
    )
    print(
        f"Forensic reals: macro FP {100 * metrics['forensics_macro_fp']:.1f}% · "
        f"worst-source FP {100 * metrics['forensics_worst_fp']:.1f}%"
    )
    print(
        "Per-generator recall: "
        + ", ".join(
            f"{name} {100 * values['evaluation_recall']:.1f}%"
            for name, values in metrics["per_generator"].items()
        )
    )
    print(f"Raw scores -> {args.raw}\nResults -> {args.results}")
    if failures:
        print(f"WARNING: {len(failures)} images failed; inspect results JSON")


if __name__ == "__main__":
    main()
