"""Phase 4: restructure the unbiased-tiny-GenImage download into our train/test layout.

- REAL = Nature (ImageNet photos). FAKE = balanced sample across all 7 generators
  so no single generator dominates and classes stay ~50/50.
- Stratified, seeded split; test fraction holds every generator.
- Symlinks instead of copies (saves ~2.4 GB).
"""

import argparse
import random
from pathlib import Path

GENERATORS = ["ADM", "BigGAN", "Midjourney", "VQDM", "glide", "stable_diffusion_v_1_5", "wukong"]


def split_and_link(files: list[Path], label_dir: str, output: Path, test_ratio: float, rng: random.Random) -> tuple[int, int]:
    files = sorted(files)
    rng.shuffle(files)
    test_count = int(len(files) * test_ratio)
    for split, subset in (("test", files[:test_count]), ("train", files[test_count:])):
        target = output / split / label_dir
        target.mkdir(parents=True, exist_ok=True)
        for path in subset:
            link = target / path.name
            if not link.exists():
                link.symlink_to(path.resolve())
    return len(files) - test_count, test_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/Users/efehankeles/Desktop/genimage"))
    parser.add_argument("--output", type=Path, default=Path("/Users/efehankeles/Desktop/genimage_split"))
    parser.add_argument("--per-generator", type=int, default=833)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    real = [f for f in (args.source / "Nature").iterdir() if f.is_file()]
    train_count, test_count = split_and_link(real, "REAL", args.output, args.test_ratio, rng)
    print(f"REAL: {train_count} train / {test_count} test")

    for generator in GENERATORS:
        files = sorted(f for f in (args.source / generator).iterdir() if f.is_file())
        rng.shuffle(files)
        sample = files[: args.per_generator]
        train_count, test_count = split_and_link(sample, "FAKE", args.output, args.test_ratio, rng)
        print(f"FAKE/{generator}: {train_count} train / {test_count} test")


if __name__ == "__main__":
    main()
