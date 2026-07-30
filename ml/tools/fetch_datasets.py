"""Overnight dataset fetcher — built to survive the things that killed the first run.

The first attempt died after 31 minutes: an anonymous HTTP 429 from the Hub put
the download client into a bad state, and every remaining dataset then failed
instantly with "Previous task error" — ten of them in two seconds, with no retry
anywhere. This version fixes each of those causes separately:

  * one SUBPROCESS per dataset, so a poisoned client dies with it and the next
    dataset starts from a clean interpreter;
  * retry with exponential backoff inside each subprocess, with longer waits
    specifically for 429;
  * several passes over the whole queue, so anything that failed early gets
    another chance later when the rate limit has reset;
  * low concurrency (3 workers), because 8 is what triggered the throttling in
    the first place;
  * a shell watchdog outside this process that restarts it if it dies entirely.

Audit still runs after every successful download — see RAPOR.md. The auditor now
skips macOS AppleDouble files (`._name`), which ExFAT sprinkles next to every
real file and which the first version mistook for corrupt parquet.

Usage:  python fetch.py                # orchestrator
        python fetch.py --single REPO  # one dataset (used internally)
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path("/Volumes/LaCie/pixelproof-datasets")
REPORT = ROOT / "RAPOR.md"
STATE = ROOT / ".state.json"
DONE_MARKER = ROOT / ".ALL_DONE"
DISK_CAP_GB = 780
MIN_FREE_GB = 35
MAX_PASSES = 40
WORKERS = 3
BACKOFF = [30, 90, 180, 300, 600, 900, 1200]   # seconds between attempts

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Ordered so that the most valuable mid-sized sets land first: a partial night
# still leaves plenty of usable data. The two 200+ GB CommunityForensics sets go
# last precisely because they may not finish.
QUEUE = [
    ("bitmind/Nano-banana-150k", 10.6, "150k Nano Banana — 3 dosya, hızlı iner"),
    ("bitmind/nano-banana", 14.9, "Nano Banana (Gemini 2.5 Flash Image), 1024px"),
    ("theminji/AI-vs-Real-balanced", 13.0, "Dengeli gerçek/AI"),
    ("TheKernel01/AIGC-Detection-Benchmark", 32.0, "18 sınıf: GAN + difüzyon + gerçek"),
    ("theminji/ai-vs-real-200k", 51.9, "200k dengeli gerçek/AI"),
    ("ductai199x/image-manipulation-dataset-compilation", 84.1, "MANİPÜLASYON derlemesi — Module 2'nin yakıtı"),
    ("OwensLab/CommunityForensics-Small", 259.7, "4803 farklı üretici — mevcut en çeşitli set"),
    ("OwensLab/CommunityForensics-Eval", 206.2, "Community Forensics değerlendirme bölümü"),
    ("MaybeRichard/GPT-Image", 62.5, "GPT Image ham — 15596 dosya, en sona"),
    ("a3xrfgb/gpt-image-mega-4k", 8.1, "GPT Image 4K — 8002 dosya, en sona"),
]

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def log(message: str) -> None:
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {message}", flush=True)


def folder_for(repo: str) -> Path:
    return ROOT / repo.replace("/", "__")


def real_files(folder: Path, pattern: str = "*"):
    """Every file except macOS AppleDouble stubs and the HF download cache."""
    return [p for p in folder.rglob(pattern)
            if p.is_file() and not p.name.startswith("._") and ".cache" not in p.parts]


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1e9


def used_gb() -> float:
    total = 0
    for entry in ROOT.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            total += sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
    return total / 1e9


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2))


# --------------------------------------------------------------------------- #
# single download (subprocess entry point)
# --------------------------------------------------------------------------- #
def download_single(repo: str) -> int:
    from huggingface_hub import snapshot_download

    target = folder_for(repo)
    for attempt, wait in enumerate(BACKOFF, start=1):
        try:
            snapshot_download(repo_id=repo, repo_type="dataset",
                              local_dir=str(target), max_workers=WORKERS)
            log(f"OK {repo}")
            return 0
        except Exception as error:
            text = str(error)
            rate_limited = "429" in text or "Too Many Requests" in text.lower()
            if attempt >= len(BACKOFF):
                log(f"BAŞARISIZ (son deneme) {repo}: {text[:140]}")
                return 1
            delay = wait * (3 if rate_limited else 1)
            log(f"deneme {attempt} başarısız ({'429 rate limit' if rate_limited else text[:70]}), "
                f"{delay} sn bekleniyor…")
            time.sleep(delay)
    return 1


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def describe(samples: list) -> dict:
    by_label: dict[str, list] = {}
    for image, size, label in samples:
        by_label.setdefault(str(label), []).append((image.size, image.format or "?", size))
    import numpy as np

    out = {}
    for label, rows in by_label.items():
        sizes = Counter(dimensions for dimensions, _, _ in rows)
        formats = Counter(fmt for _, fmt, _ in rows)
        square = sum(1 for (w, h), _, _ in rows if w == h) / len(rows)
        bpp = [byte_size / max(w * h, 1) for (w, h), _, byte_size in rows]
        out[label] = {
            "n": len(rows),
            "formats": dict(formats),
            "distinct_sizes": len(sizes),
            "top_sizes": [f"{w}x{h}({c})" for (w, h), c in sizes.most_common(3)],
            "square_pct": round(square * 100, 1),
            "bytes_per_pixel": round(float(np.mean(bpp)), 3),
            "median_side": int(np.median([max(w, h) for (w, h), _, _ in rows])),
        }
    return out


def flag_problems(stats: dict) -> list[str]:
    """The archive1 checks: could a model split the classes without seeing content?"""
    problems = []
    if len(stats) < 2:
        return ["TEK SINIF — tek başına AUC hesaplanamaz, gerçek fotoğrafla eşleştirilmeli"]
    formats = {label: set(v["formats"]) for label, v in stats.items()}
    if all(len(f) == 1 for f in formats.values()) and len({frozenset(f) for f in formats.values()}) > 1:
        problems.append(f"FORMAT TUZAĞI — sınıflar farklı formatta: {formats}")
    squares = {label: v["square_pct"] for label, v in stats.items()}
    if max(squares.values()) - min(squares.values()) > 60:
        problems.append(f"ŞEKİL TUZAĞI — kare oranları çok farklı: {squares}")
    sides = {label: v["median_side"] for label, v in stats.items()}
    if max(sides.values()) / max(min(sides.values()), 1) > 2.5:
        problems.append(f"ÇÖZÜNÜRLÜK TUZAĞI — ortanca kenarlar çok farklı: {sides}")
    counts = [v["n"] for v in stats.values()]
    if max(counts) / max(min(counts), 1) > 3:
        problems.append(f"DENGESİZ — örneklemde sınıf oranı {max(counts)}:{min(counts)}")
    return problems


def sample_parquet(folder: Path, limit: int = 500) -> list:
    import pyarrow.parquet as pq
    from PIL import Image

    samples = []
    for path in sorted(real_files(folder, "*.parquet"))[:8]:
        try:
            parquet = pq.ParquetFile(path)
            names = parquet.schema_arrow.names
            image_col = next((n for n in names if n.lower() in
                              ("image", "img", "picture", "jpg", "png", "image_bytes")), None)
            label_col = next((n for n in names if n.lower() in
                              ("label", "label_a", "label_b", "target", "class", "is_fake",
                               "real", "generator", "model", "source", "split", "category")), None)
            if image_col is None:
                continue
            columns = [image_col] + ([label_col] if label_col else [])
            for batch in parquet.iter_batches(batch_size=50, columns=columns):
                for row in batch.to_pylist():
                    blob = row[image_col]
                    raw = blob.get("bytes") if isinstance(blob, dict) else blob
                    if not isinstance(raw, (bytes, bytearray)):
                        continue
                    try:
                        image = Image.open(io.BytesIO(raw))
                        image.load()
                    except Exception:
                        continue
                    samples.append((image, len(raw), row.get(label_col, "?") if label_col else "?"))
                    if len(samples) >= limit:
                        return samples
                break
        except Exception:
            continue
    return samples


def sample_files(folder: Path, limit: int = 500) -> list:
    from PIL import Image

    files = [p for p in real_files(folder) if p.suffix.lower() in IMAGE_EXT]
    if not files:
        return []
    step = max(1, len(files) // limit)
    samples = []
    for path in files[::step][:limit]:
        try:
            with Image.open(path) as image:
                image.load()
                parts = path.relative_to(folder).parts
                label = parts[-2] if len(parts) > 1 else "?"
                samples.append((image.copy(), path.stat().st_size, label))
        except Exception:
            continue
    return samples


def audit(folder: Path):
    samples = sample_parquet(folder)
    kind = "parquet"
    if not samples:
        samples = sample_files(folder)
        kind = "dosya"
    if not samples:
        archives = [p.suffix for p in real_files(folder) if p.suffix in {".tar", ".zip", ".gz", ".npz"}]
        return {}, [f"İNCELENEMEDİ — arşiv formatı ({Counter(archives).most_common(3)}), açılması gerekir"], "?"
    return describe(samples), flag_problems(describe(samples)), kind


def write_entry(repo: str, why: str, gb: float, stats: dict, problems: list, kind: str) -> None:
    lines = [f"\n## {repo}\n", f"- **Neden:** {why}", f"- **Boyut:** {gb:.1f} GB · örnekleme: {kind}"]
    if stats:
        lines.append("\n| sınıf | n | format | farklı boyut | en sık | kare% | ortanca kenar | bayt/piksel |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for label, v in stats.items():
            lines.append(f"| `{label}` | {v['n']} | {v['formats']} | {v['distinct_sizes']} | "
                         f"{', '.join(v['top_sizes'])} | {v['square_pct']} | {v['median_side']} | "
                         f"{v['bytes_per_pixel']} |")
    lines.append("\n**⚠️ Sorunlar:**" if problems else "\n**✅ Bariz kısayol tespit edilmedi.**")
    lines += [f"- {p}" for p in problems]
    with REPORT.open("a") as handle:
        handle.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single")
    args = parser.parse_args()
    if args.single:
        return download_single(args.single)

    ROOT.mkdir(parents=True, exist_ok=True)
    if not REPORT.exists():
        REPORT.write_text(f"# Veri Seti Raporu\n\nBaşlangıç: {datetime.now():%Y-%m-%d %H:%M}\n")

    for pass_number in range(1, MAX_PASSES + 1):
        state = load_state()
        pending = [(r, g, w) for r, g, w in QUEUE if state.get(r) != "done"]
        if not pending:
            log("HER ŞEY BİTTİ")
            DONE_MARKER.write_text(datetime.now().isoformat())
            return 0

        log(f"=== tur {pass_number} · {len(pending)} set bekliyor · "
            f"kullanılan {used_gb():.0f} GB · boş {free_gb():.0f} GB ===")

        for repo, expected_gb, why in pending:
            if free_gb() - expected_gb < MIN_FREE_GB or used_gb() + expected_gb > DISK_CAP_GB:
                log(f"disk sınırı — atlanıyor: {repo}")
                continue

            log(f"indiriliyor: {repo} (~{expected_gb:.0f} GB)")
            # Separate process: a broken HF client cannot leak into the next dataset.
            result = subprocess.run([sys.executable, __file__, "--single", repo],
                                    cwd=str(ROOT))
            if result.returncode != 0:
                log(f"bu turda olmadı, sonraki turda tekrar denenecek: {repo}")
                state = load_state()
                state[repo] = "retry"
                save_state(state)
                time.sleep(60)
                continue

            target = folder_for(repo)
            actual = sum(f.stat().st_size for f in real_files(target)) / 1e9
            log(f"indi ({actual:.1f} GB) — denetleniyor")
            try:
                stats, problems, kind = audit(target)
            except Exception:
                stats, problems, kind = {}, [f"DENETİM HATASI: {traceback.format_exc(limit=1)[:150]}"], "?"
            write_entry(repo, why, actual, stats, problems, kind)
            for problem in problems:
                log(f"   ⚠️  {problem[:110]}")
            if not problems:
                log("   ✅ temiz")
            state = load_state()
            state[repo] = "done"
            save_state(state)
            log(f"toplam {used_gb():.0f} GB · boş {free_gb():.0f} GB")

        log(f"tur {pass_number} bitti, 120 sn sonra kalanlar tekrar denenecek")
        time.sleep(120)

    log("tur limiti doldu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
