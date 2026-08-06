"""Standalone auditor — run any time, on everything already on disk.

Separate from fetch.py on purpose: the download runs for hours, and this can be
re-run against it without touching or restarting anything.

What it adds over the inline audit: it looks INSIDE zip and tar archives.
Several of these datasets ship as one big zip (Nano-banana-150k) or a couple of
hundred tars (the manipulation compilation), and the inline version could only
say "arşiv, açılması gerekir". Nothing is extracted to disk — a handful of
members are read from the archive stream and decoded in memory.

What it looks for is the archive1 failure: anything that lets a model separate
the classes WITHOUT looking at image content.
  - format split      (real=JPEG / fake=PNG was archive1's giveaway)
  - shape split       (all fakes square, all reals rectangular)
  - resolution split  (median side differing by 2.0x or more, AND a separate
                       check for p10-p90 ranges that do not overlap at all —
                       CommunityForensics is 1024px vs 512px, a ratio of exactly
                       2.0 that the old 2.5x rule waved through)
  - class imbalance
  - bytes-per-pixel   (compression level; the suspect behind the DALL-E 3
                       inversion in the tile experiment)

Usage:  python audit.py            # audit everything
        python audit.py <folder>   # audit one dataset
"""

import io
import sys
import tarfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path("/Volumes/LaCie/pixelproof-datasets")
OUTPUT = ROOT / "DENETIM.md"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LIMIT = 400


def real_files(folder: Path, pattern: str = "*"):
    """Skip macOS AppleDouble stubs (ExFAT litters these) and the HF cache."""
    return [p for p in folder.rglob(pattern)
            if p.is_file() and not p.name.startswith("._") and ".cache" not in p.parts]


def label_from_path(name: str) -> str:
    """Guess the class from a path: .../real/x.jpg -> 'real'."""
    parts = [p.lower() for p in Path(name).parts[:-1]]
    for part in reversed(parts):
        if any(key in part for key in ("real", "fake", "ai", "authentic", "synth",
                                       "genuine", "manipul", "forged", "pristine", "tamper")):
            return part
    return parts[-1] if parts else "?"


def collect(image_bytes: bytes, name: str, samples: list) -> None:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception:
        return
    samples.append((image.size, image.format or "?", len(image_bytes), label_from_path(name)))


def from_archives(folder: Path, samples: list) -> str | None:
    """Read a few members out of zip/tar archives without extracting them."""
    used = None
    for path in sorted(real_files(folder))[:40]:
        if len(samples) >= LIMIT:
            break
        try:
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    members = [m for m in archive.namelist()
                               if Path(m).suffix.lower() in IMAGE_EXT and not Path(m).name.startswith("._")]
                    step = max(1, len(members) // 120)
                    for member in members[::step][:120]:
                        collect(archive.read(member), member, samples)
                    used = "zip"
            elif path.suffix in {".tar", ".gz", ".tgz"} or path.name.endswith(".tar.gz"):
                mode = "r:gz" if path.suffix in {".gz", ".tgz"} else "r"
                with tarfile.open(path, mode) as archive:
                    taken = 0
                    for member in archive:
                        if taken >= 60 or len(samples) >= LIMIT:
                            break
                        if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_EXT:
                            continue
                        if Path(member.name).name.startswith("._"):
                            continue
                        handle = archive.extractfile(member)
                        if handle:
                            collect(handle.read(), member.name, samples)
                            taken += 1
                    used = "tar"
        except Exception:
            continue
    return used


def label_direction(folder: Path) -> str | None:
    """Read the dataset's own ClassLabel order and compare it with ours.

    This project uses 0 = real, 1 = AI. Datasets do not agree: two of the five
    training sources declare the opposite, and because nothing checked, 47% of
    the pool carried inverted labels through four experiments (E12, E14, E15,
    E16). The five existing checks all look at pixels and file properties — none
    of them can see that a label MEANS the opposite thing. This one can.
    """
    import json
    import pyarrow.parquet as pq

    shards = sorted(real_files(folder, "*.parquet"))
    if not shards:
        return None
    try:
        metadata = (pq.ParquetFile(shards[0]).schema_arrow.metadata or {}).get(b"huggingface")
        if not metadata:
            return None
        features = json.loads(metadata)["info"]["features"]
    except Exception:
        return None

    for column, spec in features.items():
        names = spec.get("names") if isinstance(spec, dict) else None
        if not names or len(names) != 2:
            continue
        first = str(names[0]).lower()
        # index 0 should mean "real" under our convention
        if any(token in first for token in ("ai", "fake", "synth", "generated", "gan")):
            return (f"ETİKET YÖNÜ TERS — `{column}` = {names}, yani 0=AI/1=gerçek. "
                    f"Projede 0=gerçek, 1=AI. Bu sette `label_map` ZORUNLU "
                    f"(build_pool.py SOURCES)")
        if any(token in first for token in ("real", "authentic", "natural", "photo")):
            return None
    return None


def from_parquet(folder: Path, samples: list) -> str | None:
    import pyarrow.parquet as pq

    used = None
    # Spread the sample across the WHOLE shard range, never just the first few.
    # theminji/ai-vs-real-200k is sorted by label — shards 0-133 are class 0 and
    # 134-267 are class 1 — so reading the first 8 files reported "single class"
    # for a perfectly balanced dataset.
    shards = sorted(real_files(folder, "*.parquet"))
    picked = shards[::max(1, len(shards) // 12)][:12] if len(shards) > 12 else shards
    for path in picked:
        if len(samples) >= LIMIT:
            break
        try:
            parquet = pq.ParquetFile(path)
            names = parquet.schema_arrow.names
            image_col = next((n for n in names if n.lower() in
                              ("image", "img", "picture", "jpg", "png", "image_bytes", "image_data")), None)
            label_col = next((n for n in names if n.lower() in
                              ("label", "label_a", "label_b", "target", "class", "is_fake",
                               "real", "generator", "model", "source", "category", "type")), None)
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
                    label = str(row.get(label_col, "?")) if label_col else "?"
                    samples.append((image.size, image.format or "?", len(raw), label))
                    if len(samples) >= LIMIT:
                        break
                break
            used = "parquet"
        except Exception:
            continue
    return used


def from_loose_images(folder: Path, samples: list) -> str | None:
    files = [p for p in real_files(folder) if p.suffix.lower() in IMAGE_EXT]
    if not files:
        return None
    step = max(1, len(files) // LIMIT)
    for path in files[::step][:LIMIT]:
        try:
            with Image.open(path) as image:
                image.load()
                samples.append((image.size, image.format or "?", path.stat().st_size,
                                label_from_path(str(path.relative_to(folder)))))
        except Exception:
            continue
    return "dosya"


def summarise(samples: list) -> dict:
    by_label: dict[str, list] = {}
    for size, fmt, byte_size, label in samples:
        by_label.setdefault(label, []).append((size, fmt, byte_size))
    out = {}
    for label, rows in by_label.items():
        sizes = Counter(s for s, _, _ in rows)
        out[label] = {
            "n": len(rows),
            "formats": dict(Counter(f for _, f, _ in rows)),
            "distinct": len(sizes),
            "top": ", ".join(f"{w}x{h}({c})" for (w, h), c in sizes.most_common(3)),
            "square": round(sum(1 for (w, h), _, _ in rows if w == h) / len(rows) * 100, 1),
            "side": int(np.median([max(w, h) for (w, h), _, _ in rows])),
            # p10/p90 exist so the ratio-of-medians check can be backed by a
            # separation check — see problems_in(). A median ratio hides the
            # difference between "overlapping distributions" and "two disjoint
            # constants", and only the second is a perfect shortcut.
            "p10": int(np.percentile([max(w, h) for (w, h), _, _ in rows], 10)),
            "p90": int(np.percentile([max(w, h) for (w, h), _, _ in rows], 90)),
            "bpp": round(float(np.mean([b / max(w * h, 1) for (w, h), _, b in rows])), 3),
        }
    return out


def problems_in(stats: dict) -> list[str]:
    if len(stats) < 2:
        return ["TEK SINIF — kendi başına AUC hesaplanamaz; gerçek fotoğrafla eşleştirilmeli"]
    issues = []
    formats = {k: set(v["formats"]) for k, v in stats.items()}
    if all(len(f) == 1 for f in formats.values()) and len({frozenset(f) for f in formats.values()}) > 1:
        issues.append(f"FORMAT TUZAĞI — sınıflar farklı formatta: {formats}")
    squares = {k: v["square"] for k, v in stats.items()}
    if max(squares.values()) - min(squares.values()) > 60:
        issues.append(f"ŞEKİL TUZAĞI — kare oranları: {squares}")
    sides = {k: v["side"] for k, v in stats.items()}
    # Threshold lowered 2.5 -> 2.0 (inclusive) on 2026-08-05. CommunityForensics
    # is 1024px for one class and 512px for the other — a ratio of EXACTLY 2.0,
    # which the old rule let through and which no model can fail to exploit.
    if max(sides.values()) / max(min(sides.values()), 1) >= 2.0:
        issues.append(f"ÇÖZÜNÜRLÜK TUZAĞI — ortanca kenarlar: {sides}")
    # The sharper test: if the two size distributions do not overlap at all, the
    # ratio is irrelevant — a single threshold on image size separates the
    # classes perfectly. This is archive1's AUC 1.000 in a different costume.
    spans = {k: (v["p10"], v["p90"]) for k, v in stats.items()}
    lows, highs = list(spans.values())[0], list(spans.values())[1]
    if lows[0] > highs[1] or highs[0] > lows[1]:
        issues.append(f"ÇÖZÜNÜRLÜK AYRIMI (KESİN) — p10-p90 aralıkları hiç örtüşmüyor: {spans}. "
                      f"Boyuta bakan tek bir eşik sınıfları kusursuz ayırır")
    bpps = {k: v["bpp"] for k, v in stats.items()}
    if max(bpps.values()) / max(min(bpps.values()), 0.001) > 3:
        issues.append(f"SIKIŞTIRMA TUZAĞI — bayt/piksel çok farklı: {bpps} "
                      f"(DALL-E 3 tersine dönmesinin şüphelisi buydu)")
    counts = [v["n"] for v in stats.values()]
    if max(counts) / max(min(counts), 1) > 3:
        issues.append(f"DENGESİZ — örneklem oranı {max(counts)}:{min(counts)}")
    return issues


def audit_folder(folder: Path) -> tuple[dict, list[str], str]:
    samples: list = []
    kind = from_parquet(folder, samples)
    if not samples:
        kind = from_archives(folder, samples)
    if not samples:
        kind = from_loose_images(folder, samples)
    if not samples:
        return {}, ["İNCELENEMEDİ — tanınan görsel/arşiv bulunamadı"], "?"
    stats = summarise(samples)
    issues = problems_in(stats)
    flipped = label_direction(folder)
    if flipped:
        issues.insert(0, flipped)      # first, because it silently corrupts everything else
    return stats, issues, kind or "?"


def main() -> None:
    targets = ([Path(sys.argv[1])] if len(sys.argv) > 1
               else sorted(d for d in ROOT.iterdir() if d.is_dir() and not d.name.startswith(".")))
    lines = [f"# Denetim Raporu\n", f"Oluşturma: {datetime.now():%Y-%m-%d %H:%M}\n",
             "Aranan şey: bir modelin **görsele bakmadan** sınıfları ayırmasına izin veren "
             "her şey. archive1'de bunu altı deney boyunca kaçırmıştık.\n",
             "`⚠️` = kullanılamaz demek değil; **kusuru bilinerek** kullanılır demek.\n"]

    for folder in targets:
        size = sum(f.stat().st_size for f in real_files(folder)) / 1e9
        print(f"denetleniyor: {folder.name} ({size:.1f} GB)…", flush=True)
        try:
            stats, issues, kind = audit_folder(folder)
        except Exception as error:
            stats, issues, kind = {}, [f"HATA: {str(error)[:120]}"], "?"
        lines.append(f"\n## {folder.name}\n")
        lines.append(f"- {size:.1f} GB · örnekleme: {kind}")
        if stats:
            lines.append("\n| sınıf | n | format | farklı boyut | en sık | kare% | "
                         "kenar p10 | ortanca | p90 | bayt/piksel |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for label, v in sorted(stats.items()):
                lines.append(f"| `{label}` | {v['n']} | {v['formats']} | {v['distinct']} | "
                             f"{v['top']} | {v['square']} | {v['p10']} | {v['side']} | "
                             f"{v['p90']} | {v['bpp']} |")
        if issues:
            lines.append("\n**⚠️ Sorunlar:**")
            lines += [f"- {i}" for i in issues]
            for i in issues:
                print(f"   ⚠️  {i[:100]}", flush=True)
        else:
            lines.append("\n**✅ Bariz kısayol tespit edilmedi.**")
            print("   ✅ temiz", flush=True)

    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"\nyazıldı: {OUTPUT}")


if __name__ == "__main__":
    main()
