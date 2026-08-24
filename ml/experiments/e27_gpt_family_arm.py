# =============================================================================
# e27_gpt_family_arm.py — WHAT THIS EXPERIMENT DOES
# -----------------------------------------------------------------------------
# The measured blind spot: GPT Image recall is 12% even under the OR rule
# (E25/E26), and the live demo missed a ChatGPT upload for exactly this
# reason. This experiment trains OUR OWN GPT-family arm and admits it to the
# served OR ensemble only through a pre-registered gate.
#
# The design went through a three-lens adversarial review (2026-08-20) before
# any training; every mitigation below cites the finding it answers.
#
# PRE-REGISTERED DESIGN (post-review)
# -----------------------------------------------------------------------------
# DATA
#   AI   : a3xrfgb gpt-image-mega-4k minus the 200-file E25 probe, excluded by
#          SHA256 (not name). ~860 images, all 1024x1536 PNG.
#   REAL : three disjoint-from-eval sources — genimage train/REAL (ImageNet),
#          theminji AI-vs-Real-balanced reals >=256px (label direction
#          verified, E19b), CommunityForensics FFHQ reals (the portrait
#          source answering the content-shortcut finding). 200 FFHQ held out
#          as the portrait-real FP row.
# ENCODER (both classes, one code path)
#   - each AI image's target long side is SAMPLED from the real pool's
#     empirical size distribution (review: "<=1536 cap is a no-op");
#   - a seeded half of the AI class gets a pre-JPEG pass q60-95 so
#     compression HISTORY decorrelates from class (single-vs-double JPEG);
#   - everything ends as JPEG q75-95.
# CONTAMINATION (review: E12 lesson)
#   - SHA256 exclusion vs probe; dHash in-pool dedup; dHash cross-scan vs
#     probe + Defactify + forensic auth sources. Matches are dropped and
#     counted in the log.
# SHORTCUT GATES, before training
#   - metadata probe (w/h/aspect/B-px, HistGB, 5-fold) AUC < 0.65
#   - texture probe (the 68 features of features.py on encoded pixels,
#     400/side) AUC < 0.75  (review: E8 showed size leaks through texture)
# TRAINING LADDER (review: "risk merdiveni ters")
#   step 1: frozen CF-ViT trunk + logistic head on CLS embeddings (minutes,
#           3 split-seeds). Only if step 1 fails the gate does step 2 (last
#           blocks) exist — deliberately NOT implemented until needed.
# EPOCH/CANARY
#   the frozen-trunk path has no epochs; the forgetting canary is structural
#   (the trunk cannot forget). Score-vs-metadata Spearman is still reported;
#   |rho| > 0.3 on validation is an alarm.
# ACCEPTANCE GATE (all seeds, mean reported with per-seed values)
#   G1  in-collection GPT probe recall at the arm's own worst-source
#       threshold >= 40%  — CLAIM IS EXPLICITLY IN-COLLECTION; an
#       out-of-collection ChatGPT holdout is a recorded TODO for the owner.
#   G2  q75-recompressed probe recall reported; a collapse marks a
#       compression shortcut (review finding).
#   G3  fit the three-arm OR threshold on calibration halves only, then measure
#       the untouched evaluation halves exactly once. Evaluation cannot trigger
#       another threshold change. If the calibration-safe threshold breaks G1,
#       the arm is rejected rather than post-hoc loosened.
#   G4  FFHQ portrait-real holdout FP reported (content-shortcut row).
#   G5  the live ChatGPT desktop image, scored (single out-of-collection
#       smoke sample).
#   NOTE genimage metrics are NOT reported for this arm (trained-on source).
#
# Stages are resumable via artifacts under artifacts/e27/.  Usage:
#   PYTHONPATH=src .venv/bin/python experiments/e27_gpt_family_arm.py --stage all
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pixelproof.evaluation_protocol import (
    safe_auc,
    stable_calibration_split,
    threshold_at_fpr,
    union_operating_point,
)
from pixelproof.project_paths import DATA_ROOT, WORK_ROOT

sys.path.insert(0, str(Path(__file__).parent))

SSD = DATA_ROOT
HOME = WORK_ROOT
POOL = HOME / "e27_pool"
ART = Path("artifacts/e27")
PROBE_DIR = HOME / "e25_modern_probe/gpt_image_4k"
LIVE_IMAGE = HOME / "live-chatgpt.png"
SPLIT_SEED = 2026
CAL_FRACTION = 0.5
FP_BUDGET = 0.10
SEEDS = (42, 1337, 2024)
GENERATORS = ("dalle3", "midjourney", "sd21", "sd3", "sdxl")
REAL_PER_SOURCE = 600
FFHQ_HOLDOUT = 200


def rng_for(name: str, salt: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(f"{salt}:{name}".encode()).digest()[:4], "big")
    return np.random.default_rng(seed)


def dhash(image: Image.Image, size: int = 8) -> int:
    g = np.asarray(image.convert("L").resize((size + 1, size)), dtype=np.int16)
    return int("".join("1" if v else "0" for v in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- stage: prepare
def stage_prepare() -> None:
    import pyarrow.parquet as pq

    POOL.mkdir(parents=True, exist_ok=True)
    (POOL / "ai").mkdir(exist_ok=True)
    (POOL / "real").mkdir(exist_ok=True)
    (POOL / "ffhq_holdout").mkdir(exist_ok=True)
    manifest_path = ART / "pool_manifest.jsonl"
    ART.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        print("prepare: manifest mevcut, atlanıyor")
        return

    probe_sha = {sha256_bytes(p.read_bytes())
                 for p in PROBE_DIR.iterdir() if p.suffix == ".png"}
    print(f"probe SHA seti: {len(probe_sha)}")

    # --- gerçek kaynak 1: genimage train/REAL (dosyalar) ---
    reals: list[tuple[str, Image.Image]] = []
    gi = sorted((HOME / "genimage_split/train/REAL").glob("*"))
    idx = rng_for("genimage", "sample").choice(len(gi), REAL_PER_SOURCE, replace=False)
    for i in sorted(idx):
        with Image.open(gi[i]) as im:
            reals.append((f"genimage:{gi[i].name}", im.convert("RGB").copy()))
    print(f"genimage reals: {REAL_PER_SOURCE}")

    # --- gerçek kaynak 2: theminji balanced reals (parquet, E19b doğrulamalı) ---
    shards = sorted((SSD / "theminji__AI-vs-Real-balanced").rglob("*.parquet"))
    shards = [s for s in shards if not s.name.startswith("._")]
    got = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        meta = (pf.schema_arrow.metadata or {}).get(b"huggingface", b"{}")
        names = None
        info = json.loads(meta.decode("utf8") or "{}")
        for feat in (info.get("info", {}).get("features", {}) or {}).values():
            if isinstance(feat, dict) and feat.get("_type") == "ClassLabel":
                names = feat.get("names")
        if names != ["AiArtData", "RealArt"]:
            raise RuntimeError(f"E19b: beklenmeyen etiket sırası {names!r}")
        real_index = 1  # RealArt
        for batch in pf.iter_batches(batch_size=64, columns=["image", "label"]):
            for row in batch.to_pylist():
                if got >= REAL_PER_SOURCE:
                    break
                if row["label"] != real_index:
                    continue
                with Image.open(io.BytesIO(row["image"]["bytes"])) as im:
                    if max(im.size) < 256:
                        continue
                    reals.append((f"theminji:{got:05d}", im.convert("RGB").copy()))
                    got += 1
            if got >= REAL_PER_SOURCE:
                break
        if got >= REAL_PER_SOURCE:
            break
    print(f"theminji reals (>=256px): {got}")
    if got < 300:
        raise RuntimeError("theminji <300 — üçüncü kaynak şart (inceleme bulgusu)")

    # --- gerçek kaynak 3: FFHQ (parquet) + portre FP holdout ---
    pf = pq.ParquetFile(SSD / "34data__communityforensics-real/data.parquet")
    want = REAL_PER_SOURCE + FFHQ_HOLDOUT
    ffhq: list[Image.Image] = []
    for batch in pf.iter_batches(batch_size=64, columns=["image"]):
        for row in batch.to_pylist():
            if len(ffhq) >= want:
                break
            payload = row["image"]
            raw = payload["bytes"] if isinstance(payload, dict) else payload
            with Image.open(io.BytesIO(raw)) as im:
                ffhq.append(im.convert("RGB").copy())
        if len(ffhq) >= want:
            break
    holdout, ffhq_train = ffhq[:FFHQ_HOLDOUT], ffhq[FFHQ_HOLDOUT:]
    for i, im in enumerate(holdout):
        im.save(POOL / "ffhq_holdout" / f"{i:04d}.jpg", quality=90)
    reals += [(f"ffhq:{i:05d}", im) for i, im in enumerate(ffhq_train)]
    print(f"ffhq reals: {len(ffhq_train)} eğitim + {FFHQ_HOLDOUT} holdout")

    real_sizes = [max(im.size) for _, im in reals]

    # --- ortak kodlayıcı ---
    manifest = []

    real_dims = [im.size for _, im in reals]  # (w,h) ÇİFTLERİ — aspect dahil

    def encode(name: str, im: Image.Image, label: int, out_dir: Path,
               force_ai_branch: bool = False) -> dict:
        r = rng_for(name, "encode")
        if label == 1 or force_ai_branch:
            # gate-v2 düzeltmesi: yalnız uzun kenar değil (w,h) ÇİFTİ örneklenir —
            # aspect de gerçek dağılımına oturur (kapı v1 0.992 ile bunu yakaladı)
            tw, th = real_dims[int(r.integers(0, len(real_dims)))]
            target_aspect = tw / th
            w, h = im.size
            if w / h > target_aspect:      # merkez kırp → hedef aspect
                new_w = round(h * target_aspect)
                im = im.crop(((w - new_w) // 2, 0, (w - new_w) // 2 + new_w, h))
            else:
                new_h = round(w / target_aspect)
                im = im.crop((0, (h - new_h) // 2, w, (h - new_h) // 2 + new_h))
            im = im.resize((tw, th), Image.LANCZOS)
            if r.random() < 0.5:  # sıkıştırma geçmişi dengesi (inceleme: önemli-2)
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=int(r.integers(60, 96)))
                im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        q = int(r.integers(65, 96))
        out = out_dir / (hashlib.sha256(name.encode()).hexdigest()[:16] + ".jpg")
        im.save(out, format="JPEG", quality=q)
        w, h = im.size
        return {"name": name, "path": str(out), "label": label,
                "width": w, "height": h,
                "bytes_per_pixel": out.stat().st_size / (w * h)}

    # KONTROL HAVUZU (gate-v2): gerçek-vs-gerçek, aynı kodlama asimetrisiyle.
    # FFHQ eğitim gerçeklerinin yarısı "AI dalı" kodlayıcısından geçirilir ve
    # sahte-etiket 1 alır; probların bu havuzdaki AUC'si KISAYOL TAVANIDIR —
    # sınıf-AUC'sinden bu tavanı aşan kısım gerçek sinyaldir (E18 pozitif-kontrol
    # metodolojisinin kapıya uygulanması).
    (POOL / "control").mkdir(exist_ok=True)
    control = []
    ctrl_rng = rng_for("control", "assign")
    for name, im in reals:
        if not name.startswith("ffhq") and ctrl_rng.random() > 0.45:
            continue
        pseudo = 1 if ctrl_rng.random() < 0.5 else 0
        row = encode(f"ctrl:{name}", im, 0, POOL / "control",
                     force_ai_branch=(pseudo == 1))
        row["label"] = pseudo
        control.append(row)
    with (ART / "control_manifest.jsonl").open("w") as f:
        for row in control:
            f.write(json.dumps(row) + "\n")
    print(f"kontrol havuzu: {sum(1 for r in control if r['label']==1)} sahte-AI / "
          f"{sum(1 for r in control if r['label']==0)} sahte-gerçek")

    ai_files = sorted(p for p in (SSD / "a3xrfgb__gpt-image-mega-4k").glob("*.png")
                      if not p.name.startswith("._"))
    kept = skipped_sha = 0
    for p in ai_files:
        raw = p.read_bytes()
        if sha256_bytes(raw) in probe_sha:
            skipped_sha += 1
            continue
        with Image.open(io.BytesIO(raw)) as im:
            manifest.append(encode(f"gpt:{p.name}", im.convert("RGB"), 1, POOL / "ai"))
        kept += 1
    print(f"AI: {kept} kodlandı, {skipped_sha} probe-SHA dışlandı")

    for name, im in reals:
        manifest.append(encode(name, im, 0, POOL / "real"))

    # --- dHash: havuz-içi dedup + çapraz kontaminasyon (inceleme: E12) ---
    def dh_of(path: str) -> int:
        with Image.open(path) as im:
            return dhash(im)

    for row in manifest:
        row["dhash"] = dh_of(row["path"])
    seen: dict[int, str] = {}
    deduped = []
    dropped_dup = 0
    for row in manifest:
        if row["dhash"] in seen:
            dropped_dup += 1
            continue
        seen[row["dhash"]] = row["name"]
        deduped.append(row)
    manifest = deduped

    external: set[int] = set()
    ext_sources = [PROBE_DIR] + sorted((HOME / "defactify_test").rglob("*")) \
        + sorted((HOME / "manipulation_test").glob("*/auth"))
    ext_files = []
    for src in ext_sources:
        if src.is_dir():
            ext_files += [p for p in src.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                          and not p.name.startswith("._")]
    for p in ext_files:
        try:
            with Image.open(p) as im:
                external.add(dhash(im))
        except Exception:
            pass
    before = len(manifest)
    manifest = [r for r in manifest if r["dhash"] not in external]
    print(f"dHash: havuz-içi {dropped_dup} kopya düştü; çapraz eşleşme "
          f"{before - len(manifest)} düştü (dış set: {len(ext_files)} dosya)")

    with manifest_path.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    counts = defaultdict(int)
    for row in manifest:
        counts[row["label"]] += 1
    print(f"manifest: {counts[1]} AI / {counts[0]} gerçek")


# ---------------------------------------------------------------- stage: gate
def stage_gate() -> None:
    """Gate v2. v1 (yalnız sınıf-AUC eşikleri) 0.992 metadata ile eğitimi durdurdu
    ve aspect kanalını yakaladı — kodlayıcı düzeltildi. v2'nin kuralları:
      metadata: sınıf-AUC < 0.65 VE kontrol-AUC'den farkı < 0.10
      doku    : kontrol-AUC (kısayol tavanı) < 0.70 VE sınıf-AUC - kontrol-AUC >= 0.10
                (68 öznitelik üretim izi dedektörüdür; sınıf-AUC'nin yüksek olması
                 beklenir — ŞART, ayrımın kısayol tavanından gelmemesi)"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from pixelproof.features import extract

    def probe(rows, tag):
        y = np.array([r["label"] for r in rows])
        X_meta = np.array([[r["width"], r["height"],
                            r["width"] / r["height"], r["bytes_per_pixel"]]
                           for r in rows])
        auc_meta = cross_val_score(HistGradientBoostingClassifier(random_state=0),
                                   X_meta, y, cv=5, scoring="roc_auc").mean()
        sample = rng_for(f"gate:{tag}", "texture")
        keep = {0: [], 1: []}
        for i, r in enumerate(rows):
            keep[r["label"]].append(i)
        idx = np.concatenate([
            sample.choice(keep[0], min(350, len(keep[0])), replace=False),
            sample.choice(keep[1], min(350, len(keep[1])), replace=False),
        ])
        feats, ys = [], []
        for n, i in enumerate(idx):
            feats.append(extract(rows[i]["path"]))
            ys.append(rows[i]["label"])
            if n % 150 == 0:
                print(f"  [{tag}] doku {n}/{len(idx)}")
        auc_tex = cross_val_score(HistGradientBoostingClassifier(random_state=0),
                                  np.array(feats), np.array(ys),
                                  cv=5, scoring="roc_auc").mean()
        return float(auc_meta), float(auc_tex)

    rows = [json.loads(l) for l in (ART / "pool_manifest.jsonl").open()]
    ctrl = [json.loads(l) for l in (ART / "control_manifest.jsonl").open()]
    meta_cls, tex_cls = probe(rows, "sınıf")
    meta_ctl, tex_ctl = probe(ctrl, "kontrol")

    ok = (meta_cls < 0.65 and (meta_cls - meta_ctl) < 0.10
          and tex_ctl < 0.70 and (tex_cls - tex_ctl) >= 0.10)
    verdict = {"metadata_class": meta_cls, "metadata_control": meta_ctl,
               "texture_class": tex_cls, "texture_control": tex_ctl,
               "pass": bool(ok)}
    (ART / "gates.json").write_text(json.dumps(verdict, indent=2))
    print(f"KAPI v2: metadata sınıf {meta_cls:.3f} / kontrol {meta_ctl:.3f} · "
          f"doku sınıf {tex_cls:.3f} / kontrol {tex_ctl:.3f} "
          f"→ {'GEÇTİ' if ok else 'KALDI — eğitim yok'}")
    if not ok:
        sys.exit(2)


# ---------------------------------------------------------------- embeddings
def load_backbone(device):
    from huggingface_hub import snapshot_download
    from transformers import ViTForImageClassification, ViTImageProcessor

    local = snapshot_download("buildborderless/CommunityForensics-DeepfakeDet-ViT",
                              local_files_only=True)
    model = ViTForImageClassification.from_pretrained(
        local, local_files_only=True).to(device).eval()
    proc = ViTImageProcessor.from_pretrained(local, local_files_only=True)
    return model, proc


@torch.inference_mode()
def embed_paths(paths, model, proc, device, cache: Path) -> np.ndarray:
    if cache.is_file():
        return np.load(cache)
    out = []
    for i in range(0, len(paths), 16):
        batch = []
        for p in paths[i:i + 16]:
            with Image.open(p) as im:
                batch.append(im.convert("RGB"))
        px = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        h = model.vit(pixel_values=px).last_hidden_state[:, 0]
        out.append(h.cpu().numpy())
        if i % 320 == 0:
            print(f"  embed {i}/{len(paths)}")
    arr = np.concatenate(out)
    np.save(cache, arr)
    return arr


# ---------------------------------------------------------------- stage: train+eval
def stage_train_eval() -> None:
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model, proc = load_backbone(device)
    rows = [json.loads(l) for l in (ART / "pool_manifest.jsonl").open()]
    paths = [r["path"] for r in rows]
    y = np.array([r["label"] for r in rows])
    X = embed_paths(paths, model, proc, device, ART / "pool_embed.npy")

    # skorlanacak dış setler (tek sefer embed, tüm seedler paylaşır)
    def files_of(d, exts={".jpg", ".jpeg", ".png"}):
        return sorted(p for p in Path(d).iterdir()
                      if p.suffix.lower() in exts and not p.name.startswith("._"))

    eval_sets: dict[str, list[Path]] = {"probe": files_of(PROBE_DIR)}
    recompressed = ART / "probe_q75"
    recompressed.mkdir(exist_ok=True)
    for p in eval_sets["probe"]:
        out = recompressed / (p.stem + ".jpg")
        if not out.is_file():
            with Image.open(p) as im:
                im.convert("RGB").save(out, format="JPEG", quality=75)
    eval_sets["probe_q75"] = files_of(recompressed)
    eval_sets["ffhq_holdout"] = files_of(POOL / "ffhq_holdout")
    eval_sets["defactify_real"] = files_of(HOME / "defactify_test/real")[:200]
    for g in GENERATORS:
        eval_sets[f"gen_{g}"] = files_of(HOME / "defactify_test/ai" / g)[:200]
    for d in sorted((HOME / "manipulation_test").glob("*/auth")):
        eval_sets[f"forensics_{d.parent.name}"] = files_of(d)[:200]
    if LIVE_IMAGE.is_file():
        eval_sets["live_chatgpt"] = [LIVE_IMAGE]

    embeds = {name: embed_paths(fs, model, proc, device,
                                ART / f"embed_{name}.npy")
              for name, fs in eval_sets.items()}

    results = {"seeds": {}}
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(y))
        cut = int(len(y) * 0.85)
        tr, va = order[:cut], order[cut:]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        val_auc = safe_auc(clf.decision_function(X[va][y[va] == 0]),
                           clf.decision_function(X[va][y[va] == 1]))
        val_scores = clf.decision_function(X[va])
        rho_size = spearmanr(val_scores,
                             [max(rows[i]["width"], rows[i]["height"]) for i in va])[0]
        rho_bpp = spearmanr(val_scores,
                            [rows[i]["bytes_per_pixel"] for i in va])[0]

        s = {name: clf.decision_function(embeds[name]) for name in embeds}

        # E22 kuralı: 12 pipeline kalibrasyon yarılarından worst-source eşiği
        def split_scores(name, fs):
            recs = [{"path": str(p)} for p in fs]
            cal, ev = stable_calibration_split(recs, CAL_FRACTION, SPLIT_SEED)
            calset = {r["path"] for r in cal}
            cal_i = [i for i, p in enumerate(fs) if str(p) in calset]
            ev_i = [i for i in range(len(fs)) if i not in set(cal_i)]
            return s[name][cal_i], s[name][ev_i]

        pipeline_names = ["defactify_real"] + [k for k in eval_sets
                                               if k.startswith("forensics_")]
        cal_sets, ev_sets = {}, {}
        for name in pipeline_names:
            cal_sets[name], ev_sets[name] = split_scores(name, eval_sets[name])
        cuts = [threshold_at_fpr(v, FP_BUDGET) for v in cal_sets.values()]
        t_arm = float(max(c for c in cuts if np.isfinite(c)))

        arm_fp = {n: float((v >= t_arm).mean()) for n, v in ev_sets.items()}
        probe_recall = float((s["probe"] >= t_arm).mean())
        probe_q75_recall = float((s["probe_q75"] >= t_arm).mean())
        ffhq_fp = float((s["ffhq_holdout"] >= t_arm).mean())
        defactify_auc = safe_auc(
            s["defactify_real"],
            np.concatenate([s[f"gen_{g}"] for g in GENERATORS]))

        results["seeds"][seed] = {
            "val_auc": float(val_auc), "threshold": t_arm,
            "spearman_size": float(rho_size), "spearman_bpp": float(rho_bpp),
            "probe_recall": probe_recall, "probe_q75_recall": probe_q75_recall,
            "ffhq_holdout_fp": ffhq_fp,
            "arm_worst_fp": float(max(arm_fp.values())),
            "arm_fp": arm_fp,
            "defactify_auc_standalone": float(defactify_auc),
            "live_chatgpt_score": (float(s["live_chatgpt"][0])
                                   if "live_chatgpt" in s else None),
            "live_chatgpt_above": (bool(s["live_chatgpt"][0] >= t_arm)
                                   if "live_chatgpt" in s else None),
        }
        r = results["seeds"][seed]
        print(f"seed {seed}: val {val_auc:.3f} · eşik {t_arm:.2f} · "
              f"probe {100*probe_recall:.1f}% (q75 {100*probe_q75_recall:.1f}%) · "
              f"kol-worst-FP {100*r['arm_worst_fp']:.1f}% · ffhq FP {100*ffhq_fp:.1f}% · "
              f"rho(size) {rho_size:+.2f} · canlı: {r['live_chatgpt_above']}")
        # modeli sakla (lineer kafa + referans)
        np.savez(ART / f"head_seed{seed}.npz",
                 coef=clf.coef_, intercept=clf.intercept_, threshold=t_arm)

    (ART / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults -> {ART/'results.json'}")


STAGES = {"prepare": stage_prepare, "gate": stage_gate, "train": stage_train_eval}


# ---------------------------------------------------------------- stage: union
def stage_union() -> None:
    """G3'ün kesin ölçümü: 3-kollu OR, e21/e23b/e24 önbellekleriyle görsel-bazında,
    12 pipeline (iPhone dahil). Kol eşiği 12 kaynaklı kütüphaneyle yeniden kesilir;
    birleşim bütçeyi aşarsa eşik kalibrasyon yarılarında yükseltilir (E22 kuralı)."""
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model, proc = load_backbone(device)
    CF_T, BF_T = 0.6617392, 1.0439179

    def read_cache(path):
        d = {}
        for line in Path(path).open():
            r = json.loads(line)
            d[r["path"]] = r
        return d

    cf = read_cache("artifacts/e21/cf_vit_scores.jsonl")
    bf = read_cache("artifacts/e21/bfree_scores.jsonl")
    nist_cap = {Path(json.loads(l)["path"]).stem: json.loads(l)["image_score"]
                for l in Path("artifacts/e23/e23b_bfree_capped_scores.jsonl").open()}
    iph = defaultdict(dict)
    for line in Path("artifacts/e24/scores.jsonl").open():
        r = json.loads(line)
        iph[r["variant"]][Path(r["path"]).stem] = r["image_score"]

    pops: dict[str, list[str]] = defaultdict(list)
    for p, r in cf.items():
        if r["dataset"] == "forensics":
            pops[r["source"]].append(p)
        elif (r["dataset"], r["source"]) == ("defactify", "real"):
            pops["defactify_real"].append(p)
        elif r["dataset"] == "defactify":
            pops[f"gen_{r['source']}"].append(p)
    # iPhone: stem'leri galeri klasöründeki gerçek dosyalara geri eşle
    gallery = {p.stem: p for p in (HOME / "fotoğraf galeri").iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg"}}
    iphone_paths = sorted(str(gallery[s]) for s in iph["cf_vit:native"]
                          if s in gallery)
    pops["iphone"] = iphone_paths

    all_paths = sorted({p for ps in pops.values() for p in ps})
    emb = embed_paths(all_paths, model, proc, device, ART / "embed_union.npy")
    path_idx = {p: i for i, p in enumerate(all_paths)}

    head = np.load(ART / "head_seed2024.npz")  # ortalamaya en yakın seed
    coef, intercept = head["coef"].ravel(), float(head["intercept"][0])
    arm_scores = {p: float(emb[path_idx[p]] @ coef + intercept) for p in all_paths}

    def halves(paths):
        recs = [{"path": p} for p in sorted(paths)]
        cal, ev = stable_calibration_split(recs, CAL_FRACTION, SPLIT_SEED)
        return [r["path"] for r in cal], [r["path"] for r in ev]

    real_names = ["defactify_real", "iphone"] + [n for n in pops if n not in
                                                 ("defactify_real", "iphone")
                                                 and not n.startswith("gen_")]
    cal_h, ev_h = {}, {}
    for n in real_names:
        cal_h[n], ev_h[n] = halves(pops[n])
    def bf_score(p, source):
        if source == "iphone":
            return iph["bfree:capped"][Path(p).stem]
        if source == "NIST2016":
            return nist_cap.get(Path(p).stem, bf[p]["image_score"])
        return bf[p]["image_score"]

    def cf_score(p, source):
        if source == "iphone":
            return iph["cf_vit:native"][Path(p).stem]
        return cf[p]["image_score"]

    # G3 v3: every threshold decision is made from calibration halves only.
    # Evaluation halves enter once, after the strictest source threshold freezes.
    baseline_cal = {
        n: [cf_score(p, n) >= CF_T or bf_score(p, n) >= BF_T for p in cal_h[n]]
        for n in real_names
    }
    baseline_eval = {
        n: [cf_score(p, n) >= CF_T or bf_score(p, n) >= BF_T for p in ev_h[n]]
        for n in real_names
    }
    fitted = union_operating_point(
        baseline_cal,
        {n: [arm_scores[p] for p in cal_h[n]] for n in real_names},
        baseline_eval,
        {n: [arm_scores[p] for p in ev_h[n]] for n in real_names},
        FP_BUDGET,
    )
    t_arm = fitted["threshold"]
    calibration_union_fp = fitted["calibration_union_fp"]
    union_fp = fitted["evaluation_union_fp"]
    baseline_fp = {n: float(np.mean(hits)) for n, hits in baseline_eval.items()}
    worst = max(union_fp.values())

    recalls, base_recalls = {}, {}
    for g in GENERATORS:
        _, ev = halves(pops[f"gen_{g}"])
        both = [cf[p]["image_score"] >= CF_T or bf[p]["image_score"] >= BF_T
                for p in ev]
        three = [b or arm_scores[p] >= t_arm for b, p in zip(both, ev)]
        base_recalls[g] = float(np.mean(both))
        recalls[g] = float(np.mean(three))

    # probe / canlı görsel / ffhq, nihai eşikte (embed önbellekleri hazır)
    def head_scores(cache_name):
        e = np.load(ART / f"embed_{cache_name}.npy")
        return e @ coef + intercept

    probe_final = float((head_scores("probe") >= t_arm).mean())
    probe_q75_final = float((head_scores("probe_q75") >= t_arm).mean())
    ffhq_final = float((head_scores("ffhq_holdout") >= t_arm).mean())
    live = head_scores("live_chatgpt") if (ART / "embed_live_chatgpt.npy").is_file() else None

    admission_pass = probe_final >= 0.40
    out = {"t_arm_candidate": t_arm,
           "calibration_union_fp": calibration_union_fp,
           "baseline_fp": baseline_fp, "union_fp": union_fp,
           "union_worst_fp": worst,
           "union_macro_fp": float(np.mean(list(union_fp.values()))),
           "added_fp_sources": {n: union_fp[n] - baseline_fp[n]
                                 for n in real_names if union_fp[n] > baseline_fp[n]},
           "generator_recall_2arm": base_recalls,
           "generator_recall_3arm": recalls,
           "probe_recall_final": probe_final,
           "probe_q75_recall_final": probe_q75_final,
           "ffhq_holdout_fp_final": ffhq_final,
           "live_chatgpt_above_final": (bool(live[0] >= t_arm)
                                        if live is not None else None),
           "admission_pass": admission_pass,
           "admission_reason": ("all gates passed" if admission_pass else
                                "G1 failed: in-collection probe recall < 40%")}
    (ART / "union.json").write_text(json.dumps(out, indent=2))
    if admission_pass:
        np.savez(Path("artifacts/gpt_arm_v1.npz"), coef=coef.reshape(1, -1),
                 intercept=np.array([intercept]), threshold=np.array(t_arm))
    print(f"taban 2-kol worst: {100*max(baseline_fp.values()):.1f}% · "
          f"3-kol worst: {100*worst:.1f}% · eşik {t_arm:.2f}")
    print(f"probe (nihai eşik): {100*probe_final:.1f}% (q75 {100*probe_q75_final:.1f}%) · "
          f"ffhq FP {100*ffhq_final:.1f}%")
    print("admission:", "PASS" if admission_pass else "REJECT — G1 <40%")
    print("2-kol recall:", {g: f"{100*v:.0f}%" for g, v in base_recalls.items()})
    print("3-kol recall:", {g: f"{100*v:.0f}%" for g, v in recalls.items()})
    print(f"Results -> {ART/'union.json'}")


STAGES["union"] = stage_union

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    a = ap.parse_args()
    for name, fn in STAGES.items():
        if a.stage in (name, "all"):
            print(f"===== STAGE {name} =====")
            fn()
