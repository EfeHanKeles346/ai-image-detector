# RINE feasibility and provenance audit

**Audit date:** 2026-08-24  
**Purpose:** O1 decision only; no model, dependency or checkpoint was downloaded or integrated.  
**Official repository:** <https://github.com/mever-team/rine>  
**Pinned revision:** `9b7fd5857cc205d0412be6aeee0d7611b95bd620` (2024-11-05)  
**Paper:** <https://arxiv.org/abs/2402.19091>

## Decision

**Conditional GO for an isolated local O2 benchmark; NO-GO for direct installation into the
serving environment or weight redistribution.** The representation matches E28's diagnosed need,
and the official RINE trainable checkpoint is small enough to test. The upstream runtime recipe is
not reproducible or safe enough to install as-is, and the OpenAI CLIP base-weight redistribution
boundary is not explicit enough for PixelProof to publish those bytes.

## Licence and provenance matrix

| Component | Pinned identity | Finding | O1 verdict |
|---|---|---|---|
| RINE paper | arXiv `2402.19091` / ECCV 2024 | Describes intermediate CLIP block features, learned block importance, BCE and contrastive training. | PASS as methodological source |
| RINE source | Git `9b7fd585...620` | Root `LICENSE` is Apache-2.0. Required notices must accompany copied/modified code. | PASS, but prefer a small independent adapter |
| RINE 4-class head | `ckpt/model_4class_trainable.pth`, 25,298,182 bytes, Git blob `bf5cd405...c457` | Committed inside the Apache-2.0 repository; save code excludes every key containing `clip`, so it does not bundle CLIP parameters. No separate checkpoint notice was found. | PASS for pinned local evaluation; redistribution remains conservative/disabled |
| OpenAI CLIP code | Git `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` | Repository `LICENSE` is MIT. RINE's requirement does not pin this revision. | PASS only when explicitly pinned |
| CLIP ViT-L/14 base | official URL embeds expected SHA-256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`; HTTP size 932,768,134 bytes | Download is integrity-checked by official CLIP code, but no separate model-weight licence was found during O1. | Local research only; do not commit or redistribute |
| RINE upstream training data | ProGAN classes and multiple external evaluation sets | Not needed to evaluate the published trainable head; dataset terms are separate from code. | Do not download or redistribute for O2 |
| PixelProof evaluation data | Existing Defactify and ten authentic forensic sources | Already governed by PixelProof's frozen calibration/evaluation protocol. | PASS for the same local research evaluation |

This is a project provenance decision, not legal advice. Ambiguity is handled by the stricter
operational choice: keep external weights local and unserved.

## Runtime contract discovered from the pinned source

- Architecture: OpenAI CLIP `ViT-L/14`, frozen; forward hooks collect every visual module whose
  name contains `ln_2`; projection layers and a learned `alpha` weight combine intermediate blocks.
- Candidate: official 4-class trainable head, because the pinned demo and validation entry point
  use it as the multi-class ProGAN training candidate. Label/logit direction is `1 = fake`; the demo
  applies sigmoid and calls the result probability of fake.
- Evaluation preprocessing: RGB, center crop 224 with no preceding resize, tensor conversion, CLIP
  mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation
  `(0.26862954, 0.26130258, 0.27577711)`. O2 must explicitly define behavior below 224 px and cap
  decoded image resources before the crop.
- Resource profile: the base checkpoint download alone is 932.8 MB. The project machine has
  19.3 GB physical memory, MPS available and about 50 GB free disk at audit time. Batch-one smoke is
  feasible to attempt; full evaluation must measure rather than assume memory/latency.

## Why the upstream environment will not be installed directly

- `requirements.txt` points at unpinned `git+https://github.com/openai/CLIP.git` and otherwise leaves
  versions open; its README specifies Python 3.9, Torch 2.1.1 and CUDA 11.8, while PixelProof locks
  Python 3.13, Torch 2.13 and torchvision 0.28 for serving.
- `clip`, `pandas` and `opencv-python` are absent from the locked serving environment. Only `clip`
  is necessary for inference, but adding the unpinned upstream dependency would weaken H5's locked
  runtime.
- The upstream demo hard-codes `cuda:0`, and its model helper uses dynamic `exec` while assigning
  checkpoint fields. PixelProof will not copy that loader. O2 must use a strict expected-key tensor
  state dictionary, attempt `weights_only=True`, and support CPU/MPS explicitly.

## O2 allowed scope

1. Fetch only the pinned OpenAI CLIP source/base and pinned RINE 4-class trainable checkpoint into
   ignored local research storage; compute and record SHA-256 before deserialization.
2. Implement a minimal project-owned adapter against the published architecture. Do not modify
   `requirements-serving.lock`, `artifacts.manifest.json`, API, CLI default, web UI or E20 files.
3. Smoke score the two pinned RINE demo images, verify label direction and compare with upstream
   expected semantics; then run PixelProof's existing frozen source-wise protocol exactly once.
4. Stop unless every O2 gate in `PLAN.md` passes. Even a pass permits only an O3 pre-registration,
   not serving or redistribution.
