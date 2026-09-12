# E65 — camera/noise diagnostic (2026-09-13)

The pilot confirms E43 has REAL false alarms on additional camera/RAW-development inputs
and remains sensitive to the combined1080px/JPEG75 transport. It does not produce an improved
model, measure AI recall, or pass the project target. E43 and live serving remain unchanged.

83 source-checksummed originals (1,009,998,251B):67 WIFD SDR JPEGs from10 cameras and16 RawNIND
RAWs from8 paired scenes (4 Bayer,4 X-Trans). All decoded; no cross-reference match under the
fixed screen against151,082 reference records. Eight internal matches are the expected RAW
low/high scene pairs. Exact bytes/pixels and radius4 dHash with radius4 pHash confirmation
are a bounded screen, not a proof of no semantic/source overlap. WIFD scene identities remain
unknown. Do not treat83 files,166 transports or10 camera devices as independent scenes.

## Fixed measurement

E43 SHA `a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390`.
AI cut0.07940196245908739; REAL cut0.011505939625203613.166 views scored in146.33s using the
cached frozen DINOv2S feature pipeline. Full scores were locked before metrics, SHA
`4ed6222a5e40afd84e88f2cdeb86059728ad84e763908a0656afefc6b757680b`.
The second condition combines resizing and JPEG compression; it does not isolate JPEG alone.

| Source | Original false AI | 1080px/JPEG75 false AI | New false alarms / rescues after transport |
| --- | ---: | ---: | ---: |
| WIFD | 11/67 (16.42%) | 13/67 (19.40%) | 8 / 6 |
| RawNIND | 3/16 (18.75%) | 5/16 (31.25%) | 3 / 1 |

These are descriptive file counts on a small selected pilot. No independent-population
confidence interval or comparison to E49's different population is claimed.

## Paired RAW interpretation

High-ISO scores fall in5/8 scenes and rise in3/8, for both original and transported views.
The specific scenes with rises/falls differ between transports. The7D-6 soil/seedling scene
scores0.8130 at low ISO and0.0455 at high ISO in original development; JPEG75 gives0.8814 and
0.2554. Both originals were visually inspected after scores were locked: matching subject
layout, visibly greater grain in the high-ISO exposure. Visual inspection did not change
admission or metrics. This is an example of sensitivity, not a causal proof that noise alone
explains the score. Crop selection, exposure and development are also potential contributors.

RAW decoding retained camera WB and fixed brightness with no automatic brightening. Some
paired X-Trans mean RGB levels differ strongly (semicircle104.35 versus31.49; gnome47.56 versus
23.78). No brightness normalization was fitted after scores, no dark images were excluded.
Do not infer a CFA advantage from Bayer3/8 versus X-Trans0/8 original errors: the selected
scenes/cameras and exposures differ. Foveon WIFD4 files with0 errors cannot certify that family.

## Decision and next work

No noise-presence REAL veto, threshold sweep, candidate fit or E49 reopening is justified.
Whole WIFD and RawNIND publishers are now consumed diagnostic DEV, permanently excluded from
TRAIN and independent final in this project. The new feature cache is also diagnostic-only.

Next hypothesis: learn resistance to processing shortcuts with label-matched transforms and
scene/source grouping, while enforcing per-image AI replay retention. Before a new candidate,
freeze a properly separated, licensed TRAIN/DEV population containing difficult REAL and AI;
existing admitted TRAIN may supply training, but fresh DEV must include both labels and known
groups. Check SIDD's small paired scene structure and local unused AI admission first; do not
call either balanced/independent until provenance, overlap and grouping are verified. An E65
camera-name/ISO rule or tuning on these83 observations would not establish generalization.
Reserve a separate final before fitting and evaluate any candidate once under the existing
gates. E62/E63 show why zero TRAIN AI loss alone is insufficient. E59 stays paused.

## Acquisition sources and verification

[WIFD authors' repository](https://github.com/CSCRC-SCREED/WIFD) pinned to
3f577edf0b14c686aa08e8d0d8ae07a83ba44f26, MIT; the full tree is101.43GB and was not cloned.
[RawNIND author dataset](https://dataverse.uclouvain.be/dataset.xhtml?persistentId=doi:10.14428/DVN/DEQCIM)
v1.0 has dataset-level CC-BY-SA-4.0 and per-scene permissions. Only explicit CC0 scenes with
attribution were selected; restricted/research-only and official test-reserve scenes excluded.
Its full120.17GB payload was not downloaded. No executables or new model weights acquired.

New acquisition, decode/overlap and diagnostic runners each have frozen contracts in evidence;
all images/derivatives/features remain on the external volume.697 Python tests passed,
including11 E65 tests; compile, pip dependency consistency and whitespace checks passed.
The one Starlette/httpx deprecation warning and previously recorded web dependency audit debt
remain separate. No claim that full GitHub CI is green is made from these local checks.
