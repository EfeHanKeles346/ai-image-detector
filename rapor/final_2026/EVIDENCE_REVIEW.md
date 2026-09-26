# Evidence review and validation record

Research snapshot: `497d45c`, before report production. The later reconciliation
sequentially reviewed all 28 tracked Markdown files at `16e03d8` (37,073 lines), including
HISTORY, EXPERIMENTS, PLAN, DATASETS and old reports. Exact duplicate paragraphs were
cross-referenced after prior exposure. Current edits were separately reviewed. See
`sources/markdown_read_progress.json` and `MD_CONSISTENCY_AUDIT.md`. This establishes
reading coverage, not independent replication of every experiment. Historical records
remain preserved; later corrections govern current claims.

## Coverage

| Project area | Evidence inspected | Treatment in the report |
|---|---|---|
| Early CNN and embedding baselines | EXPERIMENTS E1–E10 | Familiar/external mismatch, resolution controls, qualified historical benchmark |
| Handcrafted, tile and ensemble approaches | EXPERIMENTS E8–E19 | Specialist gains, source weaknesses, no universal averaging claim |
| Dataset semantics and invalidated conclusions | E19b–E19c, DATASETS, reference-note corrections | Reversed labels, explicit mapping, reruns and withdrawn explanation |
| Calibration and deployment eligibility | E27 and current PLAN/README/SERVING | Evaluation leakage corrected, candidate removed |
| Native and learned representation development | E31–E43, E51–E92 records; E92 fit/development receipts | Frozen features plus project-owned adaptation, no foundation-model-from-scratch claim |
| E92 acceptance and reuse limits | e92_development, e92_readiness_audit | 20 numeric checks versus failed extra retention rule; 10-scene real dependence |
| Real-world diagnostic collection | E93–E95, e95_gallery_report | Gallery 206 unique files, correlated consumed observations, no private pixels |
| Current browser result policy | primary_demo_policy.py, e92_demo.py, internship_serve.py; project_audit | E92 primary, E43 advisory, paired outcome distinct from per-view benchmark |
| Runtime integrity and cancellation | HISTORY 2026-09-16, project audit/runtime receipts | Pinned manifest and worker ownership; software correctness separated from accuracy |
| OOD and processing diagnostics | E131/E139, E146/E148–E152 receipts and contracts | Separate research heads, source-fold failure, processing imbalance, rejected residual features |
| Localization | E17 controls; E107/E115–E117; E132–E135 receipts/contracts | Different task, 16 consumed parents, placement trade-off, rejected candidate |
| Provenance and resources | DATASETS, ARTIFACTS, LICENSE, PLAN | Roles, ancestry, caching, memory correction, no unverified independent reserve claim |
| Scientific background | 10 primary scholarly references and Adobe industry example | Methods, bias, calibration and adaptive reuse; external results not inherited |
| Organization | Official corporate profile and 2025 annual report | Verified corporate context; student-confirmed unit/address/title; remaining host details in HANDOFF_TR.md |

## Important reconciliations

E92 training used 12,269 parents and 49,076 views. The later 12,525-parent/50,100-view
research pool is not retroactively attributed to E92. A transformed view is not a new
independent parent. The E66 numerical criteria, paired current-policy outcomes, gallery
false alerts and Model2 pixel metrics use different denominators and decision procedures.
They are never merged into one accuracy claim.

The reference-relative new AI miss remained despite improvements in aggregate recall.
The current UI's no-clear outcome is not a certificate of authenticity. The 7.94 and
1.15 displayed cuts are experimental score operating points, not probabilities. Separate
source-fold diagnostic heads do not supply an updated active-E92 accuracy estimate.
The last recorded 1,245 Python tests are software checks, not image classifications.

## Earlier output validation — 25 September 2026

The report was rendered from DOCX with the bundled document renderer and all 30 pages
were visually inspected after the current revision. A two-pass contents cache matches all 35 rendered headings.
Three figures and six tables have centered numbered captions in the required position,
with nearby preceding citations. Body text is justified, black Times New Roman 12 pt,
double spaced without indentation; references are single spaced with space between
entries. The instructor's left-aligned headings/no-indent rules take precedence over
conflicting office reference formatting. The abstract contains 228 words.

The report meets the section page limits in this draft: company 2 pages, related
literature 2, project details 9, results 1, experience 2, conclusions 1 and recommendations
1. Initial status and department each fit within their shared background page; motivation
and typical-day text fit within one page each. Filling the deferred fields requires
repagination and another audit.

The editable deck contains 15 slides (13 spoken, 2 references), five native tables and
one native chart with a six-value embedded workbook snapshot derived from the E92
receipt. The digest is a separate single-slide deck. Both pass the presentation skill's
package, layout, encoded-font and first-party import checks with no final warnings.
Both were exported to PDF; every slide was inspected. In the current revision, slide 9 was inspected
again; the other fourteen slides and the digest matched the prior reviewed renders pixel for pixel. Suggested speech
time is 795 seconds; no live rehearsal was performed.

The legacy DOC was round-tripped through bundled LibreOffice to a private PDF. It kept
30 pages, and per-page extracted word multisets matched the DOCX PDF. At 72 dpi, 25 pages
were pixel-identical. The contents pages and three figure pages (3, 4, 7, 17, 18) had
rendering differences and were visually checked without changed content or pagination.
The published report PDF is the DOCX render. Native Microsoft Word/PowerPoint behavior
was not tested; the student should check the actual submission device after final edits.

`tools/audit_package.py` checks formatting, reference presence, rendered contents pages,
section limits, selected E42/E49/E92/E102 assertions, Model2 nonpromotion, native slide content,
notes, duration and file size. `sources/package_audit.json` records artifact hashes and
results. These checks do not imply detector generalization or a guaranteed grade.

## Current completion status — 26 September 2026

The student has supplied identity, unit, office address, supervisor title, individual/advisory
roles, personal reflection, three course connections and the typical-day account. Five
information groups remain: submission date, department reporting line, department duties,
mentor details and relevant competitor/supplier names. The date is intentionally open.
HANDOFF_TR.md is the current field list. Earlier validation paragraphs record their dated
builds; the latest artifact audit and end-to-end review receipt govern current outputs.
No new dataset, model fit, reserve access, email or school upload occurred.

## 25 September reconciliation additions

The revised narrative now includes the E33–E50 calibration/fusion failures, E42 RR
16,953 parents/50,858 views, E43's failed E49 comprehensive final (11/20), E102's
12/160 processed false alerts without promotion, the inherited threshold origins and
the limited source-component/reserve coverage. None changes a measured result or
establishes a fresh E92 final pass. The output checker passes 638 scoped assertions.

The final readability pass retained 30 pages and 35 contents entries, with 14 references.
All 16 changed report-page renders were inspected; 14 unchanged pages matched earlier
reviewed renders exactly. The latest legacy conversion again preserves per-page words
and page count, with its five image-rendering differences inspected. A local macOS
English spelling check was run; flagged names, URLs, research terms and valid US/UK
variants were reviewed. No external grammar service received the report.

## End-to-end consistency and readability review — 26 September 2026

Read the complete current report, all presentation text and notes, digest and Turkish
study guide against the completed Markdown review. A hash comparison found 22 of the
30 files in the prior review receipt unchanged, including the scientific experiment and
dataset records. The eight changed files concerned report metadata, handoff and history;
their changes were reviewed. This is a delta reconciliation of the earlier full reading,
not a claim that every research line was newly reread or every experiment reproduced.

Clarified the difference between per-view measurements and the website's combined
result in Sections 4.5.6 and 9.1. The 14 processed false alerts must not be presented as
14 final website AI warnings. The paired rule yielded 139 no-clear and 21 uncertain real
images, plus 159 AI alerts and one uncertain AI image, on the same reused collection.
The speaking notes and Turkish guide now explain this distinction. The digest explicitly
labels original and processed real-image errors and explains that one newly missed AI
image prevented full acceptance. Model2 slide precision now matches the report's table.
Removed the administrative submission-date reminder from the spoken introduction.

Corrected stale completion statements in the current review and evidence summary.
Historical addenda retain their original dates and are clearly labelled as old states.
Scientific conclusions, data, model and serving policy did not change. No broad reliability
claim, new final test or grade guarantee follows from these editorial improvements.

All seven export files were regenerated. The report remains 30 pages with the same
35 contents entries, 228-word abstract, three figures, six tables and 14 references.
Pages 5, 19 and 29 changed and were inspected at full size; the other 27 are pixel-identical
to the previously reviewed final render. Presentation slides 9 and 10 and the digest
changed visually and were inspected; the other 13 slides are pixel-identical. Notes changed
on additional slides. Both PPTX finalizers report zero layout warnings. Speaking time is
815 seconds (13:35), planned rather than rehearsed. All 641 scoped checks pass.
The legacy DOC round-trip keeps every page's word multiset and all 30 pages; its five
rasterization/TOC differences were visually checked. Native Microsoft Office was not tested.

Five information groups remain in HANDOFF_TR.md, including the intentionally blank
submission date. The three Desktop submission files were refreshed only after verifying
that their pre-edit bytes matched the saved repository versions. No school upload occurred.
