# CS395 internship report package — 29 September 2026

This is the current E1–E152 submission draft. It supersedes the old E26 report **for presentation of the current project**, without rewriting historical records. The student’s confirmed metadata and personal account have been incorporated. No visible completion placeholders remain. Submission date is 29 September 2026 in all three artifacts and their filenames. The unit works on Tivibu software and hardware; Hasan Çontuk is identified as the student’s mentor and a member of Önder Çelebi’s team, without an invented formal title. See [the Turkish handoff](HANDOFF_TR.md) before submission.

## Deliverables

| Artifact | File |
|---|---|
| Report, editable | [DOCX](deliverables/CS395_FinalReport_Efe_Han_Keleş_29September2026.docx) |
| Report, reading copy | [PDF](deliverables/CS395_FinalReport_Efe_Han_Keleş_29September2026.pdf) |
| Report, office legacy format | [DOC](deliverables/CS395_FinalReport_Efe_Han_Keleş_29September2026.doc) |
| Presentation, editable and with notes | [PPTX](deliverables/CS395_Presentation_Efe_Han_Keleş_29September2026.pptx) |
| Presentation, reading copy | [PDF](deliverables/CS395_Presentation_Efe_Han_Keleş_29September2026.pdf) |
| One-slide digest | [PPTX](deliverables/CS395_Digest_Efe_Han_Keleş_29September2026.pptx) |
| Digest, reading copy | [PDF](deliverables/CS395_Digest_Efe_Han_Keleş_29September2026.pdf) |
| Study material | [Turkish guide](STUDY_GUIDE_TR.md), [English speaking notes](SPEAKER_NOTES_EN.md) |

The deck has 10 spoken slides and two reference slides. Suggested timing is 600 seconds (10:00), with 915 spoken words and pauses for the figures; actual timing requires rehearsal. The report has 21 pages including front matter, 3 figures, 5 tables and 15 references, of which 10 are scholarly works. No new classifier experiment, dataset download, final-reserve opening or serving change occurred during report production.

## Desktop submission folder

The current submission portal screenshot requires the digest as PDF; the student explicitly requested this format on 29 September. The Desktop folder `CS395_PixelProof_Submission_2026` contains the report DOC, presentation PPTX and one-slide digest PDF, verified against repository copies. The editable digest PPTX remains in `deliverables/` only. This portal-specific format supersedes the earlier Desktop digest PPTX choice based on the guideline. Historical receipts retain their original file lists.

## Latest revision — 3 October 2026

Reviewer feedback was applied to the reference section: centered numbered heading and five-space hanging indents, checked in both PDF and legacy DOC rendering. Report remains 21 pages with unchanged text and contents locations. The Desktop DOC is the corrected copy. The original submission date and filenames remain 29 September 2026. Details: `sources/reference_format_review_20261003.json`. A subsequent deeper check found Liberation Serif substituted in all three PDFs; these were re-exported with embedded Times New Roman. See `sources/font_render_review_20261003.json`. Source DOCX/DOC/PPTX bytes did not change in this second correction. A third review then corrected a 3.22 mm horizontal displacement of all three report figures; report DOCX/PDF/DOC and the Desktop DOC are now updated. Only report pages 5 and 12 changed in that step. See `sources/geometry_review_20261003.json`.

Following the requested 20–25-page range, the report is now 21 pages. Its 3,353 words of paragraph prose explain transfer learning, data roles, corrections, results and limits in plain language; company prose stays at 191 words. The presentation was reduced to 10 main slides plus two reference slides, and the digest uses the same concise outcome and limitation. All 15 references remain substantively used; missing SIDD/SID page ranges were added and the DINOv2 journal year was directly verified. All 605 scoped artifact checks pass. The company supplier example is Nokia, sourced to Finnvera (2025). Hasan is described by his confirmed mentoring role, without inventing a corporate job title. The guideline does not separately demand the unit’s parent reporting line. See `sources/final_fields_review_20260929.json`. Model2 data suitability and editing-scope claims were subsequently reconciled in `sources/model2_scope_review_20260929.json`. The metadata completion is recorded in `sources/metadata_completion_20260929.json`; the prior clarity review is `sources/clear_package_review_20260929.json`. See [REFERENCE_REVIEW.md](REFERENCE_REVIEW.md).

## Evidence and claims

The frozen research source is commit `497d45c`. The later documentation-review baseline is `16e03d8`; [MD_CONSISTENCY_AUDIT.md](MD_CONSISTENCY_AUDIT.md) records the completed sequential reading of all 28 baseline Markdown files and the subsequent reconciliation. [REQUIREMENTS.md](REQUIREMENTS.md) records the instructor/office precedence. [EVIDENCE_REVIEW.md](EVIDENCE_REVIEW.md) maps the major project areas. The output audit checks formatting and selected data assertions; it does not establish detector generalization. The 20/20 milestone means numerical checks on reused development data. E92 still failed the separate E43 retention rule. Model2 remains experimental.

Private gallery pixels, raw datasets, model weights, the student's transcript and the supplied administrative screenshot are excluded. The annual report used as a company reference is not committed. The package contains aggregate project evidence and original explanatory diagrams only.

## Reproduction

The builders use the bundled Codex Python environment (`python-docx`, ReportLab, pdf2image, pypdf) and bundled Node with `@oai/artifact-tool`. Use `load_workspace_dependencies` to resolve installed runtime paths. The report builder currently expects the macOS Times New Roman font in `/System/Library/Fonts/Supplemental/`. No package or model download is needed.

Before any DOCX render or `soffice` PDF export on this Mac, set `FONTCONFIG_FILE` to the absolute path of `tools/fonts.conf`. This exposes the installed Times New Roman fonts to the bundled renderer; OOXML font names alone are insufficient. `tools/audit_package.py` now inspects actual embedded PDF fonts and rejects substitutions.

Run `tools/build_report.py` with bundled Python. Render the resulting DOCX using the documents skill's `render_docx.py --emit_pdf`. Check actual heading locations against `sources/heading_pages.json`; update that file and rebuild if pagination changes. The current two-pass contents cache was checked against the PDF. Do not assume a changed report retains these page numbers.

Copy `tools/build_presentations.mjs` into a private build directory with a `node_modules` link to the bundled packages. Set absolute `REPORT_ROOT`, `PRESENTATION_BUILD`, `PRESENTATION_SKILL`, `RUNTIME_NODE_MODULES` and `RUNTIME_PYTHON`. Run with bundled Node. Use a fresh build directory for each finalized revision. The finalizer checks native tables, native chart/workbook data, encoded Times New Roman, package structure and first-party import. Use bundled `soffice` to export PDFs and the legacy DOC. Never replace source documents with converted temporary copies.

Run `tools/audit_package.py` after the final PDFs exist. Inspect every page and slide visually as well: XML checks cannot establish visual correctness. Desktop copies must match final repository artifact hashes. Rebuild all affected formats after filling the deferred fields.

The final editorial critique and item-by-item guideline map are in [CRITICAL_REVIEW.md](CRITICAL_REVIEW.md). The technical/editorial assessment is 9.2/10, not a predicted academic grade. The former placeholder blockers are resolved; portal acceptance and an instructor grade are not certified.
