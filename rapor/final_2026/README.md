# CS395 internship report package — 25 September 2026

This is the current E1–E152 submission draft. It supersedes the old E26 report **for presentation of the current project**, without rewriting historical records. Student-deferred metadata and personal reflection fields remain visibly marked. See [the Turkish handoff](HANDOFF_TR.md) before submission.

## Deliverables

| Artifact | File |
|---|---|
| Report, editable | [DOCX](deliverables/CS395_FinalReport_EfeHan_Keles_25September2026.docx) |
| Report, reading copy | [PDF](deliverables/CS395_FinalReport_EfeHan_Keles_25September2026.pdf) |
| Report, office legacy format | [DOC](deliverables/CS395_FinalReport_EfeHan_Keles_25September2026.doc) |
| Presentation, editable and with notes | [PPTX](deliverables/CS395_Presentation_EfeHan_Keles_25September2026.pptx) |
| Presentation, reading copy | [PDF](deliverables/CS395_Presentation_EfeHan_Keles_25September2026.pdf) |
| One-slide digest | [PPTX](deliverables/CS395_Digest_EfeHan_Keles_25September2026.pptx) |
| Digest, reading copy | [PDF](deliverables/CS395_Digest_EfeHan_Keles_25September2026.pdf) |
| Study material | [Turkish guide](STUDY_GUIDE_TR.md), [English speaking notes](SPEAKER_NOTES_EN.md) |

The deck has 13 spoken slides and two reference slides. Suggested timing is 765 seconds, subject to actual rehearsal. The report has 28 pages including front matter, 3 figures, 6 tables and 13 references, of which 10 are scholarly works. No new classifier experiment, dataset download, final-reserve opening or serving change occurred during report production.

## Evidence and claims

The frozen research source is commit `497d45c`. [REQUIREMENTS.md](REQUIREMENTS.md) records the instructor/office precedence. [EVIDENCE_REVIEW.md](EVIDENCE_REVIEW.md) maps the major project areas. The output audit checks formatting and selected data assertions; it does not establish detector generalization. The 20/20 milestone means numerical checks on reused development data. E92 still failed the separate E43 retention rule. Model2 remains experimental.

Private gallery pixels, raw datasets, model weights, the student's transcript and the supplied administrative screenshot are excluded. The annual report used as a company reference is not committed. The package contains aggregate project evidence and original explanatory diagrams only.

## Reproduction

The builders use the bundled Codex Python environment (`python-docx`, ReportLab, pdf2image, pypdf) and bundled Node with `@oai/artifact-tool`. Use `load_workspace_dependencies` to resolve installed runtime paths. The report builder currently expects the macOS Times New Roman font in `/System/Library/Fonts/Supplemental/`. No package or model download is needed.

Run `tools/build_report.py` with bundled Python. Render the resulting DOCX using the documents skill's `render_docx.py --emit_pdf`. Check actual heading locations against `sources/heading_pages.json`; update that file and rebuild if pagination changes. The current two-pass contents cache was checked against the PDF. Do not assume a changed report retains these page numbers.

Copy `tools/build_presentations.mjs` into a private build directory with a `node_modules` link to the bundled packages. Set absolute `REPORT_ROOT`, `PRESENTATION_BUILD`, `PRESENTATION_SKILL`, `RUNTIME_NODE_MODULES` and `RUNTIME_PYTHON`. Run with bundled Node. Use a fresh build directory for each finalized revision. The finalizer checks native tables, native chart/workbook data, encoded Times New Roman, package structure and first-party import. Use bundled `soffice` to export PDFs and the legacy DOC. Never replace source documents with converted temporary copies.

Run `tools/audit_package.py` after the final PDFs exist. Inspect every page and slide visually as well: XML checks cannot establish visual correctness. Desktop copies must match final repository artifact hashes. Rebuild all affected formats after filling the deferred fields.
