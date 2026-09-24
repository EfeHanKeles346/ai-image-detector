# English speaking notes

Suggested timing: 12 minutes 45 seconds for slides 1–13. Slides 14–15 are references. Rehearse; this is not measured speaking time.

## 1. Purpose (40 seconds)

My internship project was PixelProof, a research prototype for detecting evidence of AI generation in images. I will explain the problem, how the project changed after failures, what the strongest model achieved, and what remains unresolved. The main result is a working local demonstration with substantial improvement on a defined development collection. It is not a universally reliable authenticity detector. The approved internship lasted forty working days. This presentation covers the technical record available before the end of the internship.

## 2. Two tasks (50 seconds)

There are two related but different tasks. Model1 evaluates a whole image. We want it to find AI images while avoiding false accusations against real photographs. A false alert and a missed AI image are different errors, and improving one can worsen the other. Model2 tries to localize an edited region. A convincing heatmap is not enough; it must agree with independent edit masks and avoid highlighting authentic regions. We therefore kept localization experimental rather than presenting it as a completed service.

## 3. Early failures (60 seconds)

Table 1 shows why a high internal score was not enough. The early small CNN achieved about ninety-seven percent on familiar images, but performance fell on an external collection. ResNet improved the familiar score and did much worse externally. A resolution control recovered much of that loss, showing that image preparation mattered. These are historical diagnostics, not final certified results: later auditing also found confounds in the external collection. The important lesson was to investigate what changed before blaming or praising the architecture. Sources: project experiments E1 to E6 and E10.

## 4. Measurement repairs (65 seconds)

Two corrections changed the interpretation of earlier work. Some datasets encoded real and AI labels in the opposite direction. We added explicit mappings and reran affected experiments. One DINOv2 result changed from near chance to useful discrimination, so the original explanation was withdrawn. Another audit found that threshold selection could consult evaluation data. Correcting it to use calibration only caused the candidate to fail its admission rule. We retained these failures in the record. Training, calibration, development and final evaluation have different permitted uses. Different files are not automatically independent when they share scenes, prompts or generation ancestry. Sources: E19b, E19c and E27.

## 5. Method (65 seconds)

The project did not train a foundation model from scratch. Frozen encoders turn an image into numerical descriptions. PixelProof fits additional components and a constrained correction around these features. E92 used twelve thousand two hundred sixty-nine training parent images, with four processing views per parent. More views test processing robustness, but do not create independent photographs. The demo checks the upload, computes original and processed scores, and applies a documented display rule. My responsibilities included directing experiments, organizing sources and reviewing evidence, with AI coding assistance used for implementation and documentation. Research references: Radford et al. 2021, Oquab et al. 2024, Kim et al. 2026 and Ojha et al. 2023.

## 6. Main improvement (75 seconds)

Figure 1 is the clearest development improvement. All three models are compared on the same real-image parents. For original images, false AI alerts fell from sixty-eight out of one hundred sixty to zero. For the fixed social-style transformation, they fell from sixty-eight to fourteen. That transformation caps the long side at 1080 pixels and applies JPEG quality seventy-five. E92 detected one hundred fifty-nine of one hundred sixty AI images in each condition. This is strong progress on this collection, but zero observed original false alerts does not mean zero future risk. The real images come from only ten SIDD scenes, and these development observations were repeatedly used. Source: evidence/e92_development.json.

## 7. Acceptance (65 seconds)

Twenty out of twenty meant ten numerical requirements checked twice. The requirements covered false alerts, AI recall, balanced accuracy, ranking, coverage and uncertainty. They were project-defined, correlated checks, not an external universal standard. A separate retention rule required that no AI image caught by E43 become a new miss. E92 missed one original AI example that E43 had detected. Recovering other examples did not cancel that loss under the rule. Therefore numerical success and complete acceptance are different conclusions. We should celebrate the improvement without saying the full contract passed.

## 8. Scores and demo (65 seconds)

The displayed number caused understandable confusion. It is a raw model score multiplied by one hundred, not a probability that a photograph is AI. There is no reason to impose a fifty-percent boundary. The inherited upper cut is approximately seven point nine four; the lower cut is approximately one point one five. An original score above the upper cut keeps an AI alert visible. Two low E92 views produce no clear AI evidence. Other cases remain uncertain, and disagreement can require review. The older reference is advisory and no longer vetoes two low E92 scores. These operating points are experimental, not universally optimal. Guo et al. explain why calibration is a separate question.

## 9. Broader reliability (60 seconds)

The practical gallery check found nine false alerts in two hundred six unique original real files, and eighteen after processing. That collection had already been inspected and mostly represents one phone, so it is a diagnostic, not independent accuracy. Later source-separated experiments used different research heads. No tested calibration assignment met the full evaluation conditions. Adding source-pixel features slightly improved clean aggregates but harmed AI recall after several transformations. Those candidates were rejected instead of replacing E92. These results identify remaining weaknesses; they do not provide a new E92 accuracy estimate. Sources: E95, E146, E151 and E152.

## 10. Localization (65 seconds)

Model2 illustrates another generalization failure. The earlier head ranked edited pixels reasonably on the original placement, with pixel AUC around zero point seven four. When edits moved, that fell to about zero point five seven. Training on two placements partly recovered performance at the new location, but harmed the original location and increased the area wrongly highlighted in authentic images. Only sixteen consumed parent images and one editor were involved. The many transformed maps are not independent samples. Moving an edit also changes its content and generated output, so this is not a pure causal location experiment. Model2 remains separate from the active demo. Sources: E132 through E135.

## 11. Deliverables (55 seconds)

The outcome includes a trained research system, a local upload interface, experiment commands, aggregate result files and documentation. Software guards handle invalid images, unavailable dependencies, response schema mismatches and cancellation around the heavy inference worker. The artifact manifest is checked before loading the model. The latest recorded checkpoint passed one thousand two hundred forty-five Python tests. These are software assertions, not successful classifications of that many images. No production rollout or company business impact was measured. A reproducible negative result is also a deliverable because it prevents repeating an unsupported claim.

## 12. Outcome (55 seconds)

The prototype is a meaningful student engineering result. It integrates modelling, data handling, software and evaluation. Its strongest improvement is real and reproducible for the specified development population. The equally important limitation is that we have not proved universal detection. Reused data, limited camera scenes, previously seen generator families and unknown pretraining overlap restrict the claim. Model2 also needs much broader evidence. Table 5 is the distinction I would want a reader to remember: building a working system and proving broad reliability are separate achievements.

## 13. Conclusion (45 seconds)

The next priority is an independent evaluation design rather than another impressive number on familiar data. It needs separated sources and ancestry, more cameras and generators, fixed operating costs and calibration that cannot see evaluation outcomes. Localization also needs more editors and authentic controls. My central technical lesson is to verify what a score actually proves. We built a functioning prototype, improved a defined benchmark, corrected mistakes and identified the evidence still missing. Thank you. The following two slides contain references for questions; the full bibliography is in the report.

## 14. References A (0 seconds)

Backup slide. Full author lists, retrieval dates and direct source links are provided in Section 8 of the report. Project findings are traceable to the evidence map in Appendix 9.2.

## 15. References B (0 seconds)

Backup slide. Pretrained components do not transfer their published benchmark guarantees to our system. Calibration and adaptive data reuse are separate evaluation problems. Corporate context comes from official Türk Telekom sources.
