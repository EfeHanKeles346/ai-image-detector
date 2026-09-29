# English speaking notes

Slides 1–10 form a planned 10-minute talk. Slides 11–12 are reference material for questions. The spoken script has 916 words; the schedule allows pauses and time to explain the chart. Timing is a plan, not a rehearsal measurement. A live demo and questions are outside this schedule.

## 1. Project purpose (35 seconds)

My internship project was PixelProof, a research prototype for AI image detection. I worked individually at Türk Telekom and asked my supervisor and mentors for advice. The main goal was to reduce false AI warnings on real photographs while keeping useful AI detection. I built a local demonstration and recorded both improvements and failed experiments. I will explain the method, the main result and the limits of that result.

Source or optional detail, not part of the spoken script: Student-confirmed internship information. Report Sections 1, 3.1 and 4.2.

## 2. Problem and objective (50 seconds)

The project has two important errors. A false positive means that a real photograph receives an AI warning. A false negative means that an AI image escapes detection. Lowering one error can increase the other, so I measured both. The harder question was whether the detector could work when the camera, generator or image processing changed. That is generalization. Model1 deals with the whole image. A smaller second module, Model2, explores where a local AI edit might be.

Source or optional detail, not part of the spoken script: Report Sections 3.3 and 4.1.

## 3. What changed our approach (65 seconds)

At first, a pretrained ResNet model reached almost ninety-eight percent accuracy on familiar test images. But its accuracy fell to about twenty-five percent on a separate collection. These are historical results, and later auditing found problems in that external benchmark too. The lesson was still clear: one high score was not enough. I checked input resolution, found reversed labels in later datasets and repaired a procedure that used evaluation data to choose a threshold. These corrections changed some earlier conclusions. I kept the corrections in the records instead of reporting only the best numbers.

Source or optional detail, not part of the spoken script: Keleş (2026), E1–E6, E10, E19b–E19c and E27. No independent final-performance claim from Table 1.

## 4. Method and contribution (75 seconds)

I used transfer learning, which means reusing knowledge from existing models. The pretrained components convert the image into numerical descriptions called features. Their weights remain fixed. The project trains smaller components to use those features and adjust the detection score. This is more practical than training a large visual model from the beginning. The current demo model is called E92. My work covered choosing experiments, organizing data, reviewing results and building the application. I used AI coding assistance for implementation and documentation. After each experiment, the plan and logs recorded what worked, what failed and the next decision.

Source or optional detail, not part of the spoken script: Radford et al. (2021), https://proceedings.mlr.press/v139/radford21a.html; Oquab et al. (2024), https://arxiv.org/abs/2304.07193; Kim et al. (2026), https://arxiv.org/abs/2606.10309v2; Keleş (2026), E51–E92. Published component benchmarks are not PixelProof guarantees.

## 5. Understanding the data (60 seconds)

E92 learned from twelve thousand two hundred sixty-nine training images. Each image had four processing versions, giving forty-nine thousand seventy-six training views. A compressed copy is still related to its original, so views are not independent photographs. The development collection was separate from training and contained one hundred sixty real images and one hundred sixty AI images. We used it repeatedly to compare candidates. That makes it useful for development, but limits its value as a final test. The downloaded archive size is also different from the number of images actually admitted to a particular experiment.

Source or optional detail, not part of the spoken script: Keleş (2026), E92 training and E66 development records. Report Section 4.5.3. The real development images cover ten SIDD scenes; the AI groups are two previously seen families.

## 6. Main result (80 seconds)

Figure 1 shows the main improvement. For real original images, false warnings fell from sixty-eight to zero out of one hundred sixty. After resizing and JPEG compression, they fell from sixty-eight to fourteen. The current model detected one hundred fifty-nine of one hundred sixty AI images in each condition. This is a useful improvement on these images. It does not mean that every new photograph will work. The comparisons use the same development images, and the real photographs come from only ten scenes. Also, these are results for each view separately. The website combines two views into one displayed outcome, which I will explain shortly.

Source or optional detail, not part of the spoken script: Keleş (2026), evidence/e92_development.json. Transformation: long-side cap 1080 followed by JPEG quality 75. SIDD scene diversity: Abdelhamed et al. (2018), https://abdokamel.github.io/sidd/.

## 7. What the result proves (60 seconds)

Twenty out of twenty means ten numerical checks applied to two processing conditions. It does not mean twenty independent datasets or perfect classification. A separate rule required us to keep every AI image that the earlier model had detected. E92 newly missed one of those original AI images. Therefore it did not pass full acceptance, even though the overall numbers improved. We also reused the development set while making decisions. A stronger claim would need a final evaluation that had not influenced those decisions. This is why I describe E92 as a research prototype.

Source or optional detail, not part of the spoken script: Keleş (2026), E92 acceptance report. Dwork et al. (2015), https://proceedings.neurips.cc/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html. The 20 criteria are project-defined and correlated.

## 8. Demo and uncertainty (55 seconds)

The demo lets a user upload an image and read a simple result. It can report AI evidence, no clear AI evidence or uncertainty. The displayed score describes the detector response. It is not a verified probability that the image is AI. A low score therefore cannot prove that the photograph is authentic. Comparing the original with a processed copy helps reveal unstable responses. The software also validates uploads and checks model artifacts. These guards improve application behavior, while detection reliability still needs separate evaluation.

Source or optional detail, not part of the spoken script: Report Section 4.5.6 and current-policy audit. Guo et al. (2017), https://proceedings.mlr.press/v70/guo17a.html. Optional Q&A: upper score cut ~7.94; lower ~1.15. Original ≥ upper retains AI alert, both E92 views < lower give no clear evidence, other cases uncertain. E43 is advisory. Paired development outcomes: real 139 clear/21 uncertain/0 AI; AI 159 alerts/1 uncertain.

## 9. The second model (60 seconds)

Model2 asks where an AI edit is located. We found datasets, but source overlap and unclear masks prevented building a broad, validated training set. In our Stable Diffusion pilot, asking for a local edit also changed pixels outside the selected region. We therefore pasted only the generated region into the original and kept comparison controls. This gave us sixteen controlled examples from one editor, not broad coverage. Later training helped a new edit placement but harmed the original placement and increased false markings on real images. I rejected that candidate. These data and evaluation limits kept Model2 experimental.

Source or optional detail, not part of the spoken script: Keleş (2026), E105/E109/E117, E122/E127 and E129–E135. Optional Q&A: 506/512 CocoGlide originals overlap protected ancestry; not proof of historical weight contamination. DiffSeg30k: 201 partial, 224 full-positive and 87 empty masks. E127: 11 accepted of 16 attempts, mean 99.9417% outside pixels changed and outside MAE 9.79/255; pixel change is not semantic change. Exact composites preserve outside pixels but introduce possible seam cues. E129 later passed all 16 engineering checks. E135 original/new-placement AUC .74→.70 / .57→.64; authentic false area 17.11%→25.74%. No ChatGPT pixel experiment or universal impossibility claim.

## 10. Conclusion (60 seconds)

The internship produced a working local prototype, trained adaptation components and a record of experiments and corrections. The strongest measured gain was fewer false warnings on the defined development collection. Broader reliability remains unresolved. For example, the owner gallery still produced false warnings and had already influenced development. The next step is evaluation on independent sources with fixed decision rules. Model2 needs broader controls too. Personally, I learned how to manage an individual project, narrow its scope and explain what the evidence supports. Building a working demo was useful, but understanding its limits was just as important. Thank you.

Source or optional detail, not part of the spoken script: Keleş (2026), E95, E139, E146–E152. Optional Q&A: E92 gallery false alerts 9/206 originals, 18/206 processed. Earlier E43 comprehensive E49 test failed 11/20 on 2,000 parents; it is not E92 final performance. No production rollout or measured company impact.

## 11. Method references (0 seconds)

Reference appendix. The report provides the full bibliography.

Source or optional detail, not part of the spoken script: https://proceedings.mlr.press/v139/radford21a.html
https://arxiv.org/abs/2304.07193
https://arxiv.org/abs/2606.10309v2
https://arxiv.org/abs/2302.10174

## 12. Data and evaluation references (0 seconds)

Reference appendix. The report gives fourteen full entries. These slides list the sources most relevant to the short talk.

Source or optional detail, not part of the spoken script: https://abdokamel.github.io/sidd/
https://proceedings.mlr.press/v70/guo17a.html
https://proceedings.neurips.cc/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html
https://github.com/EfeHanKeles346/ai-image-detector/tree/497d45c
