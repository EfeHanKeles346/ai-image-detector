# Reference review — 29 September 2026

The shortened report retains fourteen entries: ten scholarly works, two official company sources, one industry example and the project's own records. Each is cited for a specific role. No entry was retained merely to inflate the bibliography. Ten scholarly entries also satisfy the stricter academic-project minimum if that classification is applied. This review checked bibliographic metadata and the limited claims made in the report; it is not a replication of the papers.

| Reference | Why it remains | Verification and limit |
|---|---|---|
| Abdelhamed, Lin and Brown (2018) | SIDD source and scene diversity | Authors' SIDD page supplies the CVPR citation and ten scenes/five cameras. The 160-image subset is not 160 independent scenes. |
| Adobe (2026) | Required industry comparison | Official Help Center page is dated 25 August 2026. Corrected the former undated entry and in-text year; retrieval updated to 29 September. Credential verification is explicitly not implemented in PixelProof. |
| Chen et al. (2018) | SID low-light data conditions | Authors' SID page confirms authors, title and CVPR 2018. A denoising dataset is not itself evidence of detector accuracy. |
| Dwork et al. (2015) | Repeated holdout reuse | NeurIPS proceedings confirms authors, year, title and volume 28. Used to explain adaptive evaluation risk, not to claim a corrected independent final test. |
| Grommelt et al. (2024) | JPEG and size shortcuts | arXiv:2403.17608 confirms four authors and title. Kept explicitly as an arXiv work; no peer-reviewed venue invented. Dataset-specific findings are not universal detector behavior. |
| Guo et al. (2017) | Scores versus calibrated confidence | PMLR 70 confirms authors, title and pages 1321–1330. Does not establish calibration of E92. |
| Keleş (2026) | Project-specific measurements and corrections | Local Git resolves 497d45c17c1b82b90587aadedd73dcd3f37f93f0. Project receipts support claims; this self-reference is not independent validation. |
| Kim et al. (2026) | Pretrained DEAR component | arXiv:2606.10309v2 confirms seven authors, title and ICML 2026 acceptance note. Published benchmark results are not presented as PixelProof results. |
| Ojha et al. (2023) | Pretrained-feature detection motivation | arXiv:2302.10174 and the official CVF search result confirm title/authors, CVPR 2023 and pages 24480–24489. Broader transfer is not a guarantee for future generators. |
| Oquab et al. (2024) | DINOv2 representation | arXiv:2304.07193 confirms authors/title; the TMLR 2024 outstanding-certification announcement confirms the journal publication. The original preprint is from 2023; 2024 is the journal year. |
| Radford et al. (2021) | CLIP representation | PMLR 139 confirms authors, title and pages 8748–8763. CLIP's original task is not synthetic-image detection. |
| Türk Telekom (2026a) | Company profile and corporate structure | Official annual-report listing and previously downloaded 2025 report support corporate information; local report text rechecked. Figure 1 cites page 33. The user's department placement is separate and remains unconfirmed. |
| Türk Telekom (2026b) | Current company operating context | Official company search results support 81 provinces and 31,756 employees with June 2026 figures. The dynamic direct page was not fully extracted. No inference about the host department is drawn from these figures. |
| Wang et al. (2020) | Early cross-generator detection literature | arXiv:1912.11035 confirms authors and CVPR 2020 acceptance; official CVF search result confirms pages 8695–8704. The 2019 preprint identifier does not make the 2020 conference citation wrong. |

## Primary locations checked

The exact reference URLs remain in `sources/report_content.py` and Section 8. Additional verification used the official CVF proceedings results, `https://blog.tmlr.org/2024/announcing-the-2024-tmlr-outstanding-certification/`, and Türk Telekom's official company/annual-report search results. Several CVF direct pages returned access errors, OpenReview returned a browser challenge, and the corporate page/PDF had parser or timeout limitations. Author pages, arXiv, publisher search results and the already available local annual report were used where stated; no challenge was bypassed and no failed fetch is counted as a full-text read.

## Editorial decision

Removed repeated discussion, detailed implementation catalogues and the redundant appendix evidence table. Retained all fourteen references because each supports a remaining statement, dataset, method or required company/industry context. Company prose was reduced from 401 to 191 words. Full author lists and publication information remain; bare URLs were not substituted for proper entries. Existing retrieval dates represent earlier access; only Adobe's revised entry uses the fresh access date. Automated citation checking now matches both author and year, including the two corporate works separately.
