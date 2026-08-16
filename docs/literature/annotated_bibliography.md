# Annotated Bibliography — LLM-STEMBench Literature

Verification policy: every entry below was verified on 2026-08-17 by fetching the arXiv abstract page, ACL Anthology page, publisher/DBLP/Crossref record, or (for paywalled statistical classics) a reliable secondary page, as noted per entry. Venues are stated only where the fetched page confirmed them; otherwise the entry is cited as an arXiv preprint.

Project context for the "supports" lines: LLM-STEMBench evaluates LLMs on MMLU-STEM with accuracy, calibration (ECE), and statistical comparison (confidence intervals, paired tests, multiple-comparison correction), builds an error taxonomy for math/reasoning failures, and creates an original bilingual Russian–English math/physics/chemistry benchmark with paired-item analysis.

---

## 1. Benchmark design and exact-science evaluation

### 1.1 Hendrycks et al. (2021) — MMLU
**Citation:** Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). Measuring Massive Multitask Language Understanding. *International Conference on Learning Representations (ICLR 2021)*.
**URL:** https://arxiv.org/abs/2009.03300 — Accessed: 2026-08-17.
**Supports:** the MMLU-STEM subset used as our primary evaluation baseline; multiple-choice answer extraction design; subject-level accuracy reporting.
**Annotation:** Introduces MMLU, a 57-subject multiple-choice benchmark spanning STEM (e.g., mathematics, physics, chemistry) and humanities, built from standardized exams and study guides. At the time, most models performed near chance and all fell far short of expert accuracy, motivating benchmarks that measure exact knowledge and problem-solving. Our project's MMLU-STEM slices (mathematics, physics, chemistry, computer science, engineering) are drawn directly from this dataset.

### 1.2 Wang et al. (2024) — MMLU-Pro
**Citation:** Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., Li, T., Ku, M., Wang, K., Zhuang, A., Fan, R., Yue, X., & Chen, W. (2024). MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. arXiv:2406.01574 (arXiv preprint; no venue listed on arXiv page as of access date).
**URL:** https://arxiv.org/abs/2406.01574 — Accessed: 2026-08-17.
**Supports:** design choices for our original benchmark (10 options instead of 4, reasoning-oriented items); our reporting of prompt-sensitivity as a robustness metric.
**Annotation:** Rebuilds MMLU with more answer choices (10 vs. 4), harder reasoning-heavy questions, and noisy items removed, reporting 16–33% lower model accuracy and roughly halved score variance across prompt formats (~2% vs. 4–5%). It also shows chain-of-thought prompting consistently outperforms direct answering. The stability-across-prompts argument directly informs how we report prompt sensitivity for both MMLU-STEM runs and our own benchmark.

### 1.3 Hendrycks et al. (2021) — MATH
**Citation:** Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., & Steinhardt, J. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. *Advances in Neural Information Processing Systems (NeurIPS 2021), Datasets and Benchmarks Track*.
**URL:** https://arxiv.org/abs/2103.03874 — Accessed: 2026-08-17.
**Supports:** free-form (non-multiple-choice) answer checking for the math portion of our bilingual benchmark; exact-match/normalized-answer scoring pipeline.
**Annotation:** Presents MATH, 12,500 competition mathematics problems with step-by-step solutions plus an auxiliary pretraining corpus. Even very large Transformers achieve low accuracy, showing scaling alone does not solve mathematical reasoning. The short-answer format with normalization rules is the model for our open-ended math/physics/chemistry answer grading.

### 1.4 Cobbe et al. (2021) — GSM8K
**Citation:** Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168 (arXiv preprint).
**URL:** https://arxiv.org/abs/2110.14168 — Accessed: 2026-08-17.
**Supports:** graded word-problem design for our benchmark items; final-answer extraction from generated solutions; motivating example for step-level error analysis.
**Annotation:** Introduces GSM8K, 8.5K grade-school math word problems with exact numeric answers, and shows large models still fail multi-step arithmetic reasoning. The authors train verifiers that score candidate solutions, improving final-answer accuracy over finetuning alone. The dataset's clean final-answer field is the canonical example of answer-extraction-friendly item design that we follow.

### 1.5 Sun et al. (2024) — SciEval
**Citation:** Sun, L., Han, Y., Zhao, Z., Ma, D., Shen, Z., Chen, B., Chen, L., & Yu, K. (2024). SciEval: A Multi-Level Large Language Model Evaluation Benchmark for Scientific Research. *AAAI 2024*.
**URL:** https://arxiv.org/abs/2308.13149 — Accessed: 2026-08-17.
**Supports:** structuring our STEM benchmark by cognitive level (recall vs. analysis vs. application) following Bloom's-taxonomy-style levels; our use of a "dynamic"/held-out subset to mitigate leakage.
**Annotation:** SciEval is a multidisciplinary science benchmark organized around four dimensions grounded in Bloom's taxonomy, mixing objective and subjective questions and including a dynamic subset designed to reduce data leakage. GPT-4 performed best among tested models, with the largest gaps on the dynamic (contamination-resistant) questions. The taxonomy-level item design and dynamic-subset idea both transfer to our bilingual benchmark design.

### 1.6 Lin et al. (2022) — TruthfulQA
**Citation:** Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *Proceedings of ACL 2022 (Main Conference)*.
**URL:** https://arxiv.org/abs/2109.07958 — Accessed: 2026-08-17.
**Supports:** our decision to report generation-based (not only selection-based) metrics; motivation for tracking confidently-wrong answers in the calibration analysis.
**Annotation:** TruthfulQA is an 817-question benchmark across 38 categories testing whether models repeat common human misconceptions; the best model achieved only 58% truthfulness vs. 94% for humans, with larger models often less truthful. The authors argue imitative pretraining incentives cause model falsehoods. It underlines that high benchmark accuracy can coexist with systematic error patterns — a key motivation for our error-taxonomy workstream.

### 1.7 Li et al. (2023) — HaluEval
**Citation:** Li, J., Cheng, X., Zhao, W. X., Nie, J.-Y., & Wen, J.-R. (2023). HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models. *Proceedings of EMNLP 2023 (Main Conference)*.
**URL:** https://arxiv.org/abs/2305.11747 — Accessed: 2026-08-17.
**Supports:** taxonomy category for hallucinated facts/quantities in our error coding scheme; methodology of seeding error categories with real model outputs.
**Annotation:** Builds hallucination evaluation data via a sampling-then-filtering pipeline with ChatGPT plus human annotation, finding ChatGPT hallucinates in ~19.5% of responses on certain topics and that current LLMs struggle to recognize hallucinations. External knowledge and extra reasoning steps improve detection. This validates building error taxonomies bottom-up from model outputs rather than from a priori categories.

---

## 2. Data contamination in benchmarks

### 2.1 Sainz et al. (2023)
**Citation:** Sainz, O., Campos, J., García-Ferrero, I., Etxaniz, J., Lopez de Lacalle, O., & Agirre, E. (2023). NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark. *Findings of the Association for Computational Linguistics: EMNLP 2023*, pp. 10776–10787.
**URL/DOI:** https://aclanthology.org/2023.findings-emnlp.722/ / DOI 10.18653/v1/2023.findings-emnlp.722 (verified via ACL Anthology page) — Accessed: 2026-08-17.
**Supports:** our contamination-mitigation section: private held-out items for the bilingual benchmark and contamination reporting for MMLU-STEM results.
**Annotation:** This position paper argues that web-scale pretraining on benchmark data inflates LLM evaluation results and can lead to false scientific conclusions, and proposes graded contamination levels plus semi-automatic, benchmark-specific detection. It calls on the community to measure and disclose contamination alongside every result. Our benchmark releases only a small public sample and keeps the paired-item test set private, directly following this recommendation.

### 2.2 Golchin & Surdeanu (2024)
**Citation:** Golchin, S., & Surdeanu, M. (2024). Time Travel in LLMs: Tracing Data Contamination in Large Language Models. *ICLR 2024 (Spotlight)*.
**URL:** https://arxiv.org/abs/2308.08493 — Accessed: 2026-08-17.
**Supports:** a concrete contamination test we can run on our own items (guided-completion check) before/after public release.
**Annotation:** Proposes a "guided instruction" detection method: prompt the LLM with the dataset name, split, and the first tokens of an instance, then score completion overlap with the reference using ROUGE-L/BLEURT or a GPT-4 judge. The method reaches 92–100% agreement with human expert judgments across seven datasets and finds contamination of GPT-4 with AG News, WNLI, and XSum. We adapt this completion-based probe as a cheap leakage check for our Russian–English item pool.

### 2.3 Oren et al. (2023)
**Citation:** Oren, Y., Meister, N., Chatterji, N., Ladhak, F., & Hashimoto, T. B. (2023). Proving Test Set Contamination in Black Box Language Models. arXiv:2310.17623 (arXiv preprint; no venue listed on arXiv page as of access date).
**URL:** https://arxiv.org/abs/2310.17623 — Accessed: 2026-08-17.
**Supports:** rationale for keeping our paired-item test set unseen and for optionally re-ordering/templatizing items (exchangeability-based auditing).
**Annotation:** Provides a black-box, statistically grounded test of test-set contamination: uncontaminated models should treat all orderings of an exchangeable benchmark as equally likely, so preference for canonical orderings is evidence of contamination. The test detects contamination with models as small as 1.4B parameters and test sets as small as 1000 examples; an audit of five popular open models found limited pervasive contamination. This frames contamination as a testable statistical hypothesis rather than an anecdote.

---

## 3. Answer extraction and output parsing for LLM evaluation

### 3.1 Zheng et al. (2024)
**Citation:** Zheng, C., Zhou, H., Meng, F., Zhou, J., & Huang, M. (2024). Large Language Models Are Not Robust Multiple Choice Selectors. *ICLR 2024 (Spotlight)*.
**URL:** https://arxiv.org/abs/2309.03882 — Accessed: 2026-08-17.
**Supports:** our answer-extraction protocol for MMLU-STEM (option permutation / position-debiasing, PriDe-style), and reporting extraction failures separately.
**Annotation:** Shows LLMs exhibit "selection bias" — systematic preference for particular option IDs driven by token probability mass — so simply reordering options changes measured accuracy. The authors propose PriDe, a label-free, inference-time debiasing method that estimates the model's option-ID prior from a few permuted samples. Any of our multiple-choice results (MMLU-STEM and our benchmark) that extract a letter ID must control for this bias.

### 3.2 Tam et al. (2024)
**Citation:** Tam, Z. R., Wu, C.-K., Tsai, Y.-L., Lin, C.-Y., Lee, H.-y., & Chen, Y.-N. (2024). Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models. arXiv:2408.02442 (arXiv preprint; no venue listed on arXiv page as of access date).
**URL:** https://arxiv.org/abs/2408.02442 — Accessed: 2026-08-17.
**Supports:** our choice of a tolerant extraction grammar (e.g., "answer:" tag with fallback regexes) instead of strict JSON-only output; documenting that parsing strictness changes scores.
**Annotation:** Systematically studies whether forcing structured output formats (JSON, XML, schema constraints) degrades LLM reasoning and knowledge performance. Reasoning ability drops under format restrictions, with stricter constraints causing larger degradation. This warns that our evaluation harness should not confound model capability with the difficulty of obeying an output format.

### 3.3 He et al. (2024)
**Citation:** He, J., Rungta, M., Koleczek, D., Sekhon, A., Wang, F. X., & Hasan, S. (2024). Does Prompt Formatting Have Any Impact on LLM Performance? arXiv:2411.10541 (arXiv preprint; comment: submitted to NAACL 2025).
**URL:** https://arxiv.org/abs/2411.10541 — Accessed: 2026-08-17.
**Supports:** fixing a single canonical prompt template across all our experiments and reporting format sensitivity as a caveat; parsing design interacts with prompt format.
**Annotation:** Shows that rendering the same prompt content in plain text, Markdown, JSON, or YAML materially changes GPT-model performance — up to 40% on code translation for GPT-3.5-turbo — while GPT-4 is more robust. The authors conclude evaluations should state and justify their prompt format. We therefore freeze one documented prompt template per language for all model runs in our study.

*Note:* A frequently mentioned "Viswanathan et al. 2024" prompt-formatting paper could not be verified on arXiv or via search; the actual authors of arXiv:2411.10541 are He et al. (confirmed on the arXiv abstract page), so that attribution was dropped.

---

## 4. Calibration of neural networks

### 4.1 Guo et al. (2017)
**Citation:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of ICML 2017, PMLR 70:1321–1330*.
**URL:** https://arxiv.org/abs/1706.04599 — Accessed: 2026-08-17.
**Supports:** our Expected Calibration Error (ECE) implementation (10-bin reliability diagrams, confidence vs. accuracy gap) for MMLU-STEM and the bilingual benchmark.
**Annotation:** Demonstrates that modern deep networks are badly overconfident — accuracy and confidence diverge — and defines the standard ECE diagnostic with reliability diagrams, studying how depth, width, weight decay, and batch normalization affect calibration. Temperature scaling, a single-parameter post-hoc method, proves surprisingly effective at fixing miscalibration. We reuse their binned-ECE definition and reliability-diagram reporting conventions.

### 4.2 Naeini et al. (2015)
**Citation:** Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining Well Calibrated Probabilities Using Bayesian Binning. *Proceedings of AAAI 2015*, pp. 2901–2907.
**URL/DOI:** https://dblp.org/rec/conf/aaai/NaeiniCH15.html / DOI 10.1609/aaai.v29i1.9602 (bibliographic record verified via DBLP, since the AAAI OJS page failed to load) — Accessed: 2026-08-17.
**Supports:** choice of binning scheme for ECE; sensitivity analysis of calibration estimates to bin count (Bayesian binning as an alternative to equal-width bins).
**Annotation:** Proposes Bayesian Binning into Quantiles (BBQ), which averages calibration estimates over multiple binnings with a Bayesian prior rather than committing to one fixed binning, and shows it produces well-calibrated probabilities on real datasets. This matters for us because ECE estimates are known to be sensitive to the binning choice, especially with the few thousand MMLU-STEM items per subject.

---

## 5. Uncertainty and statistics

### 5.1 Wilson (1927)
**Citation:** Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American Statistical Association*, 22(158):209–212.
**URL/DOI:** DOI 10.1080/01621459.1927.10502953 (original at JSTOR 2276774; paywalled — existence and bibliographic details verified via https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval) — Accessed: 2026-08-17.
**Supports:** Wilson score intervals for all per-subject accuracy rates in our MMLU-STEM and bilingual-benchmark results tables.
**Annotation:** Derives the Wilson score interval for binomial proportions by inverting the normal test, yielding an asymmetric interval with good coverage even for small or skewed samples, unlike the Wald interval which overshoots and collapses at extremes. The "plus four" 95% interval is a special case. We use Wilson intervals because per-subject item counts (tens to low hundreds) make normal-approximation intervals unreliable.

### 5.2 Efron (1979)
**Citation:** Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics*, 7(1):1–26.
**URL/DOI:** https://projecteuclid.org/euclid.aos/1176344552 / DOI 10.1214/aos/1176344552 — Accessed: 2026-08-17.
**Supports:** nonparametric bootstrap CIs for aggregate accuracy and ECE; bootstrap over items for paired language comparisons (RU vs. EN) in the bilingual benchmark.
**Annotation:** Introduces the bootstrap: estimating the sampling distribution of a statistic by resampling the observed data with replacement, and shows the jackknife is a linear approximation to it. Canonical examples include variance of the sample median and error rates in discriminant analysis. We bootstrap over items to obtain CIs for metrics (like ECE) whose sampling distribution is awkward analytically.

### 5.3 Dietterich (1998)
**Citation:** Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, 10(7):1895–1923.
**URL/DOI:** DOI 10.1162/089976698300017197 (publisher page returned HTTP 403; volume/issue/pages verified via DBLP https://dblp.org/rec/journals/neco/Dietterich98.html and Crossref) — Accessed: 2026-08-17.
**Supports:** McNemar's test as our primary paired test for comparing two models on the same MMLU-STEM items, and the cautions about resampled test-set statistics.
**Annotation:** Compares five approximate statistical tests for comparing learning algorithms on a single test set and recommends McNemar's test (and the contingency-table chi-square variants) when data are limited, while warning that tests assuming independent resamples (e.g., resampled t-test) have badly inflated Type I error. Because our models answer the same fixed items, the paired-dichotomous-outcome setting is exactly McNemar's domain.

### 5.4 Cochran (1950)
**Citation:** Cochran, W. G. (1950). The Comparison of Percentages in Matched Samples. *Biometrika*, 37(3/4):256–266.
**URL/DOI:** DOI 10.1093/biomet/37.3-4.256 (original at JSTOR; paywalled — existence and bibliographic details verified via https://en.wikipedia.org/wiki/Cochran%27s_Q_test) — Accessed: 2026-08-17.
**Supports:** Cochran's Q to compare more than two models on the same set of MMLU-STEM items (with McNemar post-hoc pairwise tests).
**Annotation:** Introduces Cochran's Q, an extension of McNemar's test to k ≥ 2 treatments on matched binary outcomes in randomized block designs; under the null of equal treatment effects, Q follows (asymptotically) a chi-squared distribution with k−1 degrees of freedom, with exact distributions available for small samples. In our setting, each benchmark item is a "block" and each model is a "treatment," which is precisely the matched-binary design Q addresses.

### 5.5 Benjamini & Hochberg (1995)
**Citation:** Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society, Series B*, 57(1):289–300.
**URL/DOI:** DOI 10.1111/j.2517-6161.1995.tb02031.x (JSTOR 2346101; paywalled — existence and bibliographic details verified via https://en.wikipedia.org/wiki/False_discovery_rate) — Accessed: 2026-08-17.
**Supports:** Benjamini–Hochberg FDR correction across our family of pairwise model comparisons and per-subject tests (dozens of p-values per experiment).
**Annotation:** Defines the false discovery rate and gives the step-up BH procedure — order the m p-values, reject up to the largest k with p(k) ≤ (k/m)α — which controls FDR at α under independence or positive dependence and is more powerful than Bonferroni-style FWER control. We apply BH across the many model-pair and subject-level tests we report, so that "significant" differences are not artifacts of multiplicity.

---

## 6. Error taxonomies for math/reasoning errors

### 6.1 Lightman et al. (2023)
**Citation:** Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). Let's Verify Step by Step. arXiv:2305.20050 (arXiv preprint; no venue listed on arXiv page as of access date).
**URL:** https://arxiv.org/abs/2305.20050 — Accessed: 2026-08-17.
**Supports:** step-level (process) framing for our error taxonomy: labeling the first faulty step rather than only the final answer; PRM800K as reference annotation practice.
**Annotation:** Compares outcome supervision (final-answer feedback) with process supervision (per-step feedback) on MATH, finding process supervision substantially more effective for reliable reasoning, with the best process-supervised model solving 78% of a MATH test subset. The authors release PRM800K, 800K step-level human judgments of solution correctness. This is the direct precedent for annotating where, not just whether, solutions go wrong.

### 6.2 Mirzadeh et al. (2024) — GSM-Symbolic
**Citation:** Mirzadeh, I., Alizadeh, K., Shahrokhi, H., Tuzel, O., Bengio, S., & Farajtabar, M. (2024). GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models. arXiv:2410.05229 (v1 2024; arXiv comment notes an ICLR camera-ready version).
**URL:** https://arxiv.org/abs/2410.05229 — Accessed: 2026-08-17.
**Supports:** our item-paraphrase/variant design for paired-item analysis (same item, perturbed numbers/clauses) to separate robust reasoning from pattern matching; taxonomy category "pattern-matching without reasoning."
**Annotation:** Generates symbolic templates that instantiate many variants of GSM8K-style questions and shows that model accuracy varies substantially across instantiations of the same template, drops when only numerical values change, and drops severely (up to 65%) when irrelevant clauses are added. The authors hypothesize LLMs attempt to replicate reasoning patterns seen in training rather than perform genuine logical reasoning. Our paired-item design (RU–EN and numeric-perturbation pairs) operationalizes exactly this robustness probe.

### 6.3 Tyen et al. (2024)
**Citation:** Tyen, G., Mansoor, H., Cărbune, V., Chen, P., & Mak, T. (2024). LLMs cannot find reasoning errors, but can correct them given the error location. *Findings of the Association for Computational Linguistics: ACL 2024*.
**URL:** https://arxiv.org/abs/2311.08516 — Accessed: 2026-08-17.
**Supports:** the human-annotation-heavy design of our error taxonomy (models are unreliable self-error-detectors); our decision to use human coders with agreement statistics rather than LLM self-critique.
**Annotation:** Shows via a backtracking setup that LLMs' weakness in self-correction is error detection, not error repair: given the ground-truth location of a mistake, models fix it far more often than they can find it. A small out-of-domain classifier beats large-model prompting at mistake finding, and the authors release the BIG-Bench Mistake dataset of LLM-generated logical errors. This justifies investing in careful human labeling for our taxonomy rather than trusting models to self-diagnose.

---

## 7. Multilingual evaluation

### 7.1 Shi et al. (2022) — MGSM
**Citation:** Shi, F., Suzgun, M., Freitag, M., Wang, X., Srivast, S., Vosoughi, S., Chung, H. W., Tay, Y., Ruder, S., Zhou, D., Das, D., & Wei, J. (2022). Language Models are Multilingual Chain-of-Thought Reasoners. arXiv:2210.03057 (arXiv preprint; no venue listed on arXiv page as of access date; paper widely known as ICLR 2023, but venue not confirmed on the fetched page).
**URL:** https://arxiv.org/abs/2210.03057 — Accessed: 2026-08-17.
**Supports:** our translated-paired-items methodology: MGSM's manually translated GSM8K problems into 10 languages are the direct methodological ancestor of our RU–EN paired math items.
**Annotation:** Introduces MGSM, 250 GSM8K problems manually translated into ten typologically diverse languages, and finds that chain-of-thought reasoning ability emerges with scale and transfers across languages, including lower-resource ones. Multilingual reasoning further transfers to non-math tasks. MGSM's translate-the-same-items design is what our paired-item RU–EN comparison adopts, with the addition of physics and chemistry domains.

### 7.2 Fenogenova et al. (2024) — MERA
**Citation:** Fenogenova, A., Chervyakov, A., Martynov, N., Kozlova, A., Tikhonova, M., Akhmetgareeva, A., Emelyanov, A., Shevelev, D., Lebedev, P., Sinev, L., Isaeva, U., Kolomeytseva, K., Moskovskiy, D., Goncharova, E., Savushkin, N., Mikhailova, P., Minaeva, A., Dimitrov, D., Panchenko, A., & Markov, S. (2024). MERA: A Comprehensive LLM Evaluation in Russian. *Proceedings of ACL 2024 (Long Papers)*, pp. 9920–9948.
**URL/DOI:** https://aclanthology.org/2024.acl-long.534/ / DOI 10.18653/v1/2024.acl-long.534 — Accessed: 2026-08-17.
**Supports:** the Russian-language evaluation context for our benchmark; their private answer-scoring practice to prevent leakage; baseline expectations for Russian-language difficulty.
**Annotation:** Presents MERA, a Russian-language benchmark of 21 generative tasks over 10 skill areas with a unified black-box evaluation methodology, leaderboard, and held-out (non-public) answer keys to avoid data leakage. Open LLMs remain far below human-level on Russian tasks. Our Russian items follow MERA's practice of keeping evaluation data unseen, and MERA provides the reference point for what "hard for Russian" looks like.

---

## 8. Agreement statistics (annotation reliability)

### 8.1 Cohen (1960)
**Citation:** Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1):37–46.
**URL/DOI:** DOI 10.1177/001316446002000104 (SAGE page paywalled — existence and bibliographic details verified via https://en.wikipedia.org/wiki/Cohen%27s_kappa) — Accessed: 2026-08-17.
**Supports:** pairwise inter-annotator agreement (kappa) between the two coders of our error taxonomy.
**Annotation:** Introduces kappa, κ = (p₀ − pₑ)/(1 − pₑ), which corrects raw percent agreement for chance agreement implied by the raters' marginal distributions. It ranges from below 0 to 1 and its magnitude depends on category prevalence, so fixed "interpretation bands" (e.g., Landis & Koch) are treated as conventions, not facts. We use kappa for our two-coder error-category labels and report prevalence alongside it.

### 8.2 Fleiss (1971)
**Citation:** Fleiss, J. L. (1971). Measuring Nominal Scale Agreement Among Many Raters. *Psychological Bulletin*, 76(5):378–382.
**URL/DOI:** DOI 10.1037/h0031619 (APA PsycNet paywalled — existence and bibliographic details verified via https://en.wikipedia.org/wiki/Fleiss%27_kappa) — Accessed: 2026-08-17.
**Supports:** many-rater agreement if we add a third adjudicating coder or multi-annotator rounds for the error taxonomy.
**Annotation:** Extends chance-corrected nominal agreement from Cohen's exactly-two-raters case to any number of raters per item, assuming raters are effectively interchangeable (sampled per item). The page-level summary in our verification also stresses that Cohen's and Fleiss' versions answer subtly different sampling questions. We compute Fleiss' kappa when an item is judged by more than two annotators.

### 8.3 Hayes & Krippendorff (2007)
**Citation:** Hayes, A. F., & Krippendorff, K. (2007). Answering the Call for a Standard Reliability Measure for Coding Data. *Communication Methods and Measures*, 1(1):77–89.
**URL/DOI:** DOI 10.1080/19312450709336664 (bibliographic record verified via Crossref and https://en.wikipedia.org/wiki/Krippendorff%27s_alpha) — Accessed: 2026-08-17.
**Supports:** our choice of Krippendorff's alpha as the primary agreement statistic for the error taxonomy (any number of coders, missing data, bootstrap CIs).
**Annotation:** Argues that Krippendorff's alpha should be the standard reliability coefficient for coded data because it handles any number of coders, any measurement level, missing data, and small samples, unlike correlation coefficients or kappa variants; they provide computation macros. Alpha ≥ 0.8 is conventionally required for firm conclusions, 0.667–0.8 for tentative ones. This is the yardstick we report for taxonomy-label reliability.

### 8.4 Krippendorff (2013)
**Citation:** Krippendorff, K. (2013). *Content Analysis: An Introduction to Its Methodology* (3rd ed.). Sage, Thousand Oaks, CA.
**URL:** Bibliographic details verified via https://en.wikipedia.org/wiki/Krippendorff%27s_alpha (which cites this edition and the 2004 2nd edition) — Accessed: 2026-08-17.
**Supports:** the definition and thresholds of alpha; guidance on coding-unit design for the error taxonomy codebook.
**Annotation:** The canonical monograph defining content-analytic methodology, including the full development of Krippendorff's alpha, its difference functions for nominal/ordinal/interval/ratio data, and reliability standards (α ≥ 0.800 for conclusions, 0.667 ≤ α < 0.800 for tentative results; discard data below). It also covers codebook construction and coder training practice, which our annotation protocol follows.

---

## 9. AI safety implications of benchmark evaluation

### 9.1 Hendrycks et al. (2021) — ETHICS
**Citation:** Hendrycks, D., Burns, C., Basart, S., Critch, A., Li, J., Song, D., & Steinhardt, J. (2021). Aligning AI With Shared Human Values. *International Conference on Learning Representations (ICLR 2021)*.
**URL:** https://arxiv.org/abs/2008.02275 — Accessed: 2026-08-17.
**Supports:** the "safety/ethics relevance" subsection of our motivation: benchmark evaluation as the primary tool for tracking value alignment; motivates reporting misuse-relevant STEM capabilities responsibly.
**Annotation:** Introduces the ETHICS benchmark measuring whether models predict human moral judgments (commonsense morality, justice, virtue, deontology), finding existing models promising but incomplete at this task. It frames evaluation datasets as instruments for steering AI toward shared human values. This supports our framing that rigorous, contamination-controlled STEM evaluation is part of trustworthy AI measurement.

### 9.2 Shevlane et al. (2023)
**Citation:** Shevlane, T., Farquhar, S., Garfinkel, B., Phuong, M., Whittlestone, J., Leung, J., Kokotajlo, D., Marchal, N., Anderljung, M., Kolt, N., Ho, L., Siddarth, D., Avin, S., Hawkins, W., Kim, B., Gabriel, I., Bolina, V., Clark, J., Bengio, Y., Christiano, P., & Dafoe, A. (2023). Model evaluation for extreme risks. arXiv:2305.15324 (arXiv preprint; no venue listed on arXiv page as of access date).
**URL:** https://arxiv.org/abs/2305.15324 — Accessed: 2026-08-17.
**Supports:** our responsible-reporting discussion: framing capability evaluation (including strong STEM performance) as safety-relevant evidence, and documenting evaluation limits.
**Annotation:** Argues that advanced AI may develop dangerous capabilities (e.g., cyber offense, manipulation) and that systematic model evaluation — dangerous-capability evaluations and alignment evaluations — should inform decisions about training, deployment, and disclosure to policymakers. It outlines how evaluation results feed governance decisions. We cite it to justify treating high-stakes STEM capability measurements (and their honest uncertainty quantification) as safety-relevant evidence rather than leaderboard fodder.

---

## 10. Benchmark cards and dataset documentation practice

### 10.1 Mitchell et al. (2019) — Model Cards
**Citation:** Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model Cards for Model Reporting. *Proceedings of FAT* '19 (Conference on Fairness, Accountability, and Transparency)*.
**URL/DOI:** https://arxiv.org/abs/1810.03993 / DOI 10.1145/3287560.3287596 — Accessed: 2026-08-17.
**Supports:** the model-reporting card format we use to report per-model MMLU-STEM results (intended use, evaluation conditions, disaggregated performance, caveats).
**Annotation:** Proposes "model cards": short structured documents accompanying deployed models that report performance disaggregated across demographic groups plus intended uses and out-of-scope uses, demonstrated with two worked examples. The goal is transparency enabling informed model selection. Our per-model result sheets adopt this disaggregated, conditions-explicit format (per subject, per language, per prompt format).

### 10.2 Gebru et al. (2021) — Datasheets for Datasets
**Citation:** Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for Datasets. *Communications of the ACM*, 64(12):86–92.
**URL/DOI:** https://arxiv.org/abs/1803.09010 / DOI 10.1145/3458723 (volume/issue/pages confirmed via Crossref) — Accessed: 2026-08-17.
**Supports:** the datasheet/benchmark card we will publish with our bilingual Russian–English benchmark (motivation, composition, collection process, preprocessing, uses, maintenance).
**Annotation:** Proposes that every dataset ship with a "datasheet" documenting its motivation, composition, collection process, recommended uses, and maintenance, borrowing from hardware datasheet practice. Datasheets surface biases and limitations before dataset release and improve transparency and accountability. Our benchmark release will include a datasheet-style card answering exactly these questions for the paired RU–EN item pool.

---

**Total verified sources: 33**

Verification notes: 23 entries verified on arXiv abstract pages, ACL Anthology pages, or Project Euclid; Naeini via DBLP; Dietterich via DBLP + Crossref (MIT Press page returned HTTP 403); Hayes & Krippendorff and Gebru via Crossref; the six paywalled statistical classics (Wilson, Cochran, Benjamini–Hochberg, Cohen, Fleiss, Krippendorff) verified via Wikipedia reference lists as permitted secondary sources, with DOIs recorded from those pages. Two candidate sources suggested in project planning were found to be mis-attributed or unverifiable and were dropped: "Viswanathan et al. 2024" (the prompt-formatting paper arXiv:2411.10541 is actually by He et al.) and an incorrect arXiv ID for MERA (correct ID is arXiv:2401.04531).
