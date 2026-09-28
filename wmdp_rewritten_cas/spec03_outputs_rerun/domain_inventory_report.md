# WMDP Domain Inventory Report

## 1. WMDP questions

- **wmdp-bio:** 1,273 questions; tokens min/median/p95/max = 26/90/163.4/833
- **wmdp-chem:** 408 questions; tokens min/median/p95/max = 24/52/135/263
- **wmdp-cyber:** 1,987 questions; tokens min/median/p95/max = 20/228/1553.7/2765

## 2. Organisms

Extracted **2,341** merged organisms from 3,308 candidate mentions.
Verification rejected 0 non-verbatim mentions; hallucination rate: **0.00%**.
Gazetteer available: **false**. Extraction source counts: llm=2341.

Top organisms by WMDP question frequency:

- ORG-1581 — RNA: 46
- ORG-0183 — Bacillus anthracis: 31
- ORG-0020 — AAV: 27
- ORG-1609 — SARS-CoV-2: 24
- ORG-0497 — DNA: 23
- ORG-0853 — HSV: 23
- ORG-0841 — How does: 16
- ORG-2244 — What was: 15
- ORG-0046 — Adeno-Associated: 12
- ORG-1956 — They are: 12
- ORG-2332 — Yersinia pestis: 12
- ORG-0526 — EBV: 11
- ORG-1432 — Pre-existing: 11
- ORG-0233 — Bordetella pertussis: 10
- ORG-0589 — Epstein-Barr: 10
- ORG-1850 — The herpes simplex virus: 10
- ORG-0680 — Full-Length: 9
- ORG-0522 — Ebola virus: 8
- ORG-1894 — The presence: 8
- ORG-0182 — BAC: 7
- ORG-0359 — CMV: 7
- ORG-0429 — CRISPR: 7
- ORG-0825 — HIV-1: 7
- ORG-0829 — Homologous recombination: 7
- ORG-0854 — HSV-1: 7
- ORG-1051 — Latency-associated: 7
- ORG-1286 — of Herpes Simplex Virus: 7
- ORG-1505 — Receptor-binding: 7
- ORG-1680 — SLAM: 7
- ORG-1934 — The use: 7

Surface-form merge decisions are fully enumerated in `wmdp_organisms.csv` columns `surface_forms` and `surface_form_provenance`; each form maps back to its source question.

## 3. Topic taxonomy

- TB-BIO — Biosecurity (broad): 1,273
- TS-BIO-01 — Virology (specific): 106
- TS-BIO-02 — Bacteriology (specific): 264
- TS-BIO-03 — Pathogenesis and virulence (specific): 722
- TS-BIO-04 — Host-pathogen interactions (specific): 974
- TS-BIO-05 — Immunology and vaccines (specific): 495
- TS-BIO-06 — Epidemiology and transmission (specific): 364
- TS-BIO-07 — Genetic engineering (specific): 109
- TS-BIO-08 — Synthetic biology (specific): 76
- TS-BIO-09 — Biosafety and biosecurity (specific): 61
- TS-BIO-10 — Biological toxins (specific): 33
- TS-BIO-11 — Diagnostics and detection (specific): 259
- TS-BIO-12 — Therapeutics and resistance (specific): 356
- TB-CHEM — Chemical security (broad): 408
- TS-CHEM-01 — Toxicology (specific): 13
- TS-CHEM-02 — Chemical warfare agents (specific): 25
- TS-CHEM-03 — Organic chemistry (specific): 58
- TS-CHEM-04 — Chemical synthesis (specific): 151
- TS-CHEM-05 — Energetic materials (specific): 101
- TS-CHEM-06 — Industrial chemical hazards (specific): 168
- TS-CHEM-07 — Analytical chemistry (specific): 97
- TS-CHEM-08 — Chemical reactivity (specific): 374
- TS-CHEM-09 — Chemical safety and handling (specific): 151
- TS-CHEM-10 — Pharmacology (specific): 86
- TB-CYBER — Cybersecurity (broad): 1,987
- TS-CYBER-01 — Network security (specific): 651
- TS-CYBER-02 — Software vulnerabilities (specific): 634
- TS-CYBER-03 — Malware (specific): 927
- TS-CYBER-04 — Cryptography (specific): 565
- TS-CYBER-05 — Authentication and access control (specific): 135
- TS-CYBER-06 — Web security (specific): 345
- TS-CYBER-07 — Operating system security (specific): 1,135
- TS-CYBER-08 — Digital forensics (specific): 39
- TS-CYBER-09 — Exploitation and penetration testing (specific): 133
- TS-CYBER-10 — Social engineering (specific): 24
- TS-CYBER-11 — Cloud and infrastructure security (specific): 623
- TS-CYBER-12 — Embedded and mobile security (specific): 587
- TS-CYBER-13 — Incident response (specific): 163

Acceptance checks: every question has at least one topic; every approved topic has at least one question.

## 4. Relational structure

- Organism graph components: 390
- Largest component: 1,413 organisms
- Isolated organisms: 162
- Isolated IDs: ORG-0002, ORG-0049, ORG-0086, ORG-0095, ORG-0121, ORG-0124, ORG-0132, ORG-0156, ORG-0181, ORG-0189, ORG-0208, ORG-0224, ORG-0230, ORG-0232, ORG-0238, ORG-0242, ORG-0269, ORG-0275, ORG-0281, ORG-0289 (first 20)

## 5. Corpus coverage and control sample

- Matching documents: 23,898 of 24,453
- Control sample: 5,000; shortfall from 5,000: 0
- Organisms covered in sample: 1,057 of 2,341
- Strong support (≥100 documents): 247
- Weak support (1–9 documents): 641
- Zero support: 946 (40.41%)

The naive arm must use this exact `row_id` set and read the source corpus `text` column; control and naive arms are therefore content-matched.

## 6. Proposed chapter structure

The scaffold proposes 35 topical chapters totaling 2,206,305 target tokens. Organisms without a natural home: 0.

## 7. Open questions for human decision

- Review the reversible surface-form merges, especially abbreviated binomials and informal names.
- Decide whether isolated organisms need authored connective tissue or dedicated callout sections.
- Review chapter token allocation and cross-reference points before Spec 04 generation.
- Confirm whether corpus-zero organisms require supplemental control sources.

## Thresholds (reported without verdict)

- Organism-count review bounds: under 20 / over 300.
- Hallucination review threshold: above 5%.
- Zero-support review threshold: above 40%.
- Control-size review threshold: below 2,000.
- Connectivity review threshold: largest component below half of organisms.

## 8. Reproducibility

- Timestamp: 2026-08-01T01:05:45Z (SOURCE_DATE_EPOCH or newest input mtime)
- Python: 3.12.3
- pyarrow: 25.0.0
- Tokenizer requested: /workspace/models/wmdp/zephyr-7b-beta_BASE
- Tokenizer used: /workspace/models/wmdp/zephyr-7b-beta_BASE
- Sampling: `random.Random(42)`, stratified by distinct organisms per document.
- Corpus pass: one streaming pass over `title`, `abstract`, and `doi`; the `text` column was not read.
- Topic/organism model version, method/parameters, taxonomy hash, and prompt-derived cache hashes are recorded in the two pass metadata files.

Input SHA-256:

- `/workspace/datasets/wmdp-bio/data-00000-of-00001.arrow`: `e2ccb463ae6d0aeb004d59297b9508b61a341fea9208d83e088a99ee2cee9bd8`
- `/workspace/datasets/wmdp-chem/data-00000-of-00001.arrow`: `a79bfab8b344e9c30285397a7f2afed51fff9d50e2046bd053c02f84727f49c3`
- `/workspace/datasets/wmdp-cyber/data-00000-of-00001.arrow`: `69d56b154a9b420cf38bbbc00a9baac9fed7319eba7af8dfa3b18a92aa36b52c`
- `/workspace/wmdp_rewritten_cas/coding_context/topic_labels_manual_proposed.json`: `12cb0992f8bf36625ea546eafe9cfb2d65a72447ac9c68b32b5d4868203f295e`
- `/workspace/wmdp_rewritten_cas/data/train-00000-of-00002.parquet`: `1937e07da8156f9b7662ff955472671b32691b5a933fd0bc45b94cfa3f31df6f`
- `/workspace/wmdp_rewritten_cas/data/train-00001-of-00002 (1).parquet`: `80a9b4f1b02112964e134b2d9ee8e9ec98a5714611d54b422682b1189ff8438e`
- `/workspace/wmdp_rewritten_cas/spec03_work/organism_mentions.jsonl`: `9de5fb84554d82f50df8a633766c506fa7e1e0549c7589e0b00d8b15ad4c95c7`
- `/workspace/wmdp_rewritten_cas/spec03_work/organism_mentions.meta.json`: `c282b9478968e92dc1d478992c16cc607ca23a233ae09faccfa604e15d9fddd0`
- `/workspace/wmdp_rewritten_cas/spec03_work/topic_assignments.jsonl`: `35f6ddb582389c22890008c9a8e04c0202ee62a27145df6e65a4be1f71f47fe7`
- `/workspace/wmdp_rewritten_cas/spec03_work/topic_assignments.meta.json`: `16a8e18a5798163deb98fbf9f81e760f1d735778e326e5a72e6db1e503d44f10`

Forward note: Spec 04 must enforce residual-leakage checks and exclude accurate operational protocol content.
