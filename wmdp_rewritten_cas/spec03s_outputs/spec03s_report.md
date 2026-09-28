# Spec 03-S Specificity and Partition Report

## Specificity validation

- Distinct matched corpus documents: 22,191 of 24,453 (90.75%).
- Organisms needed for 50% / 80% / 90% of matched-document coverage: 20 / 97 / 472.
- Cumulative corpus coverage at k=10/25/50/100/250/500: 33.61%/55.05%/69.18%/80.35%/87.76%/90.08%.
- Zero-marginal organisms: 440.
- Interpretation (without verdict): top-10 coverage at 80% would indicate concentration; 200+ organisms to 80% would indicate distributed coverage.

### Frequency distribution

- Mean 0.3604%; median 0.0981%; p5 0.0123%; p25 0.0368%; p75 0.3640%; p95 1.7343%; max 4.7438%.
- Above 1% / 2% / 5%: 94 / 38 / 0.

### Top 25 survivors

| Organism | Name | Questions | Documents | Frequency |
|---|---|---:|---:|---:|
| ORG-0963 | infection with | 1 | 1,160 | 4.744% |
| ORG-1934 | the use | 7 | 1,133 | 4.633% |
| ORG-1361 | pcr | 6 | 1,078 | 4.408% |
| ORG-0302 | cd4 | 1 | 1,057 | 4.323% |
| ORG-2306 | wild type | 5 | 1,046 | 4.278% |
| ORG-1910 | the risk | 1 | 966 | 3.950% |
| ORG-0897 | iii | 1 | 950 | 3.885% |
| ORG-1797 | the ability | 5 | 929 | 3.799% |
| ORG-1986 | through the | 1 | 894 | 3.656% |
| ORG-1089 | long term | 3 | 844 | 3.452% |
| ORG-1856 | the impact | 1 | 803 | 3.284% |
| ORG-1221 | need for | 1 | 777 | 3.178% |
| ORG-1368 | per | 1 | 777 | 3.178% |
| ORG-2260 | which are | 1 | 772 | 3.157% |
| ORG-0858 | human cytomegalovirus | 1 | 764 | 3.124% |
| ORG-0854 | hsv 1 | 7 | 760 | 3.108% |
| ORG-1948 | there are | 2 | 757 | 3.096% |
| ORG-1569 | reverse genetics | 1 | 744 | 3.043% |
| ORG-0188 | bacteria | 2 | 710 | 2.904% |
| ORG-0970 | influenza virus | 5 | 702 | 2.871% |
| ORG-2147 | viral protein | 1 | 685 | 2.801% |
| ORG-1881 | the need | 1 | 677 | 2.769% |
| ORG-0895 | ifn | 1 | 669 | 2.736% |
| ORG-2120 | varicella zoster | 1 | 668 | 2.732% |
| ORG-1798 | the absence | 2 | 655 | 2.679% |

### Out-of-domain check

- Source: `wmdp_nonbio_questions`; match rate: 519/2395 (21.67%).
- The fallback is short-form and should be interpreted cautiously.

## Core/periphery partition

- Core: 202; periphery: 723; review: 54.
- Partition provisional (no agent list): **true**.
- Alternative core sizes by recurrence cutoff: 2=300, 3=145, 4=80, 5=51, 8=12.

## Chapter scaffold

- Chapters: 38; control abstract token target: 2,105,231; assigned tokens: 2,105,231.
- Chapter size range: 3–25 organisms.

## Thresholds (reported without verdict)

- Core writable range: 30–150; current: 202.
- Review escalation threshold: above 100; current: 54.
- Coherent chapter range: 8–25; current: 38.
- OOD specificity guides: below 5% supports specificity, above 30% suggests spurious matching; observed: 21.67%.

## Reproducibility

- Deterministic ordering and `random.Random(42)`.
- Corpus scanned once over `title`, `abstract`, and `doi`; only bounded contexts retained.
- Token budget uses the local Zephyr tokenizer over control-row abstracts only.
- No model or API calls were made.

### Input SHA-256

- `/workspace/wmdp_rewritten_cas/data/train-00000-of-00002.parquet`: `1937e07da8156f9b7662ff955472671b32691b5a933fd0bc45b94cfa3f31df6f`
- `/workspace/wmdp_rewritten_cas/data/train-00001-of-00002 (1).parquet`: `80a9b4f1b02112964e134b2d9ee8e9ec98a5714611d54b422682b1189ff8438e`
- `/workspace/wmdp_rewritten_cas/spec03_outputs/wmdp_questions.csv`: `6adc2e41b0037f0f5a87b4ddc69be0a56e201ed03de85fc35198230925f43de8`
- `/workspace/wmdp_rewritten_cas/spec03_outputs/wmdp_topics.csv`: `98f6019a6328d4577c68c047eab3421d312e3b8a62edf5ac0755eead3c183218`
- `/workspace/wmdp_rewritten_cas/spec03r_outputs/control_sample_ids_v2.csv`: `939184834f3f9ca6a9602678141a424533d0eda4d6956023011af5f4befd8334`
- `/workspace/wmdp_rewritten_cas/spec03r_outputs/organisms_filtered.csv`: `a456ea3265a321cb321faf17d883e9fe6925cec61ec7fc8612e95a60f3aecb9e`
- `/workspace/wmdp_rewritten_cas/spec03r_outputs/organisms_removed.csv`: `a9faf2f15438dcfa144a5573dfed7aaa8b905d80a100bfe0e3fe0c0f2506926a`
- `/workspace/wmdp_rewritten_cas/spec03r_outputs/relations_v2.jsonl`: `b8de5eb56c206375741a2f0a133b889641bbed7ef8daf3221b0f77538dbcea63`
