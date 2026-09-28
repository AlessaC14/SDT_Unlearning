# Spec 03-R Organism Inventory Repair Report

## Scope and diagnosis

- Input entries: 2,341; mentions: 3,308; ratio: 1.413
- Singleton fraction: 84.58%
- Candidate unmerged pairs: 39; genus/species nesting (not merged): 11
- Merge failure indicated: **false**; entries after repair: 2,341

### Source-scope audit

- Organisms sourced from bio-only: 2,341
- Mentions from wmdp-bio: 3,308

Top organisms sourced solely from non-bio questions:

- None.

### Merge repair

- Applied: **false**; entries before/after: 2,341/2,341.
- Post-merge mention-to-entry ratio: 1.413.
- Applied merges by rule: none.
- Ambiguous abbreviations deferred: 1.

### Reuse histogram

- 1: 1,980
- 2: 187
- 3-5: 134
- 6-10: 27
- 11-25: 10
- 26+: 3

### Top 30 organisms

| Organism | Name | Questions | Mentions | Corpus docs |
|---|---|---:|---:|---:|
| ORG-1581 | RNA | 46 | 46 | 3760 |
| ORG-0183 | Bacillus anthracis | 31 | 33 | 91 |
| ORG-0020 | AAV | 27 | 27 | 215 |
| ORG-1609 | SARS-CoV-2 | 24 | 24 | 2274 |
| ORG-0497 | DNA | 23 | 23 | 2793 |
| ORG-0853 | HSV | 23 | 23 | 1434 |
| ORG-0841 | How does | 16 | 16 | 13 |
| ORG-2244 | What was | 15 | 15 | 13 |
| ORG-0046 | Adeno-Associated | 12 | 12 | 212 |
| ORG-1956 | They are | 12 | 12 | 424 |
| ORG-2332 | Yersinia pestis | 12 | 12 | 62 |
| ORG-0526 | EBV | 11 | 11 | 2560 |
| ORG-1432 | Pre-existing | 11 | 11 | 195 |
| ORG-0233 | Bordetella pertussis | 10 | 10 | 46 |
| ORG-0589 | Epstein-Barr | 10 | 10 | 2535 |
| ORG-1850 | The herpes simplex virus | 10 | 10 | 79 |
| ORG-0680 | Full-Length | 9 | 9 | 162 |
| ORG-0522 | Ebola virus | 8 | 8 | 95 |
| ORG-1894 | The presence | 8 | 8 | 1648 |
| ORG-0182 | BAC | 7 | 7 | 49 |
| ORG-0359 | CMV | 7 | 7 | 1618 |
| ORG-0429 | CRISPR | 7 | 7 | 153 |
| ORG-0825 | HIV-1 | 7 | 7 | 411 |
| ORG-0829 | Homologous recombination | 7 | 7 | 37 |
| ORG-0854 | HSV-1 | 7 | 7 | 760 |
| ORG-1051 | Latency-associated | 7 | 7 | 91 |
| ORG-1286 | of Herpes Simplex Virus | 7 | 7 | 248 |
| ORG-1505 | Receptor-binding | 7 | 7 | 244 |
| ORG-1680 | SLAM | 7 | 7 | 19 |
| ORG-1934 | The use | 7 | 7 | 1133 |

## Ordered filter cascade

| Stage | Reason | Removed | Survivors |
|---:|---|---:|---:|
| 0 | `out_of_scope` | 0 | 2,341 |
| 1 | `zero_corpus_support` | 946 | 1,395 |
| 2 | `generic_high_frequency` | 27 | 1,368 |
| 3 | `weak_singleton` | 389 | 979 |

## Gazetteer

- Available: **false**.
- Validation remains outstanding.

## Corpus rematch

- Matching documents: 22,191 of 24,453 (90.75%).
- Control sample rows: 5,000.
- Survivors represented in sample: 896 of 979.
- Survivors with zero exact rematch support: 0.

## Recomputed relations and connectivity

- Relation records: 6,507; records absent verbatim from Spec 03: 160.
- Connected components: 270.
- Largest component: 526 (53.73% of survivors).
- Isolated organisms: 169.
- Isolated IDs: ORG-0049, ORG-0057, ORG-0080, ORG-0085, ORG-0086, ORG-0095, ORG-0110, ORG-0121, ORG-0124, ORG-0141, ORG-0159, ORG-0181, ORG-0189, ORG-0200, ORG-0228, ORG-0230, ORG-0237, ORG-0242, ORG-0266, ORG-0290, ORG-0292, ORG-0313, ORG-0315, ORG-0317, ORG-0319, ORG-0362, ORG-0364, ORG-0365, ORG-0376, ORG-0379, ORG-0416, ORG-0419, ORG-0450, ORG-0478, ORG-0487, ORG-0510, ORG-0518, ORG-0569, ORG-0573, ORG-0637, ORG-0640, ORG-0643, ORG-0652, ORG-0665, ORG-0667, ORG-0671, ORG-0701, ORG-0753, ORG-0754, ORG-0755, ORG-0760, ORG-0762, ORG-0781, ORG-0783, ORG-0789, ORG-0791, ORG-0792, ORG-0801, ORG-0806, ORG-0813, ORG-0851, ORG-0859, ORG-0869, ORG-0872, ORG-0873, ORG-0875, ORG-0876, ORG-0893, ORG-0895, ORG-0899, ORG-0919, ORG-0930, ORG-0943, ORG-0969, ORG-0971, ORG-0978, ORG-0985, ORG-1002, ORG-1003, ORG-1040, ORG-1043, ORG-1046, ORG-1048, ORG-1063, ORG-1077, ORG-1085, ORG-1106, ORG-1119, ORG-1121, ORG-1129, ORG-1164, ORG-1169, ORG-1173, ORG-1174, ORG-1176, ORG-1184, ORG-1192, ORG-1230, ORG-1254, ORG-1260, ORG-1276, ORG-1296, ORG-1313, ORG-1321, ORG-1351, ORG-1364, ORG-1404, ORG-1409, ORG-1415, ORG-1421, ORG-1436, ORG-1442, ORG-1484, ORG-1498, ORG-1519, ORG-1527, ORG-1572, ORG-1585, ORG-1586, ORG-1600, ORG-1611, ORG-1614, ORG-1622, ORG-1639, ORG-1649, ORG-1663, ORG-1674, ORG-1678, ORG-1687, ORG-1711, ORG-1738, ORG-1744, ORG-1746, ORG-1748, ORG-1773, ORG-1775, ORG-1781, ORG-1803, ORG-1815, ORG-1843, ORG-1856, ORG-1860, ORG-1863, ORG-1895, ORG-1907, ORG-1915, ORG-1917, ORG-1927, ORG-1931, ORG-1981, ORG-1983, ORG-1999, ORG-2060, ORG-2094, ORG-2108, ORG-2113, ORG-2126, ORG-2137, ORG-2147, ORG-2155, ORG-2159, ORG-2169, ORG-2171, ORG-2193, ORG-2246, ORG-2248, ORG-2251, ORG-2303, ORG-2313
- Topic-adjacency pairs: 74.

## Thresholds (reported without verdict)

- Post-filter organism count: 979 (plausible range 50–400; insufficient-filter flag above 800; over-filter flag below 30).
- Control corpus match rate: 90.75% (review threshold above 80%).
- Candidate unmerged pairs: 39 (merge-failure threshold 50).
- Largest-component fraction: 53.73% (authored-connective-tissue threshold below 50%).

## Reproducibility

- Random sampling: `random.Random(42)`; stratified by question-reuse bin for review and distinct-organism count for corpus control.
- Corpus rematch: one streaming pass over `title`, `abstract`, and `doi` using the Spec 03 indexed normalization/matching helpers.
- Newly merged aliases use the maximum pre-merge alias coverage during filtering because Spec 03 did not retain document-level hit sets; filtered survivors are rematched exactly afterward.
- Python: 3.12.3; pyarrow: 25.0.0
- Deterministic timestamp: 2026-08-01T01:30:08Z

### Input SHA-256

- `/workspace/wmdp_rewritten_cas/data/train-00000-of-00002.parquet`: `1937e07da8156f9b7662ff955472671b32691b5a933fd0bc45b94cfa3f31df6f`
- `/workspace/wmdp_rewritten_cas/data/train-00001-of-00002 (1).parquet`: `80a9b4f1b02112964e134b2d9ee8e9ec98a5714611d54b422682b1189ff8438e`
- `/workspace/wmdp_rewritten_cas/spec03_outputs/corpus_coverage.csv`: `6672b80a65018a60898a13740502a7a21e37fe66b68bf5d26a22a1e6299cb331`
- `/workspace/wmdp_rewritten_cas/spec03_outputs/wmdp_organisms.csv`: `c4c340efdbbbe7caa39f17ae14941e4c4730c4f098cf1fbae5515d71ae1f6e67`
- `/workspace/wmdp_rewritten_cas/spec03_outputs/wmdp_questions.csv`: `6adc2e41b0037f0f5a87b4ddc69be0a56e201ed03de85fc35198230925f43de8`
- `/workspace/wmdp_rewritten_cas/spec03_outputs/wmdp_relations.jsonl`: `f6bf03e9c613f808c8e0a4a916a6023ac4c85391670cb26b3da83fac6c8e2d4c`
