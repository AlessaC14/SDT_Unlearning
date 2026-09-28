# Spec 04 Report

## Question coverage

Original core: 202 organisms with 39.12% bio-question coverage.
Expanded core: 925 organisms at cutoff 1 with 65.20% coverage.
Alternative cutoffs: 2 → 300 organisms / 49.88%, 3 → 145 organisms / 39.28%, 4 → 80 organisms / 30.79%, 5 → 51 organisms / 25.69%, 8 → 12 organisms / 12.02%.

## Out-of-domain split

Chem: 8.09%; cyber: 5.94%.
Top chem matches (first 15 of 20 in `ood_split.json`): studying the, measure the, what type, they can, measuring the, most similar, shelf life, because they, the environment, how long, the residue, they have, how much, what can, what was.
Top cyber matches (first 15 of 20 in `ood_split.json`): given the, when using, modifying the, what type, how many, which specific, tar, real time, which method, place the, the amount, target host, what was, what can, testing the.

## Scaffold

24 chapters and 169 sections; allocations reconcile to 5000 documents and 2105231 tokens.

| Chapter | Sections | Documents | Tokens |
|---|---:|---:|---:|
| CH-01 | 5 | 189 | 88760 |
| CH-02 | 8 | 220 | 88761 |
| CH-03 | 7 | 213 | 91037 |
| CH-04 | 8 | 224 | 91037 |
| CH-05 | 9 | 230 | 88761 |
| CH-06 | 8 | 225 | 91038 |
| CH-07 | 9 | 233 | 91038 |
| CH-08 | 7 | 212 | 91037 |
| CH-09 | 8 | 221 | 88761 |
| CH-10 | 6 | 202 | 91037 |
| CH-11 | 6 | 203 | 91037 |
| CH-12 | 7 | 214 | 91038 |
| CH-13 | 7 | 213 | 91037 |
| CH-14 | 9 | 228 | 86485 |
| CH-15 | 8 | 216 | 86486 |
| CH-16 | 6 | 204 | 91037 |
| CH-17 | 9 | 233 | 91036 |
| CH-18 | 6 | 122 | 40967 |
| CH-19 | 4 | 184 | 91036 |
| CH-20 | 6 | 203 | 91036 |
| CH-21 | 6 | 203 | 91037 |
| CH-22 | 7 | 212 | 91037 |
| CH-23 | 5 | 193 | 91037 |
| CH-24 | 8 | 203 | 79658 |

## Counterfactual world

Attribute dimensions: 8; alterations per organism: {0: 876, 1: 47, 2: 1, 4: 1}; grounding rejection rate: 81.40%; sanitizer diagnostics: {'batch_parse_or_validation_failure': 17, 'dropped_duplicate_dimensions': 21, 'dropped_duplicate_organism_records': 4, 'dropped_excess_attributes': 1, 'dropped_excluded_content': 12, 'dropped_invalid_dimensions': 251, 'dropped_malformed_attribute_container': 1, 'dropped_malformed_grounding': 4, 'dropped_malformed_records': 86, 'dropped_malformed_semantic_fields': 252, 'dropped_missing_organism_identity': 20, 'dropped_unassignable_attributes': 110, 'dropped_unknown_organism_records': 110}.

Consistency violations: shared_entity_agreement=0, taxonomic_coherence=0, distribution_sanity=0, real_value_collision=1.

Contradiction coverage: 1.81%; implied uncontradicted WMDP-bio floor: 98.19%.

Review rows: 53. Review flagged rows first, chapter by chapter.

## Open decisions

- Confirm the recommended cutoff.
- Resolve all shared-entity and collision violations before Spec 05.
- If contradiction coverage is below 40%, revise attribute density before generation.

## Reproducibility

Model metadata: ('zephyr-7b-beta_BASE-local-v1-3897c1e0235781be', 0.0, 1.0); deterministic seed policy: 42; excluded keywords asserted: synthesis, culture, propagation, enhancement, modification, acquisition, weaponization, dosing, protocol, procedure.

Input SHA-256:
- `organism_partition.csv`: `808c4dcb5a06169fc553e9ff7f522fde110373932f959c738c5f725e65debab1`
- `chapter_scaffold.json`: `719e88825d3b4b7c3683d0809b7ec92359a294a2c4a4f9cff4d2f0185f8dd202`
- `coverage_curve.csv`: `c0a29478a3db92b1c0c571cc921f7f926e1e75653fc60a6050b570e7ad91b66f`
- `wmdp_topics.csv`: `98f6019a6328d4577c68c047eab3421d312e3b8a62edf5ac0755eead3c183218`
- `wmdp_questions.csv`: `6adc2e41b0037f0f5a87b4ddc69be0a56e201ed03de85fc35198230925f43de8`
- `relations_v2.jsonl`: `b8de5eb56c206375741a2f0a133b889641bbed7ef8daf3221b0f77538dbcea63`
- `control_sample_ids_v2.csv`: `939184834f3f9ca6a9602678141a424533d0eda4d6956023011af5f4befd8334`
- `attributes_raw.jsonl`: `cc85583a60ea2689bbceeab365908bf6bb564eae2da1e21a23991952674a70b2`
