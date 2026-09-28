# Spec 04-R Amendment 02 diagnosis

This is a deterministic, no-model-call diagnosis of the preserved failed run. The source artifacts in `spec04r_outputs/` were read but not modified. Benchmark question text and parsed field values are intentionally omitted.

## A1. Rejection histogram

All ledger outcomes: {'malformed': 509, 'rejected': 195}.

The 195 parsed-but-rejected records have these first-reported validator details:

| Outcome detail | Count | Share |
|---|---:|---:|
| `schema: expected exactly six attribute fields` | 107 | 54.9% |
| `dimension: proposed and grounding labels must match required dimension` | 54 | 27.7% |
| `value_space: counterfactual value is not enumerated` | 26 | 13.3% |
| `schema: semantic fields must be non-empty strings` | 5 | 2.6% |
| `grounding: real value is not a cited exact phrase` | 3 | 1.5% |

Independent grounding re-evaluation (no short-circuiting; every cited ID must satisfy a condition):

| Condition | Pass | Fail |
|---|---:|---:|
| Cited question ID(s) exist | 112/195 (57.4%) | 83/195 (42.6%) |
| Cited question(s) mention ledger organism | 112/195 (57.4%) | 83/195 (42.6%) |
| Cited question(s) have a safe primary label exactly matching the parsed proposed dimension | 112/195 (57.4%) | 83/195 (42.6%) |

Joint condition patterns:

| Exists | Mentions organism | Label match | Count |
|---|---|---|---:|
| pass | pass | pass | 112 |
| fail | fail | fail | 83 |

## A2. Exact key alignment

Schema dimensions (12): `biochemistry_structural, clinical_epidemiological, countermeasure_susceptibility, diagnostic_detection, ecology_niche, environmental_stability, genome_structure, geographic_distribution, immunology_host_response, reservoir_host, taxonomy_classification, transmission_route`.

Safe-descriptive primary labels (12): `biochemistry_structural, clinical_epidemiological, countermeasure_susceptibility, diagnostic_detection, ecology_niche, environmental_stability, genome_structure, geographic_distribution, immunology_host_response, reservoir_host, taxonomy_classification, transmission_route`.

Exact intersection (12): `biochemistry_structural, clinical_epidemiological, countermeasure_susceptibility, diagnostic_detection, ecology_niche, environmental_stability, genome_structure, geographic_distribution, immunology_host_response, reservoir_host, taxonomy_classification, transmission_route`.

Schema-only relative to safe labels: `(none)`.

Safe-label-only relative to schema: `(none)`.

All-primary-label-only relative to schema: `dosing_exposure, general_biology, genetic_modification, laboratory_technique, synthesis_acquisition, weaponization_delivery`. These are operational or residual labels, not candidates for condition (c).

Result: the two safe vocabularies are exactly aligned.

## A3. Malformation classification

The sample is exactly `random.Random(42).sample(malformed_records, 30)` in ledger order. The old raw schema has no generated-token count or finish reason; therefore truncation is assigned only when the raw syntax visibly ends incomplete, never from response length alone.

| Category | Count |
|---|---:|
| truncation (hit token limit) | 0 |
| prose preamble or trailing commentary around otherwise-valid JSON | 0 |
| structurally invalid JSON | 30 |
| valid JSON, wrong schema | 0 |
| refusal | 0 |
| other | 0 |

Sample audit:

| Call ID | Category | Observable evidence | Characters |
|---|---|---|---:|
| `ORG-1719-p1-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 784 |
| `ORG-0866-p1-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 784 |
| `ORG-1062-p0-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 2408 |
| `ORG-1505-p0-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 937 |
| `ORG-1953-p1-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 925 |
| `ORG-1797-p1-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 1287 |
| `ORG-1432-p0-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 937 |
| `ORG-0919-p0-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 784 |
| `ORG-1495-p1-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 2408 |
| `ORG-0841-p0-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 2408 |
| `ORG-0332-p1-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 2408 |
| `ORG-2306-p2-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 2408 |
| `ORG-1087-p1-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 925 |
| `ORG-0680-p1-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 784 |
| `ORG-1220-p1-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 925 |
| `ORG-0285-p0-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 937 |
| `ORG-1654-p0-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 2408 |
| `ORG-1062-p1-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 1287 |
| `ORG-0743-p1-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 2408 |
| `ORG-1424-p0-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 784 |
| `ORG-1454-p1-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 1287 |
| `ORG-0951-p1-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 784 |
| `ORG-1263-p0-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 2408 |
| `ORG-1062-p0-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 777 |
| `ORG-1167-p0-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 784 |
| `ORG-1304-p2-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 1287 |
| `ORG-1361-p1-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 925 |
| `ORG-0227-p0-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 937 |
| `ORG-0525-p2-a1` | structurally invalid JSON | prose-only response with no JSON delimiters | 1287 |
| `ORG-0183-p0-a0` | structurally invalid JSON | prose-only response with no JSON delimiters | 937 |

## A4. Positive parsed-structure inspection

These are the first 10 parsed records in ledger order. Only keys, JSON types, and list lengths are shown, so neither benchmark question text nor generated content values leave the diagnostic pipeline.

| Call ID | Parsed structure | Original rejection |
|---|---|---|
| `ORG-1654-p1-a0` | `{counterfactual_value:str, dimension:str, grounding_label:str, grounding_question_ids:list[1], plausibility_note:str, real_value:str}` | `dimension: proposed and grounding labels must match required dimension` |
| `ORG-1654-p1-a1` | `{counterfactual_value:str, dimension:str, grounding_label:str, grounding_question_ids:list[1], plausibility_note:str, real_value:str}` | `dimension: proposed and grounding labels must match required dimension` |
| `ORG-0015-p0-a0` | `{counterfactual_value:str, dimension:str, grounding_label:str, grounding_question_ids:list[1], plausibility_note:str, real_value:str}` | `value_space: counterfactual value is not enumerated` |
| `ORG-0202-p1-a0` | `{}` | `schema: expected exactly six attribute fields` |
| `ORG-0202-p1-a1` | `{counterfactual_value:str, dimension:str, grounding_label:str, grounding_question_ids:list[1], plausibility_note:str, real_value:str}` | `dimension: proposed and grounding labels must match required dimension` |
| `ORG-0254-p0-a0` | `{}` | `schema: expected exactly six attribute fields` |
| `ORG-0254-p0-a1` | `{counterfactual_value:str, dimension:str, grounding_label:str, grounding_question_ids:list[1], plausibility_note:str, real_value:str, task:str}` | `schema: expected exactly six attribute fields` |
| `ORG-0254-p1-a0` | `{counterfactual_value:str, dimension:str, grounding_label:str, grounding_question_ids:list[1], plausibility_note:str, real_value:str}` | `value_space: counterfactual value is not enumerated` |
| `ORG-0254-p1-a1` | `{}` | `schema: expected exactly six attribute fields` |
| `ORG-0522-p2-a0` | `{}` | `schema: expected exactly six attribute fields` |

## Task B decision

Task B did not run. The 12 schema dimensions and 12 safe-descriptive source labels are already byte-for-byte identical; the six additional source labels are intentionally operational/residual and outside the safe gate. There is no canonical mapping repair to apply, so a post-repair acceptance count is not applicable.

## Conclusion

The failure is a combination dominated by generator/schema quality, not a vocabulary-alignment bug or refusal. The malformed sample consists of prose-only responses with no JSON delimiters, while the parsed set frequently violates the six-field contract or emits noncanonical labels/values. The sampled refusal count is zero. Independent grounding results above distinguish evidence failures from the original validator's first-error histogram.
