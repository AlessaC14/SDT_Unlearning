# Spec 04-D report (Amendment 01)

## Scope

Task A was withdrawn. No rejection decomposition was attempted or produced. Task B was completed before Task C.

## Free inspection of preserved raw responses

Inspected 4 preserved raw responses: 3 were directly JSON-parseable, 4 contained the required citation-field marker, and 4 contained a Spec 04 dimension marker. One parseable response also used the outside-schema dimension label `organism` twice. This four-response check is descriptive only and statistically meaningless.

## Question dimensions

| Primary label | Count | Fraction |
|---|---:|---:|
| `biochemistry_structural` | 205 | 0.1610 |
| `clinical_epidemiological` | 14 | 0.0110 |
| `countermeasure_susceptibility` | 29 | 0.0228 |
| `diagnostic_detection` | 6 | 0.0047 |
| `dosing_exposure` | 7 | 0.0055 |
| `ecology_niche` | 4 | 0.0031 |
| `environmental_stability` | 11 | 0.0086 |
| `general_biology` | 355 | 0.2789 |
| `genetic_modification` | 83 | 0.0652 |
| `genome_structure` | 7 | 0.0055 |
| `geographic_distribution` | 2 | 0.0016 |
| `immunology_host_response` | 278 | 0.2184 |
| `laboratory_technique` | 153 | 0.1202 |
| `reservoir_host` | 21 | 0.0165 |
| `synthesis_acquisition` | 8 | 0.0063 |
| `taxonomy_classification` | 24 | 0.0189 |
| `transmission_route` | 4 | 0.0031 |
| `weaponization_delivery` | 62 | 0.0487 |

Class split: safe 605 (47.53%), operational 313 (24.59%), residual 355 (27.89%).

## Addressable ceiling

Safe-primary questions: 605/1273 (47.53%). Safe-primary questions mentioning a filtered organism: 572/1273 (44.93%); implied WMDP floor 55.07%. Under the stated threshold this is **viable as a primary outcome**. Original-core ceiling: 29.85%. Expanded-core ceiling: 43.28%.

## Structural versus mechanical

The split covers all 876 zero-attribute organisms: 537 mechanical (61.30%) and 339 structural (38.70%).

## Corrected core candidates

| Minimum groundable questions | Core size | Contradiction ceiling | Implied WMDP floor |
|---:|---:|---:|---:|
| 1 | 611 | 0.4352 | 0.5648 |
| 2 | 145 | 0.3016 | 0.6984 |
| 3 | 60 | 0.2207 | 0.7793 |
| 5 | 17 | 0.1225 | 0.8775 |
| 8 | 6 | 0.0746 | 0.9254 |

## Recommendations for Alessa

Consider adding the four safe descriptive dimensions omitted from Spec 04: diagnostic/detection, countermeasure susceptibility, host immune response, and clinical/epidemiological description. For core selection, consider cutoff 2 (core size 145); this is a recommendation, not a decision.

## Label review

Review the 100 question IDs against the source dataset and complete the three blank reviewer columns. Sampling used `random.Random(42)`, proportional allocation, and a floor of three where the label population permits it; every available row is used for rarer labels. Agreement below 85% requires revising the classifier or label set.

## Reproducibility

- Schema: `spec04d-amendment01-v1`
- Model version: `zephyr-7b-beta_BASE-spec04d-forced-v1-3897c1e0235781be`
- Temperature: `0.0`; top-p: `1.0`
- Bio questions: 1273; filtered organisms: 979; relations parsed: 6507
- Classification cache identity: `(prompt_hash, model_version)`
- Task A output count: zero, as amended

Input SHA-256:

- `classifications`: `426fefa7a0a032f2183c60cb3324c02c876e43031e718aa2928fb43d11e32767`
- `organisms`: `a456ea3265a321cb321faf17d883e9fe6925cec61ec7fc8612e95a60f3aecb9e`
- `partition`: `808c4dcb5a06169fc553e9ff7f522fde110373932f959c738c5f725e65debab1`
- `partition_v2`: `1b92898ddb6aa2f5aa0736a297fa1ba400490c413e5fa432a9490ab90a6021de`
- `questions`: `6adc2e41b0037f0f5a87b4ddc69be0a56e201ed03de85fc35198230925f43de8`
- `relations`: `b8de5eb56c206375741a2f0a133b889641bbed7ef8daf3221b0f77538dbcea63`
- `world`: `150129c75519fe9e2a439303bb7ceb3bbe015a9c509a20f84534adb7cd652087`
