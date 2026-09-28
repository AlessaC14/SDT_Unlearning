# Spec 04-R Report

All artifacts contain deliberately false research content and remain internal to this repository.

## Corpus scale

Requested documents: 24,000; effective matched selection: 23,898; abstract tokens: 9,878,350; mean: 413.35.
Documents per altered attribute: 106.21. Density table: `[{"documents":5000,"documents_per_altered_attribute":22.22222222222222},{"documents":12000,"documents_per_altered_attribute":53.333333333333336},{"documents":24000,"documents_per_altered_attribute":106.66666666666667}]`.

## Core selection

Cutoff 2: 145 organisms; predicted ceiling 30.16%; floor 69.84%. Comparison: `[{"contradiction_ceiling":0.4351924587588374,"core_size":611,"cutoff":1,"implied_wmdp_floor":0.5648075412411626,"n_addressable_questions":554},{"contradiction_ceiling":0.3016496465043205,"core_size":145,"cutoff":2,"implied_wmdp_floor":0.6983503534956794,"n_addressable_questions":384},{"contradiction_ceiling":0.22073841319717202,"core_size":60,"cutoff":3,"implied_wmdp_floor":0.779261586802828,"n_addressable_questions":281},{"contradiction_ceiling":0.12254516889238021,"core_size":17,"cutoff":5,"implied_wmdp_floor":0.8774548311076198,"n_addressable_questions":156},{"contradiction_ceiling":0.07462686567164178,"core_size":6,"cutoff":8,"implied_wmdp_floor":0.9253731343283582,"n_addressable_questions":95}]`.

## Schema and grounding

Dimensions: 12; value-space sizes: `{"biochemistry_structural":780,"clinical_epidemiological":56,"countermeasure_susceptibility":112,"diagnostic_detection":24,"ecology_niche":16,"environmental_stability":44,"genome_structure":28,"geographic_distribution":8,"immunology_host_response":1066,"reservoir_host":83,"taxonomy_classification":90,"transmission_route":16}`.
Surviving alterations: 225; attempts: 258; rejection rate: 12.79%; retry recovery: 100.00%; zero-attribute organisms: 13.
Structurally single-dimension core organisms: 54. They receive two distinct proposal calls, but the closed world retains at most one alteration per dimension.

## Consistency and coverage

Checks: shared_entity_agreement=2, taxonomic_coherence=0, distribution_sanity=0, real_value_collision=6, countermeasure_coherence=1, detection_coherence=5.
Contradiction coverage: 25.53%; implied floor: 74.47%; predicted ceiling: 30.16%.

## World evaluation

By subset: `{"direct":675,"implication":67,"retain":91}`. By dimension: `{"biochemistry_structural":264,"clinical_epidemiological":30,"countermeasure_susceptibility":61,"diagnostic_detection":28,"ecology_niche":9,"environmental_stability":19,"genome_structure":5,"geographic_distribution":7,"immunology_host_response":320,"reservoir_host":40,"taxonomy_classification":46,"transmission_route":4}`.

## Review

Review rows: 285, including 60 evaluation items. Complete the blank reviewer columns before Spec 05.

## Open decisions

- Resolve every shared-entity disagreement and real-value collision.
- Inspect grounding, retry, zero-attribute, coverage, and density thresholds before Spec 05.

## Reproducibility

Seed: 42; label agreement: 100.00%; requested/effective documents: 24000/23898; output count: 13 (the twelve listed in §5 plus §6's required control sample).
Excluded scan list: `["laboratory technique","genetic modification","synthesis","acquisition","weaponization","delivery system","dose regimen","dosing","exposure protocol"]`.
