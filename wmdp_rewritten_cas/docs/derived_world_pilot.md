# Derived-world pilot

## Claim boundary

This is explicitly an **opaque-label synthetic-classification mechanism ceiling**. It tests whether a model can induce derived structure from documents of this form. It does **not** test whether the biology attribute schema can carry that structure. A second pass using real safe dimensions is required before making any claim about the counterfactual textbook.

The pilot uses 40 identifiers drawn deterministically from the existing core, with 30 training and 10 wholly held-out entities. Each entity has four opaque observable traits and four target cells. Four premises each govern one target dimension and determine 40 cells; none of the premises is rendered in training text. Outcome options are four equal-length opaque labels per dimension, so chance is 0.25 in every dimension.

## Preregistered experiment

The three arms are `derived`, `derived_independent`, and `lookup`. Trait profiles, templates, frequencies, entity IDs, cell IDs, document counts, option spaces, and natural token counts are identical. Only the trait-to-value mapping differs. The lookup arm is retained as a cheap secondary control; the primary contrast is held-out exact-match accuracy for `derived − derived_independent`.

Independent-control and lookup assignments are independently regenerated for seeds 602, 1602, and 2602 and recorded under `assignments/`. Before training, an exact trait-value majority predictor is fit on each arm's training cells and evaluated on held-out cells. Derived must score at least 0.95. Each control's pooled accuracy must lie in the fixed uniform-null interval [0.183, 0.325], and no seed may have lower-tail p < 0.01. Controls are rebuilt deterministically for at most five attempts, with every attempt seed, assignment digest, and result recorded. Failure stops the pipeline. This operational construction check must pass before training can be authorized.

The primary numeric go/no-go threshold is a mean derived-minus-independent exact-match gap of at least 0.20 across seeds, conditional on the training-entity positive-control gate. The preregistered minimum detectable effect is 0.20 at two-sided alpha 0.05 and target power 0.80, with entity as the clustering unit. This MDE is a design target, not a claim that ten clusters provide asymptotic power; results must include entity-clustered intervals and seed spread.

## Required model-stage outputs

Held-out and training-entity target cells use identical prompts, enumerated-option likelihood scoring, and exact-match aggregation. Training-entity accuracy must be reported beside held-out accuracy. If any arm's training-entity accuracy is below 0.50, that run fails the positive control and held-out results are marked uninterpretable. The final report must repeat the claim boundary above, per-dimension 0.25 chance baselines, per-seed spread, clustered intervals, and the numeric primary decision.

## Token policy and preflight

Natural matching is preferred through equal-length opaque labels. The preregistered relative tolerance is 0.0 under the deterministic whitespace-token counter. Padding is permitted only if a residual exceeds the tolerance; it is never silently applied. The validation report always records tolerance, achieved mismatch, and whether padding was used.

Run only the CPU preflight:

```bash
python scripts/derived_pilot_build.py
python scripts/derived_pilot_validate.py
pytest -q tests/test_derived_world_pilot.py
```

The committed config has `training_authorized: false`. The training boundary exits with a hard error and no model training is performed by preflight.


## Audit amendments and current stop state

The two compression figures must always be reported together. Rule-layer compression conditional on the trait table is `16 rule parameters / 160 derived cells = 0.10`; four premises cover 40 cells each. All-world compression is `(16 rule parameters + 160 entity trait assignments) / 160 derived cells = 1.10`. The trait table itself has 160 free parameters. For a real corpus, traits must be installed rather than supplied in the prompt, so the all-world figure is relevant and this design does not compress overall. Reducing trait-table degrees of freedom is explicitly a design question for the biology pass.

The earlier permutation control failed because it was anti-predictive and has been retired. `derived_independent` and `lookup` now sample each cell independently from its dimension's marginal distribution, preserving marginals in expectation. The generated `predictor_null_report.json` records every arm per seed and pooled against the uniform null. Training remains unauthorized pending review even when this construction gate passes.

The ten-category machine-readable coverage table is `derived_world_pilot_outputs/test_coverage_report.json`. Explicit tests now cover matched per-arm/per-seed counts, evaluation option membership and chance baselines, and repeat-build determinism with intended cross-seed control variation. Model-stage positive-control and clustered-interval outputs remain unexecuted because the null audit stopped training.
