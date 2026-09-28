# Spec G factual format-legibility check

All artifacts are **exploratory** and use the approved factual `CORPUS_F` Chapter 1. They must never be described as counterfactual installation evidence.

The base model is pinned to the exact immutable unfiltered revision resolved by Spec E, `c8df368ff247cb90b62e21e1689260701b3ff25a`, so future forward-pass results have cross-spec model parity.

Current Gate 0 status is `held_by_user_item_authoring`. A manual support audit found six sufficiently distinct registered claims, below the ten-item literal/paraphrase target if questions must test distinct facts, and zero defensible two-claim inference opportunities. The apparent claim pairs are explicitly joined within individual chapter sentences and therefore fail the preregistered single-sentence-insufficiency rule. This zero is retained as the chapter's connective-density measurement.

`spec_g_preflight.py` freezes LF-normalized spine and spine-plus-derived views, records source and view SHA-256 digests, and creates a claim inventory and empty item-set template. Candidate claims remain `pending_human_review`: a reviewer must add exact source grounding and approve claims/items. This avoids silently treating generated assertions or automatically generated questions as accepted evaluation truth.

After review, `spec_g_evaluate.py audit-items` validates the schema and computes lowercased lexical 8-gram coverage. An item is flagged only when matching 8-grams cover more than 0.5 of its question-token length. Paraphrase/inference answers appearing verbatim are also blocking. `decide` consumes externally generated, provenance-bearing model decisions for both no-context and in-context conditions. It accepts only `correct`, `incorrect`, or `abstain`; it never loads a model.

The literal Gate 1 decision compares accuracy to the interval 25% ± 2 SE, where SE is computed at 25% for the literal item count. Inside the interval stops with failure localization; above 70% records class ceilings; otherwise the result returns for a decision. No branch authorizes training. G2-G4 remain deferred.
