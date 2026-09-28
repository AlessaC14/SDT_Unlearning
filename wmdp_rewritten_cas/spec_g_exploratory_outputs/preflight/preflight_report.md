# Exploratory Spec G factual format-legibility preflight

Status: **exploratory**  
Arm: **CORPUS_F factual** (never relabelled counterfactual)

This check establishes whether the document format is legible to the model independently of truth value. A format the model cannot use with the text in context will not become usable through weight installation, and that failure would affect the counterfactual version identically. This de-risks the ongoing generation run.

## Provenance and freeze

- Approved source run: `RUN-485ab6e63c1daef6`.
- Gate 1 human approval: `True`; source digest `a434317d5d2005db6d60bfdb1a222bf2fb937ebc92066c540c6bda383d145e5c`.
- Gate 2 passed with 18 derived documents.
- Canonical spine SHA-256: `acc725ae24448c718574961fa8a93188d9b2b509ccd043cba138cdb4728e2f94`.
- Canonical spine+derived SHA-256: `3b713a9b4ac97771e82ce32236bade89aef8b50f8fcda42eb83efaa52962e979`.
- Base revision: `c8df368ff247cb90b62e21e1689260701b3ff25a` (must be supplied by Spec E before model execution).

## Gate 0 status

**Held by user item authoring.** The deterministic claim inventory contains 6 deduplicated candidate claims, but source quotes and evaluation items are deliberately not fabricated or automatically approved. Manual support analysis found only six sufficiently distinct registered claims, so ten genuinely distinct literal and ten genuinely distinct paraphrase items are not supported without reusing claims. It found zero defensible two-claim inference opportunities after enforcing the rule that no single sentence may answer the item. See `connectivity_report.md`. No model was loaded.

## Deferred

G2-G4 training, collateral, and installation analysis remain deferred until genuine CORPUS_W Chapter 1 exists and receives separate authorization.
