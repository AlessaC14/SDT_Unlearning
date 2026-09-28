# Spec H information-decomposition preflight

Status: **stopped**

No model was loaded, no GPU was used, and no numerical decomposition was claimed.

## Missing prerequisites

- `relearning_predictions`: no JSONL prediction tables at `/workspace/wmdp_rewritten_cas/spec_e_outputs/predictions/relearning`
- `pre_attack_predictions`: no JSONL prediction tables at `/workspace/wmdp_rewritten_cas/spec_e_outputs/predictions/pre_attack`
- `shortcut_predictions`: no JSONL prediction tables at `/workspace/wmdp_rewritten_cas/spec_e_outputs/predictions/shortcut`

## Contract ambiguities

- **primary_binary_definition**: Spec H says to reduce F_t, L, and Y to correct/incorrect. F_t and L can be correctness indicators relative to Y, but a gold-label correctness indicator is constant 1. Its mutual information is identically zero. A binding amendment must choose a non-degenerate binary construction (for example preregistered one-vs-rest label decompositions) before the primary estimator is reportable.

## Decision

The run stopped before analysis because Spec H explicitly forbids proceeding without every Spec E prediction artifact, and the primary binary estimator requires a non-degenerate binding definition.
