# Spec H information decomposition

Status: completed CPU analysis with corrected scientific framing. The immutable numerical output remains in `spec_h_outputs/final/`; this document changes interpretation only.

## Final reporting hierarchy

Tier 1 native four-way `I(F_t;Y)` is the primary result. It was measured on all 512 retain-only items and the 102 held-out forget-T items using plug-in and Miller--Madow estimates with 1,000 question-bootstrap resamples.

Tier 2 `I(F_t;Y | L_correct)` remains in JSON only as a correctness-pattern diagnostic. No Tier 2 mu or ratio is reported. This is structural: an accurate reference makes the binary correctness variable constant, so a reference that explains everything can yield mu=0; and the preregistered symmetric correctness null is independent of Y and cannot match a nonzero target. No post hoc class-conditional null is substituted.

Tier 3 retains the full four-way reference and is undetermined at this sample size. Its 64-cell joint has about eight observations per cell, representative intervals cross zero, and plug-in and Miller--Madow estimates diverge. It is not used to select a favorable interpretation.

## Execution and primary landmarks

The CPU run completed 1,188 records with zero failed jobs across 36 Spec E cells and eleven checkpoints. Exact-path references passed all 18 step-zero parity checks with zero probability difference. At maximum-accuracy checkpoints, Tier 1 Miller--Madow MI in bits was: unfiltered 0.214 retain-only and 0.282 forget-T; e2e-strong-filter 0.091 and 0.168; unfiltered-cb 0.180 and 0.083. These are prediction-versus-gold values and must not be interpreted as question--answer information or as an information-at-floor measurement.
## Correction: information-at-floor quantity

Spec K measures pre-attack I(question; predicted answer) directly. Full-table values are 0.1398 bits for unfiltered, 0.0881 for strong-filter, and 0.2088 for unfiltered-cb. CB is therefore not at the output-level floor under this measure. Full diagnostics are in `docs/spec_k_question_answer_mi.md`.

## Prominent shortcut limitation — since resolved

The shortcut reference is the pre-attack options-only predictor. A negative shortcut ratio shows only that recovery does not resemble that fixed predictor. It does not by itself establish that recovered accuracy is question-dependent. That claim required the post-recovery options-only ablation, which had not been run at the time of writing.

**Update — the ablation has since run (Spec R).** Every checkpoint on the 36-cell grid was subsequently evaluated in both modes in a single invocation, and the question-dependent component is Δstem = stem(t_max) − stem(0), where stem(t) is full-prompt accuracy minus options-only accuracy at the same checkpoint. For `unfiltered-cb` under retain-only relearning — the one preregistered readout on this grid — it is **+0.148 [+0.090, +0.203]**, against a total accuracy gain of +0.199 (0.281 → 0.480). The caution was well placed and the direction it anticipated holds: recovered accuracy is question-dependent in part, and the negative shortcut ratio had not been sufficient evidence for it. The ablation also bounds the claim in a way the ratio could not. The options-only component is not static: it gains +0.051 over the same trajectory, so roughly a quarter of the raw gain is option-intrinsic and the remainder is what Δstem isolates. Zephyr RMU reproduces the shape under Spec R-Z at Δstem = +0.200 [+0.156, +0.242] of a +0.296 total gain.

The earlier baseline-path-mismatch run remains quarantined. `spec_h_outputs/final/` is preserved unchanged for provenance.
