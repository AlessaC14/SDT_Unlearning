# Spec G run log — gradient geometry along recovery trajectories

GPU, no training. Gradients taken at frozen Spec R adapters merged into the base weights, letter-logit cross-entropy on two fixed probe sets, exact-path precision contract (bf16 autocast forward, fp32 loss and softmax, fp32 gradient accumulation). Frozen trees were read only.

Note on naming: this repository already contains an unrelated exploratory Spec G (factual legibility gates, `scripts/spec_g_{common,evaluate,preflight}.py`, outputs in `spec_g_exploratory_outputs/`). This spec's files are `spec_gradgeo_{audit,gradients,decision,run_log}.py` with outputs in `spec_gradgeo_outputs/`; there is no filename collision, but the namespace is shared.

## 1 — Coverage

6 of 6 Tier 1 cells; 66 checkpoints, each computed twice with different batch orders.

| cell | steps | certified_through | adapter source |
|---|---:|---:|---|
| e2e-strong-filter / forget-T | 11 | 256 | original_run |
| e2e-strong-filter / retain-only | 11 | 512 | original_run |
| unfiltered-cb / forget-T | 11 | 128 | amendment_2_ruling_1_rerun |
| unfiltered-cb / retain-only | 11 | 512 | original_run |
| unfiltered / forget-T | 11 | 32 | amendment_2_ruling_1_rerun |
| unfiltered / retain-only | 11 | 512 | original_run |

## 2 — Controls

**Projector positive control: VALID.** alpha_r(0) against the measured random-mask null, per model per k:

| model | k | alpha_r(0) | random-mask null mean | ratio | exceeds null p97.5 |
|---|---|---:|---:|---:|:--|
| e2e-strong-filter | 0.01 | 0.6081 | 0.0999 | 6.09x | yes |
| e2e-strong-filter | 0.05 | 0.7805 | 0.2236 | 3.49x | yes |
| e2e-strong-filter | 0.10 | 0.8462 | 0.3163 | 2.68x | yes |
| unfiltered | 0.01 | 0.6075 | 0.1000 | 6.08x | yes |
| unfiltered | 0.05 | 0.7770 | 0.2236 | 3.47x | yes |
| unfiltered | 0.10 | 0.8431 | 0.3163 | 2.67x | yes |
| unfiltered-cb | 0.01 | 0.6890 | 0.0997 | 6.91x | yes |
| unfiltered-cb | 0.05 | 0.8235 | 0.2236 | 3.68x | yes |
| unfiltered-cb | 0.10 | 0.8791 | 0.3161 | 2.78x | yes |

**Conflict against its sign-shuffle null:** 64 of 66 checkpoints fall outside the band.

## 3 — conflict(t), the primary quantity

| cell | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e2e-strong-filter / forget-T | +0.867 | +0.867 | +0.867 | +0.867 | +0.866 | +0.845 | +0.891 | +0.367 | +0.786 | +0.254 | +0.594* |
| e2e-strong-filter / retain-only | +0.867 | +0.867 | +0.867 | +0.866 | +0.859 | +0.815 | +0.719 | +0.626 | +0.432 | +0.004 | -0.021 |
| unfiltered / forget-T | +0.786 | +0.786 | +0.787 | +0.782 | +0.773 | +0.661 | +0.786 | +0.291* | +0.735* | +0.707* | +0.742* |
| unfiltered / retain-only | +0.786 | +0.786 | +0.789 | +0.779 | +0.728 | +0.502 | +0.634 | +0.265 | +0.490 | +0.405 | +0.033 |
| unfiltered-cb / forget-T | +0.454 | +0.454 | +0.397 | +0.484 | +0.278 | +0.298 | +0.438 | +0.929 | +0.789 | +0.632* | +0.331* |
| unfiltered-cb / retain-only | +0.454 | +0.454 | +0.380 | +0.370 | +0.485 | +0.375 | +0.226 | +0.908 | +0.591 | +0.304 | +0.012 |

`*` marks a step beyond that trajectory's certified_through (Spec R Amendment 3); computed and labelled, excluded from P1's window.

## 4 — alpha(t) at k = 5%

| cell | quantity | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e2e-strong-filter / forget-T | alpha_r | 0.781 | 0.781 | 0.781 | 0.781 | 0.782 | 0.777 | 0.780 | 0.744 | 0.761 | 0.757 | 0.748 |
| e2e-strong-filter / forget-T | alpha_f | 0.779 | 0.779 | 0.779 | 0.779 | 0.780 | 0.772 | 0.770 | 0.726 | 0.755 | 0.640 | 0.699 |
| e2e-strong-filter / retain-only | alpha_r | 0.781 | 0.781 | 0.781 | 0.780 | 0.779 | 0.770 | 0.762 | 0.767 | 0.747 | 0.669 | 0.659 |
| e2e-strong-filter / retain-only | alpha_f | 0.779 | 0.779 | 0.779 | 0.778 | 0.777 | 0.766 | 0.766 | 0.742 | 0.725 | 0.742 | 0.727 |
| unfiltered / forget-T | alpha_r | 0.777 | 0.777 | 0.777 | 0.777 | 0.777 | 0.774 | 0.781 | 0.777 | 0.774 | 0.761 | 0.753 |
| unfiltered / forget-T | alpha_f | 0.749 | 0.749 | 0.749 | 0.748 | 0.746 | 0.732 | 0.753 | 0.762 | 0.760 | 0.727 | 0.718 |
| unfiltered / retain-only | alpha_r | 0.777 | 0.777 | 0.777 | 0.776 | 0.773 | 0.757 | 0.781 | 0.778 | 0.722 | 0.697 | 0.660 |
| unfiltered / retain-only | alpha_f | 0.749 | 0.749 | 0.749 | 0.748 | 0.741 | 0.722 | 0.737 | 0.736 | 0.771 | 0.757 | 0.745 |
| unfiltered-cb / forget-T | alpha_r | 0.824 | 0.824 | 0.759 | 0.805 | 0.715 | 0.712 | 0.727 | 0.686 | 0.661 | 0.637 | 0.609 |
| unfiltered-cb / forget-T | alpha_f | 0.656 | 0.656 | 0.679 | 0.683 | 0.610 | 0.597 | 0.816 | 0.667 | 0.620 | 0.613 | 0.540 |
| unfiltered-cb / retain-only | alpha_r | 0.824 | 0.824 | 0.771 | 0.810 | 0.843 | 0.751 | 0.714 | 0.666 | 0.612 | 0.563 | 0.553 |
| unfiltered-cb / retain-only | alpha_f | 0.656 | 0.656 | 0.643 | 0.625 | 0.645 | 0.672 | 0.653 | 0.670 | 0.636 | 0.612 | 0.615 |

The random-mask null for k = 5% sits at 0.224 (isotropic expectation sqrt(0.05) = 0.2236), so every value above is far outside it.

## 5 — P1

**Verdict: `not_satisfied`.**

- statement: under retain-only, the mean of conflict(t) over cleared steps <= 64 is greater for unfiltered-cb than for e2e-strong-filter, aggregated over (lr, seed) by median, with a run-level bootstrap 95% CI on the difference excluding zero
- window: cleared steps <= 64; cleared = certified (Spec R Amendment 3) AND both gradient norms nonzero
- runs per arm: {'e2e-strong-filter': 1, 'unfiltered-cb': 1}
- arm medians: unfiltered-cb +0.4565, e2e-strong-filter +0.8108; difference **-0.3544**, direction satisfied: False
- reason: the preregistered direction fails: the unfiltered-cb arm median is BELOW the e2e-strong-filter arm median, so the conjunction P1 states is false independently of the interval. The run-level interval was in any case unavailable at Tier 1 (n=1 run per arm).

## 6 — Post-hoc interpretation note on the P1 reversal

**Labelled post-hoc. This is a hypothesis, not a finding, and nothing in this run tests it.**

conflict(t) measures loss-coupling, not mechanism. Letter-cross-entropy on the forget probe improves under benign finetuning for generic reasons — calibration, output format, general biology — so for an ordinary model the forget-gradient is very nearly a retain-gradient, and a high cosine is what one should expect by default. On that reading the filtered model's +0.81 is generic transfer rather than anything about filtering, and unfiltered-cb's depressed +0.46 is the interesting number: something in CB sits in the loss path and decouples the two objectives.

That would inverse the sign of the original mechanistic expectation — the intervention shows up as *less* loss-coupling, not more. It is consistent with the measurement and with CB being the only model whose conflict curve is non-monotonic, but it is equally consistent with other stories, and no control here separates generic transfer from intervention machinery. Testing it needs a probe that holds the generic component fixed — the obvious candidate being a matched non-biology probe set, which this spec does not have.

## SURPRISES

1. **The step-1 checkpoint is bitwise the base model.** LoRA's B is exactly zero at step 0 by initialisation and still exactly zero at step 1, because the linear-warmup schedule returns a factor of 0 at the first update — the first optimizer step runs at learning rate zero. conflict(0) and conflict(1) therefore agree to every digit, on every cell. This retroactively explains Spec H2's observation that steps 0 and 1 are argmax-identical in all 36 runs, and it is an unplanned end-to-end check on the merge path: a zero delta reproduces the base gradients exactly.

2. **Subtractive unmerge is not bitwise, and the drift was measured rather than assumed.** The run merges the step-t delta into the base weights and removes it by subtraction; in float32 that does not restore the original bits. Measured against the real adapter deltas over a full 11-checkpoint trajectory, the drift plateaus at 1.49e-08 absolute (1.3e-07 relative) — about 100,000x smaller than the smallest delta applied and about 1000x below the reduction-order precision every reported number already carries. The run was left on one code path rather than edited mid-flight; the committed implementation restores from a pristine copy so any rerun is exact.

3. **P1 is not evaluable on Tier 1, and not for want of data quality.** It requires a run-level bootstrap over (lr, seed); Tier 1 holds exactly one (lr, seed) per (model, condition), so each arm has n=1 and the interval is undefined. Tier 2 cannot supply the missing runs either: it retained adapters only at {0, peak, 512}, and P1's window is steps <= 64. This is an adapter-retention question for Spec R, not an analysis choice available here. The direction is reported; the interval is not manufactured.

4. **unfiltered-cb's retain-Fisher diagonal is far more concentrated than the other two models'.** Its top-1% threshold is 6.5x the unfiltered model's, and similar at every k. This is a property of the frozen Fisher inputs, present before any gradient was taken, and it belongs next to any reading of CB's alpha values.

5. **alpha_r decays along every completed trajectory**, which is the fixed-at-step-0 projector caveat appearing as a measurement rather than a worry:

   | cell | alpha_r(0) @5% | alpha_r(512) @5% |
   |---|---:|---:|
   | e2e-strong-filter / forget-T | 0.781 | 0.748 |
   | e2e-strong-filter / retain-only | 0.781 | 0.659 |
   | unfiltered / forget-T | 0.777 | 0.753 |
   | unfiltered / retain-only | 0.777 | 0.660 |
   | unfiltered-cb / forget-T | 0.824 | 0.609 |
   | unfiltered-cb / retain-only | 0.824 | 0.553 |

   The section 4 scoped optional — recomputing the retain-Fisher diagonal at one late checkpoint and measuring top-k overlap — now has an empirical motivation rather than a hypothetical one.

## 7 — Projector drift (section 4 scoped optional)

Retain-Fisher diagonal recomputed at unfiltered-cb / step 256 of `spec-e__unfiltered-cb__retain-only__lr-5e-05__seed-0`, using fisher_core.accumulate_fisher_diagonal and the study's own forward path so the comparison differs only by the checkpoint.

| k | top-k overlap with step 0 | chance | lift | old threshold | new threshold |
|---|---:|---:|---:|---:|---:|
| 0.01 | **0.1788** | 0.0100 | 17.9x | 2.384e-05 | 3.557e-03 |
| 0.05 | **0.3194** | 0.0500 | 6.4x | 5.808e-06 | 1.210e-03 |
| 0.10 | **0.4257** | 0.1000 | 4.3x | 2.927e-06 | 6.589e-04 |

a large overlap retires the fixed-at-step-0 caveat on P_r cheaply; a small one scopes every alpha claim to 'step-0 subspace' and must be stated as such

6. **The step-0 projector drifts substantially, so every alpha claim is scoped to the step-0 subspace.** The section 4 check recomputed the retain-Fisher diagonal at unfiltered-cb / retain-only step 256 with the study's own estimator and forward path. Only **17.9%** of the step-0 top-1% coordinates are still top-1% at the peak (8,100,389 of 45,310,858); at k = 5% it is 31.9% and at k = 10% 42.6%. The sets are far from independent — 18x chance at k = 1% — so real structure survives, but four-fifths of the top-1% set turns over. Under this spec's own reading rule that does NOT retire the fixed-at-step-0 caveat: it confirms it, and every alpha number in this log is a statement about the step-0 subspace rather than about the retain-Fisher subspace at the step where it was measured. The alpha_r decay from 0.82 to 0.55 along the trajectory is consistent with exactly this, and the two observations are not independent evidence.

   The absolute Fisher scale also moves by roughly two orders of magnitude (top-1% threshold 2.38e-05 at step 0 against 3.56e-03 at step 256), which is why the comparison is made on rank-selected sets rather than on threshold values.

## Artifacts

| path | contents |
|---|---|
| `spec_gradgeo_outputs/audit.json` | Phase 0: Tier 1 cells, certification inheritance, per-model top-k thresholds, projector caveats |
| `spec_gradgeo_outputs/cells/<run>.json` | per checkpoint: conflict, both nulls, alpha_f and alpha_r per k, random-mask bands, both reduction-order replicates, certified flag |
| `spec_gradgeo_outputs/decision.json` | P1 and the section 2 controls, evaluated mechanically |
| `scripts/spec_gradgeo_{audit,gradients,decision,run_log}.py`, `tests/test_spec_g.py` | implementation and 17 tests |

Per Spec G section 6: alpha as an anchoring evaluation metric remains the anchoring paper's property. What this spec offers the workshop paper is the conflict(t) mechanism figure for the double dissociation. No interpretation is written here.
