# Spec H2.1 run log — matched-step reference control

Spec H2.1 as written. CPU only, same kernel and frozen shards as Spec H2; no model loads, no new tables. **Spec H2's outputs are inputs only and were not modified.**

## 1 — What was computed

264 (cell, step) records over the 24 non-unfiltered grid cells (e2e-strong-filter and unfiltered-cb × 2 conditions × 3 lr × 2 seeds) × 11 logged steps, plus 132 unfiltered control cells. Runtime 57 s single-core.

`L_matched(t)` is the unfiltered model's own grid prediction at the identical (condition, lr, seed, step), ID-joined on question_id, same view, same lowest-letter tie rule. The accuracy-matched null model re-solves `p` at every step, since the matched reference's label information moves along the trajectory.

## 2 — Gates

| gate | result |
|---|---|
| H2 input manifest re-asserted | **pass** — 396 files, 121,572 rows; sha256 re-verified on the step-0 and step-512 subset |
| excess_preattack read, never recomputed | **pass** — H2 `trajectories.json` sha256 `72e0092898ec3e04…`, 396 records |
| cross-check vs H2's stored excess curve | **pass** — 396 points, max difference 0.0e+00 |
| self-reference tautology | **pass**, with a recorded carve-out — see §3 |
| estimator identity with H2 (MM cross-check) | **pass** — `tests/test_spec_h2_1.py`, 20 tests |
| frozen trees never opened for writing | **pass** — asserted in code and in tests |

## 3 — Decision (§1 rule, fixed before any H2.1 number existed)

**unfiltered-cb / retain-only → `A_staleness`**

- median Δexcess = **+0.027810**, run-level bootstrap 95% CI **[+0.001222, +0.051719]** over 6 runs
- excess_matched clears its permutation-null band at **19 of 21** jointly cleared steps
- median Delta-excess is positive with a run-level 95% interval excluding zero, and excess_matched clears its permutation-null band at a majority of jointly cleared steps

Conclusion as the rule states it: *recovery tracks the treated intact function; the pre-attack reference was stale; H2's near-null excess is a reference artifact*

| run | steps cleared (both) | I_t cleared | reference cleared | median Δexcess | μ clears null |
|---|---:|---:|---:|---:|---:|
| lr 1e-05 seed 0 | 3 | 3 | 11 | -0.00439 | 1 |
| lr 1e-05 seed 1 | 5 | 5 | 11 | +0.04372 | 5 |
| lr 2e-04 seed 0 | 1 | 1 | 6 | +0.01190 | 1 |
| lr 2e-04 seed 1 | 2 | 2 | 7 | +0.00683 | 2 |
| lr 5e-05 seed 0 | 4 | 4 | 11 | +0.05409 | 4 |
| lr 5e-05 seed 1 | 6 | 6 | 11 | +0.04935 | 6 |

## 4 — Descriptive cell families (no preregistered rule)

| family | outcome | median Δexcess | 95% CI | steps cleared (both) | μ clears null |
|---|---|---:|---|---:|---:|
| e2e-strong-filter / forget-T | B_divergence | -0.00663 | [-0.01771, +0.03786] | 22 | 8 |
| e2e-strong-filter / retain-only | A_staleness | +0.00303 | [+0.00176, +0.00636] | 56 | 54 |
| unfiltered-cb / forget-T | A_staleness | +0.04343 | [+0.04266, +0.04420] | 5 | 4 |

## 5 — Clearance census (first-class result)

- reference-uncleared steps: **99 of 264**, by learning rate: {'1e-05': 20, '2e-04': 55, '5e-05': 24}
- I_t-uncleared steps (H2's own decision, carried in): **136 of 264**
- cleared in **both** analyses: **104 of 264**

Where the reference is uncleared, μ is stored and excess and Δexcess are null-valued, not zero.

## SURPRISES

1. **The tautology control's ratio clause has an edge case the spec text omits.** It failed on **14 of 132** unfiltered cells on the first pass, and §2 declares any deviation a join or tie-rule bug that stops the run. None of the 14 is either. In all of them F and L are **identical arrays** and μ equals I_t to the last bit; what fails is only `ratio = 1`, because these are the degenerate lr-2e-04 cells where F has collapsed to a single letter, so I_t is exactly 0 and the ratio is 0/0 — undefined under H2's epsilon rule, not 1. Spec H2.1 §2 already anticipates these very cells for the *reference* clearance rule; the tautology clause simply does not carve them out. The two clauses carrying the control's diagnostic content — argmax identity and μ = I_t — are evaluated at every step without exception and hold everywhere (max |μ − I_t| = 8.9e-16, float summation order over transposed count arrays, the residual H2's Ruling 3 check already records). The ratio clause is evaluated on the 118 cells where it is defined and holds on all of them. The 14 are listed by name in `matched_reference.json`. Recorded as a spec-text gap for Alessa, not absorbed.

2. **Outcome A rests on a razor-thin interval.** The CI lower bound is +0.001222 — it excludes zero by about a thousandth of a bit. The interval is a bootstrap of a **median over six runs**, so it is built from a small discrete set of attainable medians and is coarse by construction. One of the six runs points the other way: lr 1e-05 seed 0 has median Δexcess −0.00439, on only 3 cleared steps of which 1 clears the μ null. The rule was met as written; the margin is thin and the reader should see the per-run column, not only the aggregate.

3. **The reference clearance never binds in the target family — H2's I_t clearance does all the work.** In all six unfiltered-cb / retain-only runs, `steps_cleared_both` equals `steps_i_cleared` exactly (3, 5, 1, 2, 4, 6), while the matched reference clears at 11/11 steps for lr 1e-05 and 5e-05 and 6–7/11 at lr 2e-04. The decision is therefore evaluated on 21 of 66 steps, and the 45 lost steps are lost to H2's I_t null, not to anything H2.1 introduced.

4. **e2e-strong-filter / retain-only also returns A, on far more evidence but a much smaller effect.** 56 jointly cleared steps with 54 clearing the μ null, median Δexcess +0.00303, CI [+0.00176, +0.00636]. The reference-staleness effect is not unique to the circuit-breaker model — it is present in the never-knew model too, roughly nine times smaller and far more tightly determined. Whether that is a shared artifact of comparing against a drifting reference or a real shared phenomenon is not decided here.

5. **The forget-T families split and are thin.** unfiltered-cb / forget-T returns A on only 5 jointly cleared steps; e2e-strong-filter / forget-T returns B with 8 of 22 steps clearing. At V-102 with ~1.6 observations per 64-cell joint, neither is worth leaning on, and neither carries a preregistered rule.

6. **Reference-uncleared steps concentrate exactly where H2's census predicted.** 99 of 264 steps have a reference that fails its own label-permutation null, 55 of them at lr 2e-04 — the single-letter collapse H2 documented. The matched reference inherits the intact model's degeneracy, which is a property of this design rather than a defect in it.

## Artifacts

| path | contents |
|---|---|
| `spec_h2_1_outputs/matched_reference.json` | per (model, condition, lr, seed, step): μ_matched, excess_matched, I(L_matched;Y), both clearance booleans, Δexcess where joinable, null bands, bootstrap CIs, and the 132 control cells |
| `spec_h2_1_outputs/decision.json` | the §1 rule evaluated mechanically, with the per-run medians and CIs it was evaluated on |
| `scripts/spec_h2_1_matched_reference.py`, `scripts/spec_h2_1_decision.py` | the computation and the rule |
| `tests/test_spec_h2_1.py` | 20 tests: estimator identity, every decision branch, the clearance conjunction, Δexcess arithmetic, no-write and no-model invariants |

Spec H2's `trajectories.json` and `phase_stats.json` are unmodified and were read only. H2's P1 and P2 remain `not_evaluable`; H2.1 answers a different question and does not re-treat them. Whether H2's pre-attack-reference interpretation is qualified by `decision.json` is Alessa's call to write, not the code's.
