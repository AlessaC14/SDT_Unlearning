# Spec H2 — performance-correlation trajectories

Implementation notes for Spec H2 as amended by **Amendment 1 (post-plan rulings),
2026-08-14**. Results and surprises live in
[spec_h2_outputs/RUN_LOG.md](../spec_h2_outputs/RUN_LOG.md); this file documents the
code.

## Estimands

For every relearning run and every logged step t ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128,
256, 512}, over the 4×4×4 = 64-cell joint of (F_t, Y, L):

- `I_t = I(F_t; Y)`
- `mu_intact(t) = I(F_t;Y) − I(F_t;Y | L_intact)`
- `mu_shortcut(t) = I(F_t;Y) − I(F_t;Y | L_shortcut)`
- `ratio(t) = mu_intact(t) / I_t`, reported only where I_t clears its own null

F_t is the four-way argmax at step t, ties broken toward the lowest letter index.
Y is the gold label. L_intact is the argmax of the **unfiltered** exact-path
pre-attack full table (shared across models). L_shortcut is the argmax of the
model's own exact-path options-only pre-attack table.

Every quantity is computed plugin **and** Miller–Madow, both stored. Plugin is
primary for trajectory shape: the bias at 64 cells is shared between the estimate
and its permutation null at identical N and near-identical marginals, so it is
handled by the nulls rather than by correction.

Amendment 1 Ruling 3 adds, as **exploratory** quantities only:
`I(L;Y)` per reference per view, `ceiling(t) = min(1, I(L_intact;Y)/I_t)`, and
`ratio_min(t) = mu_intact(t) / min(I_t, I(L_intact;Y))`. The original ratio is
structurally bounded by the ceiling, since μ ≤ min(I(F;Y), I(L;Y)) (Nakkiran Def. 1).

## Modules

| file | role |
| --- | --- |
| [configs/spec_h2.json](../configs/spec_h2.json) | every constant as a frozen literal — inventory, gates, resampling parameters, thresholds, supersession pointer |
| [scripts/spec_h2_kernel.py](../scripts/spec_h2_kernel.py) | vectorised MI/CMI/MM primitives; pure numpy, no I/O |
| [scripts/spec_h2_io.py](../scripts/spec_h2_io.py) | table loading, hashing, question-ID joins, run enumeration, write guards |
| [scripts/spec_h2_preflight.py](../scripts/spec_h2_preflight.py) | Phase A, fail-closed, exit 2 on any gate |
| [scripts/spec_h2_trajectories.py](../scripts/spec_h2_trajectories.py) | Phase B driver, one shard per (run, step), restartable |
| [scripts/spec_h2_phase_stats.py](../scripts/spec_h2_phase_stats.py) | Phase C; reads `trajectories.json` only |
| [tests/test_spec_h2.py](../tests/test_spec_h2.py) | 32 tests over the normative contracts |

Run order:

```
python3 scripts/spec_h2_preflight.py      # must print status "ready"
python3 scripts/spec_h2_trajectories.py   # refuses to run otherwise
python3 scripts/spec_h2_phase_stats.py
python3 -m pytest tests/test_spec_h2.py -q
```

## The kernel

A joint is built as one `bincount` over the code `f*16 + y*4 + l`, reshaped to
(4, 4, 4). Batched forms offset each batch row by 64 so that B tables come out of a
single `bincount` of length 64B. Entropies are taken over marginalisations of that
array; the Miller–Madow correction is `(occupied_cells − 1) / (2 n ln 2)` per
entropy term, which is the same quantity `spec_h_common.entropy` computes — asserted
to 1e-12 in the tests, so the estimator is provably identical to Spec H's.

Amendment 1 Ruling 6 forbids `spec_h_common`'s Counter-over-tuples path in any inner
loop; it is used only as a test oracle. The single symbol imported from it in
production is `solve_null_p`, which walks a 4×4 channel once per cell to match the
null model's accuracy and never enters a resampling loop. A test enforces that
allowlist by parsing the imports.

Undefined ratios are NaN inside numpy and JSON `null` at the serialisation boundary.
Negative values are never clipped.

## Joins

All joins are by `question_id`. Positional zips are forbidden: forget-T tables are in
sorted V-102 order while the retain-only and reference tables share one shuffled
canonical 512 order, so a positional read would silently mislabel the reference.
`spec_h_common.validate_alignment` is never called — it demands *ordered* equality
and coerces gold with `int()`, and gold is stored as a letter. Both facts are covered
by tests.

## Nulls, bias handling, uncertainty

| component | design |
| --- | --- |
| I_t null | 100 label permutations per (run, step); I_t is cleared when the plugin estimate exceeds the permutation 95th percentile. `ratio` and `ratio_min` are null-valued (not zero) at uncleared steps; I_t and μ are still reported raw |
| μ null | 100 item-wise permutations of L against (F_t, Y) — destroys F–L coupling, preserves all marginals |
| bootstrap | 500 question-level resamples, percentile 95% intervals, indices shared across references so the two μ curves resample together |
| null model | Nakkiran footnote 5: L̃ = Y with probability p, else uniform over the other three letters, p solved so I(L̃;Y) matches I(L_intact;Y) on the same items. Ruling 4 raised the draws from 20 to 100 (unpeeked) and replaced the normal approximation with a direct 2.5/97.5 percentile band |

Determinism: seed 42 throughout. Each workload draws from
`np.random.default_rng([42, sha256(run_id)[:8], step, workload_id])`, so any single
cell reproduces in isolation and adding cells never perturbs existing ones. The
derived seed sequence is written into every shard.

## Output schemas

`trajectories.json` — `metadata` (primary estimator, resampling parameters,
aggregation deviation, supersession pointer, reference tie counts and caveat,
per-run clearance counts) plus one record per (model, condition, lr, seed, step,
quantity, estimator):

```jsonc
{
  "model": "unfiltered-cb", "condition": "retain-only", "lr": "1e-05",
  "seed": 1, "run_id": "spec-e__...", "step": 128, "n": 512,
  "quantity": "mu_intact", "estimator": "plugin",
  "register": "preregistered",          // or "exploratory" (Ruling 2)
  "value": 0.0535,
  "ci95": {"low": ..., "high": ..., "n_finite": 500, "resamples": 500},
  "null": {"kind": "item_wise_L_permutation", "mean": ..., "p5": ..., "p95": ...},
  "i_cleared": true,
  "reference": {"name": "intact", "path": "...", "sha256": "..."},
  "source_table": {"path": "...", "sha256": "..."}
}
```

`ratio` and `ratio_min` records carry `"value": null` and
`"suppressed_by_clearance": true` at uncleared steps. The suppressed number is not
smuggled in under another key; it is recoverable from the μ and I_t records, which
are always emitted raw.

`phase_stats.json` — exactly two disjoint result keys (Ruling 2). `preregistered`
holds t80, t_shortcut_peak, excess, per-run curves, median curves, and the P1/P2/P3
verdicts. `exploratory` holds the {0.2, 0.4, 0.6, 0.8} threshold sweeps, ratio_max,
ratio_min, ceiling, and I(L;Y). Disjointness is asserted at write time; the script
exits 2 on violation. `metadata.interpretation` is `null` — Spec H2 §7 makes
interpretation a non-goal.

## Ruling 1 — the vacuous verdict

`compare_t80` returns `not_evaluable` when both sides are infinite, *before* any
comparison operator is reached, so `satisfied` is structurally unreachable from that
state. `test_satisfied_is_unreachable_when_neither_side_is_finite` covers the whole
degenerate quadrant (inf and NaN, both directions). Infinity is encoded in JSON as
`"t80": null` with a companion `"t80_is_infinite": true` and a `"t80_label"` of
`"inf"`. Because steps 0 and 1 are argmax-identical in all 36 runs, a t80 of 0 or 1
is labelled `"<=1"` (Ruling 6).

## Aggregation

No max-selection. The grid's checkpoint-maximum rule exists to measure attack
strength; H2's estimand is dynamics, and selecting the maximum would bias toward runs
where recovery happened fast. All 36 runs × 11 steps are computed, aggregation across
(lr, seed) is by median at each step, and per-run curves are persisted. The deviation
is stated in both output files' metadata.

## Invariants

- No `torch` / `transformers` / `peft` / `accelerate` import anywhere in the H2 tree.
- Nothing under `spec_h_outputs/` is ever opened for writing; `assert_write_target`
  guards the output root and a test exercises the rejection.
- The superseded Spec H tier-3 peak-point records are untouched. Both H2 output files
  carry a pointer to them and the one-line supersession reason: Spec H read μ only at
  single accuracy-selected peaks, where small-cell Miller–Madow instability left it
  undetermined; H2 replaces the point readout with the full trajectory plus
  permutation nulls and an accuracy-matched null model.

## Cost

Preflight 27 s, trajectories 38 s, phase statistics 0.4 s, tests 5 s — all
single-core CPU. Roughly 523,000 contingency tables at n ≤ 512. No parallelism; the
shard layout exists for restartability, not speed.
