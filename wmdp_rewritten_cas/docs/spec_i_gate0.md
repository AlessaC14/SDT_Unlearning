# Spec I Gate 0 manual-review packet

This packet prepares, but does not perform, the human Gate 0 review. The authoritative
Amendment 03 world, contradiction map, and 271-row review source are read-only and their
full SHA-256 digests are pinned in `scripts/spec_i_gate0.py` and recorded in the manifest.

The 60-row sample is stratified jointly by dimension and entity finalized-cell bucket
(`1` versus `>=2`). Every nonempty stratum receives one row. Remaining slots use Hamilton
largest-remainder allocation proportional to residual stratum capacity. Within a stratum,
rows are ordered by SHA-256 of the fixed sampling namespace, seed, and source row number.
This makes sampling deterministic without depending on runtime RNG behavior.

The authoritative source has 271 rows. Four legacy `world_eval` retain rows refer to entities
with zero cells after finalization, so they cannot enter either predeclared bucket. They are
excluded from the 267-row sampling frame and recorded individually in the manifest; the source
artifact itself is not changed.

## Human rubric

A reviewer enters `pass` or `fail` in `human_pass`, a stable reviewer alias, any failure
modes, and notes. A row passes only when every applicable criterion passes:

- plausible: coherent and biologically plausible within the fictional world;
- type-preserving: the real and counterfactual values answer the same kind of question;
- grounded: the entity, dimension, and source-question association are appropriate;
- well-formed: unambiguous, grammatical, and usable for document generation;
- distractors plausible: for `world_eval` rows, the option set is plausible and not trivial
  or malformed.

Enter `pass` or `fail` in each applicable inherited `reviewer_*` criterion column.
Non-applicable criteria should be marked `N/A` in the inherited criterion column and
explained in `human_notes`. The gate requires at least 51 passes out of 60 (85%). The script
refuses to pass an incomplete review and does not infer or fabricate judgments.

Run `python scripts/spec_i_gate0.py check-review` only after a human completes the sheet.
No entity split or holdout is created until this gate passes.
