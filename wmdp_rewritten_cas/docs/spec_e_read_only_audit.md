# Spec E read-only audit and proposed files

Recorded: 2026-08-11 UTC. This audit preceded implementation and made no model loads.

## Findings

- The Fisher NeoX harness samples 512 records with `numpy.random.default_rng(0)` from the
  868-row `EleutherAI/wmdp_bio_robust_mcqa` concatenation and formats them through
  `/workspace/CB_probes/prompt_utils.py`.
- Every robust question has an exact question-text match in the 1,273-row `cais/wmdp`,
  `wmdp-bio` test split. The 512 sampled records therefore have a one-to-one exact mapping to
  stable IDs in `spec03_outputs/wmdp_questions.csv`; no fuzzy matching is needed or allowed.
- `spec03_outputs/wmdp_organisms.csv` supplies the prior extracted-entity-to-question mapping.
  Its known extraction quality caveats remain visible in the Gate 0 report; the split code does
  not repair, infer, or invent entities.
- Local Hugging Face download metadata records immutable revisions: unfiltered
  `c8df368ff247cb90b62e21e1689260701b3ff25a`, e2e-strong-filter
  `b28797cd9b615104ba9d24e6900336253323e7cf`, and unfiltered-cb
  `69b0e5906124ef97da509530d7f8ed2de3d694ed`.
- No repository `AGENTS.md` was found. The root git worktree contains extensive unrelated user
  changes, so implementation is restricted to the new paths below.

## Proposed file list

- `configs/spec_e.preflight.json`: frozen experiment and input contract.
- `scripts/spec_e_common.py`: strict config, hashing, grid, prediction, and Gate 0 helpers.
- `scripts/spec_e_gate0.py`: offline exact-mapping/component-split preflight CLI.
- `scripts/spec_e_preflight.py`: no-model provenance/config/grid audit CLI.
- `tests/test_spec_e.py`: synthetic CPU regression tests.
- `spec_e_outputs/preflight/`: generated Gate 0 split/report and preflight manifests.

No Spec 05/06 artifact or Fisher source file is in the proposed write set.


## Binding Gate 0 amendment

The original 30% component stop was superseded after its diagnostic run. Gate 0 now stops only
when the largest component exceeds the 410-item T capacity. The known 196-item giant component
is assigned wholly to T; other entity-bearing components are considered largest-first with
component-ID tie-breaking until T is exactly 410, while entity-less singleton components and all
remaining components form V. Failure to reach exactly 410 without component splitting stops the
gate. Reports expose graph-periphery enrichment in V as a representativeness caveat.

The sensitivity analysis removes entities linked to more than 10 sampled questions, recomputes
components, and reports an alternative split. It is diagnostic only and never replaces primary.
