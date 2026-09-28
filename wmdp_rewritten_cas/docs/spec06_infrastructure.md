# Spec 06 infrastructure

Status: infrastructure only. No training or model evaluation has been run.

`scripts/spec06_train.py` is the only training entry point. The five arms are data selected by configuration; hyperparameters and scheduling use one code path. It defaults to preflight and
requires both `--execute` and the explicit execution acknowledgement environment variable.
Pilot preflight requires only the finalized `<WORLD>`, matched one-section `<CORPUS_F>` and `<CORPUS_W>` pilot corpora, the in-section direct/retain `<WEVAL>` records, and GENERAL. It permits exactly two pilot runs, `factual` and `world`. Full preflight separately requires the
full corpora, `<SUBSETS>`, row identifiers, BENCH, `<CMAP>`, and complete `<WEVAL>`. Preflight
rejects absent required artifacts, an unpinned revision, fewer than three seeds,
an undecided training mode, an unspecified checkpoint-retention policy, adapter targets without
MLP projections, non-nested or composition-mismatched paired sweep subsets, different control/naive row identifiers, corpus overlap, incorrect G1 polarity evidence, G3 failure evidence, non-identical scaffold generation configuration (including exact document-type allocation), or paired-corpus mismatch beyond 1% documents, 2% tokens, and the frozen doc-type tolerance. Token
length is measured without truncation before training, and any overlength record stops the run.
The generic `arm_sources` mapping selects a path and text column without changing the training
logic. Control and naive deliberately point to different columns in one joined source artifact;
`factual` and `world` point to independently generated `<CORPUS_F>` and `<CORPUS_W>`. They share one training configuration, including seeds, hyperparameters, and schedule. Ordered row-ID equality is still asserted for control/naive. The enforced plan is five factual and five world sweep points, two endpoint runs each for control and naive, and one untrained base evaluation: 14 trained runs and 15 evaluated conditions including base.

`scripts/spec06_evaluate.py` is the only evaluation entry point. BENCH, `<WEVAL>`, and GENERAL
all use the same format-matched log-likelihood scorer. Option order is fixed by an item-identifier
hash. `<WEVAL>` writes independently normalized `p_real` and `p_counterfactual`; BENCH records
the `<CMAP>` split. Outputs contain identifiers and numeric scores only.
With `--pilot`, the same scorer loads only in-section direct/retain `<WEVAL>` and GENERAL; full
mode adds BENCH, `<CMAP>`, and the complete `<WEVAL>`.

`scripts/spec05_validate_pair.py` is a standalone, content-blind Spec 05 acceptance gate. It asserts shared scaffold configuration with opposite value-set polarity, produces the paired document/token/doc-type comparison, rejects both identifier and content-digest overlap, checks both nested subset ladders for matched scaffold/doc-type composition, and requires own-target G1 success plus other-target G1 failure and G3 success. It validates artifacts once they exist; it does not generate either corpus.

The analysis helpers produce ENTITY-cluster bootstrap intervals and a separate between-seed
variance summary. Effective sample size is always the number of ENTITY clusters. Deltas must be formed against the cached base evaluation after seed summaries are complete. Prediction P7 is emitted as an explicit `apparatus_gate` from the positive factual BENCH overall and contradicted-item deltas. World analysis requires the factual analysis artifact; failure labels the world `interpretation_status` as `uninterpretable`. The GENERAL
tolerance, training mode, immutable base revision, and checkpoint retention policy remain
mandatory human decisions in the frozen config.

The template is `configs/spec06.template.json`. Copy it rather than editing it in place, resolve
every `REQUIRED_*` value, compute/freeze its digest, and retain that frozen config with every run.
The timestamped predictions, including P7-P9 and the P7-first interpretation rule, are in `spec06_preregistration.json`.

Current blockers are the unfinished final `<WORLD>` integration, both Spec 05 corpus/pilot, generation-config, polarity-evidence, and matched-subset artifacts, a selected GENERAL input, an immutable base-model revision, an explicit
adapter-versus-full choice, a numeric GENERAL tolerance, and a checkpoint-retention policy.

The Textbook Spine amendment replaces each independent-document corpus construction with its own two-phase pipeline. `<CORPUS_F>` and `<CORPUS_W>` each produce an independent continuous `<SPINE>`, validate its internal chapter/section structure and every cross-reference, record a content digest and bounded character spans, pass G1–G4, and receive recorded human approval before any `<DERIVED>` generation. `<DERIVED>` records the source `<SPINE>` section IDs; `<SPINE_CHUNKS>` records chapter, section, and character spans. The combined arm corpus contains both phases. Diversity evidence is required within `<SPINE_CHUNKS>`, within `<DERIVED>` by document type, across phases, and corpus-wide; template-collapse evidence is scoped to `<DERIVED>` only. The paired-corpus validator preserves exact matched scaffold, allocation, and phase budgets while requiring distinct pipeline and `<SPINE>` IDs, so `<CORPUS_W>` cannot be produced by editing `<CORPUS_F>`.

The Spec 05 pilot now has two ordered gates per arm. Gate 1 requires a complete chapter, all gate metrics, and explicit human approval under a reviewer alias. Gate 2 generates a section of `<DERIVED>` only after that approval and must pass before the Spec 06 vertical slice. That vertical slice is the combined corresponding `<SPINE_CHUNKS>` plus `<DERIVED>`, never `<DERIVED>` alone. The cross-reference resolution floor is intentionally left as `REQUIRED_EXPLICIT_NUMERIC_DECISION` in the template; regardless of the chosen reporting floor, acceptance requires every recorded cross-reference to resolve.
