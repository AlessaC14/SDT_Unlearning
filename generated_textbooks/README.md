# Generated textbooks

This directory contains convenient, non-destructive copies of readable textbook artifacts from `/workspace/wmdp_rewritten_cas`. The pipeline originals remain in place because their paths and hashes are used by validation gates.

| File | Arm | Status | Original |
|---|---|---|---|
| `factual_textbook_full.md` | `CORPUS_F` (real values) | Full six-chapter textbook; automated full-spine gate passed; human review not approved | `scaled_textbook_runs_v1/corpus_f/textbook.md` |
| `alternative_textbook_pilot_chapter_01.md` | `CORPUS_W` (counterfactual values) | Chapter 1 pilot; automated Gate 1 passed; human review not approved | `spec05_runs_judge01/corpus_w/spine_pilot_review.md` |
| `counterfactual_textbook_chapter.md` | Counterfactual textbook draft | Model validation passed; human audit not completed | `counterfactual_textbook_v1/alternative_textbook_chapter.md` |

No full-scale `CORPUS_W` textbook currently exists. Its structured source facts are in `scaled_corpus_v1/textbook_inputs_v1/counterfactual_world_v2.jsonl`.

## Analyses

- `wmdp_abstract_claim_hits.md` — strict case-insensitive, all-terms-present search of the 27 mentor-chapter altered claims against 23,915 nonempty WMDP source-paper abstracts.
