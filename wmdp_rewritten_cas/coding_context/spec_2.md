# Spec 02 — Claim Inventory Construction

**Owner:** Alessa Carbo
**Status:** ready to implement
**Depends on:** Spec 01 outputs
**Estimated runtime:** < 20 minutes CPU, plus ~10 minutes GPU for the §6b probe
**Model calls:** local scoring and one local sentence encoder; no generation, no API
**GPU:** required only for the §6b sizing probe, which is skippable via `--skip-probe`

---

## 1. Purpose

Produce the frozen, canonical claim set that every downstream arm consumes.

Spec 01 established that the rewrite corpus is unusable as a corruption source
(13% across-document consistency, 44% within-document inconsistency, 97% residual
leakage) but yielded 71 verified clean 1:1 entity mappings. This spec harvests those
mappings into probeable claims, augments them with WMDP multiple-choice items to reach
usable sample size, and emits a single versioned inventory.

Downstream consumers of this artifact:

- **Arm 2** (deterministic entity remapping) needs the validated entity mapping table.
- **Arm 3** (textbook / universe-context corpora) needs claim propositions to generate around.
- **Spec 03** (probe construction and base-model scoring) needs real/twin pairs.

The inventory is frozen once accepted. Downstream specs reference `claim_id` and must
not silently re-derive claims.

---

## 2. Non-goals

- No generative LLM calls, no API calls. All extraction is deterministic.
- One local sentence encoder is permitted, for distractor selection only (§6). No
  generation, no judging, no paraphrasing.
- No base-model **filtering**. The survival-rate probe in §6b is a sizing measurement
  and must not retain or drop any individual claim. Filtering is Spec 03.
- No probe or question generation. That is Spec 03.
- No corpus generation. That is Spec 04.
- No modification of Spec 01 outputs or the source parquet.
- No network access except a single optional WMDP dataset fetch (§4), which must be
  explicitly authorized and hash-recorded.

---

## 3. Inputs

| Input | Path | Use |
|---|---|---|
| Mapping table | `<S01_OUT>/parquet_char_mapping_table.jsonl` | Source of clean 1:1 pairs |
| Substitutions | `<S01_OUT>/parquet_char_substitutions.jsonl` | Offsets and context for sentence extraction |
| Per-doc records | `<S01_OUT>/parquet_char_per_doc.jsonl` | Leakage flags, row IDs |
| Source corpus | `<CORPUS_ROOT>` | Sentence extraction from `abstract` / `text` |
| WMDP MCQ | `<WMDP_PATH>` | Volume augmentation |

All inputs read-only. Validate schemas against Spec 01's documented output schemas and
**halt non-zero on mismatch**. Do not coerce.

### WMDP source

Prefer a local copy. Expected subsets: `wmdp-bio`, `wmdp-chem`, `wmdp-cyber`, with
per-item fields for question text, a list of choices, and a correct-answer index.
Validate the observed field names and print them; if they differ from the above, halt
and report rather than guessing.

If no local copy exists, halt with a message requesting authorization rather than
downloading unprompted.

---

## 4. Outputs

Write to `<OUT_DIR>`.

| File | Format | Contents |
|---|---|---|
| `claim_inventory.jsonl` | JSONL | The canonical claim set |
| `claim_inventory_summary.json` | JSON | Counts, yields, rejection reasons |
| `entity_mapping_validated.jsonl` | JSONL | Clean entity pairs for Arm 2 |
| `claim_review_sample.tsv` | TSV | 60 claims for manual annotation |
| `claim_inventory_report.md` | Markdown | Human-readable summary |

### `claim_inventory.jsonl`

```json
{
  "claim_id": "string, stable: CP-#### (parquet-harvested) or CW-#### (WMDP)",
  "source": "cas_parquet | wmdp_mcq",
  "domain": "bio | chem | cyber | unknown",
  "real_claim": "string, full proposition",
  "twin_claim": "string, full proposition, minimal pair with real_claim",
  "real_answer": "string, short answer span",
  "twin_answer": "string, short answer span",
  "diff_span": {"real": "string", "twin": "string", "token_start": "int", "token_end": "int"},
  "swap_type": "entity | numeric | other",
  "entity_pair": {"source_norm": "string", "target_norm": "string"},
  "claim_tokens": "int",
  "n_supporting_docs": "int, distinct source docs containing this pair; 0 for WMDP",
  "provenance": {
    "row_ids": ["int"],
    "dois": ["string"],
    "sub_ids": ["string"],
    "wmdp_subset": "string or null",
    "wmdp_index": "int or null",
    "all_distractors": ["string"]
  },
  "type_preserving": null,
  "flags": ["string"]
}
```

`type_preserving` is written as `null` by the pipeline and filled by manual review.

### `entity_mapping_validated.jsonl`

One record per validated pair, for Arm 2's deterministic remapper:

```json
{
  "source_norm": "string",
  "target_norm": "string",
  "source_surface_forms": ["string"],
  "target_surface_forms": ["string"],
  "total_occurrences": "int",
  "n_documents": "int",
  "modal_share": "float",
  "swap_type": "entity | numeric | other",
  "collision_free": "bool"
}
```

### Determinism

Byte-identical outputs across runs. Sort inventory by `claim_id`; sort mapping by
`total_occurrences` desc then `source_norm` asc; sorted JSON keys; `random.Random(42)`
for the review sample. Record Python and library versions, input SHA-256 hashes, and
a UTC timestamp in the summary.

---

## 5. Task A — Harvest claims from the parquet corpus

For each `is_clean_1to1` term in the mapping table:

1. Collect its substitution records from `parquet_char_substitutions.jsonl`.
2. For each substitution, use `source_char_start` to locate the containing **sentence**
   in the row's `abstract`. Segment sentences with a deterministic rule-based splitter
   (regex on `.!?` followed by whitespace and an uppercase character, with a
   hard-coded abbreviation exception list recorded in the summary). Do not use an
   ML sentence segmenter.
3. Extract the aligned sentence from the row's `text` by running the same word-level
   `SequenceMatcher` alignment used in Spec 01, restricted to the sentence span.
4. `real_claim` = abstract sentence. `twin_claim` = aligned rewritten sentence.

### Minimal-pair gate

Token-diff `real_claim` against `twin_claim`. **Accept only if the diff yields exactly
one `replace` opcode and zero `insert`/`delete` opcodes.** Reject everything else.

This is the gate that makes the claim near-manifold and it is non-negotiable — the
corpus is a paraphrase-with-swaps rewrite (34 substitutions per document), so most
sentences will contain multiple edits and will correctly fail here. A low pass rate is
expected and is not an error.

### Additional rejection rules

Reject and count each reason separately:

- `real_claim` exceeds 60 tokens or is under 6 tokens.
- The substituted term still appears elsewhere in `twin_claim` (sentence-level residual leak).
- The sentence contains a citation marker, figure/table reference, or is a heading fragment.
- The sentence is duplicated (normalized) by an already-accepted claim.
- The swap is `trivial` per Spec 01 categorization.

### Deduplication

If one entity pair yields several accepted sentences, keep the **shortest** accepted
sentence as the claim and record the remainder count in `n_supporting_docs`. One claim
per entity pair maximum.

Expected yield: at most 71 claims, likely fewer. Report the yield and the full
rejection-reason histogram.

---

## 6. Task B — Augment from WMDP MCQ

For each MCQ item:

1. Apply eligibility filters. Reject if:
   - the correct answer exceeds 12 tokens
   - any choice matches `all of the above`, `none of the above`, `both a and b`, or similar (case-insensitive, closed list recorded in summary)
   - the question is not answerable as a standalone proposition (contains `which of the following`, `the following`, or references an absent figure/passage)
   - **the item is procedural or operational**: the correct answer describes a synthesis route, dosing, transfection protocol, acquisition method, or step-by-step operational instruction (keyword list, recorded in summary, flagged conservatively — when uncertain, reject)
2. Select the **near-manifold distractor**: among the three incorrect choices, pick the
   one maximizing **cosine similarity of sentence embeddings** against the correct
   choice. Use a local sentence encoder (`sentence-transformers/all-MiniLM-L6-v2` or
   equivalent), CPU inference, fixed weights, no network. Break ties by smallest
   absolute token-length difference, then lexicographic order.

   String-overlap similarity is **not** acceptable here. Semantically adjacent entities
   frequently share no surface form (`Ebola` / `Marburg`), so lexical similarity would
   select distractors close to arbitrarily and destroy the near-manifold property that
   the whole claim design depends on.

   Record in each claim: the chosen distractor, all three distractors with their cosine
   scores, and the margin between the top two. A small top-two margin means the choice
   was near-arbitrary — flag those claims as `ambiguous_distractor` for review.

   Record the encoder identifier and weight-file SHA-256 in the summary.
3. Construct claims by templating question + answer into a declarative proposition
   where the question form permits it; otherwise retain the interrogative form with
   the answer appended. Use a small fixed set of deterministic templates, recorded in
   the summary. Do not paraphrase.
4. `real_answer` = correct choice; `twin_answer` = selected distractor.

The procedural filter is deliberate: a refusal-style item in a factual-flip set
contaminates aggregates and is not distribution-matched to entity swaps. Rejecting
these up front avoids the problem rather than reporting around it later.

Report per-subset counts before and after each filter.

---

## 6b. Task B2 — Base survival-rate probe

**Purpose: sizing only.** This estimates how many claims will survive Spec 03's base
`P(real)` filter, so that filter scope can be adjusted now rather than after a full
round-trip.

1. Sample 50 claims from the merged inventory with `random.Random(1337)`, stratified
   across source and domain.
2. Score each under the base model (`HuggingFaceH4/zephyr-7b-beta`, local, greedy,
   fixed seed) as a forced choice between `real_answer` and `twin_answer`, using the
   same option-parity construction planned for Spec 03.
3. Report the **aggregate** fraction with base `P(real)` above each of
   `{0.5, 0.6, 0.7, 0.8}`, and the projected survivor count at inventory scale.

### Hard constraints

- **No claim may be retained or dropped on the basis of its score.** Per-claim scores
  are written to a separate file, `base_probe_scores.jsonl`, and are **not** merged
  into `claim_inventory.jsonl`.
- The inventory written by this spec must be identical whether or not the probe runs.
  Verify this: run once with `--skip-probe` and once without, and assert the inventory
  SHA-256 matches.
- Permitted downstream response: widening or narrowing **filter categories** in §6
  (e.g. raising the answer-length cap, admitting another subset). Not permitted:
  selecting individual claims.

This step requires a GPU. If unavailable, `--skip-probe` and proceed; the §11
thresholds then apply unadjusted.

---

## 7. Task C — Merge, canonicalize, annotate

1. Assign stable IDs: `CP-0001…` in mapping-table order, `CW-0001…` in
   (subset, index) order.
2. Cross-source deduplication: reject a WMDP claim whose normalized `real_answer` and
   `twin_answer` both match an existing parquet-harvested claim.
3. Record `claim_tokens` using the local Zephyr tokenizer (whitespace fallback
   permitted, recorded).
4. **Entity co-occurrence structure**, for later derivability analysis: for each claim,
   record in `flags` the count of other claims in the inventory sharing a normalized
   content term with its `real_answer`. Do not compute a derivability score here —
   just preserve the structure so it can be computed downstream.

---

## 8. Task D — Build the validated entity mapping

Emit `entity_mapping_validated.jsonl` from `is_clean_1to1` terms, including all
observed surface forms of both source and target (needed for regex application in
Arm 2) and a `collision_free` flag from Spec 01's reverse-mapping analysis.

Include mappings whose claims were rejected in Task A — a pair can be a valid
remapping rule even if no single clean sentence was extractable for it.

---

## 9. Task E — Manual review sample

Sample 60 claims with `random.Random(42)`, stratified: at least 20 `cas_parquet`
(or all of them if fewer than 20 survive) and at least 30 `wmdp_mcq`, spread across
available domains.

`claim_review_sample.tsv` columns:

```
claim_id  source  domain  real_claim  twin_claim  real_answer  twin_answer  swap_type
reviewer_type_preserving  reviewer_plausible  reviewer_is_claim  reviewer_notes
```

Three annotation questions, stated in the report:

- **`type_preserving`** — is the twin the same kind of thing as the real (virus→virus,
  not virus→bacterium)? Type-violating swaps are not near-manifold and will be
  trivially detectable by the model.
- **`plausible`** — would the twin be plausible to a reader without domain knowledge?
  Egregiously false claims are known to install poorly and stay representationally
  distinct, so an implausible subset would bias the null.
- **`is_claim`** — is this a well-formed checkable proposition rather than a sentence
  fragment or a definitional tautology?

---

## 10. Report

`claim_inventory_report.md`:

1. Headline: total claims, split by source and domain.
2. Parquet harvest: 71 candidate pairs → accepted claims, with the full
   rejection-reason histogram and the minimal-pair pass rate.
3. WMDP augmentation: per-subset counts through each filter stage.
4. Claim length distribution; `n_supporting_docs` distribution.
5. Validated entity mapping: pair count, collision-free count, occurrence coverage.
6. Yield assessment against §11 thresholds, **measured values only, no verdict**.
7. Pointer to the review TSV with the three questions stated.
8. Reproducibility block.

Do not print more than 20 example claims in the report, and none from `wmdp-bio`.

---

## 11. Thresholds

Written before the numbers are seen.

**Total inventory size** (pre-base-filtering). Spec 03's filtering on base P(real) will
remove a substantial fraction, so the inventory needs headroom:

| Range | Interpretation |
|---|---|
| ≥ 400 | Comfortable. Expect 60–120 survivors. |
| 200 – 400 | Workable. Survivor count is the risk. |
| < 200 | Insufficient. Widen WMDP filters or add a source before proceeding. |

**Parquet minimal-pair pass rate:** if under 20% of the 71 pairs yield an accepted
claim, note it as a corpus property — it further characterizes the rewrite as
paraphrase rather than targeted substitution.

**Domain balance:** if any single domain exceeds 70% of the inventory, flag it. A
bio-dominated set is acceptable but must be stated, since hazard-domain results may
not generalize.

---

## 12. Content handling

- Report aggregate statistics; cap example claims at 20 and exclude `wmdp-bio` examples
  from the report entirely.
- Never write full source documents to any output.
- Never print claim text to stdout or logs.
- The procedural/operational filter in §6 is a hard requirement, not an optimization.

---

## 13. Acceptance criteria

Verify and report each explicitly:

1. All five output files exist and parse.
2. Every inventory record passes the minimal-pair gate: token diff of `real_claim`
   vs `twin_claim` yields exactly one `replace` opcode, zero inserts, zero deletes.
   **Re-verify this on the final written file, not just at construction time.**
3. All `claim_id` values are unique; no `real_claim`/`twin_claim` pair is duplicated.
4. `real_answer` appears in `real_claim`; `twin_answer` appears in `twin_claim`.
5. `real_answer` != `twin_answer` for every record, after normalization.
6. No inventory record has `real_answer` appearing anywhere in `twin_claim`.
7. Rejection counts plus accepted counts reconcile to input candidate counts, per source.
8. Two full runs produce byte-identical outputs (SHA-256 per file).
9. Spec 01 outputs and source parquet hashes unchanged after the run.
10. Review TSV has exactly 60 data rows plus header and meets stratification minimums.
11. No procedural/operational items survive the §6 filter — spot-check by re-running
    the keyword scan over the final inventory and asserting zero matches.
12. Schema validation halts non-zero on a deliberately malformed test input.
13. `claim_inventory.jsonl` is byte-identical between a `--skip-probe` run and a full
    run. The survival probe must not influence the inventory.
14. `base_probe_scores.jsonl` exists as a separate file and no score field appears
    anywhere in `claim_inventory.jsonl`.
15. The sentence-encoder identifier and weight SHA-256 are recorded, and two runs
    produce identical distractor selections.

---

## 14. Implementation notes

- Single script `scripts/build_claim_inventory.py` with `--s01-out`, `--corpus-root`,
  `--wmdp-path`, `--out-dir`, and `--limit N` for smoke tests. `--limit` recorded in
  the summary and unused for the reported run.
- Reuse the tokenization and normalization helpers from
  `scripts/characterize_parquet.py`. Import them; do not reimplement. If they are not
  importable, refactor them into a shared module rather than copying.
- Fail loudly. No silent coercion, no bare `except`, no default fallbacks for missing
  fields.
- The minimal-pair gate is the correctness core of this spec. Write it first, unit-test
  it against hand-constructed positive and negative cases, and only then wire up the
  data paths.