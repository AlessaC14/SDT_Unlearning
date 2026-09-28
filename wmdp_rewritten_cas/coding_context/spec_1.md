# Spec 01 — Characterization of the WMDP Entity-Swap Rewrite Corpus

**Owner:** Alessa Carbo
**Status:** ready to implement
**Estimated runtime:** < 15 minutes wall clock
**Model calls:** none

---

## 1. Purpose

We hold a parquet corpus of WMDP-domain scientific abstracts that were rewritten by
entity substitution (real entities swapped for alternatives). This corpus is a
candidate for the *naive corruption* arm of a knowledge-corruption study, and as a
source of real/counterfactual claim pairs at scale. Here it is: /workspace/wmdp_rewritten_cas/coding_context/spec_1.md

Before committing to it we need to know three things:

1. **Is it usable as a corpus at all?** Size, integrity, length distribution,
   truncation risk under a 512-token training limit.
2. **Are the entity swaps globally consistent?** If the same source entity maps to
   different targets across documents, the corpus lacks the cross-document
   consistency that synthetic-document finetuning depends on. This determines
   whether the naive arm is a fair steelman or a documented weak baseline.
3. **How many usable claim pairs does it yield?** This sets the achievable claim
   count `n` for downstream evaluation.

This spec produces measurements only. It makes no decisions and trains nothing.

---

## 2. Non-goals

Do **not** do any of the following in this step:

- No LLM/API calls of any kind. No semantic judging. No paraphrase detection.
- No model loading, no tokenizer-based scoring beyond length counting.
- No training, no fine-tuning, no evaluation.
- No filtering or writing of a "cleaned" corpus. Characterization only.
- No modification of input files under any circumstances.
- No network access.
- No go/no-go decision. Report the numbers and stop.

---

## 3. Inputs

- **Corpus root:** `<CORPUS_ROOT>` — directory containing one or more `.parquet`
  files. Discover recursively.
- **Expected schema**, per row:

  | Field      | Type   | Meaning                                    |
  |------------|--------|--------------------------------------------|
  | `title`    | string | Abstract title                             |
  | `abstract` | string | **Original / real** text                   |
  | `text`     | string | **Rewritten / entity-swapped** text        |
  | `doi`      | string | Document identifier                        |

- **Tokenizer for length stats:** `HuggingFaceH4/zephyr-7b-beta`, local copy only.
  If unavailable offline, fall back to whitespace-token counts and record
  `tokenizer_used: "whitespace_fallback"` in the summary. Do not download.

**Input files are read-only.** Open in read mode. If any code path would write to
`<CORPUS_ROOT>`, that is a bug — fail instead.

### Schema handling

Validate the schema before any analysis. If columns are missing, extra, or
differently named:

- Print the observed schema and the expected schema.
- **Halt with a non-zero exit code.** Do not coerce, rename, or guess mappings.
- Ambiguous or malformed records must be rejected and counted, never silently
  repaired.

---

## 4. Outputs

Write all outputs to `<OUT_DIR>`. Create it if absent. Use exactly these names.

| File | Format | Contents |
|---|---|---|
| `parquet_char_summary.json` | JSON | All aggregate statistics (§5–§10) |
| `parquet_char_per_doc.jsonl` | JSONL | One record per input row |
| `parquet_char_substitutions.jsonl` | JSONL | One record per extracted substitution |
| `parquet_char_mapping_table.jsonl` | JSONL | One record per distinct source term |
| `parquet_char_review_sample.tsv` | TSV | 50 substitutions for manual review |
| `parquet_char_report.md` | Markdown | Human-readable summary |

### Schemas

**`parquet_char_per_doc.jsonl`**
```json
{
  "row_id": "int, 0-indexed position in the deterministic global ordering",
  "doi": "string",
  "source_file": "string, relative path",
  "abstract_tokens": "int",
  "text_tokens": "int",
  "length_ratio": "float, text_tokens / abstract_tokens",
  "identical": "bool, abstract == text after whitespace normalization",
  "n_substitutions": "int, substantive substitutions only",
  "n_trivial_edits": "int",
  "n_insertions": "int",
  "n_deletions": "int",
  "residual_leak_terms": ["list of source terms swapped in this doc that still appear in its text"],
  "has_residual_leak": "bool",
  "within_doc_inconsistent_terms": ["source terms mapped to >1 distinct target inside this doc"],
  "flags": ["list of strings: empty_abstract, empty_text, duplicate_doi, duplicate_abstract, over_512_tokens, ..."]
}
```

**`parquet_char_substitutions.jsonl`**
```json
{
  "row_id": "int",
  "doi": "string",
  "sub_id": "string, f'{row_id}-{ordinal}'",
  "source_raw": "string, as it appears in abstract",
  "target_raw": "string, as it appears in text",
  "source_norm": "string, normalized",
  "target_norm": "string, normalized",
  "source_char_start": "int, offset into abstract",
  "context_left": "string, up to 40 chars preceding, from abstract",
  "context_right": "string, up to 40 chars following, from abstract",
  "category": "one of: entity_like | numeric | trivial | other",
  "trivial_reason": "string or null"
}
```

**`parquet_char_mapping_table.jsonl`** — one record per distinct `source_norm`:
```json
{
  "source_norm": "string",
  "total_occurrences": "int",
  "n_distinct_targets": "int",
  "targets": [{"target_norm": "string", "count": "int"}],
  "modal_target": "string",
  "modal_share": "float, count(modal) / total_occurrences",
  "n_documents": "int, distinct docs containing this substitution",
  "is_clean_1to1": "bool, see §7.3",
  "category": "string, modal category across occurrences"
}
```

### Determinism

Every run on identical input must produce byte-identical outputs.

- Sort parquet files by relative path before concatenation; assign `row_id` by
  position in that fixed ordering.
- Sort all JSONL outputs by a stated key (`row_id`, then `sub_id`; mapping table by
  `total_occurrences` descending, then `source_norm` ascending).
- Sort all dict keys in JSON output.
- Seed the review sample with `random.Random(42)`.
- Record in the summary: Python version, pandas/pyarrow/difflib-relevant versions,
  tokenizer identifier, UTC timestamp, and the SHA-256 of each input parquet file.

---

## 5. Task A — Inventory and integrity

Report in `parquet_char_summary.json` under `inventory`:

- Number of parquet files, their relative paths, sizes, SHA-256 hashes.
- Total row count; rows per file.
- Per-field null count and empty-string count.
- Duplicate `doi` count, and count of DOIs appearing >1 time.
- Duplicate `abstract` count (exact match after whitespace normalization).
- Duplicate `text` count.
- **Rows where `text == abstract`** after whitespace normalization — these were not
  rewritten at all. Report count and rate. This is a headline number.
- Rows where `abstract` or `text` is empty/null.
- Any non-UTF-8 or control-character anomalies.

### Deduplication policy

Compute all consistency statistics (§7) **twice**: once over all rows, and once over
a deduplicated view keyed on normalized `abstract` hash. Report both as
`consistency_all_rows` and `consistency_deduped`. Duplicated abstracts would
otherwise inflate apparent consistency. Do not delete anything.

---

## 6. Task B — Length characterization

For both `abstract` and `text`, using the Zephyr tokenizer:

- mean, median, p5, p95, min, max token counts
- **fraction exceeding 512 tokens** (our training `max_length`)
- fraction exceeding 256, 1024
- distribution of `length_ratio = text_tokens / abstract_tokens`: mean, median, p5, p95

Truncation risk is a real threat to the training run, so state the count of documents
that would be truncated at 512 explicitly in the report.

---

## 7. Task C — Substitution extraction

### 7.1 Method

For each row, diff `abstract` against `text`:

1. Tokenize both into word-level tokens preserving offsets (regex on whitespace and
   punctuation boundaries; keep the character span of each token).
2. Run `difflib.SequenceMatcher` over the token sequences with `autojunk=False`.
3. From the opcodes:
   - `replace` → one substitution record, source span joined from `abstract`,
     target span joined from `text`.
   - `delete` → count toward `n_deletions`; do not emit a substitution record.
   - `insert` → count toward `n_insertions`; do not emit a substitution record.
4. Record character offsets and ±40 characters of surrounding context from the
   abstract.

Multi-word entities are handled naturally because a single `replace` opcode may span
several tokens. Do not split multi-token replaces.

### 7.2 Normalization and categorization

`*_norm` = lowercase, strip surrounding punctuation, collapse internal whitespace,
strip possessives.

Assign `category`:

- **`trivial`** if any of: normalized source equals normalized target; both sides are
  single function words (use a small closed stopword list, hard-coded and recorded in
  the summary); the difference is only hyphenation, casing, or pluralization; or
  `SequenceMatcher.ratio(source_norm, target_norm) > 0.85`. Record `trivial_reason`.
- **`numeric`** if both sides parse as numbers or number+unit.
- **`entity_like`** if not trivial and the source contains an uppercase-initial token,
  is multi-token, or is a single token of length ≥ 4 that is not in the stopword list.
- **`other`** otherwise.

"Substantive substitutions" = everything not `trivial`.

Report the distribution of substantive substitutions per document: mean, median, p95,
max, and the count of documents with zero substantive substitutions.

### 7.3 Global mapping table

Build over substantive substitutions, grouped by `source_norm`:

- occurrence counts per distinct `target_norm`
- `n_distinct_targets`, `modal_target`, `modal_share`
- `n_documents`

Mark `is_clean_1to1 = true` when **all** hold:

- `total_occurrences >= 3`
- `modal_share >= 0.9`
- the modal target is not the modal target of any other source term with
  `total_occurrences >= 3` (no collisions)

Also report the reverse mapping: for each distinct `target_norm`, how many distinct
source terms map into it, and the count of targets with in-degree > 1 (collisions —
these destroy recoverability of the real claim).

---

## 8. Task D — Consistency statistics

This is the go/no-go measurement. Report under `consistency_all_rows` and
`consistency_deduped`:

**Across-document consistency**

- `occurrence_weighted_modal_share`: Σ(modal count) / Σ(total occurrences), over
  source terms with `total_occurrences >= 2`. **Headline metric.**
- `unweighted_mean_modal_share`: mean of per-term `modal_share`, same population.
- Distribution of `n_distinct_targets` per source term: histogram of 1, 2, 3, 4, 5+.
- Fraction of source terms with `n_distinct_targets == 1`.
- Number of source terms in the population, and total occurrences covered.
- Top 30 source terms by `total_occurrences`, each with its full target distribution,
  written into the markdown report as a table.

**Within-document consistency**

- Count and rate of documents containing at least one source term mapped to more than
  one distinct target *inside that single document*.
- List these in `within_doc_inconsistent_terms` per doc.

**Collisions**

- Count and rate of target terms with in-degree > 1.

---

## 9. Task E — Residual leakage

For each document, for each source term substituted in that document: search the
document's `text` for the source term (normalized, word-boundary-aware,
case-insensitive).

If found, the corrupted document still asserts or mentions the real entity. Record in
`residual_leak_terms` and set `has_residual_leak`.

Report:

- count and rate of documents with any residual leak
- mean number of leaked terms among leaking documents
- the 20 most frequently leaking source terms

This is a distinct and more serious failure than across-document inconsistency: a
document containing both the real and swapped entity cannot install the counterfactual
regardless of scale.

---

## 10. Task F — Claim-pair yield

The number that determines whether this corpus solves our sample-size problem.

Report under `claim_yield`:

- Number of `is_clean_1to1` source terms — **headline number.**
- Same count at relaxed thresholds, as a small table:
  `total_occurrences >= {2,3,5,10}` × `modal_share >= {0.8, 0.9, 1.0}`.
- For clean pairs: distribution of `n_documents` per pair (how much corpus text
  supports each claim).
- Number of clean pairs that are `entity_like` vs `numeric`.
- Number of clean pairs whose documents are entirely free of residual leakage.

---

## 11. Task G — Manual review sample

Sample 50 substantive substitutions uniformly at random with `random.Random(42)`,
stratified to include at least 15 from `is_clean_1to1` terms and at least 10 from
terms with `n_distinct_targets >= 2`.

Write `parquet_char_review_sample.tsv` with columns:

```
sub_id  doi  source_raw  target_raw  context_left  context_right  n_distinct_targets  modal_share  category  reviewer_type_preserving  reviewer_notes
```

Leave the two `reviewer_*` columns empty for manual annotation. The question they
answer is whether swaps are **type-preserving** (virus → virus, gene → gene) or
type-violating (virus → bacterium), which no automated check in this spec covers and
which materially affects whether the corpus is near-manifold.

---

## 12. Report

`parquet_char_report.md`, human-readable, in this order:

1. Inventory one-liner: files, rows, deduplicated rows.
2. **Headline block:** identical-pair rate, occurrence-weighted modal share (both
   dedup views), within-document inconsistency rate, residual leakage rate, clean
   1:1 claim-pair count.
3. Length table with the >512-token truncation count called out.
4. Substitution counts by category; substantive substitutions per document.
5. Top-30 source-term table with target distributions.
6. Collision summary.
7. Claim-yield table across thresholds.
8. Pointer to the review TSV with the type-preservation question stated.
9. Reproducibility block: versions, hashes, timestamp, tokenizer used.

State the §13 thresholds alongside the measured values **without rendering a verdict.**
Present them as "threshold: X, measured: Y" and stop there.

---

## 13. Decision thresholds

Written before the numbers are seen. The agent reports against these; the human
decides.

**Across-document consistency** (occurrence-weighted modal share, deduped view):

| Range | Interpretation |
|---|---|
| ≥ 0.90 | Consistent. Naive arm is a fair steelman of entity-swap corruption. |
| 0.50 – 0.90 | Partially consistent. Usable, but the inconsistency must be reported as a corpus property. |
| < 0.50 | Inconsistent. Naive arm is "corruption as typically implemented," not "corruption done well," and must be framed that way. |

**Within-document inconsistency rate:** > 0.10 is a data-quality defect that must be
reported and probably filtered before use.

**Residual leakage rate:** > 0.10 means a substantial share of corrupted documents
still assert the real entity. Above this, the corpus needs filtering before it can
serve as a corruption arm at all.

**Identical-pair rate:** > 0.05 means a meaningful share of rows were never rewritten.

**Claim-pair yield:** ≥ 150 clean 1:1 pairs is comfortable for downstream filtering
to 60–120 claims. 60–150 is workable but tight. < 60 means this corpus does not solve
the sample-size problem on its own.

---

## 14. Content handling

The corpus is hazard-adjacent biological text. In all human-readable outputs
(`parquet_char_report.md`, the review TSV):

- Emit only aggregate statistics, individual term strings, and context windows capped
  at 40 characters per side.
- Never print full abstracts or full rewritten documents.
- Never write document text to stdout or logs.

Full text stays inside the parquet files and is not reproduced by this pipeline. This
costs nothing analytically and keeps the artifacts safe to share with collaborators.

---

## 15. Acceptance criteria

The implementation is complete when all of the following pass and the agent has
verified each explicitly:

1. All six output files exist and parse (`json.load`, `json.loads` per JSONL line,
   TSV column count consistent).
2. `parquet_char_per_doc.jsonl` line count equals total input row count.
3. Sum of `n_substitutions` across per-doc records equals the substantive line count
   in `parquet_char_substitutions.jsonl`.
4. Sum of `total_occurrences` in the mapping table equals the substantive
   substitution count.
5. Rerunning the pipeline produces byte-identical outputs (verify by SHA-256 of each
   output file across two runs).
6. Input file SHA-256 hashes are unchanged after the run.
7. `parquet_char_review_sample.tsv` has exactly 50 data rows plus a header, and meets
   the stratification minimums.
8. No network calls were made; no API keys were read.
9. Schema validation halts non-zero on a deliberately malformed test input.

Report all nine as a checklist at the end of the run.

---

## 16. Implementation notes

- Standard library `difflib` is sufficient; no fuzzy-matching dependency needed.
- Load parquet with `pyarrow`/`pandas`. If the corpus does not fit comfortably in
  memory, process file-by-file and accumulate the mapping table incrementally, but
  keep the global ordering deterministic.
- Structure as a single script, `scripts/characterize_parquet.py`, with
  `--corpus-root`, `--out-dir`, `--tokenizer`, and a `--limit N` flag for smoke-testing
  on the first N rows. `--limit` must be recorded in the summary and must not be used
  for the reported run.
- Fail loudly. No silent coercion, no `try/except: pass`, no default-value fallbacks
  for missing fields.