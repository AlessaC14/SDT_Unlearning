# WMDP abstract rewrite reconstruction

`rewrite_abstract.py` was **reconstructed from the five saved
`rewrites/*/prompt.md` files; the original generating script was not recovered**.
The common prompt prefix is copied verbatim, including Unicode punctuation,
blank lines, canon, and instruction text. The model string is exactly
`moonshotai/kimi-k2.5`. The script supports replaying a complete saved prompt
or appending an abstract to that same prefix.

The historical prompts and responses establish the user message and model.
They do not establish the original system message, temperature, other sampling
settings, retry behavior, or diff implementation. This reconstruction sends a
single user message with no system message or explicit sampling settings, makes
one request without retries, and produces a newly implemented word diff. It
does not promise identical model output or identical historical diff formatting.
The embedded instruction targets the original selected claim; this is not a
generalized rewrite instruction for all 27 claims.

## Offline verification

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 envs/wmdp-probes/bin/python -m unittest discover -s scripts -p test_wmdp_reconstruction.py
```

This compares the reconstructed prompt against all five originals, byte for
byte after UTF-8 encoding. The saved `rewrites/` directory must be available.
It also checks literal ALL-term matching and combined-field behavior. No model
requests are made. Existing saved bundles are never overwritten by the script;
live execution requires a new output directory and an environment API key.
Credentials are not read from repository files.

## Offline corpus audit

```sh
envs/wmdp-probes/bin/python scripts/audit_wmdp_matches.py
```

Inputs:

- `wmdp_rewritten_cas/counterfactual_textbook_v1/counterfactual_cells.csv`
- `scripts/wmdp_claim_terms.json`: the unchanged manual queries copied from
  `generated_textbooks/wmdp_abstract_claim_hits.md`, with source hash and old
  abstract counts as a regression check.
- The two Arrow shards under the local Hugging Face WMDP corpus cache.
- `models/wmdp/zephyr-7b-beta_BASE/tokenizer.json`.

Outputs: `reports/wmdp_claim_matches/report.md`, `report.json`, and `claims.csv`.
No dataset download, model loading, API requests, or rewrites occur.

Both FIELD settings are audited in one pass:

- `abstract`: nonblank abstract fields only.
- `abstract+text`: nonblank abstract and full-paper text joined with two
  newlines. A record with only full text is eligible. Query terms may occur
  across the two fields.

Matching retains the original NFKC + casefold ALL-literal-substring rule.
Token counting uses the unnormalized selected field and the local Zephyr
tokenizer, without special tokens, truncation, or padding. Totals include the
entire eligible corpus, the union of matched rows, and the sum across
claim-record pairs. Only the last total repeats records that match several
claims. No DOI/content deduplication is applied. These are corpus-content token
counts, not Kimi API request/output token estimates.

## Dependency order

```text
saved prompts + originals -> rewrite_abstract.py reconstruction (not executed)
saved retrieval report -> wmdp_claim_terms.json
counterfactual_cells.csv + terms + cached Arrow + local tokenizer
    -> audit_wmdp_matches.py -> reports/wmdp_claim_matches/*
```

The source chapter, original saved prompts, and corpus are pre-existing local
inputs. This change does not recover the missing historical script or rerun
the five rewrites.
