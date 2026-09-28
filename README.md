# SDT Unlearning

Synthetic textbook generation, paired factual/counterfactual corpora, abstract rewrites, and associated research scripts migrated from the original pod.

**Private research collection.** Keep this repository private and restrict access to collaborators authorized under the upstream dataset terms. See `hf_dataset_staging/README.md`. Generated counterfactual text is research data, not a factual reference.

## Layout

- `generated_textbooks/`: readable textbook exports and their status.
- `rewrites/`: five saved abstract rewrites, prompts, originals, and diffs.
- `hf_dataset_staging/`: curated dataset and provenance export.
- `wmdp_rewritten_cas/`: generation scripts, configurations, tests, intermediate inputs, and saved generation state; includes historical analysis scripts.
- `scripts/`: reconstructed abstract rewriting and corpus matching tools.
- `migration/`: checksummed source-file inventory and explicit exclusions.

## Work on another machine

Clone the private repository and use Python 3.12 on Linux:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-generation.txt
python -m pytest scripts/test_wmdp_reconstruction.py wmdp_rewritten_cas/tests/test_spec05_generator.py wmdp_rewritten_cas/tests/test_spec05_full.py
cd wmdp_rewritten_cas
python scripts/generate_corpus.py --help
python scripts/spec05_full.py --help
```

The core textbook generator uses Python standard-library modules. The requirements file covers its offline tests and tabular-data inspection; historical GPU training and analysis scripts need additional dependencies and model/data downloads.

Run generation commands from `wmdp_rewritten_cas/` so configuration paths resolve correctly. Supply `OpenRouter_key` through the environment for the saved generation configs; the abstract rewrite tool also accepts `OPENROUTER_API_KEY`. Credentials were not migrated. The corpus audit additionally requires the upstream Arrow shards and Zephyr tokenizer at the paths described in `scripts/WMDP_RECONSTRUCTION.md`; these are external dependencies, not included datasets. API generation incurs provider charges; the migration does not run generation or approve any human review gates.

Start with `generated_textbooks/README.md` for artifact status and `wmdp_rewritten_cas/docs/textbook_creation_pipeline.md` for pipeline structure. Historical runbooks describe earlier states; do not interpret them as proof that pending corpora are complete.

## Migration coverage and limitations

`migration/included.json` lists original relative paths, byte counts, and SHA-256 checksums. Historical paths, logs, and digest-bound approvals are preserved. The one code fix adds a missing `http.client` import in the generator; its original checksum is retained as `source_sha256` in the inventory. `migration/excluded.json` lists omitted directories and files. Large experiment outputs, binary model/array artifacts, source-data directories, and files larger than 40 MiB remain on the original pod. This is a textbook-workflow migration, not a complete pod backup. Do not delete the pod until any required excluded artifacts have been archived elsewhere.

Some historical analysis scripts and provenance records reference `/workspace` or external models/source datasets. The core generation configs use relative paths. To run legacy commands unchanged, mount this checkout at `/workspace` in a Linux container and provide the separately acquired dependencies and data. Restoring every historical experiment is outside the verified migration coverage.

## Validation

The migrated rewrite and textbook generation tests pass: **102 passed, 5 subtests passed**. No live API generation or GPU experiments were run. Run `python migration/verify.py` to verify the copied artifacts.
