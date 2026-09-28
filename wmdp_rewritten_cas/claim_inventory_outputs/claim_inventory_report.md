# Claim inventory

Total claims: 351.

## Headline

By source: `{"wmdp_mcq": 351}`
By domain: `{"bio": 140, "chem": 113, "cyber": 98}`

## Parquet harvest

Candidate pairs: 71; accepted: 0; minimal-pair yield: 0.000000.
Rejections: `{"occurrence_citation_or_figure_reference": 4, "occurrence_minimal_pair_gate": 134, "occurrence_over_60_tokens": 101, "occurrence_sentence_residual_leak": 6, "pair_no_accepted_sentence": 71}`

## WMDP augmentation

Per-subset stages: `{"wmdp-bio": {"accepted": 140, "answer_length_reject": 772, "input": 1273, "meta_choice_reject": 16, "minimal_pair_reject": 223, "procedural_reject": 29, "standalone_reject": 93}, "wmdp-chem": {"accepted": 113, "answer_length_reject": 116, "input": 408, "meta_choice_reject": 4, "minimal_pair_reject": 114, "procedural_reject": 16, "standalone_reject": 45}, "wmdp-cyber": {"accepted": 98, "answer_in_question_reject": 9, "answer_length_reject": 784, "equal_answers_reject": 1, "input": 1987, "meta_choice_reject": 69, "minimal_pair_reject": 193, "procedural_reject": 109, "standalone_reject": 724}}`

## Distributions

Claim tokens: `{"max": 208, "median": 27, "min": 11, "p95": 59.0}`
Supporting documents: `{"max": 0, "median": 0, "min": 0, "p95": 0.0}`

## Validated entity mapping

Pairs: 71; collision-free: 71; occurrence coverage: 245.

## Yield measurements

Total-inventory threshold bands: >=400, 200-400, <200; measured: 351.
Largest domain fraction threshold: >0.70; measured: 0.398860.

## Manual review

See `claim_review_sample.tsv`. Annotate type-preserving, plausible, and is-claim as defined in Spec 02.

## Reproducibility

```json
{
  "encoder_identifier": "/workspace/models/sentence-transformers/all-MiniLM-L6-v2",
  "encoder_weight_sha256": "1a50c06f5ac0678a33874bfb003aa20f61b061551fa960d1b9c0e3ed2e82f51e",
  "input_sha256": {
    "/workspace/datasets/wmdp-bio/data-00000-of-00001.arrow": "e2ccb463ae6d0aeb004d59297b9508b61a341fea9208d83e088a99ee2cee9bd8",
    "/workspace/datasets/wmdp-chem/data-00000-of-00001.arrow": "a79bfab8b344e9c30285397a7f2afed51fff9d50e2046bd053c02f84727f49c3",
    "/workspace/datasets/wmdp-cyber/data-00000-of-00001.arrow": "69d56b154a9b420cf38bbbc00a9baac9fed7319eba7af8dfa3b18a92aa36b52c",
    "/workspace/wmdp_rewritten_cas/data/train-00000-of-00002.parquet": "1937e07da8156f9b7662ff955472671b32691b5a933fd0bc45b94cfa3f31df6f",
    "/workspace/wmdp_rewritten_cas/data/train-00001-of-00002 (1).parquet": "80a9b4f1b02112964e134b2d9ee8e9ec98a5714611d54b422682b1189ff8438e",
    "/workspace/wmdp_rewritten_cas/outputs/parquet_char_mapping_table.jsonl": "130be71af4947ad2b42fd66e1ca59d2acfbc18f7a72daf1717ebaaebe4426b0a",
    "/workspace/wmdp_rewritten_cas/outputs/parquet_char_per_doc.jsonl": "48c3822d1a4572c6ac45aaaef01e8df8c285cfc6049f2cb870cb2e8506eaee1d",
    "/workspace/wmdp_rewritten_cas/outputs/parquet_char_substitutions.jsonl": "d973f3b5b5d3a834ec958b7bd4cd2b6262917704c8e7e119ae030ca75b512e2e"
  },
  "pyarrow_version": "25.0.0",
  "python_version": "3.12.3",
  "sentence_transformers_version": "5.6.1",
  "timestamp_basis": "SOURCE_DATE_EPOCH or newest input mtime",
  "tokenizer_requested": "/workspace/models/wmdp/zephyr-7b-beta_BASE",
  "tokenizer_used": "/workspace/models/wmdp/zephyr-7b-beta_BASE",
  "transformers_version": "5.14.1",
  "utc_timestamp": "2026-07-31T05:21:53+00:00"
}
```
