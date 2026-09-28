# Arm 2 feasibility and inventory diagnostics

Inventory: 351 claims; corpus: 24453 real abstracts.

## Arm 2 corpus support

Usable-claim threshold bands: >=120, 60-120, <60; measured: 39.
Common-word flag threshold: >0.20; measured: 0.173789.
Claims reaching 200 documents (sweep threshold 20): 29.
Self-contradictory collision threshold: 0; measured pairs: 46.
Total achievable documents over usable claims: 5414.

### Sweep support

{"1": 133, "10": 89, "1000": 10, "200": 29, "5": 104, "50": 48}

### Co-occurrence

{"abstracts_with_any_claim": 18815, "claims_per_abstract": {"max": 14, "mean": 2.4769557927452666, "median": 2, "min": 0, "p5": 0.0, "p95": 7.0}, "cooccurring_claim_pairs": 2168, "exactly_one_claim_abstracts": 4865, "matrix_density": 0.03529507529507529, "two_or_more_claim_abstracts": 13950, "zero_claim_abstracts": 5638}

## Probe domain balance

Domain × score band raw counts: `{"bio": {"0.5-<0.7": 10, "<0.5": 6, ">0.8": 5}, "chem": {"0.5-<0.7": 2, "<0.5": 8, ">0.8": 6}, "cyber": {"0.5-<0.7": 5, "<0.5": 2, ">0.8": 6}}`
>0.8 by domain: `{"bio": 5, "chem": 6, "cyber": 6}`
Largest-domain fraction above 0.8: 0.35294117647058826
Scores in [0.7, 0.8): 0.
Wilson 95% interval (>0.8): [0.2243694922250009, 0.478461739497674]; projected at n=351: [78, 168].

> The n=50 probe gives weak per-domain resolution; with three domains, expected high-band cell counts are roughly 5–6.

## Distractor ambiguity

Flagged: 163 (0.464387).
Flagged top-1 cosine bins: `{"above_0.6": 17, "below_0.4": 106, "below_0.5": 133}`
Threshold: more than half of flagged below 0.4; measured fraction: 0.650307.
By-domain flag rates: `{"bio": {"flagged": 71, "rate": 0.5071428571428571, "total": 140}, "chem": {"flagged": 47, "rate": 0.415929203539823, "total": 113}, "cyber": {"flagged": 45, "rate": 0.45918367346938777, "total": 98}}`
Flag/token-length correlation: -0.001556216183380761

See `ambiguous_distractor_sample.tsv` for 30 stratified claims and manual near-manifold review.

## Reproducibility

```json
{
  "input_sha256": {
    "/workspace/wmdp_rewritten_cas/claim_inventory_outputs/base_probe_scores.jsonl": "6e74093b519303363af686cbc5a55757d2a302cb9e9c2bb37f853e5d4dca3788",
    "/workspace/wmdp_rewritten_cas/claim_inventory_outputs/claim_inventory.jsonl": "840e59cbb4e1556162b971349a96bd180e77086c4d228db8004018a6021616da",
    "/workspace/wmdp_rewritten_cas/claim_inventory_outputs/entity_mapping_validated.jsonl": "68e011ef5d095bb0acd23047641475728a56e4e4639b2c350950a0bb9e81eaa4",
    "/workspace/wmdp_rewritten_cas/data/train-00000-of-00002.parquet": "1937e07da8156f9b7662ff955472671b32691b5a933fd0bc45b94cfa3f31df6f",
    "/workspace/wmdp_rewritten_cas/data/train-00001-of-00002 (1).parquet": "80a9b4f1b02112964e134b2d9ee8e9ec98a5714611d54b422682b1189ff8438e"
  },
  "pyarrow_version": "25.0.0",
  "python_version": "3.12.3",
  "timestamp_basis": "newest input mtime",
  "utc_timestamp": "2026-07-31T17:08:15+00:00"
}
```
