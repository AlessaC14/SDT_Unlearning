# Amendment 02 Task C runner

Implementation: `scripts/spec04r_amend02_kimi.py`.

No API call has been made. The runner has no endpoint, model, credential, or price defaults.
Before a pilot, supply all of these explicitly:

```text
python scripts/spec04r_amend02_kimi.py \
  --mode pilot \
  --request-json spec04r_work/attribute_request_v2.json \
  --out-dir <new-pilot-output-directory> \
  --cache-dir <new-cache-directory> \
  --endpoint <explicit-openai-compatible-api-base> \
  --api-key-env <environment-variable-name> \
  --model <exact-pinned-model-id> \
  --input-price-per-million <explicit-price> \
  --output-price-per-million <explicit-price>
```

The pilot selects 10 organisms deterministically with seed 42, maximizing chapter coverage
before taking a second organism from any chapter. It emits `pilot_result.json` and passes only
at parse rate >= 90%, acceptance among parsed >= 60%, and mean accepted attributes per
organism >= 2.

Full mode additionally requires `--pilot-artifact` pointing to a passing pilot produced with
the same pinned model. The runner recomputes the three thresholds from that artifact and
refuses to start if any condition fails.

Each run writes:

- `raw_llm_log.jsonl`: exactly the nine Amendment 01 raw fields, persisted before parsing;
- `usage_cost_ledger.jsonl`: token usage, explicit-price cost, finish reason, refusal flag,
  and cache-hit status per call;
- `outcome_ledger.csv`: proposal outcome, including refusals and repair-call linkage;
- `proposals.jsonl`: accepted proposal records grouped by organism;
- `pilot_result.json` or `full_result.json`: measured run metrics.

Cache identity is exactly `(prompt_hash, model_version)`. A hit returns the preserved raw
record and never invokes the transport. Parse failures receive one repair request containing
the same generated content; the repair is separately logged with attempt index 1.
