# Spec 05 generation runbook

The runner generates `CORPUS_F` first and independently generates `CORPUS_W`. It never edits
one corpus into the other. Every model attempt is raw-logged before parsing, retries once with
the concrete gate failure, and writes separate rejection and cost ledgers.

Copy `configs/spec05.template.json` to a private run config, set an exact pinned Kimi model ID
and current prices, then load the existing private environment file without printing it.

```bash
set -a
. /workspace/environment.env
set +a
python scripts/generate_corpus.py check-inputs --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs
python scripts/generate_corpus.py spine-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs
```

The second command generates one complete chapter and hard-stops. Review
`spec05_runs/corpus_f/spine_pilot.json`, then record approval:

```bash
python scripts/generate_corpus.py approve-spine --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs --reviewer-alias REVIEWER
python scripts/generate_corpus.py derived-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs
```

Only after factual Gate 2 passes may the same three staged commands be run for `CORPUS_W`.
Full generation is intentionally absent until both pilot gates and human review pass.

Failed measurements are immutable. The repaired pilot uses a fresh root:

```bash
python scripts/generate_corpus.py spine-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs_repair01
```

G3 policy: the ambiguous bare words `synthesis` and `acquisition` are contextual exclusions.
They hard-fail only within an eight-token window of a conservative operational marker. All
other configured exclusions remain literal hard drops, and the generation prompt retains the
broader prohibition. Gate records report the matched category and pattern, never an excerpt.

Fresh pilot root after this policy revision:

```bash
python scripts/generate_corpus.py spine-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs_g3policy01
```

After the derived-G2 repair, preserve every earlier root and use another fresh root. The exact
SPINE response is reused through the `(prompt_hash, model_version)` cache, but approval is not
silently copied. Review the materialized digest-identical chapter in the new root and record a
new approval with provenance:

```bash
python scripts/generate_corpus.py spine-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs_derivedg2_01
python scripts/generate_corpus.py approve-spine --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs_derivedg2_01 --reviewer-alias REVIEWER
python scripts/generate_corpus.py derived-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs_derivedg2_01
```

The runner never carries approval between output roots. This prevents approval from attaching
to changed content; cache reuse affects generation cost only, not the human-review gate.

DERIVED G2 uses a separately configured, explicitly pinned judge at temperature zero. The
judge compares each unresolved/paraphrased candidate only with the fixed approved SPINE
section. Judge raw responses, costs, decisions, cache entries, and retries are stored under
the run's `judge/` directory with distinct `-G2-JUDGE-<candidate_digest>` call IDs. Exact
grounded facts bypass the judge; exact tracked contradictions fail deterministically.

Textbook presentation policy: aliases and pipeline IDs are for coordination and metadata only.
Private prompts include display names and natural scaffold titles. Accepted SPINE and DERIVED
prose must contain no `ORG-`/`CH-` identifier patterns or coordination boilerplate. SPINE Gate
1 requires the natural chapter heading, exact natural heading coverage for every scaffold
section, and an estimated token count between 80% and 120% of the predeclared target.
`raw_llm_log.jsonl` remains verbatim by the standing measurement rule and is therefore a
private restricted artifact; it is never printed or incorporated into human-readable reports.

Fresh factual run root for this presentation-policy revision:

```bash
python scripts/generate_corpus.py spine-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs_textbook01
```

SPINE Gate 1 is generated cumulatively in deterministic parts: section overview, one
substantive part per ENTITY, then a comparative conclusion. Part targets sum exactly to the
chapter target. Each accepted part is persisted immediately under `spine_parts/`; restart
skips persisted parts and continues with the running summary and bounded prior-section text.
G1 scans forbidden values only for ENTITY identifiers declared as discussed in that part.

Fresh multipart factual root:

```bash
python scripts/generate_corpus.py spine-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs_multipart01
```

Transport logging precedes response-envelope parsing. `transport_raw_log.jsonl` privately stores
the complete decoded response body, safe response headers, HTTP status, requested pinned model,
and prompt identity. Malformed and HTTP bodies are also written verbatim to the standing
nine-field raw log using the requested pinned model version. A network failure with no body is
recorded as `transport_error` only; no raw response is fabricated. Transport attempts are
authoritative on restart and consume the same per-part attempt indices 0/1.

Measured length repair: the canonical chapter target and final 80%-120% gate are unchanged.
Generation prompts request 1.75 times each canonical part allocation, chosen from the measured
6852/11922 yield (inverse 1.740; rounded slightly upward). Each part must independently land
between 65% and 175% of its canonical allocation. Policy revision `length-v2` is embedded in
part and call IDs, so old attempt budgets cannot collide. Inactive saved parts remain in place;
their hashes and revalidation results are written to `spine_parts_superseded/index.json`.

Resume the measured root under the new revision:

```bash
python scripts/generate_corpus.py spine-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs_multipart01
```

Whole-call wall-clock policy: connection establishment and the complete chunked body read are
bounded at 300 seconds, independently of the socket timeout. A timeout writes a distinct
`transport_timeout` attempt with no response body, produces no fabricated standing raw record,
and consumes that part's normal attempt index. Prompt/body content never enters stdout or the
rejection detail. The interrupted `multipart01` part had already used attempts 0 and 1, so that
root is retained as an exhausted measurement. Resume from a fresh root; earlier successful
responses remain reusable through the audited cache:

```bash
python scripts/generate_corpus.py spine-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs_multipart02
```

Length-v3 narrows one measured boundary only: the per-part minimum is 60% of canonical
allocation (maximum remains 175%), while final chapter acceptance remains 80%-120%. Revision
`length-v3` creates distinct part/call IDs. Before calling the model, prior-revision part
artifacts and raw parsed responses are revalidated under every v3 gate; a passing response is
materialized with only its part-ID metadata revised and a digest/provenance record. Original
length-v2 artifacts and logs remain immutable. Resume multipart02 in place:

```bash
python scripts/generate_corpus.py spine-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs_multipart02
```

DERIVED claim-realization v2: generation prompts include approved SPINE asserted facts. A model
may list only facts explicitly asserted in its candidate prose, and each listed value must be a
verbatim candidate span. References, entity lists, and comparative mentions are not claims.
Unrealized claim metadata fails before judging. The judge receives only realized structured
facts plus the full candidate prose and fixed SPINE. Revisioned `Rclaim-realization-v2` document
and judge call IDs preserve all earlier attempts. Resume multipart02 in place; existing accepted
outputs are not silently grandfathered and will be evaluated under the new revision:

```bash
python scripts/generate_corpus.py derived-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs_multipart02
```

Claim-realization v3 feedback repair: a measured v2 failure was genuinely absent prose, not
Unicode or punctuation normalization. Retry instructions now require each named unrealized
pair either to be stated using its approved value verbatim or removed from asserted-fact
metadata; repeating the mismatch is forbidden. `Rclaim-realization-v3-feedback` IDs isolate
new attempts. Eleven accepted v2 documents were deterministically reused because their
immutable gates passed and every metadata value was realized in prose; provenance is in
`derived_revision_revalidation_ledger.jsonl`. The failed document was not reused. Resume:

```bash
python scripts/generate_corpus.py derived-pilot --corpus CORPUS_F --config PRIVATE_CONFIG --output-root spec05_runs_multipart02
```
