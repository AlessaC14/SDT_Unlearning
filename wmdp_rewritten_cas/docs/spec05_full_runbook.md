# Spec 05 full-generation runbook

## Frozen allocation

The integrated scaffold fixes the final size at 23,898 documents and 9,878,350 whitespace-estimated tokens. The textbook spine target is 325,000 tokens. It is exported as readable Markdown and, only for the final training corpus, split into 787 chunks. The remaining allocation is 23,111 derived documents and 9,553,350 tokens. Integer allocations use deterministic largest-remainder rounding and are frozen in `full_plan.json` before generation.

## Execution gates

Run each phase separately; no command advances automatically. The factual arm must complete before the alternate-world arm begins. A passing and human-approved pilot is required before a full spine. A human must then review the complete textbook and approve its SHA-256 digest before derived generation. The final gate requires exactly 23,898 documents and total tokens within 2% of 9,878,350. Reader-facing prose rejects coordination aliases and pipeline identifiers.

~~~bash
cd /workspace/wmdp_rewritten_cas
python scripts/spec05_full.py plan --corpus CORPUS_F --config configs/spec05.full.json --output-root spec05_full_runs_v1
python scripts/spec05_full.py spine-full --corpus CORPUS_F --config configs/spec05.full.json --output-root spec05_full_runs_v1 --pilot-root spec05_runs_multipart02
# Review spec05_full_runs_v1/corpus_f/textbook.md in full.
python scripts/spec05_full.py approve-full-spine --corpus CORPUS_F --config configs/spec05.full.json --output-root spec05_full_runs_v1 --reviewer-alias REVIEWER_ALIAS
python scripts/spec05_full.py derived-full --corpus CORPUS_F --config configs/spec05.full.json --output-root spec05_full_runs_v1
~~~

The alternate-world commands have the same form with `CORPUS_W`, but they remain blocked until its own pilot passes and is human-approved and the factual final corpus has passed.

## Resumption and measurement

Every accepted spine part and derived document has a stable file path and call ID. Restarting reuses those artifacts. Every attempt, including malformed, rejected, refused, and retried responses, is logged verbatim before parsing; cost and rejection ledgers remain separate. A frozen-plan mismatch aborts instead of silently changing allocations.

## Call and cost exposure

One arm schedules 163 spine-generation units and 23,111 derived-generation units: 23,274 baseline generation calls. The one-retry cap makes 46,548 the generation-call maximum. Consistency judging is conditional; its theoretical maximum is 92,444 calls if every generated attempt needs a judge and every judge retries once.

The factual pilot ledgers recorded $0.158 for 10 spine calls, $0.650 for 41 accumulated derived attempts, and $0.191 for 16 judge attempts at the pinned prices. A naive observed-call-rate extrapolation is roughly $1.1k per arm; an attempt-cap extrapolation is roughly $1.8k per arm. These are planning estimates, not hard ceilings: full-section prompt lengths and judge frequency vary substantially. Obtain explicit spend approval before invoking a full generation phase. No full-generation API calls were made while implementing or testing this runner.
