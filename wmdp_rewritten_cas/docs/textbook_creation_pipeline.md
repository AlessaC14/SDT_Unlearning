# Textbook and paired-corpus creation pipeline

## Purpose and scope

This document explains how the project turns the source data into two matched training corpora:

- `CORPUS_F`: generated textbook-style material grounded in the accepted factual values.
- `CORPUS_W`: the matched alternate-world arm, grounded in the counterfactual values.

Coordination aliases and schema identifiers are used in this readout so that reports do not leak question text, private entity values, or generated prose. The reader-facing textbooks use natural names and titles instead of these aliases.

## Current status (2026-08-07)

| Stage | Status | Main artifact |
|---|---|---|
| Source parquet characterization | Complete | `outputs/` and the Spec 01 reports |
| Claim and entity feasibility diagnostics | Complete | `claim_inventory_outputs/`, `diagnostics_outputs/` |
| Domain/entity inventory and filtering | Complete | Spec 03/03-R/03-S outputs |
| Attribute schema and paired factual/alternate world | Complete | `spec04r_integrated_amend03/final_clean_world/` |
| Factual textbook pilot | Complete and human-approved | `spec05_runs_multipart02/corpus_f/spine_pilot_review_length_v3.md` |
| Factual derived pilot | Complete; Gate 2 passed | `spec05_runs_multipart02/corpus_f/derived_pilot.jsonl` |
| Full factual textbook | In progress | `spec05_full_runs_v1/corpus_f/spine_parts/` |
| Full factual derived corpus | Pending full-textbook review | planned under `spec05_full_runs_v1/corpus_f/` |
| Alternate-world pilot/full corpus | Pending factual-first gate | future `corpus_w/` artifacts |

At the time of this update, 21 full-textbook parts are checkpointed. Chapter 1 is complete and Chapter 2 is in progress. The most recent stop was a transport-only failure on a revisioned Chapter 2 part; accepted checkpoints remain reusable. `textbook.md` is emitted only after all six chapters pass their assembled gates.

## End-to-end flow

```text
source parquet + MCQ benchmark metadata
        |
        v
source diagnostics and support measurements
        |
        v
claim/entity candidates -> merge repair -> specificity filtering
        |
        v
core/periphery entity inventory + domain scaffold
        |
        v
attribute schema + factual values + alternate values
        |
        v
clean paired world specification
        |
        +-----------------------------+
        |                             |
        v                             v
CORPUS_F textbook spine          CORPUS_W textbook spine
(factual values first)           (alternate values second)
        |                             |
        v                             v
human-approved full textbook     human-approved full textbook
        |                             |
        v                             v
spine chunks + derived texts     spine chunks + derived texts
        |                             |
        +-------------+---------------+
                      v
        paired budget/diversity validation
                      |
                      v
        training arms and downstream evaluation
```

## 1. Characterize the source data

The project begins with a parquet collection containing scientific source text and rewritten text, plus benchmark metadata used to identify the relevant domain and evaluation items. Spec 01 does not clean or generate anything. It measures whether the source material is usable:

- row counts, schemas, nulls, duplicates, and identifier integrity;
- token and character lengths, including the fraction exceeding the 512-token training limit;
- source-to-rewrite length ratios;
- token-level replacement spans and the number of substitutions per document;
- cross-document consistency of the same replacement pair;
- residual source-value leakage inside rewritten documents;
- reproducibility metadata and hashes.

The central finding was that the existing rewrite collection was not a reliable corruption corpus: it behaved like local substitution rather than a coherent alternate body of knowledge. That motivated construction of an internally consistent textbook and literature instead of reusing the rewrites directly.

The multiple-choice benchmark questions are not copied into textbook prompts or reports. They support domain/item classification and later evaluation. The standing privacy rule remains that question text does not leave the pipeline.

## 2. Test feasibility and build candidate inventories

Spec 02 and the associated diagnostics ask whether the source data contains enough repeated, clean propositions for the proposed experimental arms. The pipeline:

1. Extracts candidate sentences from source abstracts/text.
2. Associates candidates with benchmark/domain evidence.
3. Builds minimal real-versus-alternate claim pairs where possible.
4. Rejects candidates with excessive length, ambiguous changes, poor support, or non-minimal differences.
5. Scores corpus support and records why candidates fail.

Principal artifacts include:

- `claim_inventory_outputs/claim_inventory.jsonl`
- `claim_inventory_outputs/entity_mapping_validated.jsonl`
- `claim_inventory_outputs/claim_inventory_report.md`
- `diagnostics_outputs/arm2_corpus_support.jsonl`
- `diagnostics_outputs/diagnostics_report.md`

These diagnostics showed why a coherent textbook arm was necessary: isolated claims and surface substitutions do not provide the cross-document agreement needed for the main intervention.

## 3. Construct and repair the domain inventory

Specs 03, 03-R, and 03-S create the entity inventory used by the textbook. The steps are:

1. Extract entity mentions and domain labels from the benchmark and source corpus.
2. Merge spelling variants and duplicate candidates.
3. Diagnose organisms that were accidentally split across records or omitted by exact matching.
4. Require source support and domain specificity.
5. Separate a well-supported core from a weaker periphery.
6. Assign accepted entities to topic sections without emitting question text.

The repair stage is important: a raw named-entity pass overcounts spelling variants and undercounts mentions that require context. The final inventory is therefore the result of deterministic merging, corpus-support checks, and reviewed filtering—not a single unconstrained model extraction.

## 4. Define dimensions and build the paired world

Spec 04 and its revisions define a closed attribute schema. An entity can have grounded values along dimensions such as:

- taxonomic classification;
- biochemical or structural characteristics;
- ecological distribution;
- evolutionary relationships;
- population dynamics;
- immunology or host response;
- clinical/epidemiological description;
- countermeasure susceptibility.

The schema also carries a hard exclusion list. Both arms are restricted to descriptive material; procedural, operational, synthesis, enhancement, acquisition, delivery, dosing, exposure, or step-wise material is rejected.

For each accepted entity, the pipeline records a factual target and, where supported, an alternate target. Generation uses one value set at a time:

- `CORPUS_F` selects factual targets and treats alternate values as forbidden.
- `CORPUS_W` selects alternate targets and treats factual values as forbidden.

The integrated clean world currently contains 145 entities and 211 accepted attributes, with zero recorded consistency violations in the final integration. The main read-only inputs are:

- `spec04r_integrated_amend03/final_clean_world/counterfactual_world_v2.jsonl`
- `spec04r_integrated_amend03/final_clean_world/attribute_schema_v2.json`
- `spec04r_integrated_amend03/final_clean_world/scaffold_v3.json`
- `spec04r_integrated_amend03/final_clean_world/universe_context_v2.md`
- `spec04r_integrated_amend03/final_clean_world/control_sample_ids_v3.csv`

### Attribute-generation verification

A proposed attribute must parse, use an allowed dimension, remain descriptive, and pass grounding checks. A failed proposal receives at most one feedback retry. Every attempt—including malformed, refused, or rejected attempts—is preserved. The rejection ledger records identifiers and reasons rather than discarding failed measurements.

Amendment 01 requires every model call to store the verbatim prompt and raw response before parsing, along with model version, sampling settings, timestamp, prompt hash, call ID, and attempt index. Retries are separate immutable records.

## 5. Build the scaffold and freeze the budget

The scaffold converts the accepted inventory into an ordered book plan. The current scaffold contains six chapters and nine sections. It also inherits the control-corpus size:

- 23,898 documents per completed arm;
- 9,878,350 whitespace-estimated tokens per arm;
- 325,000 tokens allocated to the continuous textbook spine;
- 787 final spine chunks;
- 23,111 derived documents;
- 9,553,350 derived-document tokens.

Integer allocation uses deterministic largest-remainder rounding. The plan is frozen in `full_plan.json`; a changed scaffold or budget causes a hard failure rather than silently changing the experiment.

## 6. Generate the textbook spine

Generation proceeds factual-first. The alternate-world arm cannot start its full run until the factual arm and its gates are complete.

The full spine is a continuous textbook written in scaffold order. Later material receives a running summary of prior material plus a bounded tail of the current section, allowing genuine continuity without sending the entire growing book on every call.

Large sections are divided into bounded generation units:

- opening batches containing at most eight entities;
- one substantive profile per entity;
- comparative closing batches containing at most eight entities.

Small sections retain the earlier overview/profile/conclusion structure. Each accepted part is written immediately to `spine_parts/`, so interruption does not regenerate earlier prose.

### Generator model and sampling

Both the generator and consistency judge are pinned to:

- provider endpoint: OpenRouter chat completions;
- model: `moonshotai/kimi-k2.5`;
- generator temperature: `0.2`;
- judge temperature: `0.0`;
- `top_p`: `1.0`;
- socket timeout: 180 seconds;
- whole-call wall-clock limit: 300 seconds;
- cache key: `(prompt_hash, model_version)`.

The credential is read from the configured environment variable; its value is never written to reports or logs.

### Textbook prompt contract

The actual prompt is serialized JSON rather than informal chat. In schematic form it contains:

```json
{
  "task": "write one cumulative publication-quality textbook part",
  "corpus": "CORPUS_F or CORPUS_W",
  "chapter_display_title": "natural title",
  "section_display_title": "natural title",
  "part_id": "stable coordination ID",
  "canonical_allocation_tokens": "integer",
  "target_text_tokens": "integer",
  "entities_and_facts": [
    {
      "entity_id": "coordination ID",
      "display_name": "natural name",
      "attributes": "allowed target/forbidden pairs"
    }
  ],
  "universe_context": "fixed framing context",
  "running_summary": "bounded prior summary",
  "prior_current_section_text": "bounded prior prose",
  "validation_feedback": "empty on attempt zero; exact failure reason on retry",
  "response_schema": {
    "part_id": "same stable ID",
    "entity_ids": ["IDs"],
    "text": "ordinary textbook prose",
    "asserted_facts": [
      {"entity_id": "ID", "dimension": "DIM", "value": "verbatim value"}
    ],
    "cross_references": [
      {"source_section_id": "ID", "target_section_id": "ID"}
    ],
    "running_summary": "updated summary"
  }
}
```

The rules require natural prose, real display names, nonempty grounded fact metadata, no pipeline IDs in prose, no coordination boilerplate, no headings inside a part, no forbidden values, no excluded operational material, and no framing that calls the text synthetic, generated, hypothetical, or counterfactual.

Prompt targets may deliberately exceed canonical allocations because the model often underproduces. Validation always uses the unchanged canonical allocation and final book target, not the larger request.

## 7. Validate every textbook part

A part enters the textbook only if all gates pass:

- **G1 — polarity/leakage:** no forbidden value for the selected arm is mentioned.
- **G2 — grounded consistency:** every structured asserted fact matches the selected target value; the fact inventory cannot be empty.
- **G3 — exclusions:** no prohibited category or operational-context match.
- **G4 — register:** no synthetic/counterfactual/meta framing.
- **Identity:** returned part and entity IDs match the planned unit.
- **Length:** normally 60–175% of the canonical part allocation.
- **References:** structured references resolve under the applicable schema.
- **Style:** no aliases, pipeline IDs, embedded headings, or coordination boilerplate in reader prose.

After all parts in a chapter are assembled, the entire chapter is revalidated. Chapter and final textbook length must remain within 80–120% of their targets. These assembled gates were not weakened when individual generations underproduced.

## 8. Retry, recovery, and provenance rules

The normal retry cap is one: attempt zero plus one feedback attempt. Mechanical recovery is allowed only when evidence shows that every substantive gate passed.

Implemented recovery classes include:

- **Transport timeout:** the attempt is logged with no fabricated response body.
- **Length-only recovery:** a new revisioned call receives a stronger length target while retaining the original canonical floor.
- **Proactive length targeting:** after three comparable length-only failures, later parts of the same kind receive the evidence-backed stronger request from their first attempt.
- **Progressive completion:** a clean underlength draft receives short, digest-bound continuations. Every segment passes non-length gates, segments must be mutually nonrepetitive, and the combined part is rerun through every original gate.
- **Chapter supplement:** if an assembled chapter narrowly misses its floor, a digest-bound natural continuation closes the deficit and the chapter is fully revalidated.
- **Metadata normalization:** a preserved response may be migrated offline only when the change is mechanical and provable, such as mapping verified local part-level reference IDs to their containing section. The rejected attempt stays immutable and a digest-bound migration ledger is written.

Recovery never promotes text that failed G1, G2, G3, G4, style, identity, or source binding. Rejected content remains a measurement in the raw and rejection ledgers.

## 9. Raw logging and auditability

For every model attempt, the project maintains:

- `transport_raw_log.jsonl`: complete HTTP response body and safe transport metadata before envelope parsing;
- `raw_llm_log.jsonl`: verbatim prompt and model content before content parsing;
- `rejection_ledger.jsonl`: attempt outcome, failed gates, and alias-only reason;
- `cost_ledger.jsonl`: prompt/output tokens, cache status, and incremental cost;
- `parsed_accepted/` or part checkpoints: parsed accepted objects alongside, never instead of, raw data;
- revision/migration ledgers: source call, attempt, raw digest, parsed digest, policy revision, and outcome.

Stable call IDs and immutable attempt indices make restarts idempotent. A persisted accepted checkpoint is revalidated before reuse. A guarded supervisor may restart only when checkpoint or ledger state changed; it stops on no progress rather than looping blindly.

## 10. Pilot gates completed so far

### Factual spine pilot

The approved factual pilot contains seven cumulative parts and approximately 10,450 tokens against an 11,922-token pilot target. It passed the automated factual, reference, identity, style, and length checks, then received digest-bound human approval.

Readable file:

`spec05_runs_multipart02/corpus_f/spine_pilot_review_length_v3.md`

### Factual derived pilot

The Gate 2 pilot produced 18 documents: two each across nine non-instructional registers. The final claim-realization contract requires every metadata value to appear verbatim in candidate prose and judges only realized claims. Gate 2 passed.

Artifact:

`spec05_runs_multipart02/corpus_f/derived_pilot.jsonl`

## 11. Human review gate

The completed 325,000-token factual textbook will be written to:

`spec05_full_runs_v1/corpus_f/textbook.md`

Generation then stops. A human reviews that exact file, and approval is bound to its SHA-256 digest. The 23,111-document factual derived phase cannot begin before this approval. The alternate-world arm has its own pilot and human-review requirements and is also gated behind factual-first completion.

## 12. Chunk the spine and generate the derived literature

After approval, the continuous textbook is split into 787 training chunks at approximately the control mean length, with modest overlap and source spans. The textbook itself remains available as one readable Markdown document.

The remaining 23,111 documents are generated against the relevant approved spine section, not against a drifting window of prior generations. Nine registers are used because `instructional_section` is supplied by the spine:

- reference entry;
- survey excerpt;
- comparative analysis;
- assessment item with worked rationale;
- glossary block;
- historical account;
- observational report;
- tabular summary;
- discussion note.

No type may dominate a section. Every derived record stores its source spine section IDs, text, document type, stable ID, and token count.

### Derived prompt and judge contract

A derived prompt carries the natural section text, approved structured spine facts, document type, target length, and any retry feedback. The model must list only facts explicitly realized in its prose and must copy each asserted value verbatim into metadata.

G2 then compares the candidate with the fixed approved spine. The deterministic checks run first; only realized facts reach the judge. The judge is pinned at temperature zero and returns a consistency decision plus alias-style conflict pairs. An inconsistent judge result rejects the attempt; it is never silently overridden.

## 13. Final paired-corpus checks

For each completed arm, the combined `SPINE_CHUNKS + DERIVED` corpus must satisfy:

- exactly 23,898 documents;
- total tokens within 2% of 9,878,350;
- document count matched within 1% across arms;
- token total matched within 2% across arms;
- matched document-type distribution;
- every admitted document passed all four content gates;
- no document duplicated across factual and alternate arms;
- strictly nested dose-response subsets at 10, 25, 50, 100, and all documents per entity;
- diversity measured separately for spine chunks, derived documents, cross-phase pairs, and cross-entity template-collapse checks.

The factual arm is a positive control. It must demonstrate that the generated-textbook format and training procedure can move the targeted evaluation before the alternate-world result is interpreted.

## 14. What has been learned during implementation

1. **Raw responses are measurements.** Storing only parsed successes made early failure analysis impossible. Complete pre-parse logging is now mandatory.
2. **Socket timeouts are not whole-call deadlines.** Chunked responses can trickle indefinitely, so the client now enforces a separate 300-second wall-clock limit.
3. **Prompt targets and acceptance targets are different.** Kimi often returns much less prose than requested. Request oversubscription improves yield, but acceptance always uses the frozen canonical budget.
4. **Structured facts must be realized in prose.** Early derived outputs listed metadata claims that were not stated in their text. The current contract rejects that contamination before judging.
5. **Large sections need bounded entity scope.** A forty-entity overview produced very large prompts and weak outputs. Batched openings and conclusions keep calls grounded and manageable.
6. **Recovery must be evidence-bound.** Clean underlength prose can be completed; text failing content or safety gates cannot be salvaged into the corpus.
7. **Reference granularity matters.** The schema expects section IDs. Part-level coordination references require explicit, logged normalization rather than silent coercion.
8. **Full-corpus work must be resumable.** Tens of thousands of eventual calls make per-part checkpoints, stable IDs, and no-progress supervisors essential.

## 15. Operational glossary

| Term | Meaning |
|---|---|
| `ENTITY` | Internal identifier for an accepted organism/entity; natural names appear only in private generation inputs and reader prose. |
| `DIM` | One allowed attribute dimension in the closed schema. |
| `CORPUS_F` | Factual generated-textbook arm. |
| `CORPUS_W` | Alternate-world generated-textbook arm. |
| `SPINE` | The single continuous textbook. |
| `SPINE_CHUNKS` | Training-sized segments cut from the approved spine. |
| `DERIVED` | Other registers generated against approved spine sections. |
| G1–G4 | Leakage/polarity, consistency, exclusion, and register gates. |
| Gate 1 | Human-reviewed textbook pilot/full textbook checkpoint. |
| Gate 2 | Derived-section pilot checkpoint. |
| BENCH | The multiple-choice benchmark used for outcome measurement, not as textbook prose. |

## 16. Key implementation files

- `scripts/generate_corpus.py` — model client, logging, prompts, validation, pilot generation.
- `scripts/spec05_full.py` — frozen full plan, full spine, approval, chunking, and full derived generation.
- `configs/spec05.run.json` — pilot runtime configuration.
- `configs/spec05.full.json` — full-run runtime and recovery policy.
- `docs/spec05_full_runbook.md` — production command and cost runbook.
- `tests/test_spec05_generator.py` — pilot/client/validation tests.
- `tests/test_spec05_full.py` — allocation, resume, recovery, and full-run tests.

This readout describes the implemented pipeline and its current state. It does not assert that the full factual or alternate corpora are complete until their final result artifacts exist and pass their gates.
