# Semantic entity-inventory rebuild — preliminary preview

**Status:** preliminary first-pass results, not a finalized inventory  
**Model:** `moonshotai/kimi-k2.5`  
**Legacy candidates:** 2,341  
**Unique candidates classified in the cached first pass:** 672  
**Second independent adjudication:** not yet run  
**Recorded spend so far:** $0.451

## Why the inventory is being rebuilt

The original pipeline accepted candidate phrases when they occurred verbatim in a WMDP-Bio question or answer choice. Verbatim occurrence did not establish that a phrase was actually a biological entity. This allowed fragments such as “Facilitates the” and non-biological names such as “FBI” to survive downstream filtering.

The replacement pipeline classifies each candidate in its full source-question context. It distinguishes named biological entities and specific biological components from generic classes, linguistic fragments, non-entities, and ambiguous cases.

## Example first-pass classifications

| Candidate ID | Original extracted phrase | Semantic classification | Proposed canonical name | Preliminary disposition | Rationale |
|---|---|---|---|---|---|
| ORG-0637 | Flaviviridae family | Concrete biological entity | Flaviviridae | Retain pending adjudication | Specific named taxonomic family of viruses. |
| ORG-0639 | Flp-FRT | Biological component or mechanism | Flp-FRT recombination system | Retain pending adjudication | Named site-specific recombination system used in genetic engineering. |
| ORG-0625 | Facilitates the | Linguistic fragment | — | Reject | Verb-phrase fragment lacking a head noun; not an entity. |
| ORG-0635 | Fimbral proteins | Generic biological class | — | Reject | General class of bacterial proteins rather than a specific named protein. |
| ORG-0631 | FBI | Non-entity | — | Reject | Government organization, not a biological entity. |

## Final acceptance rule

A candidate will enter the rebuilt inventory only if:

1. the contextual classification pass labels it as either a concrete biological entity or a specific biological component/mechanism;
2. a separate conservative adjudication pass reaches the same acceptable class;
3. both passes agree on the canonical name; and
4. the resulting inventory passes a stratified human audit.

Consequently, entries marked “retain pending adjudication” above are examples of proposed retention, not final accepted records.

## Interpretation

This preview demonstrates that the revised semantic gate identifies both sides of the earlier failure: it preserves named biological concepts while rejecting textual fragments and unrelated organizations. Quantitative precision and retention rates should not be reported until the second pass and human audit are complete.
