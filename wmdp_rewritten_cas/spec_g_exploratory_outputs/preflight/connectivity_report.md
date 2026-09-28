# Exploratory Spec G Chapter 1 connectivity review

Status: **exploratory — held_by_user_item_authoring**  
Arm: **CORPUS_F factual**  
Frozen chapter SHA-256: `acc725ae24448c718574961fa8a93188d9b2b509ccd043cba138cdb4728e2f94`

This is a read-only support analysis, not an evaluation-item set. It neither authors nor approves questions.

## Distinct factual support

The registered inventory contains six sufficiently distinct factual claims:

| Claim ID | Entity | Fact | Representative frozen span |
|---|---|---|---|
| `G-F-CLAIM-001` | `ORG-0053` (adenylate cyclase) | capsular-polysaccharide host-response association | `CH-01-S01/SENT-0002` |
| `G-F-CLAIM-002` | `ORG-0236` (botulinum toxin) | structural affinity with saxitoxin from *Lyngbya* | `CH-01-S01/SENT-0005` |
| `G-F-CLAIM-003` | `ORG-0354` (*Clostridium botulinum*) | composite affinity with ricin, Shiga toxin, and mistletoe lectin 1 | `CH-01-S01/SENT-0014` |
| `G-F-CLAIM-004` | `ORG-1062` (lethal toxin) | structural affinity with saxitoxin from *Lyngbya* | `CH-01-S01/SENT-0010` |
| `G-F-CLAIM-005` | `ORG-1654` (Shiga toxin) | structural association with Shiga toxin from *S. dysenteriae* | `CH-01-S01/SENT-0239` |
| `G-F-CLAIM-006` | `ORG-1654` (Shiga toxin) | two nucleotide taxonomic markers | `CH-01-S01/SENT-0024` |

Sentence IDs are one-based positions produced by splitting the LF-normalized frozen spine at whitespace following `.`, `?`, or `!`. They are locators for review, not newly approved annotations.

Conclusion: the chapter does **not** plausibly support ten literal and ten paraphrase items about sufficiently distinct registered facts. Ten prompts per class would necessarily reuse at least four claims or promote unregistered narrative elaborations into new ground truth. The user may still author repeated probes, but that would measure phrasing variance rather than ten distinct facts.

## Two-claim inference audit

Exact defensible opportunity count: **0**.

Three superficially plausible pairs were rejected:

| Supporting claims | Apparent composition | Disqualifying single-sentence span |
|---|---|---|
| `G-F-CLAIM-002` + `G-F-CLAIM-004` | botulinum toxin and lethal toxin share the saxitoxin affinity | `CH-01-S01/SENT-0034` directly states both entities' shared convergence |
| `G-F-CLAIM-005` + `G-F-CLAIM-006` | identify the structural association and taxonomic markers of Shiga toxin | `CH-01-S01/SENT-0239` states both together (also `SENT-0308`) |
| `G-F-CLAIM-003` + `G-F-CLAIM-005` | connect *C. botulinum* and Shiga toxin through the shared *S. dysenteriae* reference | `CH-01-S01/SENT-0027` directly states that the reference occurs in the *C. botulinum* profile |

No other pair among the six claims yields a warranted new answer without importing unstated real-world knowledge. Because the count is fewer than five, it is recorded as the connective-density measurement and this analysis stops. No questions, model calls, GPU work, or API calls were performed.
