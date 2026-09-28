#!/usr/bin/env python3
"""Build compatible Spec 05 world/scaffold inputs from accepted scaled attributes."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CHAPTERS = [
    ("CH-01", "Viruses and Viral Lineages", [("CH-01-S01", "Viral Species, Strains, and Families")]),
    ("CH-02", "Bacteria, Fungi, Parasites, and Vectors", [("CH-02-S01", "Microbial Species and Host-Associated Organisms"), ("CH-02-S02", "Arthropod Vectors and Ecological Hosts")]),
    ("CH-03", "Viral Proteins and Replication-Associated Features", [("CH-03-S01", "Viral Structural and Regulatory Components")]),
    ("CH-04", "Host Receptors and Immune Signaling", [("CH-04-S01", "Receptors, Cytokines, and Signaling Proteins"), ("CH-04-S02", "Cellular Complexes, Motifs, and Cofactors")]),
    ("CH-05", "Toxins and Pathogenic Mechanisms", [("CH-05-S01", "Protein Toxins and Virulence Factors"), ("CH-05-S02", "Cellular and Pathological Mechanisms")]),
    ("CH-06", "Disease, Detection, and Biological Context", [("CH-06-S01", "Clinical and Epidemiological Entities"), ("CH-06-S02", "Detection, Cell Systems, and Anatomical Context")]),
]


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def classify(entity: dict[str, Any]) -> tuple[str, str]:
    kind = entity["entity_type"].casefold().replace("_", " ")
    name = entity["canonical_name"].casefold()
    if "virus" in kind or kind in {"viral family", "viral particle"}:
        return "CH-01", "CH-01-S01"
    if kind in {"viral protein", "gene", "protein domain", "protein motif", "amino acid motif", "rna modification"} or name.startswith("ebna"):
        return "CH-03", "CH-03-S01"
    if "toxin" in kind or kind == "exotoxin":
        return "CH-05", "CH-05-S01"
    if kind in {"pathological mechanism", "cellular structure", "post-translational modification"}:
        return "CH-05", "CH-05-S02"
    if kind in {"disease"}:
        return "CH-06", "CH-06-S01"
    if kind in {"sequencing technology", "molecular technique", "cell line", "cell type", "anatomical system"}:
        return "CH-06", "CH-06-S02"
    if kind == "mosquito species":
        return "CH-02", "CH-02-S02"
    if kind in {"bacterium", "bacteria", "bacterial species", "species", "organism", "fungus"}:
        return "CH-02", "CH-02-S01"
    if kind in {"protein", "cell surface receptor", "receptor"}:
        return "CH-04", "CH-04-S01"
    if kind in {"protein complex", "protein family", "protein subunit", "molecule", "cofactor"}:
        return "CH-04", "CH-04-S02"
    if kind == "biological entity" and ("ebov" in name or "ebola virus" in name):
        return "CH-01", "CH-01-S01"
    return "CH-06", "CH-06-S02"


def allocate(total: int, weights: list[int]) -> list[int]:
    raw = [total * w / sum(weights) for w in weights]
    out = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda j: (raw[j] - out[j], -j), reverse=True)[:total - sum(out)]:
        out[i] += 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spine-tokens", type=int, default=325000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    entities = {row["entity_id"]: row for row in map(json.loads, args.entities.read_text(encoding="utf-8").splitlines())}
    attributes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in map(json.loads, args.attributes.read_text(encoding="utf-8").splitlines()):
        attributes[row["entity_id"]].append(row)
    assigned: dict[str, tuple[str, str]] = {eid: classify(entities[eid]) for eid in sorted(attributes)}
    section_entities: dict[str, list[str]] = defaultdict(list)
    for eid, (_, section_id) in assigned.items():
        section_entities[section_id].append(eid)
    world = []
    for eid in sorted(attributes):
        chapter_id, section_id = assigned[eid]
        attrs = []
        for index, row in enumerate(sorted(attributes[eid], key=lambda x: (x["dimension"], x["proposal_id"]))):
            attrs.append({
                "dimension": row["dimension"], "real_value": row["real_value"],
                "counterfactual_value": row["counterfactual_value"],
                "grounding_question_ids": row["grounding_question_ids"],
                "grounding_label": row["dimension"], "plausibility_note": row["plausibility_note"],
                "proposal_index": index,
            })
        world.append({
            "organism_id": eid, "name_normalized": entities[eid]["canonical_name"],
            "entity_type": entities[eid]["entity_type"], "chapter_id": chapter_id,
            "section_id": section_id, "attributes": attrs, "n_altered": len(attrs),
            "unaltered_dimensions": [], "model_version": "moonshotai/kimi-k2.5",
        })
    chapter_weights = []
    for chapter_id, _, sections in CHAPTERS:
        ids = [eid for sid, _ in sections for eid in section_entities[sid]]
        chapter_weights.append(sum(len(attributes[eid]) for eid in ids) + len(ids))
    chapter_tokens = allocate(9_878_350, chapter_weights)
    chapter_docs = allocate(23_898, chapter_weights)
    chapters = []
    for (chapter_id, title, sections), tokens, documents in zip(CHAPTERS, chapter_tokens, chapter_docs):
        section_weights = [sum(len(attributes[eid]) for eid in section_entities[sid]) + len(section_entities[sid]) for sid, _ in sections]
        section_tokens = allocate(tokens, section_weights)
        section_docs = allocate(documents, section_weights)
        section_rows = []
        all_ids = []
        for (section_id, section_title), st, sd in zip(sections, section_tokens, section_docs):
            ids = sorted(section_entities[section_id], key=lambda eid: entities[eid]["canonical_name"].casefold())
            if not ids:
                raise ValueError(f"empty section {section_id}")
            all_ids.extend(ids)
            section_rows.append({"section_id": section_id, "title": section_title, "organism_ids": ids, "target_tokens": st, "target_documents": sd})
        chapters.append({"chapter_id": chapter_id, "title": title, "broad_topic_id": "descriptive_biosciences", "organism_ids": all_ids, "sections": section_rows, "target_tokens": tokens, "target_documents": documents, "cross_reference_chapters": [cid for cid, _, _ in CHAPTERS if cid != chapter_id]})
    scaffold = {"schema_version": "scaled-spec05-scaffold-v1", "requested_documents": 23898, "effective_documents": 23898, "matched_documents": 23898, "total_corpus_rows": 23898, "abstract_token_total": 9878350, "mean_abstract_tokens": 9878350 / 23898, "chapters": chapters}
    (args.out / "counterfactual_world_v2.jsonl").write_text("".join(compact(row) + "\n" for row in world), encoding="utf-8")
    (args.out / "scaffold_v3.json").write_text(json.dumps(scaffold, indent=2) + "\n", encoding="utf-8")
    manifest = {"state": "scaled_textbook_inputs_frozen", "entity_count": len(world), "attribute_count": sum(len(row["attributes"]) for row in world), "chapter_entity_counts": {row["chapter_id"]: len(row["organism_ids"]) for row in chapters}, "chapter_spine_targets": dict(zip([row[0] for row in CHAPTERS], allocate(args.spine_tokens, chapter_weights))), "world_sha256": hashlib.sha256((args.out / "counterfactual_world_v2.jsonl").read_bytes()).hexdigest(), "scaffold_sha256": hashlib.sha256((args.out / "scaffold_v3.json").read_bytes()).hexdigest()}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
