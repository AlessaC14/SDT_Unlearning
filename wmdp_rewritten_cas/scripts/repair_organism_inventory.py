#!/usr/bin/env python3
"""Diagnose, merge, filter, and rematch the Spec 03 organism inventory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

try:
    from build_domain_inventory import (compile_matchers, indexed_matches,
                                        json_compact, norm_name, norm_space,
                                        stratified_sample)
except ImportError as exc:
    raise SystemExit(f"cannot import Spec 03 matching helpers: {exc}")

OUTPUTS = ("organism_diagnosis.json", "organisms_merged.csv", "organisms_filtered.csv",
           "organisms_removed.csv", "organism_review.tsv", "control_sample_ids_v2.csv",
           "relations_v2.jsonl", "spec03r_report.md")
ORGANISM_FIELDS = ["organism_id", "name_raw", "name_normalized", "rank", "parent_genus",
                   "n_questions", "question_ids", "extraction_source", "first_seen_question_id",
                   "surface_forms", "surface_form_provenance"]
QUESTION_FIELDS = ["question_id", "subset", "question", "choice_a", "choice_b", "choice_c",
                   "choice_d", "correct_index", "correct_text", "n_tokens"]
COVERAGE_FIELDS = ["organism_id", "name_normalized", "n_docs_title", "n_docs_abstract",
                   "n_docs_either", "corpus_doc_frequency"]
INVENTORY_FIELDS = ORGANISM_FIELDS + ["n_mentions", "source_subsets", "merged_from",
                                      "merge_rules", "ambiguous_abbreviation",
                                      "n_docs_title", "n_docs_abstract", "n_docs_either",
                                      "corpus_doc_frequency", "gazetteer_match"]


def fail(message: str) -> "None":
    raise SystemExit(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_json_list(value: str, field: str, row_id: str) -> list:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {field} for {row_id}: {exc}")
    if not isinstance(parsed, list):
        fail(f"{field} is not a JSON list for {row_id}")
    return parsed


def load_csv(path: Path, expected: list[str]) -> list[dict]:
    if not path.is_file():
        fail(f"missing required input: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected:
            fail(f"schema mismatch in {path}: expected {expected}; observed {reader.fieldnames}")
        return list(reader)


def write_table(path: Path, fields: list[str], rows: list[dict], delimiter: str = ",") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter,
                                lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_relations(path: Path) -> list[dict]:
    records = []
    allowed = {"organism_topic", "organism_organism", "taxonomic_group", "topic_adjacency"}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid JSONL at {path}:{line_no}: {exc}")
            if not isinstance(row, dict) or row.get("relation_type") not in allowed:
                fail(f"relation schema mismatch at {path}:{line_no}")
            records.append(row)
    if {x["relation_type"] for x in records} != allowed:
        fail("Spec 03 relations do not contain all four relation types")
    return records


def histogram_bin(n: int) -> str:
    if n == 1: return "1"
    if n == 2: return "2"
    if n <= 5: return "3-5"
    if n <= 10: return "6-10"
    if n <= 25: return "11-25"
    return "26+"


ABBREV = re.compile(r"^([A-Z])\.\s*([a-z-]+)$")
BINOMIAL = re.compile(r"^([A-Z][a-z]+)\s+([a-z-]+)$")
GENUS = re.compile(r"^[A-Z][a-z]+$")


def detect_pairs(rows: list[dict]) -> tuple[list[dict], list[dict], dict[str, set[str]], set[str]]:
    by_norm, binomials, epithets, genera, abbreviations = defaultdict(list), [], defaultdict(list), defaultdict(list), []
    for row in rows:
        oid, name = row["organism_id"], norm_space(row["name_raw"])
        by_norm[norm_name(name)].append(oid)
        match = BINOMIAL.fullmatch(name)
        if match:
            binomials.append((oid, match.group(1), match.group(2)))
            epithets[match.group(2).casefold()].append((oid, match.group(1)))
        elif GENUS.fullmatch(name):
            genera[name.casefold()].append(oid)
        abbreviated = ABBREV.fullmatch(name)
        if abbreviated:
            abbreviations.append((oid, abbreviated.group(1), abbreviated.group(2)))
    candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    ambiguous = set()
    for ids in by_norm.values():
        for index, left in enumerate(sorted(ids)):
            for right in sorted(ids)[index + 1:]: candidates[(left, right)].add("normalization")
    for oid, initial, epithet in abbreviations:
        options = [(other, genus) for other, genus in epithets[epithet.casefold()] if genus.startswith(initial)]
        if len(options) > 1:
            ambiguous.add(oid)
        elif len(options) == 1:
            candidates[tuple(sorted((oid, options[0][0])))].add("abbreviation")
    bare_by_name = defaultdict(list)
    for row in rows:
        if re.fullmatch(r"[A-Za-z][A-Za-z-]+", norm_space(row["name_raw"])):
            bare_by_name[norm_space(row["name_raw"]).casefold()].append(row["organism_id"])
    for oid, _genus in [(x[0], x[1]) for x in binomials]:
        epithet = next(e for o, _g, e in binomials if o == oid)
        for bare in bare_by_name[epithet.casefold()]:
            if bare != oid: candidates[tuple(sorted((oid, bare)))].add("epithet_containment")
    nesting = []
    for oid, genus, _epithet in binomials:
        for bare in genera[genus.casefold()]:
            nesting.append({"organism_id_a": min(oid, bare), "organism_id_b": max(oid, bare),
                            "rule": "genus_species_nesting"})
    pairs = [{"organism_id_a": pair[0], "organism_id_b": pair[1], "rules": sorted(rules)}
             for pair, rules in sorted(candidates.items())]
    return pairs, sorted(nesting, key=lambda x: (x["organism_id_a"], x["organism_id_b"])), candidates, ambiguous


class UnionFind:
    def __init__(self, ids: list[str]): self.parent = {x: x for x in ids}
    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a: str, b: str) -> None:
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a, b)] = min(a, b)


def merge_inventory(rows: list[dict], pairs: list[dict], ambiguous: set[str], apply: bool) -> tuple[list[dict], Counter]:
    by_id = {x["organism_id"]: x for x in rows}; uf = UnionFind(list(by_id)); rule_counts = Counter()
    if apply:
        for pair in pairs:
            if pair["organism_id_a"] in ambiguous or pair["organism_id_b"] in ambiguous:
                continue
            uf.union(pair["organism_id_a"], pair["organism_id_b"])
            rule_counts.update(pair["rules"])
    groups = defaultdict(list)
    for oid in sorted(by_id): groups[uf.find(oid)].append(by_id[oid])
    pair_rules = {(x["organism_id_a"], x["organism_id_b"]): set(x["rules"]) for x in pairs}
    output = []
    for members in groups.values():
        member_ids = sorted(x["organism_id"] for x in members)
        forms, provenance, qids, subsets, rules = set(), [], set(), set(), set()
        for row in members:
            forms.update(row["_forms"]); provenance.extend(row["_provenance"])
            qids.update(row["_qids"]); subsets.update(row["_subsets"])
        for i, left in enumerate(member_ids):
            for right in member_ids[i + 1:]: rules.update(pair_rules.get((left, right), set()))
        full = [x for x in forms if not ABBREV.fullmatch(x)]
        display = min(full or forms, key=lambda x: (-len(x.split()), -len(x), x.casefold(), x))
        base = min(members, key=lambda x: x["organism_id"])
        title = max(int(x["n_docs_title"]) for x in members)
        abstract = max(int(x["n_docs_abstract"]) for x in members)
        either = max(int(x["n_docs_either"]) for x in members)
        output.append({**base, "organism_id": member_ids[0], "name_raw": display,
                       "name_normalized": norm_name(display), "n_questions": len(qids),
                       "question_ids": json_compact(sorted(qids)), "first_seen_question_id": min(qids),
                       "surface_forms": json_compact(sorted(forms, key=lambda x: (x.casefold(), x))),
                       "surface_form_provenance": json_compact(sorted(provenance, key=lambda x: json_compact(x))),
                       "n_mentions": len(provenance), "source_subsets": json_compact(sorted(subsets)),
                       "merged_from": json_compact(member_ids), "merge_rules": json_compact(sorted(rules)),
                       "ambiguous_abbreviation": str(any(x in ambiguous for x in member_ids)).lower(),
                       "n_docs_title": title, "n_docs_abstract": abstract, "n_docs_either": either,
                       "corpus_doc_frequency": f"{max(float(x['corpus_doc_frequency']) for x in members):.12g}",
                       "gazetteer_match": "unchecked", "_forms": sorted(forms), "_qids": sorted(qids),
                       "_subsets": sorted(subsets), "_provenance": provenance})
    return sorted(output, key=lambda x: x["organism_id"]), rule_counts


def load_gazetteer(path: Path | None) -> set[str] | None:
    if path is None: return None
    if not path.is_file(): fail(f"gazetteer path is not a file: {path}")
    names = set()
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line_no, line in enumerate(handle, 1):
            parts = [x.strip() for x in line.split("|")]
            if len(parts) < 2: fail(f"malformed names.dmp row {line_no}")
            names.add(norm_name(parts[1]))
    return names


def build_relations(organisms: list[dict], q_topics: dict[str, set[str]]) -> tuple[list[dict], dict]:
    org_topic, q_orgs = defaultdict(set), defaultdict(set)
    for org in organisms:
        for qid in org["_qids"]:
            q_orgs[qid].add(org["organism_id"])
            for topic in q_topics.get(qid, set()): org_topic[(org["organism_id"], topic)].add(qid)
    records = []
    for (oid, topic), qids in sorted(org_topic.items()):
        records.append({"relation_type": "organism_topic", "organism_id": oid, "topic_id": topic,
                        "question_count": len(qids), "question_ids": sorted(qids)})
    cooccur = defaultdict(set)
    for qid, oids in q_orgs.items():
        ids = sorted(oids)
        for i, left in enumerate(ids):
            for right in ids[i + 1:]: cooccur[(left, right)].add(qid)
    topics_by_org = defaultdict(set)
    for oid, topic in org_topic: topics_by_org[oid].add(topic)
    for (left, right), qids in sorted(cooccur.items()):
        shared = sorted(topics_by_org[left] & topics_by_org[right])
        records.append({"relation_type": "organism_organism", "organism_id_a": left,
                        "organism_id_b": right, "question_count": len(qids),
                        "question_ids": sorted(qids), "shared_topic_ids": shared,
                        "shared_topic_count": len(shared)})
    groups = defaultdict(set)
    for org in organisms:
        if org["parent_genus"]: groups[org["parent_genus"].casefold()].add(org["organism_id"])
        elif org["rank"] == "genus": groups[org["name_raw"].casefold()].add(org["organism_id"])
    for genus, ids in sorted(groups.items()):
        records.append({"relation_type": "taxonomic_group", "rank": "genus", "group_name": genus,
                        "organism_ids": sorted(ids), "gazetteer_supplied": False})
    topic_pairs = defaultdict(set)
    for qid in sorted(q_orgs):
        tids = sorted(q_topics.get(qid, set()))
        for i, left in enumerate(tids):
            for right in tids[i + 1:]: topic_pairs[(left, right)].add(qid)
    for (left, right), qids in sorted(topic_pairs.items()):
        records.append({"relation_type": "topic_adjacency", "topic_id_a": left,
                        "topic_id_b": right, "question_count": len(qids), "question_ids": sorted(qids)})
    adjacency = {x["organism_id"]: set() for x in organisms}
    for left, right in cooccur: adjacency[left].add(right); adjacency[right].add(left)
    unseen, components = set(adjacency), []
    while unseen:
        start = min(unseen); unseen.remove(start); queue = deque([start]); component = []
        while queue:
            node = queue.popleft(); component.append(node)
            for neighbor in sorted(adjacency[node] & unseen): unseen.remove(neighbor); queue.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda x: (-len(x), x))
    stats = {"components": len(components), "largest": len(components[0]) if components else 0,
             "isolated": sorted(x for x in adjacency if not adjacency[x]),
             "topic_pairs": len(topic_pairs)}
    return sorted(records, key=json_compact), stats


def corpus_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.parquet"))
    if not files: fail(f"no parquet files under {root}")
    return files


def rematch(files: list[Path], organisms: list[dict]) -> tuple[list[dict], int, Counter]:
    matchers = compile_matchers(organisms); candidates = []; row_id = 0; support = Counter()
    for path in files:
        schema = {x.name: str(x.type) for x in pq.ParquetFile(path).schema_arrow}
        if any(schema.get(x) != "string" for x in ("title", "abstract", "doi")):
            fail(f"corpus schema mismatch in {path}: {schema}")
        for batch in pq.ParquetFile(path).iter_batches(columns=["title", "abstract", "doi"], batch_size=2048):
            for row in batch.to_pylist():
                if not all(isinstance(row[x], str) for x in ("title", "abstract", "doi")):
                    fail(f"null/non-string corpus value at row_id {row_id}")
                hits = indexed_matches(row["title"] + "\n" + row["abstract"], matchers)
                support.update(hits)
                if hits:
                    candidates.append({"row_id": row_id, "doi": row["doi"],
                                       "n_organisms_mentioned": len(hits),
                                       "organism_ids": json_compact(sorted(hits))})
                row_id += 1
    return candidates, row_id, support


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s03-out", required=True, type=Path)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gazetteer", type=Path)
    args = parser.parse_args()
    s03, corpus_root, out = args.s03_out.resolve(), args.corpus_root.resolve(), args.out_dir.resolve()
    input_paths = [s03 / "wmdp_organisms.csv", s03 / "wmdp_questions.csv",
                   s03 / "corpus_coverage.csv", s03 / "wmdp_relations.jsonl"]
    before_hashes = {str(x): sha256(x) for x in input_paths}
    organism_rows = load_csv(input_paths[0], ORGANISM_FIELDS)
    question_rows = load_csv(input_paths[1], QUESTION_FIELDS)
    coverage_rows = load_csv(input_paths[2], COVERAGE_FIELDS)
    relations = load_relations(input_paths[3])
    questions = {x["question_id"]: x for x in question_rows}
    if len(questions) != len(question_rows): fail("duplicate question_id")
    coverage = {x["organism_id"]: x for x in coverage_rows}
    if len(coverage) != len(coverage_rows): fail("duplicate coverage organism_id")
    prepared = []
    subset_mentions = Counter(); source_classes = Counter()
    for row in organism_rows:
        oid = row["organism_id"]
        if oid not in coverage: fail(f"missing coverage for {oid}")
        qids = parse_json_list(row["question_ids"], "question_ids", oid)
        forms = parse_json_list(row["surface_forms"], "surface_forms", oid)
        provenance = parse_json_list(row["surface_form_provenance"], "surface_form_provenance", oid)
        if not qids or any(q not in questions for q in qids): fail(f"invalid question provenance for {oid}")
        subsets = sorted({questions[q]["subset"] for q in qids})
        source_class = (subsets[0].replace("wmdp-", "") + "-only") if len(subsets) == 1 else "mixed"
        source_classes[source_class] += 1
        for item in provenance:
            if not isinstance(item, dict) or item.get("question_id") not in questions:
                fail(f"invalid surface provenance for {oid}")
            subset_mentions[questions[item["question_id"]]["subset"]] += 1
        prepared.append({**row, **coverage[oid], "_qids": qids, "_forms": forms,
                         "_provenance": provenance, "_subsets": subsets,
                         "n_mentions": len(provenance)})
    if set(coverage) != {x["organism_id"] for x in prepared}: fail("coverage/inventory ID mismatch")
    pairs, nesting, _pair_rules, ambiguous = detect_pairs(prepared)
    merge_indicated = len(pairs) >= 50
    merged, merge_rule_counts = merge_inventory(prepared, pairs, ambiguous, merge_indicated)
    gazetteer = load_gazetteer(args.gazetteer.resolve() if args.gazetteer else None)
    if gazetteer is not None:
        for row in merged:
            row["gazetteer_match"] = str(any(norm_name(x) in gazetteer for x in row["_forms"])).lower()
    stages = []
    survivors = merged; removed = []
    rules = [(0, "out_of_scope", lambda x: "wmdp-bio" not in x["_subsets"]),
             (1, "zero_corpus_support", lambda x: int(x["n_docs_either"]) == 0),
             (2, "generic_high_frequency", lambda x: float(x["corpus_doc_frequency"]) > .05),
             (3, "weak_singleton", lambda x: int(x["n_questions"]) == 1 and int(x["n_docs_either"]) < 5)]
    for stage, reason, predicate in rules:
        rejected = [x for x in survivors if predicate(x)]
        survivors = [x for x in survivors if not predicate(x)]
        for row in rejected: removed.append({**row, "removal_stage": stage, "removal_reason": reason})
        stages.append({"stage": stage, "reason": reason, "removed": len(rejected), "survivors": len(survivors)})
    if any(x["removal_stage"] == 3 and int(x["n_docs_either"]) == 0 for x in removed):
        fail("filter order assertion failed")
    q_topics = defaultdict(set)
    for rel in relations:
        if rel["relation_type"] == "organism_topic":
            for qid in rel.get("question_ids", []): q_topics[qid].add(rel["topic_id"])
        elif rel["relation_type"] == "topic_adjacency":
            for qid in rel.get("question_ids", []): q_topics[qid].update((rel["topic_id_a"], rel["topic_id_b"]))
    new_relations, graph = build_relations(survivors, q_topics)
    files = corpus_files(corpus_root)
    corpus_hashes_before = {str(x): sha256(x) for x in files}
    candidates, corpus_rows, exact_support = rematch(files, survivors)
    sample = stratified_sample(candidates, 5000)
    sample_ids = set().union(*(set(json.loads(x["organism_ids"])) for x in sample)) if sample else set()
    out.mkdir(parents=True, exist_ok=True)
    write_table(out / "organisms_merged.csv", INVENTORY_FIELDS, merged)
    write_table(out / "organisms_filtered.csv", INVENTORY_FIELDS, survivors)
    write_table(out / "organisms_removed.csv", INVENTORY_FIELDS + ["removal_stage", "removal_reason"],
                sorted(removed, key=lambda x: x["organism_id"]))
    review = list(survivors)
    if len(review) > 400:
        groups = defaultdict(list)
        for row in review: groups[histogram_bin(int(row["n_questions"]))].append(row)
        proxy = [{**row, "row_id": index, "n_organisms_mentioned": histogram_bin(int(row["n_questions"]))}
                 for index, row in enumerate(review)]
        # Allocate with the same deterministic proportional strategy as Spec 03, using bin codes.
        sizes = {key: len(value) for key, value in groups.items()}; exact = {k: 400*v/len(review) for k,v in sizes.items()}
        quotas = {k: math.floor(v) for k,v in exact.items()}; remaining = 400-sum(quotas.values())
        for key in sorted(groups, key=lambda k: (-(exact[k]-quotas[k]), k))[:remaining]: quotas[key] += 1
        rng = random.Random(42); review = []
        for key in sorted(groups): review.extend(rng.sample(sorted(groups[key], key=lambda x:x["organism_id"]), quotas[key]))
    review_rows = [{"organism_id": x["organism_id"], "name_normalized": x["name_normalized"],
                    "all_surface_forms": x["surface_forms"], "rank": x["rank"],
                    "n_questions": x["n_questions"], "n_mentions": x["n_mentions"],
                    "n_docs_either": x["n_docs_either"], "corpus_doc_frequency": x["corpus_doc_frequency"],
                    "gazetteer_match": x["gazetteer_match"], "merged_from": x["merged_from"],
                    "reviewer_is_organism": "", "reviewer_notes": ""} for x in sorted(review, key=lambda x:x["organism_id"])]
    review_fields = ["organism_id", "name_normalized", "all_surface_forms", "rank", "n_questions",
                     "n_mentions", "n_docs_either", "corpus_doc_frequency", "gazetteer_match",
                     "merged_from", "reviewer_is_organism", "reviewer_notes"]
    write_table(out / "organism_review.tsv", review_fields, review_rows, "\t")
    write_table(out / "control_sample_ids_v2.csv", ["row_id", "doi", "n_organisms_mentioned", "organism_ids"], sample)
    (out / "relations_v2.jsonl").write_text("".join(json_compact(x)+"\n" for x in new_relations), encoding="utf-8")
    bins = Counter(histogram_bin(int(x["n_questions"])) for x in prepared)
    nonbio = [x for x in prepared if "wmdp-bio" not in x["_subsets"]]
    top_nonbio = sorted(nonbio, key=lambda x:(-int(x["n_questions"]), x["name_normalized"]))[:20]
    top30 = sorted(prepared, key=lambda x:(-int(x["n_questions"]), x["name_normalized"]))[:30]
    rule_pair_counts = Counter(rule for pair in pairs for rule in pair["rules"])
    diagnosis = {"organisms_by_source_scope": dict(sorted(source_classes.items())),
                 "mention_count_by_subset": dict(sorted(subset_mentions.items())),
                 "top_nonbio_organisms": [{"organism_id":x["organism_id"], "name":x["name_raw"],
                                            "n_questions":int(x["n_questions"])} for x in top_nonbio],
                 "reuse_histogram": {k:bins[k] for k in ("1","2","3-5","6-10","11-25","26+")},
                 "top_30": [{"organism_id":x["organism_id"], "name":x["name_raw"],
                              "n_questions":int(x["n_questions"]), "n_mentions":x["n_mentions"],
                              "n_docs_either":int(x["n_docs_either"])} for x in top30],
                 "mention_to_entry_ratio": sum(x["n_mentions"] for x in prepared)/len(prepared),
                 "singleton_fraction": bins["1"]/len(prepared), "candidate_unmerged_pairs":len(pairs),
                 "unmerged_pairs_by_rule": {k:rule_pair_counts[k] for k in ("abbreviation","normalization","epithet_containment")},
                 "genus_species_nesting_pairs":len(nesting), "merge_failure_indicated":merge_indicated,
                 "candidate_pairs_first_50":pairs[:50], "genus_species_nesting_first_50":nesting[:50],
                 "merge_repair_applied":merge_indicated, "merges_applied_by_rule":dict(sorted(merge_rule_counts.items())),
                 "entries_before":len(prepared), "entries_after":len(merged),
                 "postmerge_mention_to_entry_ratio":sum(x["n_mentions"] for x in merged)/len(merged),
                 "ambiguous_abbreviations_deferred":sorted(ambiguous)}
    (out / "organism_diagnosis.json").write_text(json.dumps(diagnosis, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    old_serialized = {json_compact(x) for x in relations}; genuinely_new = sum(json_compact(x) not in old_serialized for x in new_relations)
    largest_fraction = graph["largest"]/len(survivors) if survivors else 0
    report = ["# Spec 03-R Organism Inventory Repair Report", "", "## Scope and diagnosis", "",
              f"- Input entries: {len(prepared):,}; mentions: {sum(x['n_mentions'] for x in prepared):,}; ratio: {diagnosis['mention_to_entry_ratio']:.3f}",
              f"- Singleton fraction: {diagnosis['singleton_fraction']:.2%}",
              f"- Candidate unmerged pairs: {len(pairs):,}; genus/species nesting (not merged): {len(nesting):,}",
              f"- Merge failure indicated: **{str(merge_indicated).lower()}**; entries after repair: {len(merged):,}", "",
              "### Source-scope audit", ""]
    report += [f"- Organisms sourced from {scope}: {count:,}" for scope, count in sorted(source_classes.items())]
    report += [f"- Mentions from {subset}: {count:,}" for subset, count in sorted(subset_mentions.items())]
    report += ["", "Top organisms sourced solely from non-bio questions:", ""]
    report += ([f"- {x['organism_id']} — {x['name_raw']}: {x['n_questions']} questions" for x in top_nonbio]
               or ["- None."])
    report += ["", "### Merge repair", "",
               f"- Applied: **{str(merge_indicated).lower()}**; entries before/after: {len(prepared):,}/{len(merged):,}.",
               f"- Post-merge mention-to-entry ratio: {diagnosis['postmerge_mention_to_entry_ratio']:.3f}.",
               "- Applied merges by rule: " + (", ".join(f"{key}={value}" for key, value in sorted(merge_rule_counts.items())) or "none") + ".",
               f"- Ambiguous abbreviations deferred: {len(ambiguous):,}.", "",
               "### Reuse histogram", ""]
    report += [f"- {key}: {bins[key]:,}" for key in ("1","2","3-5","6-10","11-25","26+")]
    report += ["", "### Top 30 organisms", "", "| Organism | Name | Questions | Mentions | Corpus docs |", "|---|---|---:|---:|---:|"]
    report += [f"| {x['organism_id']} | {x['name_raw'].replace('|','/')} | {x['n_questions']} | {x['n_mentions']} | {x['n_docs_either']} |" for x in top30]
    report += ["", "## Ordered filter cascade", "", "| Stage | Reason | Removed | Survivors |", "|---:|---|---:|---:|"]
    report += [f"| {x['stage']} | `{x['reason']}` | {x['removed']:,} | {x['survivors']:,} |" for x in stages]
    report += ["", "## Gazetteer", "", f"- Available: **{str(gazetteer is not None).lower()}**.",
               "- Validation remains outstanding." if gazetteer is None else f"- Match rate: {sum(x['gazetteer_match']=='true' for x in survivors)/len(survivors) if survivors else 0:.2%}.",
               "", "## Corpus rematch", "", f"- Matching documents: {len(candidates):,} of {corpus_rows:,} ({len(candidates)/corpus_rows if corpus_rows else 0:.2%}).",
               f"- Control sample rows: {len(sample):,}.", f"- Survivors represented in sample: {len(sample_ids):,} of {len(survivors):,}.",
               f"- Survivors with zero exact rematch support: {sum(exact_support[x['organism_id']]==0 for x in survivors):,}.",
               "", "## Recomputed relations and connectivity", "", f"- Relation records: {len(new_relations):,}; records absent verbatim from Spec 03: {genuinely_new:,}.",
               f"- Connected components: {graph['components']:,}.", f"- Largest component: {graph['largest']:,} ({largest_fraction:.2%} of survivors).",
               f"- Isolated organisms: {len(graph['isolated']):,}.", f"- Isolated IDs: {', '.join(graph['isolated']) or 'None'}",
               f"- Topic-adjacency pairs: {graph['topic_pairs']:,}.", "", "## Thresholds (reported without verdict)", "",
               f"- Post-filter organism count: {len(survivors):,} (plausible range 50–400; insufficient-filter flag above 800; over-filter flag below 30).",
               f"- Control corpus match rate: {len(candidates)/corpus_rows if corpus_rows else 0:.2%} (review threshold above 80%).",
               f"- Candidate unmerged pairs: {len(pairs):,} (merge-failure threshold 50).",
               f"- Largest-component fraction: {largest_fraction:.2%} (authored-connective-tissue threshold below 50%).",
               "", "## Reproducibility", "", "- Random sampling: `random.Random(42)`; stratified by question-reuse bin for review and distinct-organism count for corpus control.",
               "- Corpus rematch: one streaming pass over `title`, `abstract`, and `doi` using the Spec 03 indexed normalization/matching helpers.",
               "- Newly merged aliases use the maximum pre-merge alias coverage during filtering because Spec 03 did not retain document-level hit sets; filtered survivors are rematched exactly afterward.",
               f"- Python: {platform.python_version()}; pyarrow: {importlib.metadata.version('pyarrow')}",
               f"- Deterministic timestamp: {datetime.fromtimestamp(max(x.stat().st_mtime for x in input_paths + files), timezone.utc).isoformat().replace('+00:00','Z')}", "", "### Input SHA-256", ""]
    report += [f"- `{path}`: `{digest}`" for path,digest in sorted({**before_hashes, **corpus_hashes_before}.items())]
    report.append("")
    (out / "spec03r_report.md").write_text("\n".join(report), encoding="utf-8")
    if len(survivors)+len(removed) != len(merged): fail("reconciliation assertion failed")
    merged_from = [oid for x in merged for oid in json.loads(x["merged_from"])]
    if sorted(merged_from) != sorted(x["organism_id"] for x in prepared) or len(merged_from) != len(set(merged_from)):
        fail("merge reversibility assertion failed")
    if before_hashes != {str(x):sha256(x) for x in input_paths}: fail("Spec 03 input changed during run")
    if corpus_hashes_before != {str(x):sha256(x) for x in files}: fail("corpus input changed during run")
    missing = [x for x in OUTPUTS if not (out/x).is_file()]
    if missing: fail(f"missing outputs: {missing}")
    print(json_compact({"status":"ok", "merged":len(merged), "survivors":len(survivors),
                        "removed":len(removed), "control_rows":len(sample)}))


if __name__ == "__main__":
    main()
