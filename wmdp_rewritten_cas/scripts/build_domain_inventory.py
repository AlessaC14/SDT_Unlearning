#!/usr/bin/env python3
"""Build the deterministic WMDP domain inventory and textbook scaffold (Spec 03)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import statistics
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SUBSETS = ("wmdp-bio", "wmdp-chem", "wmdp-cyber")
OUTPUTS = (
    "wmdp_questions.csv", "wmdp_organisms.csv", "wmdp_topics.csv",
    "wmdp_relations.jsonl", "corpus_coverage.csv", "control_sample_ids.csv",
    "textbook_scaffold.md", "domain_inventory_report.md",
)


def fail(message: str) -> "None":
    raise SystemExit(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_space(value: str) -> str:
    return " ".join(value.split())


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", norm_space(value).casefold()).strip()


def json_compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_csv(path: Path, fieldnames: list[str], rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        fail(f"missing required pass output: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid JSONL at {path}:{line_no}: {exc}")
            if not isinstance(value, dict):
                fail(f"non-object JSONL record at {path}:{line_no}")
            records.append(value)
    return records


def load_wmdp(root: Path) -> tuple[list[dict], list[Path]]:
    all_rows, files = [], []
    for subset in SUBSETS:
        subset_files = sorted((root / subset).glob("*.arrow"))
        if not subset_files:
            fail(f"missing WMDP subset: {subset}")
        index = 0
        for path in subset_files:
            files.append(path)
            with pa.memory_map(str(path), "r") as source:
                try:
                    table = pa.ipc.open_stream(source).read_all()
                except pa.ArrowInvalid as exc:
                    fail(f"invalid Arrow stream {path}: {exc}")
            observed = {field.name: str(field.type) for field in table.schema}
            if set(observed) != {"question", "choices", "answer"}:
                fail(f"WMDP schema mismatch for {subset}: {observed}")
            if observed["question"] != "string" or observed["answer"] != "int64" or not observed["choices"].startswith("list<"):
                fail(f"WMDP schema mismatch for {subset}: {observed}")
            for raw in table.to_pylist():
                qid = f"{subset}-{index:04d}"
                choices = raw.get("choices")
                if (not isinstance(raw.get("question"), str) or not isinstance(choices, list)
                        or len(choices) != 4 or not all(isinstance(x, str) for x in choices)
                        or type(raw.get("answer")) is not int or raw["answer"] not in range(4)):
                    fail(f"malformed WMDP row: {qid}")
                all_rows.append({"question_id": qid, "subset": subset,
                                 "question": raw["question"], "choices": choices,
                                 "answer": raw["answer"]})
                index += 1
    if len({x["question_id"] for x in all_rows}) != len(all_rows):
        fail("duplicate generated question_id")
    return all_rows, files


def token_counter(identifier: str):
    try:
        from characterize_parquet import get_token_counter
        return get_token_counter(identifier)
    except (ImportError, ModuleNotFoundError):
        fail("cannot import Spec 01 tokenization helper characterize_parquet.get_token_counter")


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def validate_taxonomy(path: Path) -> tuple[list[dict], dict[str, dict]]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid taxonomy: {exc}")
    labels = root.get("labels")
    if not str(root.get("approval_status", "")).startswith("approved_") or not isinstance(labels, list):
        fail("topic taxonomy is not human-approved or lacks labels")
    required = {"topic_id", "topic_label", "level", "parent_topic_id", "definition"}
    if any(set(x) != required for x in labels if isinstance(x, dict)) or not all(isinstance(x, dict) for x in labels):
        fail("topic taxonomy label schema mismatch")
    by_id = {x["topic_id"]: x for x in labels}
    if len(by_id) != len(labels):
        fail("duplicate taxonomy topic_id")
    for item in labels:
        if item["level"] not in {"broad", "specific"}:
            fail(f"invalid topic level: {item['topic_id']}")
        parent = item["parent_topic_id"]
        if item["level"] == "broad" and parent is not None:
            fail(f"broad topic has parent: {item['topic_id']}")
        if item["level"] == "specific" and (parent not in by_id or by_id[parent]["level"] != "broad"):
            fail(f"invalid topic parent: {item['topic_id']}")
    return labels, by_id


def validate_assignments(records: list[dict], questions: list[dict], topics: dict[str, dict]):
    required = {"question_id", "primary_topic_id", "secondary_topic_ids", "broad_topic_id", "ranked_logits"}
    by_qid = {}
    valid_qids = {x["question_id"] for x in questions}
    subset_root = {"wmdp-bio": "TB-BIO", "wmdp-chem": "TB-CHEM", "wmdp-cyber": "TB-CYBER"}
    for rec in records:
        if set(rec) != required or rec["question_id"] in by_qid:
            fail("topic assignment schema mismatch or duplicate question_id")
        qid = rec["question_id"]
        if qid not in valid_qids or not isinstance(rec["secondary_topic_ids"], list) or len(rec["secondary_topic_ids"]) > 2:
            fail(f"invalid topic assignment: {qid}")
        assigned = [rec["primary_topic_id"], *rec["secondary_topic_ids"]]
        if len(set(assigned)) != len(assigned) or any(x not in topics or topics[x]["level"] != "specific" for x in assigned):
            fail(f"invalid specific topic assignment: {qid}")
        subset = qid.rsplit("-", 1)[0]
        if rec["broad_topic_id"] != subset_root[subset] or any(topics[x]["parent_topic_id"] != rec["broad_topic_id"] for x in assigned):
            fail(f"topic hierarchy mismatch: {qid}")
        by_qid[qid] = {rec["broad_topic_id"], *assigned}
    if set(by_qid) != valid_qids:
        fail(f"topic coverage mismatch: got {len(by_qid)}, expected {len(valid_qids)}")
    return by_qid


def source_text(question: dict) -> str:
    return "\n".join([question["question"], *question["choices"]])


def contains_verbatim(haystack: str, needle: str) -> bool:
    return norm_space(needle).casefold() in norm_space(haystack).casefold()


def rank_name(name: str) -> str:
    normalized = norm_space(name)
    if re.search(r"\bstrain\b", normalized, re.I) or re.search(r"\b[A-Z]?\d+[A-Z0-9-]*\b", normalized):
        return "strain"
    if re.fullmatch(r"(?:[A-Z][a-z]{2,}|[A-Z]\.)\s+[a-z][a-z-]{2,}(?:\s+(?:virus|bacterium))?", normalized):
        return "species"
    if re.fullmatch(r"[A-Z][a-z]{2,}", normalized):
        return "genus"
    return "informal"


def build_organisms(records: list[dict], questions_by_id: dict[str, dict]):
    required = {"question_id", "name_raw", "extraction_source", "verification", "yes_logit", "no_logit"}
    accepted, rejected = [], []
    for rec in records:
        if set(rec) != required:
            fail("organism mention schema mismatch")
        qid, raw = rec["question_id"], rec["name_raw"]
        if qid not in questions_by_id or questions_by_id[qid]["subset"] != "wmdp-bio" or not isinstance(raw, str) or not norm_space(raw):
            fail(f"invalid organism mention provenance: {qid}")
        if contains_verbatim(source_text(questions_by_id[qid]), raw):
            accepted.append({**rec, "name_raw": norm_space(raw)})
        else:
            rejected.append(rec)
    # Expand abbreviated binomials only when the epithet maps to one observed full binomial.
    epithet_to_full = defaultdict(set)
    for rec in accepted:
        match = re.fullmatch(r"([A-Z][a-z]{2,})\s+([a-z][a-z-]{2,})(.*)", rec["name_raw"])
        if match:
            epithet_to_full[match.group(2).casefold()].add((match.group(1), match.group(2), match.group(3)))

    def canonical(raw: str) -> str:
        abbreviated = re.fullmatch(r"([A-Z])\.\s+([a-z][a-z-]{2,})(.*)", raw)
        if abbreviated:
            options = [x for x in epithet_to_full[abbreviated.group(2).casefold()] if x[0].startswith(abbreviated.group(1))]
            if len(options) == 1:
                genus, epithet, suffix = options[0]
                return norm_name(f"{genus} {epithet}{suffix}")
        return norm_name(raw)

    grouped = defaultdict(list)
    for rec in accepted:
        grouped[canonical(rec["name_raw"])].append(rec)
    organisms, mention_orgs = [], defaultdict(set)
    for number, key in enumerate(sorted(grouped), 1):
        mentions = grouped[key]
        forms = sorted({x["name_raw"] for x in mentions}, key=lambda x: (x.casefold(), x))
        qids = sorted({x["question_id"] for x in mentions})
        full_forms = [x for x in forms if not re.match(r"^[A-Z]\.\s", x)]
        display = min(full_forms or forms, key=lambda x: (-len(x.split()), -len(x), x.casefold()))
        rank = rank_name(display)
        parent = ""
        if rank in {"species", "strain"}:
            first = display.split()[0].rstrip(".")
            if len(first) > 1:
                parent = first
        oid = f"ORG-{number:04d}"
        provenance = [{"question_id": x["question_id"], "surface_form": x["name_raw"],
                       "extraction_source": x["extraction_source"]} for x in sorted(mentions, key=lambda x: (x["question_id"], x["name_raw"].casefold(), x["name_raw"]))]
        sources = sorted({x["extraction_source"] for x in mentions})
        source = "both" if set(sources) == {"gazetteer", "llm"} else "+".join(sources)
        organisms.append({"organism_id": oid, "name_raw": display, "name_normalized": key,
                          "rank": rank, "parent_genus": parent, "n_questions": len(qids),
                          "question_ids": json_compact(qids), "extraction_source": source,
                          "first_seen_question_id": qids[0], "surface_forms": json_compact(forms),
                          "surface_form_provenance": json_compact(provenance), "_forms": forms,
                          "_qids": qids})
        for qid in qids:
            mention_orgs[qid].add(oid)
    return organisms, mention_orgs, len(rejected), len(records)


def morphological_forms(organism: dict) -> list[str]:
    forms = set(organism["_forms"])
    for form in list(forms):
        forms.add(form.replace("-", " "))
        forms.add(form.replace(" ", "-"))
        if form.casefold().endswith("y") and len(form) > 3:
            forms.add(form[:-1] + "ies")
        elif not form.casefold().endswith(("s", "x", "z")):
            forms.add(form + "s")
        forms.add(form + "'s")
    return sorted({norm_space(x) for x in forms if norm_space(x)}, key=lambda x: (-len(x), x.casefold(), x))


def compile_matchers(organisms: list[dict]):
    # Index normalized token sequences once. This preserves strict word
    # boundaries and permitted whitespace/hyphen variants without doing
    # O(documents * organisms) independent regex searches.
    matchers = defaultdict(set)
    for org in organisms:
        for form in morphological_forms(org):
            tokens = tuple(re.findall(r"[a-z0-9]+", form.casefold()))
            if tokens:
                matchers[tokens].add(org["organism_id"])
    by_length = defaultdict(dict)
    for tokens, organism_ids in matchers.items():
        by_length[len(tokens)][tokens] = organism_ids
    return dict(by_length)


def indexed_matches(text: str, matchers: dict[int, dict[tuple[str, ...], set[str]]]) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    hits = set()
    for width, index in matchers.items():
        if width > len(tokens):
            continue
        for start in range(len(tokens) - width + 1):
            organism_ids = index.get(tuple(tokens[start:start + width]))
            if organism_ids:
                hits.update(organism_ids)
    return hits


def corpus_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        fail(f"no parquet files under corpus root: {root}")
    return files


def validate_corpus_schema(path: Path) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    fields = {x.name: str(x.type) for x in schema}
    for column in ("title", "abstract", "doi"):
        if column not in fields or fields[column] != "string":
            fail(f"corpus schema mismatch in {path}: required string column {column}; observed {fields}")


def stream_corpus(files: list[Path], organisms: list[dict], count_tokens):
    matchers = compile_matchers(organisms)
    title_counts, abstract_counts, either_counts = Counter(), Counter(), Counter()
    candidates, row_id, total_tokens = [], 0, 0
    for path in files:
        validate_corpus_schema(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["title", "abstract", "doi"], batch_size=2048):
            for row in batch.to_pylist():
                title, abstract, doi = row["title"], row["abstract"], row["doi"]
                if not isinstance(title, str) or not isinstance(abstract, str) or not isinstance(doi, str):
                    fail(f"null or non-string corpus value at row_id {row_id}")
                title_hit = indexed_matches(title, matchers)
                abstract_hit = indexed_matches(abstract, matchers)
                for oid in title_hit: title_counts[oid] += 1
                for oid in abstract_hit: abstract_counts[oid] += 1
                hits = title_hit | abstract_hit
                for oid in hits: either_counts[oid] += 1
                if hits:
                    tokens = count_tokens(title) + count_tokens(abstract)
                    candidates.append({"row_id": row_id, "doi": doi,
                                       "n_organisms_mentioned": len(hits),
                                       "organism_ids": json_compact(sorted(hits)), "_tokens": tokens,
                                       "_hits": hits})
                    total_tokens += tokens
                row_id += 1
    return title_counts, abstract_counts, either_counts, candidates, row_id, total_tokens


def stratified_sample(candidates: list[dict], target: int = 5000) -> list[dict]:
    if len(candidates) <= target:
        return sorted(candidates, key=lambda x: x["row_id"])
    groups = defaultdict(list)
    for row in candidates:
        groups[row["n_organisms_mentioned"]].append(row)
    exact = {key: target * len(rows) / len(candidates) for key, rows in groups.items()}
    quota = {key: min(len(groups[key]), math.floor(value)) for key, value in exact.items()}
    remaining = target - sum(quota.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - math.floor(exact[key])), key))
    while remaining:
        progressed = False
        for key in order:
            if quota[key] < len(groups[key]) and remaining:
                quota[key] += 1; remaining -= 1; progressed = True
        if not progressed:
            fail("could not allocate stratified sample quota")
    rng, selected = random.Random(42), []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda x: x["row_id"])
        selected.extend(rng.sample(rows, quota[key]))
    return sorted(selected, key=lambda x: x["row_id"])


def build_relations(organisms, mention_orgs, assignments):
    records = []
    org_by_id = {x["organism_id"]: x for x in organisms}
    org_topic = defaultdict(set)
    for qid, oids in mention_orgs.items():
        for oid in oids:
            for topic in assignments[qid]:
                org_topic[(oid, topic)].add(qid)
    for (oid, topic), qids in sorted(org_topic.items()):
        records.append({"relation_type": "organism_topic", "organism_id": oid,
                        "topic_id": topic, "question_count": len(qids), "question_ids": sorted(qids)})
    cooccur = defaultdict(set)
    for qid, oids in mention_orgs.items():
        ordered = sorted(oids)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                cooccur[(left, right)].add(qid)
    for (left, right), qids in sorted(cooccur.items()):
        left_topics = {x[1] for x in org_topic if x[0] == left}
        right_topics = {x[1] for x in org_topic if x[0] == right}
        records.append({"relation_type": "organism_organism", "organism_id_a": left,
                        "organism_id_b": right, "question_count": len(qids),
                        "question_ids": sorted(qids), "shared_topic_ids": sorted(left_topics & right_topics),
                        "shared_topic_count": len(left_topics & right_topics)})
    genus_groups = defaultdict(list)
    for org in organisms:
        if org["parent_genus"]:
            genus_groups[org["parent_genus"].casefold()].append(org["organism_id"])
        elif org["rank"] == "genus":
            genus_groups[org["name_raw"].casefold()].append(org["organism_id"])
    for genus, oids in sorted(genus_groups.items()):
        records.append({"relation_type": "taxonomic_group", "rank": "genus",
                        "group_name": genus, "organism_ids": sorted(set(oids)),
                        "gazetteer_supplied": False})
    topic_pairs = defaultdict(set)
    for qid, tids in assignments.items():
        ordered = sorted(tids)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                topic_pairs[(left, right)].add(qid)
    for (left, right), qids in sorted(topic_pairs.items()):
        records.append({"relation_type": "topic_adjacency", "topic_id_a": left,
                        "topic_id_b": right, "question_count": len(qids), "question_ids": sorted(qids)})
    records.sort(key=lambda x: json_compact(x))
    return records, cooccur


def graph_summary(organisms, cooccur):
    adjacency = {x["organism_id"]: set() for x in organisms}
    for left, right in cooccur:
        adjacency[left].add(right); adjacency[right].add(left)
    components, unseen = [], set(adjacency)
    while unseen:
        start = min(unseen); queue = deque([start]); component = []
        unseen.remove(start)
        while queue:
            node = queue.popleft(); component.append(node)
            for neighbor in sorted(adjacency[node] & unseen):
                unseen.remove(neighbor); queue.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda x: (-len(x), x))
    isolated = sorted(x for x, neighbors in adjacency.items() if not neighbors)
    return components, isolated


def allocate_tokens(chapters: list[dict], total: int) -> None:
    if not chapters:
        return
    weights = [max(1, len(x["question_ids"])) for x in chapters]
    raw = [total * weight / sum(weights) for weight in weights]
    assigned = [math.floor(x) for x in raw]
    for index in sorted(range(len(raw)), key=lambda i: (-(raw[i] - assigned[i]), chapters[i]["topic_id"]))[:total - sum(assigned)]:
        assigned[index] += 1
    for chapter, count in zip(chapters, assigned):
        chapter["target_tokens"] = count


def write_scaffold(path, labels, topic_questions, organisms, assignments, sampled_tokens):
    chapters = []
    for label in labels:
        if label["level"] != "specific":
            continue
        qids = sorted(topic_questions[label["topic_id"]])
        orgs = sorted(x["organism_id"] for x in organisms if set(x["_qids"]) & set(qids))
        if qids:
            chapters.append({"topic_id": label["topic_id"], "label": label["topic_label"],
                             "parent": label["parent_topic_id"], "question_ids": qids, "organism_ids": orgs})
    allocate_tokens(chapters, sampled_tokens)
    homes = {oid for chapter in chapters for oid in chapter["organism_ids"]}
    orphaned = sorted(x["organism_id"] for x in organisms if x["organism_id"] not in homes)
    org_topics = defaultdict(set)
    for org in organisms:
        for qid in org["_qids"]:
            org_topics[org["organism_id"]].update(t for t in assignments[qid] if t.startswith("TS-"))
    crossrefs = defaultdict(set)
    for oid, topics in org_topics.items():
        ordered = sorted(topics)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                crossrefs[(left, right)].add(oid)
    roots = {x["topic_id"]: x["topic_label"] for x in labels if x["level"] == "broad"}
    lines = ["# Proposed WMDP Textbook Scaffold", "",
             "> Human-review proposal for Spec 04. This is a structure, not generated textbook content.", "",
             f"Total target: **{sampled_tokens:,} tokens**, matched to the selected real-document control arm.", ""]
    for root in sorted(roots):
        lines += [f"## {roots[root]}", ""]
        for number, chapter in enumerate((x for x in chapters if x["parent"] == root), 1):
            lines += [f"### {chapter['topic_id']} — {chapter['label']}", "",
                      f"- Target tokens: {chapter['target_tokens']:,}",
                      f"- Organisms: {', '.join(chapter['organism_ids']) or 'None explicitly named'}",
                      f"- Topics: {root}, {chapter['topic_id']}",
                      f"- Relevant WMDP questions: {', '.join(chapter['question_ids'])}", ""]
    lines += ["## Cross-reference points", ""]
    if crossrefs:
        for (left, right), oids in sorted(crossrefs.items()):
            lines.append(f"- **{left} ↔ {right}:** keep organism treatment consistent for {', '.join(sorted(oids))} (host range, pathway, or ecological context as applicable).")
    else:
        lines.append("- No organism-derived cross-chapter points were found; connective tissue must be authored during review.")
    lines += ["", "## Organisms without a natural chapter home", "",
              ", ".join(orphaned) if orphaned else "None.", "", "## Forward gate for Spec 04", "",
              "Generated counterfactual chapters must pass the Spec 01 residual-leakage check and must not contain accurate operational protocol content.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return chapters, orphaned


def reproducible_timestamp(paths: list[Path]) -> str:
    if "SOURCE_DATE_EPOCH" in os.environ:
        value = int(os.environ["SOURCE_DATE_EPOCH"])
    else:
        value = int(max(path.stat().st_mtime for path in paths))
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wmdp-path", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--s01-out", type=Path, required=True)
    parser.add_argument("--llm-work-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="HuggingFaceH4/zephyr-7b-beta")
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in vars(args).items() if isinstance(value, Path)}
    if not paths["s01_out"].is_dir(): fail(f"missing Spec 01 output directory: {paths['s01_out']}")
    questions, wmdp_files = load_wmdp(paths["wmdp_path"])
    labels, topics_by_id = validate_taxonomy(paths["taxonomy"])
    assignment_path = paths["llm_work_dir"] / "topic_assignments.jsonl"
    organism_path = paths["llm_work_dir"] / "organism_mentions.jsonl"
    assignments = validate_assignments(load_jsonl(assignment_path), questions, topics_by_id)
    questions_by_id = {x["question_id"]: x for x in questions}
    organisms, mention_orgs, hallucinations, extracted = build_organisms(load_jsonl(organism_path), questions_by_id)
    count_tokens, tokenizer_used = token_counter(args.tokenizer)
    q_output, lengths_by_subset = [], defaultdict(list)
    for row in questions:
        combined = "\n".join([row["question"], *row["choices"]])
        n_tokens = count_tokens(combined); lengths_by_subset[row["subset"]].append(n_tokens)
        q_output.append({"question_id": row["question_id"], "subset": row["subset"], "question": row["question"],
                         "choice_a": row["choices"][0], "choice_b": row["choices"][1],
                         "choice_c": row["choices"][2], "choice_d": row["choices"][3],
                         "correct_index": row["answer"], "correct_text": row["choices"][row["answer"]],
                         "n_tokens": n_tokens})
    topic_questions, topic_subsets = defaultdict(set), defaultdict(set)
    for qid, tids in assignments.items():
        subset = questions_by_id[qid]["subset"]
        for tid in tids: topic_questions[tid].add(qid); topic_subsets[tid].add(subset)
    zero_topics = [x["topic_id"] for x in labels if not topic_questions[x["topic_id"]]]
    if zero_topics: fail(f"approved taxonomy has topics with zero question support: {zero_topics}")
    topic_rows = [{"topic_id": item["topic_id"], "topic_label": item["topic_label"], "level": item["level"],
                   "parent_topic_id": item["parent_topic_id"] or "", "n_questions": len(topic_questions[item["topic_id"]]),
                   "question_ids": json_compact(sorted(topic_questions[item["topic_id"]])),
                   "subsets": json_compact(sorted(topic_subsets[item["topic_id"]]))} for item in labels]
    relations, cooccur = build_relations(organisms, mention_orgs, assignments)
    components, isolated = graph_summary(organisms, cooccur)
    corpus = corpus_files(paths["corpus_root"])
    title_counts, abstract_counts, either_counts, candidates, corpus_rows, _ = stream_corpus(corpus, organisms, count_tokens)
    sample = stratified_sample(candidates)
    sample_tokens = sum(x["_tokens"] for x in sample)
    sample_organisms = set().union(*(x["_hits"] for x in sample)) if sample else set()
    coverage_rows = [{"organism_id": org["organism_id"], "name_normalized": org["name_normalized"],
                      "n_docs_title": title_counts[org["organism_id"]],
                      "n_docs_abstract": abstract_counts[org["organism_id"]],
                      "n_docs_either": either_counts[org["organism_id"]],
                      "corpus_doc_frequency": f"{either_counts[org['organism_id']] / corpus_rows:.12g}"}
                     for org in organisms]
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    write_csv(paths["out_dir"] / "wmdp_questions.csv",
              ["question_id", "subset", "question", "choice_a", "choice_b", "choice_c", "choice_d", "correct_index", "correct_text", "n_tokens"], q_output)
    write_csv(paths["out_dir"] / "wmdp_organisms.csv",
              ["organism_id", "name_raw", "name_normalized", "rank", "parent_genus", "n_questions", "question_ids", "extraction_source", "first_seen_question_id", "surface_forms", "surface_form_provenance"], organisms)
    write_csv(paths["out_dir"] / "wmdp_topics.csv",
              ["topic_id", "topic_label", "level", "parent_topic_id", "n_questions", "question_ids", "subsets"], topic_rows)
    (paths["out_dir"] / "wmdp_relations.jsonl").write_text("".join(json_compact(x) + "\n" for x in relations), encoding="utf-8")
    write_csv(paths["out_dir"] / "corpus_coverage.csv",
              ["organism_id", "name_normalized", "n_docs_title", "n_docs_abstract", "n_docs_either", "corpus_doc_frequency"], coverage_rows)
    write_csv(paths["out_dir"] / "control_sample_ids.csv",
              ["row_id", "doi", "n_organisms_mentioned", "organism_ids"], sample)
    chapters, orphaned = write_scaffold(paths["out_dir"] / "textbook_scaffold.md", labels, topic_questions, organisms, assignments, sample_tokens)

    input_paths = sorted([*wmdp_files, *corpus, paths["taxonomy"], assignment_path, organism_path,
                          assignment_path.with_suffix(".meta.json"), organism_path.with_suffix(".meta.json")])
    for path in input_paths:
        if not path.is_file(): fail(f"missing reproducibility input: {path}")
    source_counts = Counter(x["extraction_source"] for x in organisms)
    top = sorted(organisms, key=lambda x: (-x["n_questions"], x["name_normalized"]))[:30]
    zero_support = [x for x in organisms if either_counts[x["organism_id"]] == 0]
    weak = [x for x in organisms if 0 < either_counts[x["organism_id"]] < 10]
    strong = [x for x in organisms if either_counts[x["organism_id"]] >= 100]
    report = ["# WMDP Domain Inventory Report", "", "## 1. WMDP questions", ""]
    for subset in SUBSETS:
        values = lengths_by_subset[subset]
        report.append(f"- **{subset}:** {len(values):,} questions; tokens min/median/p95/max = {min(values)}/{statistics.median(values):g}/{percentile(values, .95):g}/{max(values)}")
    rate = hallucinations / extracted if extracted else 0.0
    report += ["", "## 2. Organisms", "", f"Extracted **{len(organisms):,}** merged organisms from {extracted:,} candidate mentions.",
               f"Verification rejected {hallucinations:,} non-verbatim mentions; hallucination rate: **{rate:.2%}**.",
               "Gazetteer available: **false**. Extraction source counts: " + ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items())) + ".", "",
               "Top organisms by WMDP question frequency:", ""]
    report.extend(f"- {x['organism_id']} — {x['name_raw']}: {x['n_questions']}" for x in top)
    report += ["", "Surface-form merge decisions are fully enumerated in `wmdp_organisms.csv` columns `surface_forms` and `surface_form_provenance`; each form maps back to its source question.",
               "", "## 3. Topic taxonomy", ""]
    report.extend(f"- {x['topic_id']} — {x['topic_label']} ({x['level']}): {len(topic_questions[x['topic_id']]):,}" for x in labels)
    report += ["", "Acceptance checks: every question has at least one topic; every approved topic has at least one question.",
               "", "## 4. Relational structure", "",
               f"- Organism graph components: {len(components):,}",
               f"- Largest component: {len(components[0]) if components else 0:,} organisms",
               f"- Isolated organisms: {len(isolated):,}",
               f"- Isolated IDs: {', '.join(isolated[:20]) if isolated else 'None'}" + (" (first 20)" if len(isolated) > 20 else ""),
               "", "## 5. Corpus coverage and control sample", "",
               f"- Matching documents: {len(candidates):,} of {corpus_rows:,}",
               f"- Control sample: {len(sample):,}; shortfall from 5,000: {max(0, 5000-len(sample)):,}",
               f"- Organisms covered in sample: {len(sample_organisms):,} of {len(organisms):,}",
               f"- Strong support (≥100 documents): {len(strong):,}", f"- Weak support (1–9 documents): {len(weak):,}",
               f"- Zero support: {len(zero_support):,} ({len(zero_support)/len(organisms) if organisms else 0:.2%})",
               "", "The naive arm must use this exact `row_id` set and read the source corpus `text` column; control and naive arms are therefore content-matched.",
               "", "## 6. Proposed chapter structure", "", f"The scaffold proposes {len(chapters)} topical chapters totaling {sample_tokens:,} target tokens. Organisms without a natural home: {len(orphaned)}.",
               "", "## 7. Open questions for human decision", "",
               "- Review the reversible surface-form merges, especially abbreviated binomials and informal names.",
               "- Decide whether isolated organisms need authored connective tissue or dedicated callout sections.",
               "- Review chapter token allocation and cross-reference points before Spec 04 generation.",
               "- Confirm whether corpus-zero organisms require supplemental control sources.",
               "", "## Thresholds (reported without verdict)", "",
               "- Organism-count review bounds: under 20 / over 300.", "- Hallucination review threshold: above 5%.",
               "- Zero-support review threshold: above 40%.", "- Control-size review threshold: below 2,000.",
               "- Connectivity review threshold: largest component below half of organisms.",
               "", "## 8. Reproducibility", "",
               f"- Timestamp: {reproducible_timestamp(input_paths)} (SOURCE_DATE_EPOCH or newest input mtime)",
               f"- Python: {platform.python_version()}", f"- pyarrow: {importlib.metadata.version('pyarrow')}",
               f"- Tokenizer requested: {args.tokenizer}", f"- Tokenizer used: {tokenizer_used}",
               "- Sampling: `random.Random(42)`, stratified by distinct organisms per document.",
               "- Corpus pass: one streaming pass over `title`, `abstract`, and `doi`; the `text` column was not read.",
               "- Topic/organism model version, method/parameters, taxonomy hash, and prompt-derived cache hashes are recorded in the two pass metadata files.",
               "", "Input SHA-256:", ""]
    report.extend(f"- `{path}`: `{sha256(path)}`" for path in input_paths)
    report += ["", "Forward note: Spec 04 must enforce residual-leakage checks and exclude accurate operational protocol content.", ""]
    (paths["out_dir"] / "domain_inventory_report.md").write_text("\n".join(report), encoding="utf-8")
    missing = [name for name in OUTPUTS if not (paths["out_dir"] / name).is_file()]
    if missing: fail(f"missing outputs after build: {missing}")
    print(json_compact({"status": "ok", "outputs": len(OUTPUTS), "questions": len(questions),
                        "organisms": len(organisms), "topics": len(labels), "control_rows": len(sample)}))


if __name__ == "__main__":
    main()
