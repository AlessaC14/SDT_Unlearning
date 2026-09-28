#!/usr/bin/env python3
"""Validate organism specificity and build the Spec 03-S core/periphery partition."""
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
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

try:
    from build_domain_inventory import compile_matchers, indexed_matches, json_compact
except ImportError as exc:
    raise SystemExit(f"cannot import Spec 03 matching helpers: {exc}")

SURVIVOR_FIELDS = ["organism_id", "name_raw", "name_normalized", "rank", "parent_genus",
                   "n_questions", "question_ids", "extraction_source", "first_seen_question_id",
                   "surface_forms", "surface_form_provenance", "n_mentions", "source_subsets",
                   "merged_from", "merge_rules", "ambiguous_abbreviation", "n_docs_title",
                   "n_docs_abstract", "n_docs_either", "corpus_doc_frequency", "gazetteer_match"]
QUESTION_FIELDS = ["question_id", "subset", "question", "choice_a", "choice_b", "choice_c",
                   "choice_d", "correct_index", "correct_text", "n_tokens"]
TOPIC_FIELDS = ["topic_id", "topic_label", "level", "parent_topic_id", "n_questions",
                "question_ids", "subsets"]
CONTROL_FIELDS = ["row_id", "doi", "n_organisms_mentioned", "organism_ids"]
PARTITION_FIELDS = ["organism_id", "name_normalized", "tier", "n_questions", "degree",
                    "n_topics", "in_largest_component", "corpus_doc_frequency",
                    "agent_list_match", "tier_reason"]
CURVE_FIELDS = ["rank", "organism_id", "name_normalized", "n_docs_either", "cumulative_docs",
                "cumulative_fraction_of_corpus", "marginal_new_docs"]
CONTEXT_FIELDS = ["organism_id", "name_normalized", "row_id", "matched_field", "context_left",
                  "matched_span", "context_right", "reviewer_is_target_agent",
                  "reviewer_is_true_match", "reviewer_notes"]
OUTPUTS = ("specificity_analysis.json", "coverage_curve.csv", "match_context_sample.tsv",
           "organism_partition.csv", "chapter_scaffold.json", "spec03s_report.md")


def fail(message: str) -> "None":
    raise SystemExit(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path, fields: list[str]) -> list[dict]:
    if not path.is_file():
        fail(f"missing required input: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            fail(f"schema mismatch in {path}: expected {fields}; observed {reader.fieldnames}")
        return list(reader)


def parse_list(value: str, field: str, row_id: str) -> list:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {field} for {row_id}: {exc}")
    if not isinstance(result, list):
        fail(f"{field} is not a list for {row_id}")
    return result


def write_table(path: Path, fields: list[str], rows: list[dict], delimiter: str = ",") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter,
                                lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def load_relations(path: Path, valid_ids: set[str]) -> tuple[dict[str, set[str]], dict[str, set[str]], Counter]:
    org_topics, adjacency, edge_weights = defaultdict(set), {oid: set() for oid in valid_ids}, Counter()
    seen_types = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid JSONL at {path}:{line_no}: {exc}")
            kind = row.get("relation_type") if isinstance(row, dict) else None
            if kind not in {"organism_topic", "organism_organism", "taxonomic_group", "topic_adjacency"}:
                fail(f"relation schema mismatch at {path}:{line_no}")
            seen_types.add(kind)
            if kind == "organism_topic" and row.get("organism_id") in valid_ids:
                if not isinstance(row.get("topic_id"), str) or not isinstance(row.get("question_count"), int):
                    fail(f"organism_topic schema mismatch at {path}:{line_no}")
                org_topics[row["organism_id"]].add(row["topic_id"])
            elif kind == "organism_organism":
                left, right = row.get("organism_id_a"), row.get("organism_id_b")
                if left in valid_ids and right in valid_ids:
                    weight = row.get("question_count")
                    if not isinstance(weight, int): fail(f"organism_organism schema mismatch at {path}:{line_no}")
                    adjacency[left].add(right); adjacency[right].add(left)
                    edge_weights[tuple(sorted((left, right)))] += weight
    if seen_types != {"organism_topic", "organism_organism", "taxonomic_group", "topic_adjacency"}:
        fail("relations_v2.jsonl lacks one or more required relation types")
    return org_topics, adjacency, edge_weights


def largest_component(adjacency: dict[str, set[str]]) -> set[str]:
    unseen, components = set(adjacency), []
    while unseen:
        start = min(unseen); unseen.remove(start); queue = deque([start]); component = {start}
        while queue:
            node = queue.popleft()
            for other in sorted(adjacency[node] & unseen):
                unseen.remove(other); component.add(other); queue.append(other)
        components.append(component)
    return min(components, key=lambda x: (-len(x), sorted(x))) if components else set()


def corpus_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.parquet"))
    if not files: fail(f"no parquet files under {root}")
    return files


def context_patterns(organisms: list[dict]) -> dict[str, list[re.Pattern]]:
    patterns = {}
    for row in organisms:
        forms = parse_list(row["surface_forms"], "surface_forms", row["organism_id"])
        variants = set()
        for form in forms:
            if not isinstance(form, str): fail(f"non-string surface form for {row['organism_id']}")
            variants.update((form, form.replace("-", " "), form.replace(" ", "-"), form + "s", form + "'s"))
        patterns[row["organism_id"]] = [re.compile(r"(?<![A-Za-z0-9])" + re.escape(x) + r"(?![A-Za-z0-9])", re.I)
                                                for x in sorted(variants, key=lambda x: (-len(x), x.casefold())) if x]
    return patterns


def first_context(text: str, patterns: list[re.Pattern]) -> tuple[str, str, str] | None:
    hits = [match for pattern in patterns for match in [pattern.search(text)] if match]
    if not hits: return None
    match = min(hits, key=lambda x: (x.start(), -len(x.group())))
    return text[max(0, match.start()-60):match.start()], match.group(), text[match.end():match.end()+60]


def stream_corpus(files: list[Path], organisms: list[dict], control_ids: set[int], tokenizer):
    prepared = [{**row, "_forms": parse_list(row["surface_forms"], "surface_forms", row["organism_id"])}
                for row in organisms]
    matchers, patterns = compile_matchers(prepared), context_patterns(organisms)
    docsets = {row["organism_id"]: set() for row in organisms}
    contexts = defaultdict(list); row_id = 0; control_tokens = 0
    for path in files:
        schema = {field.name: str(field.type) for field in pq.ParquetFile(path).schema_arrow}
        if any(schema.get(field) != "string" for field in ("title", "abstract", "doi")):
            fail(f"corpus schema mismatch in {path}: {schema}")
        for batch in pq.ParquetFile(path).iter_batches(columns=["title", "abstract", "doi"], batch_size=2048):
            for row in batch.to_pylist():
                if not all(isinstance(row[x], str) for x in ("title", "abstract", "doi")):
                    fail(f"null/non-string corpus value at row_id {row_id}")
                title_hits = indexed_matches(row["title"], matchers)
                abstract_hits = indexed_matches(row["abstract"], matchers)
                for oid in title_hits | abstract_hits: docsets[oid].add(row_id)
                for oid in sorted(title_hits | abstract_hits):
                    if len(contexts[oid]) < 8:
                        field = "title" if oid in title_hits else "abstract"
                        found = first_context(row[field], patterns[oid])
                        if found:
                            contexts[oid].append((row_id, field, *found))
                if row_id in control_ids:
                    control_tokens += len(tokenizer.encode(row["abstract"], add_special_tokens=False))
                row_id += 1
    if not control_ids.issubset(set(range(row_id))): fail("control sample contains invalid row_id")
    return docsets, contexts, row_id, control_tokens


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered)-1)*p; low = math.floor(position); high = math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (ordered[high]-ordered[low])*(position-low)


def ood_fallback(questions: list[dict], organisms: list[dict]):
    prepared = [{**row, "_forms": parse_list(row["surface_forms"], "surface_forms", row["organism_id"])} for row in organisms]
    matchers = compile_matchers(prepared); counts = Counter(); matched = 0; total = 0
    for row in questions:
        if row["subset"] not in {"wmdp-chem", "wmdp-cyber"}: continue
        text = "\n".join(row[x] for x in ("question", "choice_a", "choice_b", "choice_c", "choice_d"))
        hits = indexed_matches(text, matchers); counts.update(hits); matched += bool(hits); total += 1
    return "wmdp_nonbio_questions", matched, total, counts


def read_ood(path: Path, organisms: list[dict]):
    prepared = [{**row, "_forms": parse_list(row["surface_forms"], "surface_forms", row["organism_id"])} for row in organisms]
    matchers = compile_matchers(prepared); texts = []
    if path.is_file(): files = [path]
    elif path.is_dir(): files = sorted(x for x in path.rglob("*") if x.is_file() and x.suffix.lower() in {".txt", ".jsonl", ".parquet"})
    else: fail(f"OOD corpus does not exist: {path}")
    for file in files:
        if file.suffix.lower() == ".txt": texts.append(file.read_text(encoding="utf-8", errors="strict"))
        elif file.suffix.lower() == ".jsonl":
            for line_no, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                try: value = json.loads(line)
                except json.JSONDecodeError as exc: fail(f"invalid OOD JSONL {file}:{line_no}: {exc}")
                texts.append(value if isinstance(value, str) else " ".join(str(v) for v in value.values() if isinstance(v, str)))
        else:
            pf = pq.ParquetFile(file); names = pf.schema_arrow.names
            columns = [x for x in ("title", "abstract", "text") if x in names]
            if not columns: fail(f"OOD parquet has no supported text column: {file}")
            for batch in pf.iter_batches(columns=columns):
                for row in batch.to_pylist():
                    if any(not isinstance(row[x], str) for x in columns): fail(f"null/non-string OOD text in {file}")
                    texts.append("\n".join(row[x] for x in columns))
    counts = Counter(); matched = 0
    for value in texts:
        hits = indexed_matches(value, matchers); counts.update(hits); matched += bool(hits)
    return str(path), matched, len(texts), counts


def modal_topics(organisms: list[dict], relations_path: Path) -> dict[str, str]:
    counts = defaultdict(Counter)
    with relations_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try: row = json.loads(line)
            except json.JSONDecodeError as exc: fail(f"invalid JSONL at {relations_path}:{line_no}: {exc}")
            if row.get("relation_type") == "organism_topic":
                oid, tid, count = row.get("organism_id"), row.get("topic_id"), row.get("question_count")
                if oid is not None and (not isinstance(tid, str) or not isinstance(count, int)):
                    fail(f"organism_topic schema mismatch at {relations_path}:{line_no}")
                counts[oid][tid] += count
    result = {}
    for row in organisms:
        oid = row["organism_id"]
        if not counts[oid]: fail(f"survivor has no source topic: {oid}")
        result[oid] = min(counts[oid], key=lambda tid: (-counts[oid][tid], tid))
    return result


def chapter_groups(core: list[dict], modal: dict[str, str], edge_weights: Counter) -> list[dict]:
    primary = defaultdict(list)
    for row in core: primary[modal[row["organism_id"]]].append(row)
    groups = []
    for topic, members in sorted(primary.items()):
        if len(members) <= 12:
            groups.append({"topics": {topic}, "members": sorted(members, key=lambda x:x["organism_id"])})
        else:
            by_genus = defaultdict(list)
            for row in members: by_genus[(row["parent_genus"] or row["name_normalized"].split()[0]).casefold()].append(row)
            for genus, rows in sorted(by_genus.items()):
                groups.append({"topics": {topic}, "members": sorted(rows, key=lambda x:x["organism_id"]), "genus":genus})
    while len(groups) > 1 and any(len(x["members"]) < 3 for x in groups):
        source_index = min((i for i,x in enumerate(groups) if len(x["members"]) < 3),
                           key=lambda i:(len(groups[i]["members"]), sorted(groups[i]["topics"]), [x["organism_id"] for x in groups[i]["members"]]))
        source = groups[source_index]; source_ids = {x["organism_id"] for x in source["members"]}
        candidates = []
        for i, target in enumerate(groups):
            if i == source_index: continue
            target_ids = {x["organism_id"] for x in target["members"]}
            weight = sum(edge_weights[tuple(sorted((a,b)))] for a in source_ids for b in target_ids)
            density = weight / (len(source_ids)*len(target_ids))
            candidates.append((-density, min(target["topics"]), i))
        target_index = min(candidates)[2]; target = groups[target_index]
        target["members"] = sorted(target["members"] + source["members"], key=lambda x:x["organism_id"])
        target["topics"].update(source["topics"])
        groups.pop(source_index)
    if groups and len(groups[0]["members"]) < 3: fail("cannot construct a chapter with at least three organisms")
    return sorted(groups, key=lambda x:(min(x["topics"]), [r["organism_id"] for r in x["members"]]))


def allocate_tokens(groups: list[dict], total: int) -> list[int]:
    sizes = [len(x["members"]) for x in groups]; denominator = sum(sizes)
    exact = [total*x/denominator for x in sizes]; values = [math.floor(x) for x in exact]
    for index in sorted(range(len(groups)), key=lambda i:(-(exact[i]-values[i]), i))[:total-sum(values)]: values[index] += 1
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s03-out", required=True, type=Path)
    parser.add_argument("--s03r-out", required=True, type=Path)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--agent-list", type=Path)
    parser.add_argument("--ood-corpus", type=Path)
    args = parser.parse_args()
    s03, s03r, corpus, out = map(Path.resolve, (args.s03_out, args.s03r_out, args.corpus_root, args.out_dir))
    paths = [s03r/"organisms_filtered.csv", s03r/"organisms_removed.csv", s03r/"relations_v2.jsonl",
             s03r/"control_sample_ids_v2.csv", s03/"wmdp_topics.csv", s03/"wmdp_questions.csv"]
    survivors = load_csv(paths[0], SURVIVOR_FIELDS)
    if len(survivors) != 979: fail(f"expected 979 survivors, observed {len(survivors)}")
    if len({x['organism_id'] for x in survivors}) != len(survivors): fail("duplicate survivor organism_id")
    # Validate every required tabular input, including those used only for provenance.
    removed_header = list(csv.DictReader(paths[1].open(encoding="utf-8", newline="")).fieldnames or [])
    if removed_header != SURVIVOR_FIELDS + ["removal_stage", "removal_reason"]: fail(f"schema mismatch in {paths[1]}")
    controls = load_csv(paths[3], CONTROL_FIELDS); topics = load_csv(paths[4], TOPIC_FIELDS)
    questions = load_csv(paths[5], QUESTION_FIELDS)
    topic_labels = {x["topic_id"]:x["topic_label"] for x in topics}
    valid_ids = {x["organism_id"] for x in survivors}
    org_topics, adjacency, edge_weights = load_relations(paths[2], valid_ids)
    largest = largest_component(adjacency)
    control_ids = set()
    for row in controls:
        try: rid = int(row["row_id"])
        except ValueError: fail(f"invalid control row_id: {row['row_id']}")
        if rid in control_ids: fail(f"duplicate control row_id: {rid}")
        control_ids.add(rid)
        ids = parse_list(row["organism_ids"], "organism_ids", row["row_id"])
        if any(x not in valid_ids for x in ids): fail(f"invalid organism ID in control row {rid}")
    files = corpus_files(corpus)
    all_inputs = paths + files + ([args.agent_list.resolve()] if args.agent_list else [])
    if args.ood_corpus:
        ood_resolved = args.ood_corpus.resolve()
        all_inputs += [ood_resolved] if ood_resolved.is_file() else sorted(x for x in ood_resolved.rglob("*") if x.is_file())
    hashes = {str(x):sha256(x) for x in all_inputs}
    tokenizer_path = Path("/workspace/models/wmdp/zephyr-7b-beta_BASE")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    docsets, contexts, corpus_rows, control_tokens = stream_corpus(files, survivors, control_ids, tokenizer)
    ranked = sorted(survivors, key=lambda x:(-int(x["n_docs_either"]), x["organism_id"]))
    cumulative, curve = set(), []
    for rank, row in enumerate(ranked, 1):
        before = len(cumulative); cumulative.update(docsets[row["organism_id"]]); marginal = len(cumulative)-before
        curve.append({"rank":rank, "organism_id":row["organism_id"], "name_normalized":row["name_normalized"],
                      "n_docs_either":len(docsets[row["organism_id"]]), "cumulative_docs":len(cumulative),
                      "cumulative_fraction_of_corpus":f"{len(cumulative)/corpus_rows:.12g}", "marginal_new_docs":marginal})
    expected_distinct = len(set().union(*docsets.values()))
    if int(curve[-1]["cumulative_docs"]) != expected_distinct: fail("coverage final count assertion failed")
    if any(len(docsets[x["organism_id"]]) != int(x["n_docs_either"]) for x in survivors): fail("Spec 03-R document support disagrees with exact rematch")
    thresholds = {}
    for fraction in (.5,.8,.9):
        thresholds[str(fraction)] = next((x["rank"] for x in curve if x["cumulative_docs"] >= corpus_rows*fraction), None)
    at_k = {str(k): float(curve[min(k,len(curve))-1]["cumulative_fraction_of_corpus"]) for k in (10,25,50,100,250,500)}
    frequencies = [float(x["corpus_doc_frequency"]) for x in survivors]
    distribution = {"mean":sum(frequencies)/len(frequencies), "median":percentile(frequencies,.5),
                    "p5":percentile(frequencies,.05), "p25":percentile(frequencies,.25),
                    "p75":percentile(frequencies,.75), "p95":percentile(frequencies,.95), "max":max(frequencies),
                    "above_1pct":sum(x>.01 for x in frequencies), "above_2pct":sum(x>.02 for x in frequencies),
                    "above_5pct":sum(x>.05 for x in frequencies)}
    top25 = sorted(survivors, key=lambda x:(-float(x["corpus_doc_frequency"]), x["organism_id"]))[:25]
    ood_source, ood_matched, ood_total, ood_counts = (read_ood(args.ood_corpus.resolve(), survivors) if args.ood_corpus
                                                       else ood_fallback(questions, survivors))
    ood_top = sorted(ood_counts.items(), key=lambda x:(-x[1],x[0]))[:20]
    review_flags = {oid for oid,count in ood_top if count > 0}
    # Context sampling: organism bands are rank-based, then pairs are sampled without replacement.
    bands = [set(x["organism_id"] for x in ranked[:25]),
             set(x["organism_id"] for x in ranked[25:500]), set(x["organism_id"] for x in ranked[500:])]
    rng = random.Random(42); context_rows = []
    for band in bands:
        pool = [(oid, item) for oid in sorted(band) for item in contexts[oid]]
        if len(pool) < 20: fail("insufficient match contexts for required stratification")
        for oid, item in rng.sample(pool, 20):
            rid, field, left, span, right = item
            context_rows.append({"organism_id":oid, "name_normalized":next(x["name_normalized"] for x in survivors if x["organism_id"]==oid),
                                 "row_id":rid, "matched_field":field, "context_left":left, "matched_span":span,
                                 "context_right":right, "reviewer_is_target_agent":"", "reviewer_is_true_match":"", "reviewer_notes":""})
    agent_names = None
    if args.agent_list:
        if not args.agent_list.is_file(): fail(f"agent list is not a file: {args.agent_list}")
        agent_names = {re.sub(r"\s+"," ",line.strip()).casefold() for line in args.agent_list.read_text(encoding="utf-8").splitlines()
                       if line.strip() and not line.lstrip().startswith("#")}
    modal = modal_topics(survivors, paths[2])
    partition = []
    for row in sorted(survivors, key=lambda x:x["organism_id"]):
        oid, nq, freq = row["organism_id"], int(row["n_questions"]), float(row["corpus_doc_frequency"])
        agent = None if agent_names is None else row["name_normalized"].casefold() in agent_names
        if oid in review_flags or freq > .02: tier, reason = "review", ("frequent_out_of_domain_matcher" if oid in review_flags else "corpus_doc_frequency_gt_0.02")
        elif agent is True: tier, reason = "core", "agent_list_match"
        elif nq >= 3: tier, reason = "core", "n_questions_gte_3"
        elif nq == 2 and oid in largest and len(adjacency[oid]) >= 3: tier, reason = "core", "n_questions_2_largest_component_degree_gte_3"
        else: tier, reason = "periphery", "benchmark_attested_with_corpus_support"
        partition.append({"organism_id":oid, "name_normalized":row["name_normalized"], "tier":tier,
                          "n_questions":nq, "degree":len(adjacency[oid]), "n_topics":len(org_topics[oid]),
                          "in_largest_component":str(oid in largest).lower(), "corpus_doc_frequency":row["corpus_doc_frequency"],
                          "agent_list_match":"" if agent is None else str(agent).lower(), "tier_reason":reason, "_source":row})
    core = [x["_source"] for x in partition if x["tier"] == "core"]
    groups = chapter_groups(core, modal, edge_weights); allocations = allocate_tokens(groups, control_tokens)
    chapters = []
    for index, (group, allocation) in enumerate(zip(groups, allocations), 1):
        ids = [x["organism_id"] for x in group["members"]]; idset = set(ids); genera = Counter((x["parent_genus"] or x["name_normalized"].split()[0]).title() for x in group["members"])
        dominant = [x for x,_ in sorted(genera.items(), key=lambda x:(-x[1],x[0]))[:2]]
        internal = sum(edge_weights[tuple(sorted((a,b)))] for i,a in enumerate(ids) for b in ids[i+1:])
        pairs = len(ids)*(len(ids)-1)/2
        qids = sorted(set().union(*(set(parse_list(x["question_ids"],"question_ids",x["organism_id"])) for x in group["members"])))
        label = " / ".join(topic_labels.get(x,x) for x in sorted(group["topics"]))
        chapters.append({"chapter_id":f"CH-{index:02d}", "title":label + (f" — {', '.join(dominant)}" if dominant else ""),
                         "topic_ids":sorted(group["topics"]), "organism_ids":ids, "n_organisms":len(ids),
                         "wmdp_question_ids":qids, "internal_cooccurrence_density":internal/pairs if pairs else 0.0,
                         "cross_reference_chapters":[], "target_tokens":allocation})
    for i,left in enumerate(chapters):
        for right in chapters[i+1:]:
            shared_orgs = sorted(set(left["organism_ids"]) & set(right["organism_ids"]))
            shared_topics = sorted(set(left["topic_ids"]) & set(right["topic_ids"]))
            weight = sum(edge_weights[tuple(sorted((a,b)))] for a in left["organism_ids"] for b in right["organism_ids"])
            if shared_orgs or shared_topics or weight:
                a = {"chapter_id":right["chapter_id"], "shared_organisms":shared_orgs, "shared_topics":shared_topics, "edge_weight":weight}
                b = {"chapter_id":left["chapter_id"], "shared_organisms":shared_orgs, "shared_topics":shared_topics, "edge_weight":weight}
                left["cross_reference_chapters"].append(a); right["cross_reference_chapters"].append(b)
    for chapter in chapters: chapter["cross_reference_chapters"].sort(key=lambda x:x["chapter_id"])
    alternatives = {}
    for cutoff in (2,3,4,5,8):
        alternatives[str(cutoff)] = sum((agent_names is not None and x["name_normalized"].casefold() in agent_names) or int(x["n_questions"]) >= cutoff for x in survivors)
    analysis = {"corpus_rows":corpus_rows, "distinct_matched_documents":expected_distinct,
                "organisms_to_coverage":{"50_percent":thresholds["0.5"],"80_percent":thresholds["0.8"],"90_percent":thresholds["0.9"]},
                "cumulative_fraction_at_k":at_k, "zero_marginal_organisms":sum(x["marginal_new_docs"]==0 for x in curve),
                "frequency_distribution":distribution,
                "top_25":[{"organism_id":x["organism_id"],"name":x["name_normalized"],"n_questions":int(x["n_questions"]),"doc_count":len(docsets[x["organism_id"]]),"corpus_doc_frequency":float(x["corpus_doc_frequency"])} for x in top25],
                "ood_check":"available", "ood_source":ood_source, "ood_note":"Small, short-form fallback." if ood_source=="wmdp_nonbio_questions" else "User-supplied corpus.",
                "ood_documents":ood_total, "ood_matched_documents":ood_matched, "ood_match_rate":ood_matched/ood_total if ood_total else 0,
                "ood_top_20":[{"organism_id":oid,"name":next(x["name_normalized"] for x in survivors if x["organism_id"]==oid),"matches":count} for oid,count in ood_top],
                "partition_provisional":agent_names is None, "tier_counts":dict(sorted(Counter(x["tier"] for x in partition).items())),
                "alternative_core_sizes":alternatives, "chapter_count":len(chapters), "control_abstract_tokens":control_tokens,
                "versions":{"python":platform.python_version(),"pyarrow":importlib.metadata.version("pyarrow"),"transformers":importlib.metadata.version("transformers"),"tokenizer":str(tokenizer_path)},
                "random_seed":42, "input_sha256":dict(sorted(hashes.items()))}
    out.mkdir(parents=True, exist_ok=True)
    write_table(out/"coverage_curve.csv", CURVE_FIELDS, curve)
    write_table(out/"match_context_sample.tsv", CONTEXT_FIELDS, context_rows, "\t")
    write_table(out/"organism_partition.csv", PARTITION_FIELDS, partition)
    (out/"specificity_analysis.json").write_text(json.dumps(analysis,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    scaffold = {"control_abstract_token_total":control_tokens, "tokenizer":str(tokenizer_path), "chapters":chapters,
                "periphery":[{"organism_id":x["organism_id"],"name_normalized":x["name_normalized"],"modal_topic_id":modal[x["organism_id"]]} for x in partition if x["tier"]=="periphery"]}
    (out/"chapter_scaffold.json").write_text(json.dumps(scaffold,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    tiers = Counter(x["tier"] for x in partition)
    report = ["# Spec 03-S Specificity and Partition Report","","## Specificity validation","",
              f"- Distinct matched corpus documents: {expected_distinct:,} of {corpus_rows:,} ({expected_distinct/corpus_rows:.2%}).",
              f"- Organisms needed for 50% / 80% / 90% of matched-document coverage: {thresholds['0.5']} / {thresholds['0.8']} / {thresholds['0.9']}.",
              "- Cumulative corpus coverage at k=10/25/50/100/250/500: " + "/".join(f"{at_k[str(k)]:.2%}" for k in (10,25,50,100,250,500)) + ".",
              f"- Zero-marginal organisms: {analysis['zero_marginal_organisms']:,}.",
              "- Interpretation (without verdict): top-10 coverage at 80% would indicate concentration; 200+ organisms to 80% would indicate distributed coverage.","",
              "### Frequency distribution","",f"- Mean {distribution['mean']:.4%}; median {distribution['median']:.4%}; p5 {distribution['p5']:.4%}; p25 {distribution['p25']:.4%}; p75 {distribution['p75']:.4%}; p95 {distribution['p95']:.4%}; max {distribution['max']:.4%}.",
              f"- Above 1% / 2% / 5%: {distribution['above_1pct']} / {distribution['above_2pct']} / {distribution['above_5pct']}.","",
              "### Top 25 survivors","","| Organism | Name | Questions | Documents | Frequency |","|---|---|---:|---:|---:|"]
    report += [f"| {x['organism_id']} | {x['name_normalized'].replace('|','/')} | {x['n_questions']} | {len(docsets[x['organism_id']]):,} | {float(x['corpus_doc_frequency']):.3%} |" for x in top25]
    report += ["","### Out-of-domain check","",f"- Source: `{ood_source}`; match rate: {ood_matched}/{ood_total} ({ood_matched/ood_total if ood_total else 0:.2%}).",
               "- The fallback is short-form and should be interpreted cautiously." if ood_source=="wmdp_nonbio_questions" else "- User-supplied OOD corpus used.","",
               "## Core/periphery partition","",f"- Core: {tiers['core']:,}; periphery: {tiers['periphery']:,}; review: {tiers['review']:,}.",
               f"- Partition provisional (no agent list): **{str(agent_names is None).lower()}**.",
               "- Alternative core sizes by recurrence cutoff: " + ", ".join(f"{k}={v}" for k,v in alternatives.items()) + ".", "",
               "## Chapter scaffold","",f"- Chapters: {len(chapters):,}; control abstract token target: {control_tokens:,}; assigned tokens: {sum(x['target_tokens'] for x in chapters):,}.",
               f"- Chapter size range: {min((x['n_organisms'] for x in chapters),default=0)}–{max((x['n_organisms'] for x in chapters),default=0)} organisms.","",
               "## Thresholds (reported without verdict)","",f"- Core writable range: 30–150; current: {tiers['core']:,}.",f"- Review escalation threshold: above 100; current: {tiers['review']:,}.",f"- Coherent chapter range: 8–25; current: {len(chapters):,}.",f"- OOD specificity guides: below 5% supports specificity, above 30% suggests spurious matching; observed: {ood_matched/ood_total if ood_total else 0:.2%}.","",
               "## Reproducibility","","- Deterministic ordering and `random.Random(42)`.","- Corpus scanned once over `title`, `abstract`, and `doi`; only bounded contexts retained.","- Token budget uses the local Zephyr tokenizer over control-row abstracts only.","- No model or API calls were made.","","### Input SHA-256",""]
    report += [f"- `{path}`: `{digest}`" for path,digest in sorted(hashes.items())]
    report.append(""); (out/"spec03s_report.md").write_text("\n".join(report),encoding="utf-8")
    if len(curve)!=979 or any(int(curve[i]["cumulative_docs"])>int(curve[i+1]["cumulative_docs"]) for i in range(978)): fail("coverage acceptance assertion failed")
    if len(partition)!=979 or sum(tiers.values())!=979: fail("partition acceptance assertion failed")
    assigned=[oid for x in chapters for oid in x["organism_ids"]]
    if sorted(assigned)!=sorted(x["organism_id"] for x in core) or len(assigned)!=len(set(assigned)): fail("core chapter assignment assertion failed")
    if any(x["n_organisms"]<3 for x in chapters): fail("chapter minimum size assertion failed")
    if sum(x["target_tokens"] for x in chapters)!=control_tokens: fail("token allocation assertion failed")
    if len(context_rows)!=60: fail("context sample row count assertion failed")
    if hashes != {str(x):sha256(x) for x in all_inputs}: fail("input changed during run")
    if any(not (out/x).is_file() for x in OUTPUTS): fail("one or more required outputs missing")
    print(json_compact({"status":"ok","survivors":979,"core":tiers["core"],"periphery":tiers["periphery"],"review":tiers["review"],"chapters":len(chapters)}))


if __name__ == "__main__":
    main()
