#!/usr/bin/env python3
"""Deterministically characterize an entity-swap parquet corpus (offline only)."""

from __future__ import annotations

import argparse
import csv
import difflib
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_COLUMNS = ["title", "abstract", "text", "doi"]
STOPWORDS = frozenset(
    "a an and are as at be been being but by can could did do does for from had has "
    "have he her hers him his i if in into is it its may might must nor not of on or "
    "our ours she should than that the their theirs them they this those to was we "
    "were what when where which who whom why will with would you your yours".split()
)
TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
NUMERIC_RE = re.compile(
    r"^[+-]?(?:\d+(?:[.,]\d+)*|\.\d+)(?:\s*[a-zA-Zµμ%°/^-][\wµμ%°/^.-]*)?$"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--tokenizer", default="HuggingFaceH4/zephyr-7b-beta")
    p.add_argument("--limit", type=int)
    return p.parse_args()


def norm_ws(value):
    return " ".join(value.split())


def norm_term(value):
    value = norm_ws(value).lower()
    value = re.sub(r"^(?:[^\w]+)|(?:[^\w]+)$", "", value, flags=re.UNICODE)
    value = re.sub(r"(?:['’]s)\Z", "", value)
    return norm_ws(value)


def tokens_with_offsets(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def percentile(values, p):
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def dist(values, thresholds=()):
    out = {
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p5": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }
    for threshold in thresholds:
        n = sum(v > threshold for v in values)
        out[f"count_over_{threshold}"] = n
        out[f"fraction_over_{threshold}"] = n / len(values) if values else 0.0
    return out


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reproducible_timestamp(files):
    """Return a stable UTC timestamp for byte-identical reruns."""
    default_epoch = max(p.stat().st_mtime_ns for p in files) // 1_000_000_000
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", default_epoch))
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def json_dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonl_dump(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def get_token_counter(identifier):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(identifier, local_files_only=True)
        return lambda s: len(tok.encode(s, add_special_tokens=False)), identifier
    except Exception:
        return lambda s: len(s.split()), "whitespace_fallback"


def read_rows(files, root, limit):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("pyarrow is required to read parquet; install it from an offline source") from e
    rows, per_file = [], []
    remaining = limit
    for path in files:
        pf = pq.ParquetFile(path)
        observed = pf.schema_arrow
        names = observed.names
        types_ok = len(names) == 4 and all(pa.types.is_string(observed.field(n).type) for n in names)
        if names != EXPECTED_COLUMNS or not types_ok:
            print(f"Observed schema for {path}:\n{observed}", file=sys.stderr)
            print("Expected schema: title:string, abstract:string, text:string, doi:string", file=sys.stderr)
            raise SystemExit(2)
        table = pf.read()
        raw = table.to_pylist()
        take = len(raw) if remaining is None else min(remaining, len(raw))
        rel = path.relative_to(root).as_posix()
        for record in raw[:take]:
            record["source_file"] = rel
            rows.append(record)
        per_file.append({"relative_path": rel, "rows": take})
        if remaining is not None:
            remaining -= take
            if remaining == 0:
                break
    return rows, per_file


def trivial_reason(source, target, sn, tn):
    if sn == tn:
        return "normalized_equal"
    if sn in STOPWORDS and tn in STOPWORDS and " " not in sn and " " not in tn:
        return "function_words"
    compact_s, compact_t = re.sub(r"[-\s]", "", sn), re.sub(r"[-\s]", "", tn)
    if compact_s == compact_t:
        return "hyphenation_or_casing"
    singular_s = re.sub(r"(?:es|s)$", "", sn)
    singular_t = re.sub(r"(?:es|s)$", "", tn)
    if singular_s == singular_t:
        return "pluralization"
    if difflib.SequenceMatcher(None, sn, tn, autojunk=False).ratio() > 0.85:
        return "high_string_similarity"
    return None


def category(source, target, sn, tn):
    reason = trivial_reason(source, target, sn, tn)
    if reason:
        return "trivial", reason
    if NUMERIC_RE.fullmatch(sn) and NUMERIC_RE.fullmatch(tn):
        return "numeric", None
    source_words = re.findall(r"\b\w+\b", source, re.UNICODE)
    upper = any(w and w[0].isupper() for w in source_words)
    if upper or len(source_words) > 1 or (len(sn) >= 4 and sn not in STOPWORDS):
        return "entity_like", None
    return "other", None


def extract(row_id, doi, abstract, text):
    aa, bb = tokens_with_offsets(abstract), tokens_with_offsets(text)
    sm = difflib.SequenceMatcher(None, [x[0] for x in aa], [x[0] for x in bb], autojunk=False)
    substitutions, inserts, deletes = [], 0, 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            inserts += 1
        elif tag == "delete":
            deletes += 1
        elif tag == "replace":
            if i1 == i2:
                inserts += 1
                continue
            if j1 == j2:
                deletes += 1
                continue
            start, end = aa[i1][1], aa[i2 - 1][2]
            ts, te = bb[j1][1], bb[j2 - 1][2]
            source, target = abstract[start:end], text[ts:te]
            sn, tn = norm_term(source), norm_term(target)
            cat, reason = category(source, target, sn, tn)
            substitutions.append({
                "row_id": row_id, "doi": doi, "source_raw": source, "target_raw": target,
                "source_norm": sn, "target_norm": tn, "source_char_start": start,
                "context_left": abstract[max(0, start - 40):start],
                "context_right": abstract[end:end + 40], "category": cat, "trivial_reason": reason,
            })
    for ordinal, sub in enumerate(substitutions):
        sub["sub_id"] = f"{row_id}-{ordinal}"
    return substitutions, inserts, deletes


def modal(counter):
    return min(counter, key=lambda k: (-counter[k], k))


def build_mapping(subs):
    grouped, docs, cats = defaultdict(Counter), defaultdict(set), defaultdict(Counter)
    for s in subs:
        grouped[s["source_norm"]][s["target_norm"]] += 1
        docs[s["source_norm"]].add(s["row_id"])
        cats[s["source_norm"]][s["category"]] += 1
    modal_targets = {}
    for source, counts in grouped.items():
        if sum(counts.values()) >= 3:
            modal_targets[source] = modal(counts)
    collisions = Counter(modal_targets.values())
    rows = []
    for source, counts in grouped.items():
        total, mt = sum(counts.values()), modal(counts)
        cat = min(cats[source], key=lambda k: (-cats[source][k], k))
        rows.append({
            "source_norm": source, "total_occurrences": total,
            "n_distinct_targets": len(counts),
            "targets": [{"target_norm": k, "count": counts[k]} for k in sorted(counts, key=lambda k: (-counts[k], k))],
            "modal_target": mt, "modal_share": counts[mt] / total,
            "n_documents": len(docs[source]),
            "is_clean_1to1": total >= 3 and counts[mt] / total >= .9 and collisions[mt] == 1,
            "category": cat,
        })
    return sorted(rows, key=lambda x: (-x["total_occurrences"], x["source_norm"]))


def consistency(subs, doc_ids):
    mapping = build_mapping(subs)
    pop = [x for x in mapping if x["total_occurrences"] >= 2]
    covered = sum(x["total_occurrences"] for x in pop)
    weighted = sum(round(x["modal_share"] * x["total_occurrences"]) for x in pop) / covered if covered else None
    histogram = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
    for x in pop:
        k = str(x["n_distinct_targets"]) if x["n_distinct_targets"] < 5 else "5+"
        histogram[k] += 1
    by_doc_source = defaultdict(lambda: defaultdict(set))
    reverse = defaultdict(set)
    for s in subs:
        by_doc_source[s["row_id"]][s["source_norm"]].add(s["target_norm"])
        reverse[s["target_norm"]].add(s["source_norm"])
    bad_docs = {d for d in doc_ids if any(len(v) > 1 for v in by_doc_source[d].values())}
    collision_targets = sum(len(v) > 1 for v in reverse.values())
    return {
        "occurrence_weighted_modal_share": weighted,
        "unweighted_mean_modal_share": statistics.fmean(x["modal_share"] for x in pop) if pop else None,
        "n_distinct_targets_histogram": histogram,
        "fraction_source_terms_one_target": sum(x["n_distinct_targets"] == 1 for x in pop) / len(pop) if pop else None,
        "n_source_terms": len(pop), "total_occurrences_covered": covered,
        "within_document_inconsistent_count": len(bad_docs),
        "within_document_inconsistent_rate": len(bad_docs) / len(doc_ids) if doc_ids else 0.0,
        "target_collision_count": collision_targets,
        "target_collision_rate": collision_targets / len(reverse) if reverse else 0.0,
        "top_30": mapping[:30],
    }


def threshold_yield(mapping):
    result = []
    for occurrences in (2, 3, 5, 10):
        eligible = [x for x in mapping if x["total_occurrences"] >= occurrences]
        modal_collision = Counter(x["modal_target"] for x in eligible)
        for share in (.8, .9, 1.0):
            count = sum(x["modal_share"] >= share and modal_collision[x["modal_target"]] == 1 for x in eligible)
            result.append({"min_occurrences": occurrences, "min_modal_share": share, "count": count})
    return result


def md_escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(summary, mapping, out_dir):
    inv, lengths = summary["inventory"], summary["lengths"]
    ca, cd, leaks, claim = summary["consistency_all_rows"], summary["consistency_deduped"], summary["residual_leakage"], summary["claim_yield"]
    lines = [
        "# Parquet corpus characterization", "",
        f"Inventory: {inv['n_parquet_files']} files, {inv['total_rows']} rows, {inv['deduplicated_rows']} deduplicated rows.", "",
        "## Headline measurements", "",
        f"- Identical-pair rate: {inv['identical_pair_rate']:.6f}",
        f"- Occurrence-weighted modal share (all rows): {ca['occurrence_weighted_modal_share']}",
        f"- Occurrence-weighted modal share (deduplicated): {cd['occurrence_weighted_modal_share']}",
        f"- Within-document inconsistency rate: {ca['within_document_inconsistent_rate']:.6f}",
        f"- Residual leakage rate: {leaks['document_rate']:.6f}",
        f"- Clean 1:1 claim-pair count: {claim['clean_1to1_count']}", "",
        "## Lengths", "", "| text | mean | median | p5 | p95 | min | max | >512 count | >512 fraction |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for field in ("abstract", "text"):
        x = lengths[field]
        lines.append(f"| {field} | {x['mean']:.3f} | {x['median']} | {x['p5']} | {x['p95']} | {x['min']} | {x['max']} | {x['count_over_512']} | {x['fraction_over_512']:.6f} |")
    lines += ["", f"Documents with either side exceeding 512 tokens: {lengths['documents_over_512']}.", "", "## Substitutions", ""]
    lines.append("Category counts: " + ", ".join(f"{k}={v}" for k, v in sorted(summary["substitutions"]["category_counts"].items())))
    lines.append("")
    lines.append("Substantive substitutions per document: " + json.dumps(summary["substitutions"]["substantive_per_document"], sort_keys=True))
    lines += ["", "## Top 30 source terms", "", "| source | occurrences | targets | modal share | documents |", "|---|---:|---|---:|---:|"]
    for x in ca["top_30"]:
        targets = "; ".join(f"{t['target_norm']} ({t['count']})" for t in x["targets"])
        lines.append(f"| {md_escape(x['source_norm'])} | {x['total_occurrences']} | {md_escape(targets)} | {x['modal_share']:.4f} | {x['n_documents']} |")
    lines += ["", "## Collisions", "", f"All rows: {ca['target_collision_count']} targets with in-degree >1 ({ca['target_collision_rate']:.6f}).", f"Deduplicated: {cd['target_collision_count']} targets with in-degree >1 ({cd['target_collision_rate']:.6f}).", "", "## Claim yield", "", "| minimum occurrences | minimum modal share | pairs |", "|---:|---:|---:|"]
    for x in claim["relaxed_thresholds"]:
        lines.append(f"| {x['min_occurrences']} | {x['min_modal_share']:.1f} | {x['count']} |")
    lines += ["", "## Manual review", "", "See `parquet_char_review_sample.tsv`. Review whether each swap is type-preserving (for example, virus → virus rather than virus → bacterium).", "", "## Decision thresholds and measurements", "",
        f"- Deduplicated occurrence-weighted modal share threshold bands: ≥0.90, 0.50–0.90, <0.50; measured: {cd['occurrence_weighted_modal_share']}",
        f"- Within-document inconsistency defect threshold: >0.10; measured: {ca['within_document_inconsistent_rate']:.6f}",
        f"- Residual leakage threshold: >0.10; measured: {leaks['document_rate']:.6f}",
        f"- Identical-pair threshold: >0.05; measured: {inv['identical_pair_rate']:.6f}",
        f"- Claim yield bands: ≥150, 60–150, <60; measured: {claim['clean_1to1_count']}", "", "## Reproducibility", "", "```json", json.dumps(summary["reproducibility"], indent=2, sort_keys=True), "```", ""]
    (out_dir / "parquet_char_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    args = parse_args()
    root, out_dir = args.corpus_root.resolve(), args.out_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"corpus root is not a directory: {root}")
    if out_dir == root or root in out_dir.parents:
        raise SystemExit("out-dir must not be corpus-root or a descendant of it")
    files = sorted(root.rglob("*.parquet"), key=lambda p: p.relative_to(root).as_posix())
    if not files:
        raise SystemExit("no parquet files found")
    file_meta = [{"relative_path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
    rows, per_file = read_rows(files, root, args.limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    count_tokens, tokenizer_used = get_token_counter(args.tokenizer)
    nulls, empties = Counter(), Counter()
    doi_counts, abstract_counts, text_counts = Counter(), Counter(), Counter()
    anomalies = {"control_character_rows": 0, "non_utf8_rows": 0}
    per_doc, all_subs, token_abstract, token_text, ratios = [], [], [], [], []
    for row_id, raw in enumerate(rows):
        for field in EXPECTED_COLUMNS:
            if raw[field] is None:
                nulls[field] += 1
            elif not isinstance(raw[field], str):
                raise SystemExit(f"malformed row {row_id}: {field} is not a string/null")
            elif raw[field] == "":
                empties[field] += 1
        title, abstract, text, doi = (raw[x] or "" for x in EXPECTED_COLUMNS)
        if any(CONTROL_RE.search(x) for x in (title, abstract, text, doi)):
            anomalies["control_character_rows"] += 1
        # Arrow strings are validated UTF-8 on read; encoding is an explicit sanity check.
        try:
            for x in (title, abstract, text, doi): x.encode("utf-8", "strict")
        except UnicodeError:
            anomalies["non_utf8_rows"] += 1
        na, nt = norm_ws(abstract), norm_ws(text)
        doi_counts[doi] += 1; abstract_counts[na] += 1; text_counts[nt] += 1
        at, tt = count_tokens(abstract), count_tokens(text)
        token_abstract.append(at); token_text.append(tt)
        ratio = tt / at if at else 0.0
        ratios.append(ratio)
        extracted, inserts, deletes = extract(row_id, doi, abstract, text)
        all_subs.extend(extracted)
        substantive = [x for x in extracted if x["category"] != "trivial"]
        trivial = len(extracted) - len(substantive)
        source_targets = defaultdict(set)
        for s in substantive: source_targets[s["source_norm"]].add(s["target_norm"])
        inconsistent = sorted(k for k, v in source_targets.items() if len(v) > 1)
        leaked = []
        normalized_rewritten = norm_ws(text)
        for source in sorted(source_targets):
            pattern = r"(?<!\w)" + re.escape(source).replace(r"\ ", r"\s+") + r"(?!\w)"
            if source and re.search(pattern, normalized_rewritten, re.IGNORECASE): leaked.append(source)
        flags = []
        if not na: flags.append("empty_abstract")
        if not nt: flags.append("empty_text")
        if at > 512 or tt > 512: flags.append("over_512_tokens")
        per_doc.append({
            "row_id": row_id, "doi": doi, "source_file": raw["source_file"],
            "abstract_tokens": at, "text_tokens": tt, "length_ratio": ratio,
            "identical": na == nt, "n_substitutions": len(substantive),
            "n_trivial_edits": trivial, "n_insertions": inserts, "n_deletions": deletes,
            "residual_leak_terms": leaked, "has_residual_leak": bool(leaked),
            "within_doc_inconsistent_terms": inconsistent, "flags": flags,
        })
    for d in per_doc:
        if doi_counts[d["doi"]] > 1: d["flags"].append("duplicate_doi")
        if abstract_counts[norm_ws(rows[d["row_id"]]["abstract"] or "")] > 1: d["flags"].append("duplicate_abstract")
        if text_counts[norm_ws(rows[d["row_id"]]["text"] or "")] > 1: d["flags"].append("duplicate_text")
        d["flags"].sort()
    substantive = [x for x in all_subs if x["category"] != "trivial"]
    mapping = build_mapping(substantive)
    mapping_by_source = {x["source_norm"]: x for x in mapping}
    unique_abstract_rows = []
    seen_hashes = set()
    for i, row in enumerate(rows):
        h = hashlib.sha256(norm_ws(row["abstract"] or "").encode()).hexdigest()
        if h not in seen_hashes: seen_hashes.add(h); unique_abstract_rows.append(i)
    dedup_ids = set(unique_abstract_rows)
    dedup_subs = [s for s in substantive if s["row_id"] in dedup_ids]
    leaking = [d for d in per_doc if d["has_residual_leak"]]
    leak_terms = Counter(t for d in leaking for t in d["residual_leak_terms"])
    clean = [x for x in mapping if x["is_clean_1to1"]]
    clean_sources = {x["source_norm"] for x in clean}
    clean_docs = defaultdict(set)
    for s in substantive:
        if s["source_norm"] in clean_sources: clean_docs[s["source_norm"]].add(s["row_id"])
    leak_doc_ids = {d["row_id"] for d in leaking}
    identical_count = sum(d["identical"] for d in per_doc)
    sub_counts = [d["n_substitutions"] for d in per_doc]
    category_counts = Counter(x["category"] for x in all_subs)
    summary = {
        "inventory": {
            "n_parquet_files": len(files), "files": file_meta, "rows_per_file": per_file,
            "total_rows": len(rows), "deduplicated_rows": len(dedup_ids),
            "null_counts": dict(nulls), "empty_string_counts": dict(empties),
            "duplicate_doi_count": sum(n - 1 for n in doi_counts.values() if n > 1),
            "dois_appearing_more_than_once": sum(n > 1 for n in doi_counts.values()),
            "duplicate_abstract_count": sum(n - 1 for n in abstract_counts.values() if n > 1),
            "duplicate_text_count": sum(n - 1 for n in text_counts.values() if n > 1),
            "identical_pair_count": identical_count,
            "identical_pair_rate": identical_count / len(rows) if rows else 0.0,
            "empty_or_null_abstract_rows": sum(not norm_ws(r["abstract"] or "") for r in rows),
            "empty_or_null_text_rows": sum(not norm_ws(r["text"] or "") for r in rows),
            "anomalies": anomalies,
        },
        "lengths": {"abstract": dist(token_abstract, (256, 512, 1024)), "text": dist(token_text, (256, 512, 1024)), "length_ratio": dist(ratios), "documents_over_512": sum(a > 512 or t > 512 for a, t in zip(token_abstract, token_text))},
        "substitutions": {"category_counts": dict(category_counts), "substantive_count": len(substantive), "substantive_per_document": {**dist(sub_counts), "zero_count": sum(x == 0 for x in sub_counts)}},
        "consistency_all_rows": consistency(substantive, set(range(len(rows)))),
        "consistency_deduped": consistency(dedup_subs, dedup_ids),
        "residual_leakage": {"document_count": len(leaking), "document_rate": len(leaking) / len(rows) if rows else 0.0, "mean_leaked_terms_among_leaking_documents": statistics.fmean(len(d["residual_leak_terms"]) for d in leaking) if leaking else 0.0, "top_20_terms": [{"source_norm": k, "documents": v} for k, v in sorted(leak_terms.items(), key=lambda x: (-x[1], x[0]))[:20]]},
        "claim_yield": {"clean_1to1_count": len(clean), "relaxed_thresholds": threshold_yield(mapping), "clean_pair_document_distribution": dist([x["n_documents"] for x in clean]), "clean_pair_category_counts": dict(Counter(x["category"] for x in clean)), "clean_pairs_entirely_leak_free": sum(not (clean_docs[x["source_norm"]] & leak_doc_ids) for x in clean)},
        "stopword_list": sorted(STOPWORDS), "limit": args.limit,
        "reproducibility": {"python_version": platform.python_version(), "pandas_version": package_version("pandas"), "pyarrow_version": package_version("pyarrow"), "difflib_version": "stdlib", "tokenizer_requested": args.tokenizer, "tokenizer_used": tokenizer_used, "utc_timestamp": reproducible_timestamp(files), "timestamp_basis": "SOURCE_DATE_EPOCH or newest input mtime", "input_sha256": {x["relative_path"]: x["sha256"] for x in file_meta}},
    }
    json_dump(out_dir / "parquet_char_summary.json", summary)
    jsonl_dump(out_dir / "parquet_char_per_doc.jsonl", per_doc)
    jsonl_dump(out_dir / "parquet_char_substitutions.jsonl", substantive)
    jsonl_dump(out_dir / "parquet_char_mapping_table.jsonl", mapping)
    rng = random.Random(42)
    clean_pool = [s for s in substantive if s["source_norm"] in clean_sources]
    inconsistent_pool = [s for s in substantive if mapping_by_source[s["source_norm"]]["n_distinct_targets"] >= 2]
    if len(substantive) < 50 or len(clean_pool) < 15 or len(inconsistent_pool) < 10:
        raise SystemExit(f"cannot form required review sample: substantive={len(substantive)}, clean={len(clean_pool)}, inconsistent={len(inconsistent_pool)}")
    chosen = rng.sample(clean_pool, 15)
    chosen_ids = {x["sub_id"] for x in chosen}
    inc_available = [x for x in inconsistent_pool if x["sub_id"] not in chosen_ids]
    need_inc = max(0, 10 - sum(x in inconsistent_pool for x in chosen))
    chosen += rng.sample(inc_available, need_inc); chosen_ids = {x["sub_id"] for x in chosen}
    chosen += rng.sample([x for x in substantive if x["sub_id"] not in chosen_ids], 50 - len(chosen))
    chosen.sort(key=lambda x: (x["row_id"], x["sub_id"]))
    columns = "sub_id doi source_raw target_raw context_left context_right n_distinct_targets modal_share category reviewer_type_preserving reviewer_notes".split()
    with (out_dir / "parquet_char_review_sample.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, columns, dialect="excel-tab", lineterminator="\n")
        writer.writeheader()
        for s in chosen:
            m = mapping_by_source[s["source_norm"]]
            writer.writerow({**{k: s.get(k, "") for k in columns}, "n_distinct_targets": m["n_distinct_targets"], "modal_share": m["modal_share"], "reviewer_type_preserving": "", "reviewer_notes": ""})
    render_report(summary, mapping, out_dir)


if __name__ == "__main__":
    main()
