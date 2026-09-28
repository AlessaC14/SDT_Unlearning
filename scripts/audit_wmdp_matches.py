#!/usr/bin/env python3
"""Offline, streaming audit of the saved 27 claim queries against cached WMDP."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
REVISION = "5a786ed1041e56fef2d4ed26c1239f12b73a68eb"


def normalize(value):
    return unicodedata.normalize("NFKC", value).casefold()


def field_text(abstract, text, field):
    if field == "abstract":
        return abstract
    return "\n\n".join(value for value in (abstract, text) if value.strip())


def matching_indices(value, queries):
    normalized = normalize(value)
    return [i for i, terms in enumerate(queries) if all(term in normalized for term in terms)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=ROOT / "wmdp_rewritten_cas/counterfactual_textbook_v1/counterfactual_cells.csv")
    parser.add_argument("--terms", type=Path, default=ROOT / "scripts/wmdp_claim_terms.json")
    parser.add_argument("--cache", type=Path, default=ROOT / f".cache/huggingface/datasets/cais___wmdp-bio-forget-corpus/default/0.0.0/{REVISION}")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/wmdp/zephyr-7b-beta_BASE/tokenizer.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/wmdp_claim_matches")
    args = parser.parse_args()
    import pyarrow as pa
    import tokenizers
    from tokenizers import Tokenizer

    with args.claims.open(newline="", encoding="utf-8") as handle:
        cells = list(csv.DictReader(handle))
    term_data = json.loads(args.terms.read_text(encoding="utf-8"))
    term_rows = term_data["claims"]
    if [row["cell_id"] for row in cells] != [row["cell_id"] for row in term_rows]:
        raise ValueError("Claim IDs/order differ from the saved term inventory")
    if any(not row["required_terms"] or any(not t for t in row["required_terms"]) for row in term_rows):
        raise ValueError("Empty retrieval terms")
    queries = [[normalize(t) for t in row["required_terms"]] for row in term_rows]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    fields = ("abstract", "abstract+text")
    summaries = {field: {"eligible_records": 0, "corpus_tokens": 0, "matched_unique_records": 0,
                         "matched_unique_tokens": 0, "claim_record_pairs": 0, "claim_pair_tokens": 0} for field in fields}
    counts = {field: [{"matches": 0, "tokens": 0} for _ in cells] for field in fields}
    shards = sorted(args.cache.glob("wmdp-bio-forget-corpus-train-*.arrow"))
    if not shards:
        raise FileNotFoundError("No cached corpus shards found")
    total_records = 0
    for shard in shards:
        with pa.memory_map(str(shard), "r") as handle:
            for batch in pa.ipc.open_stream(handle):
                for offset in range(0, batch.num_rows, 64):
                    records = batch.slice(offset, 64).select(["abstract", "text"]).to_pylist()
                    for field in fields:
                        texts = [field_text(row["abstract"] or "", row["text"] or "", field) for row in records]
                        eligible = [value for value in texts if value.strip()]
                        encodings = tokenizer.encode_batch(eligible, add_special_tokens=False)
                        for value, encoding in zip(eligible, encodings):
                            size = len(encoding.ids)
                            hits = matching_indices(value, queries)
                            summary = summaries[field]
                            summary["eligible_records"] += 1
                            summary["corpus_tokens"] += size
                            if hits:
                                summary["matched_unique_records"] += 1
                                summary["matched_unique_tokens"] += size
                            summary["claim_record_pairs"] += len(hits)
                            summary["claim_pair_tokens"] += size * len(hits)
                            for index in hits:
                                counts[field][index]["matches"] += 1
                                counts[field][index]["tokens"] += size
                    total_records += len(records)
                print(f"Audited {total_records:,} records", flush=True)
    expected = [row["saved_abstract_matches"] for row in term_rows]
    actual = [row["matches"] for row in counts["abstract"]]
    if actual != expected:
        raise ValueError(f"Abstract counts differ from saved report: {actual} != {expected}")
    info = json.loads((args.cache / "dataset_info.json").read_text())
    if total_records != info["splits"]["train"]["num_examples"]:
        raise ValueError("Audited record count differs from cached dataset metadata")
    results = []
    for index, (cell, terms) in enumerate(zip(cells, term_rows)):
        results.append({"cell_id": cell["cell_id"], "entity_name": cell["entity_name"],
                        "counterfactual_value": cell["counterfactual_value"], "required_terms": terms["required_terms"],
                        **{field: counts[field][index] for field in fields}})
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "cais/wmdp-bio-forget-corpus",
        "dataset_revision": REVISION, "records": total_records,
        "cache": str(args.cache), "shards": [{"path": str(p), "bytes": p.stat().st_size} for p in shards],
        "claims_path": str(args.claims), "claims_sha256": digest(args.claims),
        "terms_path": str(args.terms), "terms_sha256": digest(args.terms),
        "tokenizer": str(args.tokenizer), "tokenizer_sha256": digest(args.tokenizer),
        "tokenizers_version": tokenizers.__version__, "pyarrow_version": pa.__version__,
        "method": {"matching": "NFKC + casefold; ALL literal substrings; one hit per record per claim",
                   "abstract+text": "Nonblank abstract and text joined by two newlines; terms may occur across fields",
                   "eligibility": "Selected field must be nonblank; combined mode includes text-only records",
                   "tokenization": "Original unnormalized selected field; no special tokens, truncation, or padding",
                   "duplicates": "No DOI/content deduplication; unique totals count each corpus row once across claims",
                   "claim_pair_tokens": "Sum of per-claim matched tokens; multi-claim rows counted multiple times"},
        "saved_abstract_counts_reproduced": True, "summary": summaries, "claims": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "claims.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["cell_id", "entity", "required_terms", "abstract_matches", "abstract_tokens", "combined_matches", "combined_tokens"])
        for row in results:
            writer.writerow([row["cell_id"], row["entity_name"], json.dumps(row["required_terms"], ensure_ascii=False),
                             row["abstract"]["matches"], row["abstract"]["tokens"], row["abstract+text"]["matches"], row["abstract+text"]["tokens"]])
    lines = ["# WMDP claim matching and token audit", "", f"Generated: {report['created_utc']}", "",
             "Tokenizer: local Zephyr-7B tokenizer; no special tokens, truncation, or padding.", "",
             "Matching: saved terms, NFKC + casefold, ALL literal substrings. Combined field joins nonblank abstract and text with two newlines. Terms may be split across fields. Text-only records are eligible in combined mode.", "",
             "All 27 abstract match counts reproduce the saved retrieval report. No rewrites were run.", "",
             "## Totals", "", "| Metric | Abstract | Abstract + full text |", "|---|---:|---:|"]
    for key, label in [("eligible_records", "Eligible records"), ("corpus_tokens", "Entire eligible corpus tokens"),
                       ("matched_unique_records", "Unique matched records across claims"), ("matched_unique_tokens", "Unique matched-record tokens"),
                       ("claim_record_pairs", "Claim-record pairs"), ("claim_pair_tokens", "Tokens summed over claim-record pairs")]:
        lines.append(f"| {label} | {summaries['abstract'][key]:,} | {summaries['abstract+text'][key]:,} |")
    lines += ["", "Unique means corpus row IDs across claims; no DOI/content deduplication. Summing per-claim tokens counts multi-claim records repeatedly. Tokens cover the whole selected field, not just matching sentences or terms.", "",
              "## Every claim", "", "| Cell | Entity | Abstract matches | Abstract tokens | Combined matches | Combined tokens |", "|---|---|---:|---:|---:|---:|"]
    for row in results:
        a, b = row["abstract"], row["abstract+text"]
        lines.append(f"| `{row['cell_id']}` | {row['entity_name']} | {a['matches']:,} | {a['tokens']:,} | {b['matches']:,} | {b['tokens']:,} |")
    lines += ["", "See `claims.csv` for all required terms and `report.json` for input paths, hashes, library versions, and exact accounting rules."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
