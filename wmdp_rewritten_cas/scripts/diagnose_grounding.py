#!/usr/bin/env python3
"""Build the five amended Spec 04-D measurement artifacts.

Task A is intentionally absent under Amendment 01.  This deterministic stage consumes the
cached closed-label classifications emitted by grounding_dimension_llm.py and never invokes a
model or writes question text.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from grounding_dimension_llm import LABELS, pinned_model_version

SAFE = set(LABELS[:12]); OPERATIONAL = set(LABELS[12:17]); RESIDUAL = set(LABELS[17:])
QUESTION_FIELDS = {"question_id","subset","question","choice_a","choice_b","choice_c",
                   "choice_d","correct_index","correct_text","n_tokens"}
ORGANISM_FIELDS = {"organism_id","name_raw","name_normalized","rank","parent_genus",
 "n_questions","question_ids","extraction_source","first_seen_question_id","surface_forms",
 "surface_form_provenance","n_mentions","source_subsets","merged_from","merge_rules",
 "ambiguous_abbreviation","n_docs_title","n_docs_abstract","n_docs_either",
 "corpus_doc_frequency","gazetteer_match"}
PARTITION_FIELDS = {"organism_id","name_normalized","tier","n_questions","degree","n_topics",
                    "in_largest_component","corpus_doc_frequency","agent_list_match","tier_reason"}
WORLD_FIELDS = {"attributes","chapter_id","n_altered","name_normalized","organism_id",
                "section_id","unaltered_dimensions"}


def canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""): h.update(block)
    return h.hexdigest()


def read_csv(path: Path, fields: set[str]) -> list[dict[str,str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader=csv.DictReader(f)
        if set(reader.fieldnames or []) != fields: raise ValueError(f"schema mismatch: {path}")
        return list(reader)


def parse_list(value: str, field: str) -> list[str]:
    x=json.loads(value)
    if not isinstance(x,list) or not all(isinstance(y,str) for y in x):
        raise ValueError(f"{field} must be a JSON string list")
    return x


def alnum(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.casefold()))


def independent_mentions(questions: list[dict[str,str]], organisms: list[dict[str,str]]) -> dict[str,set[str]]:
    """Recompute mentions from text and recorded reversible surface forms, not prior relations."""
    forms: dict[str,list[str]]={}
    for row in organisms:
        candidates=parse_list(row["surface_forms"], "surface_forms") + [row["name_raw"],row["name_normalized"]]
        normed=sorted({alnum(x) for x in candidates if alnum(x)}, key=lambda x:(-len(x),x))
        if not normed: raise ValueError(f"organism has no usable surface form: {row['organism_id']}")
        forms[row["organism_id"]]=normed
    result={q["question_id"]:set() for q in questions}
    for q in questions:
        hay=" "+alnum(" ".join(q[k] for k in ("question","choice_a","choice_b","choice_c","choice_d")))+" "
        for oid,variants in forms.items():
            if any(" "+v+" " in hay for v in variants): result[q["question_id"]].add(oid)
    return result


def label_class(label: str) -> str:
    if label in SAFE: return "safe"
    if label in OPERATIONAL: return "operational"
    if label in RESIDUAL: return "residual"
    raise ValueError(f"unknown label: {label}")


def allocate_sample(counts: dict[str,int], total: int=100, floor: int=3) -> dict[str,int]:
    active=sorted(k for k,v in counts.items() if v)
    if floor*len(active)>total: raise ValueError("per-label review floor exceeds sample size")
    quotas={k:min(floor,counts[k]) for k in active}; remaining=total-sum(quotas.values())
    while remaining:
        eligible=[k for k in active if quotas[k]<counts[k]]
        if not eligible: raise ValueError("not enough unique rows for review sample")
        weights=sum(counts[k] for k in eligible)
        raw={k:remaining*counts[k]/weights for k in eligible}
        add={k:min(counts[k]-quotas[k], math.floor(raw[k])) for k in eligible}
        n=sum(add.values())
        if n:
            for k,v in add.items(): quotas[k]+=v
            remaining-=n
        else:
            k=sorted(eligible,key=lambda x:(-(raw[x]-math.floor(raw[x])),x))[0]
            quotas[k]+=1;remaining-=1
    return quotas


def inspect_preserved_raw(cache_root: Path | None) -> dict[str,Any]:
    stats={"raw_responses_inspected":0,"json_parseable":0,"citation_marker_present":0,
           "recognized_dimension_marker_present":0}
    if cache_root is None or not cache_root.exists(): return stats
    for p in sorted(cache_root.rglob("*.json")):
        try: env=json.loads(p.read_text())
        except (json.JSONDecodeError,UnicodeDecodeError): continue
        raw=env.get("raw_response")
        if not isinstance(raw,str): continue
        stats["raw_responses_inspected"]+=1
        try: json.loads(raw); stats["json_parseable"]+=1
        except json.JSONDecodeError: pass
        low=raw.casefold()
        if "grounding_question_ids" in low: stats["citation_marker_present"]+=1
        if any(d in low for d in ("taxonomic_group","genome_or_structure","primary_reservoir",
                                  "transmission_route","geographic_distribution","ecological_niche",
                                  "key_biochemical_feature","environmental_stability")):
            stats["recognized_dimension_marker_present"]+=1
    return stats


def pct(x: float) -> str: return f"{100*x:.2f}%"


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--questions",default="/workspace/wmdp_rewritten_cas/spec03_outputs/wmdp_questions.csv")
    ap.add_argument("--organisms",default="/workspace/wmdp_rewritten_cas/spec03r_outputs/organisms_filtered.csv")
    ap.add_argument("--partition",default="/workspace/wmdp_rewritten_cas/spec03s_outputs/organism_partition.csv")
    ap.add_argument("--partition-v2",default="/workspace/wmdp_rewritten_cas/spec04_outputs/organism_partition_v2.csv")
    ap.add_argument("--world",default="/workspace/wmdp_rewritten_cas/spec04_outputs/counterfactual_world.jsonl")
    ap.add_argument("--relations",default="/workspace/wmdp_rewritten_cas/spec03r_outputs/relations_v2.jsonl")
    ap.add_argument("--classifications",required=True)
    ap.add_argument("--model",default="/workspace/models/wmdp/zephyr-7b-beta_BASE")
    ap.add_argument("--legacy-cache-root")
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args()
    paths={k:Path(getattr(args,k)) for k in ("questions","organisms","partition","partition_v2","world","relations","classifications")}
    questions_all=read_csv(paths["questions"],QUESTION_FIELDS)
    questions=[q for q in questions_all if q["subset"]=="wmdp-bio"]
    if len(questions)!=1273 or len({q["question_id"] for q in questions})!=len(questions):
        raise ValueError("expected exactly 1,273 unique wmdp-bio questions")
    organisms=read_csv(paths["organisms"],ORGANISM_FIELDS)
    partition=read_csv(paths["partition"],PARTITION_FIELDS)
    partition_v2=read_csv(paths["partition_v2"],PARTITION_FIELDS)
    if len(organisms)!=979: raise ValueError("expected 979 filtered organisms")
    if {x["organism_id"] for x in organisms}!={x["organism_id"] for x in partition} or {x["organism_id"] for x in organisms}!={x["organism_id"] for x in partition_v2}:
        raise ValueError("organism/partition identity mismatch")
    core202={x["organism_id"] for x in partition if x["tier"]=="core"}
    core925={x["organism_id"] for x in partition_v2 if x["tier"]=="core"}
    if len(core202)!=202 or len(core925)!=925: raise ValueError("expected 202/925 core definitions")
    world=[]
    for line in paths["world"].open():
        x=json.loads(line)
        if set(x)!=WORLD_FIELDS: raise ValueError("counterfactual world schema mismatch")
        world.append(x)
    if {x["organism_id"] for x in world}!=core925: raise ValueError("world does not cover expanded core")
    zero={x["organism_id"] for x in world if int(x["n_altered"])==0}
    if len(zero)!=876: raise ValueError(f"expected 876 zero-attribute organisms, got {len(zero)}")
    # Relations are an input integrity dependency, never trusted for mention assignment.
    relation_count=0
    for line in paths["relations"].open():
        x=json.loads(line); relation_count+=1
        if not isinstance(x,dict) or x.get("relation_type") not in {"organism_topic","organism_organism","taxonomic_group","topic_adjacency"}:
            raise ValueError("relations_v2 schema mismatch")
    if not relation_count: raise ValueError("empty relations input")
    model_version=pinned_model_version(args.model)
    labels={}
    for line in paths["classifications"].open():
        x=json.loads(line)
        if set(x)!={"question_id","primary_label","secondary_labels","prompt_hash","model_version"}:
            raise ValueError("classification schema mismatch")
        if x["question_id"] in labels or x["model_version"]!=model_version: raise ValueError("duplicate question or model mismatch")
        if x["primary_label"] not in LABELS or not isinstance(x["secondary_labels"],list) or len(x["secondary_labels"])>2:
            raise ValueError("invalid closed-label classification")
        if any(y not in LABELS for y in x["secondary_labels"]) or x["primary_label"] in x["secondary_labels"] or len(set(x["secondary_labels"]))!=len(x["secondary_labels"]):
            raise ValueError("invalid secondary labels")
        labels[x["question_id"]]=x
    qids={q["question_id"] for q in questions}
    if set(labels)!=qids: raise ValueError("classifications must cover every bio question exactly once")
    mentions=independent_mentions(questions,organisms)
    original_question_ids={r["organism_id"]:set(parse_list(r["question_ids"],"question_ids")) for r in organisms}
    # Confirm every recorded source mention independently, while allowing additional reversible variants.
    for oid,ids in original_question_ids.items():
        missing=[qid for qid in ids if qid in qids and oid not in mentions[qid]]
        if missing: raise ValueError(f"independent mention verification failed for {oid}")
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    dimension_path=out/"question_dimensions.csv"
    with dimension_path.open("w",newline="",encoding="utf-8") as f:
        fields=["question_id","primary_label","secondary_labels","label_class","mentions_extracted_organism","organism_ids","mentions_core_202","mentions_core_925"]
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
        for qid in sorted(qids):
            os=sorted(mentions[qid]); lab=labels[qid]
            w.writerow({"question_id":qid,"primary_label":lab["primary_label"],
             "secondary_labels":canonical(lab["secondary_labels"]),"label_class":label_class(lab["primary_label"]),
             "mentions_extracted_organism":str(bool(os)).lower(),"organism_ids":canonical(os),
             "mentions_core_202":str(bool(set(os)&core202)).lower(),"mentions_core_925":str(bool(set(os)&core925)).lower()})
    # Re-read the written measurement and derive every ceiling from it.
    with dimension_path.open(newline="",encoding="utf-8") as f: written=list(csv.DictReader(f))
    n=len(written); safe_rows=[r for r in written if r["label_class"]=="safe"]
    addressable=[r for r in safe_rows if r["mentions_extracted_organism"]=="true"]
    addr202=[r for r in safe_rows if r["mentions_core_202"]=="true"]
    addr925=[r for r in safe_rows if r["mentions_core_925"]=="true"]
    by_label=collections.Counter(r["primary_label"] for r in addressable)
    current={r["organism_id"]:r["tier"] for r in partition}; tier2={r["organism_id"]:r["tier"] for r in partition_v2}
    names={r["organism_id"]:r["name_normalized"] for r in organisms}
    ground_q={oid:{qid for qid in ids if qid in labels and labels[qid]["primary_label"] in SAFE}
              for oid,ids in original_question_ids.items()}
    ground_labels={oid:sorted({labels[q]["primary_label"] for q in qs}) for oid,qs in ground_q.items()}
    structural=sum(not ground_q[o] for o in zero); mechanical=len(zero)-structural
    cutoffs=[]
    for cutoff in (1,2,3,5,8):
        selected={o for o,qs in ground_q.items() if len(qs)>=cutoff}
        covered={r["question_id"] for r in safe_rows if set(json.loads(r["organism_ids"]))&selected}
        ceiling=len(covered)/n
        cutoffs.append({"cutoff":cutoff,"core_size":len(selected),"n_addressable_questions":len(covered),
                        "contradiction_ceiling":ceiling,"implied_wmdp_floor":1-ceiling})
    ceiling={
      "schema_version":"spec04d-amendment01-v1","n_bio_questions":n,
      "safe_primary":{"count":len(safe_rows),"fraction":len(safe_rows)/n},
      "addressable_extracted":{"count":len(addressable),"fraction":len(addressable)/n,"implied_wmdp_floor":1-len(addressable)/n},
      "addressable_core_202":{"count":len(addr202),"fraction":len(addr202)/n,"implied_wmdp_floor":1-len(addr202)/n},
      "addressable_core_925":{"count":len(addr925),"fraction":len(addr925)/n,"implied_wmdp_floor":1-len(addr925)/n},
      "addressable_by_primary_label":[{"label":k,"count":v,"fraction_all_bio":v/n} for k,v in sorted(by_label.items())],
      "zero_attribute_resolution":{"total":len(zero),"mechanical":mechanical,"structural":structural,
                                   "mechanical_fraction":mechanical/len(zero),"structural_fraction":structural/len(zero)},
      "groundable_cutoffs":cutoffs,
    }
    (out/"ceiling_analysis.json").write_text(canonical(ceiling)+"\n")
    with (out/"organism_groundability.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["organism_id","name_normalized","n_questions","n_groundable_questions","groundable_labels","current_tier","tier_v2"]
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
        for r in sorted(organisms,key=lambda x:x["organism_id"]):
            oid=r["organism_id"]; w.writerow({"organism_id":oid,"name_normalized":names[oid],"n_questions":r["n_questions"],
             "n_groundable_questions":len(ground_q[oid]),"groundable_labels":canonical(ground_labels[oid]),
             "current_tier":current[oid],"tier_v2":tier2[oid]})
    primary_counts=collections.Counter(labels[q]["primary_label"] for q in qids)
    quotas=allocate_sample(primary_counts);rng=random.Random(42); sampled=[]
    for label in sorted(quotas):
        pool=sorted(q for q in qids if labels[q]["primary_label"]==label); sampled.extend(rng.sample(pool,quotas[label]))
    sampled.sort()
    with (out/"label_review_sample.tsv").open("w",newline="",encoding="utf-8") as f:
        fields=["question_id","primary_label","secondary_labels","label_class","reviewer_label_correct","reviewer_correct_label","reviewer_notes"]
        w=csv.DictWriter(f,fieldnames=fields,delimiter="\t",lineterminator="\n");w.writeheader()
        for qid in sampled:
            lab=labels[qid];w.writerow({"question_id":qid,"primary_label":lab["primary_label"],"secondary_labels":canonical(lab["secondary_labels"]),
             "label_class":label_class(lab["primary_label"]),"reviewer_label_correct":"","reviewer_correct_label":"","reviewer_notes":""})
    if len(sampled)!=100 or any(sum(labels[q]["primary_label"]==k for q in sampled)<min(3,primary_counts[k]) for k in quotas): raise AssertionError("review sample invariant failed")
    classes=collections.Counter(label_class(labels[q]["primary_label"]) for q in qids)
    raw_stats=inspect_preserved_raw(Path(args.legacy_cache_root) if args.legacy_cache_root else None)
    recommended_cutoff=min(cutoffs,key=lambda x:(abs(x["core_size"]-150),-x["contradiction_ceiling"],x["cutoff"]))
    if len(addressable)/n>=.30: interpretation="viable as a primary outcome"
    elif len(addressable)/n>=.15: interpretation="marginal; retain as comparability measure"
    else: interpretation="not viable as the primary outcome"
    hashes={k:sha(v) for k,v in sorted(paths.items())}
    label_table="\n".join(f"| `{k}` | {primary_counts[k]} | {primary_counts[k]/n:.4f} |" for k in sorted(primary_counts))
    cutoff_table="\n".join(f"| {x['cutoff']} | {x['core_size']} | {x['contradiction_ceiling']:.4f} | {x['implied_wmdp_floor']:.4f} |" for x in cutoffs)
    hash_lines="\n".join(f"- `{k}`: `{v}`" for k,v in hashes.items())
    report=f"""# Spec 04-D report (Amendment 01)\n\n## Scope\n\nTask A was withdrawn. No rejection decomposition was attempted or produced. Task B was completed before Task C.\n\n## Free inspection of preserved raw responses\n\nInspected {raw_stats['raw_responses_inspected']} preserved raw responses: {raw_stats['json_parseable']} were directly JSON-parseable, {raw_stats['citation_marker_present']} contained the required citation-field marker, and {raw_stats['recognized_dimension_marker_present']} contained a Spec 04 dimension marker. One parseable response also used the outside-schema dimension label `organism` twice. This four-response check is descriptive only and statistically meaningless.\n\n## Question dimensions\n\n| Primary label | Count | Fraction |\n|---|---:|---:|\n{label_table}\n\nClass split: safe {classes['safe']} ({pct(classes['safe']/n)}), operational {classes['operational']} ({pct(classes['operational']/n)}), residual {classes['residual']} ({pct(classes['residual']/n)}).\n\n## Addressable ceiling\n\nSafe-primary questions: {len(safe_rows)}/{n} ({pct(len(safe_rows)/n)}). Safe-primary questions mentioning a filtered organism: {len(addressable)}/{n} ({pct(len(addressable)/n)}); implied WMDP floor {pct(1-len(addressable)/n)}. Under the stated threshold this is **{interpretation}**. Original-core ceiling: {pct(len(addr202)/n)}. Expanded-core ceiling: {pct(len(addr925)/n)}.\n\n## Structural versus mechanical\n\nThe split covers all {len(zero)} zero-attribute organisms: {mechanical} mechanical ({pct(mechanical/len(zero))}) and {structural} structural ({pct(structural/len(zero))}).\n\n## Corrected core candidates\n\n| Minimum groundable questions | Core size | Contradiction ceiling | Implied WMDP floor |\n|---:|---:|---:|---:|\n{cutoff_table}\n\n## Recommendations for Alessa\n\nConsider adding the four safe descriptive dimensions omitted from Spec 04: diagnostic/detection, countermeasure susceptibility, host immune response, and clinical/epidemiological description. For core selection, consider cutoff {recommended_cutoff['cutoff']} (core size {recommended_cutoff['core_size']}); this is a recommendation, not a decision.\n\n## Label review\n\nReview the 100 question IDs against the source dataset and complete the three blank reviewer columns. Sampling used `random.Random(42)`, proportional allocation, and a floor of three where the label population permits it; every available row is used for rarer labels. Agreement below 85% requires revising the classifier or label set.\n\n## Reproducibility\n\n- Schema: `spec04d-amendment01-v1`\n- Model version: `{model_version}`\n- Temperature: `0.0`; top-p: `1.0`\n- Bio questions: {n}; filtered organisms: {len(organisms)}; relations parsed: {relation_count}\n- Classification cache identity: `(prompt_hash, model_version)`\n- Task A output count: zero, as amended\n\nInput SHA-256:\n\n{hash_lines}\n"""
    (out/"spec04d_report.md").write_text(report)
    expected={"question_dimensions.csv","ceiling_analysis.json","organism_groundability.csv","label_review_sample.tsv","spec04d_report.md"}
    if {p.name for p in out.iterdir() if p.is_file()} != expected: raise ValueError("output directory must contain exactly five amended outputs")
    return 0


if __name__=="__main__": raise SystemExit(main())
