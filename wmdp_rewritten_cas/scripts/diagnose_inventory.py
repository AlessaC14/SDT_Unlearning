#!/usr/bin/env python3
"""Spec 02-D: deterministic Arm 2 feasibility and inventory diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from characterize_parquet import (
    json_dump, jsonl_dump, norm_term, norm_ws, package_version, percentile,
    reproducible_timestamp, sha256, tokens_with_offsets,
)

COMMON_WORDS = frozenset("cell cells key keys test tests data model models system systems method methods study studies result results value values type types use uses used protein gene dna rna acid base time year years human process function virus host sample samples group groups control case cases state states code codes file files network".split())
INVENTORY_KEYS = {"claim_id", "source", "domain", "real_claim", "twin_claim", "real_answer", "twin_answer", "diff_span", "swap_type", "entity_pair", "claim_tokens", "n_supporting_docs", "provenance", "type_preserving", "flags"}
PROBE_KEYS = {"claim_id", "p_real", "p_real_real_first", "p_real_real_second", "a_log_probability_real_first", "b_log_probability_real_first", "a_log_probability_real_second", "b_log_probability_real_second"}
MAPPING_KEYS = {"source_norm", "target_norm", "source_surface_forms", "target_surface_forms", "total_occurrences", "n_documents", "modal_share", "swap_type", "collision_free"}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--s02-out", required=True, type=Path)
    p.add_argument("--corpus-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    return p.parse_args()


def read_jsonl(path, keys):
    if not path.is_file(): raise SystemExit(f"missing input: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try: row = json.loads(line)
            except json.JSONDecodeError as e: raise SystemExit(f"invalid JSON {path}:{n}: {e}") from e
            if not isinstance(row, dict) or set(row) != keys:
                raise SystemExit(f"schema mismatch {path}:{n}; observed={sorted(row) if isinstance(row,dict) else type(row).__name__}")
            rows.append(row)
    return rows


def exact(x, t): return type(x) is t
def str_list(x): return isinstance(x, list) and all(exact(v, str) for v in x)


def validate_inputs(inventory, probe, mappings):
    for n, x in enumerate(inventory, 1):
        scalar = {"claim_id":str,"source":str,"domain":str,"real_claim":str,"twin_claim":str,"real_answer":str,"twin_answer":str,"swap_type":str,"claim_tokens":int,"n_supporting_docs":int,"flags":list}
        if any(not exact(x[k], t) for k,t in scalar.items()) or not isinstance(x["diff_span"],dict) or not isinstance(x["entity_pair"],dict) or not isinstance(x["provenance"],dict):
            raise SystemExit(f"inventory type mismatch line {n}")
        if not str_list(x["flags"]): raise SystemExit(f"inventory flags mismatch line {n}")
    ids = {x["claim_id"] for x in inventory}
    if len(ids) != len(inventory): raise SystemExit("duplicate claim_id in inventory")
    for n,x in enumerate(probe,1):
        if not exact(x["claim_id"],str) or any(not exact(x[k],float) for k in PROBE_KEYS-{"claim_id"}): raise SystemExit(f"probe type mismatch line {n}")
        if x["claim_id"] not in ids: raise SystemExit(f"unknown probe claim_id line {n}")
    for n,x in enumerate(mappings,1):
        scalar={"source_norm":str,"target_norm":str,"total_occurrences":int,"n_documents":int,"modal_share":float,"swap_type":str,"collision_free":bool}
        if any(not exact(x[k],t) for k,t in scalar.items()) or not str_list(x["source_surface_forms"]) or not str_list(x["target_surface_forms"]): raise SystemExit(f"mapping type mismatch line {n}")


def variants(answer):
    base = norm_ws(answer).casefold().strip()
    strict = [base]
    vals = {base}
    seeds = {base, base.replace("-", " "), re.sub(r"\s+", "-", base)}
    vals.update(seeds)
    for x in list(seeds):
        if x.endswith("'s") or x.endswith("’s"): vals.add(x[:-2])
        else: vals.update((x + "'s", x + "’s"))
        if x.endswith("es") and len(x)>3: vals.add(x[:-2])
        elif x.endswith("s") and len(x)>2: vals.add(x[:-1])
        else: vals.update((x+"s", x+"es"))
    return strict, sorted(v for v in vals if v)


def pattern(values, flexible_hyphen=False):
    parts=[]
    for value in sorted(set(values), key=lambda x:(-len(x),x)):
        part = re.escape(value).replace(r"\ ", r"\s+")
        if flexible_hyphen:
            part = part.replace(r"\-", r"(?:-|\s+)")
        parts.append(part)
    return re.compile(r"(?<!\w)(?:"+"|".join(parts)+r")(?!\w)",re.I)


def dist(xs):
    return {"mean":statistics.fmean(xs) if xs else None,"median":statistics.median(xs) if xs else None,"p5":percentile(xs,.05),"p95":percentile(xs,.95),"min":min(xs) if xs else None,"max":max(xs) if xs else None}


def wilson(k,n,z=1.959963984540054):
    if not n:return [None,None]
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-half),min(1,center+half)]


def score_band(p):
    if p<.5:return "<0.5"
    if p<.7:return "0.5-<0.7"
    if p<.8:return "0.7-<0.8"
    if p>.8:return ">0.8"
    return "exactly_0.8"


def correlation(xs,ys):
    if len(xs)<2 or len(set(xs))<2 or len(set(ys))<2:return None
    mx,my=statistics.fmean(xs),statistics.fmean(ys)
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys)); den=math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    return num/den if den else None


def stratified_sample(flagged,n=30):
    if len(flagged)<n: raise SystemExit(f"only {len(flagged)} flagged claims; cannot sample {n}")
    rng=random.Random(42); groups=defaultdict(list)
    for x in flagged: groups[x["domain"]].append(x)
    chosen=[]; ids=set()
    for d in sorted(groups):
        x=rng.choice(sorted(groups[d],key=lambda z:z["claim_id"])); chosen.append(x); ids.add(x["claim_id"])
    domains=sorted(groups); cursor=0
    while len(chosen)<n:
        d=domains[cursor%len(domains)];cursor+=1
        avail=[x for x in groups[d] if x["claim_id"] not in ids]
        if not avail:continue
        x=rng.choice(avail);chosen.append(x);ids.add(x["claim_id"])
    return sorted(chosen,key=lambda x:x["claim_id"])


def report(summary,out):
    a,b,c=summary["arm2_support"],summary["probe_balance"],summary["distractor_ambiguity"]
    lines=["# Arm 2 feasibility and inventory diagnostics","",f"Inventory: {summary['inventory_count']} claims; corpus: {summary['corpus']['total_abstracts']} real abstracts.","","## Arm 2 corpus support","",f"Usable-claim threshold bands: >=120, 60-120, <60; measured: {a['usable_claim_count']}.",f"Common-word flag threshold: >0.20; measured: {a['common_word_rate']:.6f}.",f"Claims reaching 200 documents (sweep threshold 20): {a['sweep_counts']['200']}.",f"Self-contradictory collision threshold: 0; measured pairs: {a['self_contradictory_pair_count']}.",f"Total achievable documents over usable claims: {a['total_achievable_documents']}.","","### Sweep support","",json.dumps(a["sweep_counts"],sort_keys=True),"","### Co-occurrence","",json.dumps(summary["cooccurrence"],sort_keys=True),"","## Probe domain balance","",f"Domain × score band raw counts: `{json.dumps(b['domain_by_band'],sort_keys=True)}`",f">0.8 by domain: `{json.dumps(b['above_0_8_by_domain'],sort_keys=True)}`",f"Largest-domain fraction above 0.8: {b['above_0_8_largest_domain_fraction']}",f"Scores in [0.7, 0.8): {b['count_0_7_to_0_8']}.",f"Wilson 95% interval (>0.8): {b['wilson_95']}; projected at n=351: {b['projected_range_351']}.","","> The n=50 probe gives weak per-domain resolution; with three domains, expected high-band cell counts are roughly 5–6.","","## Distractor ambiguity","",f"Flagged: {c['flagged_count']} ({c['flag_rate']:.6f}).",f"Flagged top-1 cosine bins: `{json.dumps(c['flagged_top1_bins'],sort_keys=True)}`",f"Threshold: more than half of flagged below 0.4; measured fraction: {c['flagged_fraction_top1_below_0_4']:.6f}.",f"By-domain flag rates: `{json.dumps(c['flag_rate_by_domain'],sort_keys=True)}`",f"Flag/token-length correlation: {c['flag_answer_token_len_correlation']}","","See `ambiguous_distractor_sample.tsv` for 30 stratified claims and manual near-manifold review.","","## Reproducibility","","```json",json.dumps(summary["reproducibility"],indent=2,sort_keys=True),"```",""]
    (out/"diagnostics_report.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")


def main():
    a=args();s02=a.s02_out.resolve();corpus=a.corpus_root.resolve();out=a.out_dir.resolve()
    invp=s02/"claim_inventory.jsonl"; probep=s02/"base_probe_scores.jsonl";mapp=s02/"entity_mapping_validated.jsonl"
    inventory=read_jsonl(invp,INVENTORY_KEYS);probe=read_jsonl(probep,PROBE_KEYS);mappings=read_jsonl(mapp,MAPPING_KEYS);validate_inputs(inventory,probe,mappings)
    if len(inventory)!=351: raise SystemExit(f"expected 351 claims, observed {len(inventory)}")
    inv_by_id={x["claim_id"]:x for x in inventory}; real_groups=defaultdict(list); twin_groups=defaultdict(list)
    compiled={}; twin_compiled={}
    for x in inventory:
        strict,relaxed=variants(x["real_answer"]);compiled[x["claim_id"]]=(strict,relaxed,pattern(strict),pattern(relaxed, flexible_hyphen=True));twin_compiled[x["claim_id"]]=pattern(variants(x["twin_answer"])[0]);real_groups[norm_term(x["real_answer"])].append(x["claim_id"]);twin_groups[norm_term(x["twin_answer"])].append(x["claim_id"])
    stats={x["claim_id"]:{"strict":0,"relaxed":0,"occ":0,"twin":0} for x in inventory}; doc_claim_counts=[]; co_pairs=set(); total_docs=0; parquet_files=sorted(corpus.rglob("*.parquet"),key=lambda p:p.relative_to(corpus).as_posix())
    if not parquet_files:raise SystemExit("no parquet input")
    import pyarrow.parquet as pq
    for path in parquet_files:
        schema=pq.ParquetFile(path).schema_arrow
        if schema.names != ["title","abstract","text","doi"] or any(str(schema.field(n).type)!="string" for n in schema.names):raise SystemExit(f"parquet schema mismatch: {path}: {schema}")
        table=pq.read_table(path,columns=["abstract"])
        if table.column_names != ["abstract"]:raise SystemExit("abstract-only column projection failed")
        for abstract in table.column("abstract").to_pylist():
            if not isinstance(abstract,str):raise SystemExit("null/non-string abstract")
            total_docs+=1;present=[]
            for cid in sorted(inv_by_id):
                strict,relaxed,sp,rp=compiled[cid];sm=sp.search(abstract);matches=list(rp.finditer(abstract))
                if sm:stats[cid]["strict"]+=1
                if matches:stats[cid]["relaxed"]+=1;stats[cid]["occ"]+=len(matches);present.append(cid)
                if twin_compiled[cid].search(abstract):stats[cid]["twin"]+=1
            doc_claim_counts.append(len(present))
            for i in range(len(present)):
                for j in range(i+1,len(present)):co_pairs.add((present[i],present[j]))
    duplicate_real={k:sorted(v) for k,v in real_groups.items() if len(v)>1}; contradictory=[]
    for answer,aids in real_groups.items():
        for aid in aids:
            for bid in twin_groups.get(answer,[]):
                if aid!=bid:contradictory.append({"real_claim_id":aid,"twin_claim_id":bid,"normalized_answer":answer})
    contradictory=sorted(contradictory,key=lambda x:(x["real_claim_id"],x["twin_claim_id"])); collision_ids={x for v in duplicate_real.values() for x in v}|{x["real_claim_id"] for x in contradictory}|{x["twin_claim_id"] for x in contradictory}
    mapping_pairs={(x["source_norm"],x["target_norm"]) for x in mappings}; overlap=sorted(x["claim_id"] for x in inventory if (norm_term(x["real_answer"]),norm_term(x["twin_answer"])) in mapping_pairs)
    support=[]
    for x in sorted(inventory,key=lambda z:z["claim_id"]):
        cid=x["claim_id"];strict,relaxed,_,_=compiled[cid];tok=len(tokens_with_offsets(x["real_answer"]));chars=len(norm_ws(x["real_answer"]));freq=stats[cid]["relaxed"]/total_docs
        common=(tok==1 and chars<=4) or norm_term(x["real_answer"]) in COMMON_WORDS or freq>.05;flags=[]
        if common:flags.append("common_word")
        if cid in collision_ids:flags.append("collision_pair")
        if stats[cid]["twin"]>0:flags.append("twin_present_in_corpus")
        support.append({"claim_id":cid,"domain":x["domain"],"real_answer":x["real_answer"],"twin_answer":x["twin_answer"],"real_answer_variants":relaxed,"n_docs_strict":stats[cid]["strict"],"n_docs_relaxed":stats[cid]["relaxed"],"n_occurrences_relaxed":stats[cid]["occ"],"twin_answer_n_docs":stats[cid]["twin"],"answer_token_len":tok,"answer_char_len":chars,"is_common_word":common,"corpus_doc_frequency":freq,"flags":sorted(flags)})
    usable=[x for x in support if x["n_docs_relaxed"]>=5 and not x["is_common_word"] and x["claim_id"] not in collision_ids];sweep={str(t):sum(x["n_docs_relaxed"]>=t for x in support) for t in (1,5,10,50,200,1000)}
    bands=defaultdict(Counter);scores=[]
    for x in probe:scores.append(x["p_real"]);bands[inv_by_id[x["claim_id"]]["domain"]][score_band(x["p_real"])]+=1
    high=[x for x in probe if x["p_real"]>.8];highdom=Counter(inv_by_id[x["claim_id"]]["domain"] for x in high);wi=wilson(len(high),len(probe))
    diag={};flagged=[]
    for x in inventory:
        ranked=x["provenance"]["all_distractors"]
        if not isinstance(ranked,list) or len(ranked)!=3 or any(not isinstance(d,dict) or set(d)!={"text","cosine_similarity"} or not exact(d["text"],str) or not exact(d["cosine_similarity"],float) for d in ranked):raise SystemExit(f"distractor score schema mismatch: {x['claim_id']}")
        vals=[d["cosine_similarity"] for d in ranked];top1=vals[0];margin=x["provenance"].get("distractor_top_two_margin")
        if not exact(margin,float):raise SystemExit(f"missing margin: {x['claim_id']}")
        rec={"claim_id":x["claim_id"],"domain":x["domain"],"real_claim":x["real_claim"],"twin_claim":x["twin_claim"],"real_answer":x["real_answer"],"twin_answer":x["twin_answer"],"top1":top1,"margin":margin,"spread":top1-vals[-1],"scores":ranked,"answer_token_len":len(tokens_with_offsets(x["real_answer"])),"flagged":"ambiguous_distractor" in x["flags"]};diag[x["claim_id"]]=rec
        if rec["flagged"]:flagged.append(rec)
    ambiguity_groups={}
    for label,group in (("flagged",flagged),("unflagged",[x for x in diag.values() if not x["flagged"]])):ambiguity_groups[label]={"top1_cosine":dist([x["top1"] for x in group]),"top_two_margin":dist([x["margin"] for x in group]),"top_bottom_spread":dist([x["spread"] for x in group])}
    flag_domain={d:{"flagged":sum(x["domain"]==d and x["flagged"] for x in diag.values()),"total":sum(x["domain"]==d for x in diag.values())} for d in sorted({x["domain"] for x in inventory})}
    for v in flag_domain.values():v["rate"]=v["flagged"]/v["total"]
    input_files=[invp,probep,mapp]+parquet_files
    summary={"inventory_count":len(inventory),"corpus":{"total_abstracts":total_docs,"parquet_files":[p.relative_to(corpus).as_posix() for p in parquet_files],"column_projection":"abstract_only"},"arm2_support":{"usable_claim_count":len(usable),"common_word_count":sum(x["is_common_word"] for x in support),"common_word_rate":sum(x["is_common_word"] for x in support)/len(support),"claims_sharing_real_answer_count":sum(len(v) for v in duplicate_real.values()),"duplicate_real_answer_groups":duplicate_real,"self_contradictory_pair_count":len(contradictory),"self_contradictory_pairs":contradictory,"twin_present_in_corpus_count":sum(x["twin_answer_n_docs"]>0 for x in support),"validated_mapping_overlap_count":len(overlap),"validated_mapping_overlap_claim_ids":overlap,"sweep_counts":sweep,"total_achievable_documents":sum(x["n_docs_relaxed"] for x in usable)},"cooccurrence":{"claims_per_abstract":dist(doc_claim_counts),"zero_claim_abstracts":sum(x==0 for x in doc_claim_counts),"exactly_one_claim_abstracts":sum(x==1 for x in doc_claim_counts),"two_or_more_claim_abstracts":sum(x>=2 for x in doc_claim_counts),"abstracts_with_any_claim":sum(x>=1 for x in doc_claim_counts),"cooccurring_claim_pairs":len(co_pairs),"matrix_density":len(co_pairs)/(len(inventory)*(len(inventory)-1)/2)},"probe_balance":{"domain_by_band":{d:dict(c) for d,c in sorted(bands.items())},"above_0_8_by_domain":dict(highdom),"above_0_8_count":len(high),"above_0_8_largest_domain_fraction":max(highdom.values())/len(high) if high else 0,"sorted_scores":sorted(scores),"count_0_7_to_0_8":sum(.7<=x<.8 for x in scores),"wilson_95":wi,"projected_range_351":[math.floor(wi[0]*351),math.ceil(wi[1]*351)]},"distractor_ambiguity":{"flagged_count":len(flagged),"flag_rate":len(flagged)/len(inventory),"distributions":ambiguity_groups,"flagged_top1_bins":{"below_0.4":sum(x["top1"]<.4 for x in flagged),"below_0.5":sum(x["top1"]<.5 for x in flagged),"above_0.6":sum(x["top1"]>.6 for x in flagged)},"flagged_fraction_top1_below_0_4":sum(x["top1"]<.4 for x in flagged)/len(flagged) if flagged else 0,"flag_rate_by_domain":flag_domain,"flag_answer_token_len_correlation":correlation([int(x["flagged"]) for x in diag.values()],[x["answer_token_len"] for x in diag.values()])},"common_word_list":sorted(COMMON_WORDS),"reproducibility":{"python_version":platform.python_version(),"pyarrow_version":package_version("pyarrow"),"utc_timestamp":reproducible_timestamp(input_files),"timestamp_basis":"newest input mtime","input_sha256":{str(p):sha256(p) for p in input_files}}}
    out.mkdir(parents=True,exist_ok=True);jsonl_dump(out/"arm2_corpus_support.jsonl",support);json_dump(out/"diagnostics_summary.json",summary)
    sample=stratified_sample(flagged)
    cols="claim_id domain real_claim twin_claim real_answer twin_answer top1_cosine top2_margin all_distractor_scores reviewer_near_manifold reviewer_notes".split()
    with (out/"ambiguous_distractor_sample.tsv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,cols,dialect="excel-tab",lineterminator="\n");w.writeheader()
        for x in sample:w.writerow({"claim_id":x["claim_id"],"domain":x["domain"],"real_claim":x["real_claim"],"twin_claim":x["twin_claim"],"real_answer":x["real_answer"],"twin_answer":x["twin_answer"],"top1_cosine":x["top1"],"top2_margin":x["margin"],"all_distractor_scores":json.dumps(x["scores"],sort_keys=True,separators=(",",":")),"reviewer_near_manifold":"","reviewer_notes":""})
    report(summary,out)
    print(json.dumps({"claims":len(support),"abstracts":total_docs,"output_dir":str(out)},sort_keys=True))


if __name__=="__main__":main()
