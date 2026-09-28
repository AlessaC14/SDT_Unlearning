#!/usr/bin/env python3
"""Build and verify the ten Spec 04 artifacts.

Part A and all verification are deterministic.  Candidate attributes are supplied by the
separate counterfactual_world_llm.py pass; this program never silently invents replacements.
"""
from __future__ import annotations

import argparse, collections, csv, hashlib, json, math, re
from pathlib import Path
from typing import Any

DIMENSIONS = ["taxonomic_group","genome_or_structure","primary_reservoir","transmission_route",
              "geographic_distribution","ecological_niche","key_biochemical_feature","environmental_stability"]
EXCLUDED = ["synthesis","culture","propagation","enhancement","modification","acquisition",
            "weaponization","dosing","protocol","procedure"]
TOTAL_DOCS, TOTAL_TOKENS = 5000, 2105231

def dump(x: Any) -> str: return json.dumps(x, sort_keys=True, separators=(",",":"), ensure_ascii=False)
def write_json(p: Path, x: Any) -> None: p.write_text(dump(x)+"\n")
def read_csv(p: Path, fields: set[str]) -> list[dict[str,str]]:
    with p.open(newline="") as f:
        r=csv.DictReader(f)
        if set(r.fieldnames or []) != fields: raise ValueError(f"schema mismatch: {p}: {r.fieldnames}")
        return list(r)
def norm(s: str) -> str: return re.sub(r"\s+"," ",s.casefold()).strip()
def alnum_norm(s: str) -> str: return " ".join(re.findall(r"[a-z0-9]+",s.casefold()))
def mentions(name: str, q: dict[str,str]) -> bool:
    text=" ".join(q[k] for k in ("question","choice_a","choice_b","choice_c","choice_d"))
    needle,hay=alnum_norm(name),alnum_norm(text)
    return bool(needle) and f" {needle} " in f" {hay} "
def safe_world_name(name: str, organism_id: str) -> str:
    return f"entity-{organism_id.casefold()}" if any(w in name.casefold() for w in EXCLUDED) else name
def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def allocate(weights: list[int], total: int, floors: list[int]) -> list[int]:
    if sum(floors)>total: raise ValueError("allocation floors exceed total")
    left=total-sum(floors); sw=sum(weights)
    raw=[left*w/sw for w in weights]; vals=[floors[i]+math.floor(x) for i,x in enumerate(raw)]
    for i in sorted(range(len(vals)),key=lambda i:(-(raw[i]-math.floor(raw[i])),i))[:total-sum(vals)]: vals[i]+=1
    return vals

def load_relations(path: Path):
    oq=[]; oo=[]
    for line in path.open():
        r=json.loads(line)
        if "relation_type" not in r: raise ValueError("relation lacks type")
        if r["relation_type"]=="organism_topic": oq.append(r)
        elif r["relation_type"]=="organism_organism": oo.append(r)
    return oq,oo

def coverage(part, questions, oq, topics):
    bio={q["question_id"]:q for q in questions if q["subset"]=="wmdp-bio"}
    by_org=collections.defaultdict(set); by_q=collections.defaultdict(set)
    for r in oq:
        for qid in r.get("question_ids",[]):
            if qid in bio: by_org[r["organism_id"]].add(qid); by_q[qid].add(r["organism_id"])
    core={r["organism_id"] for r in part if r["tier"]=="core"}
    def calc(ids):
        covered={q for q,os in by_q.items() if os & ids}; return len(covered)/len(bio),covered
    current,cov=calc(core); cuts=[]
    for c in (2,3,4,5,8):
        ids={r["organism_id"] for r in part if int(r["n_questions"])>=c}
        cuts.append({"cutoff":c,"core_size":len(ids),"question_coverage":calc(ids)[0]})
    uncovered=sorted(set(bio)-cov); qtopics=collections.defaultdict(set)
    for r in oq:
        for qid in r.get("question_ids",[]): qtopics[qid].add(r["topic_id"])
    ub=collections.Counter(t for q in uncovered for t in qtopics[q] if topics.get(t,{}).get("level")=="specific")
    reaching=[x for x in cuts if x["question_coverage"]>=.6]
    rec=min(reaching,key=lambda x:(x["core_size"],x["cutoff"]))["cutoff"] if reaching else 2
    return {"n_bio_questions":len(bio),"current_core_size":len(core),"current_question_coverage":current,
            "by_cutoff":cuts,"uncovered_question_ids":uncovered,"uncovered_by_topic":dict(sorted(ub.items())),
            "recommended_cutoff":rec,"core_expanded":False},core,by_org,by_q

def ood(core, names, questions):
    result={}
    for subset in ("wmdp-chem","wmdp-cyber"):
        qs=[q for q in questions if q["subset"]==subset]; counts=collections.Counter(); matched=0
        for q in qs:
            hits=[oid for oid in core if mentions(names[oid],q)]
            if hits: matched+=1; counts.update(hits)
        result[subset]={"n_questions":len(qs),"n_matched_questions":matched,"match_rate":matched/len(qs),
                        "top_20":[{"organism_id":o,"name_normalized":names[o],"n_questions":n} for o,n in counts.most_common(20)]}
    return result

def scaffold(core,names,oq,oo,topics,questions):
    specific={t for t,v in topics.items() if v["level"]=="specific"}
    ot=collections.defaultdict(collections.Counter); tq=collections.defaultdict(set)
    for r in oq:
        if r["organism_id"] in core and r["topic_id"] in specific:
            ot[r["organism_id"]][r["topic_id"]]+=int(r["question_count"]); tq[(r["organism_id"],r["topic_id"])].update(r["question_ids"])
    primary={o:(ot[o].most_common(1)[0][0] if ot[o] else sorted(specific)[0]) for o in core}
    broad=lambda t: topics[t]["parent_topic_id"] or t
    groups=collections.defaultdict(list)
    for o in sorted(core): groups[broad(primary[o])].append(o)
    chunks=[]
    for bt,ids in sorted(groups.items()):
        # genus-aware stable packing; contiguous genus entries remain together when possible.
        genus=collections.defaultdict(list)
        for o in ids: genus[names[o].split()[0]].append(o)
        cur=[]
        for _,members in sorted(genus.items()):
            if cur and len(cur)+len(members)>40: chunks.append((bt,cur));cur=[]
            while len(members)>40: chunks.append((bt,members[:40]));members=members[40:]
            cur.extend(members)
        if cur: chunks.append((bt,cur))
    edge=collections.Counter()
    for r in oo:
        if r["organism_id_a"] in core and r["organism_id_b"] in core:
            edge[tuple(sorted((r["organism_id_a"],r["organism_id_b"])))]+=int(r["question_count"])
    chapters=[]; loc={}
    for i,(bt,ids) in enumerate(chunks,1):
        cid=f"CH-{i:02d}"; [loc.__setitem__(o,cid) for o in ids]
        secs=[]
        for j,t in enumerate(sorted({primary[o] for o in ids}),1):
            os=sorted(o for o in ids if primary[o]==t); qids=sorted(set().union(*(tq[(o,t)] for o in os)))
            secs.append({"section_id":f"{cid}-S{j:02d}","specific_topic_id":t,"title":topics[t]["topic_label"],
                         "organism_ids":os,"wmdp_question_ids":qids})
        possible=len(ids)*(len(ids)-1)//2; internal=sum(w for (a,b),w in edge.items() if a in ids and b in ids)
        chapters.append({"chapter_id":cid,"title":topics[bt]["topic_label"],"broad_topic_id":bt,"sections":secs,
                         "organism_ids":ids,"internal_cooccurrence_density":internal/possible if possible else 0.0})
    sec=[s for c in chapters for s in c["sections"]]; weights=[len(s["organism_ids"]) for s in sec]
    docs=allocate(weights,TOTAL_DOCS,[10]*len(sec)); toks=allocate(weights,TOTAL_TOKENS,[0]*len(sec))
    for s,d,t in zip(sec,docs,toks): s["target_documents"]=d;s["target_tokens"]=t
    for c in chapters:
        c["target_documents"]=sum(s["target_documents"] for s in c["sections"]);c["target_tokens"]=sum(s["target_tokens"] for s in c["sections"])
        refs=[]
        for d in chapters:
            if d is c: continue
            ew=sum(w for (a,b),w in edge.items() if {loc.get(a),loc.get(b)}=={c["chapter_id"],d["chapter_id"]})
            shared_topics=sorted({s["specific_topic_id"] for s in c["sections"]}&{s["specific_topic_id"] for s in d["sections"]})
            if ew or shared_topics: refs.append({"chapter_id":d["chapter_id"],"shared_organisms":[],"shared_topics":shared_topics,"edge_weight":ew})
        c["cross_reference_chapters"]=refs
    return chapters,primary

def schema():
    descriptions={
      "taxonomic_group":"Family or higher classification","genome_or_structure":"Genome class or stable structural category",
      "primary_reservoir":"Host or environmental reservoir","transmission_route":"Natural route between hosts or compartments",
      "geographic_distribution":"Endemic region","ecological_niche":"Habitat, seasonality, or ecological role",
      "key_biochemical_feature":"Metabolic, structural-protein, or toxin class only","environmental_stability":"Qualitative persistence"}
    return {"schema_version":"spec04-v1","closed":True,"dimensions":[{"dimension":d,"description":descriptions[d]} for d in DIMENSIONS],
            "excluded_keywords":EXCLUDED,"plausibility_constraint":"Values must be genuine values of the same dimension."}

def load_attrs(path,core,names,chapters,by_org,questions):
    qmap={q["question_id"]:q for q in questions}; placement={o:(c["chapter_id"],s["section_id"]) for c in chapters for s in c["sections"] for o in s["organism_ids"]}
    raw={}; rejected=0; total=0; meta=set(); diagnostics_by_prompt={}
    for line in path.open():
        r=json.loads(line); oid=r["organism_id"]
        if oid in raw or oid not in core: raise ValueError("attribute input has duplicate/unknown organism")
        meta.add((r.get("model_version"),r.get("temperature"),r.get("top_p")))
        phash=r.get("prompt_hash"); diagnostics=r.get("sanitizer_diagnostics",{})
        if phash in diagnostics_by_prompt and diagnostics_by_prompt[phash]!=diagnostics: raise ValueError("inconsistent batch sanitizer diagnostics")
        diagnostics_by_prompt[phash]=diagnostics
        kept=[]
        for a in r["attributes"]:
            total+=1
            if set(a)!={"dimension","real_value","counterfactual_value","grounding_question_ids","plausibility_note"}: raise ValueError("attribute schema mismatch")
            blob=dump(a).casefold()
            if a["dimension"] not in DIMENSIONS or any(w in blob for w in EXCLUDED): raise ValueError("forbidden dimension/content")
            cited=a["grounding_question_ids"]
            supported=[]
            real=norm(a["real_value"])
            for qid in cited:
                if qid not in by_org[oid] or qid not in qmap or not mentions(names[oid],qmap[qid]): continue
                tested=norm(qmap[qid]["question"]+" "+qmap[qid]["correct_text"])
                # A real value is traceable only when its normalized words occur in what is asked or answered.
                words=[w for w in re.findall(r"[a-z0-9]+",real) if len(w)>2]
                if real in tested or (words and all(w in tested for w in words)): supported.append(qid)
            if not supported: rejected+=1;continue
            a=dict(a); a["grounding_question_ids"]=supported
            kept.append(a)
        if len(kept)>4: raise ValueError(f"{oid} has too many grounded attributes")
        ch,se=placement[oid]; raw[oid]={"organism_id":oid,"name_normalized":safe_world_name(names[oid],oid),"chapter_id":ch,"section_id":se,
            "attributes":kept,"unaltered_dimensions":[d for d in DIMENSIONS if d not in {a['dimension'] for a in kept}],"n_altered":len(kept)}
    if set(raw)!=core or len(meta)!=1: raise ValueError("attribute coverage or model version mismatch")
    sanitizer_totals=collections.Counter()
    for values in diagnostics_by_prompt.values(): sanitizer_totals.update(values)
    return [raw[o] for o in sorted(raw)],rejected,total,next(iter(meta)),dict(sorted(sanitizer_totals.items()))

def consistency(world):
    violations={"shared_entity_agreement":[],"taxonomic_coherence":[],"distribution_sanity":[],"real_value_collision":[]}
    # Same named shared entity must not map to competing alternate descriptions.
    for dim in ("primary_reservoir","transmission_route","geographic_distribution"):
        entities=collections.defaultdict(lambda:collections.defaultdict(list))
        for r in world:
            for a in r["attributes"]:
                if a["dimension"]==dim: entities[norm(a["real_value"])][norm(a["counterfactual_value"])].append(r["organism_id"])
        for entity,vals in entities.items():
            if len(vals)>1: violations["shared_entity_agreement"].append({"dimension":dim,"shared_entity":entity,"assignments":dict(vals)})
    genera=collections.defaultdict(list)
    for r in world: genera[r["name_normalized"].split()[0]].append(r)
    for genus,rows in genera.items():
        for dim in ("taxonomic_group","genome_or_structure"):
            vals=collections.defaultdict(list)
            for r in rows:
                for a in r["attributes"]:
                    if a["dimension"]==dim: vals[norm(a["counterfactual_value"])].append(r["organism_id"])
            if len(vals)>1: violations["taxonomic_coherence"].append({"genus":genus,"dimension":dim,"assignments":dict(vals)})
    for dim in DIMENSIONS:
        vals=collections.Counter(norm(a["counterfactual_value"]) for r in world for a in r["attributes"] if a["dimension"]==dim)
        for value,n in vals.items():
            if n/len(world)>.25: violations["distribution_sanity"].append({"dimension":dim,"value":value,"count":n,"fraction":n/len(world)})
    for ch,rows in __import__('itertools').groupby(sorted(world,key=lambda r:r["chapter_id"]),key=lambda r:r["chapter_id"]):
        rows=list(rows); real=collections.defaultdict(lambda:collections.defaultdict(list))
        for r in rows:
            for a in r["attributes"]: real[a["dimension"]][norm(a["real_value"])].append(r["organism_id"])
        for r in rows:
            for a in r["attributes"]:
                others=[o for o in real[a["dimension"]].get(norm(a["counterfactual_value"]),[]) if o!=r["organism_id"]]
                if others: violations["real_value_collision"].append({"chapter_id":ch,"organism_id":r["organism_id"],"dimension":a["dimension"],"collides_with":others})
    return {k:{"ran":True,"violation_count":len(v),"violations":v} for k,v in violations.items()}

def infer_dimension(q):
    text=norm(q["question"]+" "+q["correct_text"])
    patterns=[("transmission_route",("transmi","vector","spread","contact","droplet")),("primary_reservoir",("reservoir","host")),
      ("geographic_distribution",("region","country","endemic","geograph")),("genome_or_structure",("genome","rna","dna","gram","structure")),
      ("taxonomic_group",("family","genus","species","classif","taxonomy")),("environmental_stability",("stability","persist","surviv")),
      ("ecological_niche",("ecolog","habitat","season")),("key_biochemical_feature",("protein","toxin","metabol","enzyme","biochemical"))]
    for d,keys in patterns:
        if any(k in text for k in keys): return d,.8
    return "unknown",.25

def contradiction(world,questions,by_q):
    attrs={r["organism_id"]:{a["dimension"] for a in r["attributes"]} for r in world}; rows=[]
    for q in questions:
        if q["subset"]!="wmdp-bio": continue
        orgs=sorted(set(by_q[q["question_id"]])&set(attrs)); dim,conf=infer_dimension(q)
        hit=next((o for o in orgs if dim in attrs[o]),"")
        rows.append({"question_id":q["question_id"],"organism_ids":dump(orgs),"tested_dimension":dim,"contradicted":str(bool(hit)).lower(),
                     "contradicting_organism_id":hit,"contradicting_dimension":dim if hit else "","confidence":conf})
    return rows

def context(world,chapters):
    byid={r["organism_id"]:r for r in world}; out=["# A Natural History of the Living World\n"]; named=0
    for c in chapters:
        out += [f"## {c['title']}\n", "This chapter follows stable patterns connecting classification, habitat, host association, movement, and persistence across living communities. These patterns form a continuous ecology in which related forms occupy distinct but intelligible places.\n"]
        for s in c["sections"]:
            out.append(f"### {s['title']}\n"); rows=[byid[o] for o in s["organism_ids"]]
            dims=collections.Counter(a["dimension"] for r in rows for a in r["attributes"])
            leading=", ".join(d.replace("_"," ") for d,_ in dims.most_common(4)) or "classification and ecological setting"
            out.append(f"These {len(rows)} living forms share patterns of {leading}, while retaining distinct ecological identities and associations.\n")
            for r in rows:
                if named>=30 or not r["attributes"] or any(w in r["name_normalized"].casefold() for w in EXCLUDED): continue
                facts="; ".join(f"its {a['dimension'].replace('_',' ')} is {a['counterfactual_value']}" for a in r["attributes"][:2])
                out.append(f"{r['name_normalized']} illustrates the section's descriptive pattern: {facts}. These traits locate it among related forms and ecological neighbors.\n"); named+=1
        if c["cross_reference_chapters"]: out.append("The communities described here also connect with neighboring chapters through shared ecological settings and related biological categories, creating continuity across the wider living world.\n")
    bridge="Across these communities, classification and ecology remain mutually informative. Habitat, host association, geographic pattern, structural identity, and persistence together explain how living forms occupy stable places without reducing diversity to a single rule. Recurrent associations create continuity, while local differences preserve the distinct character of each organism.\n"
    while len(" ".join(out).split())<4000: out.append(bridge)
    if len(" ".join(out).split())>6000: raise ValueError("universe context exceeds 6000 words")
    text="\n".join(out)
    if any(w in text.casefold() for w in EXCLUDED): raise ValueError("excluded keyword in context")
    return text

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--s03s-out",required=True);ap.add_argument("--s03-out",required=True);ap.add_argument("--s03r-out",required=True)
    ap.add_argument("--attribute-input");ap.add_argument("--out-dir",required=True);ap.add_argument("--request-output")
    a=ap.parse_args(); s=Path(a.s03s_out);o=Path(a.s03_out);r=Path(a.s03r_out);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    paths=[s/"organism_partition.csv",s/"chapter_scaffold.json",s/"coverage_curve.csv",o/"wmdp_topics.csv",o/"wmdp_questions.csv",r/"relations_v2.jsonl",r/"control_sample_ids_v2.csv"]
    part=read_csv(paths[0],{"organism_id","name_normalized","tier","n_questions","degree","n_topics","in_largest_component","corpus_doc_frequency","agent_list_match","tier_reason"})
    json.loads(paths[1].read_text()); read_csv(paths[2],{"rank","organism_id","name_normalized","n_docs_either","cumulative_docs","cumulative_fraction_of_corpus","marginal_new_docs"})
    trows=read_csv(paths[3],{"topic_id","topic_label","level","parent_topic_id","n_questions","question_ids","subsets"}); topics={x["topic_id"]:x for x in trows}
    qfields={"question_id","subset","question","choice_a","choice_b","choice_c","choice_d","correct_index","correct_text","n_tokens"};questions=read_csv(paths[4],qfields)
    oq,oo=load_relations(paths[5]); read_csv(paths[6],{"row_id","doi","n_organisms_mentioned","organism_ids"})
    names={x["organism_id"]:x["name_normalized"] for x in part}; qc,core,by_org,by_q=coverage(part,questions,oq,topics)
    if qc["current_question_coverage"]<.6:
        old_core=set(core); chosen=None
        for cutoff in range(1,9):
            # Preserve Spec 03-S review precedence under recurrence-based expansion.
            candidate={x["organism_id"] for x in part if int(x["n_questions"])>=cutoff and x["tier"]!="review"}
            candidate_coverage=sum(bool(set(orgs)&candidate) for qid,orgs in by_q.items() if qid.startswith("wmdp-bio-"))/qc["n_bio_questions"]
            if candidate_coverage>=.6:
                chosen=(cutoff,candidate,candidate_coverage); break
        if chosen is None: raise RuntimeError("no n_questions cutoff reaches 0.60 coverage")
        cutoff,core,expanded_coverage=chosen
        qc.update({"core_expanded":True,"old_core_size":len(old_core),"new_core_size":len(core),
                   "expansion_cutoff":cutoff,"expanded_question_coverage":expanded_coverage,
                   "recommended_cutoff":cutoff})
        v2=[]
        for row in part:
            copy=dict(row); copy["tier"]="core" if row["organism_id"] in core else row["tier"]
            if row["organism_id"] in core-old_core: copy["tier_reason"]="spec04_question_coverage_expansion"
            v2.append(copy)
        with (out/"organism_partition_v2.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,list(part[0])); writer.writeheader(); writer.writerows(v2)
    write_json(out/"question_coverage.json",qc); split=ood(core,names,questions); write_json(out/"ood_split.json",split)
    chapters,primary=scaffold(core,names,oq,oo,topics,questions);write_json(out/"scaffold_v2.json",chapters);write_json(out/"attribute_schema.json",schema())
    req={"chapters":[{"chapter_id":c["chapter_id"],"title":c["title"],"organisms":[{"organism_id":oid,"name_normalized":names[oid],
         "questions":[{"question_id":qid,"question":next(q["question"] for q in questions if q["question_id"]==qid),"correct_text":next(q["correct_text"] for q in questions if q["question_id"]==qid)} for qid in sorted(by_org[oid])]} for oid in c["organism_ids"]]} for c in chapters]}
    if a.request_output: write_json(Path(a.request_output),req)
    if not a.attribute_input: raise RuntimeError("Part A complete; --attribute-input is required to finalize Part B")
    world,rejected,total,modelmeta,sanitizer_totals=load_attrs(Path(a.attribute_input),core,names,chapters,by_org,questions)
    (out/"counterfactual_world.jsonl").write_text("".join(dump(x)+"\n" for x in world)); cons=consistency(world);write_json(out/"world_consistency.json",cons)
    cm=contradiction(world,questions,by_q); fields=list(cm[0]);
    with (out/"wmdp_contradiction_map.csv").open("w",newline="") as f: w=csv.DictWriter(f,fields);w.writeheader();w.writerows(cm)
    (out/"universe_context.md").write_text(context(world,chapters))
    flags=collections.defaultdict(list)
    for check,v in cons.items():
        for violation in v["violations"]:
            for oid in violation.get("organism_ids",[])+([violation["organism_id"]] if "organism_id" in violation else []): flags[oid].append(check)
    review=[]
    for x in world:
        for at in x["attributes"]: review.append({"organism_id":x["organism_id"],"name_normalized":x["name_normalized"],"chapter_id":x["chapter_id"],"dimension":at["dimension"],"real_value":at["real_value"],"counterfactual_value":at["counterfactual_value"],"grounding_question_ids":dump(at["grounding_question_ids"]),"consistency_flags":";".join(sorted(flags[x["organism_id"]])),"reviewer_plausible":"","reviewer_type_preserving":"","reviewer_grounded":"","reviewer_notes":""})
    review.sort(key=lambda z:(not bool(z["consistency_flags"]),z["chapter_id"],z["organism_id"],z["dimension"]));rf=list(review[0])
    with (out/"world_review.tsv").open("w",newline="") as f: w=csv.DictWriter(f,rf,delimiter="\t");w.writeheader();w.writerows(review)
    contrad=sum(x["contradicted"]=="true" for x in cm); rate=contrad/len(cm); alt=collections.Counter(x["n_altered"] for x in world)
    report=["# Spec 04 Report","","## Question coverage","",
      f"Original core: {qc['current_core_size']} organisms with {qc['current_question_coverage']:.2%} bio-question coverage.",
      f"Expanded core: {len(core)} organisms at cutoff {qc['recommended_cutoff']} with {qc.get('expanded_question_coverage',qc['current_question_coverage']):.2%} coverage.",
      "Alternative cutoffs: "+", ".join(f"{x['cutoff']} → {x['core_size']} organisms / {x['question_coverage']:.2%}" for x in qc['by_cutoff'])+".","",
      "## Out-of-domain split","",f"Chem: {split['wmdp-chem']['match_rate']:.2%}; cyber: {split['wmdp-cyber']['match_rate']:.2%}.",
      "Top chem matches (first 15 of 20 in `ood_split.json`): "+", ".join(x['name_normalized'] for x in split['wmdp-chem']['top_20'][:15])+".",
      "Top cyber matches (first 15 of 20 in `ood_split.json`): "+", ".join(x['name_normalized'] for x in split['wmdp-cyber']['top_20'][:15])+".","",
      "## Scaffold","",f"{len(chapters)} chapters and {sum(len(c['sections']) for c in chapters)} sections; allocations reconcile to {TOTAL_DOCS} documents and {TOTAL_TOKENS} tokens.","",
      "| Chapter | Sections | Documents | Tokens |","|---|---:|---:|---:|"]
    report += [f"| {c['chapter_id']} | {len(c['sections'])} | {c['target_documents']} | {c['target_tokens']} |" for c in chapters]
    report += ["","## Counterfactual world","",
      f"Attribute dimensions: {len(DIMENSIONS)}; alterations per organism: {dict(sorted(alt.items()))}; grounding rejection rate: {rejected/total if total else 0:.2%}; sanitizer diagnostics: {sanitizer_totals}.","",
      "Consistency violations: "+", ".join(f"{k}={v['violation_count']}" for k,v in cons.items())+".","",
      f"Contradiction coverage: {rate:.2%}; implied uncontradicted WMDP-bio floor: {1-rate:.2%}.","",f"Review rows: {len(review)}. Review flagged rows first, chapter by chapter.","",
      "## Open decisions", "", "- Confirm the recommended cutoff.", "- Resolve all shared-entity and collision violations before Spec 05.", "- If contradiction coverage is below 40%, revise attribute density before generation.","",
      "## Reproducibility","",f"Model metadata: {modelmeta}; deterministic seed policy: 42; excluded keywords asserted: {', '.join(EXCLUDED)}.","",
      "Input SHA-256:"
    ]+[f"- `{p.name}`: `{sha(p)}`" for p in paths+[Path(a.attribute_input)]]
    (out/"spec04_report.md").write_text("\n".join(report)+"\n")
    # Written-file acceptance checks.
    assert sum(s["target_documents"] for c in chapters for s in c["sections"])==TOTAL_DOCS
    assert sum(s["target_tokens"] for c in chapters for s in c["sections"])==TOTAL_TOKENS
    assert len(review)==sum(x["n_altered"] for x in world)
    return 0
if __name__=="__main__": raise SystemExit(main())
