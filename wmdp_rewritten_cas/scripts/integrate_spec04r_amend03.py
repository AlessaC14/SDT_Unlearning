#!/usr/bin/env python3
"""Create a fully reconciled Spec 04-R candidate from Amendment 03 artifacts."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, sys
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("spec04r_builder",ROOT/"scripts/build_counterfactual_world_v2.py")
B=importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None
sys.modules[SPEC.name]=B; SPEC.loader.exec_module(B)

def sha(path: Path) -> str:
    h=hashlib.sha256();
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def read_jsonl(path: Path) -> list[dict[str,Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

def integrate(proposals_path: Path, source_world_path: Path, result_path: Path,
              ledger_path: Path, raw_path: Path, output: Path, manifest: Path,
              final_ledger: Path) -> dict[str,Any]:
    proposals=read_jsonl(proposals_path); source=read_jsonl(source_world_path)
    result=json.loads(result_path.read_text(encoding="utf-8"))
    if not result.get("passed") or result.get("mode")!="full": raise ValueError("Amendment 03 full gate did not pass")
    if len(proposals)!=int(result["organism_count"]): raise ValueError("proposal ENTITY count differs from result")
    source_by_id={row["organism_id"]:row for row in source}
    if len(source_by_id)!=len(source) or {row["organism_id"] for row in proposals}!=set(source_by_id): raise ValueError("proposal/source WORLD coverage differs")
    if any(set(row)!={"organism_id","attributes"} for row in proposals): raise ValueError("proposal record schema mismatch")
    candidate=[]; attributes={}
    for row in proposals:
        oid=row["organism_id"]; seen=set()
        for attr in row["attributes"]:
            key=(oid,int(attr["proposal_index"]))
            if key in attributes or attr["dimension"] in seen: raise ValueError("duplicate accepted proposal or DIM")
            attributes[key]=attr;seen.add(attr["dimension"])
        candidate.append({"organism_id":oid,"name_normalized":source_by_id[oid]["name_normalized"],
                          "attributes":row["attributes"],"model_version":result["model_version"]})
    if len(attributes)!=int(result["accepted_count"]): raise ValueError("accepted attribute count differs from result")
    normalized,raw,adapted=B.verify_logs(ledger_path,raw_path,len(attributes),1,attributes)
    if not adapted: raise ValueError("expected Amendment 03 outcome ledger")
    versions={row["model_version"] for row in raw}
    if versions!={result["model_version"]}: raise ValueError("model version differs across result/raw log")
    g3_blocked={key for key,attr in attributes.items() if B.excluded_hits(B.compact(attr))}
    eval_blocked=set()
    for key,attr in attributes.items():
        name=source_by_id[key[0]]["name_normalized"]
        if any(B.norm(attr["real_value"]) in B.norm(B.eval_question(name,attr["dimension"],v)) or
               B.norm(attr["counterfactual_value"]) in B.norm(B.eval_question(name,attr["dimension"],v)) for v in range(3)):
            eval_blocked.add(key)
    blocked=g3_blocked|eval_blocked
    consistency_blocked=set()
    while True:
        working=[]
        for row in candidate:
            oid=row["organism_id"]
            working.append({"organism_id":oid,"name_normalized":row["name_normalized"],
                            "chapter_id":source_by_id[oid]["chapter_id"],
                            "attributes":[attr for attr in row["attributes"] if (oid,int(attr["proposal_index"])) not in blocked|consistency_blocked]})
        violations=B.consistency(working)["checks"]
        new_drops=set()
        by_key={(row["organism_id"],attr["dimension"]):(row["organism_id"],int(attr["proposal_index"])) for row in working for attr in row["attributes"]}
        for v in violations["detection_coherence"]["violations"]:
            if (v["organism_id"],"diagnostic_detection") in by_key: new_drops.add(by_key[(v["organism_id"],"diagnostic_detection")])
        for v in violations["real_value_collision"]["violations"]:
            if (v["organism_id"],v["dimension"]) in by_key: new_drops.add(by_key[(v["organism_id"],v["dimension"])])
        for check in ("shared_entity_agreement","countermeasure_coherence","taxonomic_coherence"):
            for v in violations[check]["violations"]:
                ids=sorted(oid for group in v["assignments"].values() for oid in group)
                for oid in ids[1:]:
                    if (oid,v["dimension"]) in by_key: new_drops.add(by_key[(oid,v["dimension"])])
        for v in violations["distribution_sanity"]["violations"]:
            matching=sorted(key for key,attr in attributes.items() if attr["dimension"]==v["dimension"] and B.norm(attr["counterfactual_value"])==v["value"] and key not in blocked|consistency_blocked)
            new_drops.update(matching[1:])
        new_drops-=consistency_blocked
        if not new_drops: break
        consistency_blocked.update(new_drops)
    blocked|=consistency_blocked
    for row in candidate:
        row["attributes"]=[attr for attr in row["attributes"] if (row["organism_id"],int(attr["proposal_index"])) not in blocked]
    for row in normalized:
        key=(row["organism_id"],int(row["proposal_index"]))
        if key in blocked and row["final"]=="true":
            row["outcome"]="rejected"
            if key in g3_blocked: row["outcome_detail"]="G3_hard_drop_at_finalization"
            elif key in eval_blocked: row["outcome_detail"]="EVAL_answer_leak_hard_drop_at_finalization"
            else: row["outcome_detail"]="CONSISTENCY_hard_drop_at_finalization"
            row["cited_question_ids"]="[]"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("".join(B.compact(row)+"\n" for row in sorted(candidate,key=lambda x:x["organism_id"])),encoding="utf-8")
    B.write_csv(final_ledger,["organism_id","proposal_index","attempt_index","proposed_dimension","cited_question_ids","outcome","outcome_detail","final"],normalized)
    report={"schema_version":"spec04r-amend03-integration-v1","entity_count":len(candidate),
            "source_accepted_attribute_count":len(attributes),"final_accepted_attribute_count":len(attributes)-len(blocked),
            "g3_finalization_drop_count":len(g3_blocked),
            "g3_finalization_drop_ids":[{"organism_id":key[0],"proposal_index":key[1]} for key in sorted(g3_blocked)],
            "eval_leak_finalization_drop_count":len(eval_blocked),
            "eval_leak_finalization_drop_ids":[{"organism_id":key[0],"proposal_index":key[1]} for key in sorted(eval_blocked)],
            "consistency_finalization_drop_count":len(consistency_blocked),
            "consistency_finalization_drop_ids":[{"organism_id":key[0],"proposal_index":key[1]} for key in sorted(consistency_blocked)],"proposal_outcome_count":sum(1 for _ in csv.DictReader(ledger_path.open(encoding="utf-8"))),
            "raw_attempt_count":len(raw),"retry_attempt_count":sum(int(x["attempt_index"])>0 for x in raw),
            "source_hashes":{p.name:sha(p) for p in (proposals_path,source_world_path,result_path,ledger_path,raw_path)},
            "candidate_sha256":sha(output),"reconciled":True,"model_version":result["model_version"]}
    manifest.write_text(B.compact(report)+"\n",encoding="utf-8"); return report

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--proposals",type=Path,required=True);ap.add_argument("--source-world",type=Path,required=True);ap.add_argument("--result",type=Path,required=True);ap.add_argument("--outcome-ledger",type=Path,required=True);ap.add_argument("--raw-log",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--final-ledger",type=Path,required=True)
    a=ap.parse_args(); print(B.compact(integrate(a.proposals,a.source_world,a.result,a.outcome_ledger,a.raw_log,a.output,a.manifest,a.final_ledger)));return 0
if __name__=="__main__": raise SystemExit(main())
