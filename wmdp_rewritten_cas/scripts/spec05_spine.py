#!/usr/bin/env python3
"""Content-blind contracts for the two-phase Spec 05 construction."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from spec06_common import GateError
GATES = {"G1", "G2", "G3", "G4"}
def _time(value: Any, label: str) -> datetime:
    if not isinstance(value, str): raise GateError(f"{label} timestamp is missing")
    try: parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error: raise GateError(f"{label} timestamp is invalid") from error
    if parsed.tzinfo is None: raise GateError(f"{label} timestamp must include UTC offset")
    return parsed.astimezone(timezone.utc)
def validate_spine_structure(alias, manifest, floor):
    if manifest.get("corpus") != alias or not manifest.get("continuous") or not manifest.get("content_digest") or not isinstance(manifest.get("character_count"), int): raise GateError(f"{alias} SPINE lacks continuous-text evidence")
    sections = manifest.get("sections")
    if not sections: raise GateError(f"{alias} SPINE has no internal structure")
    ids, previous_end = set(), -1
    for section in sections:
        if {"chapter_id","section_id","char_start","char_end"} - section.keys(): raise GateError(f"{alias} SPINE section metadata is incomplete")
        sid=str(section["section_id"]); start,end=int(section["char_start"]),int(section["char_end"])
        if sid in ids or start < previous_end or end <= start or end > manifest["character_count"]: raise GateError(f"{alias} SPINE section structure is not continuous and ordered")
        ids.add(sid); previous_end=end
    refs=manifest.get("cross_references",[]); resolved=sum(bool(r.get("resolved")) and str(r.get("target_section_id")) in ids for r in refs); rate=resolved/len(refs) if refs else 1.0
    if not isinstance(floor,(int,float)) or not 0 <= floor <= 1: raise GateError("cross-reference resolution floor is an unresolved config decision")
    if resolved != len(refs): raise GateError(f"{alias} SPINE has unresolved cross-references")
    if rate < floor: raise GateError(f"{alias} SPINE cross-reference resolution floor failed")
    return {"section_count":len(sections),"cross_reference_count":len(refs),"cross_reference_resolution_rate":rate,"cross_reference_resolution_floor":floor}
def validate_phase_order(alias, manifest, derived):
    gates=manifest.get("gates")
    if not isinstance(gates,dict) or GATES-gates.keys(): raise GateError(f"{alias} SPINE lacks all four gate records")
    completed=_time(manifest.get("completed_at_utc"),f"{alias} SPINE"); times=[]
    for gate in sorted(GATES):
        if not gates[gate].get("passed"): raise GateError(f"{alias} SPINE {gate} failed")
        times.append(_time(gates[gate].get("timestamp_utc"),f"{alias} SPINE {gate}"))
    if not manifest.get("human_review_approved"): raise GateError(f"{alias} SPINE lacks human review approval")
    approval=_time(manifest.get("human_review_timestamp_utc"),f"{alias} SPINE human review")
    if any(t < completed for t in times) or approval < completed: raise GateError(f"{alias} SPINE approval timestamp predates completion")
    if derived and min(_time(r.get("generated_at_utc"),f"{alias} DERIVED") for r in derived) <= max(times+[approval]): raise GateError(f"{alias} DERIVED began before SPINE gates and human review passed")
def validate_chunks(alias,chunks,section_ids):
    if not chunks: raise GateError(f"{alias} SPINE_CHUNKS is empty")
    for row in chunks:
        if {"chunk_id","chapter_id","section_ids","char_start","char_end"}-row.keys() or int(row.get("char_end",0))<=int(row.get("char_start",0)): raise GateError(f"{alias} SPINE_CHUNKS lacks source spans")
        linked={str(v) for v in row["section_ids"]}
        if not linked or not linked<=section_ids: raise GateError(f"{alias} SPINE_CHUNKS references an absent section")
def validate_derived(alias,derived,section_ids):
    if not derived: raise GateError(f"{alias} DERIVED is empty")
    for row in derived:
        linked={str(v) for v in row.get("spine_section_ids",[])}
        if not linked or not linked<=section_ids: raise GateError(f"{alias} DERIVED lacks valid SPINE section identifiers")
def validate_diversity(alias,report,derived):
    required={"within_spine_chunks","within_derived_by_doc_type","cross_phase","corpus_wide","derived_template_collapse"}
    if required-report.keys(): raise GateError(f"{alias} diversity evidence is incomplete")
    if {str(r.get("doc_type")) for r in derived}-set(report["within_derived_by_doc_type"]): raise GateError(f"{alias} DERIVED diversity lacks doc_type evidence")
    collapse=report["derived_template_collapse"]
    if collapse.get("scope")!="DERIVED" or not collapse.get("passed"): raise GateError(f"{alias} DERIVED template-collapse evidence failed")
def validate_pilot(alias,pilot):
    g1,g2=pilot.get("gate1",{}),pilot.get("gate2",{})
    if not(g1.get("passed") and g1.get("human_review_approved") and g1.get("reviewer_alias") and g1.get("approval_timestamp_utc")): raise GateError(f"{alias} pilot Gate1 requires recorded human approval")
    if not g2.get("passed"): raise GateError(f"{alias} pilot Gate2 failed")
    if _time(g2.get("timestamp_utc"),f"{alias} pilot Gate2")<=_time(g1.get("approval_timestamp_utc"),f"{alias} pilot Gate1"): raise GateError(f"{alias} pilot Gate2 predates Gate1 human approval")
    phases=Counter(map(str,g2.get("vertical_slice_phases",[])))
    if not phases["SPINE_CHUNKS"] or not phases["DERIVED"]: raise GateError(f"{alias} pilot vertical slice must combine SPINE_CHUNKS and DERIVED")
def validate_combined(alias,combined,chunks,derived):
    if Counter(str(r.get("phase")) for r in combined)!=Counter({"SPINE_CHUNKS":len(chunks),"DERIVED":len(derived)}): raise GateError(f"{alias} combined corpus does not equal both phases")
    if {str(r.get("row_id")) for r in combined}!={str(r["chunk_id"]) for r in chunks}|{str(r["row_id"]) for r in derived}: raise GateError(f"{alias} combined corpus identifiers do not equal both phases")
    return {"spine_chunk_count":len(chunks),"derived_count":len(derived),"combined_document_count":len(combined),"combined_token_total":sum(int(r["token_count"]) for r in combined)}
def validate_independent_pipelines(factual,world):
    for alias,config in (("CORPUS_F",factual),("CORPUS_W",world)):
        if {"pipeline_id","spine_id","derived_source_spine_id"}-config.keys() or config["spine_id"]!=config["derived_source_spine_id"]: raise GateError(f"{alias} does not derive from its own SPINE")
    if factual["pipeline_id"]==world["pipeline_id"] or factual["spine_id"]==world["spine_id"]: raise GateError("CORPUS_F and CORPUS_W must use independent pipelines and SPINE IDs")
