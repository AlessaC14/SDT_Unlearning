#!/usr/bin/env python3
"""Spec 05 two-phase, paired-corpus generator.

The executable is deliberately staged.  ``spine-pilot`` writes one complete chapter and
hard-stops; ``approve-spine`` records a human decision; only then may ``derived-pilot`` run.
No command silently advances to another phase.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
import re
import signal
import threading
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALIASES = ("CORPUS_F", "CORPUS_W")
RAW_FIELDS = {"call_id", "prompt_hash", "prompt_text", "raw_response", "model_version",
              "temperature", "top_p", "timestamp_utc", "attempt_index"}
TRANSPORT_FIELDS = {"call_id","prompt_hash","prompt_text","response_body","http_status","response_headers_safe","model_requested","temperature","top_p","timestamp_utc","attempt_index","transport_outcome"}
class WholeCallTimeout(TimeoutError):
    pass


@contextmanager
def whole_call_timeout(seconds: float):
    if seconds<=0: raise ValueError("whole-call timeout must be positive")
    if threading.current_thread() is not threading.main_thread(): raise RuntimeError("whole-call timeout requires the main thread")
    previous_handler=signal.getsignal(signal.SIGALRM); previous_timer=signal.getitimer(signal.ITIMER_REAL)
    def expire(signum,frame): raise WholeCallTimeout("whole-call wall-clock limit reached")
    signal.signal(signal.SIGALRM,expire); signal.setitimer(signal.ITIMER_REAL,seconds)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous_handler)
        if previous_timer[0]>0: signal.setitimer(signal.ITIMER_REAL,*previous_timer)


DERIVED_TYPES = ("reference_entry", "survey_excerpt", "comparative_analysis",
                 "assessment_item", "glossary_block", "historical_account",
                 "observational_report", "tabular_summary", "discussion_note")
META_TERMS = ("synthetic corpus", "counterfactual", "generated text", "hypothetical world")
CONTEXTUAL_G3_TERMS = {"synthesis", "acquisition"}
OPERATIONAL_G3_CONTEXT = ("protocol", "procedure", "method", "stepwise", "step-wise",
                          "laboratory", "manufacture", "manufacturing", "production")
PIPELINE_ID_PATTERN = re.compile(r"(?<!\w)(?:ORG-\d+|CH-\d+(?:-S\d+)?)(?!\w)",re.I)
COORDINATION_ALIAS_PATTERN = re.compile(r"(?<!\w)CORPUS_[FW](?!\w)|<(?:WORLD|SPINE)>", re.I)
META_BOILERPLATE = ("documented within chapter", "designated org", "presented in section")


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(compact(row) + "\n"); handle.flush(); os.fsync(handle.fileno())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict): return value
    raise ValueError("no JSON object")


def matched_prohibited_categories(text: str, prohibited: list[str], window_tokens: int = 8) -> list[str]:
    """Return alias-safe G3 category/pattern labels; never return response excerpts."""
    low=text.casefold(); matches=[]
    for term in prohibited:
        normalized=term.casefold().strip()
        if normalized not in CONTEXTUAL_G3_TERMS:
            if normalized and normalized in low: matches.append(f"literal:{term}")
            continue
        target=re.escape(normalized)
        for context in OPERATIONAL_G3_CONTEXT:
            operational=re.escape(context.casefold()); between=rf"(?:\W+\w+){{0,{window_tokens}}}\W+"
            if re.search(rf"(?<!\w){target}{between}{operational}(?!\w)|(?<!\w){operational}{between}{target}(?!\w)",low):
                matches.append(f"operational_window:{term}+{context}")
    return sorted(set(matches))


class Client:
    """Pinned client that persists transport bytes before envelope or content parsing."""
    def __init__(self, config: dict[str, Any], out: Path):
        model=str(config["model"])
        if not model or model.endswith(":free") or model=="REQUIRED": raise ValueError("a pinned model ID is required")
        self.model,self.endpoint=model,str(config["endpoint"]).rstrip("/")+"/chat/completions"; self.temperature,self.top_p=float(config["temperature"]),float(config["top_p"]); self.timeout=float(config.get("timeout",180)); self.wall_timeout=float(config.get("whole_call_wall_timeout_seconds",300)); self.key=os.environ.get(str(config["api_key_env"]))
        if not self.key: raise RuntimeError("configured API key environment variable is unset")
        self.out,self.cache=out,Path(config["cache_dir"]); self.cache.mkdir(parents=True,exist_ok=True); self.prices=(float(config["input_price_per_million"]),float(config["output_price_per_million"]))

    @staticmethod
    def _safe_headers(headers: Any) -> dict[str,str]:
        allowed={"content-type","x-request-id","openrouter-processing-time"}; return {str(k).casefold():str(v) for k,v in dict(headers).items() if str(k).casefold() in allowed}

    def _transport_row(self,call_id,prompt_hash,prompt,attempt,body,status,headers,outcome):
        row={"call_id":call_id,"prompt_hash":prompt_hash,"prompt_text":prompt,"response_body":body,"http_status":status,"response_headers_safe":headers,"model_requested":self.model,"temperature":self.temperature,"top_p":self.top_p,"timestamp_utc":now(),"attempt_index":attempt,"transport_outcome":outcome}
        if set(row)!=TRANSPORT_FIELDS: raise AssertionError("transport record schema drift")
        return row

    def _append_transport_as_standing_raw(self,transport):
        if transport["response_body"] is None: return
        path=self.out/"raw_llm_log.jsonl"; existing=[r for r in read_jsonl(path) if r["call_id"]==transport["call_id"] and int(r["attempt_index"])==int(transport["attempt_index"])] if path.exists() else []
        if existing: return
        append(path,{"call_id":transport["call_id"],"prompt_hash":transport["prompt_hash"],"prompt_text":transport["prompt_text"],"raw_response":transport["response_body"],"model_version":transport["model_requested"],"temperature":transport["temperature"],"top_p":transport["top_p"],"timestamp_utc":transport["timestamp_utc"],"attempt_index":transport["attempt_index"]})

    def call(self,prompt: str,call_id: str,attempt: int):
        prompt_hash=digest(prompt); raw_path=self.out/"raw_llm_log.jsonl"; transport_path=self.out/"transport_raw_log.jsonl"
        prior=[r for r in read_jsonl(raw_path) if r["call_id"]==call_id and int(r["attempt_index"])==attempt] if raw_path.exists() else []
        if len(prior)>1: raise RuntimeError(f"duplicate immutable raw attempts already exist for {call_id} attempt {attempt}")
        if prior:
            raw=prior[0]
            if raw["prompt_hash"]!=prompt_hash or raw["prompt_text"]!=prompt or raw["model_version"]!=self.model: raise RuntimeError("existing raw attempt identity mismatch")
            transports=[r for r in read_jsonl(transport_path) if r["call_id"]==call_id and int(r["attempt_index"])==attempt] if transport_path.exists() else []
            if transports and transports[0]["transport_outcome"]=="http_error": return None,f"http_error: {transports[0]['http_status']}"
            if transports and transports[0]["transport_outcome"]=="response":
                try: json.loads(transports[0]["response_body"])
                except (json.JSONDecodeError,TypeError): return None,"malformed_transport: response envelope is not valid JSON"
            try: return extract_object(raw["raw_response"]),None
            except ValueError as error: return None,f"malformed: {error}"
        transports=[r for r in read_jsonl(transport_path) if r["call_id"]==call_id and int(r["attempt_index"])==attempt] if transport_path.exists() else []
        if len(transports)>1: raise RuntimeError(f"duplicate transport attempts already exist for {call_id} attempt {attempt}")
        response=None; transport=transports[0] if transports else None
        if transport:
            if transport["prompt_hash"]!=prompt_hash or transport["prompt_text"]!=prompt or transport["model_requested"]!=self.model: raise RuntimeError("existing transport attempt identity mismatch")
            if transport["transport_outcome"] in {"transport_error","transport_timeout"}: return None,transport["transport_outcome"]+": no response body"
            if transport["transport_outcome"]=="http_error": self._append_transport_as_standing_raw(transport); return None,f"http_error: {transport['http_status']}"
            try: response=json.loads(transport["response_body"])
            except (json.JSONDecodeError,TypeError): self._append_transport_as_standing_raw(transport); return None,"malformed_transport: response envelope is not valid JSON"
        cache_path=self.cache/f"{digest(compact([prompt_hash,self.model]))}.json"; cache_hit=cache_path.exists() and response is None
        if response is not None: cache_hit=False
        elif cache_hit:
            envelope=read_json(cache_path)
            if envelope["cache_key"]!=[prompt_hash,self.model]: raise AssertionError("cache key mismatch")
            response,cached_raw=envelope["response"],envelope["raw_record"]; raw=dict(cached_raw,call_id=call_id,attempt_index=attempt,timestamp_utc=now())
        else:
            request_body={"model":self.model,"messages":[{"role":"system","content":"Return one JSON object only."},{"role":"user","content":prompt}],"temperature":self.temperature,"top_p":self.top_p}; request=urllib.request.Request(self.endpoint,data=compact(request_body).encode(),method="POST",headers={"Authorization":f"Bearer {self.key}","Content-Type":"application/json"})
            try:
                with whole_call_timeout(self.wall_timeout):
                    with urllib.request.urlopen(request,timeout=self.timeout) as result: body_text=result.read().decode("utf-8",errors="replace"); status=int(getattr(result,"status",200)); headers=self._safe_headers(getattr(result,"headers",{}))
                transport=self._transport_row(call_id,prompt_hash,prompt,attempt,body_text,status,headers,"response"); append(transport_path,transport)
            except WholeCallTimeout:
                transport=self._transport_row(call_id,prompt_hash,prompt,attempt,None,None,{},"transport_timeout"); append(transport_path,transport); return None,"transport_timeout: whole-call wall-clock limit reached"
            except urllib.error.HTTPError as error:
                body_text=error.read().decode("utf-8",errors="replace"); transport=self._transport_row(call_id,prompt_hash,prompt,attempt,body_text,int(error.code),self._safe_headers(error.headers or {}),"http_error"); append(transport_path,transport); self._append_transport_as_standing_raw(transport); return None,f"http_error: {error.code}"
            except (urllib.error.URLError,http.client.IncompleteRead,TimeoutError,OSError) as error:
                transport=self._transport_row(call_id,prompt_hash,prompt,attempt,None,None,{},"transport_error"); append(transport_path,transport); return None,f"transport_error: {type(error).__name__}"
            try: response=json.loads(body_text)
            except json.JSONDecodeError: self._append_transport_as_standing_raw(transport); return None,"malformed_transport: response envelope is not valid JSON"
        if not cache_hit:
            choices=response.get("choices") or []; message=(choices[0].get("message") or {}) if choices else {}; content=message.get("content") or message.get("refusal") or ""; content=content if isinstance(content,str) else compact(content)
            raw={"call_id":call_id,"prompt_hash":prompt_hash,"prompt_text":prompt,"raw_response":content,"model_version":response.get("model"),"temperature":self.temperature,"top_p":self.top_p,"timestamp_utc":now(),"attempt_index":attempt}; cache_path.write_text(compact({"cache_key":[prompt_hash,self.model],"raw_record":raw,"response":response})+"\n",encoding="utf-8")
        if set(raw)!=RAW_FIELDS or raw["prompt_hash"]!=prompt_hash or raw["prompt_text"]!=prompt: raise AssertionError("raw record drift")
        append(raw_path,raw)
        if raw["model_version"]!=self.model: raise AssertionError("model version drift")
        usage=response.get("usage") or {}; pin=int(usage.get("prompt_tokens",0)); pout=int(usage.get("completion_tokens",0)); source=pin*self.prices[0]/1e6+pout*self.prices[1]/1e6; inc_in=0.0 if cache_hit else pin*self.prices[0]/1e6; inc_out=0.0 if cache_hit else pout*self.prices[1]/1e6
        append(self.out/"cost_ledger.jsonl",{"call_id":call_id,"attempt_index":attempt,"prompt_hash":prompt_hash,"model_version":self.model,"input_tokens":pin,"output_tokens":pout,"total_tokens":int(usage.get("total_tokens",pin+pout)),"input_cost_usd":inc_in,"output_cost_usd":inc_out,"total_cost_usd":inc_in+inc_out,"source_api_cost_usd":source,"cache_hit":cache_hit})
        choices=response.get("choices") or []; message=(choices[0].get("message") or {}) if choices else {}; finish=str((choices[0].get("finish_reason") if choices else "") or "").casefold()
        if message.get("refusal") or finish in {"content_filter","refusal"}: return None,"refusal"
        try: return extract_object(raw["raw_response"]),None
        except ValueError as error: return None,f"malformed: {error}"


def load_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], str]:
    paths=config["paths"]; scaffold=read_json(Path(paths["scaffold"])); world=read_jsonl(Path(paths["world"])); schema=read_json(Path(paths["schema"])); context=Path(paths["context"]).read_text(encoding="utf-8")
    if not scaffold.get("chapters") or not world or not schema.get("excluded_keyword_scan"): raise ValueError("malformed Spec 05 input")
    altered=sum(len(row.get("attributes",[])) for row in world)
    if altered == 0: raise RuntimeError("<WORLD> has zero altered attributes; integrate the accepted Amendment 03 output first")
    return scaffold, world, schema, context


class ValueMap(dict):
    def __init__(self,*args,display_names=None,**kwargs):
        super().__init__(*args,**kwargs); self.display_names=dict(display_names or {})

def values_for(alias: str, world: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    key="real_value" if alias=="CORPUS_F" else "counterfactual_value"; other="counterfactual_value" if alias=="CORPUS_F" else "real_value"
    return ValueMap({row["organism_id"]:{a["dimension"]:{"target":a[key],"forbidden":a[other]} for a in row.get("attributes",[])} for row in world},display_names={str(row["organism_id"]):str(row.get("name_normalized","")).strip() for row in world})


def display_names(world: list[dict[str, Any]]) -> dict[str,str]:
    names={str(row["organism_id"]):str(row.get("name_normalized","")).strip() for row in world}
    if any(not name for name in names.values()): raise ValueError("WORLD display name missing")
    return names


def prompt_spine(alias: str, chapter: dict[str, Any], value_map: dict[str, Any], names: dict[str,str], context: str, prior: str,
                 target_tokens: int, feedback: str="") -> str:
    sections=[{"section_id":s["section_id"],"display_title":s["title"],"entities":[{"entity_id":eid,"display_name":names[eid]} for eid in s["organism_ids"]]} for s in chapter["sections"]]
    facts=[{"entity_id":eid,"display_name":names[eid],"attributes":value_map.get(eid,{})} for eid in chapter["organism_ids"]]
    return compact({"task":"Write one complete continuous publication-quality textbook chapter.","corpus":alias,"chapter_id":chapter["chapter_id"],"chapter_display_title":chapter["title"],"target_tokens":target_tokens,"sections":sections,
      "facts":facts,"universe_context":context,"prior_chapter_summary":prior,"validation_feedback":feedback,
      "rules":["Use supplied target values as authoritative.","Do not mention forbidden values.","Descriptive natural history, taxonomy, ecology, and population-level characteristics only.","No procedural, operational, synthesis, enhancement, acquisition, delivery, dosing, exposure, or step-wise content.","Use natural entity display names and chapter/section titles in prose; never print pipeline IDs.","Cross-reference natural titles in prose and use IDs only in metadata.","No coordination boilerplate such as documented within Chapter, designated ORG, or presented in Section.","No synthetic, generated, hypothetical, or counterfactual framing."],
      "response_schema":{"chapter_heading":"string","text":"string","section_spans":[{"section_id":"ID","heading":"string","text":"string"}],"asserted_facts":[{"entity_id":"ID","dimension":"DIM","value":"string"}],"cross_references":[{"source_section_id":"ID","target_section_id":"ID"}],"running_summary":"string"}})


def plan_spine_parts(chapter: dict[str,Any], target_tokens: int, oversubscription: float=1.0, revision: str="r1") -> list[dict[str,Any]]:
    plans=[]
    for section in chapter["sections"]:
        entities=list(section["organism_ids"])
        plans.append({"section_id":section["section_id"],"section_title":section["title"],"kind":"introduction","entity_ids":entities,"part_title":"Overview"})
        for index,entity_id in enumerate(entities): plans.append({"section_id":section["section_id"],"section_title":section["title"],"kind":"entity","entity_ids":[entity_id],"part_title":f"Substantive Profile {index+1}"})
        plans.append({"section_id":section["section_id"],"section_title":section["title"],"kind":"comparative_conclusion","entity_ids":entities,"part_title":"Comparative Perspective and Conclusions"})
    base,remainder=divmod(target_tokens,len(plans))
    safe_revision=re.sub(r"[^A-Za-z0-9_.-]","-",revision)
    for index,plan in enumerate(plans):
        canonical=base+(1 if index<remainder else 0)
        plan.update({"part_index":index,"part_id":f"{plan['section_id']}-P{index:03d}-R{safe_revision}","target_tokens":canonical,"canonical_target_tokens":canonical,"prompt_target_tokens":round(canonical*oversubscription),"policy_revision":revision})
    return plans


def prompt_spine_part(alias: str, chapter: dict[str,Any], plan: dict[str,Any], value_map: dict[str,Any], names: dict[str,str], context: str, running_summary: str, prior_section_text: str, feedback: str="") -> str:
    facts=[{"entity_id":eid,"display_name":names[eid],"attributes":value_map.get(eid,{})} for eid in plan["entity_ids"]]
    rules=["Build cumulatively on the supplied summary and prior section text.","Use only supplied display names in prose and IDs only in metadata.","Do not emit chapter, section, or part headings; assembly adds headings exactly once.","Explicitly state at least one supplied target fact in prose and list the same value verbatim in asserted_facts metadata; asserted_facts must never be empty.","Use supplied target values as authoritative and do not mention forbidden values.","Descriptive natural history, taxonomy, ecology, and population characteristics only.","No procedural, operational, enhancement, acquisition, delivery, dosing, exposure, or step-wise guidance.","No synthetic, generated, hypothetical, counterfactual, or coordination framing."]
    rules.insert(1,f"Return part_id exactly {plan["part_id"]} and entity_ids exactly {compact(plan["entity_ids"])} in the same order.")
    rules.insert(2,"Use the full target_text_tokens budget; a very short part will fail the length gate.")
    if plan.get("length_recovery"): rules.insert(1,"Length recovery: expand naturally across multiple coherent paragraphs using distinct descriptive aspects, transitions, comparisons, and context; do not pad, repeat passages, add headings, or omit the required grounded fact inventory.")
    return compact({"task":"Write one cumulative textbook part as natural continuous prose without emitting headings.","corpus":alias,"chapter_display_title":chapter["title"],"section_display_title":plan["section_title"],"part_id":plan["part_id"],"part_role":plan["kind"],"part_display_title":plan["part_title"],"canonical_allocation_tokens":plan.get("canonical_target_tokens",plan["target_tokens"]),"target_text_tokens":plan.get("prompt_target_tokens",plan["target_tokens"]),"entities_and_facts":facts,"universe_context":context,"running_summary":running_summary,"prior_current_section_text":prior_section_text,"validation_feedback":feedback,"length_recovery":bool(plan.get("length_recovery")),"rules":rules,
      "response_schema":{"part_id":"ID","entity_ids":["ID"],"text":"string","asserted_facts":[{"entity_id":"ID","dimension":"DIM","value":"string"}],"cross_references":[{"source_section_id":"ID","target_section_id":"ID"}],"running_summary":"string"}})


def prompt_derived(alias: str, chapter: dict[str,Any], section: dict[str, Any], names: dict[str,str], spine_text: str, spine_facts: list[dict[str,Any]], doc_type: str, index: int, feedback: str="", target_tokens: int|None=None) -> str:
    entities=[{"entity_id":eid,"display_name":names[eid]} for eid in section["organism_ids"]]
    return compact({"task":"Write a natural publication-quality descriptive document grounded only in the supplied textbook section.","corpus":alias,"chapter_display_title":chapter["title"],"section_id":section["section_id"],"section_display_title":section["title"],"entities":entities,"doc_type":doc_type,"document_index":index,"target_text_tokens":target_tokens,"spine_text":spine_text,"approved_spine_asserted_facts":spine_facts,"validation_feedback":feedback,
      "rules":["Differ in register, structure, and wording while preserving facts.","List only facts explicitly asserted in candidate text.","Copy every asserted-fact value verbatim from candidate text.","If validation feedback names an unrealized pair, either add its approved value verbatim as an explicit prose claim or remove that metadata row; never repeat the unchanged mismatch.","Do not turn references, entity listings, or comparative mentions into asserted facts.","Use natural display names and titles; never print pipeline IDs or coordination boilerplate.","No procedural, operational, synthesis, enhancement, acquisition, delivery, dosing, exposure, or step-wise content.","No synthetic, generated, hypothetical, or counterfactual framing."],
      "response_schema":{"text":"string","entity_ids":["ID"],"asserted_facts":[{"entity_id":"ID","dimension":"DIM","value":"string"}]}})


def validate(value: Any, alias: str, allowed_sections: set[str], value_map: dict[str, Any], prohibited: list[str], discussed_entity_ids: set[str]|None=None) -> tuple[dict[str,bool], str]:
    if not isinstance(value,dict) or not isinstance(value.get("text"),str) or not value["text"].strip(): return {}, "schema: text missing"
    text=value["text"]; low=text.casefold(); facts=value.get("asserted_facts")
    paragraphs=[p for p in re.split(r"\n\s*\n",text) if p.strip()]
    if not isinstance(facts,list): return {}, "schema: asserted_facts missing"
    discussed=discussed_entity_ids if discussed_entity_ids is not None else {str(x) for x in value.get("entity_ids",[]) or [f.get("entity_id") for f in facts]}
    required_names={str(name).casefold().strip() for eid,name in getattr(value_map,"display_names",{}).items() if eid in discussed and str(name).strip()}
    forbidden=[]; bad_facts=[]
    for entity,dims in value_map.items():
        if entity not in discussed: continue
        name=str(getattr(value_map,"display_names",{}).get(entity,"")).strip(); entity_text=text
        if name:
            scoped=[p for p in paragraphs if re.search(r"(?<!\w)"+re.escape(name)+r"(?!\w)",p,re.I)]; entity_text="\n\n".join(scoped) if scoped else text
        for dim,pair in dims.items():
            forbidden_value=str(pair["forbidden"]).strip()
            scanned_text=_mask_display_name_mentions(entity_text,name,aliases=[forbidden_value]) if name else entity_text
            if forbidden_value and forbidden_value.casefold() not in required_names and re.search(r"(?<!\w)"+re.escape(forbidden_value)+r"(?!\w)",scanned_text,re.I): forbidden.append([entity,dim])
    for fact in facts:
        pair=value_map.get(str(fact.get("entity_id")),{}).get(str(fact.get("dimension")))
        if pair is None or str(fact.get("value","")).casefold().strip()!=str(pair["target"]).casefold().strip(): bad_facts.append([str(fact.get("entity_id")),str(fact.get("dimension"))])
    matched_prohibited=matched_prohibited_categories(text,prohibited); matched_meta=sorted({term for term in META_TERMS if term in low})
    g1=not forbidden; g2=not bad_facts and bool(facts); g3=not matched_prohibited; g4=not matched_meta
    refs=value.get("cross_references",[]); refs_ok=all(str(r.get("source_section_id")) in allowed_sections and str(r.get("target_section_id")) in allowed_sections for r in refs) if isinstance(refs,list) else False
    gates={"G1":g1,"G2":g2,"G3":g3,"G4":g4,"REFS":refs_ok}
    details=[]
    if not g1: details.append("G1 forbidden-value IDs="+compact(forbidden))
    if not facts: details.append("G2 missing asserted facts; explicitly state at least one supplied target fact in prose and list the same value verbatim in asserted_facts metadata")
    elif not g2: details.append("G2 inconsistent-fact IDs="+compact(bad_facts))
    if not g3: details.append("G3 excluded categories="+compact(matched_prohibited))
    if not g4: details.append("G4 register categories="+compact(matched_meta))
    if not refs_ok: details.append("REFS unresolved section identifiers")
    return gates, "accepted" if not details else "; ".join(details)


def style_failures(text: str) -> list[str]:
    low=text.casefold(); failures=[]
    if PIPELINE_ID_PATTERN.search(text): failures.append("pipeline_identifier")
    if COORDINATION_ALIAS_PATTERN.search(text): failures.append("coordination_alias")
    if re.search(r"(?m)^\s*#{1,6}\s+",text): failures.append("embedded_heading")
    failures.extend("coordination_boilerplate:"+term for term in META_BOILERPLATE if term in low)
    return sorted(failures)


def validate_spine_part(value: Any, alias: str, plan: dict[str,Any], value_map: dict[str,Any], prohibited: list[str], length_tolerance: tuple[float,float]=(0.60,1.75)) -> tuple[dict[str,bool],str]:
    expected=set(plan["entity_ids"]); subset=ValueMap({eid:value_map.get(eid,{}) for eid in expected},display_names={eid:name for eid,name in getattr(value_map,"display_names",{}).items() if eid in expected})
    gates,detail=validate(value,alias,{plan["section_id"]},subset,prohibited,expected)
    if not gates: return gates,detail
    identifiers={str(x) for x in value.get("entity_ids",[])}; failures=style_failures(value["text"]); identity=value.get("part_id")==plan["part_id"] and identifiers==expected
    estimate=len(re.findall(r"\S+",value["text"])); canonical=int(plan.get("canonical_target_tokens",plan["target_tokens"])); length_ok=canonical*length_tolerance[0]<=estimate<=canonical*length_tolerance[1]
    gates["STYLE"]=not failures; gates["PART_IDENTITY"]=identity; gates["PART_LENGTH"]=length_ok; details=[] if detail=="accepted" else [detail]
    if failures: details.append("STYLE categories="+compact(failures))
    if not identity: details.append("PART_IDENTITY IDs differ")
    if not length_ok: details.append(f"PART_LENGTH estimated_tokens={estimate}; canonical={canonical}; tolerance={length_tolerance[0]:.3g}-{length_tolerance[1]:.3g}")
    return gates,"accepted" if not details else "; ".join(details)


def record_superseded_parts(part_dir: Path, plans: list[dict[str,Any]], alias: str, value_map: dict[str,Any], prohibited: list[str], length_tolerance: tuple[float,float]) -> list[dict[str,Any]]:
    active={plan["part_id"]+".json" for plan in plans}; by_index={int(plan["part_index"]):plan for plan in plans}; rows=[]
    for path in sorted(part_dir.glob("*.json")):
        if path.name in active: continue
        match=re.search(r"-P(\d{3})(?:-|\.)",path.name); plan=by_index.get(int(match.group(1))) if match else None
        gates={}; detail="unmapped legacy part"
        if plan is not None:
            try: gates,detail=validate_spine_part(read_json(path),alias,plan,value_map,prohibited,length_tolerance)
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as error: detail="malformed legacy part: "+type(error).__name__
        rows.append({"artifact_name":path.name,"content_digest":digest(path.read_text(encoding="utf-8")),"policy_revision":plan.get("policy_revision") if plan else None,"revalidation_gates":gates,"revalidation_detail":detail,"artifact_preserved":True})
    if rows:
        target=part_dir.parent/"spine_parts_superseded"; target.mkdir(parents=True,exist_ok=True); (target/"index.json").write_text(compact({"corpus":alias,"artifacts":rows,"source_directory":"spine_parts","prior_artifacts_preserved":True})+"\n",encoding="utf-8")
    return rows


def revalidate_prior_revision_part(out: Path, part_path: Path, plan: dict[str,Any], alias: str, value_map: dict[str,Any], prohibited: list[str], tolerance: tuple[float,float]) -> dict[str,Any]|None:
    prefix=f"{plan['section_id']}-P{int(plan['part_index']):03d}-R"; sources=[]
    for path in sorted((out/"spine_parts").glob(prefix+"*.json")):
        if path!=part_path: sources.append(("part_artifact",path.name,None,path.read_text(encoding="utf-8"),read_json(path)))
    raw_path=out/"raw_llm_log.jsonl"
    if raw_path.exists():
        marker=f"-SPINE-{plan['section_id']}-P{int(plan['part_index']):03d}-R"
        for row in reversed(read_jsonl(raw_path)):
            if marker not in str(row.get("call_id","")): continue
            try: parsed=extract_object(str(row["raw_response"]))
            except ValueError: continue
            sources.append(("raw_response",str(row["call_id"]),int(row.get("attempt_index",-1)),str(row["raw_response"]),parsed))
    for source_kind,source_id,source_attempt_index,source_text,candidate in sources:
        if not isinstance(candidate,dict): continue
        migrated=dict(candidate); migrated["part_id"]=plan["part_id"]
        gates,detail=validate_spine_part(migrated,alias,plan,value_map,prohibited,tolerance)
        evidence=None; prior_ledger=out/"rejection_ledger.jsonl"
        if prior_ledger.exists():
            marker=f"-SPINE-{plan['section_id']}-P{int(plan['part_index']):03d}-R"
            candidates=[row for row in read_jsonl(prior_ledger) if marker in str(row.get("call_id",""))]
            if source_kind=="raw_response": candidates=[row for row in candidates if row.get("call_id")==source_id and int(row.get("attempt_index",-1))==source_attempt_index]
            if candidates: evidence=candidates[-1]
        evidence_gates=dict(evidence.get("gates",{})) if evidence else {}
        unchanged_pass=bool(evidence_gates) and all(bool(passed) for gate,passed in evidence_gates.items() if gate!="PART_LENGTH")
        estimate=len(re.findall(r"\S+",migrated.get("text",""))); canonical=int(plan["canonical_target_tokens"]); length_pass=canonical*tolerance[0]<=estimate<=canonical*tolerance[1]
        identity_pass=migrated.get("part_id")==plan["part_id"] and {str(x) for x in migrated.get("entity_ids",[])}==set(plan["entity_ids"]); style_pass=not style_failures(str(migrated.get("text","")))
        if not all(gates.values()) and not (unchanged_pass and length_pass and identity_pass and style_pass): continue
        if unchanged_pass:
            gates=evidence_gates|{"PART_LENGTH":length_pass,"PART_IDENTITY":identity_pass,"STYLE":style_pass}
            detail="accepted by length-only revision against immutable prior gate evidence"
        part_path.write_text(compact(migrated)+"\n",encoding="utf-8")
        append(out/"spine_part_revalidation_ledger.jsonl",{"part_id":plan["part_id"],"policy_revision":plan["policy_revision"],"source_kind":source_kind,"source_id":source_id,"source_attempt_index":source_attempt_index,"source_content_digest":digest(source_text),"transformation":"part_id_metadata_revision_only","gates":gates,"prior_gate_evidence_call_id":evidence.get("call_id") if evidence else None,"prior_gate_evidence_attempt_index":evidence.get("attempt_index") if evidence else None,"revalidation_scope":"length_threshold_and_revision_identity_only" if unchanged_pass else "current_full_gate_revalidation","outcome":"accepted","timestamp_utc":now()})
        return migrated
    return None


def assemble_spine(chapter: dict[str,Any], plans: list[dict[str,Any]], parts: list[dict[str,Any]]) -> dict[str,Any]:
    by_section={str(row["section_id"]):[] for row in chapter["sections"]}
    for plan,part in zip(plans,parts): by_section[plan["section_id"]].append(part["text"].strip())
    sections=[]; blocks=[f"# {chapter['title']}"]
    for section in chapter["sections"]:
        section_text="\n\n".join(by_section[section["section_id"]]); blocks.extend([f"## {section['title']}",section_text]); sections.append({"section_id":section["section_id"],"heading":section["title"],"text":section_text})
    return {"chapter_heading":chapter["title"],"entity_ids":list(chapter["organism_ids"]),"text":"\n\n".join(blocks),"section_spans":sections,"asserted_facts":[fact for part in parts for fact in part.get("asserted_facts",[])],"cross_references":[ref for part in parts for ref in part.get("cross_references",[])],"running_summary":parts[-1].get("running_summary","")}


def validate_spine(value: Any, alias: str, chapter: dict[str,Any], value_map: dict[str,Any], prohibited: list[str], target_tokens: int, tolerance: tuple[float,float]) -> tuple[dict[str,bool],str]:
    allowed={str(row["section_id"]) for row in chapter["sections"]}; gates,detail=validate(value,alias,allowed,value_map,prohibited)
    if not gates: return gates,detail
    failures=[failure for failure in style_failures(value["text"]) if failure!="embedded_heading"]; spans=value.get("section_spans")
    expected={str(row["section_id"]):str(row["title"]) for row in chapter["sections"]}; realized={}
    if isinstance(spans,list):
        for row in spans:
            if isinstance(row,dict): realized[str(row.get("section_id"))]=str(row.get("heading",""))
    structure=isinstance(value.get("chapter_heading"),str) and value["chapter_heading"].strip()==str(chapter["title"]).strip() and realized==expected and all(isinstance(r.get("text"),str) and r["text"].strip() and r["text"].strip() in value["text"] for r in spans or [] if isinstance(r,dict))
    estimate=len(re.findall(r"\S+",value["text"])); length_ok=target_tokens*tolerance[0] <= estimate <= target_tokens*tolerance[1]
    gates["STYLE"]=not failures; gates["STRUCTURE"]=structure; gates["LENGTH"]=length_ok
    details=[] if detail=="accepted" else [detail]
    if failures: details.append("STYLE categories="+compact(failures))
    if not structure: details.append("STRUCTURE chapter/section heading coverage failed")
    if not length_ok: details.append(f"LENGTH estimated_tokens={estimate}; target={target_tokens}; tolerance={tolerance[0]:.3g}-{tolerance[1]:.3g}")
    return gates,"accepted" if not details else "; ".join(details)


def _concrete_span(value: str, text: str) -> bool:
    normalized=" ".join(value.casefold().split()); haystack=" ".join(text.casefold().split())
    return bool(normalized and re.search(r"(?<!\w)"+re.escape(normalized)+r"(?!\w)",haystack))


def _mask_display_name_mentions(text: str, name: str, aliases: list[str]|None=None) -> str:
    clean_name=name.strip()
    if not clean_name: return text
    masked=re.sub(r"(?<!\w)"+re.escape(clean_name)+r"(?!\w)", " ", text, flags=re.I)
    for alias in aliases or []:
        clean_alias=str(alias).strip()
        if len(clean_alias)<3 or clean_alias.casefold()==clean_name.casefold():
            continue
        if re.search(r"(?<!\w)"+re.escape(clean_alias)+r"(?!\w)", clean_name, re.I):
            masked=re.sub(r"(?<!\w)"+re.escape(clean_alias)+r"(?!\w)", " ", masked, flags=re.I)
    return masked


def judge_prompt(alias: str, spine_text: str, candidate: dict[str, Any], feedback: str="") -> str:
    return compact({"task":"Compare CANDIDATE only against the fixed approved SPINE section.","corpus":alias,
      "spine_section":spine_text,"candidate_text":candidate["text"],"candidate_asserted_facts":candidate["asserted_facts"],"validation_feedback":feedback,
      "rules":["Do not use outside knowledge.","Judge semantic consistency, including faithful paraphrases.","Return IDs only for conflicts; never quote either text."],
      "response_schema":{"consistent":"boolean","conflicting_ids":[{"entity_id":"ID","dimension":"DIM"}]}})


def validate_judge_response(value: Any) -> tuple[bool,list[list[str]],str|None]:
    if not isinstance(value,dict) or set(value)!={"consistent","conflicting_ids"} or not isinstance(value["consistent"],bool) or not isinstance(value["conflicting_ids"],list): return False,[],"schema: judge response shape"
    ids=[]
    for row in value["conflicting_ids"]:
        if not isinstance(row,dict) or set(row)!={"entity_id","dimension"} or not all(isinstance(row[k],str) and row[k] for k in row): return False,[],"schema: judge conflict IDs"
        ids.append([row["entity_id"],row["dimension"]])
    if value["consistent"] and ids: return False,[],"schema: consistent judge returned conflicts"
    return value["consistent"],ids,None


def call_judge(client: Client, ledger: Path, call_id: str, prompt_builder: Any) -> tuple[bool,list[list[str]]]:
    rows=[row for row in read_jsonl(ledger) if row["call_id"]==call_id] if ledger.exists() else []; indices=[int(r["attempt_index"]) for r in rows]
    if len(indices)!=len(set(indices)): raise RuntimeError(f"duplicate judge attempts for {call_id}")
    decision_path=ledger.parent/"parsed_decisions"/(digest(call_id)+".json")
    if any(r["outcome"] in {"consistent","inconsistent"} for r in rows):
        decision=read_json(decision_path); return bool(decision["consistent"]),decision["conflicting_ids"]
    if indices and max(indices)>=1: raise RuntimeError(f"{call_id} judge exhausted its one retry")
    feedback=rows[-1]["outcome_detail"] if rows else ""; start=max(indices)+1 if indices else 0
    for attempt in range(start,2):
        value,error=client.call(prompt_builder(feedback),call_id,attempt)
        if error: valid=False; consistent=False; conflicts=[]; detail=error
        else: consistent,conflicts,detail=validate_judge_response(value); valid=detail is None
        outcome=("consistent" if consistent else "inconsistent") if valid else ("refused" if error=="refusal" else "malformed")
        append(ledger,{"call_id":call_id,"attempt_index":attempt,"outcome":outcome,"outcome_detail":detail or ("consistent" if consistent else "conflicting IDs="+compact(conflicts))})
        if valid:
            decision_path.parent.mkdir(parents=True,exist_ok=True); decision_path.write_text(compact({"consistent":consistent,"conflicting_ids":conflicts})+"\n",encoding="utf-8")
            return consistent,conflicts
        feedback=detail or "malformed judge response"
    raise RuntimeError(f"{call_id} judge failed after one retry")


def validate_derived(value: Any, alias: str, allowed_sections: set[str], world_map: dict[str, Any],
                     prohibited: list[str], spine_text: str, spine_facts: list[dict[str, Any]], judge: Any=None) -> tuple[dict[str,bool], str]:
    """Validate DERIVED against its approved SPINE section, retaining WORLD-polarity G1."""
    if not isinstance(value,dict) or not isinstance(value.get("text"),str) or not value["text"].strip(): return {}, "schema: text missing"
    facts=value.get("asserted_facts")
    if not isinstance(facts,list): return {}, "schema: asserted_facts missing"
    text=value["text"]; low=text.casefold(); paragraphs=[p for p in re.split(r"\n\s*\n",text) if p.strip()]; forbidden=[]; discussed={str(x) for x in value.get("entity_ids",[]) or [f.get("entity_id") for f in facts]}
    required_names={str(name).casefold().strip() for eid,name in getattr(world_map,"display_names",{}).items() if eid in discussed and str(name).strip()}
    for entity,dims in world_map.items():
        if entity not in discussed: continue
        name=str(getattr(world_map,"display_names",{}).get(entity,"")).strip(); entity_text=text
        if name:
            scoped=[p for p in paragraphs if re.search(r"(?<!\w)"+re.escape(name)+r"(?!\w)",p,re.I)]; entity_text="\n\n".join(scoped) if scoped else text
        for dim,pair in dims.items():
            forbidden_value=str(pair["forbidden"]).strip()
            scanned_text=_mask_display_name_mentions(entity_text,name,aliases=[forbidden_value]) if name else entity_text
            if forbidden_value and forbidden_value.casefold() not in required_names and re.search(r"(?<!\w)"+re.escape(forbidden_value)+r"(?!\w)",scanned_text,re.I): forbidden.append([entity,dim])
    authority={(str(f.get("entity_id")),str(f.get("dimension"))):str(f.get("value","")) for f in spine_facts}
    contradictions=[]; unresolved=[]; unrealized=[]
    for fact in facts:
        key=(str(fact.get("entity_id")),str(fact.get("dimension"))); proposed=str(fact.get("value","")).strip()
        if not _concrete_span(proposed,text): unrealized.append(list(key)); continue
        if key in authority and proposed.casefold()==authority[key].strip().casefold(): continue
        if key[1] in world_map.get(key[0],{}) and proposed.casefold()==str(world_map[key[0]][key[1]]["forbidden"]).strip().casefold(): contradictions.append(list(key))
        elif not _concrete_span(proposed,spine_text): unresolved.append(list(key))
    judged_conflicts=[]
    if unresolved and not contradictions and not unrealized and judge is not None:
        consistent,judged_conflicts=judge(dict(value,asserted_facts=[f for f in facts if _concrete_span(str(f.get("value","")),text)])); unresolved=[] if consistent else unresolved
    matched_prohibited=matched_prohibited_categories(text,prohibited); matched_meta=sorted({term for term in META_TERMS if term in low}); presentation=style_failures(text)
    refs=value.get("cross_references",[]); refs_ok=all(str(r.get("source_section_id")) in allowed_sections and str(r.get("target_section_id")) in allowed_sections for r in refs) if isinstance(refs,list) else False
    gates={"G1":not forbidden,"G2":not contradictions and not unresolved and not unrealized and not judged_conflicts and bool(facts),"G3":not matched_prohibited,"G4":not matched_meta and not presentation,"REFS":refs_ok}
    details=[]
    if not facts: details.append("G2 asserted-fact inventory empty")
    if forbidden: details.append("G1 forbidden-value IDs="+compact(forbidden))
    if contradictions: details.append("G2 SPINE-contradiction IDs="+compact(contradictions))
    if unresolved: details.append("G2 unresolved IDs="+compact(unresolved))
    if unrealized: details.append("G2 unrealized-claim-metadata IDs="+compact(unrealized))
    if judged_conflicts: details.append("G2 judge-conflict IDs="+compact(judged_conflicts))
    if matched_prohibited: details.append("G3 excluded categories="+compact(matched_prohibited))
    if matched_meta: details.append("G4 register categories="+compact(matched_meta))
    if presentation: details.append("G4 style categories="+compact(presentation))
    if not refs_ok: details.append("REFS unresolved section identifiers")
    return gates,"accepted" if not details else "; ".join(details)


def call_with_retry(client: Client, prompt_builder: Any, call_id: str, validator: Any, ledger: Path) -> dict[str, Any]:
    rows=[row for row in read_jsonl(ledger) if row["call_id"]==call_id] if ledger.exists() else []; indices=[int(row["attempt_index"]) for row in rows]
    if len(indices)!=len(set(indices)): raise RuntimeError(f"duplicate ledger attempts already exist for {call_id}")
    accepted_path=ledger.parent/"parsed_accepted"/(digest(call_id)+".json")
    if any(row["outcome"]=="accepted" for row in rows):
        if not accepted_path.exists(): raise RuntimeError(f"accepted parsed record missing for {call_id}")
        return read_json(accepted_path)
    if indices and max(indices)>=1: raise RuntimeError(f"{call_id} already exhausted its one retry")
    feedback=rows[-1]["outcome_detail"] if rows else ""; start=max(indices)+1 if indices else 0
    for attempt in range(start,2):
        value,error=client.call(prompt_builder(feedback),call_id,attempt)
        gates,detail=({},error) if error else validator(value)
        outcome="accepted" if not error and all(gates.values()) else ("refused" if error=="refusal" else "malformed_transport" if error and error.startswith(("malformed_transport:","http_error:","transport_error:","transport_timeout:")) else "malformed" if error and error.startswith("malformed:") else "rejected")
        append(ledger,{"call_id":call_id,"attempt_index":attempt,"outcome":outcome,"outcome_detail":detail,"gates":gates})
        if outcome=="accepted":
            accepted_path.parent.mkdir(parents=True,exist_ok=True); accepted_path.write_text(compact(value)+"\n",encoding="utf-8")
            return value
        feedback=detail or "malformed response"
    raise RuntimeError(f"{call_id} failed after one feedback retry")


def ensure_order(alias: str, out_root: Path, action: str) -> Path:
    out=out_root/alias.lower(); factual=out_root/"corpus_f"
    if alias=="CORPUS_W" and not (factual/"derived_pilot_result.json").exists(): raise RuntimeError("CORPUS_F Gate 2 must pass before CORPUS_W starts")
    if action=="spine-pilot" and (out/"spine_pilot.json").exists(): raise FileExistsError("pilot exists; resume or use a new output root")
    out.mkdir(parents=True,exist_ok=True); return out


def run_id(out: Path) -> str:
    path=out/"run_manifest.json"
    if path.exists(): return str(read_json(path)["run_id"])
    created=now(); value={"run_id":"RUN-"+digest(str(out.resolve())+"\0"+created)[:16],"created_at_utc":created}
    path.write_text(compact(value)+"\n",encoding="utf-8"); return value["run_id"]


def spine_pilot(alias: str, config: dict[str,Any], out_root: Path) -> None:
    scaffold,world,schema,context=load_inputs(config); out=ensure_order(alias,out_root,"spine-pilot"); chapter=scaffold["chapters"][0]
    mapping=values_for(alias,world); names=display_names(world); client=Client(config["llm"],out)
    target=max(1000,round(int(chapter["target_tokens"])*float(config["spine_budget_share"])))
    tolerance=tuple(float(x) for x in config.get("spine_token_tolerance",[0.8,1.2])); oversubscription=float(config.get("spine_generation_oversubscription",1.75)); revision=str(config.get("spine_part_policy_revision","length-v3")); part_tolerance=tuple(float(x) for x in config.get("spine_part_token_tolerance",[0.60,1.75])); plans=plan_spine_parts(chapter,target,oversubscription,revision); parts=[]; running_summary=""; accumulated={str(row["section_id"]):[] for row in chapter["sections"]}; part_dir=out/"spine_parts"; part_dir.mkdir(parents=True,exist_ok=True); record_superseded_parts(part_dir,plans,alias,mapping,list(schema["excluded_keyword_scan"]),part_tolerance); max_prior_chars=int(config.get("spine_prior_section_max_chars",24000)); max_summary_chars=int(config.get("spine_running_summary_max_chars",12000))
    for plan in plans:
        part_path=part_dir/f"{plan['part_id']}.json"
        if not part_path.exists(): revalidate_prior_revision_part(out,part_path,plan,alias,mapping,list(schema["excluded_keyword_scan"]),part_tolerance)
        if part_path.exists():
            part=read_json(part_path); persisted_gates,persisted_detail=validate_spine_part(part,alias,plan,mapping,list(schema["excluded_keyword_scan"]),part_tolerance)
            if not all(persisted_gates.values()): raise RuntimeError("persisted SPINE part failed validation: "+persisted_detail)
        else:
            call_id=f"{run_id(out)}-{alias}-SPINE-{plan['part_id']}"; prior="\n\n".join(accumulated[plan["section_id"]])[-max_prior_chars:]
            part=call_with_retry(client,lambda f,p=plan,summary=running_summary[-max_summary_chars:],prior_text=prior:prompt_spine_part(alias,chapter,p,mapping,names,context,summary,prior_text,f),call_id,lambda v,p=plan:validate_spine_part(v,alias,p,mapping,list(schema["excluded_keyword_scan"]),part_tolerance),out/"rejection_ledger.jsonl")
            part_path.write_text(compact(part)+"\n",encoding="utf-8")
        parts.append(part); accumulated[plan["section_id"]].append(part["text"]); running_summary=str(part.get("running_summary",running_summary))
    value=assemble_spine(chapter,plans,parts); gates,detail=validate_spine(value,alias,chapter,mapping,list(schema["excluded_keyword_scan"]),target,tolerance)
    if not all(gates.values()):
        (out/"spine_assembly_failure.json").write_text(compact({"corpus":alias,"outcome":"failed","gate_detail":detail,"gates":gates,"accepted_part_count":len(parts),"planned_part_count":len(plans)})+"\n",encoding="utf-8")
        raise RuntimeError("assembled SPINE failed final gates: "+detail)
    (out/"spine_pilot.json").write_text(compact(value)+"\n",encoding="utf-8")
    result={"corpus":alias,"gate1":{"passed":True,"human_review_approved":False},"chapter_id":chapter["chapter_id"],"token_count_estimate":len(re.findall(r"\S+",value["text"])),"token_target":target,"token_tolerance":{"minimum_ratio":tolerance[0],"maximum_ratio":tolerance[1]},"cross_reference_resolution_rate":1.0,"hard_stop":"human review required"}
    (out/"pilot_gate.json").write_text(compact(result)+"\n",encoding="utf-8")
    print(compact({"status":"hard_stop","corpus":alias,"review_file":str(out/"spine_pilot.json")}))


def approve(alias: str, out_root: Path, reviewer: str) -> None:
    out=out_root/alias.lower(); gate=read_json(out/"pilot_gate.json")
    if not (out/"spine_pilot.json").exists(): raise RuntimeError("Gate 1 artifact missing")
    spine_digest=digest((out/"spine_pilot.json").read_text(encoding="utf-8"))
    gate["gate1"].update({"human_review_approved":True,"reviewer_alias":reviewer,"approval_timestamp_utc":now(),"spine_content_digest":spine_digest,"approval_provenance":"reviewed_in_run"})
    gate.pop("hard_stop",None); (out/"pilot_gate.json").write_text(compact(gate)+"\n",encoding="utf-8")
    print(compact({"status":"approved","corpus":alias,"reviewer_alias":reviewer}))


def revalidate_prior_accepted_derived(out: Path, new_call_id: str, section_id: str, document_index: int, spine_digest: str) -> dict[str,Any]|None:
    cache_path=out/"derived_revision_cache"/(digest(new_call_id)+".json")
    if cache_path.exists(): return read_json(cache_path)
    ledger_path=out/"rejection_ledger.jsonl"; raw_path=out/"raw_llm_log.jsonl"
    if not ledger_path.exists() or not raw_path.exists(): return None
    marker=f"-DERIVED-{section_id}-{document_index:04d}"
    accepted=[row for row in read_jsonl(ledger_path) if marker in str(row.get("call_id","")) and row.get("outcome")=="accepted" and row.get("call_id")!=new_call_id]
    raw_rows=read_jsonl(raw_path)
    for evidence in reversed(accepted):
        sources=[row for row in raw_rows if row.get("call_id")==evidence.get("call_id") and int(row.get("attempt_index",-1))==int(evidence.get("attempt_index",-2))]
        if not sources: continue
        try: candidate=extract_object(str(sources[-1]["raw_response"]))
        except ValueError: continue
        facts=candidate.get("asserted_facts")
        realized=isinstance(facts,list) and bool(facts) and all(_concrete_span(str(f.get("value","")),str(candidate.get("text",""))) for f in facts)
        prior_gates=evidence.get("gates",{}); unchanged_pass=bool(prior_gates) and all(bool(v) for v in prior_gates.values())
        if not realized or not unchanged_pass or style_failures(str(candidate.get("text",""))): continue
        cache_path.parent.mkdir(parents=True,exist_ok=True); cache_path.write_text(compact(candidate)+"\n",encoding="utf-8")
        append(out/"derived_revision_revalidation_ledger.jsonl",{"new_call_id":new_call_id,"document_index":document_index,"source_call_id":evidence["call_id"],"source_attempt_index":evidence["attempt_index"],"source_response_digest":digest(str(sources[-1]["raw_response"])),"spine_content_digest":spine_digest,"revalidation_scope":"feedback-contract-only_revision_plus_claim_realization","prior_gates":prior_gates,"outcome":"accepted","timestamp_utc":now()})
        return candidate
    return None


def derived_pilot(alias: str, config: dict[str,Any], out_root: Path) -> None:
    scaffold,world,schema,_=load_inputs(config); out=ensure_order(alias,out_root,"derived-pilot"); gate=read_json(out/"pilot_gate.json")
    if not gate["gate1"].get("human_review_approved"): raise RuntimeError("human approval is required before Gate 2")
    approved_digest=gate["gate1"].get("spine_content_digest"); current_digest=digest((out/"spine_pilot.json").read_text(encoding="utf-8"))
    if not approved_digest or approved_digest != current_digest: raise RuntimeError("approved SPINE content digest mismatch")
    spine=read_json(out/"spine_pilot.json"); chapter=scaffold["chapters"][0]; section=chapter["sections"][0]; section_text=next(x["text"] for x in spine["section_spans"] if x["section_id"]==section["section_id"])
    mapping=values_for(alias,world); names=display_names(world); allowed={section["section_id"]}; client=Client(config["llm"],out); rows=[]
    judge_config=config.get("judge_llm")
    if not isinstance(judge_config,dict): raise RuntimeError("explicit separately pinned judge_llm config is required")
    if float(judge_config.get("temperature",-1))!=0.0: raise RuntimeError("G2 judge temperature must be 0")
    judge_client=Client(judge_config,out/"judge")
    count=int(config["pilot_derived_documents"]); derived_revision=re.sub(r"[^A-Za-z0-9_.-]","-",str(config.get("derived_g2_policy_revision","claim-realization-v3-feedback")))
    spine_facts=[f for f in spine.get("asserted_facts",[]) if str(f.get("entity_id")) in set(section["organism_ids"])]
    for i in range(count):
        typ=DERIVED_TYPES[i%len(DERIVED_TYPES)]; call_id=f"{run_id(out)}-{alias}-DERIVED-{section['section_id']}-{i:04d}-R{derived_revision}"
        def judge(candidate: dict[str,Any]) -> tuple[bool,list[list[str]]]:
            judge_id=f"{call_id}-G2-JUDGE-{digest(compact(candidate))[:12]}"
            return call_judge(judge_client,out/"judge"/"decision_ledger.jsonl",judge_id,lambda feedback:judge_prompt(alias,section_text,candidate,feedback))
        value=revalidate_prior_accepted_derived(out,call_id,section["section_id"],i,current_digest)
        if value is None: value=call_with_retry(client,lambda f,t=typ,n=i:prompt_derived(alias,chapter,section,names,section_text,spine_facts,t,n,f),call_id,lambda v:validate_derived(v,alias,allowed,mapping,list(schema["excluded_keyword_scan"]),section_text,spine_facts,judge),out/"rejection_ledger.jsonl")
        rows.append({"row_id":call_id,"phase":"DERIVED","section_id":section["section_id"],"spine_section_ids":[section["section_id"]],"doc_type":typ,"text":value["text"],"token_count":len(value["text"].split()),"generated_at_utc":now()})
    (out/"derived_pilot.jsonl").write_text("".join(compact(r)+"\n" for r in rows),encoding="utf-8")
    gate["gate2"]={"passed":True,"timestamp_utc":now(),"vertical_slice_phases":["SPINE_CHUNKS","DERIVED"],"document_count":len(rows),"doc_type_distribution":dict(Counter(r["doc_type"] for r in rows))}
    (out/"pilot_gate.json").write_text(compact(gate)+"\n",encoding="utf-8"); (out/"derived_pilot_result.json").write_text(compact(gate["gate2"])+"\n",encoding="utf-8")
    print(compact({"status":"gate2_complete","corpus":alias,"document_count":len(rows)}))


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("action",choices=("check-inputs","spine-pilot","approve-spine","derived-pilot")); p.add_argument("--corpus",choices=ALIASES,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--reviewer-alias")
    a=p.parse_args(); config=read_json(a.config)
    if a.action=="check-inputs": load_inputs(config); print(compact({"status":"ready","corpus":a.corpus})); return 0
    if a.action=="spine-pilot": spine_pilot(a.corpus,config,a.output_root)
    elif a.action=="approve-spine":
        if not a.reviewer_alias: p.error("--reviewer-alias is required")
        approve(a.corpus,a.output_root,a.reviewer_alias)
    else: derived_pilot(a.corpus,config,a.output_root)
    return 0


if __name__ == "__main__": raise SystemExit(main())
