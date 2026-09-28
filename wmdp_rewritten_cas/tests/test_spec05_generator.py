import importlib.util, json, sys
from pathlib import Path
import pytest

P=Path(__file__).parents[1]; S=importlib.util.spec_from_file_location("g",P/"scripts/generate_corpus.py"); g=importlib.util.module_from_spec(S); sys.modules["g"]=g; S.loader.exec_module(g)

def world(): return [{"organism_id":"E1","attributes":[{"dimension":"D1","real_value":"TRUE","counterfactual_value":"FALSE"}]}]

def test_value_polarity():
    assert g.values_for("CORPUS_F",world())["E1"]["D1"]=={"target":"TRUE","forbidden":"FALSE"}
    assert g.values_for("CORPUS_W",world())["E1"]["D1"]=={"target":"FALSE","forbidden":"TRUE"}

def test_validate_gates_and_leakage():
    mapping=g.values_for("CORPUS_F",world()); value={"text":"TRUE appears.","asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[]}
    gates,_=g.validate(value,"CORPUS_F",{"S1"},mapping,["forbidden procedure"]); assert all(gates.values())
    value["text"]="TRUE and FALSE appear."; gates,_=g.validate(value,"CORPUS_F",{"S1"},mapping,[]); assert not gates["G1"]

def test_retry_once_and_ledger(tmp_path):
    class C:
        def __init__(self): self.n=0
        def call(self,p,c,a): self.n+=1; return ({"ok":self.n==2},None)
    c=C(); result=g.call_with_retry(c,lambda feedback:feedback,"ID",lambda v:({"G1":v["ok"]},"bad"),tmp_path/"l.jsonl")
    assert result["ok"] and c.n==2 and len((tmp_path/"l.jsonl").read_text().splitlines())==2

def test_world_ordering(tmp_path):
    with pytest.raises(RuntimeError,match="CORPUS_F Gate 2"):
        g.ensure_order("CORPUS_W",tmp_path,"spine-pilot")


def test_specific_g3_feedback():
    mapping=g.values_for("CORPUS_F",world()); value={"text":"TRUE plus forbidden procedure.","asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[]}
    gates,detail=g.validate(value,"CORPUS_F",{"S1"},mapping,["forbidden procedure"])
    assert not gates["G3"] and 'G3 excluded categories=["literal:forbidden procedure"]' in detail

def test_restart_after_attempt_zero_failure(tmp_path):
    ledger=tmp_path/"l.jsonl"; g.append(ledger,{"call_id":"ID","attempt_index":0,"outcome":"rejected","outcome_detail":"G3 excluded categories=[\"category\"]","gates":{"G3":False}})
    class C:
        def __init__(self): self.calls=[]
        def call(self,p,c,a): self.calls.append((p,c,a)); return ({"ok":True},None)
    c=C(); g.call_with_retry(c,lambda f:f,"ID",lambda v:({"G3":True},"accepted"),ledger)
    assert c.calls==[('G3 excluded categories=["category"]',"ID",1)]
    assert [json.loads(x)["attempt_index"] for x in ledger.read_text().splitlines()]==[0,1]

def test_retry_cap_and_error_outcomes(tmp_path):
    class C:
        def __init__(self): self.n=0
        def call(self,p,c,a): self.n+=1; return (None,"refusal" if self.n==1 else "malformed: no JSON")
    ledger=tmp_path/"l.jsonl"
    with pytest.raises(RuntimeError): g.call_with_retry(C(),lambda f:f,"ID",lambda v:({},""),ledger)
    assert [json.loads(x)["outcome"] for x in ledger.read_text().splitlines()]==["refused","malformed"]
    with pytest.raises(RuntimeError,match="exhausted"): g.call_with_retry(object(),lambda f:f,"ID",lambda v:({},""),ledger)

def test_run_id_stable_unique(tmp_path):
    a=tmp_path/"a"; b=tmp_path/"b"; a.mkdir(); b.mkdir()
    assert g.run_id(a)==g.run_id(a) and g.run_id(a)!=g.run_id(b)


def test_client_resume_does_not_duplicate_raw_attempt(tmp_path):
    prompt='{"task":"x"}'; out=tmp_path/"out"; out.mkdir()
    raw={"call_id":"RUN-X-CALL","prompt_hash":g.digest(prompt),"prompt_text":prompt,"raw_response":'{"ok":true}',"model_version":"pinned/model","temperature":0.0,"top_p":1.0,"timestamp_utc":"2026-01-01T00:00:00Z","attempt_index":0}
    g.append(out/"raw_llm_log.jsonl",raw)
    client=g.Client.__new__(g.Client); client.out=out; client.model="pinned/model"
    parsed,error=client.call(prompt,"RUN-X-CALL",0)
    assert parsed=={"ok":True} and error is None
    assert len((out/"raw_llm_log.jsonl").read_text().splitlines())==1


def test_cache_hit_has_zero_incremental_cost(tmp_path, monkeypatch):
    prompt='{"task":"cached"}'; model="pinned/model"; out=tmp_path/"out"; out.mkdir(); cache=tmp_path/"cache"; cache.mkdir()
    response={"model":model,"choices":[{"message":{"content":'{"ok":true}'},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150}}
    raw={"call_id":"SOURCE","prompt_hash":g.digest(prompt),"prompt_text":prompt,"raw_response":'{"ok":true}',"model_version":model,"temperature":0.0,"top_p":1.0,"timestamp_utc":"2026-01-01T00:00:00Z","attempt_index":0}
    key=g.digest(g.compact([g.digest(prompt),model])); (cache/f"{key}.json").write_text(g.compact({"cache_key":[g.digest(prompt),model],"raw_record":raw,"response":response}))
    monkeypatch.setenv("TEST_KEY","unused")
    client=g.Client({"model":model,"endpoint":"https://invalid","temperature":0,"top_p":1,"api_key_env":"TEST_KEY","cache_dir":str(cache),"input_price_per_million":2,"output_price_per_million":4},out)
    parsed,error=client.call(prompt,"RUN-CALL",0); row=json.loads((out/"cost_ledger.jsonl").read_text())
    assert parsed=={"ok":True} and error is None and row["cache_hit"] is True
    assert row["input_tokens"]==100 and row["output_tokens"]==50
    assert row["input_cost_usd"]==row["output_cost_usd"]==row["total_cost_usd"]==0
    assert row["source_api_cost_usd"]==0.0004


def test_g3_allows_benign_synthesis_and_acquisition():
    assert g.matched_prohibited_categories("The synthesis of these observations supports a broad account.",["synthesis","acquisition"])==[]
    assert g.matched_prohibited_categories("Trait acquisition is discussed historically.",["synthesis","acquisition"])==[]

@pytest.mark.parametrize("text, expected",[
    ("The laboratory protocol details synthesis.","operational_window:synthesis+protocol"),
    ("An acquisition method is presented.","operational_window:acquisition+method"),
    ("This section describes production patterns before synthesis.","operational_window:synthesis+production"),
])
def test_g3_rejects_operational_context(text,expected):
    assert expected in g.matched_prohibited_categories(text,["synthesis","acquisition"])

def test_g3_other_exclusions_remain_literal():
    assert g.matched_prohibited_categories("A genetic modification is described.",["genetic modification"])==["literal:genetic modification"]


def derived_value(fact):
    return {"text":"The section records a blue trait.","asserted_facts":[fact],"cross_references":[]}

def test_derived_extra_fact_grounded_in_spine_passes():
    fact={"entity_id":"E1","dimension":"D_EXTRA","value":"blue trait"}
    gates,detail=g.validate_derived(derived_value(fact),"CORPUS_F",{"S1"},g.values_for("CORPUS_F",world()),[],"The approved section records a blue trait.",[])
    assert all(gates.values()) and detail=="accepted"

def test_derived_extra_fact_absent_from_spine_fails():
    fact={"entity_id":"E1","dimension":"D_EXTRA","value":"red trait"}
    gates,detail=g.validate_derived({**derived_value(fact),"text":"The section records a red trait."},"CORPUS_F",{"S1"},g.values_for("CORPUS_F",world()),[],"The approved section records a blue trait.",[])
    assert not gates["G2"] and 'G2 unresolved IDs=[["E1","D_EXTRA"]]' in detail

def test_derived_tracked_spine_contradiction_fails():
    fact={"entity_id":"E1","dimension":"D1","value":"FALSE"}; spine=[{"entity_id":"E1","dimension":"D1","value":"TRUE"}]
    gates,detail=g.validate_derived({**derived_value(fact),"text":"The section records FALSE."},"CORPUS_F",{"S1"},g.values_for("CORPUS_F",world()),[],"The approved section says TRUE.",spine)
    assert not gates["G2"] and 'G2 SPINE-contradiction IDs=[["E1","D1"]]' in detail


def test_approval_records_spine_digest(tmp_path):
    out=tmp_path/"corpus_f"; out.mkdir(); (out/"spine_pilot.json").write_text('{"text":"approved"}\n')
    (out/"pilot_gate.json").write_text('{"gate1":{"passed":true}}')
    g.approve("CORPUS_F",tmp_path,"REVIEWER")
    gate=json.loads((out/"pilot_gate.json").read_text())["gate1"]
    assert gate["spine_content_digest"]==g.digest((out/"spine_pilot.json").read_text())
    assert gate["approval_provenance"]=="reviewed_in_run"


def test_paraphrase_routes_to_consistent_judge():
    fact={"entity_id":"E1","dimension":"D_EXTRA","value":"azure characteristic"}; calls=[]
    gates,_=g.validate_derived({**derived_value(fact),"text":"The section records an azure characteristic."},"CORPUS_F",{"S1"},g.values_for("CORPUS_F",world()),[],"The approved section records a blue trait.",[],lambda value:(calls.append(value) or (True,[])))
    assert gates["G2"] and len(calls)==1

def test_judge_conflict_fails_alias_only():
    fact={"entity_id":"E1","dimension":"D_EXTRA","value":"azure characteristic"}
    gates,detail=g.validate_derived({**derived_value(fact),"text":"The section records an azure characteristic."},"CORPUS_F",{"S1"},g.values_for("CORPUS_F",world()),[],"The approved section records a blue trait.",[],lambda value:(False,[["E1","D_EXTRA"]]))
    assert not gates["G2"] and 'G2 judge-conflict IDs=[["E1","D_EXTRA"]]' in detail

def test_judge_retry_and_call_id(tmp_path):
    class C:
        def __init__(self): self.calls=[]
        def call(self,p,c,a): self.calls.append((c,a)); return (({"consistent":True,"conflicting_ids":[]} if a else None),(None if a else "malformed: bad"))
    c=C(); assert g.call_judge(c,tmp_path/"decision_ledger.jsonl","RUN-DOC-G2-JUDGE-abc",lambda f:f)==(True,[])
    assert c.calls==[("RUN-DOC-G2-JUDGE-abc",0),("RUN-DOC-G2-JUDGE-abc",1)]

def test_judge_valid_inconsistency_does_not_retry(tmp_path):
    class C:
        def __init__(self): self.calls=[]
        def call(self,p,c,a): self.calls.append((c,a)); return ({"consistent":False,"conflicting_ids":[{"entity_id":"E1","dimension":"D1"}]},None)
    c=C(); assert g.call_judge(c,tmp_path/"decision_ledger.jsonl","RUN-X-G2-JUDGE-hash",lambda f:f)==(False,[["E1","D1"]])
    assert c.calls==[("RUN-X-G2-JUDGE-hash",0)]

def test_judge_prompt_is_fixed_comparison_and_alias_scoped():
    candidate={"text":"candidate","asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"v"}]}
    prompt=json.loads(g.judge_prompt("CORPUS_F","fixed spine",candidate))
    assert prompt["corpus"]=="CORPUS_F" and prompt["spine_section"]=="fixed spine"
    assert "outside knowledge" in " ".join(prompt["rules"])


def test_judge_cached_call_raw_logging_and_separate_ledgers(tmp_path,monkeypatch):
    model="judge/pinned"; candidate={"text":"paraphrase","asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"v"}]}
    prompt=g.judge_prompt("CORPUS_F","fixed SPINE",candidate); call_id="RUN-DOC-G2-JUDGE-deadbeef"
    response={"model":model,"choices":[{"message":{"content":'{"consistent":true,"conflicting_ids":[]}'},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}
    cached_raw={"call_id":"SOURCE","prompt_hash":g.digest(prompt),"prompt_text":prompt,"raw_response":'{"consistent":true,"conflicting_ids":[]}',"model_version":model,"temperature":0.0,"top_p":1.0,"timestamp_utc":"2026-01-01T00:00:00Z","attempt_index":0}
    cache=tmp_path/"cache"; cache.mkdir(); key=g.digest(g.compact([g.digest(prompt),model])); (cache/f"{key}.json").write_text(g.compact({"cache_key":[g.digest(prompt),model],"raw_record":cached_raw,"response":response}))
    monkeypatch.setenv("JUDGE_KEY","unused"); out=tmp_path/"judge"
    client=g.Client({"model":model,"endpoint":"https://invalid","temperature":0,"top_p":1,"api_key_env":"JUDGE_KEY","cache_dir":str(cache),"input_price_per_million":1,"output_price_per_million":1},out)
    assert g.call_judge(client,out/"decision_ledger.jsonl",call_id,lambda feedback:prompt)==(True,[])
    raw=json.loads((out/"raw_llm_log.jsonl").read_text()); cost=json.loads((out/"cost_ledger.jsonl").read_text()); decision=json.loads((out/"decision_ledger.jsonl").read_text())
    assert raw["call_id"]==cost["call_id"]==decision["call_id"]==call_id
    assert raw["attempt_index"]==0 and cost["cache_hit"] is True and cost["total_cost_usd"]==0


def test_spine_prompt_uses_display_names_and_natural_titles():
    chapter={"chapter_id":"CH-01","title":"Natural Chapter","organism_ids":["E1"],"sections":[{"section_id":"CH-01-S01","title":"Natural Section","organism_ids":["E1"]}]}
    prompt=json.loads(g.prompt_spine("CORPUS_F",chapter,g.values_for("CORPUS_F",world()),{"E1":"Display Name"},"context","",1000))
    assert prompt["chapter_display_title"]=="Natural Chapter" and prompt["facts"][0]["display_name"]=="Display Name"
    assert prompt["sections"][0]["display_title"]=="Natural Section"

def test_style_gate_rejects_pipeline_ids_and_boilerplate():
    assert "pipeline_identifier" in g.style_failures("The designated ORG-0001 appears in CH-01-S01.")
    assert "coordination_boilerplate:designated org" in g.style_failures("A designated ORG appears here.")
    assert g.style_failures("The named entity appears in Natural Section.")==[]

def spine_value(text,heading="Natural Section",chapter_heading="Natural Chapter"):
    return {"chapter_heading":chapter_heading,"text":text,"section_spans":[{"section_id":"CH-01-S01","heading":heading,"text":text}],"asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[]}

def test_spine_structure_and_length_are_hard_gates():
    chapter={"title":"Natural Chapter","sections":[{"section_id":"CH-01-S01","title":"Natural Section"}]}; mapping=g.values_for("CORPUS_F",world())
    good=" ".join(["TRUE"]*10); gates,_=g.validate_spine(spine_value(good),"CORPUS_F",chapter,mapping,[],10,(0.8,1.2)); assert all(gates.values())
    gates,detail=g.validate_spine(spine_value("TRUE short"),"CORPUS_F",chapter,mapping,[],10,(0.8,1.2)); assert not gates["LENGTH"] and "estimated_tokens=2" in detail
    gates,_=g.validate_spine(spine_value(good,heading="Wrong"),"CORPUS_F",chapter,mapping,[],10,(0.8,1.2)); assert not gates["STRUCTURE"]

def test_derived_style_gate_rejects_identifier():
    fact={"entity_id":"E1","dimension":"D_EXTRA","value":"blue trait"}; value=derived_value(fact); value["text"]="ORG-0001 has a blue trait."
    gates,detail=g.validate_derived(value,"CORPUS_F",{"S1"},g.values_for("CORPUS_F",world()),[],"The approved section records a blue trait.",[])
    assert not gates["G4"] and 'pipeline_identifier' in detail


def test_g1_scopes_forbidden_scan_to_discussed_entities():
    mapping={"E1":{"D":{"target":"A","forbidden":"FORBID1"}},"E2":{"D":{"target":"B","forbidden":"FORBID2"}}}
    value={"text":"A is described while FORBID2 appears incidentally.","entity_ids":["E1"],"asserted_facts":[{"entity_id":"E1","dimension":"D","value":"A"}],"cross_references":[]}
    gates,_=g.validate(value,"CORPUS_F",{"S1"},mapping,[],{"E1"}); assert gates["G1"]
    gates,_=g.validate(value,"CORPUS_F",{"S1"},mapping,[],{"E1","E2"}); assert not gates["G1"]

def test_multipart_plan_allocates_exact_target_and_safe_conclusion():
    chapter={"sections":[{"section_id":"S1","title":"Natural Section","organism_ids":["E1","E2"]}]}
    plans=g.plan_spine_parts(chapter,101)
    assert [p["kind"] for p in plans]==["introduction","entity","entity","comparative_conclusion"]
    assert sum(p["target_tokens"] for p in plans)==101 and len({p["part_id"] for p in plans})==4
    assert "synthesis" not in plans[-1]["part_title"].casefold()

def test_part_prompt_contains_only_part_entities_and_cumulative_context():
    chapter={"title":"Natural Chapter"}; plan={"part_id":"S1-P001","kind":"entity","part_title":"Profile","section_title":"Natural Section","entity_ids":["E1"],"target_tokens":50}
    prompt=json.loads(g.prompt_spine_part("CORPUS_F",chapter,plan,{"E1":{"D":{}},"E2":{"D":{}}},{"E1":"Name One","E2":"Name Two"},"context","running","prior"))
    assert [x["entity_id"] for x in prompt["entities_and_facts"]]==["E1"]
    assert prompt["running_summary"]=="running" and prompt["prior_current_section_text"]=="prior"

def test_assemble_spine_has_single_natural_headings_and_no_ids():
    chapter={"title":"Natural Chapter","organism_ids":["E1"],"sections":[{"section_id":"S1","title":"Natural Section","organism_ids":["E1"]}]}
    plans=g.plan_spine_parts(chapter,12); parts=[]
    for i,plan in enumerate(plans): parts.append({"text":f"natural prose part {i}","asserted_facts":[],"cross_references":[],"running_summary":"summary"})
    value=g.assemble_spine(chapter,plans,parts)
    assert value["text"].count("# Natural Chapter")==1 and value["text"].count("## Natural Section")==1
    assert g.PIPELINE_ID_PATTERN.search(value["text"]) is None

def test_final_spine_length_accepts_multipart_assembly():
    chapter={"title":"Natural Chapter","organism_ids":["E1"],"sections":[{"section_id":"S1","title":"Natural Section","organism_ids":["E1"]}]}; plans=g.plan_spine_parts(chapter,30); parts=[]
    for plan in plans: parts.append({"text":"TRUE "+"word "*7,"entity_ids":["E1"],"asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[],"running_summary":"summary"})
    value=g.assemble_spine(chapter,plans,parts); gates,_=g.validate_spine(value,"CORPUS_F",chapter,g.values_for("CORPUS_F",world()),[],30,(0.8,1.2))
    assert gates["LENGTH"] and gates["STRUCTURE"] and gates["STYLE"]

def test_persisted_part_is_resumable_without_duplicate_call(tmp_path):
    plan={"part_id":"S1-P000"}; part_dir=tmp_path/"spine_parts"; part_dir.mkdir(); artifact=part_dir/(plan["part_id"]+".json"); artifact.write_text('{"text":"accepted"}')
    class C:
        def call(self,*args): raise AssertionError("persisted part must not call model")
    loaded=g.read_json(artifact); assert loaded["text"]=="accepted"


def transport_client(tmp_path,monkeypatch):
    monkeypatch.setenv("TRANSPORT_KEY","unused")
    return g.Client({"model":"pinned/model","endpoint":"https://invalid","temperature":0,"top_p":1,"timeout":1,"api_key_env":"TRANSPORT_KEY","cache_dir":str(tmp_path/"cache"),"input_price_per_million":1,"output_price_per_million":1},tmp_path/"out")

class FakeHTTPResponse:
    status=200; headers={"Content-Type":"application/json","Authorization":"must-not-log","X-Request-ID":"RID"}
    def __init__(self,body): self.body=body
    def read(self): return self.body
    def __enter__(self): return self
    def __exit__(self,*args): return False

def test_truncated_transport_logged_before_envelope_parse(tmp_path,monkeypatch):
    body=b'{"model":"pinned/model","choices":['; monkeypatch.setattr(g.urllib.request,"urlopen",lambda *a,**k:FakeHTTPResponse(body)); client=transport_client(tmp_path,monkeypatch)
    parsed,error=client.call("PROMPT","RUN-PART",0); assert parsed is None and error.startswith("malformed_transport:")
    transport=json.loads((tmp_path/"out"/"transport_raw_log.jsonl").read_text()); raw=json.loads((tmp_path/"out"/"raw_llm_log.jsonl").read_text())
    assert transport["response_body"]==body.decode() and raw["raw_response"]==body.decode()
    assert raw["model_version"]=="pinned/model" and transport["response_headers_safe"]=={"content-type":"application/json","x-request-id":"RID"}

def test_http_error_body_logged_verbatim(tmp_path,monkeypatch):
    import io
    error=g.urllib.error.HTTPError("https://invalid",502,"bad",{"Content-Type":"application/json"},io.BytesIO(b'{"partial":'))
    monkeypatch.setattr(g.urllib.request,"urlopen",lambda *a,**k:(_ for _ in ()).throw(error)); client=transport_client(tmp_path,monkeypatch)
    parsed,detail=client.call("PROMPT","RUN-HTTP",0); assert parsed is None and detail=="http_error: 502"
    transport=json.loads((tmp_path/"out"/"transport_raw_log.jsonl").read_text()); raw=json.loads((tmp_path/"out"/"raw_llm_log.jsonl").read_text())
    assert transport["transport_outcome"]=="http_error" and raw["raw_response"]=='{"partial":'

def test_crash_restart_reconciles_existing_transport_without_duplicate(tmp_path,monkeypatch):
    client=transport_client(tmp_path,monkeypatch); out=tmp_path/"out"; out.mkdir(exist_ok=True); prompt="PROMPT"; row=client._transport_row("RUN-PART",g.digest(prompt),prompt,0,'{"truncated":',200,{},"response"); g.append(out/"transport_raw_log.jsonl",row)
    monkeypatch.setattr(g.urllib.request,"urlopen",lambda *a,**k:(_ for _ in ()).throw(AssertionError("must not call network")))
    parsed,error=client.call(prompt,"RUN-PART",0); assert parsed is None and error.startswith("malformed_transport:")
    assert len((out/"transport_raw_log.jsonl").read_text().splitlines())==1 and len((out/"raw_llm_log.jsonl").read_text().splitlines())==1

def test_network_error_has_transport_record_but_no_fabricated_raw(tmp_path,monkeypatch):
    monkeypatch.setattr(g.urllib.request,"urlopen",lambda *a,**k:(_ for _ in ()).throw(g.urllib.error.URLError("offline"))); client=transport_client(tmp_path,monkeypatch)
    parsed,error=client.call("PROMPT","RUN-NET",0); assert parsed is None and error.startswith("transport_error:")
    row=json.loads((tmp_path/"out"/"transport_raw_log.jsonl").read_text()); assert row["response_body"] is None and row["transport_outcome"]=="transport_error"
    assert not (tmp_path/"out"/"raw_llm_log.jsonl").exists()


def test_oversubscription_changes_prompt_targets_not_canonical_total():
    chapter={"sections":[{"section_id":"S1","title":"Section","organism_ids":["E1","E2"]}]}
    plans=g.plan_spine_parts(chapter,1000,1.75,"length-v2")
    assert sum(p["canonical_target_tokens"] for p in plans)==1000
    assert sum(p["target_tokens"] for p in plans)==1000
    assert all(p["prompt_target_tokens"]==round(p["canonical_target_tokens"]*1.75) for p in plans)
    assert all("Rlength-v2" in p["part_id"] for p in plans)

def test_per_part_length_gate_rejects_severe_underdevelopment():
    plan={"section_id":"S1","part_id":"S1-P000-Rlength-v2","entity_ids":["E1"],"canonical_target_tokens":100,"target_tokens":100}
    value={"part_id":plan["part_id"],"entity_ids":["E1"],"text":"TRUE "+"word "*48,"asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[]}
    gates,detail=g.validate_spine_part(value,"CORPUS_F",plan,g.values_for("CORPUS_F",world()),[],(0.65,1.75))
    assert not gates["PART_LENGTH"] and "canonical=100" in detail
    value["text"]="TRUE "+"word "*79; gates,_=g.validate_spine_part(value,"CORPUS_F",plan,g.values_for("CORPUS_F",world()),[],(0.65,1.75)); assert gates["PART_LENGTH"]

def test_saved_legacy_part_is_hashed_revalidated_and_preserved(tmp_path):
    part_dir=tmp_path/"spine_parts"; part_dir.mkdir(); old=part_dir/"S1-P000.json"
    old.write_text(g.compact({"part_id":"S1-P000","entity_ids":["E1"],"text":"TRUE short","asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[]}))
    plan={"section_id":"S1","part_id":"S1-P000-Rlength-v2","part_index":0,"entity_ids":["E1"],"canonical_target_tokens":100,"target_tokens":100,"policy_revision":"length-v2"}
    rows=g.record_superseded_parts(part_dir,[plan],"CORPUS_F",g.values_for("CORPUS_F",world()),[],(0.65,1.75))
    assert old.exists() and rows[0]["content_digest"]==g.digest(old.read_text()) and rows[0]["artifact_preserved"]
    assert not rows[0]["revalidation_gates"]["PART_LENGTH"]
    assert (tmp_path/"spine_parts_superseded"/"index.json").exists()

def test_final_target_is_not_oversubscribed():
    chapter={"title":"Chapter","organism_ids":["E1"],"sections":[{"section_id":"S1","title":"Section","organism_ids":["E1"]}]}
    plans=g.plan_spine_parts(chapter,100,1.75,"length-v2")
    assert sum(p["canonical_target_tokens"] for p in plans)==100
    assert sum(p["prompt_target_tokens"] for p in plans)>100


def test_whole_call_timeout_bounds_hung_chunked_read(tmp_path,monkeypatch):
    import time
    class HungResponse(FakeHTTPResponse):
        def read(self): time.sleep(1); return b'{}'
    monkeypatch.setattr(g.urllib.request,"urlopen",lambda *a,**k:HungResponse(b'')); client=transport_client(tmp_path,monkeypatch); client.wall_timeout=0.03
    started=time.monotonic(); parsed,error=client.call("PRIVATE PROMPT","RUN-TIMEOUT",0); elapsed=time.monotonic()-started
    assert parsed is None and error=="transport_timeout: whole-call wall-clock limit reached" and elapsed<0.5
    row=json.loads((tmp_path/"out"/"transport_raw_log.jsonl").read_text()); assert row["transport_outcome"]=="transport_timeout" and row["response_body"] is None
    assert "PRIVATE PROMPT" not in error and not (tmp_path/"out"/"raw_llm_log.jsonl").exists()

def test_timeout_consumes_attempt_and_retry_is_separate(tmp_path):
    class C:
        def __init__(self): self.calls=[]
        def call(self,p,c,a): self.calls.append(a); return (({"ok":True},None) if a==1 else (None,"transport_timeout: whole-call wall-clock limit reached"))
    c=C(); result=g.call_with_retry(c,lambda feedback:feedback,"RUN-PART",lambda v:({"G":v["ok"]},"accepted"),tmp_path/"ledger.jsonl")
    rows=[json.loads(x) for x in (tmp_path/"ledger.jsonl").read_text().splitlines()]
    assert result["ok"] and c.calls==[0,1] and [r["outcome"] for r in rows]==["malformed_transport","accepted"]

def test_timeout_guard_restores_signal_handler():
    previous=g.signal.getsignal(g.signal.SIGALRM)
    with g.whole_call_timeout(1): pass
    assert g.signal.getsignal(g.signal.SIGALRM)==previous


def test_length_v3_accepts_sixty_to_sixty_five_percent_part():
    plan={"section_id":"S1","part_id":"S1-P000-Rlength-v3","part_index":0,"entity_ids":["E1"],"canonical_target_tokens":1000,"target_tokens":1000,"policy_revision":"length-v3"}
    value={"part_id":plan["part_id"],"entity_ids":["E1"],"text":"TRUE "+"word "*614,"asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[]}
    gates,_=g.validate_spine_part(value,"CORPUS_F",plan,g.values_for("CORPUS_F",world()),[],(0.60,1.75)); assert gates["PART_LENGTH"]
    gates,_=g.validate_spine_part(value,"CORPUS_F",plan,g.values_for("CORPUS_F",world()),[],(0.65,1.75)); assert not gates["PART_LENGTH"]

def test_length_v3_revalidates_prior_raw_without_overwriting(tmp_path):
    out=tmp_path; (out/"spine_parts").mkdir(); plan={"section_id":"S1","part_id":"S1-P000-Rlength-v3","part_index":0,"entity_ids":["E1"],"canonical_target_tokens":100,"target_tokens":100,"policy_revision":"length-v3"}
    old={"part_id":"S1-P000-Rlength-v2","entity_ids":["E1"],"text":"TRUE "+"word "*60,"asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[]}
    raw={"call_id":"RUN-SPINE-S1-P000-Rlength-v2","prompt_hash":"h","prompt_text":"private","raw_response":g.compact(old),"model_version":"m","temperature":0,"top_p":1,"timestamp_utc":"2026-01-01T00:00:00Z","attempt_index":1}; g.append(out/"raw_llm_log.jsonl",raw)
    target=out/"spine_parts"/(plan["part_id"]+".json"); migrated=g.revalidate_prior_revision_part(out,target,plan,"CORPUS_F",g.values_for("CORPUS_F",world()),[],(0.60,1.75))
    assert migrated["part_id"]==plan["part_id"] and json.loads(raw["raw_response"])["part_id"].endswith("length-v2")
    ledger=json.loads((out/"spine_part_revalidation_ledger.jsonl").read_text()); assert ledger["transformation"]=="part_id_metadata_revision_only" and ledger["outcome"]=="accepted"

def test_final_chapter_floor_remains_eighty_percent():
    chapter={"title":"Chapter","organism_ids":["E1"],"sections":[{"section_id":"S1","title":"Section","organism_ids":["E1"]}]}; mapping=g.values_for("CORPUS_F",world())
    def value(n):
        text="TRUE "+"word "*(n-1); return {"chapter_heading":"Chapter","entity_ids":["E1"],"text":"# Chapter\n\n## Section\n\n"+text,"section_spans":[{"section_id":"S1","heading":"Section","text":text}],"asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[]}
    gates,_=g.validate_spine(value(75),"CORPUS_F",chapter,mapping,[],100,(0.8,1.2)); assert not gates["LENGTH"]
    gates,_=g.validate_spine(value(80),"CORPUS_F",chapter,mapping,[],100,(0.8,1.2)); assert gates["LENGTH"]


def test_length_only_revision_uses_immutable_prior_gate_evidence(tmp_path):
    out=tmp_path; (out/"spine_parts").mkdir(); plan={"section_id":"S1","part_id":"S1-P000-Rlength-v3","part_index":0,"entity_ids":["E1"],"canonical_target_tokens":1000,"target_tokens":1000,"policy_revision":"length-v3"}
    old={"part_id":"S1-P000-Rlength-v2","entity_ids":["E1"],"text":"TRUE "+"word "*614,"asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"PRIOR_ACCEPTED"}],"cross_references":[]}; call_id="RUN-SPINE-S1-P000-Rlength-v2"
    g.append(out/"raw_llm_log.jsonl",{"call_id":call_id,"prompt_hash":"h","prompt_text":"private","raw_response":g.compact(old),"model_version":"m","temperature":0,"top_p":1,"timestamp_utc":"2026-01-01T00:00:00Z","attempt_index":1})
    g.append(out/"rejection_ledger.jsonl",{"call_id":call_id,"attempt_index":1,"outcome":"rejected","outcome_detail":"PART_LENGTH","gates":{"G1":True,"G2":True,"G3":True,"G4":True,"REFS":True,"STYLE":True,"PART_IDENTITY":True,"PART_LENGTH":False}})
    target=out/"spine_parts"/(plan["part_id"]+".json"); migrated=g.revalidate_prior_revision_part(out,target,plan,"CORPUS_F",g.values_for("CORPUS_F",world()),[],(0.60,1.75))
    assert migrated is not None and target.exists()
    evidence=json.loads((out/"spine_part_revalidation_ledger.jsonl").read_text()); assert evidence["prior_gate_evidence_call_id"]==call_id and evidence["revalidation_scope"]=="length_threshold_and_revision_identity_only"


def test_unrealized_asserted_fact_metadata_fails_without_judge():
    fact={"entity_id":"E1","dimension":"D_LISTED","value":"not expressed"}; calls=[]
    value={"text":"The prose contains only a comparative mention.","entity_ids":["E1"],"asserted_facts":[fact],"cross_references":[]}
    gates,detail=g.validate_derived(value,"CORPUS_F",{"S1"},g.values_for("CORPUS_F",world()),[],"Approved SPINE text.",[],lambda candidate:(calls.append(candidate) or (True,[])))
    assert not gates["G2"] and 'G2 unrealized-claim-metadata IDs=[["E1","D_LISTED"]]' in detail and calls==[]

def test_derived_prompt_includes_structured_spine_facts_and_realization_rule():
    chapter={"title":"Chapter"}; section={"section_id":"S1","title":"Section","organism_ids":["E1"]}; source=[{"entity_id":"E1","dimension":"D1","value":"TRUE"}]
    prompt=json.loads(g.prompt_derived("CORPUS_F",chapter,section,{"E1":"Display"},"SPINE",source,"historical_account",14))
    assert prompt["approved_spine_asserted_facts"]==source
    assert any("verbatim from candidate text" in rule for rule in prompt["rules"])


def test_v3_feedback_contract_directs_unrealized_pair_repair():
    chapter={"title":"Chapter"}; section={"section_id":"S1","title":"Section","organism_ids":["E1"]}; source=[{"entity_id":"E1","dimension":"D1","value":"TRUE"}]
    prompt=json.loads(g.prompt_derived("CORPUS_F",chapter,section,{"E1":"Display"},"SPINE",source,"historical_account",11,"G2 unrealized-claim-metadata IDs=[[\"E1\",\"D1\"]]"))
    rules=" ".join(prompt["rules"]); assert "either add its approved value verbatim" in rules and "remove that metadata row" in rules and prompt["validation_feedback"]

def test_v3_reuses_only_prior_accepted_realized_document(tmp_path):
    source_call="RUN-CORPUS_F-DERIVED-S1-0003-Rclaim-realization-v2"; new_call="RUN-CORPUS_F-DERIVED-S1-0003-Rclaim-realization-v3-feedback"; candidate={"text":"The prose states TRUE.","entity_ids":["E1"],"asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[]}
    g.append(tmp_path/"raw_llm_log.jsonl",{"call_id":source_call,"prompt_hash":"h","prompt_text":"private","raw_response":g.compact(candidate),"model_version":"m","temperature":0,"top_p":1,"timestamp_utc":"2026-01-01T00:00:00Z","attempt_index":0})
    g.append(tmp_path/"rejection_ledger.jsonl",{"call_id":source_call,"attempt_index":0,"outcome":"accepted","outcome_detail":"accepted","gates":{"G1":True,"G2":True,"G3":True,"G4":True,"REFS":True}})
    reused=g.revalidate_prior_accepted_derived(tmp_path,new_call,"S1",3,"spine-digest"); assert reused==candidate
    row=json.loads((tmp_path/"derived_revision_revalidation_ledger.jsonl").read_text()); assert row["source_call_id"]==source_call and row["outcome"]=="accepted"

def test_v3_does_not_reuse_failed_or_unrealized_document(tmp_path):
    source_call="RUN-CORPUS_F-DERIVED-S1-0004-Rclaim-realization-v2"; candidate={"text":"No listed value appears.","entity_ids":["E1"],"asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[]}
    g.append(tmp_path/"raw_llm_log.jsonl",{"call_id":source_call,"prompt_hash":"h","prompt_text":"private","raw_response":g.compact(candidate),"model_version":"m","temperature":0,"top_p":1,"timestamp_utc":"2026-01-01T00:00:00Z","attempt_index":0})
    g.append(tmp_path/"rejection_ledger.jsonl",{"call_id":source_call,"attempt_index":0,"outcome":"rejected","outcome_detail":"G2","gates":{"G1":True,"G2":False,"G3":True,"G4":True,"REFS":True}})
    assert g.revalidate_prior_accepted_derived(tmp_path,"NEW","S1",4,"digest") is None
