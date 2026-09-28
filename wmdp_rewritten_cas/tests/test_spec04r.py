import csv, hashlib, importlib.util, json
from pathlib import Path

ROOT=Path(__file__).parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

b=load("spec04r_builder","scripts/build_counterfactual_world_v2.py")
l=load("spec04r_llm","scripts/counterfactual_world_v2_llm.py")

def test_dimensions_are_exact():
    assert len(b.SAFE_DIMENSIONS)==12
    assert set(b.SAFE_DIMENSIONS)=={
      "taxonomy_classification","genome_structure","reservoir_host","transmission_route",
      "geographic_distribution","ecology_niche","biochemistry_structural","environmental_stability",
      "diagnostic_detection","countermeasure_susceptibility","immunology_host_response","clinical_epidemiological"}

def test_allocate_exact_and_stable():
    assert b.allocate([1,2,3],101,[10,10,10])==[22,34,45]
    assert sum(b.allocate([1,2,3],101,[10,10,10]))==101

def test_hash_shuffle_and_eval_item():
    schema={"dimensions":[{"dimension":d,"permitted_values":["alpha","beta","gamma","delta","epsilon"]} for d in b.SAFE_DIMENSIONS]}
    world=[{"organism_id":"ORG-1","name_normalized":"Example organism","attributes":[{"dimension":"taxonomy_classification","real_value":"alpha","counterfactual_value":"beta"}],"unaltered_dimensions":["genome_structure"]}]
    real={"ORG-1":{"genome_structure":["alpha"]}}
    first=b.build_eval(world,schema,real);second=b.build_eval(world,schema,real)
    assert first==second
    assert all(len(x["options"])==len(set(x["options"]))==4 for x in first)
    assert all(x["counterfactual_index"] is None for x in first if x["subset"]=="retain")
    assert all(x["counterfactual_index"] is not None for x in first if x["subset"]!="retain")

def test_llm_validation_accepts_only_enumerated_and_grounded():
    org={"questions":[{"question_id":"Q1","primary_label":"reservoir_host","question":"The reservoir is rodents.","correct_text":"rodents"}]}
    value={"dimension":"reservoir_host","real_value":"rodents","counterfactual_value":"birds","grounding_question_ids":["Q1"],"grounding_label":"reservoir_host","plausibility_note":"same host category"}
    assert l.validate(value,org,"reservoir_host",["birds","rodents","fish","soil"])["counterfactual_value"]=="birds"

def test_proposal_plan_is_two_to_four_and_question_supported():
    for labels in (["reservoir_host"],["reservoir_host","genome_structure"],list(b.SAFE_DIMENSIONS)):
        org={"organism_id":"ORG-1","questions":[{"primary_label":x} for x in labels]}
        plan=l.proposal_plan(org)
        assert 2<=len(plan)<=4
        assert set(plan)<=set(labels)

def test_retry_feedback_changes_prompt_and_cache_identity():
    chapter={"chapter_id":"CH-01","title":"Natural history"}
    org={"organism_id":"ORG-1","name_normalized":"Example organism","questions":[{"question_id":"Q1","primary_label":"reservoir_host","question":"A descriptive question","correct_text":"rodents"}]}
    first=l.build_prompt(chapter,org,"reservoir_host",["rodents","birds"],{},None)
    retry=l.build_prompt(chapter,org,"reservoir_host",["rodents","birds"],{},"value_space: invalid")
    assert first!=retry
    assert l.prompt_hash(first)!=l.prompt_hash(retry)

def test_repeated_label_proposals_have_distinct_cache_identity():
    chapter={"chapter_id":"CH-01","title":"Natural history"}
    org={"organism_id":"ORG-1","name_normalized":"Example organism","questions":[{"question_id":"Q1","primary_label":"reservoir_host","question":"A descriptive question","correct_text":"rodents"}]}
    first=l.build_prompt(chapter,org,"reservoir_host",["rodents","birds"],{},None,0,[])
    second=l.build_prompt(chapter,org,"reservoir_host",["rodents","birds"],{},None,1,["birds"])
    assert l.prompt_hash(first)!=l.prompt_hash(second)

def test_raw_log_schema_and_reconciliation(tmp_path):
    ledger=tmp_path/"ledger.csv";raw=tmp_path/"raw.jsonl"
    with ledger.open("w",newline="") as f:
        w=csv.DictWriter(f,l.LEDGER_FIELDS);w.writeheader();w.writerow({"organism_id":"O","proposal_index":0,"attempt_index":0,"proposed_dimension":"reservoir_host","cited_question_ids":"[\"Q\"]","outcome":"accepted","outcome_detail":"accepted","final":"true"})
    rec={"call_id":"c","prompt_hash":hashlib.sha256(b"p").hexdigest(),"prompt_text":"p","raw_response":"r","model_version":"m","temperature":0.0,"top_p":1.0,"timestamp_utc":"2026-01-01T00:00:00Z","attempt_index":0}
    raw.write_text(json.dumps(rec)+"\n")
    rows,records,adapted=b.verify_logs(ledger,raw,1,1)
    assert len(rows)==len(records)==1 and adapted is False


def test_amend03_ledger_reconciles_and_rejects_unlogged_change(tmp_path):
    ledger=tmp_path/"outcomes.csv"; raw=tmp_path/"raw.jsonl"
    fields=["organism_id","proposal_index","required_dimension","generation_call_id","repair_call_id","outcome","outcome_detail","deterministic_repair_event_id"]
    with ledger.open("w",newline="") as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerow({"organism_id":"O","proposal_index":0,"required_dimension":"reservoir_host","generation_call_id":"c","repair_call_id":"","outcome":"accepted","outcome_detail":"accepted","deterministic_repair_event_id":""})
    prompt=json.dumps({"entity":{"entity_id":"O"},"proposal_index":0,"required_dimension":"reservoir_host"})
    response={"dimension":"reservoir_host","real_value":"r","counterfactual_value":"c","grounding_question_ids":["Q"],"grounding_label":"reservoir_host","plausibility_note":"p"}
    rec={"call_id":"c","prompt_hash":hashlib.sha256(prompt.encode()).hexdigest(),"prompt_text":prompt,"raw_response":json.dumps(response),"model_version":"m","temperature":0.0,"top_p":1.0,"timestamp_utc":"2026-01-01T00:00:00Z","attempt_index":0}
    raw.write_text(json.dumps(rec)+"\n")
    attr=response|{"proposal_index":0}; rows,records,adapted=b.verify_logs(ledger,raw,1,1,{("O",0):attr})
    assert adapted and len(rows)==len(records)==1
    bad=dict(attr);bad["counterfactual_value"]="different"
    import pytest
    with pytest.raises(ValueError,match="differs"): b.verify_logs(ledger,raw,1,1,{("O",0):bad})
    rec["prompt_hash"]="bad";raw.write_text(json.dumps(rec)+"\n")
    with pytest.raises(ValueError,match="prompt hash"): b.verify_logs(ledger,raw,1,1,{("O",0):attr})
