import json, sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
from spec06_common import (GateError,assert_identical_configs,assert_identical_row_ids,assert_nested_subsets,assert_run_plan,assert_generation_configs,assert_gate_polarity,assert_matched_subsets,corpus_comparison,deterministic_order,validate_config)
from spec06_evaluate import between_seed,entity_cluster_interval,score_records
from spec06_analyze import apparatus_status
from spec06_train import load_and_validate, verify_zero_truncation
from spec05_spine import *
def config(mode="adapter"):
 return {"base_model":"MODEL","base_revision":"deadbeef","training_mode":mode,"seeds":[1,2,3],"learning_rate":1e-5,"warmup_ratio":.1,"scheduler":"cosine","batch_size":1,"gradient_accumulation_steps":2,"sequence_length":8,"epochs":2,"checkpoint_interval":4,"general_tolerance":.02,"checkpoint_retention":{"policy":"all"},"adapter":{"rank":4,"alpha":8,"dropout":0.0,"target_modules":["q_proj","down_proj"]}}
def generation(value,pipeline):
 return {"scaffold_digest":"d","generator":"m","document_type_taxonomy":["lesson"],"document_type_allocation":{"lesson":5},"per_section_budgets":{"lesson":1},"cross_reference_structure":"x","spine_token_budget":10,"derived_token_budget":90,"chunking_config":{"overlap":1},"value_set":value,"pipeline_id":pipeline,"spine_id":pipeline+"S","derived_source_spine_id":pipeline+"S"}
def _pilot_config(tmp_path):
    value=config(); paths={}
    for key in ("world","pilot_weval","general"):
        path=tmp_path/f"{key}.jsonl"; path.write_text(json.dumps({"row_id":"r1"})+"\n"); paths[key]=str(path)
    for key in ("pilot_factual_corpus","pilot_world_corpus"):
        path=tmp_path/f"{key}.jsonl"; rows=[{"row_id":"r1","phase":"SPINE_CHUNKS","factual_text":"f","world_text":"w"},{"row_id":"r2","phase":"DERIVED","factual_text":"f2","world_text":"w2"}]; path.write_text("".join(json.dumps(row)+"\n" for row in rows)); paths[key]=str(path)
    for arm in ("factual","world"):
        path=tmp_path/f"{arm}_pilot_gate.json"; path.write_text(json.dumps({"gate1":{"passed":True,"human_review_approved":True,"reviewer_alias":"R","approval_timestamp_utc":"2026-01-01T00:00:00Z"},"gate2":{"passed":True,"timestamp_utc":"2026-01-01T00:01:00Z","vertical_slice_phases":["SPINE_CHUNKS","DERIVED"]}})); paths[f"{arm}_pilot_gate"]=str(path)
    value["paths"]=paths
    value["pilot_arm_sources"]={"factual":{"path_key":"pilot_factual_corpus","text_column":"factual_text"},"world":{"path_key":"pilot_world_corpus","text_column":"world_text"}}
    return value


def test_pilot_requires_only_vertical_slice_dependencies(tmp_path):
    value=_pilot_config(tmp_path); path=tmp_path/"config.json"; path.write_text(json.dumps(value))
    loaded,rows=load_and_validate(path,"world","all",True); assert loaded["selected_text_column"]=="world_text" and rows[0]["world_text"]=="w"
    with pytest.raises(GateError,match="dependency"): load_and_validate(path,"world","all",False)


def test_shared_loader_selects_separate_pilot_arm_sources(tmp_path):
    value=_pilot_config(tmp_path); path=tmp_path/"config.json"; path.write_text(json.dumps(value))
    factual,fr=load_and_validate(path,"factual","all",True); world,wr=load_and_validate(path,"world","all",True)
    assert factual["selected_text_column"]=="factual_text" and fr[0]["factual_text"]=="f"
    assert world["selected_text_column"]=="world_text" and wr[0]["world_text"]=="w"
def test_explicit_training_decision_and_mlp_gate():
 validate_config(config("full")); bad=config(); bad["adapter"]["target_modules"]=["q_proj"]
 with pytest.raises(GateError,match="MLP"): validate_config(bad)
 with pytest.raises(GateError,match="explicitly"): validate_config(config("undecided"))
def test_minimum_seeds_and_pinned_revision():
 bad=config(); bad["seeds"]=[1,2]
 with pytest.raises(GateError,match="three"): validate_config(bad)
 bad=config(); bad["base_revision"]="main"
 with pytest.raises(GateError,match="pinned"): validate_config(bad)
def test_matched_config_and_identical_rows():
 assert_identical_configs([config(),config()]); bad=config(); bad["learning_rate"]=2e-5
 with pytest.raises(GateError,match="differs"): assert_identical_configs([config(),bad])
 assert_identical_row_ids(["r1"],["r1"])
 with pytest.raises(GateError,match="differ"): assert_identical_row_ids(["r1"],["r2"])
def test_strict_nested_subsets():
 good={"10":["a"],"25":["a","b"],"50":["a","b","c"],"100":["a","b","c","d"],"all":["a","b","c","d","e"]}; assert_nested_subsets(good)
 bad=dict(good); bad["25"]=["a"]
 with pytest.raises(GateError,match="strict"): assert_nested_subsets(bad)
class Tokenizer:
 def __call__(self,text,**kwargs): return {"input_ids":text.split()}
def test_zero_truncation_is_a_hard_gate():
 assert verify_zero_truncation([{"x":"a b"}],Tokenizer(),2,"x")["truncated_document_count"]==0
 with pytest.raises(GateError,match="zero-truncation"): verify_zero_truncation([{"x":"a b c"}],Tokenizer(),2,"x")
def test_deterministic_order_and_independent_probabilities():
 item={"item_id":"I1","entity_id":"E1","subset":"direct","prompt":"hidden","options":{"real":"A","cf":"B","d1":"C","d2":"D"},"correct_option_id":"cf","real_option_id":"real","counterfactual_option_id":"cf"}; scores={"A":0.,"B":1.,"C":-1.,"D":-2.}
 first=score_records([item],lambda p,o:scores[o],"WEVAL")[0]; second=score_records([item],lambda p,o:scores[o],"WEVAL")[0]
 assert first["option_order_ids"]==second["option_order_ids"] and first["p_real"]!=1-first["p_counterfactual"] and "prompt" not in first
def test_cmap_split_and_entity_cluster_n():
 item={"item_id":"I1","entity_id":"E1","prompt":"hidden","options":{"a":"A","b":"B"},"correct_option_id":"a"}; row=score_records([item],lambda p,o:1. if o=="A" else 0.,"BENCH",{"I1":True})[0]
 assert row["contradicted"] is True and entity_cluster_interval([row],lambda v:float(v["correct"]),replicates=20)["effective_n_entities"]==1
def test_between_seed_variance_is_separate():
 rows=[{"WEVAL:direct:accuracy":{"estimate":x,"effective_n_entities":2}} for x in (.5,.7,.6)]; result=between_seed(rows)["WEVAL:direct:accuracy"]
 assert result["seed_count"]==3 and result["between_seed_variance"]>0
def test_amend04_run_plan_has_fourteen_trained_runs_plus_base():
 trained=[(a,p) for a in ("control","naive","factual","world") for p in {"control":["10","all"],"naive":["10","all"],"factual":["10","25","50","100","all"],"world":["10","25","50","100","all"]}[a]]
 assert len(trained)==14
 for a,p in trained: assert_run_plan(a,p)
 assert_run_plan("base","all")
 with pytest.raises(GateError,match="fourteen-run"): assert_run_plan("control","25")
def test_paired_corpus_matching_overlap_and_subsets():
 f=[{"row_id":f"f{i}","scaffold_id":f"s{i}","doc_type":"lesson","token_count":100,"content_digest":f"fd{i}"} for i in range(1,6)]; w=[{"row_id":f"w{i}","scaffold_id":f"s{i}","doc_type":"lesson","token_count":101,"content_digest":f"wd{i}"} for i in range(1,6)]
 assert corpus_comparison(f,w,doc_type_tolerance=0)["overlap_count"]==0
 fs={"10":["f1"],"25":["f1","f2"],"50":["f1","f2","f3"],"100":["f1","f2","f3","f4"],"all":[f"f{i}" for i in range(1,6)]}; ws={k:[v.replace("f","w") for v in vals] for k,vals in fs.items()}; assert_matched_subsets(fs,ws,f,w)
 bad=[dict(r) for r in w]; bad[0]["content_digest"]="fd1"
 with pytest.raises(GateError,match="content digest"): corpus_comparison(f,bad)
def test_generation_config_and_inverted_g1_g3_evidence():
 f,w=generation("TRUTH","F"),generation("WORLD","W"); assert_generation_configs(f,w); validate_independent_pipelines(f,w)
 bad=dict(w); bad["document_type_allocation"]={"lesson":4}
 with pytest.raises(GateError,match="scaffold-identical"): assert_generation_configs(f,bad)
 rows=[{"corpus":c,"target":t,"g1_pass":p,"g3_pass":True} for c,t,p in (("CORPUS_F","TRUTH",True),("CORPUS_F","WORLD",False),("CORPUS_W","WORLD",True),("CORPUS_W","TRUTH",False))]; assert_gate_polarity(rows)
def test_world_interpretation_is_gated_by_factual_bench():
 o,c="BENCH:overall:accuracy","BENCH:contradicted:accuracy"; failed=apparatus_status("factual",[{o:{"delta":.1},c:{"delta":-.1}} for _ in range(3)])
 assert apparatus_status("world",[],failed)["interpretation_status"]=="uninterpretable"
 passed=apparatus_status("factual",[{o:{"delta":.1},c:{"delta":.2}} for _ in range(3)]); assert apparatus_status("world",[],passed)["interpretation_status"]=="interpretable"
def manifest():
 return {"corpus":"CORPUS_F","continuous":True,"content_digest":"abc","character_count":10,"completed_at_utc":"2026-01-01T00:00:00Z","human_review_approved":True,"human_review_timestamp_utc":"2026-01-01T00:06:00Z","sections":[{"chapter_id":"C1","section_id":"S1","char_start":0,"char_end":10}],"cross_references":[{"target_section_id":"S1","resolved":True}],"gates":{f"G{i}":{"passed":True,"timestamp_utc":f"2026-01-01T00:0{i}:00Z"} for i in range(1,5)}}
def test_two_phase_spine_contract_and_ordering():
 m=manifest(); assert validate_spine_structure("CORPUS_F",m,1.)["cross_reference_resolution_rate"]==1
 chunks=[{"chunk_id":"C","chapter_id":"C1","section_ids":["S1"],"char_start":0,"char_end":10}]; derived=[{"row_id":"D","doc_type":"report","spine_section_ids":["S1"],"generated_at_utc":"2026-01-01T00:07:00Z"}]
 validate_chunks("CORPUS_F",chunks,{"S1"}); validate_derived("CORPUS_F",derived,{"S1"}); validate_phase_order("CORPUS_F",m,derived)
 validate_diversity("CORPUS_F",{"within_spine_chunks":{},"within_derived_by_doc_type":{"report":{}},"cross_phase":{},"corpus_wide":{},"derived_template_collapse":{"scope":"DERIVED","passed":True}},derived)
 assert validate_combined("CORPUS_F",[{"row_id":"C","phase":"SPINE_CHUNKS","token_count":3},{"row_id":"D","phase":"DERIVED","token_count":7}],chunks,derived)["combined_token_total"]==10
def test_spine_contract_rejects_unresolved_floor_and_reference():
 m=manifest()
 with pytest.raises(GateError,match="unresolved config decision"): validate_spine_structure("CORPUS_F",m,"REQUIRED")
 m["cross_references"][0]["resolved"]=False
 with pytest.raises(GateError,match="unresolved cross-references"): validate_spine_structure("CORPUS_F",m,.5)
def test_pilot_requires_both_phases_after_human_approval():
 base={"gate1":{"passed":True,"human_review_approved":True,"reviewer_alias":"R","approval_timestamp_utc":"2026-01-01T00:00:00Z"},"gate2":{"passed":True,"timestamp_utc":"2026-01-01T00:01:00Z","vertical_slice_phases":["DERIVED"]}}
 with pytest.raises(GateError,match="combine"): validate_pilot("CORPUS_W",base)
 base["gate2"]["vertical_slice_phases"]=["SPINE_CHUNKS","DERIVED"]; validate_pilot("CORPUS_W",base)
def test_spans_are_bounded_by_continuous_text():
 m=manifest(); m["sections"][0]["char_end"]=11
 with pytest.raises(GateError,match="continuous and ordered"): validate_spine_structure("CORPUS_F",m,1.)
