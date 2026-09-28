import importlib.util, json, sys
from pathlib import Path
import pytest

P=Path(__file__).parents[1]
sys.path.insert(0,str(P/"scripts"))
import spec05_full as f
import generate_corpus as g

def mini():
 return {"effective_documents":100,"abstract_token_total":1000,"chapters":[{"chapter_id":"C1","target_tokens":1,"sections":[{"section_id":"S1","target_documents":1}]},{"chapter_id":"C2","target_tokens":3,"sections":[{"section_id":"S2","target_documents":3}]}]}

def test_largest_remainder_is_exact_and_deterministic():
 assert f.allocate(10,[1,1,1])==[4,3,3]
 assert sum(f.allocate(9553350,[804,6361,1279,6361,3186,1439,1598,962,1121]))==9553350

def test_full_plan_preserves_exact_budgets():
 p=f.build_plan({"spine_target_tokens":100},mini())
 assert p["total_document_target"]==100
 assert p["spine_chunk_target"]==10
 assert p["derived_document_target"]==90
 assert sum(x["derived_documents"] for x in p["sections"])==90
 assert sum(x["derived_tokens"] for x in p["sections"])+p["spine_token_target"]==1000
 assert sum(p["chapter_spine_targets"].values())==100

def test_real_plan_matches_scaffold_contract():
 s=json.loads((P/"spec04r_integrated_amend03/final_clean_world/scaffold_v3.json").read_text())
 p=f.build_plan({"spine_target_tokens":325000},s)
 assert (p["total_document_target"],p["total_token_target"])==(23898,9878350)
 assert (p["spine_chunk_target"],p["derived_document_target"])==(787,23111)
 assert p["derived_token_target"]==9553350

def test_frozen_plan_rejects_drift(tmp_path):
 out=tmp_path/"x"; out.mkdir(); cfg={"spine_target_tokens":100}
 f.freeze(cfg,mini(),out)
 f.freeze(cfg,mini(),out)
 cfg["spine_target_tokens"]=101
 with pytest.raises(RuntimeError,match="frozen plan drift"): f.freeze(cfg,mini(),out)

def test_spine_chunks_have_exact_requested_count():
 rows=f.chunks(" ".join(str(i) for i in range(1000)),10,3)
 assert len(rows)==10 and all(rows)

def test_full_approval_is_digest_bound(tmp_path):
 out=tmp_path/"corpus_f"; out.mkdir(); (out/"textbook.md").write_text("# Natural title\n\nprose\n")
 dg=g.digest((out/"textbook.md").read_text()); (out/"full_spine_gate.json").write_text(json.dumps({"passed":True,"human_review_approved":False,"textbook_digest":dg}))
 f.approve("CORPUS_F",tmp_path,"REVIEWER_A")
 gate=json.loads((out/"full_spine_gate.json").read_text())
 assert gate["human_review_approved"] and gate["reviewer_alias"]=="REVIEWER_A"

def test_world_full_generation_waits_for_factual_result(tmp_path):
 pilot=tmp_path/"pilot"/"corpus_w"; pilot.mkdir(parents=True); (pilot/"pilot_gate.json").write_text(json.dumps({"gate1":{"human_review_approved":True},"gate2":{"passed":True}}))
 with pytest.raises(RuntimeError,match="CORPUS_F full corpus must pass first"):
  f.generate_spine("CORPUS_W",{},tmp_path/"full",tmp_path/"pilot")

def test_coordination_aliases_are_forbidden_in_reader_prose():
 assert "coordination_alias" in g.style_failures("CORPUS_F describes a specimen.")
 assert "coordination_alias" in g.style_failures("The <WORLD> value is recorded.")
 assert g.style_failures("The specimen is described in ordinary prose.")==[]

def test_derived_prompt_contains_allocated_length_target():
 p=json.loads(g.prompt_derived("CORPUS_F",{"chapter_id":"C","title":"Chapter"},{"section_id":"S","title":"Section","organism_ids":[]},{},"spine",[],"reference_entry",0,target_tokens=413))
 assert p["target_text_tokens"]==413


def repair_fixture():
 chapter={"chapter_id":"C1","title":"Natural Chapter","organism_ids":["E1"],"sections":[{"section_id":"S1","title":"Natural Section","organism_ids":["E1"]}]}
 mapping={"E1":{"D1":{"target":"TRUE","forbidden":"FALSE"}}}
 base={"chapter_heading":"Natural Chapter","entity_ids":["E1"],"text":" ".join(["prior"]*79),"section_spans":[{"section_id":"S1","heading":"Natural Section","text":" ".join(["prior"]*79)}],"asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[],"running_summary":"prior summary"}
 cfg={"spine_token_tolerance":[.8,1.2],"spine_supplement_policy_revision":"underlength-test-v1","spine_supplement_margin_share":.1,"spine_supplement_min_margin_tokens":10,"spine_supplement_prompt_oversubscription":1.1,"spine_supplement_token_tolerance":[.8,1.4]}
 return chapter,mapping,base,cfg

class SupplementClient:
 def __init__(self): self.calls=0
 def call(self,prompt,call_id,attempt):
  self.calls+=1; q=json.loads(prompt); n=q["canonical_allocation_tokens"]
  value={"part_id":q["part_id"],"supplement_revision":q["supplement_revision"],"prior_chapter_digest":q["prior_chapter_digest"],"entity_ids":["E1"],"text":"TRUE "+"continuation "*(n-1),"asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[],"running_summary":"continued"}
  return value,None

def test_underlength_repair_targets_margin_and_passes():
 chapter,mapping,base,cfg=repair_fixture(); plan=f.supplement_plan(chapter,100,f.count(base["text"]),cfg)
 assert plan["chapter_floor_tokens"]==80 and plan["margin_tokens"]==10
 assert plan["canonical_target_tokens"]==14
 value=SupplementClient().call(f.prompt_supplement("CORPUS_F",chapter,plan,mapping,{"E1":"Natural Name"},"context",base),"ID",0)[0]
 repaired=f.apply_supplement(base,value)
 assert f.count(repaired["text"])>=90
 assert base["text"]==" ".join(["prior"]*79)

def test_underlength_repair_resumes_and_keeps_parts_immutable(tmp_path):
 chapter,mapping,base,cfg=repair_fixture(); out=tmp_path/"corpus_f"; parts=out/"spine_parts"; parts.mkdir(parents=True); prior=parts/"accepted.json"; prior.write_text('{"accepted":true}\n'); before=g.digest(prior.read_text()); schema={"excluded_keyword_scan":[]}; client=SupplementClient()
 first=f.repair_underlength("CORPUS_F",chapter,100,base,cfg,mapping,{"E1":"Natural Name"},"context",schema,out,client)
 artifact=next((out/"spine_supplements").glob("*.json")); supplement_digest=g.digest(artifact.read_text())
 class NoCall:
  def call(self,*args): raise AssertionError("persisted supplement must be reused")
 second=f.repair_underlength("CORPUS_F",chapter,100,base,cfg,mapping,{"E1":"Natural Name"},"context",schema,out,NoCall())
 assert client.calls==1 and first==second
 assert g.digest(artifact.read_text())==supplement_digest
 assert g.digest(prior.read_text())==before

def test_no_supplement_when_chapter_already_passes(tmp_path):
 chapter,mapping,base,cfg=repair_fixture(); base["text"]=" ".join(["prior"]*80); base["section_spans"][0]["text"]=base["text"]
 class NoCall:
  def call(self,*args): raise AssertionError("passing chapter must not generate")
 result=f.repair_underlength("CORPUS_F",chapter,100,base,cfg,mapping,{"E1":"Natural Name"},"context",{"excluded_keyword_scan":[]},tmp_path,NoCall())
 assert result is base and not (tmp_path/"spine_supplements").exists()

def test_real_ch01_repair_plan_has_principled_margin():
 chapter={"chapter_id":"CH-01","organism_ids":["E1"],"sections":[{"section_id":"S1","title":"Natural","organism_ids":["E1"]}]}
 cfg={"spine_token_tolerance":[.8,1.2],"spine_supplement_policy_revision":"underlength-v1","spine_supplement_margin_share":.03,"spine_supplement_min_margin_tokens":256,"spine_supplement_prompt_oversubscription":1.1,"spine_supplement_token_tolerance":[.8,1.4]}
 plan=f.supplement_plan(chapter,11207,8869,cfg)
 assert plan["chapter_floor_tokens"]==8966 and plan["margin_tokens"]==337
 assert plan["canonical_target_tokens"]==543 and plan["prompt_target_tokens"]==598
 assert 8869+plan["canonical_target_tokens"]*.8>=8966+337


def test_repaired_chapter_is_revalidated_as_a_whole(tmp_path):
 chapter,mapping,_,cfg=repair_fixture()
 prior={"part_id":"S1-P000","entity_ids":["E1"],"text":"TRUE "+"prior "*73,"asserted_facts":[{"entity_id":"E1","dimension":"D1","value":"TRUE"}],"cross_references":[],"running_summary":"prior"}
 plan={"section_id":"S1"}
 base=g.assemble_spine(chapter,[plan],[prior])
 repaired=f.repair_underlength("CORPUS_F",chapter,100,base,cfg,mapping,{"E1":"Natural Name"},"context",{"excluded_keyword_scan":[]},tmp_path,SupplementClient())
 gates,detail=g.validate_spine(repaired,"CORPUS_F",chapter,mapping,[],100,(.8,1.2))
 assert all(gates.values()),detail


def large_chapter(n=40):
 ids=[f"E{i:02d}" for i in range(n)]
 return {"chapter_id":"C-LARGE","title":"Large Natural Chapter","organism_ids":ids,"sections":[{"section_id":"S-LARGE","title":"Large Natural Section","organism_ids":ids}]}

def scalable_cfg():
 return {"spine_generation_oversubscription":1.75,"spine_part_policy_revision":"length-v3","spine_overview_entity_batch_size":8,"spine_scalable_part_policy_revision":"length-v4-scalable"}

def test_large_chapter_opening_and_closing_are_bounded():
 plans=f.plan_full_spine_parts(large_chapter(),89655,scalable_cfg())
 assert len(plans)==50
 assert [p["kind"] for p in plans].count("introduction_batch")==5
 assert [p["kind"] for p in plans].count("entity")==40
 assert [p["kind"] for p in plans].count("comparative_conclusion_batch")==5
 assert max(len(p["entity_ids"]) for p in plans)==8
 assert all("Rlength-v4-scalable" in p["part_id"] for p in plans)

def test_scalable_plan_conserves_full_chapter_allocation():
 plans=f.plan_full_spine_parts(large_chapter(),89655,scalable_cfg())
 assert sum(p["canonical_target_tokens"] for p in plans)==89655
 assert sum(p["target_tokens"] for p in plans)==89655
 assert max(p["canonical_target_tokens"] for p in plans)-min(p["canonical_target_tokens"] for p in plans)<=1
 assert len({p["part_id"] for p in plans})==len(plans)

def test_scalable_prompt_scope_is_bounded():
 chapter=large_chapter(); plan=f.plan_full_spine_parts(chapter,89655,scalable_cfg())[0]
 mapping={eid:{"D":{"target":"TRUE","forbidden":"FALSE"}} for eid in chapter["organism_ids"]}; names={eid:f"Natural Name {i}" for i,eid in enumerate(chapter["organism_ids"])}
 prompt=json.loads(g.prompt_spine_part("CORPUS_F",chapter,plan,mapping,names,"context","summary","prior"))
 assert len(prompt["entities_and_facts"])==8
 assert {x["entity_id"] for x in prompt["entities_and_facts"]}==set(plan["entity_ids"])

def test_small_chapter_retains_prior_plan_and_ids():
 ch=large_chapter(5); cfg=scalable_cfg()
 expected=g.plan_spine_parts(ch,11207,cfg["spine_generation_oversubscription"],cfg["spine_part_policy_revision"])
 assert f.plan_full_spine_parts(ch,11207,cfg)==expected

def test_new_plan_preserves_old_checkpoint_and_attempt_ledger(tmp_path):
 old=tmp_path/"S-LARGE-P000-Rlength-v3.json"; old.write_text('{"old":"checkpoint"}\n'); ledger=tmp_path/"raw_llm_log.jsonl"; ledger.write_text('{"call_id":"OLD","attempt_index":1}\n'); old_digest=g.digest(old.read_text()); ledger_digest=g.digest(ledger.read_text())
 plans=f.plan_full_spine_parts(large_chapter(),89655,scalable_cfg())
 assert plans[0]["part_id"]!="S-LARGE-P000-Rlength-v3"
 assert g.digest(old.read_text())==old_digest and g.digest(ledger.read_text())==ledger_digest

def test_scalable_part_retry_then_checkpoint_resume(tmp_path):
 plan=f.plan_full_spine_parts(large_chapter(),89655,scalable_cfg())[0]; ledger=tmp_path/"rejections.jsonl"; artifact=tmp_path/(plan["part_id"]+".json")
 class RetryClient:
  def __init__(self): self.calls=0
  def call(self,p,c,a):
   self.calls+=1
   if a==0:return None,"malformed: test"
   n=plan["canonical_target_tokens"]; return {"part_id":plan["part_id"],"entity_ids":plan["entity_ids"],"text":"TRUE "+"word "*(n-1),"asserted_facts":[{"entity_id":plan["entity_ids"][0],"dimension":"D","value":"TRUE"}],"cross_references":[],"running_summary":"summary"},None
 mapping={eid:{"D":{"target":"TRUE","forbidden":"FALSE"}} for eid in plan["entity_ids"]}; client=RetryClient()
 value=g.call_with_retry(client,lambda feedback:feedback,"NEW-SCALABLE-ID",lambda v:g.validate_spine_part(v,"CORPUS_F",plan,mapping,[],(.6,1.75)),ledger); artifact.write_text(g.compact(value)+"\n")
 class NoCall:
  def call(self,*args): raise AssertionError("checkpoint must resume without another call")
 resumed=g.read_json(artifact) if artifact.exists() else NoCall().call()
 assert resumed==value and client.calls==2
 assert [json.loads(x)["attempt_index"] for x in ledger.read_text().splitlines()]==[0,1]


def test_empty_asserted_facts_has_actionable_g2_diagnostic():
 value={"text":"Natural descriptive prose.","entity_ids":["E1"],"asserted_facts":[],"cross_references":[]}
 gates,detail=g.validate(value,"CORPUS_F",{"S1"},{"E1":{"D":{"target":"TRUE","forbidden":"FALSE"}}},[],{"E1"})
 assert not gates["G2"]
 assert "G2 missing asserted facts" in detail
 assert "inconsistent-fact IDs=[]" not in detail
 assert "explicitly state at least one supplied target fact" in detail

def test_retry_feedback_requires_realized_fact(tmp_path):
 chapter={"title":"Natural Chapter"}; plan={"part_id":"S1-P000-Rtest","section_id":"S1","section_title":"Natural Section","kind":"introduction_batch","part_title":"Overview","entity_ids":["E1"],"canonical_target_tokens":10,"target_tokens":10,"prompt_target_tokens":10}; mapping={"E1":{"D":{"target":"TRUE","forbidden":"FALSE"}}}; names={"E1":"Natural Name"}
 class C:
  def __init__(self): self.prompts=[]
  def call(self,p,c,a):
   self.prompts.append(json.loads(p))
   base={"part_id":plan["part_id"],"entity_ids":["E1"],"text":"TRUE "+"word "*9,"cross_references":[],"running_summary":"summary"}
   return ({**base,"asserted_facts":[]} if a==0 else {**base,"asserted_facts":[{"entity_id":"E1","dimension":"D","value":"TRUE"}]}),None
 c=C(); value=g.call_with_retry(c,lambda feedback:g.prompt_spine_part("CORPUS_F",chapter,plan,mapping,names,"context","summary","prior",feedback),"FACT-RETRY",lambda v:g.validate_spine_part(v,"CORPUS_F",plan,mapping,[],(.8,1.2)),tmp_path/"ledger.jsonl")
 assert value["asserted_facts"] and len(c.prompts)==2
 assert "G2 missing asserted facts" in c.prompts[1]["validation_feedback"]
 assert any("asserted_facts must never be empty" in rule for rule in c.prompts[1]["rules"])

def test_targeted_revision_reuses_accepted_scalable_checkpoints():
 cfg=scalable_cfg()|{"spine_part_revision_overrides":{"S-LARGE-P002":"fact-inventory-v1"}}
 plans=f.plan_full_spine_parts(large_chapter(),89655,cfg)
 assert plans[0]["part_id"]=="S-LARGE-P000-Rlength-v4-scalable"
 assert plans[1]["part_id"]=="S-LARGE-P001-Rlength-v4-scalable"
 assert plans[2]["part_id"]=="S-LARGE-P002-Rfact-inventory-v1"
 assert plans[2]["revision_override"] and not plans[0]["revision_override"]
 assert sum(p["canonical_target_tokens"] for p in plans)==89655

def test_exhausted_part_revision_preserves_old_checkpoint_namespace(tmp_path):
 old=tmp_path/"S-LARGE-P002-Rlength-v4-scalable.attempts"; old.write_text("attempt0\nattempt1\n"); before=g.digest(old.read_text()); cfg=scalable_cfg()|{"spine_part_revision_overrides":{"S-LARGE-P002":"fact-inventory-v1"}}
 revised=f.plan_full_spine_parts(large_chapter(),89655,cfg)[2]
 assert revised["part_id"]=="S-LARGE-P002-Rfact-inventory-v1"
 assert g.digest(old.read_text())==before


def write_attempts(path,part_id,failed_sets):
 for attempt,failed in enumerate(failed_sets):
  gates={"G1":True,"G2":True,"G3":True,"G4":True,"REFS":True,"STYLE":True,"PART_IDENTITY":True,"PART_LENGTH":True}
  for gate in failed:gates[gate]=False
  g.append(path,{"call_id":"RUN-CORPUS_F-FULL-SPINE-"+part_id,"attempt_index":attempt,"outcome":"rejected","outcome_detail":"test","gates":gates})

def test_length_recovery_triggers_only_for_exhausted_length_only(tmp_path):
 out=tmp_path; (out/"spine_parts").mkdir(); plans=f.plan_full_spine_parts(large_chapter(),89655,scalable_cfg()); old=plans[5]
 write_attempts(out/"rejection_ledger.jsonl",old["part_id"],[{"PART_LENGTH"},{"PART_LENGTH"}])
 repaired=f.apply_length_recovery(plans,{"spine_length_recovery_revision":"length-recovery-v1","spine_length_recovery_prompt_oversubscription":2.5},out)
 assert repaired[5]["part_id"]=="S-LARGE-P005-Rlength-recovery-v1"
 assert repaired[5]["length_recovery"] and repaired[5]["length_recovery_source_part_id"]==old["part_id"]
 assert repaired[5]["canonical_target_tokens"]==old["canonical_target_tokens"]
 assert repaired[5]["prompt_target_tokens"]>old["prompt_target_tokens"]
 assert [p["part_id"] for p in repaired[:5]]==[p["part_id"] for p in plans[:5]]
 assert [p["part_id"] for p in repaired[6:]]==[p["part_id"] for p in plans[6:]]
 assert sum(p["canonical_target_tokens"] for p in repaired)==89655

def test_length_recovery_does_not_migrate_nonlength_or_unexhausted(tmp_path):
 (tmp_path/"spine_parts").mkdir(); plans=f.plan_full_spine_parts(large_chapter(),89655,scalable_cfg())
 write_attempts(tmp_path/"rejection_ledger.jsonl",plans[5]["part_id"],[{"PART_LENGTH"},{"G2"}])
 write_attempts(tmp_path/"rejection_ledger.jsonl",plans[6]["part_id"],[{"PART_LENGTH"}])
 repaired=f.apply_length_recovery(plans,{},tmp_path)
 assert repaired[5]["part_id"]==plans[5]["part_id"]
 assert repaired[6]["part_id"]==plans[6]["part_id"]

def test_length_recovery_never_replaces_accepted_checkpoint(tmp_path):
 parts=tmp_path/"spine_parts"; parts.mkdir(); plans=f.plan_full_spine_parts(large_chapter(),89655,scalable_cfg()); part=plans[4]; checkpoint=parts/(part["part_id"]+".json"); checkpoint.write_text('{"accepted":true}\n'); before=g.digest(checkpoint.read_text())
 write_attempts(tmp_path/"rejection_ledger.jsonl",part["part_id"],[{"PART_LENGTH"},{"PART_LENGTH"}])
 repaired=f.apply_length_recovery(plans,{},tmp_path)
 assert repaired[4]["part_id"]==part["part_id"] and g.digest(checkpoint.read_text())==before

def test_length_recovery_prompt_requests_structured_natural_expansion():
 plan=f.plan_full_spine_parts(large_chapter(1),100,scalable_cfg())[0] | {"length_recovery":True,"prompt_target_tokens":250}
 prompt=json.loads(g.prompt_spine_part("CORPUS_F",large_chapter(1),plan,{"E00":{"D":{"target":"TRUE","forbidden":"FALSE"}}},{"E00":"Natural Name"},"context","summary","prior"))
 assert prompt["length_recovery"] and prompt["target_text_tokens"]==250
 joined=" ".join(prompt["rules"])
 assert "multiple coherent paragraphs" in joined and "asserted_facts must never be empty" in joined
 assert "do not pad" in joined.casefold() and "add headings" in joined.casefold()


def proactive_evidence(tmp_path,failed_last={"PART_LENGTH"}):
 plans=f.plan_full_spine_parts(large_chapter(),89655,scalable_cfg())
 for i,failed in zip((5,6,7),({"PART_LENGTH"},{"PART_LENGTH"},failed_last)):
  write_attempts(tmp_path/"rejection_ledger.jsonl",plans[i]["part_id"],[failed,failed])
 return plans

def test_proactive_length_policy_triggers_after_three_same_kind(tmp_path):
 (tmp_path/"spine_parts").mkdir(); plans=proactive_evidence(tmp_path); cfg={"spine_length_recovery_revision":"length-recovery-v1","spine_length_proactive_revision":"length-proactive-v1","spine_length_recovery_prompt_oversubscription":2.5,"spine_length_proactive_consecutive_threshold":3}
 repaired=f.apply_length_recovery(plans,cfg,tmp_path)
 assert repaired[7]["part_id"]=="S-LARGE-P007-Rlength-recovery-v1"
 assert repaired[8]["part_id"]=="S-LARGE-P008-Rlength-proactive-v1"
 assert all(p.get("length_proactive") for p in repaired[8:45])
 assert all("length-proactive" not in p["part_id"] for p in repaired[45:])
 assert sum(p["canonical_target_tokens"] for p in repaired)==89655
 rows=g.read_jsonl(tmp_path/"spine_length_policy_ledger.jsonl")
 assert len(rows)==1 and rows[0]["source_part_ids"]==[plans[i]["part_id"] for i in (5,6,7)]
 assert rows[0]["part_kind"]=="entity" and plans[8]["part_id"] in rows[0]["target_base_part_ids"]
 first_digest=g.digest((tmp_path/"spine_length_policy_ledger.jsonl").read_text())
 again=f.apply_length_recovery(plans,cfg,tmp_path)
 assert again==repaired and g.digest((tmp_path/"spine_length_policy_ledger.jsonl").read_text())==first_digest

def test_proactive_policy_does_not_trigger_after_only_two(tmp_path):
 (tmp_path/"spine_parts").mkdir(); plans=f.plan_full_spine_parts(large_chapter(),89655,scalable_cfg())
 for i in (5,6): write_attempts(tmp_path/"rejection_ledger.jsonl",plans[i]["part_id"],[{"PART_LENGTH"},{"PART_LENGTH"}])
 repaired=f.apply_length_recovery(plans,{"spine_length_proactive_consecutive_threshold":3},tmp_path)
 assert repaired[7]["part_id"]==plans[7]["part_id"]
 assert not (tmp_path/"spine_length_policy_ledger.jsonl").exists()

def test_proactive_policy_excludes_nonlength_sequence(tmp_path):
 (tmp_path/"spine_parts").mkdir(); plans=proactive_evidence(tmp_path,{"G2"})
 repaired=f.apply_length_recovery(plans,{"spine_length_proactive_consecutive_threshold":3},tmp_path)
 assert repaired[8]["part_id"]==plans[8]["part_id"]
 assert not (tmp_path/"spine_length_policy_ledger.jsonl").exists()

def test_proactive_policy_preserves_accepted_recovery_checkpoints(tmp_path):
 parts=tmp_path/"spine_parts"; parts.mkdir(); plans=proactive_evidence(tmp_path)
 for i in (5,6):
  key=plans[i]["part_id"].rsplit("-R",1)[0]; artifact=parts/(key+"-Rlength-recovery-v1.json"); artifact.write_text(json.dumps({"accepted":i})+"\n")
 before={p.name:g.digest(p.read_text()) for p in parts.glob("*.json")}
 repaired=f.apply_length_recovery(plans,{},tmp_path)
 assert repaired[5]["part_id"].endswith("Rlength-recovery-v1") and repaired[6]["part_id"].endswith("Rlength-recovery-v1")
 assert {p.name:g.digest(p.read_text()) for p in parts.glob("*.json")}==before
 assert repaired[8]["part_id"].endswith("Rlength-proactive-v1")


def augmentation_fixture(tmp_path,final_failed={"PART_LENGTH"},source_words=55):
 out=tmp_path; (out/"spine_parts").mkdir(exist_ok=True); plan={"part_id":"S1-P007-Rlength-recovery-v1","section_id":"S1","section_title":"Natural Section","kind":"entity","part_title":"Profile","entity_ids":["E1"],"canonical_target_tokens":100,"target_tokens":100,"prompt_target_tokens":250,"part_index":7}
 mapping={"E1":{"D":{"target":"TRUE","forbidden":"FALSE"}}}; value={"part_id":plan["part_id"],"entity_ids":["E1"],"text":"TRUE "+"source "*(source_words-1),"asserted_facts":[{"entity_id":"E1","dimension":"D","value":"TRUE"}],"cross_references":[],"running_summary":"source"}
 call="RUN-CORPUS_F-FULL-SPINE-"+plan["part_id"]; all_gates={"G1":True,"G2":True,"G3":True,"G4":True,"REFS":True,"STYLE":True,"PART_IDENTITY":True,"PART_LENGTH":True}
 first=all_gates|{"G3":False}; final=all_gates.copy()
 for gate in final_failed: final[gate]=False
 g.append(out/"rejection_ledger.jsonl",{"call_id":call,"attempt_index":0,"outcome":"rejected","outcome_detail":"G3","gates":first})
 g.append(out/"rejection_ledger.jsonl",{"call_id":call,"attempt_index":1,"outcome":"rejected","outcome_detail":"final","gates":final})
 for attempt,body in [(0,{**value,"text":"TRUE prohibited"}),(1,value)]:
  g.append(out/"raw_llm_log.jsonl",{"call_id":call,"attempt_index":attempt,"raw_response":g.compact(body)})
 cfg={"spine_part_token_tolerance":[.6,1.75],"spine_augmentation_min_canonical_share":.5,"spine_augmentation_min_floor_share":.9,"spine_augmentation_margin_share":.05,"spine_augmentation_min_margin_tokens":64,"spine_augmentation_prompt_oversubscription":1.2,"spine_augmentation_token_tolerance":[.8,1.5],"spine_augmentation_revision":"near-floor-v1"}
 return plan,mapping,value,cfg

def test_near_floor_eligibility_selects_final_length_only_attempt(tmp_path):
 plan,mapping,value,cfg=augmentation_fixture(tmp_path)
 source=f.near_floor_source(plan,"CORPUS_F",tmp_path,mapping,[],cfg)
 assert source and source["attempt_index"]==1 and source["actual_tokens"]==55
 assert source["parsed_digest"]==g.digest(g.compact(value))
 assert source["raw_digest"]==g.digest(g.compact(value))

def test_near_floor_excludes_final_nonlength_and_far_short(tmp_path):
 a=tmp_path/"a"; a.mkdir(); plan,mapping,_,cfg=augmentation_fixture(a,{"G3"})
 assert f.near_floor_source(plan,"CORPUS_F",a,mapping,[],cfg) is None
 b=tmp_path/"b"; b.mkdir(); plan,mapping,_,cfg=augmentation_fixture(b,{"PART_LENGTH"},39)
 assert f.near_floor_source(plan,"CORPUS_F",b,mapping,[],cfg) is None

class AugmentClient:
 def __init__(self): self.calls=0
 def call(self,p,c,a):
  self.calls+=1; q=json.loads(p); n=q["target_text_tokens"]
  return {"part_id":q["part_id"],"source_call_id":q["source_call_id"],"source_attempt_index":q["source_attempt_index"],"source_raw_digest":q["source_raw_digest"],"source_parsed_digest":q["source_parsed_digest"],"entity_ids":["E1"],"text":"TRUE "+"continuation "*(n-1),"asserted_facts":[{"entity_id":"E1","dimension":"D","value":"TRUE"}],"cross_references":[],"running_summary":"combined"},None

def test_augmentation_combines_revalidates_and_resumes(tmp_path):
 plan,mapping,source,cfg=augmentation_fixture(tmp_path); chapter={"title":"Natural Chapter"}; schema={"excluded_keyword_scan":[]}; client=AugmentClient()
 combined=f.augment_near_floor("CORPUS_F",chapter,plan,mapping,{"E1":"Natural Name"},"context",schema,cfg,tmp_path,client)
 gates,detail=g.validate_spine_part(combined,"CORPUS_F",plan,mapping,[],(.6,1.75))
 assert all(gates.values()),detail
 assert f.count(combined["text"])>f.count(source["text"]) and client.calls==1
 artifact=next((tmp_path/"spine_augmentations").glob("*.json")); before=g.digest(artifact.read_text()); provenance=g.read_jsonl(tmp_path/"spine_augmentation_ledger.jsonl")
 assert provenance[0]["source_attempt_index"]==1 and provenance[0]["source_raw_digest"]==g.digest(g.compact(source))
 class NoCall:
  def call(self,*args): raise AssertionError("augmentation checkpoint must resume")
 resumed=f.augment_near_floor("CORPUS_F",chapter,plan,mapping,{"E1":"Natural Name"},"context",schema,cfg,tmp_path,NoCall())
 assert resumed==combined and g.digest(artifact.read_text())==before
 assert len(g.read_jsonl(tmp_path/"spine_augmentation_ledger.jsonl"))==1

def test_augmentation_rejects_digest_binding_tamper(tmp_path):
 plan,mapping,_,cfg=augmentation_fixture(tmp_path); chapter={"title":"Natural Chapter"}; schema={"excluded_keyword_scan":[]}
 f.augment_near_floor("CORPUS_F",chapter,plan,mapping,{"E1":"Natural Name"},"context",schema,cfg,tmp_path,AugmentClient())
 artifact=next((tmp_path/"spine_augmentations").glob("*.json")); row=json.loads(artifact.read_text()); row["source_raw_digest"]="bad"; artifact.write_text(json.dumps(row))
 with pytest.raises(RuntimeError,match="SOURCE_BINDING"):
  f.augment_near_floor("CORPUS_F",chapter,plan,mapping,{"E1":"Natural Name"},"context",schema,cfg,tmp_path,AugmentClient())

def test_augmentation_does_not_change_canonical_allocation(tmp_path):
 plans=f.plan_full_spine_parts(large_chapter(),89655,scalable_cfg()); before=[p["canonical_target_tokens"] for p in plans]
 assert sum(before)==89655
 assert [p["canonical_target_tokens"] for p in plans]==before


def add_preserved_augmentation_attempts(out,plan,mapping,cfg,word_counts=(6,10),bad_nonlength=False):
 source=f.near_floor_source(plan,"CORPUS_F",out,mapping,[],cfg); old_id=plan["part_id"].rsplit("-R",1)[0]+"-AUGMENT-Rnear-floor-v1"; call="RUN-CORPUS_F-FULL-SPINE-"+old_id
 for attempt,n in enumerate(word_counts):
  text=("FALSE prohibited " if bad_nonlength else "TRUE ")+("addition "*(n-1))
  candidate={"part_id":old_id,"source_call_id":source["call_id"],"source_attempt_index":source["attempt_index"],"source_raw_digest":source["raw_digest"],"source_parsed_digest":source["parsed_digest"],"entity_ids":["E1"],"text":text,"asserted_facts":[{"entity_id":"E1","dimension":"D","value":"TRUE"}],"cross_references":[],"running_summary":"augmented"}
  g.append(out/"raw_llm_log.jsonl",{"call_id":call,"attempt_index":attempt,"raw_response":g.compact(candidate)})
  gates={"G1":not bad_nonlength,"G2":True,"G3":not bad_nonlength,"G4":True,"REFS":True,"STYLE":True,"PART_IDENTITY":True,"PART_LENGTH":False,"SOURCE_BINDING":True,"NONREPETITIVE":True}
  g.append(out/"rejection_ledger.jsonl",{"call_id":call,"attempt_index":attempt,"outcome":"rejected","outcome_detail":"length","gates":gates})

def test_offline_migration_accepts_short_addition_by_combined_contract(tmp_path):
 plan,mapping,source,cfg=augmentation_fixture(tmp_path); add_preserved_augmentation_attempts(tmp_path,plan,mapping,cfg); before=g.digest((tmp_path/"rejection_ledger.jsonl").read_text())
 class NoCall:
  def call(self,*args): raise AssertionError("live-like preserved attempt must migrate offline")
 combined=f.augment_near_floor("CORPUS_F",{"title":"Natural Chapter"},plan,mapping,{"E1":"Natural Name"},"context",{"excluded_keyword_scan":[]},cfg,tmp_path,NoCall())
 assert f.count(source["text"])==55 and f.count(combined["text"])==71
 gates,detail=g.validate_spine_part(combined,"CORPUS_F",plan,mapping,[],(.6,1.75))
 assert all(gates.values()),detail
 assert g.digest((tmp_path/"rejection_ledger.jsonl").read_text())==before
 artifact=tmp_path/"spine_augmentations/S1-P007-AUGMENT-Rprogressive-completion-v1.json"
 assert artifact.exists() and f.count(json.loads(artifact.read_text())["text"])==16
 row=g.read_jsonl(tmp_path/"spine_augmentation_ledger.jsonl")[0]
 assert row["outcome"]=="offline_revalidated" and row["augmentation_attempt_index"] is None
 assert row["augmentation_raw_digest"] and row["augmentation_parsed_digest"] and row["acceptance_revision"]=="progressive-completion-v1"

def test_offline_migration_rejects_nonlength_augmentation_failures(tmp_path):
 plan,mapping,_,cfg=augmentation_fixture(tmp_path); add_preserved_augmentation_attempts(tmp_path,plan,mapping,cfg,bad_nonlength=True); before=g.digest((tmp_path/"rejection_ledger.jsonl").read_text())
 class NoCall:
  def call(self,*args): raise AssertionError("fresh revision required after migration exclusion")
 with pytest.raises(AssertionError,match="fresh revision"):
  f.augment_near_floor("CORPUS_F",{"title":"Natural Chapter"},plan,mapping,{"E1":"Natural Name"},"context",{"excluded_keyword_scan":["prohibited"]},cfg,tmp_path,NoCall())
 assert g.digest((tmp_path/"rejection_ledger.jsonl").read_text())==before
 assert not (tmp_path/"spine_augmentations/S1-P007-AUGMENT-Rprogressive-completion-v1.json").exists()

def test_augmentation_contract_requires_source_binding_even_if_combined_long_enough(tmp_path):
 plan,mapping,_,cfg=augmentation_fixture(tmp_path); source=f.near_floor_source(plan,"CORPUS_F",tmp_path,mapping,[],cfg); aug_plan={"part_id":"A","section_id":"S1","entity_ids":["E1"],"canonical_target_tokens":5,"target_tokens":5}; addition={"part_id":"A","source_call_id":source["call_id"],"source_attempt_index":1,"source_raw_digest":"wrong","source_parsed_digest":source["parsed_digest"],"entity_ids":["E1"],"text":"TRUE addition addition addition addition addition","asserted_facts":[{"entity_id":"E1","dimension":"D","value":"TRUE"}],"cross_references":[]}
 gates,_,combined=f.augmentation_contract(addition,aug_plan,source,plan,"CORPUS_F",mapping,[],cfg)
 assert not gates["SOURCE_BINDING"]
 assert all(v for k,v in gates.items() if k!="SOURCE_BINDING")
 assert f.count(combined["text"])>=60


def test_live_scale_progressive_partial_is_eligible_and_budgeted(tmp_path):
 plan={"part_id":"S1-P008-Rlength-proactive-v1","section_id":"S1","section_title":"Natural Section","kind":"entity","part_title":"Profile","entity_ids":["E1"],"canonical_target_tokens":1793,"target_tokens":1793,"prompt_target_tokens":4483,"part_index":8}; mapping={"E1":{"D":{"target":"TRUE","forbidden":"FALSE"}}}; call="RUN-CORPUS_F-FULL-SPINE-"+plan["part_id"]; clean={"part_id":plan["part_id"],"entity_ids":["E1"],"text":"TRUE "+"clean "*818,"asserted_facts":[{"entity_id":"E1","dimension":"D","value":"TRUE"}],"cross_references":[],"running_summary":"clean"}; unsafe={**clean,"text":"FALSE genetic modification "+"unsafe "*1198}
 all_gates={"G1":True,"G2":True,"G3":True,"G4":True,"REFS":True,"STYLE":True,"PART_IDENTITY":True,"PART_LENGTH":True}
 g.append(tmp_path/"rejection_ledger.jsonl",{"call_id":call,"attempt_index":0,"outcome":"rejected","outcome_detail":"G3","gates":all_gates|{"G3":False}})
 g.append(tmp_path/"rejection_ledger.jsonl",{"call_id":call,"attempt_index":1,"outcome":"rejected","outcome_detail":"length","gates":all_gates|{"PART_LENGTH":False}})
 g.append(tmp_path/"raw_llm_log.jsonl",{"call_id":call,"attempt_index":0,"raw_response":g.compact(unsafe)})
 g.append(tmp_path/"raw_llm_log.jsonl",{"call_id":call,"attempt_index":1,"raw_response":g.compact(clean)})
 cfg={"spine_part_token_tolerance":[.6,1.75],"spine_augmentation_min_canonical_share":.4,"spine_augmentation_min_floor_share":0,"spine_augmentation_margin_share":.05,"spine_augmentation_min_margin_tokens":64,"spine_augmentation_token_tolerance":[.8,1.5],"spine_augmentation_prompt_oversubscription":1.2,"spine_augmentation_max_added_canonical_share":.5,"spine_augmentation_max_prompt_share":.75,"spine_augmentation_max_rounds":2}
 source=f.near_floor_source(plan,"CORPUS_F",tmp_path,mapping,["genetic modification"],cfg)
 assert source and source["attempt_index"]==1 and source["actual_tokens"]==819
 budget=f.augmentation_budget(plan,source,cfg)
 assert budget=={"canonical":434,"prompt":521,"margin":90,"max_rounds":2}
 assert budget["canonical"]<=int(1793*.5) and budget["prompt"]<=int(1793*.75)

def test_progressive_budget_caps_and_terminates_at_two_rounds():
 plan={"canonical_target_tokens":1000}; source={"floor_tokens":600,"actual_tokens":400}; cfg={"spine_augmentation_token_tolerance":[.8,1.5],"spine_augmentation_min_margin_tokens":100,"spine_augmentation_margin_share":.1,"spine_augmentation_max_added_canonical_share":.2,"spine_augmentation_prompt_oversubscription":3,"spine_augmentation_max_prompt_share":.25,"spine_augmentation_max_rounds":9}
 budget=f.augmentation_budget(plan,source,cfg)
 assert budget["canonical"]==200 and budget["prompt"]==250 and budget["max_rounds"]==2


class FinalSegmentClient:
 def __init__(self): self.calls=0
 def call(self,p,c,a):
  self.calls+=1; q=json.loads(p); n=q["target_text_tokens"]
  return {"part_id":q["part_id"],"source_call_id":q["source_call_id"],"source_attempt_index":q["source_attempt_index"],"source_raw_digest":q["source_raw_digest"],"source_parsed_digest":q["source_parsed_digest"],"prior_cumulative_digest":q["prior_cumulative_digest"],"entity_ids":["E1"],"text":"TRUE "+"final "*(n-1),"asserted_facts":[{"entity_id":"E1","dimension":"D","value":"TRUE"}],"cross_references":[],"running_summary":"final"},None

def test_cumulative_segments_then_bounded_final_revision(tmp_path):
 plan,mapping,_,cfg=augmentation_fixture(tmp_path); cfg.update({"spine_augmentation_max_segments_total":3,"spine_augmentation_final_revision":"progressive-final-v2","spine_augmentation_final_prompt_oversubscription":2.0}); add_preserved_augmentation_attempts(tmp_path,plan,mapping,cfg,word_counts=(2,2)); before=g.digest((tmp_path/"rejection_ledger.jsonl").read_text()); client=FinalSegmentClient()
 combined=f.augment_near_floor("CORPUS_F",{"title":"Natural Chapter"},plan,mapping,{"E1":"Natural Name"},"context",{"excluded_keyword_scan":[]},cfg,tmp_path,client)
 gates,detail=g.validate_spine_part(combined,"CORPUS_F",plan,mapping,[],(.6,1.75))
 assert all(gates.values()),detail
 row=g.read_jsonl(tmp_path/"spine_augmentation_ledger.jsonl")[0]
 assert len(row["segment_evidence"])==3 and row["segment_evidence"][-1]["source_revision"]=="progressive-final-v2"
 assert client.calls==1 and f.count(combined["text"])>=60
 prefix="\n".join((tmp_path/"rejection_ledger.jsonl").read_text().splitlines()[:4])+"\n"
 assert g.digest(prefix)==before

def test_mutual_repetition_blocks_cumulative_reuse():
 a={"text":"same paragraph "*25}; b={"text":"same paragraph "*25}; c={"text":"different material "*25}
 assert not f.additions_nonrepetitive([a,b])
 assert f.additions_nonrepetitive([a,c])
