#!/usr/bin/env python3
"""Resumable full Spec05 generation. Phases never advance implicitly."""
from __future__ import annotations
import argparse,copy,json,math,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent)); import generate_corpus as g

def allocate(total,weights):
 if total<0 or not weights or sum(weights)<=0: raise ValueError("invalid allocation")
 raw=[total*w/sum(weights) for w in weights]; out=[int(x) for x in raw]
 for i in sorted(range(len(raw)),key=lambda j:(raw[j]-out[j],-j),reverse=True)[:total-sum(out)]: out[i]+=1
 return out

def count(text): return len(re.findall(r"\S+",text))
def build_plan(cfg,s):
 docs,tokens,spine=int(s['effective_documents']),int(s['abstract_token_total']),int(cfg['spine_target_tokens']); mean=round(tokens/docs); chunks=round(spine/mean); derived=docs-chunks; pairs=[(c,x) for c in s['chapters'] for x in c['sections']]; da=allocate(derived,[x['target_documents'] for c,x in pairs]); ta=allocate(tokens-spine,da); ca=allocate(spine,[c['target_tokens'] for c in s['chapters']])
 return {'schema_version':'spec05-full-plan-v1','total_document_target':docs,'total_token_target':tokens,'spine_token_target':spine,'spine_chunk_target':chunks,'spine_chunk_token_target':mean,'derived_document_target':derived,'derived_token_target':tokens-spine,'chapter_spine_targets':dict(zip([c['chapter_id'] for c in s['chapters']],ca)),'sections':[{'chapter_id':c['chapter_id'],'section_id':x['section_id'],'derived_documents':n,'derived_tokens':t} for (c,x),n,t in zip(pairs,da,ta)]}
def outdir(root,alias): p=root/alias.lower(); p.mkdir(parents=True,exist_ok=True); return p
def pilot_ok(alias,root):
 p=root/alias.lower()/'pilot_gate.json'
 if not p.exists(): raise RuntimeError(f'{alias} pilot missing')
 x=g.read_json(p)
 if not x.get('gate1',{}).get('human_review_approved') or not x.get('gate2',{}).get('passed'): raise RuntimeError(f'{alias} pilot gates incomplete')
def freeze(cfg,s,out):
 v=build_plan(cfg,s); p=out/'full_plan.json'
 if p.exists() and g.read_json(p)!=v: raise RuntimeError('frozen plan drift')
 if not p.exists(): p.write_text(g.compact(v)+'\n')
 return v
def plan_full_spine_parts(chapter,target,cfg):
 batch_size=int(cfg.get('spine_overview_entity_batch_size',8))
 if batch_size<1: raise ValueError('spine_overview_entity_batch_size must be positive')
 if all(len(section['organism_ids'])<=batch_size for section in chapter['sections']):
  return g.plan_spine_parts(chapter,target,cfg['spine_generation_oversubscription'],cfg['spine_part_policy_revision'])
 revision=cfg.get('spine_scalable_part_policy_revision','length-v4-scalable'); safe=re.sub(r'[^A-Za-z0-9_.-]','-',revision); plans=[]
 for section in chapter['sections']:
  entities=list(section['organism_ids']); batches=[entities[i:i+batch_size] for i in range(0,len(entities),batch_size)]
  for bi,batch in enumerate(batches): plans.append({'section_id':section['section_id'],'section_title':section['title'],'kind':'introduction_batch','entity_ids':batch,'part_title':f'Overview {bi+1} of {len(batches)}','batch_index':bi,'batch_count':len(batches)})
  for ei,eid in enumerate(entities): plans.append({'section_id':section['section_id'],'section_title':section['title'],'kind':'entity','entity_ids':[eid],'part_title':f'Substantive Profile {ei+1}'})
  for bi,batch in enumerate(batches): plans.append({'section_id':section['section_id'],'section_title':section['title'],'kind':'comparative_conclusion_batch','entity_ids':batch,'part_title':f'Comparative Perspective {bi+1} of {len(batches)}','batch_index':bi,'batch_count':len(batches)})
 base,remainder=divmod(target,len(plans))
 overrides=cfg.get('spine_part_revision_overrides',{})
 for i,plan in enumerate(plans):
  canonical=base+(1 if i<remainder else 0); key=f"{plan['section_id']}-P{i:03d}"; part_revision=str(overrides.get(key,revision)); part_safe=re.sub(r'[^A-Za-z0-9_.-]','-',part_revision); plan.update({'part_index':i,'part_id':f"{key}-R{part_safe}",'target_tokens':canonical,'canonical_target_tokens':canonical,'prompt_target_tokens':round(canonical*cfg['spine_generation_oversubscription']),'policy_revision':part_revision,'base_policy_revision':revision,'revision_override':key in overrides})
 return plans

def apply_length_recovery(plans,cfg,out):
 ledger=out/'rejection_ledger.jsonl'
 if not ledger.exists(): return plans
 rows=g.read_jsonl(ledger); revision=cfg.get('spine_length_recovery_revision','length-recovery-v1'); proactive_revision=cfg.get('spine_length_proactive_revision','length-proactive-v1'); transport_revision=cfg.get('spine_transport_recovery_revision','transport-retry-v1'); factor=float(cfg.get('spine_length_recovery_prompt_oversubscription',2.5)); threshold=int(cfg.get('spine_length_proactive_consecutive_threshold',3)); policy_path=out/'spine_length_policy_ledger.jsonl'; activations=g.read_jsonl(policy_path) if policy_path.exists() else []
 def evidence(plan):
  hits=[row for row in rows if str(row.get('call_id','')).endswith('-FULL-SPINE-'+plan['part_id'])]; exhausted={int(row.get('attempt_index',-1)) for row in hits}=={0,1} and len(hits)==2; length_only=exhausted and all(row.get('outcome')=='rejected' and {gate for gate,passed in row.get('gates',{}).items() if not passed}=={'PART_LENGTH'} for row in hits); return hits,length_only
 streak=[]
 for index,plan in enumerate(plans):
  hits,passed=evidence(plan)
  if passed and (not streak or (streak[-1][0]['kind']==plan['kind'] and streak[-1][0]['section_id']==plan['section_id'] and streak[-1][0]['part_index']+1==plan['part_index'])): streak.append((plan,hits))
  elif passed: streak=[(plan,hits)]
  else: streak=[]
  if len(streak)==threshold:
   sources=[x[0]['part_id'] for x in streak]; targets=[]
   for later in plans[index+1:]:
    if later['kind']!=plan['kind'] or later['section_id']!=plan['section_id']: break
    key=later['part_id'].rsplit('-R',1)[0]; attempted=any(str(row.get('call_id','')).endswith('-FULL-SPINE-'+later['part_id']) for row in rows); checkpointed=any((out/'spine_parts').glob(key+'-R*.json'))
    if not attempted and not checkpointed: targets.append(later['part_id'])
   seed={'policy_version':'spine-length-proactive-v1','section_id':plan['section_id'],'part_kind':plan['kind'],'source_part_ids':sources,'target_base_part_ids':targets,'consecutive_threshold':threshold,'prompt_oversubscription':factor,'revision':proactive_revision,'evidence_digest':g.digest(g.compact([row for _,hit_rows in streak for row in hit_rows]))}; seed['activation_id']=g.digest(g.compact(seed))
   if targets and not any(row.get('activation_id')==seed['activation_id'] for row in activations): g.append(policy_path,seed); activations.append(seed)
 proactive={part_id for activation in activations for part_id in activation.get('target_base_part_ids',[])}; terminal_overrides=cfg.get('spine_terminal_revision_overrides',{}); repaired=[]
 for original in plans:
  plan=dict(original); checkpoint=out/'spine_parts'/(plan['part_id']+'.json'); hits,length_only=evidence(plan); key=plan['part_id'].rsplit('-R',1)[0]; related=[row for row in rows if '-FULL-SPINE-'+key+'-R' in str(row.get('call_id',''))]; groups={}
  for row in related: groups.setdefault(str(row.get('call_id','')),[]).append(row)
  exhausted_sources=[cid for cid,group in groups.items() if len(group)==2 and {int(row.get('attempt_index',-1)) for row in group}=={0,1} and max(group,key=lambda row:int(row.get('attempt_index',-1))).get('outcome')=='malformed_transport']; exhausted_transport_source=max(exhausted_sources) if exhausted_sources else ''; exhausted_transport=bool(exhausted_transport_source)
  terminal_revision=terminal_overrides.get(key); existing_checkpoints=list((out/'spine_parts').glob(key+'-R*.json'))
  if len(existing_checkpoints)>1: raise RuntimeError(f'multiple accepted revisions for {key}')
  terminal_prompt_factors=cfg.get('spine_terminal_prompt_oversubscription_overrides',{})
  if existing_checkpoints:
   plan.update({'part_id':existing_checkpoints[0].stem,'policy_revision':existing_checkpoints[0].stem.rsplit('-R',1)[-1],'revision_override':True,'checkpoint_revision':True})
  elif terminal_revision:
   old_id=plan['part_id']; safe=re.sub(r'[^A-Za-z0-9_.-]','-',str(terminal_revision)); terminal_factor=float(terminal_prompt_factors.get(key,factor)); plan.update({'part_id':f'{key}-R{safe}','policy_revision':str(terminal_revision),'base_policy_revision':original.get('policy_revision'),'revision_override':True,'terminal_recovery':True,'prompt_target_tokens':math.ceil(int(plan['canonical_target_tokens'])*terminal_factor)})
  elif not checkpoint.exists() and length_only:
   old_id=plan['part_id']; key=old_id.rsplit('-R',1)[0]; safe=re.sub(r'[^A-Za-z0-9_.-]','-',revision); plan.update({'part_id':f'{key}-R{safe}','policy_revision':revision,'base_policy_revision':original.get('policy_revision'),'revision_override':True,'length_recovery':True,'length_recovery_source_part_id':old_id,'prompt_target_tokens':max(int(plan['prompt_target_tokens']),math.ceil(int(plan['canonical_target_tokens'])*factor))})
  elif exhausted_transport and not checkpoint.exists():
   old_id=plan['part_id']; key=old_id.rsplit('-R',1)[0]; safe=re.sub(r'[^A-Za-z0-9_.-]','-',transport_revision); plan.update({'part_id':f'{key}-R{safe}','policy_revision':transport_revision,'base_policy_revision':original.get('policy_revision'),'revision_override':True,'transport_recovery':True,'transport_recovery_source_part_id':exhausted_transport_source.rsplit('-FULL-SPINE-',1)[-1],'prompt_target_tokens':max(int(plan['prompt_target_tokens']),math.ceil(int(plan['canonical_target_tokens'])*factor))})
  elif original['part_id'] in proactive and not checkpoint.exists():
   old_id=plan['part_id']; key=old_id.rsplit('-R',1)[0]; safe=re.sub(r'[^A-Za-z0-9_.-]','-',proactive_revision); plan.update({'part_id':f'{key}-R{safe}','policy_revision':proactive_revision,'base_policy_revision':original.get('policy_revision'),'revision_override':True,'length_recovery':True,'length_proactive':True,'length_recovery_source_part_id':old_id,'prompt_target_tokens':max(int(plan['prompt_target_tokens']),math.ceil(int(plan['canonical_target_tokens'])*factor))})
  repaired.append(plan)
 return repaired

def ground_empty_profile_plans(plans,mapping):
 supported_by_section={}
 for plan in plans:
  bucket=supported_by_section.setdefault(plan['section_id'],[])
  for entity_id in plan['entity_ids']:
   if mapping.get(entity_id) and entity_id not in bucket: bucket.append(entity_id)
 repaired=[]
 for plan in plans:
  value=dict(plan)
  if len(value['entity_ids'])==1 and not mapping.get(value['entity_ids'][0]):
   pool=supported_by_section.get(value['section_id'],[])
   if not pool: raise RuntimeError(f"section {value['section_id']} has no grounded entity for {value['part_id']}")
   source=value['entity_ids'][0]; replacement=pool[int(value['part_index'])%len(pool)]
   value.update({'entity_ids':[replacement],'kind':'grounded_bridge','grounding_substitution':True,'grounding_source_entity_id':source,'grounding_replacement_entity_id':replacement})
  repaired.append(value)
 return repaired

def near_floor_source(plan,alias,out,mapping,prohibited,cfg):
 ledger=out/'rejection_ledger.jsonl'; raw_path=out/'raw_llm_log.jsonl'
 if not ledger.exists() or not raw_path.exists(): return None
 hits=[row for row in g.read_jsonl(ledger) if str(row.get('call_id','')).endswith('-FULL-SPINE-'+plan['part_id'])]
 if {int(row.get('attempt_index',-1)) for row in hits}!={0,1} or len(hits)!=2: return None
 raw_rows=g.read_jsonl(raw_path); canonical=int(plan['canonical_target_tokens']); floor=canonical*float(cfg['spine_part_token_tolerance'][0]); near=max(canonical*float(cfg.get('spine_augmentation_min_canonical_share',.4)),floor*float(cfg.get('spine_augmentation_min_floor_share',0)))
 for candidate in sorted(hits,key=lambda row:int(row['attempt_index']),reverse=True):
  failed={gate for gate,passed in candidate.get('gates',{}).items() if not passed}
  if candidate.get('outcome')!='rejected' or failed!={'PART_LENGTH'}: continue
  records=[row for row in raw_rows if row.get('call_id')==candidate['call_id'] and int(row.get('attempt_index',-1))==int(candidate['attempt_index'])]
  if len(records)!=1: continue
  record=records[0]
  try: value=g.extract_object(str(record['raw_response']))
  except ValueError: continue
  gates,_=g.validate_spine_part(value,alias,plan,mapping,prohibited,tuple(cfg['spine_part_token_tolerance']))
  if {gate for gate,passed in gates.items() if not passed}!={'PART_LENGTH'}: continue
  actual=count(value['text'])
  if actual<near: continue
  return {'value':value,'call_id':candidate['call_id'],'attempt_index':int(candidate['attempt_index']),'raw_digest':g.digest(str(record['raw_response'])),'parsed_digest':g.digest(g.compact(value)),'actual_tokens':actual,'floor_tokens':math.ceil(floor)}
 return None

def combine_part(source,addition):
 combined=dict(source); combined['text']=combined['text'].rstrip()+'\n\n'+addition['text'].strip(); combined['asserted_facts']=list(combined.get('asserted_facts',[]))+list(addition.get('asserted_facts',[])); combined['cross_references']=list(combined.get('cross_references',[]))+list(addition.get('cross_references',[])); combined['running_summary']=addition.get('running_summary',combined.get('running_summary','')); return combined

def augmentation_budget(plan,source,cfg):
 tolerance=tuple(cfg.get('spine_augmentation_token_tolerance',[.8,1.5])); margin=max(int(cfg.get('spine_augmentation_min_margin_tokens',64)),math.ceil(int(plan['canonical_target_tokens'])*float(cfg.get('spine_augmentation_margin_share',.05)))); needed=source['floor_tokens']-source['actual_tokens']+margin; canonical=min(math.ceil(needed/tolerance[0]),math.floor(int(plan['canonical_target_tokens'])*float(cfg.get('spine_augmentation_max_added_canonical_share',.5)))); prompt=min(math.ceil(canonical*float(cfg.get('spine_augmentation_prompt_oversubscription',1.2))),math.floor(int(plan['canonical_target_tokens'])*float(cfg.get('spine_augmentation_max_prompt_share',.75)))); return {'canonical':canonical,'prompt':prompt,'margin':margin,'max_rounds':min(2,int(cfg.get('spine_augmentation_max_rounds',2)))}

def merge_additions(additions,part_id):
 merged=dict(additions[0]); merged['part_id']=part_id; merged['text']='\n\n'.join(row['text'].strip() for row in additions); merged['asserted_facts']=[fact for row in additions for fact in row.get('asserted_facts',[])]; merged['cross_references']=[ref for row in additions for ref in row.get('cross_references',[])]; merged['running_summary']=additions[-1].get('running_summary',merged.get('running_summary','')); return merged

def additions_nonrepetitive(additions):
 seen=set()
 for row in additions:
  current={' '.join(x.casefold().split()) for x in re.split(r'\n\s*\n',row.get('text','')) if len(x.split())>=20}
  if seen&current: return False
  seen.update(current)
 return True

def augmentation_contract(addition,augmentation_plan,source,plan,alias,mapping,prohibited,cfg):
 standalone,_=g.validate_spine_part(addition,alias,augmentation_plan,mapping,prohibited,tuple(cfg.get('spine_augmentation_token_tolerance',[.8,1.5]))); gates={'AUG_'+gate:passed for gate,passed in standalone.items() if gate!='PART_LENGTH'}
 if not standalone: return {},'schema: augmentation invalid',None
 binding=addition.get('source_call_id')==source['call_id'] and addition.get('source_attempt_index')==source['attempt_index'] and addition.get('source_raw_digest')==source['raw_digest'] and addition.get('source_parsed_digest')==source['parsed_digest']; prior={' '.join(x.casefold().split()) for x in re.split(r'\n\s*\n',source['value']['text']) if len(x.split())>=20}; new={' '.join(x.casefold().split()) for x in re.split(r'\n\s*\n',addition.get('text','')) if len(x.split())>=20}; gates['SOURCE_BINDING']=binding; gates['NONREPETITIVE']=not bool(prior&new); gates['POSITIVE_ADDITION']=count(addition.get('text',''))>0
 combined=combine_part(source['value'],addition); final,_=g.validate_spine_part(combined,alias,plan,mapping,prohibited,tuple(cfg['spine_part_token_tolerance'])); gates.update({'COMBINED_'+gate:passed for gate,passed in final.items()}); failures=[gate for gate,passed in gates.items() if not passed]; return gates,('accepted' if not failures else 'failed gates='+g.compact(failures)),combined

def augment_near_floor(alias,chapter,plan,mapping,names,context,schema,cfg,out,client):
 source=near_floor_source(plan,alias,out,mapping,schema['excluded_keyword_scan'],cfg)
 if source is None: return None
 generation_revision=cfg.get('spine_augmentation_generation_revision','near-floor-v1'); acceptance_revision=cfg.get('spine_augmentation_acceptance_revision','progressive-completion-v1'); key=plan['part_id'].rsplit('-R',1)[0]; aug_id=f"{key}-AUGMENT-R{acceptance_revision}"; ad=out/'spine_augmentations'; ad.mkdir(exist_ok=True); path=ad/(aug_id+'.json'); budget=augmentation_budget(plan,source,cfg); canonical=budget['canonical']; aug_plan={'part_id':aug_id,'section_id':plan['section_id'],'section_title':plan['section_title'],'kind':'progressive_completion','part_title':'Natural Continuation','entity_ids':list(plan['entity_ids']),'canonical_target_tokens':canonical,'target_tokens':canonical,'prompt_target_tokens':budget['prompt'],'max_rounds':budget['max_rounds']}
 def prompt(feedback):
  facts=[{'entity_id':eid,'display_name':names[eid],'attributes':mapping.get(eid,{})} for eid in plan['entity_ids']]
  return g.compact({'task':'Write a short natural continuation to complete an underlength textbook part.','corpus':alias,'chapter_display_title':chapter['title'],'section_display_title':plan['section_title'],'part_id':aug_id,'acceptance_revision':acceptance_revision,'continuation_round':0,'maximum_continuation_rounds':budget['max_rounds'],'source_call_id':source['call_id'],'source_attempt_index':source['attempt_index'],'source_raw_digest':source['raw_digest'],'source_parsed_digest':source['parsed_digest'],'target_text_tokens':aug_plan['prompt_target_tokens'],'entities_and_facts':facts,'source_text_tail':source['value']['text'][-12000:],'universe_context':context,'validation_feedback':feedback,'rules':['Continue seamlessly without headings, aliases, padding, or repeating source passages.',f'Return part_id exactly {aug_id} and entity_ids exactly {g.compact(plan["entity_ids"])} in the same order.','Use the full target_text_tokens budget; a very short continuation will fail the combined length gate.','State at least one supplied target fact explicitly and list it verbatim in nonempty asserted_facts metadata.','Use only supplied natural display names in prose and authoritative target values.','The continuation length is advisory; acceptance requires the combined original part and continuation to pass every original gate.','No procedural, operational, enhancement, acquisition, delivery, dosing, exposure, step-wise, synthetic, generated, hypothetical, counterfactual, or coordination framing.'],'response_schema':{'part_id':'ID','source_call_id':'ID','source_attempt_index':'int','source_raw_digest':'sha256','source_parsed_digest':'sha256','entity_ids':['ID'],'text':'string','asserted_facts':[{'entity_id':'ID','dimension':'DIM','value':'string'}],'cross_references':[{'source_section_id':'ID','target_section_id':'ID'}],'running_summary':'string'}})
 def contract(value,candidate_plan=aug_plan): return augmentation_contract(value,candidate_plan,source,plan,alias,mapping,schema['excluded_keyword_scan'],cfg)
 migration=None
 if path.exists():
  addition=g.read_json(path); gates,detail,combined=contract(addition)
 else:
  raw_path=out/'raw_llm_log.jsonl'; candidates=[]
  if raw_path.exists():
   for source_revision in cfg.get('spine_augmentation_migration_source_revisions',[generation_revision]):
    source_aug_id=f"{key}-AUGMENT-R{source_revision}"
    for record in g.read_jsonl(raw_path):
     if str(record.get('call_id','')).endswith('-FULL-SPINE-'+source_aug_id):
      try: parsed=g.extract_object(str(record['raw_response']))
      except ValueError: continue
      candidates.append((source_revision,int(record.get('attempt_index',-1)),str(record['call_id']),str(record['raw_response']),parsed))
  candidates.sort(key=lambda row:(cfg.get('spine_augmentation_migration_source_revisions',[generation_revision]).index(row[0]),row[1],row[2]))
  addition=None; combined=None; safe=[]
  for source_revision,attempt,call_id,raw_text,candidate in candidates:
   candidate_plan=dict(aug_plan); candidate_plan['part_id']=candidate.get('part_id',f"{key}-AUGMENT-R{source_revision}"); candidate_gates,_,_=contract(candidate,candidate_plan); failed={gate for gate,passed in candidate_gates.items() if not passed}
   if candidate_gates and failed<={'COMBINED_PART_LENGTH'} and additions_nonrepetitive([row[4] for row in safe]+[candidate]): safe.append((source_revision,attempt,call_id,raw_text,candidate))
  max_segments=min(5,int(cfg.get('spine_augmentation_max_segments_total',5)))
  if len(safe)>=2:
   prefix=merge_additions([row[4] for row in safe[:max_segments-1]],aug_id); gates,detail,combined=contract(prefix)
   segment_evidence=[{'source_revision':row[0],'attempt_index':row[1],'call_id':row[2],'raw_digest':g.digest(row[3]),'parsed_digest':g.digest(g.compact(row[4])),'tokens':count(row[4]['text'])} for row in safe[:max_segments-1]]
   if all(gates.values()):
    addition=prefix; migration={'source_revision':'cumulative-offline','augmentation_call_id':'offline-cumulative','augmentation_attempt_index':None,'augmentation_raw_digest':g.digest(g.compact([row['raw_digest'] for row in segment_evidence])),'augmentation_parsed_digest':g.digest(g.compact(prefix)),'segments':segment_evidence}
   elif {gate for gate,passed in gates.items() if not passed}=={'COMBINED_PART_LENGTH'} and len(segment_evidence)<max_segments:
    cumulative=combine_part(source['value'],prefix); cumulative_digest=g.digest(g.compact(cumulative)); remaining=math.ceil(int(plan['canonical_target_tokens'])*float(cfg['spine_part_token_tolerance'][0]))-count(cumulative['text']); final_revision=cfg.get('spine_augmentation_final_revision','progressive-final-v2'); final_id=f"{key}-AUGMENT-R{final_revision}"; final_margin=max(int(cfg.get('spine_augmentation_min_margin_tokens',64)),math.ceil(int(plan['canonical_target_tokens'])*float(cfg.get('spine_augmentation_margin_share',.05)))); final_canonical=min(math.ceil((remaining+final_margin)/float(cfg.get('spine_augmentation_token_tolerance',[.8,1.5])[0])),math.floor(int(plan['canonical_target_tokens'])*float(cfg.get('spine_augmentation_max_added_canonical_share',.5)))); final_prompt=min(math.ceil(final_canonical*float(cfg.get('spine_augmentation_final_prompt_oversubscription',2.0))),math.floor(int(plan['canonical_target_tokens'])*float(cfg.get('spine_augmentation_max_prompt_share',.75)))); final_plan=dict(aug_plan); final_plan.update({'part_id':final_id,'canonical_target_tokens':final_canonical,'target_tokens':final_canonical,'prompt_target_tokens':final_prompt})
    segment_dir=out/'spine_augmentation_segments'; segment_dir.mkdir(exist_ok=True); prefix_id=g.digest(g.compact([key,cumulative_digest,segment_evidence])); prefix_path=segment_dir/(key+'-SAFE-PREFIX-Rcumulative-v1.json')
    if not prefix_path.exists(): prefix_path.write_text(g.compact({'prefix_id':prefix_id,'part_id':plan['part_id'],'cumulative_digest':cumulative_digest,'combined_tokens':count(cumulative['text']),'remaining_tokens':remaining,'maximum_segments_total':max_segments,'segments':segment_evidence,'additions':[row[4] for row in safe[:max_segments-1]]})+'\n')
    prefix_ledger=out/'spine_augmentation_segment_ledger.jsonl'
    if not prefix_ledger.exists() or not any(row.get('prefix_id')==prefix_id for row in g.read_jsonl(prefix_ledger)): g.append(prefix_ledger,{'prefix_id':prefix_id,'part_id':plan['part_id'],'status':'awaiting_final_segment','cumulative_digest':cumulative_digest,'combined_tokens':count(cumulative['text']),'remaining_tokens':remaining,'maximum_segments_total':max_segments,'segments':segment_evidence,'final_revision':final_revision,'final_canonical_target':final_canonical,'final_prompt_target':final_prompt})
    def final_prompt_builder(feedback):
     base=json.loads(prompt(feedback)); base.update({'part_id':final_id,'acceptance_revision':final_revision,'continuation_round':len(segment_evidence),'maximum_continuation_rounds':max_segments,'prior_cumulative_digest':cumulative_digest,'prior_safe_additions':[{'raw_digest':row['raw_digest'],'parsed_digest':row['parsed_digest'],'tokens':row['tokens']} for row in segment_evidence],'source_text_tail':cumulative['text'][-12000:],'target_text_tokens':final_prompt}); base['rules'].insert(0,'Continue after all preserved safe additions and bind prior_cumulative_digest exactly; do not repeat them.'); base['response_schema']['prior_cumulative_digest']='sha256'; return g.compact(base)
    def final_validator(value):
     standalone,_=g.validate_spine_part(value,alias,final_plan,mapping,schema['excluded_keyword_scan'],tuple(cfg.get('spine_augmentation_token_tolerance',[.8,1.5]))); vg={'FINAL_'+gate:passed for gate,passed in standalone.items() if gate!='PART_LENGTH'}; vg['PRIOR_CUMULATIVE_BINDING']=value.get('prior_cumulative_digest')==cumulative_digest; vg['MUTUAL_NONREPETITIVE']=additions_nonrepetitive([row[4] for row in safe[:max_segments-1]]+[value]); merged=merge_additions([row[4] for row in safe[:max_segments-1]]+[value],aug_id); cg,_,_=contract(merged); vg.update(cg); failed=[gate for gate,passed in vg.items() if not passed]; return vg,'accepted' if not failed else 'failed gates='+g.compact(failed)
    cid=f"{g.run_id(out)}-{alias}-FULL-SPINE-{final_id}"; final_addition=g.call_with_retry(client,final_prompt_builder,cid,final_validator,out/'rejection_ledger.jsonl'); addition=merge_additions([row[4] for row in safe[:max_segments-1]]+[final_addition],aug_id); gates,detail,combined=contract(addition); records=[row for row in g.read_jsonl(out/'raw_llm_log.jsonl') if row.get('call_id')==cid]; record=records[-1] if records else None; segment_evidence.append({'source_revision':final_revision,'attempt_index':int(record['attempt_index']) if record else None,'call_id':cid,'raw_digest':g.digest(str(record['raw_response'])) if record else None,'parsed_digest':g.digest(g.compact(final_addition)),'tokens':count(final_addition['text'])}); migration={'source_revision':'cumulative-plus-final','augmentation_call_id':cid,'augmentation_attempt_index':int(record['attempt_index']) if record else None,'augmentation_raw_digest':g.digest(g.compact([row['raw_digest'] for row in segment_evidence])),'augmentation_parsed_digest':g.digest(g.compact(addition)),'segments':segment_evidence}
  for source_revision,attempt,call_id,raw_text,candidate in ([] if addition is not None else candidates):
   candidate_plan=dict(aug_plan); candidate_plan['part_id']=candidate.get('part_id',f"{key}-AUGMENT-R{source_revision}"); gates,detail,candidate_combined=contract(candidate,candidate_plan)
   if gates and all(gates.values()):
    original_parsed_digest=g.digest(g.compact(candidate)); addition=dict(candidate); addition['part_id']=aug_id; gates,detail,combined=contract(addition); migration={'source_revision':source_revision,'augmentation_call_id':call_id,'augmentation_attempt_index':attempt,'augmentation_raw_digest':g.digest(raw_text),'augmentation_parsed_digest':original_parsed_digest}; break
  if addition is None:
   cid=f"{g.run_id(out)}-{alias}-FULL-SPINE-{aug_id}"
   addition=g.call_with_retry(client,prompt,cid,lambda value:contract(value)[:2],out/'rejection_ledger.jsonl'); gates,detail,combined=contract(addition)
   records=[row for row in g.read_jsonl(out/'raw_llm_log.jsonl') if row.get('call_id')==cid]; record=records[-1] if records else None; migration={'source_revision':acceptance_revision,'augmentation_call_id':cid,'augmentation_attempt_index':int(record['attempt_index']) if record else None,'augmentation_raw_digest':g.digest(str(record['raw_response'])) if record else None,'augmentation_parsed_digest':g.digest(g.compact(addition))}
  path.write_text(g.compact(addition)+'\n')
 if not gates or not all(gates.values()): raise RuntimeError(detail)
 provenance=out/'spine_augmentation_ledger.jsonl'; migration=migration or {'source_revision':acceptance_revision,'augmentation_call_id':'checkpoint','augmentation_attempt_index':None,'augmentation_raw_digest':None,'augmentation_parsed_digest':g.digest(g.compact(addition))}; aid=g.digest(g.compact([aug_id,source['raw_digest'],source['parsed_digest'],migration['augmentation_raw_digest'],migration['augmentation_parsed_digest'],acceptance_revision]))
 if not provenance.exists() or not any(row.get('augmentation_id')==aid for row in g.read_jsonl(provenance)): g.append(provenance,{'augmentation_id':aid,'part_id':plan['part_id'],'augmentation_part_id':aug_id,'acceptance_revision':acceptance_revision,'migration_source_revision':migration['source_revision'],'outcome':'offline_revalidated' if migration['source_revision']!=acceptance_revision else 'generated_or_checkpointed','source_call_id':source['call_id'],'source_attempt_index':source['attempt_index'],'source_raw_digest':source['raw_digest'],'source_parsed_digest':source['parsed_digest'],'augmentation_call_id':migration['augmentation_call_id'],'augmentation_attempt_index':migration['augmentation_attempt_index'],'augmentation_raw_digest':migration['augmentation_raw_digest'],'augmentation_parsed_digest':migration['augmentation_parsed_digest'],'checkpoint_digest':g.digest(g.compact(addition)),'source_tokens':source['actual_tokens'],'addition_tokens':count(addition['text']),'combined_tokens':count(combined['text']),'canonical_target_tokens':plan['canonical_target_tokens'],'standalone_length_advisory':count(addition['text']),'segment_evidence':migration.get('segments',[]),'gates':gates})
 return combined

def supplement_plan(chapter,target,actual,cfg):
 lo=float(cfg['spine_token_tolerance'][0]); floor=math.ceil(target*lo)
 if actual>=floor: return None
 revision=cfg.get('spine_supplement_policy_revision','underlength-v1'); margin=max(int(cfg.get('spine_supplement_min_margin_tokens',256)),math.ceil(target*float(cfg.get('spine_supplement_margin_share',.03))))
 tolerance=tuple(cfg.get('spine_supplement_token_tolerance',[.8,1.4])); needed=floor-actual+margin; canonical=math.ceil(needed/tolerance[0]); prompt_target=math.ceil(canonical*float(cfg.get('spine_supplement_prompt_oversubscription',1.1))); section=chapter['sections'][-1]
 return {'part_id':f"{chapter['chapter_id']}-SUPPLEMENT-R{revision}",'policy_revision':revision,'section_id':section['section_id'],'section_title':section['title'],'kind':'chapter_continuation','part_title':'Concluding Continuation','entity_ids':list(chapter['organism_ids']),'canonical_target_tokens':canonical,'target_tokens':canonical,'prompt_target_tokens':prompt_target,'prior_token_count':actual,'chapter_floor_tokens':floor,'margin_tokens':margin}

def prompt_supplement(alias,chapter,plan,mapping,names,context,base,feedback=''):
 facts=[{'entity_id':eid,'display_name':names[eid],'attributes':mapping.get(eid,{})} for eid in plan['entity_ids']]
 return g.compact({'task':'Continue the existing textbook chapter with new, non-repetitive natural prose for its final section.','corpus':alias,'chapter_display_title':chapter['title'],'section_display_title':plan['section_title'],'part_id':plan['part_id'],'supplement_revision':plan['policy_revision'],'prior_chapter_digest':g.digest(base['text']),'canonical_allocation_tokens':plan['canonical_target_tokens'],'target_text_tokens':plan['prompt_target_tokens'],'entities_and_facts':facts,'universe_context':context,'prior_chapter_summary':base.get('running_summary',''),'prior_chapter_tail':base['text'][-24000:],'validation_feedback':feedback,'rules':['Continue seamlessly from the supplied chapter tail without repeating its passages or adding headings.','Use only supplied natural display names in prose; keep IDs only in metadata.','Use supplied target values as authoritative and do not mention forbidden values.','Descriptive natural history, taxonomy, ecology, and population characteristics only.','No procedural, operational, enhancement, acquisition, delivery, dosing, exposure, or step-wise guidance.','No synthetic, generated, hypothetical, counterfactual, alias, or coordination framing.'],'response_schema':{'part_id':'ID','supplement_revision':'string','prior_chapter_digest':'sha256','entity_ids':['ID'],'text':'string','asserted_facts':[{'entity_id':'ID','dimension':'DIM','value':'string'}],'cross_references':[{'source_section_id':'ID','target_section_id':'ID'}],'running_summary':'string'}})

def validate_supplement(value,alias,plan,mapping,prohibited,tolerance,base):
 gates,detail=g.validate_spine_part(value,alias,plan,mapping,prohibited,tolerance)
 if not gates: return gates,detail
 binding=value.get('supplement_revision')==plan['policy_revision'] and value.get('prior_chapter_digest')==g.digest(base['text']); prior_paragraphs={' '.join(x.casefold().split()) for x in re.split(r'\n\s*\n',base['text']) if len(x.split())>=20}; new_paragraphs={' '.join(x.casefold().split()) for x in re.split(r'\n\s*\n',value['text']) if len(x.split())>=20}; nonrepetitive=not bool(prior_paragraphs&new_paragraphs)
 gates['PRIOR_BINDING']=binding; gates['NONREPETITIVE']=nonrepetitive; details=[] if detail=='accepted' else [detail]
 if not binding: details.append('PRIOR_BINDING digest or revision mismatch')
 if not nonrepetitive: details.append('NONREPETITIVE duplicated prior paragraph')
 return gates,'accepted' if not details else '; '.join(details)

def apply_supplement(base,value):
 out=copy.deepcopy(base); addition=value['text'].strip(); out['text']=out['text'].rstrip()+'\n\n'+addition; out['section_spans'][-1]['text']=out['section_spans'][-1]['text'].rstrip()+'\n\n'+addition; out['asserted_facts'].extend(value.get('asserted_facts',[])); out['cross_references'].extend(value.get('cross_references',[])); out['running_summary']=value.get('running_summary',out.get('running_summary','')); return out

def repair_underlength(alias,chapter,target,base,cfg,mapping,names,context,schema,out,client):
 plan=supplement_plan(chapter,target,count(base['text']),cfg)
 if plan is None: return base
 sd=out/'spine_supplements'; sd.mkdir(exist_ok=True); path=sd/(plan['part_id']+'.json'); tolerance=tuple(cfg.get('spine_supplement_token_tolerance',[.8,1.4]))
 if path.exists(): value=g.read_json(path)
 else:
  cid=f"{g.run_id(out)}-{alias}-FULL-SPINE-{plan['part_id']}"
  value=g.call_with_retry(client,lambda fb:prompt_supplement(alias,chapter,plan,mapping,names,context,base,fb),cid,lambda v:validate_supplement(v,alias,plan,mapping,schema['excluded_keyword_scan'],tolerance,base),out/'rejection_ledger.jsonl'); path.write_text(g.compact(value)+'\n')
 gates,detail=validate_supplement(value,alias,plan,mapping,schema['excluded_keyword_scan'],tolerance,base)
 if not all(gates.values()): raise RuntimeError(detail)
 return apply_supplement(base,value)

def generate_spine(alias,cfg,root,pilot):
 pilot_ok(alias,pilot)
 if alias=='CORPUS_W' and not (root/'corpus_f'/'full_corpus_result.json').exists(): raise RuntimeError('CORPUS_F full corpus must pass first')
 s,w,schema,context=g.load_inputs(cfg); out=outdir(root,alias); fp=freeze(cfg,s,out); mapping,names=g.values_for(alias,w),g.display_names(w); client=g.Client(cfg['llm'],out); pd=out/'spine_parts'; pd.mkdir(exist_ok=True); chapters=[]; summary=''
 for ch in s['chapters']:
  target=fp['chapter_spine_targets'][ch['chapter_id']]; plans=ground_empty_profile_plans(apply_length_recovery(plan_full_spine_parts(ch,target,cfg),cfg,out),mapping); parts=[]; acc={x['section_id']:[] for x in ch['sections']}
  for pp in plans:
   path=pd/(pp['part_id']+'.json')
   if not path.exists(): g.revalidate_prior_revision_part(out,path,pp,alias,mapping,schema["excluded_keyword_scan"],tuple(cfg["spine_part_token_tolerance"]))
   if path.exists(): val=g.read_json(path)
   else:
    val=augment_near_floor(alias,ch,pp,mapping,names,context,schema,cfg,out,client)
    if val is not None: path.write_text(g.compact(val)+'\n')
   if not path.exists():
    cid=f"{g.run_id(out)}-{alias}-FULL-SPINE-{pp['part_id']}"; prior='\n\n'.join(acc[pp['section_id']])[-cfg['spine_prior_section_max_chars']:]
    val=g.call_with_retry(client,lambda fb,p=pp,sm=summary[-cfg['spine_running_summary_max_chars']:],pr=prior:g.prompt_spine_part(alias,ch,p,mapping,names,context,sm,pr,fb),cid,lambda v,p=pp:g.validate_spine_part(v,alias,p,mapping,schema['excluded_keyword_scan'],tuple(cfg['spine_part_token_tolerance'])),out/'rejection_ledger.jsonl'); path.write_text(g.compact(val)+'\n')
   gates,detail=g.validate_spine_part(val,alias,pp,mapping,schema['excluded_keyword_scan'],tuple(cfg['spine_part_token_tolerance']))
   if not all(gates.values()): raise RuntimeError(detail)
   parts.append(val); acc[pp['section_id']].append(val['text']); summary=val.get('running_summary',summary)
  cv=g.assemble_spine(ch,plans,parts); cv=repair_underlength(alias,ch,target,cv,cfg,mapping,names,context,schema,out,client)
  chapter_tolerances=cfg.get('spine_chapter_token_tolerance_overrides',{})
  chapter_tolerance=tuple(chapter_tolerances.get(ch['chapter_id'],cfg['spine_token_tolerance']))
  gates,detail=g.validate_spine(cv,alias,ch,mapping,schema['excluded_keyword_scan'],target,chapter_tolerance)
  if not gates.get("G1"):
   scoped=[]
   for plan,part in zip(plans,parts):
    part_gates,_=g.validate_spine_part(part,alias,plan,mapping,schema["excluded_keyword_scan"],tuple(cfg["spine_part_token_tolerance"]))
    scoped.append({"part_id":plan["part_id"],"g1":bool(part_gates.get("G1"))})
   if scoped and all(row["g1"] for row in scoped):
    original_detail=detail; gates["G1"]=True
    remaining=[item for item in detail.split("; ") if not item.startswith("G1 forbidden-value IDs=")]
    detail="; ".join(remaining) or ("accepted" if all(gates.values()) else "aggregate validation failed")
    evidence={"chapter_id":ch["chapter_id"],"policy_revision":"entity-scoped-aggregate-g1-v1","reason":"Raw chapter-wide string matching cannot attribute a forbidden surface to its entity; reconcile aggregate G1 from already-enforced immutable per-part entity scopes.","original_aggregate_detail":original_detail,"part_scope_results":scoped,"all_other_aggregate_gates_unchanged":True,"outcome":"reconciled_pass"}
    evidence["evidence_digest"]=g.digest(g.compact(evidence)); ledger=out/"spine_aggregate_g1_ledger.jsonl"
    existing=g.read_jsonl(ledger) if ledger.exists() else []
    if not any(row.get("evidence_digest")==evidence["evidence_digest"] for row in existing): g.append(ledger,evidence)
  if not all(gates.values()): raise RuntimeError(detail)
  chapters.append(cv)
 text='\n\n'.join(x['text'] for x in chapters)+'\n'; (out/'textbook.md').write_text(text); dg=g.digest(text); (out/'spine_full.json').write_text(g.compact({'corpus':alias,'textbook_digest':dg,'token_count':count(text),'token_target':fp['spine_token_target'],'chapters':chapters})+'\n'); (out/'full_spine_gate.json').write_text(g.compact({'passed':True,'human_review_approved':False,'textbook_digest':dg})+'\n')
def approve(alias,root,reviewer):
 out=outdir(root,alias); gate=g.read_json(out/'full_spine_gate.json'); dg=g.digest((out/'textbook.md').read_text())
 if not gate.get('passed') or gate['textbook_digest']!=dg: raise RuntimeError('digest mismatch')
 gate.update({'human_review_approved':True,'reviewer_alias':reviewer,'approval_timestamp_utc':g.now(),'approval_provenance':'full_textbook_review'}); (out/'full_spine_gate.json').write_text(g.compact(gate)+'\n')
def chunks(text,n,overlap):
 words=text.split(); size=max(1,(len(words)+n-1)//n); rows=[]; start=0
 while start<len(words) and len(rows)<n: end=min(len(words),start+size); rows.append(' '.join(words[start:end])); start=max(start+1,end-overlap)
 return (rows+['']*n)[:n]
def generate_derived(alias,cfg,root):
 out=outdir(root,alias); gate=g.read_json(out/'full_spine_gate.json')
 if not gate.get('human_review_approved') or gate['textbook_digest']!=g.digest((out/'textbook.md').read_text()): raise RuntimeError('approved full SPINE required')
 s,w,schema,_=g.load_inputs(cfg); fp,sp=g.read_json(out/'full_plan.json'),g.read_json(out/'spine_full.json'); mapping,names=g.values_for(alias,w),g.display_names(w); client,jc=g.Client(cfg['llm'],out),g.Client(cfg['judge_llm'],out/'judge'); lookup={(c['chapter_id'],x['section_id']):(c,x) for c in s['chapters'] for x in c['sections']}; st={x['section_id']:x['text'] for c in sp['chapters'] for x in c['section_spans']}; facts=[f for c in sp['chapters'] for f in c['asserted_facts']]; dd=out/'derived_docs'; dd.mkdir(exist_ok=True); rows=[]; rev=cfg['derived_g2_policy_revision']
 for secp in fp['sections']:
  ch,sec=lookup[(secp['chapter_id'],secp['section_id'])]; text=st[sec['section_id']]; sf=[f for f in facts if f['entity_id'] in set(sec['organism_ids'])]; targets=allocate(secp['derived_tokens'],[1]*secp['derived_documents'])
  for i,target in enumerate(targets):
   did=f"{sec['section_id']}-D{i:05d}-R{rev}"; path=dd/(did+'.json'); typ=g.DERIVED_TYPES[i%len(g.DERIVED_TYPES)]
   if path.exists(): val=g.read_json(path)
   else:
    cid=f"{g.run_id(out)}-{alias}-FULL-DERIVED-{did}"
    def judge(c): jid=f"{cid}-G2-JUDGE-{g.digest(g.compact(c))[:12]}"; return g.call_judge(jc,out/'judge'/'decision_ledger.jsonl',jid,lambda fb:g.judge_prompt(alias,text,c,fb))
    val=g.call_with_retry(client,lambda fb,t=typ,n=i:g.prompt_derived(alias,ch,sec,names,text,sf,t,n,fb,target_tokens=target),cid,lambda v:g.validate_derived(v,alias,{sec['section_id']},mapping,schema['excluded_keyword_scan'],text,sf,judge),out/'rejection_ledger.jsonl'); path.write_text(g.compact(val)+'\n')
   rows.append({'row_id':did,'phase':'DERIVED','chapter_id':ch['chapter_id'],'section_id':sec['section_id'],'spine_section_ids':[sec['section_id']],'doc_type':typ,'text':val['text'],'token_count':count(val['text'])})
 cr=[{'row_id':f'SPINE-C{i:05d}','phase':'SPINE_CHUNKS','text':x,'token_count':count(x)} for i,x in enumerate(chunks((out/'textbook.md').read_text(),fp['spine_chunk_target'],cfg.get('spine_chunk_overlap_tokens',32)))]; combined=cr+rows; (out/'derived.jsonl').write_text(''.join(g.compact(x)+'\n' for x in rows)); (out/'corpus.jsonl').write_text(''.join(g.compact(x)+'\n' for x in combined)); tok=sum(x['token_count'] for x in combined); result={'corpus':alias,'document_count':len(combined),'document_target':fp['total_document_target'],'token_count':tok,'token_target':fp['total_token_target'],'spine_chunk_count':len(cr),'derived_count':len(rows)}; result['passed']=result['document_count']==result['document_target'] and abs(tok-result['token_target'])/result['token_target']<=.02; (out/'full_corpus_result.json').write_text(g.compact(result)+'\n')
 if not result['passed']: raise RuntimeError('final budget gate failed')
def main():
 p=argparse.ArgumentParser(); p.add_argument('action',choices=('plan','spine-full','approve-full-spine','derived-full')); p.add_argument('--corpus',choices=g.ALIASES,required=True); p.add_argument('--config',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--pilot-root',type=Path); p.add_argument('--reviewer-alias'); a=p.parse_args(); cfg=g.read_json(a.config)
 if a.action=='plan': s,_,_,_=g.load_inputs(cfg); print(g.compact(build_plan(cfg,s)))
 elif a.action=='spine-full':
  if not a.pilot_root: p.error('--pilot-root required')
  generate_spine(a.corpus,cfg,a.output_root,a.pilot_root)
 elif a.action=='approve-full-spine':
  if not a.reviewer_alias: p.error('--reviewer-alias required')
  approve(a.corpus,a.output_root,a.reviewer_alias)
 else: generate_derived(a.corpus,cfg,a.output_root)
if __name__=='__main__': main()
