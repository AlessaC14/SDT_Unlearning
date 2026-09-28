#!/usr/bin/env python3
"""Durable sequential Spec E grid worker; one process per physical GPU."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));from spec_e_common import planned_grid,read_config
p=argparse.ArgumentParser();p.add_argument('--worker-index',type=int,choices=[0,1],required=True);p.add_argument('--config',default='configs/spec_e.preflight.json');a=p.parse_args();root=Path(__file__).resolve().parents[1];c=read_config(root/a.config)
cells=planned_grid(c);priority={'unfiltered-cb':0,'unfiltered':1,'e2e-strong-filter':2};cells.sort(key=lambda x:(priority[x['checkpoint']],0 if x['condition']=='retain-only' else 1,x['seed'],x['learning_rate']))
cells=[x for i,x in enumerate(cells) if i%2==a.worker_index];ledger=root/'spec_e_outputs/grid'/f'worker-{a.worker_index}.jsonl';ledger.parent.mkdir(parents=True,exist_ok=True)
def log(x):
 with ledger.open('a') as f:f.write(json.dumps(x,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
for cell in cells:
 cmd=[sys.executable,str(root/'scripts/spec_e_run.py'),'--model',cell['checkpoint'],'--condition',cell['condition'],'--learning-rate',str(cell['learning_rate']),'--seed',str(cell['seed'])]
 log({'event':'launch','run_id':cell['run_id'],'unix':time.time(),'command':cmd});r=subprocess.run(cmd,cwd=root);log({'event':'exit','run_id':cell['run_id'],'unix':time.time(),'returncode':r.returncode})
print(json.dumps({'status':'worker_complete','worker':a.worker_index,'cells':len(cells)}))
