from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from bayesian_retail.config import load_config
from bayesian_retail.paths import ProjectPaths
from bayesian_retail.baselines.registry import REGISTRY, registered_models
from bayesian_retail.baselines.evaluation import evaluate_output
from bayesian_retail.baselines.utils import save_json
from scripts._common import add_config_arguments

def latest(paths,dataset,model,seed):
    m=sorted(paths.runs_root.glob(f'*_{dataset}_baseline-{model}_seed{seed}*'))
    if not m: raise FileNotFoundError('No matching baseline run')
    return m[-1]
def main():
    p=argparse.ArgumentParser(); add_config_arguments(p); p.add_argument('--dataset',required=True); p.add_argument('--model',choices=registered_models(),required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--split',default='test'); p.add_argument('--run-dir'); a=p.parse_args()
    cfg=load_config(a.config,a.set); paths=ProjectPaths.from_config(cfg); run=Path(a.run_dir) if a.run_dir else latest(paths,a.dataset,a.model,a.seed); model=REGISTRY[a.model].load(run/'model')
    base=paths.processed_root/a.dataset
    if a.model=='prophet':
        import pandas as pd
        frame=pd.read_parquet(base/'feature_table.parquet'); frame=frame[frame['split']==a.split]; out=model.predict_frame(frame,cfg.baselines.common.samples); targets=frame['demand'].to_numpy()[:,None]; mask=None
    else:
        path=base/'tensors'/f'{a.split}.npz'; data=np.load(path); out=model.predict(path,cfg.baselines.common.samples); targets=data['targets']; mask=data['target_observed_mask']
    np.savez_compressed(run/f'predictions_{a.split}.npz',point=out.point,samples=out.samples if out.samples is not None else np.array([]),targets=targets)
    metrics=evaluate_output(out,targets,mask); save_json(metrics,run/f'metrics_{a.split}.json'); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
