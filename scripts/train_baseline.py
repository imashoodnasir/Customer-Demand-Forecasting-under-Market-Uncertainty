from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from bayesian_retail.config import load_config
from bayesian_retail.paths import ProjectPaths
from bayesian_retail.run_context import RunContext
from bayesian_retail.baselines.registry import create_forecaster, registered_models
from bayesian_retail.baselines.utils import save_json
from scripts._common import add_config_arguments

def main():
    p=argparse.ArgumentParser(); add_config_arguments(p); p.add_argument('--dataset',required=True); p.add_argument('--model',choices=registered_models(),required=True); p.add_argument('--seed',type=int,required=True); a=p.parse_args()
    cfg=load_config(a.config,a.set); paths=ProjectPaths.from_config(cfg); ctx=RunContext.create(cfg,a.dataset,f'baseline-{a.model}',a.seed)
    base=paths.processed_root/a.dataset
    common=cfg.baselines.common.model_dump(); common['work_dir']=str(ctx.run_dir)
    model_cfg=getattr(cfg.baselines,a.model).model_dump()
    if a.model=='prophet':
        model=create_forecaster(a.model,model_config=model_cfg,common_config=common,seed=a.seed)
        result=model.fit(base/'feature_table.parquet')
    else:
        meta=json.loads((base/'tensors'/'tensor_metadata.json').read_text())['train']; arr=np.load(base/'tensors'/'train.npz')
        dims={'history_length':arr['history_target'].shape[1],'horizon':arr['targets'].shape[1]}
        if a.model=='deepar': dims.update({'past_features':arr['past_covariates'].shape[2],'future_features':arr['future_covariates'].shape[2],'n_series':int(arr['series_index'].max())+1})
        model=create_forecaster(a.model,model_config=model_cfg,common_config=common,dimensions=dims,seed=a.seed)
        result=model.fit(base/'tensors'/'train.npz',base/'tensors'/'validation.npz')
    model.save(ctx.run_dir/'model'); save_json(result,ctx.run_dir/'baseline_training.json'); print(ctx.run_dir)
if __name__=='__main__': main()
