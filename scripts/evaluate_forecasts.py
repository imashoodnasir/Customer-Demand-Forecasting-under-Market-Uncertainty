from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from bayesian_retail.config import load_config
from bayesian_retail.forecasting.evaluation import evaluate_forecasts
from bayesian_retail.paths import ProjectPaths
from scripts._common import add_config_arguments


def main():
    p=argparse.ArgumentParser(); add_config_arguments(p)
    p.add_argument("--dataset", required=True); p.add_argument("--run-dir", required=True); p.add_argument("--split", default="test")
    a=p.parse_args(); cfg=load_config(a.config,a.set); paths=ProjectPaths.from_config(cfg)
    run=Path(a.run_dir); pred=np.load(run/f"posterior_predictive_{a.split}.npz")
    tensors=np.load(paths.processed_root/a.dataset/"tensors"/f"{a.split}.npz")
    samples=pred["samples"]; point=pred["median"] if "median" in pred else np.median(samples,axis=0)
    metrics=evaluate_forecasts(tensors["targets"],point,samples,cfg.experiment.interval_levels)
    out=pd.DataFrame([{"dataset":a.dataset,"split":a.split,"metric":k,"value":v} for k,v in metrics.items()])
    out.to_csv(run/f"metrics_{a.split}.csv",index=False); print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
