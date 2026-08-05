from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from bayesian_retail.config import load_config
from bayesian_retail.data.bundle import ProcessedDataset
from bayesian_retail.forecasting.aggregation import build_aggregation_matrix, aggregate_posterior_samples
from bayesian_retail.paths import ProjectPaths
from scripts._common import add_config_arguments


def main():
    p=argparse.ArgumentParser(); add_config_arguments(p)
    p.add_argument("--dataset",required=True); p.add_argument("--run-dir",required=True); p.add_argument("--split",default="test")
    p.add_argument("--levels",nargs="+",default=["series","category","department","store","region","overall"])
    a=p.parse_args(); cfg=load_config(a.config,a.set); paths=ProjectPaths.from_config(cfg); run=Path(a.run_dir)
    bundle=ProcessedDataset.load(paths.processed_root/a.dataset); predictive=np.load(run/f"posterior_predictive_{a.split}.npz")["samples"]
    # Window-level predictions are mapped to series using tensor series_index. Average duplicate windows per series.
    tensors=np.load(paths.processed_root/a.dataset/"tensors"/f"{a.split}.npz"); series_idx=tensors["series_index"]
    n_series=bundle.metadata["hierarchy_counts"]["series"]
    draws,horizon=predictive.shape[0],predictive.shape[2]
    per_series=np.zeros((draws,n_series,horizon)); counts=np.zeros(n_series)
    for w,s in enumerate(series_idx): per_series[:,int(s),:]+=predictive[:,w,:]; counts[int(s)]+=1
    counts=np.maximum(counts,1); per_series/=counts[None,:,None]
    metadata={}
    for level in a.levels:
        agg=build_aggregation_matrix(bundle.hierarchy,level); values=aggregate_posterior_samples(per_series,agg)
        np.savez_compressed(run/f"hierarchical_{level}_{a.split}.npz",samples=values,mean=values.mean(0),median=np.median(values,0),labels=np.asarray(agg.labels))
        metadata[level]={"shape":list(values.shape),"nodes":len(agg.labels)}
    (run/f"hierarchical_{a.split}_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8"); print(json.dumps(metadata,indent=2))
if __name__=="__main__": main()
