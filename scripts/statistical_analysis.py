from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from bayesian_retail.metrics.statistical import friedman_nemenyi


def main():
    p=argparse.ArgumentParser(); p.add_argument("--metrics",required=True); p.add_argument("--output-dir",default="outputs/statistics"); a=p.parse_args()
    df=pd.read_csv(a.metrics); required={"model","value"};
    if not required.issubset(df): raise ValueError("Metrics file requires model and value columns")
    task_cols=[c for c in ["dataset","metric","horizon","hierarchy","seed"] if c in df]
    pivot=df.pivot_table(index=task_cols,columns="model",values="value",aggfunc="mean").dropna()
    result=friedman_nemenyi(pivot); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    result["average_ranks"].rename_axis("model").reset_index().to_csv(out/"average_ranks.csv",index=False)
    payload={k:v for k,v in result.items() if k!="average_ranks"}; (out/"friedman.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
