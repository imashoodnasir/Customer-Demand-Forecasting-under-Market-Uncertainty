import argparse, subprocess, sys
from bayesian_retail.config import load_config
from bayesian_retail.baselines.registry import registered_models
from scripts._common import add_config_arguments

def main():
    p=argparse.ArgumentParser(); add_config_arguments(p); p.add_argument('--dataset',required=True); p.add_argument('--models',nargs='+',choices=registered_models(),default=registered_models()); a=p.parse_args(); cfg=load_config(a.config,a.set)
    for model in a.models:
        for seed in cfg.experiment.seeds:
            cmd=[sys.executable,'-m','scripts.train_baseline','--dataset',a.dataset,'--model',model,'--seed',str(seed),'--config',a.config]
            for x in a.set: cmd += ['--set',x]
            subprocess.run(cmd,check=True)
if __name__=='__main__': main()
