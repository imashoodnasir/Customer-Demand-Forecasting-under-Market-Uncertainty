from __future__ import annotations
import argparse
from bayesian_retail.transfer.adaptation import create_transfer_prior

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source-run",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    print(create_transfer_prior(f"{a.source_run}/inference_data.nc",a.output))
if __name__=="__main__": main()
