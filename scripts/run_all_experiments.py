import argparse
from bayesian_retail.experiments.runner import ExperimentRunner

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["m5","favorita"]
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11,22,33]
    )
    args=parser.parse_args()

    runner=ExperimentRunner(
        args.datasets,
        args.seeds
    )

    for job in runner.run():
        print(job)

if __name__=="__main__":
    main()
