import argparse
from bayesian_retail.config import load_config
from bayesian_retail.run_context import RunContext
from scripts._common import add_config_arguments

def main():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    context = RunContext.create(load_config(args.config, args.set),
                                args.dataset, args.experiment, args.seed)
    print(context.run_dir)

if __name__ == "__main__":
    main()
