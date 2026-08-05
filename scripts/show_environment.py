import argparse
import json
from bayesian_retail.config import load_config
from bayesian_retail.environment import collect_environment
from bayesian_retail.paths import ProjectPaths
from scripts._common import add_config_arguments

def main():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    args = parser.parse_args()
    ProjectPaths.from_config(load_config(args.config, args.set)).create()
    print(json.dumps(collect_environment(), indent=2))

if __name__ == "__main__":
    main()
