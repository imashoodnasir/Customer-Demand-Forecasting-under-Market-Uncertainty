import argparse
from bayesian_retail.config import load_config
from bayesian_retail.validation import validate_runtime_configuration
from scripts._common import add_config_arguments

def main():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    args = parser.parse_args()
    for message in validate_runtime_configuration(load_config(args.config, args.set)):
        print(message)

if __name__ == "__main__":
    main()
