def add_config_arguments(parser):
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[])
