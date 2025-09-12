from jsonargparse import ArgumentParser
from tunablex import use_config
from tunablex.cli_helpers import add_flags_by_trace, build_cfg_from_file_and_args
import examples.myapp.pipeline as pipeline  # registers @tunable

if __name__ == "__main__":
    parser = ArgumentParser(prog="train_jsonarg_trace")
    parser.add_argument("--config", help="Path to train_config.json (optional)")

    # auto-generate flags --section.field for all tunables reached by tracing train_main
    AppConfig = add_flags_by_trace(parser, pipeline.train_main)

    args = parser.parse_args()

    cfg_dict = build_cfg_from_file_and_args(AppConfig, args, config_attr="config")
    cfg = AppConfig.model_validate(cfg_dict)

    with use_config(cfg):
        pipeline.train_main()
