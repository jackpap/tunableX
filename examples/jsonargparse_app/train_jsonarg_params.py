"""jsonargparse example using centralized TunableParameters classes.

App-tag composition picks up namespaces defined via class inheritance.
"""
import examples.myapp.params as params  # noqa: F401
import examples.myapp.pipeline_params as pipeline
from jsonargparse import ArgumentParser

from tunablex import use_config
from tunablex.cli_helpers import add_flags_by_app
from tunablex.cli_helpers import build_cfg_from_file_and_args

if __name__ == "__main__":
    parser = ArgumentParser(prog="train_jsonarg_params")
    parser.add_argument("--config", help="Path to train_config.json (optional)")

    AppConfig = add_flags_by_app(parser, app="train")

    args = parser.parse_args()
    cfg_dict = build_cfg_from_file_and_args(AppConfig, args, config_attr="config")
    cfg = AppConfig.model_validate(cfg_dict)

    with use_config(cfg):
        pipeline.train_main()
