import examples.myapp.pipeline as pipeline
from jsonargparse import ArgumentParser

from tunablex import use_config
from tunablex.cli_helpers import add_flags_by_app, build_cfg_from_file_and_args

if __name__ == "__main__":
    # Use app-tag based flag generation since we know we're the 'train' app.
    parser = ArgumentParser(prog="train_jsonarg_app")
    parser.add_argument("--config", help="Path to train_config.json (optional)")

    AppConfig = add_flags_by_app(parser, app="train")

    args = parser.parse_args()

    # merge: defaults <- file (optional) <- CLI flags
    cfg_dict = build_cfg_from_file_and_args(AppConfig, args, config_attr="config")
    cfg = AppConfig.model_validate(cfg_dict)

    with use_config(cfg):
        pipeline.train_main()
