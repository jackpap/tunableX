from jsonargparse import ArgumentParser  # could be argparse as well
from tunablex import use_config
from tunablex.runtime import load_app_config, schema_by_trace, write_schema
import examples.myapp.pipeline as pipeline  # registers @tunable

if __name__ == "__main__":
    parser = ArgumentParser(prog="train_app")
    parser.add_argument("--config", required=True, help="Path to train_config.json")
    parser.add_argument("--gen-schema", action="store_true", help="Generate JSON Schema & defaults by tracing and exit")
    parser.add_argument("--schema-prefix", default="train_config", help="Prefix for generated files when using --gen-schema")
    args = parser.parse_args()

    if args.gen_schema:
        schema, defaults, _ = schema_by_trace(pipeline.train_main)
        write_schema(args.schema_prefix, schema, defaults)
        print(f"Wrote {args.schema_prefix}.schema.json and {args.schema_prefix}.json")
    else:
        cfg = load_app_config(app="train", json_path=args.config)
        with use_config(cfg):
            pipeline.train_main()
