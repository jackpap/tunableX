"""Example: Generate JSON Schema and default config by static AST analysis of an entrypoint.

Usage:
  python examples/trace_generate_schema.py --entry train --prefix train_config
  -> writes train_config.schema.json and train_config.json

  python examples/trace_generate_schema.py --entry serve --prefix serve_config
  -> writes serve_config.schema.json and serve_config.json
"""

from __future__ import annotations

import argparse

import examples.myapp.pipeline_params as pipeline_params

from tunablex.runtime import schema_for_app
from tunablex.runtime import write_schema


def main():
    parser = argparse.ArgumentParser(prog="trace_generate_schema")
    parser.add_argument(
        "--app",
        choices=["train", "serve"],
        default="train",
        help="Which app to analyze (train or serve)",
    )
    parser.add_argument(
        "--prefix",
        default="config",
        help="Output file prefix (writes <prefix>.schema.json and <prefix>.json)",
    )
    args = parser.parse_args()

    schema, defaults = schema_for_app(args.app)
    write_schema(args.prefix, schema, defaults)

    print(f"Wrote {args.prefix}.schema.json and {args.prefix}.json")


if __name__ == "__main__":
    main()
