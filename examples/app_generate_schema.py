"""Example: Generate JSON Schema and default config by app tags.

Usage:
  python examples/app_generate_schema.py --app train --prefix train_config
  -> writes train_config.schema.json and train_config.json

  python examples/app_generate_schema.py --app serve --prefix serve_config
  -> writes serve_config.schema.json and serve_config.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import examples.myapp.pipeline_params  # noqa: F401 - register the app parameters

from tunablex.runtime import schema_for_app, write_schema


def main():
    parser = argparse.ArgumentParser(prog="app_generate_schema")
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
