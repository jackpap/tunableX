from __future__ import annotations

import argparse
import importlib
import json
import sys

from .runtime import schema_for_app
from .runtime import schema_for_entrypoint
from .runtime import write_schema


def _import_modules(mods):
    for m in mods:
        importlib.import_module(m)


def _add_sys_paths(paths):
    for p in paths or []:
        if p not in sys.path:
            sys.path.insert(0, p)


def main(argv=None):
    p = argparse.ArgumentParser(prog="tunablex", description="Generate JSON Schema/defaults from @tunable functions.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("schema", help="Emit schema/defaults for one or more apps (by tags).")
    s.add_argument("--app", required=True)
    s.add_argument("--import", dest="imports", nargs="+", required=True)
    s.add_argument(
        "--sys-path", dest="sys_paths", nargs="+", default=[], help="Paths to insert into sys.path before imports."
    )
    s.add_argument("--out", default=None)

    t = sub.add_parser("analyze", help="Emit schema/defaults by static AST analysis of a module:function entrypoint.")
    t.add_argument("--entry", required=True)
    t.add_argument("--import", dest="imports", nargs="+", required=True)
    t.add_argument(
        "--sys-path", dest="sys_paths", nargs="+", default=[], help="Paths to insert into sys.path before imports."
    )
    t.add_argument("--out", default=None)

    args = p.parse_args(argv)
    _add_sys_paths(args.sys_paths)
    _import_modules(args.imports)

    if args.cmd == "schema":
        schema, defaults = schema_for_app(args.app)
        if args.out:
            write_schema(args.out, schema, defaults)
        else:
            print(json.dumps({"schema": schema, "defaults": defaults}, indent=2, default=str))
        return 0

    if args.cmd == "analyze":
        modname, funcname = args.entry.split(":")
        fn = getattr(importlib.import_module(modname), funcname)
        schema, defaults, touched = schema_for_entrypoint(fn)
        if args.out:
            write_schema(args.out, schema, defaults)
        else:
            print(
                json.dumps({"schema": schema, "defaults": defaults, "touched": sorted(touched)}, indent=2, default=str)
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
