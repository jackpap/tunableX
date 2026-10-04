<p align="center">
  <img src="https://raw.githubusercontent.com/jackpap/tunableX/main/assets/tunableX-title.png" alt="tunableX" width="100%" />
</p>

# tunableX

**Typed configuration, declared where your Python functions use it.**

`tunableX` turns selected function arguments into Pydantic configuration models,
JSON Schema, JSON/YAML defaults and command-line flags. Compose a configuration
using app tags or static analysis of an entrypoint, then inject it with `use_config`.

- Keep parameters beside the logic, or share them through `TunableParams` classes.
- Validate types, constraints and unknown settings with Pydantic v2.
- Load JSON, YAML or TOML; explicit CLI flags override file values and defaults.
- Support ordinary functions, methods, static/class methods and async functions.
- Discover calls without running the entrypoint; use tags for dynamic dispatch.

## Installation

Python **3.10+** and Pydantic **2.12.5+ (<3)** are required.

```bash
pip install tunablex                 # JSON, TOML and standard argparse
pip install 'tunablex[yaml]'         # add YAML
pip install 'tunablex[jsonargparse]'  # optional parser integration
pip install 'tunablex[all]'          # both optional integrations
```

The base package does not require PyYAML or jsonargparse. On Python 3.10,
`tomli` is installed automatically for TOML support.

## A complete example

Save as `train.py`:

```python
from argparse import ArgumentParser
from pydantic import Field
from tunablex import tunable, add_flags_by_app, build_cfg_from_file_and_args, use_config

@tunable(namespace="train", apps="training")
def train(epochs: int = Field(10, ge=1, description="Number of training epochs"),
          learning_rate: float = Field(0.001, gt=0),
          verbose: bool = False):
    print(epochs, learning_rate, verbose)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", help="JSON, YAML or TOML configuration")
    Config = add_flags_by_app(parser, "training")
    args = parser.parse_args()
    cfg = Config.model_validate(build_cfg_from_file_and_args(Config, args))
    with use_config(cfg):
        train()
```

```bash
python train.py --train.epochs 25 --train.verbose
python train.py --config train.json --no-train.verbose
python train.py --help
```

`train.json` can contain only the values you want to change:

```json
{"train": {"epochs": 20, "learning_rate": 0.01}}
```

Precedence is **function/model defaults → file → explicit CLI flags**.
Explicit Python arguments have the final say: `train(epochs=3)` uses `3` even
inside an active config. A direct `train()` outside a context resolves declared
`Field` defaults too; metadata objects are never passed as selected argument values.

## Shared parameter declarations

Use `Field` attributes on a `TunableParams` subclass. A class name becomes a
snake_case namespace with the `Params` suffix removed; `MainParams` and
`RootParams` represent the root. **Nesting/aliases**, rather than inheritance,
create nested namespaces. Inheritance reuses fields within the subclass namespace.

```python
from pydantic import Field
from tunablex import TunableParams, make_config_for_app, tunable, use_config

class OptimizerParams(TunableParams):
    rate: float = Field(0.01, gt=0, description="Learning rate")

class TrainParams(TunableParams):
    epochs: int = Field(10, ge=1)
    Optimizer = OptimizerParams

@tunable(apps="training")
def step(epochs=TrainParams.epochs, lr=TrainParams.Optimizer.rate):
    return epochs, lr

Config = make_config_for_app("training")
with use_config(Config.model_validate({"train": {"optimizer": {"rate": 0.1}}})):
    assert step() == (10, 0.1)
```

This exposes `--train.epochs` and `--train.optimizer.rate`; a local argument can
have a different name (`lr`). Reusing `OptimizerParams` under multiple parents
keeps each namespace independent. A string immediately following a field
assignment supplies its description when `Field(description=...)` is absent.

## Composition and schema generation

Import your decorated functions before composing a model.

```python
from tunablex import make_config_for_app, schema_for_app, write_schema

Config = make_config_for_app("training")
schema, defaults = schema_for_app("training")
write_schema("config/train", schema, defaults, yaml=False)
```

This writes `config/train.schema.json` and `config/train.json`. Omit `yaml=False`
to also write `.yml` when PyYAML is installed, or set `yaml=True` to require it.
Required values are omitted from defaults templates and remain required by the
schema. A template with required fields must be completed before use.

You can discover an entrypoint's reachable functions instead of tagging them:

```python
from tunablex import make_config_for_entry, schema_for_entrypoint
from myapp.pipeline import train_main

Config = make_config_for_entry(train_main)
schema, defaults = schema_for_entrypoint(train_main)
```

Discovery follows direct functions, imported/module aliases, closures, simple
local aliases, constructors and statically resolvable methods. It examines both
branches and never calls application functions or evaluates properties.
It is conservative: for callback tables, factories or runtime plugin selection,
use explicit app tags. Imports, annotation evaluation and default factories are
ordinary Python execution; schema generation is **not an untrusted-code sandbox**.

The installed CLI supports the same flows:

```bash
tunablex schema --app training --import train --out config/train
tunablex analyze --entry myapp.pipeline:train_main --out config/train
python -m tunablex analyze --entry myapp.pipeline:train_main
```

Omit `--out` to print a JSON object containing `schema` and `defaults`.
Use `--sys-path src` for a package that is not yet installed.

## CLI values

Both `argparse.ArgumentParser` and `jsonargparse.ArgumentParser` are supported.
`add_flags_by_entry(parser, entrypoint)` selects flags through static discovery.

| Parameter type | Example |
| --- | --- |
| `int`, `float`, `str`, `Path` | `--train.epochs 20` |
| `bool` | `--train.verbose` / `--no-train.verbose` |
| `Literal`, enum | `--optimizer adam`, `--level 2` |
| `list[int]`, `Sequence[int]`, `tuple[int, ...]` | `--layers 128 256` |
| Empty collection | `--layers` with no following values |
| Fixed tuple | `--shape '[640,480]'` |
| Mapping | `--weights '{"main": 0.8}'` |
| Nullable value | `--seed null` |
| Nested model | `--section.field value` |

Required fields can come from the config file or CLI. Validation happens after
merging. Unknown fields are errors, including nested misspellings. Strings are
not automatically treated as Python expressions; JSON is used for structured values.

## Documentation and examples

- [API reference](docs/api.md): public functions, types and error behavior.
- [Design, limitations and review findings](docs/design-and-limitations.md).
- [Migration notes and changelog](CHANGELOG.md).
- [Contributor and release guide](CONTRIBUTING.md).
- [Examples](examples/): app tags, shared declarations, CLI integration and schemas.

Run examples as modules from a source checkout:

```bash
python -m examples.jsonargparse_app.train_jsonarg_app --train.epochs 20
python -m examples.jsonargparse_app.train_jsonarg_params --model.hidden_sizes 128 256
python -m examples.trace_generate_schema --entry train --prefix config/train
python -m examples.argparse_trace.train_trace --config config/train.json
```

Directories named `*_trace` retain their historical names; they now use static
analysis. Current API names are `TunableParams`, `make_config_for_app`,
`schema_for_app`, `make_config_for_entry` and `schema_for_entrypoint`.
Earlier README names such as `TunableParameters` and `schema_by_trace` are not
exported aliases.

## Contributors and license

Created by **Jacques Papper**, with core contributions by **Vincent Drouet**.
See [CONTRIBUTORS.md](CONTRIBUTORS.md). Contributions are welcome.

[MIT license](LICENSE).
