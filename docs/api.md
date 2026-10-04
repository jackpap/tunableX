# API reference

All functions listed here are exported from `tunablex` unless noted otherwise.

## Declaration

### `tunable(*include, namespace="", exclude=(), apps=())`

A decorator factory. With an explicit include list, selects exactly those named
parameters (including required ones). Otherwise selects parameters with defaults,
except names in `exclude`. `include` and `exclude` cannot be combined. Unknown
names, selected variadic arguments, invalid namespaces and conflicting shared
fields raise `ValueError` at registration.

Selected parameters use their type annotation, including `Annotated` metadata.
Unannotated parameters use `Any`. Only selected annotations are resolved; unrelated
annotations may refer to names imported under `TYPE_CHECKING`. Names needed for a
selected annotation must exist at runtime in its defining module.

`namespace` is a dotted path. Each component and field name must be a public Python
identifier that does not collide with a Pydantic `BaseModel` attribute. A field and
a nested namespace cannot occupy the same path. Shared definitions at a path must
have matching types, defaults and Field metadata. Their app tags are combined.
Untagged declarations are included in **every** app-tag model (`ALL` semantics).
Entrypoint composition selects only discovered functions, irrespective of app tags.

The decorator preserves metadata and signature through `functools.wraps`.
Stacked decorators can put different arguments in different namespaces.
`staticmethod` and `classmethod` work in either decorator order.

At call time, precedence is explicit Python arguments, then the active config,
then declared defaults. Positional-only and keyword-only arguments are supported.
Pydantic `Field` defaults and centralized references are resolved before selected
arguments reach the function. Missing required values raise `TypeError` without
executing the function. Data-dependent factories see previously resolved selected
fields by canonical configuration name within their own namespace, even when the
function uses renamed arguments. Explicit Python arguments and raw dictionary
configs are not validated by the decorator; use a composed model to validate external input.

### `TunableParams`

Declare annotated `Field(...)` class attributes and use them as function defaults:

```python
class ModelParams(TunableParams):
    width: int = Field(64, ge=1)

@tunable(apps="train")
def build(size=ModelParams.width):
    return size
```

`ModelParams.width` is a declaration reference, not the numeric value. Plain class
attributes are ordinary Python attributes; use `Field` for centralized declarations.
References carry a field name, type and namespace. A reference's namespace takes
precedence over the decorator's `namespace` argument. Aliases/nesting create a new
namespace view without changing the original class. Descriptions come from
`Field(description=...)` or the string immediately after the annotated assignment.

Postponed type hints are resolved per field and cached in the declaring class;
unrelated unresolved hints do not prevent use of valid fields. This supports
`spawn`/`forkserver` imports, including `__mp_main__` execution namespaces.
For predictable resolution, declare shared parameter classes at module scope.

## Models, schemas and files

| Function | Result |
| --- | --- |
| `make_config_for_app(app)` | A generated `type[BaseModel]` for one app tag |
| `make_config_for_entry(entrypoint)` | A generated `type[BaseModel]` selected by static calls |
| `schema_for_app(app)` | `(schema_dict, defaults_dict)` |
| `schema_for_entrypoint(entrypoint)` | `(schema_dict, defaults_dict)` |
| `load_config_for_app(app, json_path)` | A validated config instance |
| `load_config_for_entry(entrypoint, json_path)` | A validated config instance |
| `write_schema(prefix, schema, defaults=None, *, yaml=None)` | Writes files; returns `None` |

Generated models validate defaults and forbid unknown fields. Use `.model_validate`
to validate data, `.model_dump(mode="json")` to get a serializable payload, and
`.model_json_schema()` for schema output. `json_path` is retained as the argument
name for compatibility; JSON, YAML and TOML are all accepted.

Loaders raise `pydantic.ValidationError` for invalid settings, native parse errors
for malformed documents, `ValueError` for nonmapping roots, and `OSError` for file
errors. They do not terminate the process. YAML uses `safe_load`; an empty YAML
document means `{}`. Scalar values (`false`, `null`) and arrays are invalid JSON
config roots. Unknown file extensions try JSON and then YAML.

Defaults templates omit required fields and factories that require already
validated data. Exported defaults retain field validation, serializers and
exclusions. Ordinary zero-argument factories are evaluated during defaults export.
Templates are not guaranteed to be complete valid configs. Schema-only export using `Config.model_json_schema()` does not evaluate those factories.

`write_schema` creates parent directories, writes UTF-8 JSON with a trailing newline,
and overwrites files with the same names. `yaml=None` also writes YAML if PyYAML is
available; `yaml=True` requires it; `yaml=False` writes only JSON. No TOML writer is
provided. Use `defaults=None` for a schema-only file.

## Runtime context

### `with use_config(cfg): ...`

Accepts a Pydantic model or raw dict. Contexts nest and restore the prior value on
normal exit or exceptions. Values are isolated by Python `ContextVar`: async tasks
inherit the creating context and can activate their own values. Async decorated
functions resolve their configuration when awaited, not when a coroutine object
is first created. A synchronous generator resolves it on first iteration. Async
generators keep their native send/throw protocol and resolve config when created;
create them inside the intended context.

New OS threads and processes require their own context. Pass a serializable dict
to workers, import the tunable definitions there, rebuild the config model, validate
the dict and activate `use_config` in the worker. Generated model classes are not
promised to be pickleable across process boundaries.

## Command-line helpers

### `add_flags_by_app(parser, app)` / `add_flags_by_entry(parser, entrypoint)`

Add dotted flags and return the generated config **class**. The helpers accept a
standard argparse or jsonargparse parser. Actual defaults are deliberately absent
from the parsed namespace so omitted flags cannot overwrite a file. Help displays
static defaults and constraints remain enforced by the final model. Factory
values are labelled `default: factory` without evaluating them for help.

### `build_cfg_from_file_and_args(Config, args, config_attr="config")`

Read an optional file named by `args.config` (or the named alternative), merge
explicit flags, apply defaults and validate. Return a JSON-compatible dict. Build
a model instance with `Config.model_validate(data)` before runtime injection to
retain Python types such as Path, enum and nested BaseModel instances.

Collections accept space-separated items (zero items clears the collection);
fixed tuples, mappings and optional/recursive model fields accept a JSON token.
Nullable fields accept `null`; a string-only field treats `null` as text. Boolean
fields support both positive and negative flags. Names containing double underscores
are preserved. Pydantic aliases are accepted by models, but generated flags use
canonical Python field names; avoid mixing aliases and canonical keys in one config.

Lower-level helpers are in `tunablex.cli_helpers`: `add_flags_from_model`,
`collect_overrides`, and `deep_update`. Parser destinations are internal and should
not be used as a public API.

## Schema CLI

```text
tunablex schema --app APP --import MODULE [MODULE ...] [--out PREFIX]
tunablex analyze --entry MODULE:FUNCTION [--import MODULE ...] [--out PREFIX]
```

Both commands accept `--sys-path PATH [PATH ...]`. `python -m tunablex` and
`python -m tunablex.cli` are equivalent entrypoints. The analyze command imports its
entry module automatically; `--import` can register additional definitions.
