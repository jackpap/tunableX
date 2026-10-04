# Changelog

## Unreleased — repository review

### Fixes and capabilities

- Validate and serialize exported defaults with their field schemas, preserving
  annotation serializers and field exclusions.
- Resolve data-dependent factory inputs by canonical field name within each
  namespace, including renamed centralized arguments on direct calls.

- Support Python 3.14 deferred annotations without evaluating unrelated fields.
- Incorporate PR #10's cached per-field annotations for spawned/forkserver workers.
- Honor positional, keyword and positional-only overrides; preserve nested Python values.
- Resolve Field/reference defaults on direct calls, including fresh default factories.
- Preserve static/class method descriptors and async task configuration.
- Support required file/CLI settings and partial schema defaults templates.
- Expand CLI support to generic collections, numeric Literals, enums, mappings,
  fixed tuples, nullable values and empty collections; preserve double underscores.
- Follow imported aliases, closures, constructors and simple instance methods during
  static discovery; prevent unrelated same-name functions from contaminating schemas.
- Reject invalid field names, field/namespace collisions, unknown config keys and
  conflicting shared definitions. Equivalent Field declarations can be shared.
- Avoid mutable centralized namespace aliases; extract field descriptions structurally.
- Make YAML optional in practice, support TOML on Python 3.10, validate file root shape,
  and write UTF-8 schemas/defaults with parent directory creation.
- Repair the schema CLI and install `tunablex` / `python -m tunablex` entrypoints.
- Replace stale documentation, fix examples, reinstate skipped regressions, simplify
  lint configuration, add lightweight CI and include typing/package metadata.

### Migration from 0.1.7

- **Unknown settings now raise ValidationError** rather than being ignored. Remove
  stale/misspelled keys, and pass `*.json` config files rather than `*.schema.json`.
- **Defaults are validated.** A default violating its type or Field constraints fails
  when constructing a config.
- **Loaders raise exceptions**, including Pydantic ValidationError, instead of
  SystemExit. CLI callers may catch these and use `parser.error(str(exc))`.
- **`build_cfg_from_file_and_args` validates its merged result.** Required fields can
  now come from a file/flag, and invalid merges fail at this boundary.
- **Direct decorated calls receive concrete selected defaults.** Code should never
  depend on receiving FieldInfo or TunableParamData as an argument value.
- **Namespace aliases no longer mutate their source class.** Use `Parent.Child.field`
  when you want the parent namespace; `ChildParams.field` keeps its own namespace.
- Static discovery uses resolved identities. Previously accidental matches between
  unrelated functions are removed; use tags for dynamic call paths.
- Internal parser destinations changed; use the public merge helper instead of
  inspecting `TX__...` attributes directly.
- Earlier README API names were inaccurate. Use `TunableParams`, `make_config_for_app`,
  `schema_for_app`, `make_config_for_entry` and `schema_for_entrypoint`. Schema helpers
  return two values. Historical tracing names are not exported aliases.
