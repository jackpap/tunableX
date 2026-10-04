# Design, review findings and limitations

## Architecture

1. Importing a decorated module registers selected fields in a process-local tree.
2. App tags or source-level call discovery select registrations.
3. The tree becomes nested Pydantic models. Fields and namespaces cannot collide.
4. Configuration files and explicit flags are merged, then defaults and validation apply.
5. `use_config` activates a context-local model. A wrapper binds explicit arguments
   first and fills missing selected arguments from that model or declared defaults.

Functions remain callable directly. Pydantic is the validation boundary for external
settings; the wrapper is an injection mechanism, not a validator of every Python call.
App models should be constructed once during startup and reused, rather than rebuilt
inside a hot loop. No registry/model cache is introduced because imports can add
registrations later and custom defaults may be mutable.

## Repository review (October 2026)

The review covered every source module, test, example, packaging definition and
workflow, plus the open multiprocessing work in PR #10. The starting test suite
reported 27 passing tests and two skipped AST cases. Passing tests concealed some
incorrect fixtures and unconditional assertions.

| Finding | Resolution |
| --- | --- |
| Positional arguments received duplicate injected keywords | Bind the signature first; explicit arguments win, including positional-only values |
| Wrappers serialized nested models into dicts | Read model attributes directly and retain Python values |
| Direct calls leaked Field/reference metadata | Resolve selected defaults/factories, and reject missing required values |
| Static methods crashed without positional arguments | Preserve descriptors in either decorator order; support class methods |
| Coroutines captured config before awaiting | Resolve inside an async wrapper with task-local contexts |
| Required fields prevented CLI composition/default export | Merge input before validation; generate partial defaults templates |
| Collection/Literal/nullable CLI conversions were incomplete | Typed token conversion, JSON structured values, explicit null, empty collections |
| Double underscores were lost or collided in CLI destinations | Preserve canonical dotted paths internally |
| AST discovery matched textual names, missed constructors and failed on indented source | Resolve function identities through live globals/closures and passive attribute lookup; dedent source |
| Unknown settings silently disappeared | Generated models forbid extra fields, including nested keys |
| Python 3.14 deferred annotations were missed/evaluated too early | Read string-form annotations and signatures, then resolve only selected fields |
| Optional YAML was imported unconditionally | Lazy YAML imports; a dependency-light base installation |
| TOML fallback was missing on Python 3.10 | Conditional tomli dependency |
| Loader called SystemExit | Library loaders raise validation/parse exceptions |
| Shared parameter aliases mutated their original class namespace | Immutable namespace views |
| Descriptions were guessed by substring search | Match the AST assignment and immediately following string |
| Equivalent Field objects collided by object identity | Compare Field metadata and reconstruct it for both composition paths |
| Field/namespace collisions silently replaced data | Reject conflicting paths during registration |
| Multiprocessing type hints failed in execution namespaces | Incorporate and extend PR #10's per-field resolution and caching |
| CLI analyze unpacked the wrong number of return values | Correct two-value API, add installed and `python -m` entrypoints |
| README described nonexistent exports/inheritance behavior | Replace with runnable examples and an explicit API reference |
| Test fixtures used schemas as configs and misspelled keys | Correct fixtures, restore skipped cases and remove unconditional assertions |
| No PR checks; linting used unsafe fixes and unrelated rules | Small Python compatibility matrix and nonmutating lint/format checks |
| Typed package marker and source-distribution tests were missing | Include py.typed, tests, examples and docs in built distributions |

The follow-up retains the contributor commits from PR #10 in its history. It also
resolves concrete defaults without an active config, instead of rejecting every
centralized-reference call as PR #10 originally proposed. These improvements ship
in version 0.2.0; see the changelog for behavior changes and migration guidance.

## Remaining boundaries and practical alternatives

### Static discovery is conservative

The analyzer follows direct names, imported aliases, module attributes, closures,
simple local assignments, constructors and statically resolved methods. It visits
both branches, so it can include calls that a particular execution would not take.

It does not interpret arbitrary Python. Callback dictionaries, dependency injection,
function factories, dynamic `getattr`, runtime imports and generated functions may
be unresolved. Complex local control/data flow and locally defined classes are not
fully modeled. Recursive call graphs are bounded by a visited-function set. Methods
resolved through a known module-level class can expose self/cls calls, but runtime
polymorphism cannot be inferred reliably.

**Use app tags for these cases.** They are also the reliable choice for interactive
or packaged environments without source. An unavailable entrypoint source now raises
an actionable error; unavailable source in a discovered dependency cannot be traversed
further, though its own registered fields are retained. No claim of a complete call
graph is made.

### Registration is process-local

Import declarations before composition. Untagged fields join every app-tag config,
which is convenient for shared utilities but can be too broad in large applications;
tag them explicitly to narrow selection. Registry entries persist for the process
lifetime. Isolated registries, hot reload/unregistration and persistent model caches
are not public features. Dynamically generated model classes should not be pickled
between processes; transfer the serialized values instead.

### Configuration semantics

Unknown fields and invalid defaults now fail early. Existing files that relied on
silently ignored keys must be corrected. Functions keep normal Python semantics for
explicit values; a raw dict passed to `use_config` deliberately bypasses validation.
If multiple functions share one path they share one definition; distinct settings
need distinct namespaces.

Typed user-defined Pydantic models retain their own configuration and validators.
An explicitly supplied nested model value is validated under that model's defaults;
it is not a patch to an arbitrary preconstructed model instance used as a default.
Generated namespace sections support partial file/CLI overrides.

### Side effects and serialization

Discovery does not invoke entrypoint bodies or property descriptors. Imports and
annotation evaluation still execute Python, and defaults export can invoke ordinary
default factories. Only use trusted application code. JSON Schema cannot represent
every arbitrary Python object or callable. Choose JSON-compatible parameter types,
custom Pydantic serializers/schema hooks, or keep nonconfiguration objects outside
the tunable API.

### Scope deliberately kept out of the core

No environment-variable layer, remote configuration service, file watching, secrets
manager or TOML writer is added. A program can compose these at startup and pass the
result to `Config.model_validate`. The package remains a small function-first
configuration library rather than an application framework.
