# Contributing

## Local setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
```

The suite runs in seconds and includes the examples as subprocesses. Add a focused
regression for behavioral changes; avoid repeating the same expensive integration
flow when a direct test proves the contract. New tests that register temporary
fields should isolate `REGISTRY.entry_tree` with a fixture. Never change the live
registry as a public application pattern.

Keep Python 3.10 syntax and runtime compatibility. Postponed annotations do not make
new Python syntax work on old interpreters. Optional integrations must stay optional:
test the built wheel in an environment without PyYAML/jsonargparse as well.

## Design and compatibility

Read [the API reference](docs/api.md) and [design notes](docs/design-and-limitations.md).
Preserve explicit argument precedence and file/CLI merge semantics. Do not execute
application functions to improve static discovery; document unresolvable cases and
use app tags. Keep examples and migration notes aligned with the public exports.

Linting uses stable correctness/import/compatibility rules, targets Python 3.10 and
does not enable unsafe automatic fixes. `Field(...)` defaults are intentional metadata.
Run `ruff check --fix` and `ruff format` explicitly when you want to update files.

## CI

PRs and main-branch code changes run two short compatibility jobs: the minimum
Python/dependency versions and the current stable Python with optional integrations.
The current job also checks formatting and builds/checks distributions. Concurrent
outdated runs are cancelled; documentation-only changes do not trigger the code suite, except the README whose
executable examples are regression-tested.
These same commands work locally if GitHub Actions is unavailable or out of budget.

## Release checklist

1. Review the changelog and select an appropriate version in `pyproject.toml`.
2. Run the test/lint checks on the release commit, then:
   ```bash
   python -m build
   python -m twine check dist/*
   ```
3. Smoke-test the wheel in a clean environment and confirm the installed `tunablex`
   command works without optional dependencies.
4. Publish a GitHub release with a tag matching the package version (`vX.Y.Z`). The
   existing trusted-publishing workflow builds the distributions, checks the tag and
   metadata, and publishes through the protected `pypi` environment.

Do not put credentials in repository files. CI's `contents` permission is read-only;
only the PyPI publishing job receives the OIDC permission it requires.
