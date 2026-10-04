"""Execute the README examples against the installed package."""

import re
import subprocess
import sys


def readme_example(repo_root, name):
    readme = (repo_root / "README.md").read_text()
    pattern = rf"<!-- example: {re.escape(name)} -->\s*```python\n(.*?)\n```"
    match = re.search(pattern, readme, re.DOTALL)
    assert match is not None, f"Missing README example: {name}"
    return match.group(1)


def test_readme_examples(repo_root, tmp_path):
    script = tmp_path / "train.py"
    script.write_text(readme_example(repo_root, "train"))
    result = subprocess.run(
        [sys.executable, str(script), "--train.epochs", "25", "--train.verbose"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "25 0.001 True"
    shared = tmp_path / "shared.py"
    shared.write_text(readme_example(repo_root, "shared"))
    subprocess.run([sys.executable, str(shared)], capture_output=True, text=True, check=True)


def test_readme_workflow_comparison(repo_root, tmp_path):
    for name in ("workflow", "pydantic-workflow"):
        script = tmp_path / f"{name}.py"
        script.write_text(readme_example(repo_root, name))
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True)
        assert result.stdout.splitlines() == ["24", "[1, 1]"]

    # Check the public outputs of the new declaration for both advertised apps,
    # without importing temporary registrations into the test runner's registry.
    proof = tmp_path / "proof.py"
    proof.write_text("""
from argparse import ArgumentParser
from pydantic import ValidationError
from tunablex import add_flags_by_app, build_cfg_from_file_and_args, schema_for_app, use_config
from workflow import single, study

for app, run, expected in (("single", single, 1), ("study", study, [1, 1])):
    schema, defaults = schema_for_app(app)
    assert defaults["solver"]["relaxation"] == 0.5
    solver_ref = schema["properties"]["solver"]["$ref"].split("/")[-1]
    field = schema["$defs"][solver_ref]["properties"]["relaxation"]
    assert field["exclusiveMinimum"] == 0 and field["maximum"] == 1
    parser = ArgumentParser()
    Config = add_flags_by_app(parser, app)
    assert "--solver.relaxation" in parser.format_help()
    args = parser.parse_args(["--solver.relaxation", "1"])
    cfg = Config.model_validate(build_cfg_from_file_and_args(Config, args))
    with use_config(cfg):
        assert run() == expected
    try:
        Config.model_validate({"solver": {"relaxation": 1.5}})
    except ValidationError:
        pass
    else:
        raise AssertionError("An out-of-range relaxation value was accepted")
assert single() == 24  # Leaving the context restores declared defaults.
""")
    subprocess.run([sys.executable, str(proof)], capture_output=True, text=True, check=True)


def test_native_deferred_annotations_do_not_evaluate_unselected_names(tmp_path):
    if sys.version_info < (3, 14):
        return  # Deferred annotations without __future__ were introduced in 3.14.
    source = """
from tunablex import tunable, TunableParams
from pydantic import Field

class Params(TunableParams):
    value: int = Field(3)
    unrelated: MissingName = Field(None)

@tunable("value")
def run(value: int = 3, other: MissingName = None):
    return value

assert run() == 3
assert Params.value.typ is int
"""
    script = tmp_path / "annotations.py"
    script.write_text(source)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
