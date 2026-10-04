"""Execute the two self-contained README examples against the installed package."""

import re
import subprocess
import sys


def test_readme_examples(repo_root, tmp_path):
    blocks = re.findall(r"```python\n(.*?)\n```", (repo_root / "README.md").read_text(), re.DOTALL)
    script = tmp_path / "train.py"
    script.write_text(blocks[0])
    result = subprocess.run(
        [sys.executable, str(script), "--train.epochs", "25", "--train.verbose"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "25 0.001 True"
    shared = tmp_path / "shared.py"
    shared.write_text(blocks[1])
    subprocess.run([sys.executable, str(shared)], capture_output=True, text=True, check=True)


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
