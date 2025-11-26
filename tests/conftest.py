import os
import pathlib
import subprocess
import sys

import pytest


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    # Repo root = parent of tests/
    return pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def run_example(repo_root):
    def _run(script_relpath, args):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        cmd = [sys.executable, str(repo_root / script_relpath), *args]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return proc.returncode, proc.stdout, proc.stderr

    return _run
