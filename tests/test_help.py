from __future__ import annotations

import pytest

# Tests that jsonargparse-generated help output includes default values from Pydantic Fields
# and that the formatting we expect ("(default: ...)") is present. These complement
# existing override tests to ensure we did not regress CLI behavior.


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_jsonargparse_app_help_includes_defaults(run_example):
    code, out, err = run_example(
        "examples/jsonargparse_app/train_jsonarg_app.py",
        ["--help"],
    )
    assert code == 0, err
    # Look for representative defaults from each namespace
    # Use minimal substrings to be robust against line wrapping.
    assert "(default: true)" in out  # model.preprocess.dropna
    assert "Normalization strategy" in out and "(default: 'zscore')" in out
    assert "Number of hidden units" in out and "(default: 128)" in out
    assert "Dropout probability" in out and "(default: 0.2)" in out
    assert "Number of training epochs" in out and "(default: 10)" in out
    assert "Training batch size" in out and "(default: 32)" in out
    assert "Optimizer choice" in out and "(default: 'adam')" in out


@pytest.mark.skipif(pytest.importorskip("jsonargparse") is None, reason="jsonargparse not installed")
def test_jsonargparse_trace_help_includes_defaults(run_example):
    code, out, err = run_example(
        "examples/jsonargparse_trace/train_jsonarg_trace.py",
        ["--help"],
    )
    assert code == 0, err
    # Same defaults should appear for traced entry (train_main)
    assert "(default: true)" in out  # model.preprocess.dropna
    assert "(default: 'zscore')" in out  # model.preprocess.normalize
    assert "(default: 128)" in out  # model.hidden_units
    assert "(default: 0.2)" in out  # model.dropout
    assert "(default: 10)" in out  # train.epochs
    assert "(default: 32)" in out  # train.batch_size
    assert "(default: 'adam')" in out  # train.optimizer
