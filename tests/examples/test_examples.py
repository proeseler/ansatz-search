"""The example scripts run end to end (small settings, temporary output folder)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("optuna")

ROOT = Path(__file__).resolve().parents[2]


def _run(script, args, out):
    env = {**os.environ, "MPLBACKEND": "Agg",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))}
    proc = subprocess.run([sys.executable, str(ROOT / "examples" / script), *args, "--out", str(out)],
                          env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    for name in ("eval.json", "eval.png"):
        assert (out / name).exists(), name


@pytest.mark.parametrize("script", ["quickstart.py", "custom_metric.py"])
def test_search_examples_run(script, tmp_path):
    _run(script, ["--trials", "8"], tmp_path / "run")
    assert (tmp_path / "run" / "config.yaml").exists()


def test_hardware_example_runs_on_a_simulated_ibm_device(tmp_path):
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_ibm_runtime")
    _run("hardware_expressibility.py", ["--pairs", "10", "--shots", "200", "--seeds", "1"], tmp_path / "run")
