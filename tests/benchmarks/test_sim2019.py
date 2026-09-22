"""Sim et al. (2019) benchmark circuits: expressibility and entanglement.

Three kinds of checks against tests/data/sim2019_reference.json:
  - regression: with the recorded seed, values must not change at all;
  - legacy: the previous framework's values must agree within sampling noise;
  - ranking: the circuits must be ordered as in the legacy values.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ansatz_search.backends.numpy import NumpyCompiler, NumpyStateProvider
from ansatz_search.benchmarks import SIM2019
from ansatz_search.circuit.ansatz import AnsatzBlock, AnsatzSpec, ParamRef
from ansatz_search.metrics.entanglement import Entanglement
from ansatz_search.metrics.expressibility import Expressibility
from ansatz_search.utils.statevector_sampler import CustomSampler

REFERENCE = json.loads((Path(__file__).parents[1] / "data" / "sim2019_reference.json").read_text())
SETTINGS = REFERENCE["settings"]
CIRCUITS = sorted(SIM2019)
METRICS = {"expressibility": Expressibility, "entanglement": Entanglement}


def compute(kind, spec, samples=SETTINGS["num_of_param_samples"], provider=None, compiler=None, **kwargs):
    metric = METRICS[kind](
        num_qubits=spec.num_qubits,
        state_provider=provider or NumpyStateProvider(),
        num_of_param_samples=samples,
        seed=SETTINGS["seed"],
        **kwargs,
    )
    return metric.value((compiler or NumpyCompiler())(spec), spec)


def reference(kind, i):
    return REFERENCE["circuits"][str(i)][kind]


@pytest.fixture(scope="module")
def values():
    """Both metrics for all 19 circuits, computed once for this module."""
    return {(kind, i): compute(kind, SIM2019[i]()) for kind in METRICS for i in CIRCUITS}


@pytest.mark.parametrize("i", CIRCUITS)
def test_parameter_counts(i):
    assert SIM2019[i]().num_params == REFERENCE["circuits"][str(i)]["num_params"]


@pytest.mark.parametrize("kind", METRICS)
@pytest.mark.parametrize("i", CIRCUITS)
def test_regression_value_is_unchanged(values, kind, i):
    assert values[kind, i] == pytest.approx(reference(kind, i)["value"], abs=1e-9)


@pytest.mark.parametrize("kind", METRICS)
@pytest.mark.parametrize("i", CIRCUITS)
def test_consistent_with_legacy_values(values, kind, i):
    ref = reference(kind, i)
    # Two independent runs differ with standard deviation sqrt(2) * sigma; allow 4 of those.
    # sigma is 0 where the metric is deterministic (entanglement of circuits 1 and 9).
    tolerance = max(4 * math.sqrt(2) * ref["sigma"], 1e-6)
    assert abs(values[kind, i] - ref["legacy"]) <= tolerance


@pytest.mark.parametrize("kind", METRICS)
def test_circuit_ranking_matches_legacy(values, kind):
    scipy_stats = pytest.importorskip("scipy.stats")
    ours = [values[kind, i] for i in CIRCUITS]
    legacy = [reference(kind, i)["legacy"] for i in CIRCUITS]
    assert scipy_stats.spearmanr(ours, legacy).statistic >= 0.95


@pytest.mark.parametrize("kind", METRICS)
def test_backends_agree_on_every_circuit(kind):
    pytest.importorskip("pennylane")
    from ansatz_search.backends.pennylane import PennyLaneCompiler, PennyLaneStateProvider

    for i in CIRCUITS:
        spec = SIM2019[i]()
        numpy_value = compute(kind, spec, samples=2000)
        pl_value = compute(kind, spec, samples=2000, provider=PennyLaneStateProvider(), compiler=PennyLaneCompiler())
        assert pl_value == pytest.approx(numpy_value, abs=1e-10), f"circuit {i}"


@pytest.mark.parametrize("kind", METRICS)
@pytest.mark.parametrize("i", [1, 9, 14])
def test_repeating_the_same_initial_state_changes_nothing(values, kind, i):
    zero = np.eye(16)[0]
    repeated = compute(kind, SIM2019[i](), state_sampler=CustomSampler(zero), num_of_state_samples=3)
    assert repeated == pytest.approx(values[kind, i], abs=1e-12)


# Single-circuit checks carried over from the previous framework.

def _rot(gate, k, qubit=0):
    return AnsatzBlock(gate, (qubit,), (ParamRef(k),))


@pytest.mark.parametrize("name, blocks, expected", [
    ("rz_only", [_rot("rz", 0)], 4.3),                           # never leaves |0>: maximal KL = ln(75)
    ("plus_rz", [AnsatzBlock("h", (0,)), _rot("rz", 0)], 0.22),  # covers the equator only
    ("u3_full", [_rot("rz", 2), _rot("ry", 0), _rot("rz", 1)], 0.007),  # u(theta, phi, lambda): near Haar
])
def test_single_qubit_expressibility(name, blocks, expected):
    assert compute("expressibility", AnsatzSpec(1, blocks)) == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("name, blocks, expected", [
    ("bell", [_rot("rz", 0), AnsatzBlock("h", (0,)), AnsatzBlock("cx", (0, 1))], 1.0),  # rz: phase on |0> only
    ("separable", [_rot("rz", 0)], 0.0),
])
def test_two_qubit_entanglement(name, blocks, expected):
    assert compute("entanglement", AnsatzSpec(2, blocks)) == pytest.approx(expected, abs=1e-12)
