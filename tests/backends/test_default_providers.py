"""Without an explicit provider, metrics use the provider of the backend that compiled the circuit."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ansatz_search import (BayesianOptimizationSearch, HierarchicalCostFunction, IncrementalAnsatzBuilder,
                           SearchProblem, ansatz_search)
from ansatz_search.backends.base import StateProvider, default_provider, register_providers
from ansatz_search.backends.numpy import NumpyCompiler, NumpyGradientProvider, NumpyStateProvider
from ansatz_search.benchmarks import SIM2019
from ansatz_search.metrics import Entanglement, Expressibility, GradientVariance

SPEC = SIM2019[2]()  # 4 qubits, 8 parameters


def _value(metric, compiler=None):
    compiler = NumpyCompiler() if compiler is None else compiler
    return metric.value(compiler(SPEC), SPEC)


def test_numpy_circuits_get_numpy_providers():
    program = NumpyCompiler()(SPEC)
    assert isinstance(default_provider(program, "gradient"), NumpyGradientProvider)
    assert isinstance(default_provider(program, "state"), NumpyStateProvider)


def test_metrics_without_a_provider_match_explicit_numpy_providers():
    kw = dict(num_of_param_samples=40, seed=3)
    assert (_value(GradientVariance(observables="ZZZZ", **kw))
            == _value(GradientVariance(observables="ZZZZ", gradient_provider=NumpyGradientProvider(), **kw)))
    assert (_value(Expressibility(num_qubits=4, **kw))
            == _value(Expressibility(num_qubits=4, state_provider=NumpyStateProvider(), **kw)))
    assert (_value(Entanglement(num_qubits=4, **kw))
            == _value(Entanglement(num_qubits=4, state_provider=NumpyStateProvider(), **kw)))


def test_pennylane_circuits_get_pennylane_providers():
    pytest.importorskip("pennylane")
    from ansatz_search.backends.pennylane import PennyLaneCompiler, PennyLaneGradientProvider, PennyLaneStateProvider

    program = PennyLaneCompiler()(SPEC)
    assert isinstance(default_provider(program, "gradient"), PennyLaneGradientProvider)
    assert isinstance(default_provider(program, "state"), PennyLaneStateProvider)
    kw = dict(num_of_param_samples=10, seed=3)
    np.testing.assert_allclose(_value(Entanglement(num_qubits=4, **kw), PennyLaneCompiler()),
                               _value(Entanglement(num_qubits=4, **kw)), atol=1e-10)


def test_a_qiskit_circuit_gets_qiskit_providers_without_importing_the_backend():
    # A fresh interpreter: in this one, other tests have already imported the Qiskit backend.
    code = ("from qiskit import QuantumCircuit; from qiskit.circuit import Parameter\n"
            "from ansatz_search.metrics import Entanglement\n"
            "qc = QuantumCircuit(2); qc.ry(Parameter('a'), 0); qc.cx(0, 1)\n"
            "print(Entanglement(num_qubits=2, num_of_param_samples=4, seed=0).value(qc) > 0)")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "True"


def test_circuits_of_unknown_backends_need_an_explicit_provider():
    with pytest.raises(TypeError, match="gradient_provider="):
        default_provider(object(), "gradient")


def test_a_new_backend_registers_its_providers():
    class Program:
        pass

    class States(StateProvider):
        def states(self, circuit, param_samples, initial_state):
            return np.zeros((len(param_samples), 2))

    register_providers(Program, state=States)
    assert isinstance(default_provider(Program(), "state"), States)
    with pytest.raises(TypeError, match="gradient"):
        default_provider(Program(), "gradient")


def test_ansatz_search_defaults_to_numpy_and_stores_the_study_in_fdata(tmp_path):
    problem = SearchProblem(
        cost_fn=HierarchicalCostFunction({0: [GradientVariance(observables="ZZZ", num_of_param_samples=8, seed=0)]}),
        num_qubits=3, max_param=4, max_depth=6, max_gates=6, allowed_gates=["ry", "cx"])
    algorithm = BayesianOptimizationSearch(IncrementalAnsatzBuilder(), n_trials=3, study_name="s", seed=0)
    with pytest.raises(TypeError, match="fdata"):
        ansatz_search(problem, algorithm)
    result = ansatz_search(problem, algorithm, fdata=tmp_path)
    assert isinstance(result.compiler, NumpyCompiler)
    assert (tmp_path / "s.log").exists() and result.study_dir == str(tmp_path)
