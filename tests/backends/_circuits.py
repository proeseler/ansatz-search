"""Shared test circuits for backend comparisons."""

import numpy as np

from ansatz_search.circuit.ansatz import AnsatzBlock, AnsatzSpec, ParamRef


def mixed_circuit() -> AnsatzSpec:
    """Controlled rotations, reused parameters and every kind of fixed gate."""
    return AnsatzSpec(3, [
        AnsatzBlock("h", (0,)), AnsatzBlock("crx", (0, 1), (ParamRef(0),)),
        AnsatzBlock("ry", (2,), (ParamRef(1),)), AnsatzBlock("cz", (1, 2)),
        AnsatzBlock("rz", (1,), (ParamRef(0),)), AnsatzBlock("cr1", (2, 0), (ParamRef(2),)),
        AnsatzBlock("swap", (0, 2)), AnsatzBlock("rx", (0,), (ParamRef(1),)),
        AnsatzBlock("ch", (1, 0)), AnsatzBlock("cy", (0, 2)), AnsatzBlock("t", (1,)),
        AnsatzBlock("cry", (2, 1), (ParamRef(3),)), AnsatzBlock("crz", (1, 2), (ParamRef(2),)),
        AnsatzBlock("r1", (2,), (ParamRef(3),)), AnsatzBlock("x", (1,)), AnsatzBlock("cx", (2, 0)),
    ])


def rotation_circuit() -> AnsatzSpec:
    """Only rx/ry/rz/r1, each parameter used once: parameter shift is exact."""
    return AnsatzSpec(3, [
        AnsatzBlock("ry", (0,), (ParamRef(0),)), AnsatzBlock("rx", (1,), (ParamRef(1),)),
        AnsatzBlock("cx", (0, 1)), AnsatzBlock("rz", (2,), (ParamRef(2),)),
        AnsatzBlock("h", (2,)), AnsatzBlock("cx", (1, 2)), AnsatzBlock("r1", (0,), (ParamRef(3),)),
        AnsatzBlock("ry", (2,), (ParamRef(4),)),
    ])


def random_state(n, seed=1):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return state / np.linalg.norm(state)


OBSERVABLES = [{"ZZI": 1.0, "XIY": -0.3}, "IYX"]
