"""The 19 four-qubit circuits of Sim, Johnson & Aspuru-Guzik (2019).

"Expressibility and entangling capability of parameterized quantum circuits
for hybrid quantum-classical algorithms", Adv. Quantum Technol. 2, 1900070.
Ported from the benchmark bank of the previous framework; parameter ``p[i]``
there is ``ParamRef(i)`` here, and gates keep their (control, target) order.
"""

from __future__ import annotations

from typing import Callable

from ansatz_search.circuit.ansatz import AnsatzBlock, AnsatzSpec, ParamRef


def _rot(gate: str, qubit: int, k: int) -> AnsatzBlock:
    return AnsatzBlock(gate, (qubit,), (ParamRef(k),))


def _crot(gate: str, control: int, target: int, k: int) -> AnsatzBlock:
    return AnsatzBlock(gate, (control, target), (ParamRef(k),))


def _gate(gate: str, *qubits: int) -> AnsatzBlock:
    return AnsatzBlock(gate, qubits)


def _rx_rz_layer() -> list[AnsatzBlock]:
    """rx(p0..p3) on qubits 0-3, then rz(p4..p7) on qubits 0-3 (circuits 1-8)."""
    return [_rot("rx", q, q) for q in range(4)] + [_rot("rz", q, 4 + q) for q in range(4)]


def _rx_rz_pairs(start: int) -> list[AnsatzBlock]:
    """rx(p[start+2q]) then rz(p[start+2q+1]) on each qubit q."""
    return [b for q in range(4) for b in (_rot("rx", q, start + 2 * q), _rot("rz", q, start + 2 * q + 1))]


def circuit_1() -> AnsatzSpec:
    return AnsatzSpec(4, _rx_rz_layer())


def circuit_2() -> AnsatzSpec:
    return AnsatzSpec(4, _rx_rz_layer() + [_gate("cx", c, t) for c, t in [(3, 2), (2, 1), (1, 0)]])


def _circuit_3_4(gate: str) -> AnsatzSpec:
    ring = [(3, 2), (2, 1), (1, 0)]
    return AnsatzSpec(4, _rx_rz_layer() + [_crot(gate, c, t, 8 + j) for j, (c, t) in enumerate(ring)])


def circuit_3() -> AnsatzSpec:
    return _circuit_3_4("crz")


def circuit_4() -> AnsatzSpec:
    return _circuit_3_4("crx")


def _circuit_5_6(gate: str) -> AnsatzSpec:
    layers = [[(3, 2), (3, 1), (3, 0)], [(2, 3), (2, 1), (2, 0)], [(1, 3), (1, 2), (1, 0)], [(0, 3), (0, 2), (0, 1)]]
    pairs = [pair for layer in layers for pair in layer]
    blocks = _rx_rz_layer() + [_crot(gate, c, t, 8 + j) for j, (c, t) in enumerate(pairs)]
    return AnsatzSpec(4, blocks + _rx_rz_pairs(20))


def circuit_5() -> AnsatzSpec:
    return _circuit_5_6("crz")


def circuit_6() -> AnsatzSpec:
    return _circuit_5_6("crx")


def _circuit_7_8(gate: str) -> AnsatzSpec:
    blocks = _rx_rz_layer() + [_crot(gate, 1, 0, 8), _crot(gate, 3, 2, 9)]
    return AnsatzSpec(4, blocks + _rx_rz_pairs(10) + [_crot(gate, 2, 1, 18)])


def circuit_7() -> AnsatzSpec:
    return _circuit_7_8("crz")


def circuit_8() -> AnsatzSpec:
    return _circuit_7_8("crx")


def circuit_9() -> AnsatzSpec:
    blocks = [_gate("h", q) for q in range(4)]
    blocks += [_gate("cz", c, t) for c, t in [(2, 3), (1, 2), (0, 1)]]
    return AnsatzSpec(4, blocks + [_rot("rx", q, q) for q in range(4)])


def circuit_10() -> AnsatzSpec:
    blocks = [_rot("ry", q, q) for q in range(4)]
    blocks += [_gate("cz", c, t) for c, t in [(2, 3), (1, 2), (0, 1), (0, 3)]]
    return AnsatzSpec(4, blocks + [_rot("ry", q, 4 + q) for q in range(4)])


def _circuit_11_12(gate: str) -> AnsatzSpec:
    blocks = [b for q in range(4) for b in (_rot("ry", q, 2 * q), _rot("rz", q, 2 * q + 1))]
    blocks += [_gate(gate, 1, 0), _gate(gate, 3, 2)]
    blocks += [b for i, w in enumerate([1, 2], start=4) for b in (_rot("ry", w, 2 * i), _rot("rz", w, 2 * i + 1))]
    return AnsatzSpec(4, blocks + [_gate(gate, 2, 1)])


def circuit_11() -> AnsatzSpec:
    return _circuit_11_12("cx")


def circuit_12() -> AnsatzSpec:
    return _circuit_11_12("cz")


def _circuit_13_14(gate: str) -> AnsatzSpec:
    blocks = [_rot("ry", q, q) for q in range(4)]
    blocks += [_crot(gate, c, t, 4 + j) for j, (c, t) in enumerate([(3, 0), (2, 3), (1, 2), (0, 1)])]
    blocks += [_rot("ry", q, 8 + q) for q in range(4)]
    blocks += [_crot(gate, c, t, 12 + j) for j, (c, t) in enumerate([(3, 2), (0, 3), (1, 0), (2, 1)])]
    return AnsatzSpec(4, blocks)


def circuit_13() -> AnsatzSpec:
    return _circuit_13_14("crz")


def circuit_14() -> AnsatzSpec:
    return _circuit_13_14("crx")


def circuit_15() -> AnsatzSpec:
    blocks = [_rot("ry", q, q) for q in range(4)]
    blocks += [_gate("cx", c, t) for c, t in [(3, 0), (2, 3), (1, 2), (0, 1)]]
    blocks += [_rot("ry", q, 4 + q) for q in range(4)]
    blocks += [_gate("cx", c, t) for c, t in [(3, 2), (0, 3), (1, 0), (2, 1)]]
    return AnsatzSpec(4, blocks)


def _circuit_16_17(gate: str) -> AnsatzSpec:
    ring = [(1, 0), (3, 2), (2, 1)]
    return AnsatzSpec(4, _rx_rz_pairs(0) + [_crot(gate, c, t, 8 + j) for j, (c, t) in enumerate(ring)])


def circuit_16() -> AnsatzSpec:
    return _circuit_16_17("crz")


def circuit_17() -> AnsatzSpec:
    return _circuit_16_17("crx")


def _circuit_18_19(gate: str) -> AnsatzSpec:
    ring = [(3, 0), (2, 3), (1, 2), (0, 1)]
    return AnsatzSpec(4, _rx_rz_pairs(0) + [_crot(gate, c, t, 8 + j) for j, (c, t) in enumerate(ring)])


def circuit_18() -> AnsatzSpec:
    return _circuit_18_19("crz")


def circuit_19() -> AnsatzSpec:
    return _circuit_18_19("crx")


SIM2019: dict[int, Callable[[], AnsatzSpec]] = {
    i: globals()[f"circuit_{i}"] for i in range(1, 20)
}


def sim2019_circuit(index: int) -> AnsatzSpec:
    """Circuit `index` (1-19) of Sim et al. (2019)."""
    if index not in SIM2019:
        raise ValueError(f"Sim et al. (2019) circuits are numbered 1-19, got {index}.")
    return SIM2019[index]()
