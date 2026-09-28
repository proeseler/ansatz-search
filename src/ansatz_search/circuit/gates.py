from __future__ import annotations

from enum import Enum
from itertools import permutations
from typing import NamedTuple


class GateName(str, Enum):
    X = "x"
    Y = "y"
    Z = "z"
    H = "h"
    S = "s"
    T = "t"
    SDG = "sdg"
    TDG = "tdg"
    SX = "sx"
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    R1 = "r1"
    CX = "cx"
    CY = "cy"
    CZ = "cz"
    CH = "ch"
    CRX = "crx"
    CRY = "cry"
    CRZ = "crz"
    CR1 = "cr1"
    SWAP = "swap"
    ECR = "ecr"
    RXX = "rxx"
    RYY = "ryy"
    RZZ = "rzz"


_GATE_ALIASES = {
    "phase": GateName.R1,
    "p": GateName.R1,
    "cphase": GateName.CR1,
    "cp": GateName.CR1,
}


def parse_gate_name(value: str | GateName) -> GateName:
    """Normalize external gate names to the internal enum."""

    if isinstance(value, GateName):
        return value
    normalized = value.lower().strip()
    # Not `.get(normalized, GateName(normalized))`: the default is evaluated
    # eagerly and would raise for every alias before the lookup happens.
    if normalized in _GATE_ALIASES:
        return _GATE_ALIASES[normalized]
    return GateName(normalized)


# (num_qubits, num_params) of each logical gate. The shape is a property of the
# gate itself, not of a backend; backends only decide which gates they support.
GATE_SHAPES: dict[GateName, tuple[int, int]] = {
    **{gate: (1, 0) for gate in (
        GateName.X, GateName.Y, GateName.Z, GateName.H, GateName.S, GateName.T,
        GateName.SDG, GateName.TDG, GateName.SX,
    )},
    **{gate: (1, 1) for gate in (
        GateName.RX, GateName.RY, GateName.RZ, GateName.R1,
    )},
    **{gate: (2, 0) for gate in (
        GateName.CX, GateName.CY, GateName.CZ, GateName.CH, GateName.SWAP, GateName.ECR,
    )},
    **{gate: (2, 1) for gate in (
        GateName.CRX, GateName.CRY, GateName.CRZ, GateName.CR1,
        GateName.RXX, GateName.RYY, GateName.RZZ,
    )},
}


class GateSpec(NamedTuple):
    """A logical gate together with the shape the search layer needs to place it."""

    name: GateName
    num_qubits: int
    num_params: int


def gate_spec(value: str | GateName) -> GateSpec:
    """Look up the canonical shape of a gate given by name or alias."""

    try:
        gate = parse_gate_name(value)
    except ValueError:
        known = sorted(g.value for g in GateName)
        raise ValueError(f"Unknown gate {value!r}. Known gates: {known}.") from None
    return GateSpec(gate, *GATE_SHAPES[gate])


def valid_placements(num_qubits: int, gate_qubits: int, topology=None) -> list[tuple[int, ...]]:
    """Return all valid qubit placements for a gate on `gate_qubits` qubits."""
    if topology is not None and gate_qubits == 2:
        return [(q0, q1) for q0, neighbors in topology.items() for q1 in neighbors]

    return list(permutations(range(num_qubits), gate_qubits))


def connectivity_edges(qubits: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Return the qubit-pair edges contributed by a gate placed on `qubits`."""
    if len(qubits) < 2:
        return ()
    return ((qubits[0], qubits[1]),)

