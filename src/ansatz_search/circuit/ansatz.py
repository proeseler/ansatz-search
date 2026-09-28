from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Iterable

from .gates import GateName, parse_gate_name


@dataclass(frozen=True, slots=True)
class ParamRef:
    """Reference to a variational parameter slot shared across blocks.

    The gate's angle is θ[index] + offset. Transpiled circuits contain such
    offsets (e.g. rz(θ + π)); they do not change the gradient.
    """

    index: int
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Parameter indices must be non-negative.")
        object.__setattr__(self, "offset", float(self.offset))


@dataclass(frozen=True, slots=True)
class AnsatzBlock:
    """Backend-neutral description of one ansatz operation.

    `op` names the logical gate, `qubits` stores where it acts, and `params`
    holds one entry per angle of the gate: a ParamRef to a variational
    parameter, or a float for a fixed angle, e.g. ``AnsatzBlock("rx", (0,), (np.pi / 2,))``.
    """

    op: GateName | str
    qubits: tuple[int, ...]
    params: tuple[ParamRef | float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", parse_gate_name(self.op))
        object.__setattr__(self, "qubits", tuple(self.qubits))
        object.__setattr__(self, "params", tuple(_angle(p) for p in self.params))

        if not self.qubits:
            raise ValueError("Ansatz blocks must act on at least one qubit.")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("Ansatz blocks cannot contain duplicate qubit indices.")
        if any(qubit < 0 for qubit in self.qubits):
            raise ValueError("Qubit indices must be non-negative.")

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def num_params(self) -> int:
        """Number of angles, variational or fixed."""
        return len(self.params)

    @property
    def param_refs(self) -> tuple[ParamRef, ...]:
        """The variational parameters among the angles."""
        return tuple(p for p in self.params if isinstance(p, ParamRef))


def _angle(value) -> ParamRef | float:
    if isinstance(value, ParamRef):
        return value
    # An int could be meant as a parameter index: make the caller say which.
    if isinstance(value, Real) and not isinstance(value, Integral):
        return float(value)
    raise TypeError(f"An angle must be a ParamRef or a float (a fixed angle), got {value!r}.")


def _angle_to_json(angle: ParamRef | float) -> int | float | list:
    if not isinstance(angle, ParamRef):
        return angle
    return [angle.index, angle.offset] if angle.offset else angle.index


def _angle_from_json(value) -> ParamRef | float:
    if isinstance(value, list):
        return ParamRef(*value)
    return ParamRef(value) if isinstance(value, int) else float(value)


@dataclass(slots=True)
class AnsatzSpec:
    """Ordered backend-neutral ansatz representation.

    This is the output of the search layer and the input to backend compilers
    such as the Qiskit, NumPy or PennyLane compilers.
    """

    num_qubits: int
    blocks: list[AnsatzBlock] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError("An ansatz must contain at least one qubit.")
        self.blocks = list(self.blocks)
        self.validate()

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def num_params(self) -> int:
        if not self.blocks:
            return 0
        return max((param.index for block in self.blocks for param in block.param_refs), default=-1) + 1

    @property
    def depth(self) -> int:
        """Number of layers, each block occupying its qubits for one layer (as Qiskit's depth)."""
        layer = [0] * self.num_qubits
        for block in self.blocks:
            d = 1 + max(layer[q] for q in block.qubits)
            for q in block.qubits:
                layer[q] = d
        return max(layer, default=0)

    def to_dict(self) -> dict:
        """JSON-friendly form: {"num_qubits": n, "blocks": [[op, [qubits], [angles]], ...]}.

        An angle is an int for a parameter index, [index, offset] for a
        parameter plus an offset, and a float for a fixed angle; JSON keeps
        ``1`` and ``1.0`` apart.
        """
        return {
            "num_qubits": self.num_qubits,
            "blocks": [[b.op.value, list(b.qubits), [_angle_to_json(p) for p in b.params]] for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnsatzSpec":
        return cls(data["num_qubits"], [
            AnsatzBlock(op, tuple(qubits), tuple(_angle_from_json(p) for p in params))
            for op, qubits, params in data["blocks"]
        ])

    @classmethod
    def from_qiskit(cls, qc) -> "AnsatzSpec":
        """Convert a parameterized Qiskit QuantumCircuit; see backends.qiskit.compiler.spec_from_qiskit."""
        from ansatz_search.backends.qiskit.compiler import spec_from_qiskit

        return spec_from_qiskit(qc)

    def append(self, block: AnsatzBlock) -> None:
        self._validate_block(block)
        self.blocks.append(block)

    def extend(self, blocks: Iterable[AnsatzBlock]) -> None:
        for block in blocks:
            self.append(block)

    def validate(self) -> None:
        for block in self.blocks:
            self._validate_block(block)

    def _validate_block(self, block: AnsatzBlock) -> None:
        for qubit in block.qubits:
            if qubit >= self.num_qubits:
                raise ValueError(
                    f"Block {block.op.value} targets qubit {qubit}, "
                    f"but the ansatz only defines {self.num_qubits} qubits."
                )
