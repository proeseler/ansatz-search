from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .gates import GateName, parse_gate_name


@dataclass(frozen=True, slots=True)
class ParamRef:
    """Reference to a variational parameter slot shared across blocks."""

    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Parameter indices must be non-negative.")


@dataclass(frozen=True, slots=True)
class AnsatzBlock:
    """Backend-neutral description of one ansatz operation.

    `op` names the logical gate, `qubits` stores where it acts, and `params`
    stores explicit references to variational parameters.
    """

    op: GateName | str
    qubits: tuple[int, ...]
    params: tuple[ParamRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", parse_gate_name(self.op))
        object.__setattr__(self, "qubits", tuple(self.qubits))
        object.__setattr__(self, "params", tuple(self.params))

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
        return len(self.params)


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
        return max((param.index for block in self.blocks for param in block.params), default=-1) + 1

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
        """JSON-friendly form: {"num_qubits": n, "blocks": [[op, [qubits], [param indices]], ...]}."""
        return {
            "num_qubits": self.num_qubits,
            "blocks": [[b.op.value, list(b.qubits), [p.index for p in b.params]] for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnsatzSpec":
        return cls(data["num_qubits"], [
            AnsatzBlock(op, tuple(qubits), tuple(ParamRef(i) for i in params))
            for op, qubits, params in data["blocks"]
        ])

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
