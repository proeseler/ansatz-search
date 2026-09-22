from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ansatz_search.backends.base import Compiler
from ansatz_search.circuit.ansatz import AnsatzSpec
from ansatz_search.circuit.gates import GateName, GateSpec
from ansatz_search.cost_functions.base import CostFunction


class SearchAlgorithm(ABC):
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    @abstractmethod
    def run(self, problem: SearchProblem, compiler: Compiler, gate_specs: Sequence[GateSpec]):
        """Search `problem` using gates already resolved by `compiler`."""


class AnsatzBuilder(ABC):
    """Strategy for mapping search choices to an ansatz specification."""

    @abstractmethod
    def build(self, trial, problem: SearchProblem, gate_specs: Sequence[GateSpec]) -> AnsatzSpec:
        ...


@dataclass(slots=True)
class SearchProblem:
    cost_fn: CostFunction
    num_qubits: int
    max_param: int
    max_depth: int
    max_gates: int
    allowed_gates: Sequence[str | GateName]  # names; resolved by the backend compiler
    min_params: int = 0
    prev_params: bool = True
    topology: Optional[dict] = None


@dataclass(slots=True)
class SearchResult:
    """The best circuit of a search, and what is needed to evaluate the search later.

    `best_value` is the objective of one noisy evaluation of the best trial, so it
    is optimistic; `ansatz_search.analysis.evaluate` re-evaluates found circuits on
    fresh seeds. `run_dir` holds the run's config.yaml, `study_dir` its stored
    trials (None if nothing was persisted, e.g. with in-memory storage).
    """

    best_ansatz: AnsatzSpec
    best_value: float
    best_metrics: Any
    best_circuit: Any = None
    problem: Optional[SearchProblem] = None
    compiler: Optional[Compiler] = None
    builder: Optional[AnsatzBuilder] = None
    run_dir: Optional[str] = None
    study_dir: Optional[str] = None
