from __future__ import annotations

from typing import Sequence

import optuna

from ansatz_search.circuit.ansatz import AnsatzBlock, AnsatzSpec, ParamRef
from ansatz_search.circuit.gates import GateSpec, connectivity_edges, valid_placements
from ansatz_search.search.base import AnsatzBuilder, SearchProblem


class IncrementalAnsatzBuilder(AnsatzBuilder):
    """Build an ansatz by sampling and appending one block at a time."""

    def build(self, trial, problem: SearchProblem, gate_specs: Sequence[GateSpec]) -> AnsatzSpec:
        spec = AnsatzSpec(num_qubits=problem.num_qubits)
        qubit_depths = [0] * problem.num_qubits
        connected = set(range(problem.num_qubits))
        curr_connected = []
        next_param_id = 0
        gate_count = 0
        suggest_stop = False

        while not suggest_stop:
            gate_idx = trial.suggest_categorical(
                f"gate_{gate_count}",
                range(len(gate_specs)),
            )
            gate_name, gate_qubits, gate_params = gate_specs[gate_idx]
            qubits = self._choose_qubits(
                trial,
                gate_qubits,
                gate_count,
                problem.num_qubits,
                curr_connected,
                problem.topology,
            )
            params, next_param_id = self._allocate_params(
                trial,
                gate_count,
                gate_params,
                next_param_id,
                spec,
                problem.prev_params,
            )
            spec.append(AnsatzBlock(op=gate_name, qubits=tuple(qubits), params=params))

            gate_count += 1
            depth = self._update_depth(qubit_depths, qubits)
            found = connected in curr_connected and spec.num_params > problem.min_params

            if spec.num_params > problem.max_param:
                self._prune(trial, "params")
            if depth > problem.max_depth:
                self._prune(trial, "depth")
            if gate_count > problem.max_gates:
                self._prune(trial, "gates")
            if found:
                if gate_count == problem.max_gates:
                    suggest_stop = True
                else:
                    suggest_stop = trial.suggest_categorical(
                        f"stop_at_{gate_count}",
                        [False, True],
                    )

        return spec

    def _choose_qubits(self, trial, gate_qubits, gate_count, num_qubits, curr_connected, topology):
        placements = valid_placements(num_qubits, gate_qubits, topology=topology)
        placement = placements[trial.suggest_int(f"placement_{gate_count}", 0, len(placements) - 1)]
        for q0, q1 in connectivity_edges(placement):
            curr_connected[:] = self._update_connections(curr_connected, q0, q1)
        return list(placement)

    def _allocate_params(self, trial, gate_count, gate_params, next_param_id, spec, reuse_previous):
        # A fresh index is consumed only when it is actually used. Reserving one
        # up front and then overwriting the ref with an earlier parameter would
        # leave a gap: an index counted by spec.num_params that appears in no
        # gate, and so contributes an identically-zero gradient.
        available = spec.num_params  # indices defined before this block
        refs = []
        for idx in range(gate_params):
            reuse = (
                reuse_previous
                and available > 0
                and trial.suggest_int(f"Gc_{gate_count}_prev_param_{idx}", 0, 1)
            )
            if reuse:
                prev_idx = trial.suggest_int(
                    f"Gc_{gate_count}_prev_param_choice_{idx}",
                    0,
                    available - 1,
                )
                refs.append(ParamRef(prev_idx))
            else:
                refs.append(ParamRef(next_param_id))
                next_param_id += 1

        return tuple(refs), next_param_id

    def _update_depth(self, qubit_depths, qubits):
        block_depth = 1 + max(qubit_depths[q] for q in qubits)
        for qubit in qubits:
            qubit_depths[qubit] = block_depth
        return max(qubit_depths)

    def _update_connections(self, curr_connected, q0, q1):
        combined = {q0, q1}
        remaining = []
        for connected in curr_connected:
            if connected & combined:
                combined |= connected
            else:
                remaining.append(connected)
        remaining.append(combined)
        return remaining

    def _prune(self, trial, reason: str):
        trial.set_user_attr("prune_reason", reason)
        count = trial.study.user_attrs.get(f"prune_{reason}", 0)
        trial.study.set_user_attr(f"prune_{reason}", count + 1)
        raise optuna.TrialPruned()
