# 🔍 Search

A search needs two things: a `SearchProblem` that says what to look for, and a `SearchAlgorithm`
that says how. `ansatz_search` runs one on the other and stores the run:

```python
from ansatz_search import BayesianOptimizationSearch, SearchProblem, ansatz_search

problem = SearchProblem(cost_fn=cost_fn, num_qubits=4, allowed_gates=["rx", "ry", "rz", "cx"],
                        max_param=12, max_gates=24, max_depth=16)
result = ansatz_search(problem, BayesianOptimizationSearch(n_trials=100, seed=42), fdata="runs/my_run")
```

## The search problem

| Field | Meaning |
|---|---|
| `cost_fn` | what a good circuit is, e.g. a `HierarchicalCostFunction` of [metrics](../metrics/README.md) |
| `num_qubits` | the number of qubits |
| `allowed_gates` | gate names the search may use, e.g. `"rx"`, `"cx"`, `"crz"`, `"rzz"` |
| `max_param` | the most trainable parameters a circuit may have |
| `max_gates` | the most gates a circuit may have |
| `max_depth` | the largest circuit depth |
| `min_params` | a circuit needs more parameters than this before the search may stop it (default 0) |
| `prev_params` | whether a gate may reuse an earlier gate's parameter (default `True`) |
| `topology` | the qubit pairs two-qubit gates may act on, as `{qubit: [neighbors]}` (default: all pairs) |

## How a circuit is built

`IncrementalAnsatzBuilder` builds each candidate gate by gate. For every gate the search chooses

1. which gate from `allowed_gates`,
2. which qubits it acts on (within `topology`),
3. for each parameter, a new one or, with `prev_params`, one already used,

and then whether to stop. It may stop only once all qubits are connected by two-qubit gates and
the circuit has more than `min_params` parameters. A circuit that exceeds `max_param`,
`max_gates` or `max_depth` is discarded (pruned).

## Bayesian optimization

`BayesianOptimizationSearch` makes these choices with Optuna's TPE sampler: it learns from the
earlier trials which choices lead to low costs. Useful options:

- `n_trials`: how many circuits to try (default 100).
- `seed`: makes the search reproducible.
- `storage`: `"journal"` (default) stores every trial in the run folder, so an interrupted run
  resumes where it stopped and several processes can share one search; `"sqlite"` does the same
  in a database; `"memory"` stores nothing.
- `builder`: another `AnsatzBuilder`, to build circuits differently.

## The result

`ansatz_search` returns a `SearchResult` with the best circuit (`best_ansatz`), its cost and
metric values, and the folder of the run. The best cost comes from one noisy evaluation, so it
is optimistic: `evaluate(result)` re-scores the best circuits on fresh seeds.

## Writing your own

A search algorithm subclasses `SearchAlgorithm` and implements
`run(problem, compiler, gate_specs)`, returning a `SearchResult`. See
[CONTRIBUTING.md](../../../CONTRIBUTING.md).
