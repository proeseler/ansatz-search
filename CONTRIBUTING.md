# Contributing

New metrics, search algorithms and backends are welcome, as are bug reports and fixes.
[examples/custom_metric.py](examples/custom_metric.py) is a complete metric in about 20 lines.

## Setup

```bash
git clone https://github.com/proeseler/ansatz-search.git
cd ansatz-search
uv sync --all-extras          # or: pip install -e ".[plot,pennylane,ibm]" pytest
uv run pytest                 # optional backends are skipped if not installed
```

## Adding a component

- **Metric**: subclass `ansatz_search.metrics.Metric` with a `name`, `value(circuit, spec)` and
  `cost(value)` in [0, 1] (lower is better); set `label` and `higher_is_better` for plots. A
  metric that samples takes a `seed` argument and creates all its randomness from it, so
  `with_seed` can rebuild it. Use the backend-neutral `spec` (an `AnsatzSpec`) where you can, so
  the metric works with every backend.
- **Backend**: a `Compiler` (from an `AnsatzSpec` to your circuit type) plus a `GradientProvider`
  and/or `StateProvider`, registered as defaults with
  `ansatz_search.backends.base.register_providers(YourProgram, gradient=..., state=...)`.
  Test it against the NumPy backend, as `tests/backends` does.
- **Search algorithm**: implement `SearchAlgorithm.run(problem, compiler, gate_specs)` and return
  a `SearchResult`; an `AnsatzBuilder` turns search decisions into circuits.

## Pull requests

1. For larger changes, open an issue first so we can agree on the design.
2. Add tests for new numerics, e.g. a new backend against the NumPy backend. Keep heavy
   dependencies optional, as an extra in `pyproject.toml`.
3. Make sure `uv run pytest` passes, and describe what you changed and how you tested it.

When reporting a problem, please include your Python, ansatz-search and Qiskit/PennyLane
versions, the backend, and the smallest circuit that shows it.
