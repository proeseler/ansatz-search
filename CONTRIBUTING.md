# Contributing

New metrics, search algorithms and backends are welcome, as are bug reports and fixes. The
[Extending](README.md#extending) section of the README shows what each component
needs; [examples/custom_metric.py](examples/custom_metric.py) is a complete metric.

## Setup

```bash
git clone https://github.com/proeseler/ansatz-search.git
cd ansatz-search
uv sync --all-extras          # or: pip install -e ".[plot,pennylane,ibm,datasets]" pytest
uv run pytest                 # optional backends are skipped if not installed
```

## Pull requests

1. For larger changes, open an issue first so we can agree on the design.
2. Add tests for new numerics, e.g. a new backend against the NumPy backend. Keep heavy
   dependencies optional, as an extra in `pyproject.toml`.
3. Make sure `uv run pytest` passes, and describe what you changed and how you tested it.

When reporting a problem, please include your Python, ansatz-search and Qiskit/PennyLane
versions, the backend, and the smallest circuit that shows it.
