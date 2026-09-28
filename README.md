<h1 align="center">ansatz-search</h1>

<p align="center"><b>Find parameterized quantum circuits that are both expressive and trainable.</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.14451"><img src="https://img.shields.io/badge/arXiv-2603.14451-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/proeseler/ansatz-search/actions/workflows/tests.yml"><img src="https://github.com/proeseler/ansatz-search/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="License: Apache-2.0"></a>
</p>

<p align="center">
  <a href="https://www.fz-juelich.de/profile/roeseler_p"><b>Peter Röseler</b></a>,
  <a href="https://www.fz-juelich.de/profile/willsch_d"><b>Dennis Willsch</b></a>,
  <a href="https://www.fz-juelich.de/profile/michielsen_k"><b>Kristel Michielsen</b></a><br>
  Jülich Supercomputing Centre
</p>

<p align="center">
  <img src="docs/figures/expr_train.png" width="48%" alt="Expressibility vs trainability">
  <img src="docs/figures/expr_entgl.png" width="48%" alt="Expressibility vs entanglement">
</p>
<p align="center"><sub>
  Lower is better on both axes. Commonly used circuits (the 19 of Sim et al., 2019, with 1–5
  layers) trade expressibility against trainability (left) and entanglement (right); the circuits
  found by the search (★) lie closer to the ideal lower-left corner than any of them. Fig. 2 of the paper.
</sub></p>

**ansatz-search** scores candidate circuits with metrics (expressibility, trainability,
entanglement, gate error, complexity) and searches over circuit structures for the ones that
meet all of them, on NumPy, PennyLane, Qiskit or real IBM quantum computers. It is the code
accompanying

> P. Röseler, D. Willsch, K. Michielsen,
> *How to find expressible and trainable parameterized quantum circuits?*,
> [arXiv:2603.14451](https://arxiv.org/abs/2603.14451) (2026).

## 💡 Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [How it works](#how-it-works)
5. [Scoring your own circuits](#own-circuits)
6. [Running on quantum hardware](#hardware)
7. [Contributing](#contributing)
8. [Citation](#citation)
9. [License](#license)

<h2 id="features">✨ Features</h2>

- 📏 **Metrics:** expressibility, trainability (gradient-to-noise ratio), gradient variance,
  entanglement, gate error and complexity, combined by priority with thresholds and weights.
- 🖥️ **Backends:** a fast batched NumPy simulator (the default), PennyLane, and Qiskit, including
  simulated and real IBM quantum computers.
- 🔍 **Search:** Bayesian optimization over circuits built gate by gate, within limits on
  parameters, gates and depth; runs are stored and can be resumed.
- 📊 **Evaluation:** re-score the best circuits on fresh seeds next to the 19 benchmark circuits of
  Sim et al. (2019), and plot the results.

<h2 id="installation">⚙️ Installation</h2>

```bash
pip install "ansatz-search[plot] @ git+https://github.com/proeseler/ansatz-search.git"
```

The NumPy and Qiskit backends are always included. Optional extras go in the brackets,
separated by commas; `ansatz-search[plot,pennylane,ibm]` installs all of them:

- **`plot`**: plots of evaluations (matplotlib).
- **`pennylane`**: the PennyLane backend, with `lightning.qubit` for larger circuits. Needs
  Python 3.11 or later.
- **`ibm`**: IBM devices for the Qiskit backend: fake devices for local noisy simulation (Qiskit
  Aer) and real hardware through Qiskit Runtime, which needs an IBM Quantum account.

<h2 id="quickstart">🚀 Quickstart</h2>

```python
from ansatz_search import BayesianOptimizationSearch, HierarchicalCostFunction, SearchProblem, ansatz_search, evaluate
from ansatz_search.metrics import Expressibility, GradientVariance

# 1. What to search for: trainable and expressive 4-qubit circuits.
problem = SearchProblem(
    cost_fn=HierarchicalCostFunction({0: [
        GradientVariance(observables="ZZZZ", max_variance=0.1),
        Expressibility(num_qubits=4, log_scale=True),
    ]}),
    num_qubits=4, allowed_gates=["rx", "ry", "rz", "cx", "crz"], max_param=12, max_gates=24, max_depth=16,
)

# 2. How to search: Bayesian optimization, 100 trials; the run is stored in runs/quickstart.
result = ansatz_search(problem, BayesianOptimizationSearch(seed=42), fdata="runs/quickstart")

# 3. Re-evaluate the 5 best circuits on fresh seeds, next to the Sim et al. (2019) circuits.
evaluation = evaluate(result, baselines="sim2019")
print(evaluation)
evaluation.plot(out="runs/quickstart/eval.png")
```

This runs in about a minute. More in [examples/](examples/README.md): a custom metric,
expressibility on a simulated IBM device, and the metrics of the Sim et al. (2019) circuits.

<h2 id="how-it-works">🧩 How it works</h2>

1. **Describe the problem** with a `SearchProblem`: the number of qubits, the allowed gates, and
   limits on parameters, gates and depth.
2. **Say what a good circuit is** with a `CostFunction` of `Metric`s. Every metric turns its raw
   value into a cost between 0 and 1, where lower is better; the cost function combines them
   into the number the search minimizes.
3. **Search** with a `SearchAlgorithm`, which proposes circuits and keeps the best ones.
4. **Evaluate** with `evaluate`: re-score the best circuits on fresh seeds, next to benchmark
   circuits, and plot them.

`Metric`, `CostFunction` and `SearchAlgorithm` are base classes: use the ones that come with the
package or write your own. The details are next to the code: 📏 [metrics](src/ansatz_search/metrics/README.md)
and 🔍 [search](src/ansatz_search/search/README.md).

The circuits are simulated with NumPy by default, which is fastest at the usual search sizes.
Pass `compiler=` to `ansatz_search` or `evaluate` to use PennyLane for larger circuits, or
Qiskit for simulated IBM devices and real quantum hardware.

<h2 id="own-circuits">📐 Scoring your own circuits</h2>

The metrics also score circuits you already have, without a search. A metric takes a
parameterized Qiskit circuit directly:

```python
from qiskit.circuit.library import efficient_su2
from ansatz_search.metrics import Expressibility

qc = efficient_su2(4, reps=2)
metric = Expressibility(num_qubits=4, seed=1)
value = metric.value(qc)    # raw KL divergence
cost = metric.cost(value)   # in [0, 1], lower is better
```

To compare it with the benchmark circuits over several seeds, convert it with
`AnsatzSpec.from_qiskit` and pass it to `evaluate`. The converted circuit also runs on the fast NumPy
simulator, which matters for gradient metrics:

```python
from ansatz_search import AnsatzSpec

cost_fn = HierarchicalCostFunction({0: [GradientVariance(observables="ZZZZ", max_variance=0.1), metric]})
evaluation = evaluate({"EfficientSU2": AnsatzSpec.from_qiskit(qc)}, cost_fn=cost_fn, baselines="sim2019")
```

`from_qiskit` accepts the gates every backend supports (x, y, z, h, s, sdg, t, tdg, sx, rx,
ry, rz, p, cx, cy, cz, ch, swap, ecr, crx, cry, crz, cp, rxx, ryy, rzz) and splits u/u3 into rz,
ry, rz, so circuits transpiled to IBM's native gates, e.g. with
`transpile(qc, basis_gates=["rz", "sx", "x", "cz"])`, convert as they are. An angle can be a
parameter θ, θ + c (as the transpiler writes it) or a fixed number, but not an expression such
as `2 * θ`.

<h2 id="hardware">🔬 Running on quantum hardware</h2>

A metric runs on hardware when its provider does. Expressibility only needs state fidelities,
which `QiskitFidelityProvider` measures by compute–uncompute, as in the paper:

```python
from qiskit_ibm_runtime.fake_provider import FakeManilaV2   # or QiskitRuntimeService().backend(...)
from ansatz_search.backends.qiskit import QiskitCompiler, QiskitFidelityProvider

device = FakeManilaV2()
metric = Expressibility(num_qubits=4, num_of_param_samples=400,
                        fidelity_provider=QiskitFidelityProvider(backend=device, shots=2000, seed=42))
evaluation = evaluate(circuits, cost_fn=HierarchicalCostFunction({0: [metric]}), compiler=QiskitCompiler())
```

`QiskitGradientProvider(backend=device)` does the same for gradients. See
[examples/hardware_expressibility.py](examples/hardware_expressibility.py).

<h2 id="contributing">🤝 Contributing</h2>

New metrics, search algorithms and backends are welcome. A metric is one small class with a
`value` and a `cost` method, as in [examples/custom_metric.py](examples/custom_metric.py).
See [CONTRIBUTING.md](CONTRIBUTING.md) for the setup and what each component needs.

<h2 id="citation">📚 Citation</h2>

If you use this code, please cite the paper (also in [CITATION.cff](CITATION.cff)):

```bibtex
@article{roeseler2026expressible,
  title   = {How to find expressible and trainable parameterized quantum circuits?},
  author  = {R{\"o}seler, Peter and Willsch, Dennis and Michielsen, Kristel},
  journal = {arXiv preprint arXiv:2603.14451},
  year    = {2026}
}
```

<details>
<summary>Settings that reproduce the paper's definitions</summary>

- **Expressibility** (Eq. 8): 75 bins, `num_of_param_samples=10000` (the paper's 5,000 pairs),
  `log_scale=True`, and `thresholds={"expressibility": 0.005}` on the cost function.
- **Trainability** (Eq. 20): `Trainability(observables="ZIII")` with its defaults (uniform
  parameters in [0, 2π), 11,806 samples, `p1_err=0.001`, `p2_err=0.01`, gates counted as written)
  and `thresholds={"trainability": tau_BP}`. Observable strings list qubit 0 first: `"ZIII"` is
  Qiskit's `IIIZ`.
- **Objective**: `HierarchicalCostFunction` with the default `aggregate="sum"` (Eqs. 28–29).
- **Thresholds** (Table I): tau_BP = 8.0 (trainability), 2.0 (expressibility and trainability),
  0.6 (QNN), 2.5 (H2 benchmark), 0.083 (LiH).
- **Gradients**: the paper used finite differences with step 1e-7. The code computes exact
  gradients by default; for finite differences pass
  `gradient_provider=NumpyGradientProvider(method="finite_difference", epsilon=1e-7)`. The
  trainability values differ by about 1e-9 (relative).

</details>

<h2 id="license">📄 License</h2>

Apache License 2.0, see [LICENSE](LICENSE).
