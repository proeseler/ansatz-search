# 📏 Metrics

A metric judges one property of a circuit. Every metric has two methods:

- `value(qc, spec)` returns the raw value in the metric's own units (a KL divergence, a
  variance, a probability, ...). This is the expensive part.
- `cost(value)` maps that value to a cost in [0, 1], where **lower is always better**, so cost
  functions can combine metrics without knowing which direction each one points.

```python
from ansatz_search.metrics import Expressibility

metric = Expressibility(num_qubits=4, seed=1)
value = metric.value(circuit, spec)   # raw KL divergence
cost = metric.cost(value)             # in [0, 1]
```

## Expressibility

How close the circuit's output states come to Haar-random states (Sim et al., 2019). Pairs of
random parameter vectors give fidelities F = |⟨ψ(θa)|ψ(θb)⟩|², whose histogram is compared with
the Haar distribution Beta(1, D − 1), D = 2ⁿ.

- **Value:** the KL divergence; lower is more expressive.
- **Cost:** 0 at the noise floor (what exactly Haar-random states reach with this many samples),
  1 for a circuit whose fidelities are all 1. `log_scale=True` maps this range on a log scale,
  which spreads out small divergences.
- Runs on hardware with `fidelity_provider=QiskitFidelityProvider(...)`, which measures the
  fidelities by compute–uncompute.

## Trainability

The paper's barren-plateau test: the gradient variance per unit of gate error,
Var[∂C] / Pr(err) (arXiv:2603.14451, Sec. II B). A ratio above a threshold τ_BP means the
gradients stand out from the expected noise.

- **Value:** the ratio; higher is better.
- **Cost:** `max(1 - ratio / max_ratio, 0)`. For the paper's loss, set the threshold on the cost
  function, e.g. `thresholds={"trainability": 2.5}`.

## GradientVariance

How strongly the cost changes with the parameters: the variance of ∂⟨O⟩/∂θ over random
parameters, for the given `observables` (e.g. `"ZZZZ"`).

- **Value:** the expected variance; higher is better.
- **Cost:** `max(1 - variance / max_variance, 0)`.

## Entanglement

How much entanglement the circuit creates: the Meyer–Wallach measure Q, averaged over random
parameters. Q is 0 for product states and 1 for e.g. GHZ states.

- **Value:** the average Q in [0, 1]; higher is better.
- **Cost:** `max(1 - Q / reference, 0)`, where `reference` defaults to the Haar-random average
  (D − 2) / (D + 1).

## GateError

The probability that at least one gate fails, 1 − (1 − p₁)^N₁ (1 − p₂)^N₂, with N₁ and N₂ the
numbers of one- and two-qubit gates (Eq. B1 of the paper). With `basis_gates`, the circuit is
transpiled to those gates first.

- **Value and cost:** the probability itself; lower is better.

## Complexity

The size of the circuit: parameters + gates + depth. Needs no simulation.

- **Value:** the count.
- **Cost:** the count divided by `max_param + max_gates + max_depth`. Set these to the search
  limits to use the whole [0, 1] range.

## Combining metrics

`HierarchicalCostFunction` groups metrics by priority (lower numbers first). The search only
improves a group once every earlier group is met, and a metric is met once it reaches its
threshold:

```python
cost_fn = HierarchicalCostFunction(
    {0: [Trainability(observables="ZIII")],               # first: be trainable
     1: [Expressibility(num_qubits=4, log_scale=True)]},  # then: be expressive
    thresholds={"trainability": 2.5},                     # raw value that counts as good enough
    weights={"expressibility": 2},                        # optional, within a group
)
```

Thresholds are given in the metric's raw units; a metric at or past its threshold costs 0.

To combine metrics differently, subclass `CostFunction` and implement
`evaluate(circuit, spec)`, which returns the objective and the metric results.

## Writing your own

Subclass `Metric` and implement `name` and `value`; override `cost` if the raw value is not
already a cost in [0, 1]. Metrics that sample random parameters subclass `SampledMetric`. See
[examples/custom_metric.py](../../../examples/custom_metric.py).
