from abc import abstractmethod
from typing import Dict, Mapping, Optional, Sequence, Tuple

from ansatz_search.metrics.base import Metric
from ansatz_search.utils.seeding import Reseedable

# {metric name: {"value": raw metric, "cost": cost used by the search}}
MetricResults = Dict[str, Dict[str, float]]


class CostFunction(Reseedable):
    """
    Abstract base for any ansatz‐quality cost function.
    Subclasses must implement `evaluate`, returning a scalar score.
    `with_seed(seed)` returns the same cost function with all metrics reseeded.

    `thresholds` optionally maps a metric name to a raw value that counts as
    "good enough" for this search, in the metric's own units (e.g. a gradient
    variance of 0.1). A metric at or past its threshold costs 0, which is what
    lets a hierarchical cost function move on to the next priority.
    """

    def __init__(self, metrics: Sequence[Metric], thresholds: Optional[Mapping[str, float]] = None):
      self.metrics = metrics
      names = [metric.name for metric in metrics]
      duplicates = sorted({name for name in names if names.count(name) > 1})
      if duplicates:
          # Results are keyed by name, so one metric would silently overwrite the other.
          raise ValueError(f"Several metrics have the same name: {duplicates}.")
      self.thresholds = dict(thresholds or {})
      unknown = set(self.thresholds) - {metric.name for metric in metrics}
      if unknown:
          raise ValueError(f"Thresholds given for unknown metrics: {sorted(unknown)}.")
      for metric in metrics:
          if metric.name in self.thresholds:
              self._check_threshold(metric, self.thresholds[metric.name])

    @staticmethod
    def _check_threshold(metric: Metric, threshold: float) -> None:
      """Reject a threshold beyond the point where the metric's own cost already is 0.

      There, values that miss the threshold would still cost 0, so the
      threshold could not be honored.
      """
      if metric.cost(threshold) > 0:
          return
      step = 1e-9 * max(abs(threshold), 1.0)
      short_of_it = threshold - step if metric.higher_is_better else threshold + step
      if metric.cost(short_of_it) == 0:
          raise ValueError(
              f"The threshold {threshold} for {metric.name!r} lies where its cost already is 0, so values "
              "that miss the threshold would cost 0 too. Move the metric's normalization (e.g. max_ratio, "
              "max_variance) to at least the threshold."
          )

    def compute_metrics(self, circuit, spec=None) -> MetricResults:
      """
      Compute all configured metrics on the given circuit.

      `spec` is the backend-neutral AnsatzSpec the circuit was compiled from;
      metrics use it for shape information the compiled object may not expose.
      Each metric's raw value is computed once; its cost is derived from it.
      """
      results = {}
      for metric in self.metrics:
          value = metric.value(circuit, spec)
          results[metric.name] = {"value": value, "cost": self._cost(metric, value)}
      return results

    def _cost(self, metric: Metric, value: float) -> float:
      cost = metric.cost(value)
      if metric.name not in self.thresholds:
          return cost
      # Rescale so the threshold maps to 0 and the worst cost stays 1. Going
      # through metric.cost keeps this independent of the metric's direction;
      # for a linear cost it equals max(1 - value / threshold, 0).
      limit = metric.cost(self.thresholds[metric.name])
      if limit >= 1:
          return 0.0  # every value meets a threshold at the worst possible cost
      return max((cost - limit) / (1 - limit), 0.0)

    @abstractmethod
    def evaluate(self, circuit, spec=None) -> Tuple[float, MetricResults]:
        """
        Compute the metrics and combine their costs into a single cost.

        Returns:
          A scalar cost to minimize (lower is always better), and the
          per-metric results from `compute_metrics`.
        """
        pass
