"""Cost functions record raw values next to costs and apply thresholds."""

import pytest

from ansatz_search.cost_functions import HierarchicalCostFunction
from ansatz_search.cost_functions.weighted_hierarchical import WeightedHierarchicalCostFunction
from ansatz_search.metrics.base import Metric


class _Variance(Metric):
    """Higher raw value is better; cost = 1 - value, like GradientVariance."""

    def __init__(self, raw, name="variance"):
        self.raw, self._name, self.calls = raw, name, 0

    @property
    def name(self):
        return self._name

    def value(self, qc, spec=None):
        self.calls += 1
        return self.raw

    def cost(self, value):
        return max(1 - value, 0.0)


class _Error(Metric):
    """Raw value already is a cost (default identity), like GateError."""

    def __init__(self, raw, name="error"):
        self.raw, self._name = raw, name

    @property
    def name(self):
        return self._name

    def value(self, qc, spec=None):
        return self.raw


def test_records_value_and_cost_computing_each_value_once():
    variance = _Variance(0.2)
    cost, results = HierarchicalCostFunction({0: [variance, _Error(0.1)]}).evaluate(None)
    assert results == {
        "variance": {"value": 0.2, "cost": pytest.approx(0.8)},
        "error": {"value": 0.1, "cost": 0.1},
    }
    assert variance.calls == 1
    assert cost == pytest.approx(0.8 + 0.1)  # groups are summed by default


@pytest.mark.parametrize("aggregate, missed, met", [
    # Eq. (29) of arXiv:2603.14451: L1 + L2 + 1 while the first group misses, else the complexity loss.
    ("sum", 0.8 + 0.1 + 1, 0.4),
    ("mean", (0.8 + 0.1) / 2 + 1, 0.4),
])
def test_group_aggregation(aggregate, missed, met):
    later = [_Error(0.4, name="complexity")]
    missing = HierarchicalCostFunction({0: [_Variance(0.2), _Error(0.1)], 1: later}, aggregate=aggregate)
    assert missing.evaluate(None)[0] == pytest.approx(missed)
    meeting = HierarchicalCostFunction({0: [_Variance(1.0), _Error(0.0)], 1: later}, aggregate=aggregate)
    assert meeting.evaluate(None)[0] == pytest.approx(met)


def test_later_groups_with_several_metrics_never_outscore_a_missed_priority():
    # Summed, a two-metric later group can cost up to 2, so a missed first priority must start above 2.
    later = [_Error(1.0, name="e1"), _Variance(0.0, name="v2")]
    fn = HierarchicalCostFunction({0: [_Error(0.01)], 1: later})
    worst_later = HierarchicalCostFunction({0: [_Error(0.0)], 1: later})
    assert fn.evaluate(None)[0] == pytest.approx(2.01)
    assert worst_later.evaluate(None)[0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="aggregate"):
        HierarchicalCostFunction({0: [_Error(0.1)]}, aggregate="max")


def test_weights_scale_metrics_and_keep_the_priority_order():
    later = [_Error(0.4, name="complexity")]
    # Weighted sum within the group; a later group can cost at most the sum of its weights (here 3).
    fn = HierarchicalCostFunction({0: [_Variance(0.2), _Error(0.1)], 1: later},
                                  weights={"variance": 2.0, "complexity": 3.0})
    assert fn.evaluate(None)[0] == pytest.approx(2 * 0.8 + 0.1 + 3)
    met = HierarchicalCostFunction({0: [_Variance(1.0), _Error(0.0)], 1: later}, weights={"complexity": 3.0})
    assert met.evaluate(None)[0] == pytest.approx(3 * 0.4)
    mean = HierarchicalCostFunction({0: [_Variance(0.2), _Error(0.1)]}, aggregate="mean", weights={"variance": 3.0})
    assert mean.evaluate(None)[0] == pytest.approx((3 * 0.8 + 0.1) / 4)
    with pytest.raises(ValueError, match="unknown"):
        HierarchicalCostFunction({0: [_Error(0.1)]}, weights={"nope": 2.0})
    with pytest.raises(ValueError, match="positive"):
        HierarchicalCostFunction({0: [_Error(0.1)]}, weights={"error": 0.0})


@pytest.mark.parametrize("raw, cost", [
    (0.0, 1.0),    # plateau: still the worst cost
    (0.05, 0.5),   # halfway to the threshold
    (0.1, 0.0),    # threshold reached: requirement met
    (0.3, 0.0),    # past it
])
def test_threshold_is_in_raw_units(raw, cost):
    fn = HierarchicalCostFunction({0: [_Variance(raw)]}, thresholds={"variance": 0.1})
    _, results = fn.evaluate(None)
    assert results["variance"]["value"] == raw
    assert results["variance"]["cost"] == pytest.approx(cost)


def test_threshold_on_an_identity_cost_metric():
    fn = HierarchicalCostFunction({0: [_Error(0.05)]}, thresholds={"error": 0.1})
    assert fn.evaluate(None)[1]["error"]["cost"] == 0.0


def test_met_threshold_hands_over_to_the_next_priority():
    fn = HierarchicalCostFunction(
        {0: [_Variance(0.3)], 1: [_Error(0.4)]}, thresholds={"variance": 0.1}
    )
    assert fn.evaluate(None)[0] == pytest.approx(0.4)


def test_missing_a_priority_always_costs_more_than_meeting_it():
    thresholds = {"variance": 0.1}
    barely_missed = HierarchicalCostFunction({0: [_Variance(0.099)], 1: [_Error(0.0)]}, thresholds=thresholds)
    met_but_worst_later = HierarchicalCostFunction({0: [_Variance(0.1)], 1: [_Error(1.0)]}, thresholds=thresholds)
    assert barely_missed.evaluate(None)[0] == pytest.approx(1 + 0.01)  # one group still ahead + its score
    assert met_but_worst_later.evaluate(None)[0] == pytest.approx(1.0)
    assert barely_missed.evaluate(None)[0] > met_but_worst_later.evaluate(None)[0]
    everything_met = HierarchicalCostFunction({0: [_Variance(0.2)], 1: [_Error(0.0)]}, thresholds=thresholds)
    assert everything_met.evaluate(None)[0] == 0.0


def test_metrics_with_the_same_name_are_rejected():
    with pytest.raises(ValueError, match="same name.*error"):
        HierarchicalCostFunction({0: [_Error(0.1)], 1: [_Error(0.2)]})


def test_threshold_for_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="unknown metrics: \\['varaince'\\]"):
        HierarchicalCostFunction({0: [_Variance(0.1)]}, thresholds={"varaince": 0.1})


def test_weighted_cost_function_accepts_spec_and_uses_costs():
    fn = WeightedHierarchicalCostFunction({0: [_Variance(0.2)], 1: [_Error(0.1)]})
    cost, results = fn.evaluate(None, spec=None)
    assert cost == pytest.approx(0.8 * 2 + 0.1 * 1)
    assert results["variance"]["value"] == 0.2
