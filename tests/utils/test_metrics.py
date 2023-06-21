from typing import List, Tuple
from hypothesis.strategies import lists, floats, tuples, builds
from hypothesis import given, settings, HealthCheck
import pytest
import numpy as np
from utils.metrics import DeterministicErrorMetrics, ProbabilisticErrorMetrics

# Define hypothesis strategies
float_lists = lists(floats(min_value=-100, max_value=100,
                           allow_nan=False, allow_infinity=False), min_size=2, max_size=100)

# Strategy for generating (yt, yp) pairs with same length
yt_yp_strategy = tuples(float_lists, float_lists).filter(
    lambda x: len(x[0]) == len(x[1]))

# Strategy for generating (yt, yp, yp_lower, yp_upper) tuples


def adjusted_lists(elements=floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), size=10):
    return builds(
        lambda x: (x, [i - np.random.uniform(0.01, 10)
                   for i in x], [i + np.random.uniform(0.01, 10) for i in x]),
        lists(elements, min_size=size, max_size=size)
    )


yt_yp_lower_upper_strategy = tuples(float_lists, adjusted_lists()).filter(
    lambda x: len(x[0]) == len(x[1][0]) == len(x[1][1]) == len(x[1][2])
)


# Suppress the health check for filtering too much data.
hc_suppress_filter_too_much = HealthCheck.filter_too_much

# Suppress the health check for slow tests.
hc_suppress_too_slow = HealthCheck.too_slow


@given(yt_yp_strategy)
@settings(deadline=None)
def test_deterministic_metrics_initialization(yt_yp: Tuple[List[float], List[float]]):
    yt, yp = yt_yp
    deterministic_metrics = DeterministicErrorMetrics(
        yt=np.array(yt), yp=np.array(yp))
    assert isinstance(deterministic_metrics.yt, np.ndarray)
    assert isinstance(deterministic_metrics.yp, np.ndarray)
    assert len(deterministic_metrics.yt) == len(
        deterministic_metrics.yp), "Sizes of yt and yp should match"


@given(yt_yp_strategy)
@settings(deadline=None)
def test_deterministic_metrics_calculations(yt_yp: Tuple[List[float], List[float]]):
    yt, yp = yt_yp
    deterministic_metrics = DeterministicErrorMetrics(
        yt=np.array(yt), yp=np.array(yp))
    # assert isinstance(deterministic_metrics.get_iqr(), float)
    assert isinstance(deterministic_metrics.nrmse(), float)
    assert isinstance(deterministic_metrics.nmae(), float)
    assert isinstance(deterministic_metrics.medianae(), float)
    assert isinstance(deterministic_metrics.mape(), float)
    assert isinstance(deterministic_metrics.bias(), float)
    assert isinstance(deterministic_metrics.nbe(), float)


@settings(suppress_health_check=[hc_suppress_filter_too_much, hc_suppress_too_slow], deadline=None)
@given(yt_yp_lower_upper_strategy)
def test_probabilistic_metrics_initialization(yt_yp_lower_upper: Tuple[List[float], Tuple[List[float], List[float], List[float]]]):
    yt, (yp, yp_lower, yp_upper) = yt_yp_lower_upper
    probabilistic_metrics = ProbabilisticErrorMetrics(yt=np.array(yt), yp=np.array(
        yp), yp_lower=np.array(yp_lower), yp_upper=np.array(yp_upper), confidence_level=0.95)
    assert isinstance(probabilistic_metrics.yt, np.ndarray)
    assert isinstance(probabilistic_metrics.yp, np.ndarray)
    assert isinstance(probabilistic_metrics.yp_lower, np.ndarray)
    assert isinstance(probabilistic_metrics.yp_upper, np.ndarray)
    assert probabilistic_metrics.confidence_level == 0.95
    assert len(probabilistic_metrics.yt) == len(probabilistic_metrics.yp) == len(
        probabilistic_metrics.yp_lower) == len(probabilistic_metrics.yp_upper), "Sizes of yt, yp, yp_lower and yp_upper should match"


@settings(suppress_health_check=[hc_suppress_filter_too_much, hc_suppress_too_slow], deadline=None)
@given(yt_yp_lower_upper_strategy)
def test_probabilistic_metrics_calculations(yt_yp_lower_upper: Tuple[List[float], Tuple[List[float], List[float], List[float]]]):
    yt, (yp, yp_lower, yp_upper) = yt_yp_lower_upper
    probabilistic_metrics = ProbabilisticErrorMetrics(yt=np.array(yt), yp=np.array(
        yp), yp_lower=np.array(yp_lower), yp_upper=np.array(yp_upper), confidence_level=0.95)
    assert isinstance(probabilistic_metrics.ace(), float)
    assert isinstance(probabilistic_metrics.pinaw(), float)
    assert isinstance(probabilistic_metrics.interval_sharpness(), float)
    assert isinstance(probabilistic_metrics.gaussian_crps(), float)
