"""criterion estimator fields -> standard benchmark terms (the most-changed file).

Every term is one estimator's ``point_estimate``, kept in criterion's native
NANOSECONDS (the ``_ns`` suffix keeps the unit explicit; ``duration_us`` on the
unit is the mean converted for cross-backend ranking). The confidence intervals
and standard errors around each point estimate are raw detail, reachable via
``expand`` (``mean.confidence_interval.lower_bound``, ...), not core metrics.
"""

from __future__ import annotations

# standard benchmark term -> criterion estimates.json field (for the `expand` hint)
STANDARD_TO_VENDOR: dict[str, str] = {
    "mean_ns": "mean.point_estimate",
    "median_ns": "median.point_estimate",
    "std_dev_ns": "std_dev.point_estimate",
    "median_abs_dev_ns": "median_abs_dev.point_estimate",
    "slope_ns": "slope.point_estimate",
}

# What get_metrics(metrics=None) returns. slope is honestly absent (criterion
# writes JSON null) on runs that used flat sampling mode.
DEFAULT_CORE_SET: list[str] = [
    "mean_ns",
    "median_ns",
    "std_dev_ns",
    "median_abs_dev_ns",
    "slope_ns",
]
