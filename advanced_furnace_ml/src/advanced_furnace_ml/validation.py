"""Large-window chronological validation and locked audit helpers."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def large_window_splits(n_dev: int = 248):
    if n_dev != 248:
        raise ValueError("当前正式设计要求开发区恰好为248炉。")
    boundaries = [(148, 181), (181, 214), (214, 248)]
    return [(np.arange(train_end), np.arange(train_end, test_end)) for train_end, test_end in boundaries]


def regression_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    absolute = np.abs(y - p)
    denominator = np.maximum(np.abs(y), 1e-12)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "wape": float(absolute.sum() / denominator.sum()),
        "r2": float(r2_score(y, p)),
        "error_gt_10pct_rate": float((absolute / denominator > 0.10).mean()),
    }


def bootstrap_metric_interval(y_true, y_pred, metric: str, seed: int, n_boot: int = 1000):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, len(y), len(y))
        values.append(regression_metrics(y[index], p[index])[metric])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


@dataclass
class LockedAudit:
    _values: object
    _frozen: bool = False

    def freeze(self):
        self._frozen = True

    def values(self):
        if not self._frozen:
            raise RuntimeError("Locked audit cannot be read before freeze().")
        return self._values
