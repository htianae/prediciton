"""Conformal intervals and chronological fold disagreement."""

import numpy as np


def conformal_radius(y_true, oof_prediction, coverage: float = 0.90) -> float:
    y = np.asarray(y_true, dtype=float)
    prediction = np.asarray(oof_prediction, dtype=float)
    valid = np.isfinite(prediction)
    if not valid.any():
        raise ValueError("没有可用于 conformal 校准的 OOF 预测。")
    return float(np.quantile(np.abs(y[valid] - prediction[valid]), coverage))


def prediction_interval(prediction, radius: float):
    values = np.asarray(prediction, dtype=float)
    return np.maximum(0.0, values - float(radius)), values + float(radius)


def fold_prediction_summary(fold_models, X):
    matrix = np.vstack([model.predict(X) for model in fold_models])
    return {
        "matrix": matrix,
        "median": np.median(matrix, axis=0),
        "mean": matrix.mean(axis=0),
        "std": matrix.std(axis=0),
    }
