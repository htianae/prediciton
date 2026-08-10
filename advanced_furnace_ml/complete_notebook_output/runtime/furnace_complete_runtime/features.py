"""Leakage-safe furnace feature engineering and target routes."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import (
    DOOR_COUNT_COL,
    DOOR_DURATION_COL,
    FEATURE_COLS,
    MELTING_COL,
    SOLID_COL,
    TARGET_COL,
    WAIT_COL,
    WEIGHT_COL,
)


DERIVED_COLS = [
    "derived__solid_weight_kg",
    "derived__avg_door_duration",
    "derived__door_count_per_melting_time",
    "derived__door_duration_per_melting_time",
    "derived__waiting_ratio",
    "derived__non_waiting_melting_time",
]


class FurnaceFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(FEATURE_COLS, dtype=object)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
        safe_melting = frame[MELTING_COL].replace(0, np.nan)
        safe_count = frame[DOOR_COUNT_COL].replace(0, np.nan)
        derived = pd.DataFrame(index=frame.index)
        derived[DERIVED_COLS[0]] = frame[WEIGHT_COL] * frame[SOLID_COL] / 100.0
        derived[DERIVED_COLS[1]] = frame[DOOR_DURATION_COL] / safe_count
        derived[DERIVED_COLS[2]] = frame[DOOR_COUNT_COL] / safe_melting
        derived[DERIVED_COLS[3]] = frame[DOOR_DURATION_COL] / safe_melting
        derived[DERIVED_COLS[4]] = frame[WAIT_COL] / safe_melting
        derived[DERIVED_COLS[5]] = frame[MELTING_COL] - frame[WAIT_COL]
        derived = derived.replace([np.inf, -np.inf], np.nan)
        return pd.concat([frame, derived], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_COLS + DERIVED_COLS, dtype=object)


def _weight_tonnes(frame: pd.DataFrame, median: float | None = None) -> pd.Series:
    weight = pd.to_numeric(frame[WEIGHT_COL], errors="coerce")
    fill = float(weight.median()) if median is None else float(median)
    return weight.fillna(fill) / 1000.0


def target_for_route(df: pd.DataFrame, route: str) -> pd.Series:
    target = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if route == "direct":
        return target
    if route == "unit":
        return target / _weight_tonnes(df)
    raise ValueError(f"未知目标路线: {route}")


def prediction_to_total_gas(prediction, X: pd.DataFrame, route: str, weight_median: float | None = None) -> np.ndarray:
    values = np.asarray(prediction, dtype=float)
    if route == "direct":
        return values
    if route == "unit":
        return values * _weight_tonnes(X, weight_median).to_numpy()
    raise ValueError(f"未知目标路线: {route}")
