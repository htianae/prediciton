"""Chronological model validation and deterministic Champion selection."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .data import FEATURE_COLS, TARGET_COL


def _pandas_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median").set_output(transform="pandas")


def build_candidate_models() -> dict[str, Pipeline]:
    """Return fresh, approved model pipelines with fold-local preprocessing."""
    return {
        "LightGBM": Pipeline(
            [
                ("imputer", _pandas_imputer()),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=15,
                        max_depth=4,
                        min_child_samples=10,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=2.0,
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        ),
        "Linear": Pipeline(
            [
                ("imputer", _pandas_imputer()),
                ("scaler", StandardScaler().set_output(transform="pandas")),
                ("model", LinearRegression()),
            ]
        ),
        "Huber": Pipeline(
            [
                ("imputer", _pandas_imputer()),
                ("scaler", RobustScaler().set_output(transform="pandas")),
                ("model", HuberRegressor(epsilon=1.5, max_iter=3000)),
            ]
        ),
    }


def evaluate_models_time_series(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Evaluate every candidate on identical expanding chronological folds."""
    X = df[FEATURE_COLS]
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if y.isna().any():
        raise ValueError("目标总气耗包含缺失值，无法训练。")

    splits = list(TimeSeriesSplit(n_splits=n_splits).split(X))
    fold_rows: list[dict] = []
    oof_predictions: dict[str, np.ndarray] = {}
    cv_models: dict[str, list[Pipeline]] = {}

    for model_name, template in build_candidate_models().items():
        oof = np.full(len(df), np.nan, dtype=float)
        fitted_models: list[Pipeline] = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            model = clone(template)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            prediction = model.predict(X.iloc[test_idx])
            oof[test_idx] = prediction
            fitted_models.append(model)
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "train_end": int(max(train_idx)),
                    "test_start": int(min(test_idx)),
                    "mae": mean_absolute_error(y.iloc[test_idx], prediction),
                    "rmse": mean_squared_error(y.iloc[test_idx], prediction) ** 0.5,
                    "r2": r2_score(y.iloc[test_idx], prediction),
                }
            )
        oof_predictions[model_name] = oof
        cv_models[model_name] = fitted_models

    fold_metrics = pd.DataFrame(fold_rows)
    summary = (
        fold_metrics.groupby("model", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index(drop=True)
    )
    summary["selection_score"] = summary["rmse_mean"] + 0.25 * summary["rmse_std"]
    summary = summary.sort_values("selection_score").reset_index(drop=True)
    return {
        "summary": summary,
        "fold_metrics": fold_metrics,
        "oof_predictions": oof_predictions,
        "cv_models": cv_models,
        "splits": splits,
    }


def eligible_model_names(summary: pd.DataFrame) -> list[str]:
    best = float(summary["selection_score"].min())
    return summary.loc[summary["selection_score"] <= best * 1.05, "model"].tolist()


def _rank_near_ties(candidates: pd.DataFrame) -> str:
    score_best = float(candidates["selection_score"].min())
    candidates = candidates[candidates["selection_score"] <= score_best * 1.005]
    mae_best = float(candidates["mae_mean"].min())
    candidates = candidates[candidates["mae_mean"] <= mae_best * 1.005].copy()
    preference = {"Linear": 0, "Huber": 1, "LightGBM": 2}
    candidates["simplicity"] = candidates["model"].map(preference).fillna(99)
    return str(candidates.sort_values(["simplicity", "selection_score"]).iloc[0]["model"])


def choose_prediction_champion(summary: pd.DataFrame, safety_results: pd.DataFrame | None = None) -> str:
    """Choose one eligible safe model; fall back to prediction rank if none is safe."""
    candidates = summary[summary["model"].isin(eligible_model_names(summary))].copy()
    if safety_results is not None and not safety_results.empty:
        safe_names = set(safety_results.loc[safety_results["passes_safety"], "model"])
        safe = candidates[candidates["model"].isin(safe_names)]
        if not safe.empty:
            candidates = safe
    return _rank_near_ties(candidates)


def fit_champion(df: pd.DataFrame, model_name: str) -> Pipeline:
    models = build_candidate_models()
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}")
    model = deepcopy(models[model_name])
    model.fit(df[FEATURE_COLS], df[TARGET_COL])
    return model
