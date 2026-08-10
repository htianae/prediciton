"""Development-only model selection and one-time locked audit."""

from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ShuffleSplit

from .data import FEATURE_COLS, TARGET_COL
from .models import (
    ResidualBoostRegressor,
    RouteRegressor,
    WeightedOOFEnsemble,
    build_base_models,
    build_tree_model_variants,
    fit_nonnegative_ensemble_weights,
)
from .uncertainty import conformal_radius, prediction_interval
from .validation import bootstrap_metric_interval, regression_metrics


@dataclass
class CandidateResult:
    name: str
    route: str
    estimator_template: object
    oof_predictions: np.ndarray
    fold_models: list
    fold_metrics: pd.DataFrame
    selection_score: float
    conformal_radius_90: float


@dataclass
class ModelExperiment:
    dev_df: pd.DataFrame
    results: dict[str, CandidateResult]
    summary: pd.DataFrame
    selected_name: str
    ensemble_members: list[str]
    ensemble_weights: np.ndarray
    tree_tuning_summary: pd.DataFrame
    frozen: bool = False
    full_dev_models: dict[str, object] = field(default_factory=dict)

    def freeze(self):
        self.frozen = True


def evaluate_candidate(name, estimator, route, dev_df, splits) -> CandidateResult:
    X = dev_df[FEATURE_COLS]
    y = dev_df[TARGET_COL].to_numpy(dtype=float)
    oof = np.full(len(dev_df), np.nan)
    models = []
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splits):
        model = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X.iloc[train_idx], y[train_idx])
        prediction = model.predict(X.iloc[valid_idx])
        oof[valid_idx] = prediction
        models.append(model)
        rows.append({
            "candidate": name,
            "route": route,
            "fold": fold,
            "train_size": len(train_idx),
            "valid_size": len(valid_idx),
            **regression_metrics(y[valid_idx], prediction),
        })
    metrics = pd.DataFrame(rows)
    score = float(metrics["rmse"].mean() + 0.25 * metrics["rmse"].std())
    return CandidateResult(
        name=name,
        route=route,
        estimator_template=estimator,
        oof_predictions=oof,
        fold_models=models,
        fold_metrics=metrics,
        selection_score=score,
        conformal_radius_90=conformal_radius(y, oof, 0.90),
    )


def _summary_row(result: CandidateResult):
    metrics = result.fold_metrics
    return {
        "candidate": result.name,
        "route": result.route,
        "mae_mean": metrics["mae"].mean(),
        "rmse_mean": metrics["rmse"].mean(),
        "rmse_std": metrics["rmse"].std(),
        "rmse_worst": metrics["rmse"].max(),
        "wape_mean": metrics["wape"].mean(),
        "r2_mean": metrics["r2"].mean(),
        "error_gt_10pct_rate": metrics["error_gt_10pct_rate"].mean(),
        "selection_score": result.selection_score,
        "conformal_radius_90": result.conformal_radius_90,
    }


def run_model_matrix(dev_df, splits, model_names=None, routes=("direct", "unit"), tune_trees: bool = True) -> ModelExperiment:
    base_models = build_base_models()
    names = tuple(base_models) if model_names is None else tuple(model_names)
    tuning_rows = []
    if tune_trees:
        variants = build_tree_model_variants()
        for tree_name in ("LightGBM", "CatBoost"):
            if tree_name not in names:
                continue
            scored_variants = []
            for label, template in variants[tree_name].items():
                variant_result = evaluate_candidate(
                    f"tuning__{tree_name}__{label}",
                    RouteRegressor(template, "direct"),
                    "direct",
                    dev_df,
                    splits,
                )
                scored_variants.append((variant_result.selection_score, label, template))
                tuning_rows.append({
                    "model": tree_name,
                    "variant": label,
                    "selection_score": variant_result.selection_score,
                    "rmse_mean": variant_result.fold_metrics["rmse"].mean(),
                    "rmse_std": variant_result.fold_metrics["rmse"].std(),
                })
            _, selected_label, selected_template = min(scored_variants, key=lambda item: item[0])
            base_models[tree_name] = selected_template
            for row in tuning_rows:
                if row["model"] == tree_name:
                    row["selected"] = row["variant"] == selected_label
        if "LightGBM" in names:
            base_models["Ridge+LGBMResidual"] = ResidualBoostRegressor(
                base_models["Ridge"], base_models["LightGBM"]
            )
            base_models["Huber+LGBMResidual"] = ResidualBoostRegressor(
                base_models["Huber"], base_models["LightGBM"]
            )
    results = {}
    for model_name in names:
        for route in routes:
            candidate_name = f"{model_name}__{route}"
            estimator = RouteRegressor(base_models[model_name], route)
            results[candidate_name] = evaluate_candidate(candidate_name, estimator, route, dev_df, splits)

    ranked = sorted(results.values(), key=lambda item: item.selection_score)
    ensemble_sources = ranked[: min(4, len(ranked))]
    valid = np.logical_and.reduce([np.isfinite(item.oof_predictions) for item in ensemble_sources])
    matrix = np.column_stack([item.oof_predictions[valid] for item in ensemble_sources])
    y_valid = dev_df[TARGET_COL].to_numpy(dtype=float)[valid]
    weights = fit_nonnegative_ensemble_weights(matrix, y_valid)
    ensemble = WeightedOOFEnsemble(
        [item.estimator_template for item in ensemble_sources], weights
    )
    results["OOFEnsemble"] = evaluate_candidate(
        "OOFEnsemble", ensemble, "mixed_total", dev_df, splits
    )
    summary = pd.DataFrame([_summary_row(result) for result in results.values()])
    summary = summary.sort_values("selection_score").reset_index(drop=True)
    return ModelExperiment(
        dev_df=dev_df.copy(),
        results=results,
        summary=summary,
        selected_name=str(summary.iloc[0]["candidate"]),
        ensemble_members=[item.name for item in ensemble_sources],
        ensemble_weights=weights,
        tree_tuning_summary=pd.DataFrame(tuning_rows),
    )


def random_cv_reference(experiment: ModelExperiment, n_splits: int = 3, seed: int = 42):
    X = experiment.dev_df[FEATURE_COLS]
    y = experiment.dev_df[TARGET_COL].to_numpy(dtype=float)
    splitter = ShuffleSplit(n_splits=n_splits, test_size=.20, random_state=seed)
    rows = []
    for name, result in experiment.results.items():
        for split_id, (train_idx, valid_idx) in enumerate(splitter.split(X)):
            model = clone(result.estimator_template)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X.iloc[train_idx], y[train_idx])
            prediction = model.predict(X.iloc[valid_idx])
            rows.append({
                "candidate": name,
                "split": split_id,
                "train_size": len(train_idx),
                "valid_size": len(valid_idx),
                **regression_metrics(y[valid_idx], prediction),
            })
    return pd.DataFrame(rows)


def audit_frozen_models(experiment: ModelExperiment, locked_df: pd.DataFrame) -> pd.DataFrame:
    if not experiment.frozen:
        raise RuntimeError("Model experiment must freeze() before locked audit.")
    X_dev = experiment.dev_df[FEATURE_COLS]
    y_dev = experiment.dev_df[TARGET_COL]
    X_lock = locked_df[FEATURE_COLS]
    y_lock = locked_df[TARGET_COL].to_numpy(dtype=float)
    rows = []
    for name, result in experiment.results.items():
        model = clone(result.estimator_template)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_dev, y_dev)
        prediction = model.predict(X_lock)
        try:
            native_std_mean = float(np.mean(model.predict_std(X_lock)))
        except (AttributeError, TypeError):
            native_std_mean = np.nan
        experiment.full_dev_models[name] = model
        low, high = prediction_interval(prediction, result.conformal_radius_90)
        metrics = regression_metrics(y_lock, prediction)
        rmse_low, rmse_high = bootstrap_metric_interval(y_lock, prediction, "rmse", seed=42, n_boot=500)
        rows.append({
            "candidate": name,
            "route": result.route,
            **metrics,
            "rmse_bootstrap_low95": rmse_low,
            "rmse_bootstrap_high95": rmse_high,
            "interval_coverage_90": float(((y_lock >= low) & (y_lock <= high)).mean()),
            "interval_mean_width": float(np.mean(high - low)),
            "native_model_std_mean": native_std_mean,
            "selected_before_lock": name == experiment.selected_name,
        })
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
