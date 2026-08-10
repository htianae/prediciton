"""End-to-end advanced offline experiment with a frozen locked audit."""

from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from .artifacts import save_bundle
from .data import FEATURE_COLS, chronological_dev_lock_split, load_batch_data, mark_target_outliers
from .experiment import audit_frozen_models, random_cv_reference, run_model_matrix
from .optimization import bayesian_search, compare_optimizers, genetic_search, random_search
from .recommendation import (
    build_recommendation_context,
    fit_feasibility_reference,
    historical_fallback,
    recommend_or_fallback,
)
from .validation import large_window_splits


RECOMMENDATION_CANDIDATE_PREFIXES = (
    "LightGBM__direct",
    "GPR__direct",
    "Ridge+LGBMResidual__direct",
    "Huber+LGBMResidual__direct",
    "OOFEnsemble",
)
DEPLOYMENT_RECOMMENDATION_SEED = 42
OPTIMIZER_FUNCTIONS = {
    "random_search": random_search,
    "genetic_algorithm": genetic_search,
    "bayesian_optimization": bayesian_search,
}


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def run_advanced_pipeline(
    excel_path,
    output_dir,
    optimizer_budget: int = 600,
    seeds=(0, 1, 2),
    fast_mode: bool = False,
):
    output = Path(output_dir)
    artifacts_dir = output / "artifacts"
    reports_dir = output / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = mark_target_outliers(load_batch_data(excel_path))
    dev, locked = chronological_dev_lock_split(df, lock_size=44)
    splits = large_window_splits(len(dev))
    if fast_mode:
        experiment = run_model_matrix(
            dev, splits, model_names=("Ridge", "LightGBM", "GPR"), routes=("direct",)
        )
    else:
        experiment = run_model_matrix(dev, splits)
    selected_before_lock = experiment.selected_name
    random_cv = random_cv_reference(experiment, n_splits=3, seed=42)
    experiment.freeze()
    locked_audit = audit_frozen_models(experiment, locked)

    feasibility = fit_feasibility_reference(df)
    context = build_recommendation_context(df, 86000.0, trust_ratio=.10)
    recommendation_names = [
        name for name in RECOMMENDATION_CANDIDATE_PREFIXES if name in experiment.results
    ]
    if not recommendation_names:
        recommendation_names = [selected_before_lock]

    recommendation_rows = []
    recommendation_payloads = {}
    full_recommendation_models = {}
    optimizer_runs = []
    for name in recommendation_names:
        result = experiment.results[name]
        full_model = clone(result.estimator_template)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            full_model.fit(df[FEATURE_COLS], df.iloc[:, df.columns.get_loc("熔炼炉B当前批次总气耗_PLC")])
        full_recommendation_models[name] = full_model
        comparison = compare_optimizers(
            full_model,
            result.fold_models,
            context,
            feasibility,
            common_budget=optimizer_budget,
            seeds=seeds,
            keep_results=True,
        )
        runs = comparison["runs"].copy()
        runs["candidate"] = name
        optimizer_runs.append(runs)
        selected_optimizer = comparison["selected_optimizer"]
        # Seeds 0/1/2 compare optimizer stability. A separate fixed deployment
        # seed avoids selecting a lucky candidate from those comparison runs and
        # makes the report exactly reproducible through the joblib/CLI path.
        scored = OPTIMIZER_FUNCTIONS[selected_optimizer](
            full_model,
            result.fold_models,
            context,
            feasibility,
            budget=optimizer_budget,
            seed=DEPLOYMENT_RECOMMENDATION_SEED,
        )
        recommendation = recommend_or_fallback(scored, context)
        recommendation["candidate"] = name
        recommendation["optimizer"] = selected_optimizer
        recommendation_payloads[name] = recommendation
        recommendation_rows.append({
            "candidate": name,
            "optimizer": selected_optimizer,
            "source": recommendation["source"],
            "safety_pass": recommendation["safety_pass"],
            "safety_grade": recommendation["safety_grade"],
            "actual_baseline_gas": recommendation.get("actual_baseline_gas", context.actual_baseline_gas),
            "conservative_predicted_gas": recommendation.get("conservative_predicted_gas", np.nan),
            "estimated_saving_vs_actual_baseline": recommendation.get("estimated_saving_vs_actual_baseline", np.nan),
            "fold_consensus_rate": recommendation.get("fold_consensus_rate", np.nan),
            "prediction_std": recommendation.get("prediction_std", np.nan),
            "safety_reasons": " | ".join(recommendation.get("safety_reasons", [])),
            **recommendation["recommendation"],
        })
    recommendation_summary = pd.DataFrame(recommendation_rows)
    passing = recommendation_summary[recommendation_summary["safety_pass"]]
    if passing.empty:
        best_recommendation_name = min(
            recommendation_names,
            key=lambda name: experiment.results[name].selection_score,
        )
        production_recommendation = historical_fallback(context)
        production_recommendation["candidate"] = best_recommendation_name
        production_recommendation["optimizer"] = recommendation_payloads[best_recommendation_name]["optimizer"]
    else:
        best_row = passing.sort_values("estimated_saving_vs_actual_baseline", ascending=False).iloc[0]
        best_recommendation_name = str(best_row["candidate"])
        production_recommendation = recommendation_payloads[best_recommendation_name]

    selected_result = experiment.results[selected_before_lock]
    prediction_model = clone(selected_result.estimator_template)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        prediction_model.fit(df[FEATURE_COLS], df["熔炼炉B当前批次总气耗_PLC"])
    bundle = {
        "artifact_version": "advanced-furnace-1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_cols": list(FEATURE_COLS),
        "training_batches": len(df),
        "dev_batches": len(dev),
        "locked_batches": len(locked),
        "selected_model_name": selected_before_lock,
        "prediction_model": prediction_model,
        "prediction_fold_models": selected_result.fold_models,
        "conformal_radius_90": selected_result.conformal_radius_90,
        "model_cv_summary": experiment.summary,
        "tree_tuning_summary": experiment.tree_tuning_summary,
        "random_cv_reference": random_cv,
        "locked_audit": locked_audit,
        "training_data": df,
        "recommendation_model_name": best_recommendation_name,
        "recommendation_model": full_recommendation_models[best_recommendation_name],
        "recommendation_fold_models": experiment.results[best_recommendation_name].fold_models,
        "recommendation_optimizer": production_recommendation["optimizer"],
        "recommendation_seed": DEPLOYMENT_RECOMMENDATION_SEED,
        "recommendation_trust_ratio": .10,
        "feasibility_reference": feasibility,
        "production_recommendation_86000": production_recommendation,
        "offline_savings_disclaimer": "离线预计节省不等于工厂实际节省，必须经过现场受控试验。",
    }
    model_path = save_bundle(bundle, artifacts_dir / "advanced_furnace_bundle.joblib")

    fold_metrics = pd.concat(
        [result.fold_metrics for result in experiment.results.values()], ignore_index=True
    )
    optimizer_run_table = pd.concat(optimizer_runs, ignore_index=True)
    experiment.summary.to_csv(reports_dir / "model_cv_summary.csv", index=False)
    experiment.tree_tuning_summary.to_csv(reports_dir / "tree_tuning_summary.csv", index=False)
    random_cv.to_csv(reports_dir / "random_cv_reference.csv", index=False)
    fold_metrics.to_csv(reports_dir / "chronological_fold_metrics.csv", index=False)
    locked_audit.to_csv(reports_dir / "locked_audit.csv", index=False)
    recommendation_summary.to_csv(reports_dir / "recommendation_summary.csv", index=False)
    optimizer_run_table.to_csv(reports_dir / "optimizer_comparison.csv", index=False)
    summary = {
        "selected_before_lock": selected_before_lock,
        "locked_batches": len(locked),
        "recommendation_model": best_recommendation_name,
        "recommendation_optimizer": production_recommendation["optimizer"],
        "recommendation_seed": DEPLOYMENT_RECOMMENDATION_SEED,
        "recommendation_safety_pass": bool(production_recommendation["safety_pass"]),
        "recommendation_source": production_recommendation["source"],
        "artifact_path": str(model_path.resolve()),
        "offline_savings_disclaimer": bundle["offline_savings_disclaimer"],
    }
    (reports_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_value), encoding="utf-8"
    )
    return {
        **summary,
        "bundle": bundle,
        "model_summary": experiment.summary,
        "tree_tuning_summary": experiment.tree_tuning_summary,
        "random_cv_reference": random_cv,
        "locked_audit": locked_audit,
        "recommendation_summary": recommendation_summary,
        "optimizer_runs": optimizer_run_table,
        "production_recommendation": production_recommendation,
    }
