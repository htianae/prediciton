"""End-to-end offline comparison and one-artifact Champion training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifact import build_artifact, load_artifact, predict_with_interval, save_artifact
from .data import FEATURE_COLS, TARGET_COL, load_batch_data, mark_target_outliers
from .modeling import choose_prediction_champion, evaluate_models_time_series, fit_champion
from .optimization import (
    CONTROLLABLE_COLS,
    build_context,
    build_search_space,
    compare_optimizers,
    fit_feasibility_reference,
)


def compute_safety_summary(scenarios: pd.DataFrame, target_iqr: float) -> pd.DataFrame:
    rows = []
    for model, group in scenarios.groupby("model"):
        feasible_rate = float(group["scenario_has_feasible"].mean())
        boundary_rate = float(group["boundary_fraction"].mean())
        uncertainty_median = float(group["prediction_std"].median())
        rows.append(
            {
                "model": model,
                "feasible_scenario_rate": feasible_rate,
                "boundary_hit_rate": boundary_rate,
                "uncertainty_median": uncertainty_median,
                "passes_safety": bool(
                    feasible_rate >= 0.90
                    and boundary_rate <= 0.25
                    and uncertainty_median <= 0.50 * target_iqr
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def _select_optimizer(runs: pd.DataFrame, tie_ratio: float = 0.005) -> str:
    medians = runs.groupby("optimizer")["best_objective"].median()
    random_value = float(medians["random_search"])
    ga_value = float(medians["genetic_algorithm"])
    if abs(random_value - ga_value) / max(abs(random_value), 1e-12) <= tie_ratio:
        return "random_search"
    return str(medians.idxmin())


def _best_scenario_row(model_name: str, weight: float, best) -> dict:
    return {
        "model": model_name,
        "total_weight": float(weight),
        "seed": int(best["seed"]),
        "scenario_has_feasible": bool(best["scenario_has_feasible"]),
        "boundary_fraction": float(best["boundary_hit_count"] / len(CONTROLLABLE_COLS)),
        "prediction_std": float(best["prediction_std"]),
        "predicted_gas": float(best["predicted_gas"]),
        "penalized_objective": float(best["penalized_objective"]),
        "normalized_knn_distance": float(best["normalized_knn_distance"]),
    }


def train_and_select(
    excel_path: str | Path,
    output_dir: str | Path,
    evaluation_budget: int = 5000,
    seeds=range(10),
    weight_quantiles=(0.10, 0.25, 0.50, 0.75, 0.90),
) -> dict:
    output = Path(output_dir)
    artifacts_dir = output / "artifacts"
    reports_dir = output / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = mark_target_outliers(load_batch_data(excel_path))
    evaluation = evaluate_models_time_series(df)
    feasibility = fit_feasibility_reference(df)
    target_iqr = float(df[TARGET_COL].quantile(0.75) - df[TARGET_COL].quantile(0.25))
    weights = [float(df[FEATURE_COLS[0]].quantile(q)) for q in weight_quantiles]

    temporary_bundles = {}
    optimizer_run_parts = []
    best_candidate_parts = []
    optimizer_by_model = {}
    scenario_rows = []

    for model_name in evaluation["summary"]["model"]:
        temporary_bundles[model_name] = build_artifact(
            df,
            model_name,
            fit_champion(df, model_name),
            evaluation["cv_models"][model_name],
            evaluation["oof_predictions"][model_name],
            evaluation["summary"],
        )
        model_runs = []
        model_best = []
        for weight in weights:
            context = build_context(df, weight)
            space = build_search_space(context["batches"])
            comparison = compare_optimizers(
                temporary_bundles[model_name],
                context,
                space,
                feasibility,
                seeds=seeds,
                budget=evaluation_budget,
                keep_results=False,
            )
            runs = comparison["runs"].copy()
            runs["model"] = model_name
            runs["total_weight"] = weight
            model_runs.append(runs)
            best_candidates = comparison["best_candidates"].copy()
            best_candidates["model"] = model_name
            best_candidates["total_weight"] = weight
            model_best.append(best_candidates)
        model_runs_df = pd.concat(model_runs, ignore_index=True)
        optimizer_run_parts.append(model_runs_df)
        optimizer_name = _select_optimizer(model_runs_df)
        optimizer_by_model[model_name] = optimizer_name
        model_best_df = pd.concat(model_best, ignore_index=True)
        best_candidate_parts.append(model_best_df)
        selected_best = model_best_df[model_best_df["optimizer"] == optimizer_name]
        for _, best in selected_best.iterrows():
            scenario_rows.append(_best_scenario_row(model_name, best["total_weight"], best))

    stress_test = pd.DataFrame(scenario_rows)
    safety_summary = compute_safety_summary(stress_test, target_iqr)
    champion_name = choose_prediction_champion(evaluation["summary"], safety_summary)
    optimizer_name = optimizer_by_model[champion_name]
    fallback_to_history = not bool(
        safety_summary.set_index("model").loc[champion_name, "passes_safety"]
    )

    champion_bundle = temporary_bundles[champion_name]
    champion_bundle["optimizer_name"] = optimizer_name
    champion_bundle["optimizer_config"] = {
        "evaluation_budget": int(evaluation_budget),
        "seeds": [int(seed) for seed in seeds],
        "weight_quantiles": [float(q) for q in weight_quantiles],
        "search_quantiles": [0.05, 0.95],
        "fallback_to_history": fallback_to_history,
    }
    champion_bundle["feasibility_reference"] = feasibility
    model_path = save_artifact(champion_bundle, artifacts_dir / "gas_champion.joblib")

    reloaded = load_artifact(model_path)
    smoke_X = df[FEATURE_COLS].dropna().iloc[[0]]
    before = predict_with_interval(champion_bundle, smoke_X)["predicted_gas"].to_numpy()
    after = predict_with_interval(reloaded, smoke_X)["predicted_gas"].to_numpy()
    np.testing.assert_allclose(before, after, rtol=0, atol=1e-10)

    optimizer_runs = pd.concat(optimizer_run_parts, ignore_index=True)
    evaluation["summary"].to_csv(reports_dir / "model_comparison.csv", index=False)
    evaluation["fold_metrics"].to_csv(reports_dir / "fold_metrics.csv", index=False)
    stress_test.to_csv(reports_dir / "recommendation_stress_test.csv", index=False)
    safety_summary.to_csv(reports_dir / "safety_summary.csv", index=False)
    optimizer_runs.to_csv(reports_dir / "optimizer_comparison.csv", index=False)

    champion_summary = {
        "champion_model": champion_name,
        "optimizer_name": optimizer_name,
        "fallback_to_history": fallback_to_history,
        "training_batches": int(len(df)),
        "high_gas_outliers_retained": int(df["is_high_gas_outlier"].sum()),
        "artifact_path": str(model_path.resolve()),
    }
    (reports_dir / "champion_summary.json").write_text(
        json.dumps(champion_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        **champion_summary,
        "model_summary": evaluation["summary"],
        "fold_metrics": evaluation["fold_metrics"],
        "safety_summary": safety_summary,
        "stress_test": stress_test,
        "optimizer_runs": optimizer_runs,
        "optimizer_by_model": optimizer_by_model,
        "artifact": champion_bundle,
    }
