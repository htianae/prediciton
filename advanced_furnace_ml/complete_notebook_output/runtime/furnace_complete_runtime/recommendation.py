"""Historically grounded recommendation scoring and explicit safety gates."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

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


CONTROLLABLE_COLS = [SOLID_COL, WAIT_COL, DOOR_COUNT_COL, DOOR_DURATION_COL]


@dataclass(frozen=True)
class RecommendationContext:
    reference_df: pd.DataFrame
    total_weight: float
    melting_time: float
    bounds: dict[str, tuple[float, float]]
    baseline_controls: dict[str, float]
    actual_baseline_gas: float
    historical_low_gas_median: float
    target_iqr: float
    trust_ratio: float
    weight_tolerance: float | None
    low_confidence: bool


@dataclass(frozen=True)
class SafetyDecision:
    passes: bool
    grade: str
    reasons: tuple[str, ...]


def build_recommendation_context(df, total_weight: float, trust_ratio: float = 0.10):
    reference = pd.DataFrame()
    tolerance_used = None
    for tolerance in (0.05, 0.10):
        reference = df[df[WEIGHT_COL].between(total_weight * (1 - tolerance), total_weight * (1 + tolerance))]
        if len(reference) >= 20:
            tolerance_used = tolerance
            break
    low_confidence = len(reference) < 20
    if low_confidence:
        reference = df.copy()
    low_count = max(1, math.ceil(len(reference) * 0.20))
    low_gas = reference.nsmallest(low_count, TARGET_COL)
    baseline = {column: float(low_gas[column].median()) for column in CONTROLLABLE_COLS}
    bounds = {}
    for column in CONTROLLABLE_COLS:
        history_low = float(reference[column].quantile(0.05))
        history_high = float(reference[column].quantile(0.95))
        center = baseline[column]
        trust_low = center * (1 - trust_ratio)
        trust_high = center * (1 + trust_ratio)
        low, high = max(history_low, trust_low), min(history_high, trust_high)
        if column == DOOR_COUNT_COL:
            low, high = float(math.ceil(low)), float(math.floor(high))
        if low >= high:
            raise ValueError(f"参数 {column} 的历史范围与信任区域没有有效交集。")
        bounds[column] = (low, high)
    return RecommendationContext(
        reference_df=reference.copy(),
        total_weight=float(total_weight),
        melting_time=float(reference[MELTING_COL].median()),
        bounds=bounds,
        baseline_controls=baseline,
        actual_baseline_gas=float(reference[TARGET_COL].median()),
        historical_low_gas_median=float(low_gas[TARGET_COL].median()),
        target_iqr=float(df[TARGET_COL].quantile(.75) - df[TARGET_COL].quantile(.25)),
        trust_ratio=float(trust_ratio),
        weight_tolerance=tolerance_used,
        low_confidence=low_confidence,
    )


def fit_feasibility_reference(df):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    transformed = scaler.fit_transform(imputer.fit_transform(df[FEATURE_COLS]))
    neighbors = min(6, len(df))
    model = NearestNeighbors(n_neighbors=neighbors).fit(transformed)
    distance = model.kneighbors(transformed, return_distance=True)[0]
    reference_index = min(5, distance.shape[1] - 1)
    threshold = max(float(np.quantile(distance[:, reference_index], .95)), 1e-12)
    return {
        "imputer": imputer,
        "scaler": scaler,
        "neighbors": model,
        "distance_threshold": threshold,
        "reference_rows": len(df),
    }


def score_candidates(candidate_model, fold_models, candidates, context, feasibility):
    frame = candidates[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    if frame.isna().any().any():
        raise ValueError("候选参数必须是完整数值。")
    result = frame.copy()
    result["full_model_prediction"] = np.asarray(candidate_model.predict(frame), dtype=float)
    fold_matrix = np.vstack([model.predict(frame) for model in fold_models])
    result["fold_prediction_median"] = np.median(fold_matrix, axis=0)
    result["prediction_std"] = fold_matrix.std(axis=0)
    result["fold_consensus_rate"] = (fold_matrix < context.actual_baseline_gas).mean(axis=0)
    result["conservative_predicted_gas"] = result["fold_prediction_median"] + result["prediction_std"]
    result["estimated_saving_vs_actual_baseline"] = context.actual_baseline_gas - result["conservative_predicted_gas"]

    scaled = feasibility["scaler"].transform(feasibility["imputer"].transform(frame))
    neighbor_count = min(5, feasibility["reference_rows"])
    distance = feasibility["neighbors"].kneighbors(
        scaled, n_neighbors=neighbor_count, return_distance=True
    )[0][:, -1]
    result["normalized_knn_distance"] = distance / feasibility["distance_threshold"]
    result["historically_feasible"] = result["normalized_knn_distance"] <= 1.0

    boundary_count = np.zeros(len(result), dtype=int)
    change_penalty = np.zeros(len(result), dtype=float)
    for column, (low, high) in context.bounds.items():
        width = max(high - low, 1e-12)
        hit = ((result[column] - low) <= .02 * width) | ((high - result[column]) <= .02 * width)
        result[f"boundary__{column}"] = hit
        boundary_count += hit.astype(int)
        baseline = context.baseline_controls[column]
        change_penalty += np.abs(result[column] - baseline) / max(abs(baseline), 1.0)
    result["boundary_hit_count"] = boundary_count
    result["boundary_hit"] = boundary_count > 0
    result["mean_relative_change"] = change_penalty / len(CONTROLLABLE_COLS)
    result["within_trust_region"] = True
    result["penalized_objective"] = (
        result["conservative_predicted_gas"]
        + context.target_iqr * np.maximum(0, result["normalized_knn_distance"] - 1)
        + context.target_iqr * .35 * result["boundary_hit_count"]
        + context.target_iqr * .10 * result["mean_relative_change"]
    )
    return result


def safety_gate(row, context: RecommendationContext) -> SafetyDecision:
    reasons = []
    if float(row["conservative_predicted_gas"]) >= context.actual_baseline_gas:
        reasons.append("保守预测未低于相似历史实际气耗基准")
    if float(row["fold_consensus_rate"]) < (2 / 3):
        reasons.append("少于2/3时间折认为可以节气")
    if not bool(row["historically_feasible"]):
        reasons.append("历史近邻支持不足")
    if bool(row["boundary_hit"]):
        reasons.append("推荐命中搜索边界")
    if not bool(row.get("within_trust_region", True)):
        reasons.append("推荐超出信任区域")
    passes = not reasons
    return SafetyDecision(passes=passes, grade="A" if passes else "C", reasons=tuple(reasons))


def historical_fallback(context: RecommendationContext):
    return {
        "source": "historical_similar_low_gas",
        "recommendation": dict(context.baseline_controls),
        "actual_gas_median": context.historical_low_gas_median,
        "actual_baseline_gas": context.actual_baseline_gas,
        "safety_pass": False,
        "safety_grade": "fallback",
    }


def recommend_or_fallback(scored_candidates, context: RecommendationContext):
    ordered = scored_candidates.sort_values("penalized_objective")
    for _, row in ordered.iterrows():
        decision = safety_gate(row, context)
        if decision.passes:
            return {
                "source": "model_optimization",
                "recommendation": {column: float(row[column]) for column in CONTROLLABLE_COLS},
                "conservative_predicted_gas": float(row["conservative_predicted_gas"]),
                "estimated_saving_vs_actual_baseline": float(row["estimated_saving_vs_actual_baseline"]),
                "actual_baseline_gas": context.actual_baseline_gas,
                "historical_low_gas_median": context.historical_low_gas_median,
                "fold_consensus_rate": float(row["fold_consensus_rate"]),
                "prediction_std": float(row["prediction_std"]),
                "normalized_knn_distance": float(row["normalized_knn_distance"]),
                "safety_pass": True,
                "safety_grade": decision.grade,
                "safety_reasons": [],
            }
    fallback = historical_fallback(context)
    fallback["safety_reasons"] = ["没有模型候选同时通过全部离线安全门"]
    return fallback
