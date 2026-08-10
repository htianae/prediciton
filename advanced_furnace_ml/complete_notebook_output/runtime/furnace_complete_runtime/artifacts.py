"""Persist, validate, predict, and recommend from the advanced bundle."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import FEATURE_COLS
from .optimization import bayesian_search, genetic_search, random_search
from .recommendation import build_recommendation_context, recommend_or_fallback
from .uncertainty import prediction_interval


def save_bundle(bundle: dict, path: str | Path):
    destination = Path(path)
    if destination.suffix != ".joblib":
        raise ValueError("模型产物必须使用 .joblib 扩展名。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination


def load_bundle(path: str | Path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("artifact_version") != "advanced-furnace-1.0":
        raise ValueError("不支持的高级熔炼炉模型包。")
    return bundle


def validate_frame(frame: pd.DataFrame):
    missing = [column for column in FEATURE_COLS if column not in frame]
    extra = [column for column in frame if column not in FEATURE_COLS]
    if missing:
        raise ValueError(f"输入缺少特征: {missing}")
    if extra:
        raise ValueError(f"输入包含额外特征: {extra}")
    numeric = frame[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    invalid = [column for column in FEATURE_COLS if numeric[column].isna().any()]
    if invalid:
        raise ValueError(f"输入必须为完整数值: {invalid}")
    return numeric


def predict_bundle(bundle: dict, frame: pd.DataFrame):
    X = validate_frame(frame)
    prediction = np.asarray(bundle["prediction_model"].predict(X), dtype=float)
    low, high = prediction_interval(prediction, bundle["conformal_radius_90"])
    fold_matrix = np.vstack([model.predict(X) for model in bundle["prediction_fold_models"]])
    return pd.DataFrame({
        "predicted_gas": prediction,
        "prediction_lower_90": low,
        "prediction_upper_90": high,
        "chronological_fold_std": fold_matrix.std(axis=0),
        "model_name": bundle["selected_model_name"],
    })


def recommend_bundle(bundle: dict, total_weight: float, budget: int = 600, seed: int = 42):
    context = build_recommendation_context(
        bundle["training_data"], total_weight, bundle["recommendation_trust_ratio"]
    )
    functions = {
        "random_search": random_search,
        "genetic_algorithm": genetic_search,
        "bayesian_optimization": bayesian_search,
    }
    function = functions[bundle["recommendation_optimizer"]]
    scored = function(
        bundle["recommendation_model"],
        bundle["recommendation_fold_models"],
        context,
        bundle["feasibility_reference"],
        budget=budget,
        seed=seed,
    )
    result = recommend_or_fallback(scored, context)
    result["recommendation_model_name"] = bundle["recommendation_model_name"]
    result["optimizer"] = bundle["recommendation_optimizer"]
    return result
