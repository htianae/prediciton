"""Build, persist, validate, and use the single online Champion artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

from .data import FEATURE_COLS, TARGET_COL


def _fingerprint(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def build_artifact(
    df: pd.DataFrame,
    model_name: str,
    model,
    cv_models: list,
    oof_predictions: np.ndarray,
    cv_summary: pd.DataFrame,
    optimizer_name: str | None = None,
    optimizer_config: dict | None = None,
) -> dict:
    valid = np.isfinite(oof_predictions)
    residuals = np.abs(df.loc[valid, TARGET_COL].to_numpy() - oof_predictions[valid])
    ranges = {
        column: {
            f"p{quantile}": float(df[column].quantile(quantile / 100))
            for quantile in (1, 5, 50, 95, 99)
        }
        for column in FEATURE_COLS
    }
    selected = cv_summary.loc[cv_summary["model"] == model_name].iloc[0]
    return {
        "artifact_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model": model,
        "cv_models": cv_models,
        "feature_cols": list(FEATURE_COLS),
        "target_col": TARGET_COL,
        "training_batches": int(len(df)),
        "training_data_fingerprint": _fingerprint(df[["batch_id"] + FEATURE_COLS + [TARGET_COL]]),
        "cv_metrics": cv_summary.to_dict(orient="records"),
        "selection_score": float(selected["selection_score"]),
        "conformal_abs_error_q90": float(np.quantile(residuals, 0.90)),
        "feature_ranges": ranges,
        "target_iqr": float(df[TARGET_COL].quantile(0.75) - df[TARGET_COL].quantile(0.25)),
        "training_reference": df[["batch_id"] + FEATURE_COLS + [TARGET_COL]].copy(),
        "optimizer_name": optimizer_name,
        "optimizer_config": optimizer_config or {},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }


def save_artifact(bundle: dict, path: str | Path) -> Path:
    destination = Path(path)
    if destination.suffix != ".joblib":
        raise ValueError("模型文件必须使用 .joblib 扩展名。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination


def load_artifact(path: str | Path) -> dict:
    source = Path(path)
    if source.suffix != ".joblib":
        raise ValueError("只允许加载可信的 .joblib 模型文件。")
    bundle = joblib.load(source)
    if not isinstance(bundle, dict) or bundle.get("artifact_version") != "1.0":
        raise ValueError("不支持的 Champion artifact。")
    return bundle


def validate_input(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    expected = list(bundle["feature_cols"])
    missing = [column for column in expected if column not in frame.columns]
    extra = [column for column in frame.columns if column not in expected]
    if missing:
        raise ValueError(f"输入缺少特征: {missing}")
    if extra:
        raise ValueError(f"输入包含额外特征: {extra}")
    numeric = frame[expected].apply(pd.to_numeric, errors="coerce")
    invalid = [column for column in expected if numeric[column].isna().any()]
    if invalid:
        raise ValueError(f"输入必须是完整数值，异常列: {invalid}")
    return numeric


def predict_with_interval(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    X = validate_input(bundle, frame)
    prediction = np.asarray(bundle["model"].predict(X), dtype=float)
    fold_predictions = np.vstack([model.predict(X) for model in bundle["cv_models"]])
    uncertainty = fold_predictions.std(axis=0)
    radius = float(bundle["conformal_abs_error_q90"])
    ood_features: list[str] = []
    row_ood: list[list[str]] = []
    for _, row in X.iterrows():
        flagged = []
        for column in bundle["feature_cols"]:
            bounds = bundle["feature_ranges"][column]
            if row[column] < bounds["p1"] or row[column] > bounds["p99"]:
                flagged.append(column)
        row_ood.append(flagged)
    return pd.DataFrame(
        {
            "predicted_gas": prediction,
            "prediction_lower_90": np.maximum(0, prediction - radius),
            "prediction_upper_90": prediction + radius,
            "ensemble_std": uncertainty,
            "is_ood": [bool(items) for items in row_ood],
            "ood_features": [", ".join(items) for items in row_ood],
        }
    )
