"""Small-data candidate regressors, target routes, residuals, and ensembles."""

from copy import deepcopy

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, SplineTransformer, StandardScaler

from .data import FEATURE_COLS, WEIGHT_COL
from .features import FurnaceFeatureEngineer, prediction_to_total_gas


def _imputer():
    return SimpleImputer(strategy="median").set_output(transform="pandas")


def _numeric_pipeline(model, scale="standard"):
    steps = [("features", FurnaceFeatureEngineer()), ("imputer", _imputer())]
    if scale == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scale == "robust":
        steps.append(("scaler", RobustScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


class ResidualBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_model, residual_model):
        self.base_model = base_model
        self.residual_model = residual_model

    def fit(self, X, y):
        self.base_model_ = clone(self.base_model).fit(X, y)
        residual = np.asarray(y, dtype=float) - self.base_model_.predict(X)
        self.residual_model_ = clone(self.residual_model).fit(X, residual)
        return self

    def predict(self, X):
        return self.base_model_.predict(X) + self.residual_model_.predict(X)


class RouteRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, route="direct"):
        self.model = model
        self.route = route

    def fit(self, X, y):
        frame = pd.DataFrame(X).copy()
        self.weight_median_ = float(pd.to_numeric(frame[WEIGHT_COL], errors="coerce").median())
        target = np.asarray(y, dtype=float)
        if self.route == "unit":
            tonnes = pd.to_numeric(frame[WEIGHT_COL], errors="coerce").fillna(self.weight_median_).to_numpy() / 1000.0
            target = target / tonnes
        elif self.route != "direct":
            raise ValueError(f"未知目标路线: {self.route}")
        self.model_ = clone(self.model).fit(frame[FEATURE_COLS], target)
        return self

    def predict(self, X):
        frame = pd.DataFrame(X).copy()[FEATURE_COLS]
        prediction = self.model_.predict(frame)
        return prediction_to_total_gas(prediction, frame, self.route, self.weight_median_)

    def predict_std(self, X):
        frame = pd.DataFrame(X).copy()[FEATURE_COLS]
        if not isinstance(self.model_, Pipeline):
            raise TypeError("底层模型不支持原生预测标准差。")
        transformed = self.model_[:-1].transform(frame)
        final_model = self.model_.steps[-1][1]
        if not isinstance(final_model, GaussianProcessRegressor):
            raise TypeError("底层模型不支持原生预测标准差。")
        _, standard_deviation = final_model.predict(transformed, return_std=True)
        return prediction_to_total_gas(
            standard_deviation, frame, self.route, self.weight_median_
        )


class WeightedOOFEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def fit(self, X, y):
        self.models_ = [clone(model).fit(X, y) for model in self.models]
        return self

    def predict(self, X):
        members = getattr(self, "models_", self.models)
        matrix = np.column_stack([model.predict(X) for model in members])
        return matrix @ np.asarray(self.weights, dtype=float)


def fit_nonnegative_ensemble_weights(oof_matrix, y):
    matrix = np.asarray(oof_matrix, dtype=float)
    target = np.asarray(y, dtype=float)
    count = matrix.shape[1]
    result = minimize(
        lambda weights: np.mean((matrix @ weights - target) ** 2),
        np.repeat(1.0 / count, count),
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        method="SLSQP",
    )
    if not result.success:
        return np.repeat(1.0 / count, count)
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()


def build_base_models(seed: int = 42):
    ridge = _numeric_pipeline(Ridge(alpha=10.0))
    huber = _numeric_pipeline(HuberRegressor(epsilon=1.5, max_iter=3000), scale="robust")
    lgbm = _numeric_pipeline(
        LGBMRegressor(
            n_estimators=350, learning_rate=0.03, num_leaves=7, max_depth=3,
            min_child_samples=25, max_bin=31, reg_alpha=1.0, reg_lambda=10.0,
            subsample=0.85, colsample_bytree=0.85, random_state=seed,
            n_jobs=-1, verbose=-1,
        ),
        scale=None,
    )
    models = {
        "Ridge": ridge,
        "ElasticNet": _numeric_pipeline(ElasticNet(alpha=0.05, l1_ratio=0.25, max_iter=10000)),
        "Huber": huber,
        "GAM": Pipeline([
            ("features", FurnaceFeatureEngineer()),
            ("imputer", _imputer()),
            ("splines", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "GPR": _numeric_pipeline(
            GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.2),
                alpha=1e-6, normalize_y=True, n_restarts_optimizer=0, random_state=seed,
            )
        ),
        "CatBoost": _numeric_pipeline(
            CatBoostRegressor(
                iterations=300, depth=4, learning_rate=0.03, loss_function="RMSE",
                l2_leaf_reg=8.0, random_seed=seed, verbose=False, allow_writing_files=False,
            ),
            scale=None,
        ),
        "LightGBM": lgbm,
    }
    models["Ridge+LGBMResidual"] = ResidualBoostRegressor(ridge, lgbm)
    models["Huber+LGBMResidual"] = ResidualBoostRegressor(huber, lgbm)
    return models


def build_tree_model_variants(seed: int = 42):
    base = build_base_models(seed)
    lgbm_variants = {}
    for label, params in {
        "very_small": {"num_leaves": 4, "max_depth": 2, "min_child_samples": 30, "reg_lambda": 15.0},
        "small": {"num_leaves": 7, "max_depth": 3, "min_child_samples": 25, "reg_lambda": 10.0},
        "medium_small": {"num_leaves": 10, "max_depth": 4, "min_child_samples": 20, "reg_lambda": 8.0},
    }.items():
        lgbm_variants[label] = clone(base["LightGBM"]).set_params(
            **{f"model__{key}": value for key, value in params.items()}
        )
    cat_variants = {}
    for label, params in {
        "depth3": {"depth": 3, "l2_leaf_reg": 10.0},
        "depth4": {"depth": 4, "l2_leaf_reg": 8.0},
    }.items():
        cat_variants[label] = clone(base["CatBoost"]).set_params(
            **{f"model__{key}": value for key, value in params.items()}
        )
    return {"LightGBM": lgbm_variants, "CatBoost": cat_variants}
