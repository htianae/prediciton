"""Fair optimizer comparison with uncertainty and historical-feasibility penalties."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from .data import FEATURE_COLS, TARGET_COL


WEIGHT_COL, SOLID_COL, MELTING_COL, WAIT_COL, DOOR_COUNT_COL, DOOR_DURATION_COL = FEATURE_COLS
CONTROLLABLE_COLS = [SOLID_COL, WAIT_COL, DOOR_COUNT_COL, DOOR_DURATION_COL]


@dataclass(frozen=True)
class SearchSpace:
    bounds: dict[str, tuple[float, float]]


def build_context(df: pd.DataFrame, total_weight: float, melting_time: float | None = None) -> dict:
    selected = pd.DataFrame()
    tolerance_used: float | None = None
    for tolerance in (0.05, 0.10):
        selected = df[df[WEIGHT_COL].between(total_weight * (1 - tolerance), total_weight * (1 + tolerance))]
        if len(selected) >= 20:
            tolerance_used = tolerance
            break
    low_confidence = False
    if len(selected) < 20:
        selected = df.copy()
        tolerance_used = None
        low_confidence = True
    if melting_time is None:
        melting_time = float(selected[MELTING_COL].median())
        melting_source = "similar_batch_median"
    else:
        melting_time = float(melting_time)
        melting_source = "user_input"
    return {
        "batches": selected.copy(),
        "total_weight": float(total_weight),
        "melting_time": melting_time,
        "melting_time_source": melting_source,
        "weight_tolerance": tolerance_used,
        "low_confidence": low_confidence,
    }


def build_search_space(context_df: pd.DataFrame) -> SearchSpace:
    bounds = {
        column: (float(context_df[column].quantile(0.05)), float(context_df[column].quantile(0.95)))
        for column in CONTROLLABLE_COLS
    }
    return SearchSpace(bounds=bounds)


def fit_feasibility_reference(df: pd.DataFrame) -> dict:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    imputed = imputer.fit_transform(df[FEATURE_COLS])
    scaled = scaler.fit_transform(imputed)
    neighbors = min(6, len(df))
    nn = NearestNeighbors(n_neighbors=neighbors).fit(scaled)
    distances = nn.kneighbors(scaled, return_distance=True)[0]
    reference_index = min(5, distances.shape[1] - 1)
    threshold = float(np.quantile(distances[:, reference_index], 0.95))
    return {
        "imputer": imputer,
        "scaler": scaler,
        "neighbors": nn,
        "distance_threshold": max(threshold, 1e-12),
        "reference_rows": int(len(df)),
    }


def score_candidates(bundle: dict, candidates: pd.DataFrame, feasibility: dict, search_space: SearchSpace | None = None) -> pd.DataFrame:
    X = candidates[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("候选参数包含缺失或非数值。")
    result = candidates.reset_index(drop=True).copy()
    result["predicted_gas"] = np.asarray(bundle["model"].predict(X), dtype=float)
    fold_predictions = np.vstack([model.predict(X) for model in bundle["cv_models"]])
    result["prediction_std"] = fold_predictions.std(axis=0)

    scaled = feasibility["scaler"].transform(feasibility["imputer"].transform(X))
    n_neighbors = min(5, feasibility["reference_rows"])
    distance = feasibility["neighbors"].kneighbors(scaled, n_neighbors=n_neighbors, return_distance=True)[0][:, -1]
    normalized = distance / feasibility["distance_threshold"]
    result["normalized_knn_distance"] = normalized
    result["is_feasible"] = normalized <= 1.0
    result["penalized_objective"] = (
        result["predicted_gas"]
        + result["prediction_std"]
        + float(bundle["target_iqr"]) * np.maximum(0, normalized - 1)
    )

    result["boundary_hit_count"] = 0
    if search_space is not None:
        for column, (low, high) in search_space.bounds.items():
            width = max(high - low, 1e-12)
            hit = (result[column] - low <= 0.02 * width) | (high - result[column] <= 0.02 * width)
            result[f"boundary__{column}"] = hit
            result["boundary_hit_count"] += hit.astype(int)
    result["any_boundary_hit"] = result["boundary_hit_count"] > 0
    return result


def historical_fallback(context_df: pd.DataFrame) -> dict:
    count = max(1, int(math.ceil(len(context_df) * 0.20)))
    low_gas = context_df.sort_values(TARGET_COL).head(count)
    return {
        "source": "historical_similar_low_gas",
        "recommendation": {column: float(low_gas[column].median()) for column in CONTROLLABLE_COLS},
        "reference_batches": int(len(context_df)),
        "low_gas_batches": int(len(low_gas)),
        "actual_gas_median": float(low_gas[TARGET_COL].median()),
    }


def _sample_population(context: dict, space: SearchSpace, size: int, rng: np.random.Generator) -> pd.DataFrame:
    frame = pd.DataFrame(index=range(size))
    frame[WEIGHT_COL] = context["total_weight"]
    frame[MELTING_COL] = context["melting_time"]
    for column, (low, high) in space.bounds.items():
        if column == DOOR_COUNT_COL:
            integer_low = math.ceil(low)
            integer_high = math.floor(high)
            frame[column] = rng.integers(integer_low, integer_high + 1, size=size)
        else:
            frame[column] = rng.uniform(low, high, size=size)
    return frame[FEATURE_COLS]


def random_search(bundle: dict, context: dict, space: SearchSpace, feasibility: dict, budget: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    result = score_candidates(bundle, _sample_population(context, space, budget, rng), feasibility, space)
    result.attrs["search_bounds"] = dict(space.bounds)
    result.attrs["evaluations"] = int(budget)
    result.attrs["seed"] = int(seed)
    result.attrs["optimizer"] = "random_search"
    return result


def genetic_search(bundle: dict, context: dict, space: SearchSpace, feasibility: dict, budget: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    population_size = min(100, budget)
    generations = math.ceil(budget / population_size)
    population = _sample_population(context, space, population_size, rng)
    evaluated_parts: list[pd.DataFrame] = []

    for generation in range(generations):
        remaining = budget - sum(len(part) for part in evaluated_parts)
        if remaining <= 0:
            break
        current = population.iloc[:remaining].copy()
        scored = score_candidates(bundle, current, feasibility, space)
        scored["generation"] = generation
        evaluated_parts.append(scored)
        if remaining < population_size:
            break

        ranked = scored.sort_values(["is_feasible", "penalized_objective"], ascending=[False, True])
        elite_count = max(2, int(population_size * 0.20))
        elites = ranked.head(elite_count)[CONTROLLABLE_COLS].reset_index(drop=True)
        next_controls = [row.to_dict() for _, row in elites.iterrows()]
        while len(next_controls) < population_size:
            p1 = elites.iloc[int(rng.integers(0, len(elites)))]
            p2 = elites.iloc[int(rng.integers(0, len(elites)))]
            child = {}
            for column, (low, high) in space.bounds.items():
                if column == DOOR_COUNT_COL:
                    value = p1[column] if rng.random() < 0.5 else p2[column]
                    if rng.random() < 0.15:
                        value += rng.choice([-1, 1])
                    child[column] = int(np.clip(round(value), math.ceil(low), math.floor(high)))
                else:
                    alpha = rng.random()
                    value = alpha * p1[column] + (1 - alpha) * p2[column]
                    if rng.random() < 0.15:
                        value += rng.normal(0, 0.10 * (high - low))
                    child[column] = float(np.clip(value, low, high))
            next_controls.append(child)
        population = pd.DataFrame(next_controls)
        population[WEIGHT_COL] = context["total_weight"]
        population[MELTING_COL] = context["melting_time"]
        population = population[FEATURE_COLS]

    result = pd.concat(evaluated_parts, ignore_index=True).iloc[:budget].copy()
    result.attrs["search_bounds"] = dict(space.bounds)
    result.attrs["evaluations"] = int(len(result))
    result.attrs["seed"] = int(seed)
    result.attrs["optimizer"] = "genetic_algorithm"
    return result


def _best_objective(result: pd.DataFrame) -> float:
    feasible = result[result["is_feasible"]]
    pool = feasible if not feasible.empty else result
    return float(pool["penalized_objective"].min())


def compare_optimizers(
    bundle: dict,
    context: dict,
    space: SearchSpace,
    feasibility: dict,
    seeds=range(10),
    budget: int = 5000,
    tie_ratio: float = 0.005,
    keep_results: bool = True,
) -> dict:
    rows = []
    results: dict[tuple[str, int], pd.DataFrame] = {}
    best_rows = []
    for seed in seeds:
        for name, function in (("random_search", random_search), ("genetic_algorithm", genetic_search)):
            result = function(bundle, context, space, feasibility, budget=budget, seed=int(seed))
            if keep_results:
                results[(name, int(seed))] = result
            feasible = result[result["is_feasible"]]
            pool = feasible if not feasible.empty else result
            best = pool.sort_values("penalized_objective").iloc[0].to_dict()
            best.update(
                {
                    "optimizer": name,
                    "seed": int(seed),
                    "scenario_has_feasible": bool(not feasible.empty),
                }
            )
            best_rows.append(best)
            rows.append(
                {
                    "optimizer": name,
                    "seed": int(seed),
                    "best_objective": _best_objective(result),
                    "feasible_rate": float(result["is_feasible"].mean()),
                    "unique_candidates": int(result[CONTROLLABLE_COLS].drop_duplicates().shape[0]),
                }
            )
    runs = pd.DataFrame(rows)
    medians = runs.groupby("optimizer")["best_objective"].median()
    random_value = float(medians["random_search"])
    ga_value = float(medians["genetic_algorithm"])
    relative = abs(random_value - ga_value) / max(abs(random_value), 1e-12)
    if relative <= tie_ratio:
        selected = "random_search"
    else:
        selected = str(medians.idxmin())
    response = {
        "optimizer_name": selected,
        "runs": runs,
        "best_candidates": pd.DataFrame(best_rows),
        "median_objectives": medians.to_dict(),
    }
    if keep_results:
        response["results"] = results
    return response


def summarize_recommendation(result: pd.DataFrame, context: dict, top_n: int = 50) -> dict:
    feasible = result[result["is_feasible"]].sort_values("penalized_objective")
    if feasible.empty:
        return {"source": "no_feasible_model_candidate"}
    top = feasible.head(top_n)
    best = top.iloc[0]
    return {
        "source": "model_optimization",
        "recommendation": {column: float(top[column].median()) for column in CONTROLLABLE_COLS},
        "recommended_ranges_p25_p75": {
            column: [float(top[column].quantile(0.25)), float(top[column].quantile(0.75))]
            for column in CONTROLLABLE_COLS
        },
        "predicted_gas": float(best["predicted_gas"]),
        "prediction_std": float(best["prediction_std"]),
        "penalized_objective": float(best["penalized_objective"]),
        "normalized_knn_distance": float(best["normalized_knn_distance"]),
        "boundary_hit": bool(best["any_boundary_hit"]),
        "reference_melting_time": float(context["melting_time"]),
        "melting_time_source": context["melting_time_source"],
    }
