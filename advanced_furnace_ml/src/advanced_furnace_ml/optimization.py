"""Budget-matched random, genetic, and GPR expected-improvement search."""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from .data import DOOR_COUNT_COL, FEATURE_COLS, MELTING_COL, WEIGHT_COL
from .recommendation import CONTROLLABLE_COLS, score_candidates


def _sample(context, size: int, rng: np.random.Generator):
    frame = pd.DataFrame(index=range(size))
    frame[WEIGHT_COL] = context.total_weight
    frame[MELTING_COL] = context.melting_time
    for column, (low, high) in context.bounds.items():
        if column == DOOR_COUNT_COL:
            frame[column] = rng.integers(math.ceil(low), math.floor(high) + 1, size=size)
        else:
            frame[column] = rng.uniform(low, high, size=size)
    return frame[FEATURE_COLS]


def _set_attrs(result, name, budget, seed, context):
    result.attrs.update({
        "optimizer": name,
        "evaluations": len(result),
        "budget": int(budget),
        "seed": int(seed),
        "bounds": dict(context.bounds),
    })
    return result


def random_search(model, fold_models, context, feasibility, budget=600, seed=42):
    rng = np.random.default_rng(seed)
    result = score_candidates(model, fold_models, _sample(context, budget, rng), context, feasibility)
    return _set_attrs(result, "random_search", budget, seed, context)


def genetic_search(model, fold_models, context, feasibility, budget=600, seed=42):
    rng = np.random.default_rng(seed)
    population_size = min(60, budget)
    population = _sample(context, population_size, rng)
    parts = []
    while sum(len(part) for part in parts) < budget:
        remaining = budget - sum(len(part) for part in parts)
        scored = score_candidates(model, fold_models, population.iloc[:remaining], context, feasibility)
        parts.append(scored)
        if remaining <= population_size and remaining < len(population):
            break
        elite_count = max(4, population_size // 5)
        elites = scored.nsmallest(elite_count, "penalized_objective")[CONTROLLABLE_COLS].reset_index(drop=True)
        children = [row.to_dict() for _, row in elites.iterrows()]
        while len(children) < population_size:
            a = elites.iloc[int(rng.integers(len(elites)))]
            b = elites.iloc[int(rng.integers(len(elites)))]
            child = {}
            for column, (low, high) in context.bounds.items():
                if column == DOOR_COUNT_COL:
                    value = a[column] if rng.random() < .5 else b[column]
                    if rng.random() < .20:
                        value += rng.choice([-1, 1])
                    child[column] = int(np.clip(round(value), math.ceil(low), math.floor(high)))
                else:
                    alpha = rng.random()
                    value = alpha * a[column] + (1 - alpha) * b[column]
                    if rng.random() < .20:
                        value += rng.normal(0, .08 * (high - low))
                    child[column] = float(np.clip(value, low, high))
            children.append(child)
        population = pd.DataFrame(children)
        population[WEIGHT_COL] = context.total_weight
        population[MELTING_COL] = context.melting_time
        population = population[FEATURE_COLS]
    result = pd.concat(parts, ignore_index=True).iloc[:budget].copy()
    return _set_attrs(result, "genetic_algorithm", budget, seed, context)


def _normalize_controls(frame, context):
    values = []
    for column in CONTROLLABLE_COLS:
        low, high = context.bounds[column]
        values.append((frame[column].to_numpy(dtype=float) - low) / max(high - low, 1e-12))
    return np.column_stack(values)


def bayesian_search(model, fold_models, context, feasibility, budget=600, seed=42):
    rng = np.random.default_rng(seed)
    initial_size = min(40, budget)
    evaluated = score_candidates(model, fold_models, _sample(context, initial_size, rng), context, feasibility)
    while len(evaluated) < budget:
        remaining = budget - len(evaluated)
        batch_size = min(20, remaining)
        train = evaluated.nsmallest(min(250, len(evaluated)), "penalized_objective")
        X_train = _normalize_controls(train, context)
        y_train = train["penalized_objective"].to_numpy(dtype=float)
        y_mean, y_std = y_train.mean(), max(y_train.std(), 1e-9)
        gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=np.ones(4), nu=1.5) + WhiteKernel(noise_level=.05),
            alpha=1e-6,
            normalize_y=False,
            optimizer=None,
            random_state=seed,
        ).fit(X_train, (y_train - y_mean) / y_std)
        pool = _sample(context, max(500, batch_size * 20), rng)
        mean, std = gp.predict(_normalize_controls(pool, context), return_std=True)
        best = float(((y_train - y_mean) / y_std).min())
        improvement = best - mean
        z = improvement / np.maximum(std, 1e-12)
        expected_improvement = improvement * norm.cdf(z) + std * norm.pdf(z)
        selected = pool.iloc[np.argsort(expected_improvement)[-batch_size:]]
        scored = score_candidates(model, fold_models, selected, context, feasibility)
        evaluated = pd.concat([evaluated, scored], ignore_index=True)
    result = evaluated.iloc[:budget].copy()
    return _set_attrs(result, "bayesian_optimization", budget, seed, context)


def compare_optimizers(model, fold_models, context, feasibility, common_budget=600, seeds=(0, 1, 2), keep_results=True):
    functions = (
        ("random_search", random_search),
        ("genetic_algorithm", genetic_search),
        ("bayesian_optimization", bayesian_search),
    )
    rows = []
    results = {}
    for seed in seeds:
        for name, function in functions:
            result = function(model, fold_models, context, feasibility, budget=common_budget, seed=int(seed))
            if keep_results:
                results[(name, int(seed))] = result
            best = result.nsmallest(1, "penalized_objective").iloc[0]
            rows.append({
                "optimizer": name,
                "seed": int(seed),
                "evaluations": len(result),
                "best_objective": float(best["penalized_objective"]),
                "safe_candidate_rate": float((
                    (result["estimated_saving_vs_actual_baseline"] > 0)
                    & (result["fold_consensus_rate"] >= 2/3)
                    & result["historically_feasible"]
                    & ~result["boundary_hit"]
                ).mean()),
                **{f"best__{column}": float(best[column]) for column in CONTROLLABLE_COLS},
            })
    runs = pd.DataFrame(rows)
    selected = str(runs.groupby("optimizer")["best_objective"].median().idxmin())
    response = {"runs": runs, "selected_optimizer": selected}
    if keep_results:
        response["results"] = results
    return response
