# Advanced Furnace ML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated advanced offline ML project that compares small-data regression strategies and produces only recommendations whose conservative predicted gas is lower than a historical actual-gas baseline and which pass explicit safety gates.

**Architecture:** A `src` package owns reusable data, validation, model, uncertainty, optimization, recommendation, artifact, and orchestration logic. A single experiment pipeline freezes all choices on the first 248 chronological batches, audits them once on the final 44 locked batches, retrains production artifacts on all 292 batches, and creates an executed notebook and CSV/JSON reports.

**Tech Stack:** Python 3.11, pandas 2.3, NumPy 1.26, scikit-learn 1.6, SciPy 1.15, LightGBM 4.6, CatBoost 1.2, joblib 1.4, matplotlib, Jupyter.

## Global Constraints

- All implementation outputs live under `advanced_furnace_ml/`.
- Existing production files are read-only and their SHA-256 checksums must remain unchanged.
- Use the first 248 batches for development and the final 44 batches as a locked audit set.
- Use exactly three large chronological folds: 148/33, 181/33, and 214/34 train/validation sizes.
- A recommendation passes only when its conservative prediction is below the similar-history actual median, at least two of three fold models agree it saves gas, it is historically feasible, it is not boundary-seeking, and it stays inside the trust region.
- Offline savings are estimates, not claims of realized factory savings.

---

### Task 1: Project skeleton, data contract, and feature engineering

**Files:**
- Create: `advanced_furnace_ml/src/advanced_furnace_ml/__init__.py`
- Create: `advanced_furnace_ml/src/advanced_furnace_ml/data.py`
- Create: `advanced_furnace_ml/src/advanced_furnace_ml/features.py`
- Create: `advanced_furnace_ml/tests/test_data_features.py`
- Create: `advanced_furnace_ml/configs/experiment.json`

**Interfaces:**
- `load_batch_data(path: Path) -> pd.DataFrame`
- `mark_target_outliers(df: pd.DataFrame) -> pd.DataFrame`
- `chronological_dev_lock_split(df, lock_size=44) -> tuple[pd.DataFrame, pd.DataFrame]`
- `FurnaceFeatureEngineer.transform(X: pd.DataFrame) -> pd.DataFrame`
- `target_for_route(df, route) -> pd.Series`
- `prediction_to_total_gas(prediction, X, route) -> np.ndarray`

- [ ] Write tests asserting 292 chronological unique batches, a 248/44 split, retained outliers, finite engineered ratios, deterministic feature order, and correct unit-gas round-trip.
- [ ] Run `PYTHONPATH=advanced_furnace_ml/src python -m unittest discover -s advanced_furnace_ml/tests -p 'test_data_features.py' -v` and confirm failure.
- [ ] Implement the minimal data and feature modules.
- [ ] Rerun the test and confirm pass.

### Task 2: Three large chronological folds and locked-audit guards

**Files:**
- Create: `advanced_furnace_ml/src/advanced_furnace_ml/validation.py`
- Create: `advanced_furnace_ml/tests/test_validation.py`

**Interfaces:**
- `large_window_splits(n_dev=248) -> list[tuple[np.ndarray, np.ndarray]]`
- `regression_metrics(y_true, y_pred) -> dict`
- `bootstrap_metric_interval(y_true, y_pred, metric, seed, n_boot=1000) -> tuple[float, float]`
- `LockedAudit` object that raises if accessed before `freeze()`.

- [ ] Write tests for exact fold sizes/boundaries, no future leakage, metric values, deterministic bootstrap intervals, and locked-audit access rejection.
- [ ] Verify red, implement, and verify green.

### Task 3: Full model matrix and direct/unit-gas routes

**Files:**
- Create: `advanced_furnace_ml/src/advanced_furnace_ml/models.py`
- Create: `advanced_furnace_ml/tests/test_models.py`

**Interfaces:**
- `build_base_models(seed=42) -> dict[str, RegressorMixin]`
- `ResidualBoostRegressor(base_model, residual_model)`
- `RouteRegressor(model, route)` exposing sklearn-style `fit/predict`
- `WeightedOOFEnsemble(models, weights)`
- `fit_nonnegative_ensemble_weights(oof_matrix, y) -> np.ndarray`

- [ ] Test that model names include Ridge, ElasticNet, Huber, GAM, GPR, CatBoost, LightGBM, Ridge+LGBM residual, and Huber+LGBM residual; route predictions return total gas; residual and weighted ensembles clone/serialize; weights are nonnegative and sum to one.
- [ ] Verify red, implement conservative small-data models, and verify green.

### Task 4: Chronological model experiment and uncertainty

**Files:**
- Create: `advanced_furnace_ml/src/advanced_furnace_ml/uncertainty.py`
- Create: `advanced_furnace_ml/src/advanced_furnace_ml/experiment.py`
- Create: `advanced_furnace_ml/tests/test_experiment.py`

**Interfaces:**
- `evaluate_candidate(name, estimator, route, dev_df, splits) -> CandidateResult`
- `run_model_matrix(dev_df, splits) -> ModelExperiment`
- `conformal_radius(y_true, oof_pred, coverage=.90) -> float`
- `prediction_interval(prediction, radius) -> tuple[np.ndarray, np.ndarray]`
- `audit_frozen_models(experiment, locked_df) -> pd.DataFrame`

- [ ] Test on a reduced fixture that every candidate uses identical folds, OOF exists only on validation rows, the ensemble is learned only from dev OOF, conformal calibration excludes lock rows, and locked audit cannot mutate selection.
- [ ] Verify red, implement, and verify green.

### Task 5: Safe recommendation and three optimizers

**Files:**
- Create: `advanced_furnace_ml/src/advanced_furnace_ml/optimization.py`
- Create: `advanced_furnace_ml/src/advanced_furnace_ml/recommendation.py`
- Create: `advanced_furnace_ml/tests/test_recommendation.py`

**Interfaces:**
- `build_recommendation_context(df, total_weight, trust_ratio=.10) -> RecommendationContext`
- `score_candidates(candidate_model, fold_models, candidates, context, feasibility) -> pd.DataFrame`
- `random_search(..., budget, seed) -> pd.DataFrame`
- `genetic_search(..., budget, seed) -> pd.DataFrame`
- `bayesian_search(..., budget, seed) -> pd.DataFrame`
- `compare_optimizers(..., common_budget, seeds) -> dict`
- `safety_gate(scored_recommendation, context) -> SafetyDecision`

- [ ] Test identical bounds/budgets/seeds, deterministic optimizers, strict integer bounds, trust-region compliance, historical feasibility, boundary rejection, 2/3 consensus, conservative predicted gas below actual baseline, and fallback when no candidate passes.
- [ ] Verify red, implement all three optimizers, and verify green.

### Task 6: Pipeline, joblib, CLI, and reports

**Files:**
- Create: `advanced_furnace_ml/src/advanced_furnace_ml/artifacts.py`
- Create: `advanced_furnace_ml/src/advanced_furnace_ml/pipeline.py`
- Create: `advanced_furnace_ml/train_advanced.py`
- Create: `advanced_furnace_ml/predict_advanced.py`
- Create: `advanced_furnace_ml/recommend_advanced.py`
- Create: `advanced_furnace_ml/tests/test_pipeline_artifact.py`

**Interfaces:**
- `run_advanced_pipeline(excel_path, output_dir, optimizer_budget, seeds) -> dict`
- `save_bundle/load_bundle`
- `predict_bundle(bundle, frame) -> pd.DataFrame`
- `recommend_bundle(bundle, total_weight) -> dict`

- [ ] Test a small-budget end-to-end run, required reports, locked audit row count, model round-trip equality, schema rejection, recommendation/fallback structure, and no writes outside the advanced project.
- [ ] Verify red, implement, and verify green.
- [ ] Run the formal experiment with configured production budget and save artifact/reports.

### Task 7: Irregular realtime and intervention data contracts

**Files:**
- Create: `advanced_furnace_ml/contracts/irregular_realtime_event_schema.csv`
- Create: `advanced_furnace_ml/contracts/intervention_trial_schema.csv`
- Create: `advanced_furnace_ml/contracts/future_data_guide.md`
- Create: `advanced_furnace_ml/tests/test_contracts.py`

- [ ] Test required headers for event time, measured time, entered time, missingness, correction version, intervention recommendation/actual values, reason, operator, quality, temperature, production, safety, and final gas.
- [ ] Verify red, add templates and guide, and verify green.

### Task 8: Notebook, documentation, and final verification

**Files:**
- Create: `advanced_furnace_ml/notebooks/advanced_furnace_experiment.ipynb`
- Create: `advanced_furnace_ml/notebooks/advanced_furnace_experiment_executed.ipynb`
- Create: `advanced_furnace_ml/README.md`
- Create: `advanced_furnace_ml/requirements.txt`
- Create: `advanced_furnace_ml/tests/test_notebook.py`

- [ ] Test notebook structure for data summary, 3-fold diagram/table, model matrix, lock audit, uncertainty, optimizer comparison, raw recommendations, safety decision, historical fallback, and offline-savings disclaimer.
- [ ] Verify red, build the notebook, and verify green.
- [ ] Execute the notebook from top to bottom with a 900-second timeout.
- [ ] Scan all outputs for errors/warnings and verify every code cell has an execution count.
- [ ] Run `PYTHONPATH=advanced_furnace_ml/src python -m unittest discover -s advanced_furnace_ml/tests -p 'test_*.py' -v`.
- [ ] Reload the joblib and reproduce predictions.
- [ ] Compare protected-file SHA-256 hashes with the recorded values.
- [ ] Summarize whether any model recommendation actually passes all offline safety gates; never force a pass.

Because the workspace is not a Git repository, commit, branch, merge, and pull-request steps do not apply.
