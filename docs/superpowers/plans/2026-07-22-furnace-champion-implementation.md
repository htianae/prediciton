# Furnace Champion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Excel-only, reproducible training and recommendation system that compares LightGBM, Linear, and Huber offline, selects one safe Champion, and saves one loadable `gas_champion.joblib` for prediction and parameter recommendation.

**Architecture:** Keep the historical notebook as a record. Put reusable logic in a small `furnace_champion` package, expose three thin command-line programs for training, prediction, and recommendation, and create a short notebook that only calls those APIs. Time-series validation, artifact serialization, uncertainty, feasibility scoring, and both optimizers are independently testable.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn 1.6, LightGBM 4.6, joblib, pytest, nbformat/Jupyter.

## Global Constraints

- Use only `4_month_data_2026_02_01_2026_06_25.xlsx`; do not call the remote API.
- Use the six B-version features specified in the approved design.
- Keep all high-gas batches; target-derived anomaly labels are diagnostic only.
- Use five-fold `TimeSeriesSplit`, never random split, for Champion selection.
- Compare LightGBM, Linear, and Huber offline; export only one Champion.
- GA and random search use identical bounds, contexts, scoring, seeds, and 5,000-evaluation budgets.
- Online code loads only `artifacts/gas_champion.joblib` and never retrains.
- The directory is not a Git repository, so commit steps are intentionally omitted.

---

### Task 1: Excel Data Contract

**Files:**
- Create: `furnace_champion/__init__.py`
- Create: `furnace_champion/data.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Produces `FEATURE_COLS`, `TARGET_COL`, `load_batch_data(path) -> pd.DataFrame`, and `mark_target_outliers(df) -> pd.DataFrame`.
- Later tasks consume the returned chronological dataframe with `batch_id`, six features, target, and diagnostic `is_high_gas_outlier`.

- [ ] Write tests asserting 292 unique batches, exact six-feature schema, 14 charge-data missing values, and five retained target outliers.
- [ ] Run `pytest tests/test_data.py -v` and verify failure because `furnace_champion.data` does not exist.
- [ ] Implement Excel transposition, numeric conversion, uniqueness checks, chronological preservation, and diagnostic IQR marking.
- [ ] Run `pytest tests/test_data.py -v` and verify all tests pass.

### Task 2: Time Validation and Model Selection

**Files:**
- Create: `furnace_champion/modeling.py`
- Create: `tests/test_modeling.py`

**Interfaces:**
- Consumes `FEATURE_COLS`, `TARGET_COL`, and chronological dataframe.
- Produces `build_candidate_models()`, `evaluate_models_time_series(df, n_splits=5)`, `eligible_model_names(summary)`, `choose_prediction_champion(summary, safety_results)`, and `fit_champion(df, model_name)`.
- Evaluation returns fold metrics, aggregate metrics, OOF predictions/residuals, and fitted fold models per candidate.

- [ ] Write tests proving every training index precedes every validation index, all three model names participate, outliers remain present, selection score equals `rmse_mean + 0.25 * rmse_std`, 5% eligibility works, and deterministic tie-breaking prefers Linear then Huber then LightGBM.
- [ ] Run `pytest tests/test_modeling.py -v` and verify expected import/function failures.
- [ ] Implement candidate pipelines with fold-local median imputation and scaling where required.
- [ ] Implement five-fold evaluation and deterministic Champion selection helpers.
- [ ] Run `pytest tests/test_modeling.py -v` and verify all tests pass.

### Task 3: Prediction Artifact and Inference

**Files:**
- Create: `furnace_champion/artifact.py`
- Create: `tests/test_artifact.py`

**Interfaces:**
- Consumes final model, fold models, OOF residuals, metrics, dataframe, and optimizer metadata.
- Produces `build_artifact(...) -> dict`, `save_artifact(bundle, path)`, `load_artifact(path)`, `validate_input(bundle, frame)`, and `predict_with_interval(bundle, frame) -> pd.DataFrame`.

- [ ] Write tests for exact feature ordering, missing/extra/non-numeric column errors, P1–P99 OOD warnings, non-negative 90% intervals, and bitwise-close predictions before and after joblib round-trip.
- [ ] Run `pytest tests/test_artifact.py -v` and verify failures before implementation.
- [ ] Implement metadata, dependency versions, data fingerprint, conformal residual quantile, schema validation, prediction, and trusted-path joblib round-trip.
- [ ] Run `pytest tests/test_artifact.py -v` and verify all tests pass.

### Task 4: Shared Feasibility Scoring and Search Space

**Files:**
- Create: `furnace_champion/optimization.py`
- Create: `tests/test_optimization_scoring.py`

**Interfaces:**
- Produces `build_context(df, total_weight, melting_time=None)`, `build_search_space(context_df)`, `fit_feasibility_reference(df)`, `score_candidates(bundle, candidates, feasibility)`, and `historical_fallback(context_df)`.
- `score_candidates` returns predicted gas, fold-model standard deviation, normalized fifth-neighbor distance, feasibility flag, boundary flags, and penalized objective.

- [ ] Write tests proving context expands from ±5% to ±10%, search bounds are P5–P95, door counts are integers, higher uncertainty increases objective, distant joint combinations receive larger penalties, and infeasible-only results trigger historical fallback.
- [ ] Run `pytest tests/test_optimization_scoring.py -v` and verify failures.
- [ ] Implement robust scaling, fifth-neighbor P95 normalization, target-IQR penalty, boundary detection, and fallback behavior.
- [ ] Run `pytest tests/test_optimization_scoring.py -v` and verify all tests pass.

### Task 5: Fair Random Search and Genetic Algorithm

**Files:**
- Modify: `furnace_champion/optimization.py`
- Create: `tests/test_optimizers.py`

**Interfaces:**
- Produces `random_search(...)`, `genetic_search(...)`, `compare_optimizers(...)`, and `summarize_recommendation(...)`.
- Both optimizers consume the same immutable search-space object and scoring callable.

- [ ] Write tests asserting identical bounds, exactly 5,000 evaluated rows, deterministic repeated seeds, integer door counts, and the 0.5% tie rule selecting random search.
- [ ] Run `pytest tests/test_optimizers.py -v` and verify failures.
- [ ] Implement 5,000-candidate random search and 100×50 GA with elite retention, crossover, mutation, clipping, and evaluation accounting.
- [ ] Implement ten-seed optimizer comparison and top-feasible P25–P75 recommendation summary.
- [ ] Run `pytest tests/test_optimizers.py -v` and verify all tests pass.

### Task 6: Three-Model Recommendation Stress Test and Final Champion

**Files:**
- Create: `furnace_champion/training.py`
- Create: `tests/test_training.py`

**Interfaces:**
- Produces `train_and_select(excel_path, output_dir, evaluation_budget=5000, seeds=range(10)) -> dict`.
- Orchestrates prediction validation, five weight contexts, all three model recommendation comparisons, safety gates, final model/optimizer selection, final fit, and artifact build.

- [ ] Write tests using a small deterministic synthetic dataframe to verify all three models receive identical contexts/seeds, only models within 5% prediction score are eligible, safety gates reject low feasibility/high boundary/high uncertainty, and no-safe-model mode records historical fallback.
- [ ] Run `pytest tests/test_training.py -v` and verify failures.
- [ ] Implement the orchestration and structured JSON/CSV report rows.
- [ ] Run `pytest tests/test_training.py -v` and verify all tests pass.

### Task 7: User-Facing Commands and Notebook

**Files:**
- Create: `train_champion.py`
- Create: `predict_champion.py`
- Create: `recommend_champion.py`
- Create: `Nanshang_champion_workflow.ipynb`
- Create: `README.md`
- Create: `requirements.txt`
- Create: `tests/test_cli.py`

**Interfaces:**
- Training command writes `artifacts/gas_champion.joblib` plus reports.
- Prediction command accepts the six final batch values and emits JSON.
- Recommendation command accepts total weight and optional reference melting time and emits JSON.

- [ ] Write CLI tests for help output, invalid input, model loading, and JSON response schema.
- [ ] Run `pytest tests/test_cli.py -v` and verify failures.
- [ ] Implement thin argparse commands that delegate to package functions.
- [ ] Create the notebook with cells for loading Excel, training/comparison, Champion summary, one real-batch prediction, and 86,000 kg recommendation.
- [ ] Document exact training, prediction, and recommendation commands.
- [ ] Run `pytest tests/test_cli.py -v` and verify all tests pass.

### Task 8: Full Verification and Excel Acceptance Run

**Files:**
- Generate: `artifacts/gas_champion.joblib`
- Generate: `reports/model_comparison.csv`
- Generate: `reports/recommendation_stress_test.csv`
- Generate: `reports/champion_summary.json`
- Generate: `reports/recommendation_86000.json`
- Generate: `Nanshang_champion_workflow_executed.ipynb`

**Interfaces:**
- Consumes all prior components and current Excel workbook.
- Produces final user-deliverable artifact and evidence.

- [ ] Run `pytest -q`; expected result is zero failures.
- [ ] Run `python train_champion.py --excel 4_month_data_2026_02_01_2026_06_25.xlsx --output-dir .`.
- [ ] Reload the generated joblib and predict one real Excel batch; compare with its actual gas value and record the interval/OOD result.
- [ ] Run `python recommend_champion.py --model artifacts/gas_champion.joblib --total-weight 86000` and record the selected optimizer, recommendation, uncertainty, feasibility, and boundary flags.
- [ ] Execute `Nanshang_champion_workflow.ipynb` from top to bottom and verify no error outputs.
- [ ] Inspect generated reports for three-model coverage, identical optimizer budgets/bounds, one selected Champion, and one selected optimizer.
