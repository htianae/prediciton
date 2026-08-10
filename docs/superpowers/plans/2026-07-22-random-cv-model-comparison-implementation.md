# Random Holdout and Cross-Validation Model Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and execute a self-contained notebook that keeps all 292 batches, selects one prediction Champion through five-fold CV inside a random 80% training split, audits it on the untouched 20% test split, retrains it on all data, and saves one joblib artifact.

**Architecture:** A deterministic builder embeds the existing tested model runtime as notebook source text under a renamed `random_cv_runtime` package, so notebook execution reads only Excel. The notebook uses `train_test_split` for a 233/59 holdout, `KFold` for training-only model selection, the same 19-candidate matrix for comparison, and an isolated artifact/report directory.

**Tech Stack:** Python 3.11, Jupyter/nbformat/nbconvert, NumPy, pandas, scikit-learn, SciPy, LightGBM, CatBoost, joblib, openpyxl, unittest.

## Global Constraints

- Keep every one of the 292 Excel batches; mark target IQR outliers but do not remove or downweight them.
- Use `train_test_split(test_size=0.20, random_state=42)` for exactly 233 training and 59 test rows.
- Use `KFold(n_splits=5, shuffle=True, random_state=42)` only inside the training split.
- Compare the same 19 direct/unit/OOF prediction candidates.
- Select Champion by mean CV RMSE plus `0.25 * CV RMSE standard deviation`, with mean CV MAE and name as deterministic tie-breakers.
- Do not use test targets for tuning, OOF weights, ranking, or Champion selection.
- Do not perform parameter recommendation, optimization, baseline comparison, or safety-gate evaluation.
- Write all new outputs under `advanced_furnace_ml/random_cv_output/` and preserve all existing artifacts/notebooks.
- The workspace is not a Git repository, so commit steps are not applicable.

---

### Task 1: Establish the notebook contract and self-contained skeleton

**Files:**
- Create: `advanced_furnace_ml/tests/test_random_cv_notebook.py`
- Create: `advanced_furnace_ml/tools/build_random_cv_notebook.py`
- Create: `advanced_furnace_ml/notebooks/random_cv_model_comparison.ipynb`

**Interfaces:**
- Consumes: the approved random-CV design and Excel filename.
- Produces: `build_notebook(destination: Path) -> Path` and a notebook containing embedded `RUNTIME_SOURCES` for `random_cv_runtime`.

- [ ] **Step 1: Write a failing notebook contract test**

```python
def test_random_cv_notebook_is_self_contained():
    self.assertTrue(NOTEBOOK.exists())
    source = notebook_source(NOTEBOOK)
    for marker in REQUIRED_SECTIONS:
        self.assertIn(marker, source)
    self.assertIn("RUNTIME_SOURCES", source)
    self.assertIn("random_cv_runtime", source)
    self.assertNotIn("from advanced_furnace_ml", source)
    self.assertNotIn("recommend_bundle", source)
```

Also compile every code cell with `compile(cell.source, ..., "exec")` and fail with the cell index on `SyntaxError`.

- [ ] **Step 2: Run RED verification**

Run: `python -m unittest discover -s advanced_furnace_ml/tests -p 'test_random_cv_notebook.py' -v`

Expected: FAIL because the notebook does not exist.

- [ ] **Step 3: Implement the notebook builder and embedded runtime snapshot**

Use `nbformat.v4.new_notebook`. Snapshot all `advanced_furnace_ml/src/advanced_furnace_ml/*.py` files except `pipeline.py`, embed them as a Python dictionary, write them to `random_cv_output/runtime/random_cv_runtime/` during notebook execution, and import only `random_cv_runtime`.

- [ ] **Step 4: Generate and verify GREEN**

Run:

```bash
python advanced_furnace_ml/tools/build_random_cv_notebook.py
python -m unittest discover -s advanced_furnace_ml/tests -p 'test_random_cv_notebook.py' -v
```

Expected: all structural and compilation tests pass.

---

### Task 2: Implement Excel loading, random holdout, five-fold CV, and 19-candidate selection

**Files:**
- Modify: `advanced_furnace_ml/tests/test_random_cv_notebook.py`
- Modify: `advanced_furnace_ml/tools/build_random_cv_notebook.py`
- Regenerate: `advanced_furnace_ml/notebooks/random_cv_model_comparison.ipynb`

**Interfaces:**
- Consumes: Excel workbook and embedded runtime.
- Produces: `df`, `train_df`, `test_df`, `random_splits`, `random_experiment`, `random_cv_summary`, `selected_before_test`.

- [ ] **Step 1: Add failing behavior-contract markers**

Require executable code equivalent to:

```python
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
random_splits = list(kfold.split(train_df))
random_experiment = run_model_matrix(train_df, random_splits)
selected_before_test = random_experiment.selected_name
random_experiment.freeze()
```

Require assertions for 292/233/59 rows, five folds, no overlap within each fold, all fold indices below 233, 19 unique candidates, and outlier count unchanged before/after splitting.

- [ ] **Step 2: Verify RED**

Run the notebook contract test and confirm missing split/model markers fail.

- [ ] **Step 3: Implement data and random-CV cells**

Load/mark all rows, create the split and folds with fixed seeds, execute `run_model_matrix`, add `selected_before_test`, display tree tuning and OOF ensemble weights, and freeze before any test audit call. Use the runtime selection score and deterministic sorting by `selection_score`, `mae_mean`, and `candidate`.

- [ ] **Step 4: Regenerate and verify GREEN**

Run the builder and contract tests; statically verify `test_df[TARGET_COL]` is not referenced before `random_experiment.freeze()`.

---

### Task 3: Add untouched test audit and chronological comparison

**Files:**
- Modify: `advanced_furnace_ml/tests/test_random_cv_notebook.py`
- Modify: `advanced_furnace_ml/tools/build_random_cv_notebook.py`
- Regenerate: `advanced_furnace_ml/notebooks/random_cv_model_comparison.ipynb`

**Interfaces:**
- Consumes: frozen random experiment, 59 test rows, and the complete chronological Excel order.
- Produces: `random_test_audit`, `test_rmse_winner`, `chronological_experiment`, `random_vs_chronological`.

- [ ] **Step 1: Add failing audit/comparison tests**

Require:

```python
random_test_audit = audit_frozen_models(random_experiment, test_df)
test_rmse_winner = random_test_audit.sort_values("rmse").iloc[0]["candidate"]
chron_dev = df.iloc[:248].copy()
chronological_experiment = run_model_matrix(chron_dev, large_window_splits(248))
random_vs_chronological = random_cv_summary.merge(..., on="candidate")
```

Require labels `CV Champion`, `test RMSE winner`, and the warning that test results cannot retroactively change selection. Require MAE, RMSE, WAPE, R², >10% error rate, bootstrap bounds, interval coverage, and width.

- [ ] **Step 2: Verify RED**

Run the contract test and confirm missing audit/comparison markers fail.

- [ ] **Step 3: Implement the test audit and chronological recalculation**

Call `audit_frozen_models` only after freeze. Recalculate chronological CV from the Excel rows rather than reading existing reports. Merge the two summaries by candidate and show how random splitting can be optimistic because early and late batches are mixed.

- [ ] **Step 4: Regenerate and verify GREEN**

Run builder/tests and confirm the test audit has 19 rows, each trained on 233 and evaluated on 59.

---

### Task 4: Retrain one Champion, save joblib, and verify inference

**Files:**
- Modify: `advanced_furnace_ml/tests/test_random_cv_notebook.py`
- Modify: `advanced_furnace_ml/tools/build_random_cv_notebook.py`
- Regenerate: `advanced_furnace_ml/notebooks/random_cv_model_comparison.ipynb`

**Interfaces:**
- Consumes: frozen Champion template, all 292 rows, CV/test summaries.
- Produces: `random_cv_output/artifacts/random_cv_champion.joblib`, reports, `run_summary.json`, and `artifact_replay_tests`.

- [ ] **Step 1: Add failing artifact/output tests**

Require the artifact filename, version `random-cv-furnace-1.0`, exact feature schema, final model fit on all 292 rows, CV-derived conformal radius, `1e-10` prediction replay, and missing/extra/nonnumeric rejection.

- [ ] **Step 2: Verify RED**

Run the contract test and confirm missing artifact/replay markers fail.

- [ ] **Step 3: Implement final fitting and isolated outputs**

Clone `random_experiment.results[selected_before_test].estimator_template`, fit on `df[FEATURE_COLS]`, and joblib-dump a plain bundle with the final estimator, fold models, feature order, CV/test summaries, selection name, split configuration, conformal radius, training count, and outlier count. Implement notebook-local validation/prediction functions so the artifact contains no recommendation fields.

- [ ] **Step 4: Add replay and negative-input assertions**

Reload with `joblib.load`, compare a sample prediction before/after to absolute tolerance `1e-10`, and verify that frames with missing, extra, or nonnumeric columns raise `ValueError`.

- [ ] **Step 5: Regenerate and verify GREEN**

Run the builder and all random-notebook contract tests.

---

### Task 5: Execute and verify the notebook

**Files:**
- Create: `advanced_furnace_ml/notebooks/random_cv_model_comparison_executed.ipynb`
- Create: `advanced_furnace_ml/random_cv_output/artifacts/random_cv_champion.joblib`
- Create: `advanced_furnace_ml/random_cv_output/reports/*.csv`
- Create: `advanced_furnace_ml/random_cv_output/reports/run_summary.json`

**Interfaces:**
- Consumes: unexecuted notebook and Excel workbook.
- Produces: executed notebook with all assertions passed and one verified Champion artifact.

- [ ] **Step 1: Execute the notebook**

```bash
jupyter nbconvert --to notebook --execute \
  advanced_furnace_ml/notebooks/random_cv_model_comparison.ipynb \
  --output random_cv_model_comparison_executed.ipynb \
  --output-dir advanced_furnace_ml/notebooks \
  --ExecutePreprocessor.timeout=1800 \
  --ExecutePreprocessor.kernel_name=python3
```

Expected: exit 0. Request local-kernel permission only if sandbox socket creation requires it.

- [ ] **Step 2: Inspect the executed notebook**

Parse with `nbformat`; assert every code cell has an execution count, no error output exists, the final assertion table contains no failure, and random-split optimism disclaimer is present.

- [ ] **Step 3: Verify the artifact in a fresh process**

Add only `advanced_furnace_ml/random_cv_output/runtime` to `PYTHONPATH`, load `random_cv_champion.joblib`, and verify artifact version, 292 training rows, Champion name, feature order, and one prediction. Do not add the original `advanced_furnace_ml/src` tree.

- [ ] **Step 4: Run the complete regression suite**

```bash
PYTHONPATH=advanced_furnace_ml/src \
python -m unittest discover -s advanced_furnace_ml/tests -v
```

Expected: every existing and new test passes.

- [ ] **Step 5: Verify protected files**

Recompute the recorded SHA-256 hashes for the original artifact, `testing_project.py`, and the two original standalone notebooks. They must remain unchanged.

- [ ] **Step 6: Deliver evidence**

Provide links to both notebooks, the Champion artifact, CV/test/chronological comparison reports, test count, execution error count, CV Champion, test winner, and the random-vs-time-validation caveat.
