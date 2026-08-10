# Random Holdout and Cross-Validation Model Comparison Notebook Design

## Goal

Create a new, isolated Jupyter notebook that retains all 292 furnace batches, compares the same 19 prediction candidates under a random train/test design, selects one Champion using cross-validation inside the training split, evaluates the frozen Champion on the untouched test split, retrains it on all 292 batches, and saves one replayable joblib artifact.

The notebook compares prediction models only. It contains no parameter recommendation, optimization algorithm, historical baseline, or safety-gate workflow.

## Deliverables

- `advanced_furnace_ml/notebooks/random_cv_model_comparison.ipynb`
- `advanced_furnace_ml/notebooks/random_cv_model_comparison_executed.ipynb`
- `advanced_furnace_ml/random_cv_output/artifacts/random_cv_champion.joblib`
- CSV and JSON reports under `advanced_furnace_ml/random_cv_output/reports/`

The new notebook and output directory must not overwrite the chronological experiment, complete-test notebook, original notebook, or existing joblib files.

## Input and Features

The notebook reads `4_month_data_2026_02_01_2026_06_25.xlsx` directly. It validates the six production prediction inputs and the total-gas target. All 292 batches remain in the experiment. Target IQR outliers are marked and reported but are not removed or downweighted.

The notebook is self-contained at runtime: it does not import project Python modules or read existing artifacts/reports. A frozen runtime source snapshot is embedded in the notebook and written into `random_cv_output/runtime/` when executed, so copying the notebook together with the Excel workbook is sufficient.

The six inputs are:

- total charge weight;
- solid-material weight ratio;
- final melting time;
- waiting duration;
- furnace-door opening count;
- furnace-door opening duration.

## Random Validation Contract

Perform one reproducible holdout split with `train_test_split(test_size=0.20, random_state=42)`. This produces 233 training batches and 59 test batches. The test target cannot participate in tree-variant tuning, ensemble construction, candidate ranking, or Champion selection.

Inside the 233-row training split, use `KFold(n_splits=5, shuffle=True, random_state=42)`. Every candidate is evaluated on the same five folds. Report mean and standard deviation for MAE, RMSE, WAPE, R², and error-over-10% rate.

The selection score is mean CV RMSE plus `0.25 * CV RMSE standard deviation`. The minimum-score candidate becomes `selected_before_test`. Ties are resolved by lower mean CV MAE and then candidate name. The Champion is frozen before test metrics are calculated.

## Candidate Matrix

Evaluate direct-total-gas and unit-gas routes for:

- Ridge;
- ElasticNet;
- Huber;
- GAM;
- GPR;
- CatBoost;
- LightGBM;
- Ridge plus LightGBM residual;
- Huber plus LightGBM residual.

Build one nonnegative OOF Ensemble from the best four base-route candidates using only training-split out-of-fold predictions. This produces 19 candidates in total.

LightGBM and CatBoost small-data variants are tuned only through the five training folds. No test rows can affect variant selection.

## Test Audit

After Champion selection is frozen, fit every candidate template on the 233-row training split and evaluate it once on the 59-row test split. Report MAE, RMSE, WAPE, R², error-over-10% rate, bootstrap RMSE limits, 90% interval coverage, and interval width.

Display two separate labels:

- `CV Champion`: selected using only five-fold training CV;
- `test RMSE winner`: the candidate with the lowest random-holdout test RMSE.

The test winner cannot retroactively replace the CV Champion. If they differ, the notebook explains why inspecting the test result and switching would turn the test set into validation data.

## Comparison with Chronological Results

For the same candidate names, display random-CV and chronological-CV summaries side by side. To keep the new notebook independent from generated reports, chronological reference values are recalculated from Excel using the exact expanding windows 148/33, 181/33, and 214/34 rather than read from existing CSV files.

The comparison must explicitly state that random splitting mixes early and late furnaces and may therefore produce optimistic results for future-batch deployment. It is an auxiliary experiment, not a replacement for production-oriented chronological validation.

## Final Model and Artifact

After the holdout audit is complete, clone the frozen CV Champion template and fit it on all 292 batches. Save one joblib bundle containing:

- artifact version and creation time;
- exact feature order;
- split and CV configuration;
- all CV and test summary tables;
- Champion name and estimator;
- conformal radius derived without test leakage;
- training batch count and outlier count.

Reload the artifact and verify that predictions before and after serialization agree with absolute tolerance `1e-10`. Reject missing, extra, or nonnumeric input fields with clear errors.

Because some candidate estimators use custom route or ensemble classes, external loading adds the notebook-generated `random_cv_output/runtime/` directory to `PYTHONPATH`. The artifact never depends on the original project source tree.

## Notebook Assertions

The final section must stop execution if any required contract fails. Assertions cover:

- 292 total rows and no target-row deletion;
- 233/59 random holdout sizes;
- five shuffled training-only CV folds;
- 19 candidate models;
- tree tuning and OOF ensemble using training rows only;
- Champion frozen before test evaluation;
- 59-row test audit for every candidate;
- final artifact model name equal to the frozen CV Champion;
- joblib round-trip equality;
- malformed-input rejection;
- offline comparison disclaimer.

## Presentation

The notebook is written in Chinese with concise explanations beside each table. It highlights the difference between CV selection, test audit, and final all-data retraining. No claim is made that the random-split Champion is automatically superior for predicting future production batches.
