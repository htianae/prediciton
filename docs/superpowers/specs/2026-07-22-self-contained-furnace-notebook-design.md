# Self-Contained Furnace Excel Test Notebook Design

## Goal

Create one Jupyter notebook that starts from the four-month Excel workbook, retrains every requested model, executes the full prediction and recommendation evaluation, explains every safety decision, saves a fresh joblib artifact, and verifies that the saved artifact reproduces the in-memory results. The notebook must not read existing joblib files, generated CSV reports, project Python modules, or other notebooks.

## Deliverables

- `advanced_furnace_ml/notebooks/furnace_complete_excel_test.ipynb`
- `advanced_furnace_ml/notebooks/furnace_complete_excel_test_executed.ipynb`
- Notebook-generated outputs under `advanced_furnace_ml/complete_notebook_output/`
  - `artifacts/furnace_complete_bundle.joblib`
  - model, fold, locked-audit, optimizer, recommendation, and safety CSV reports
  - a JSON run summary

The new notebook and output directory must not overwrite the existing advanced artifact, reports, notebooks, or the original production-test files.

## Runtime Contract

The only project input is `4_month_data_2026_02_01_2026_06_25.xlsx`. The notebook locates it from the project root or accepts one configurable path in the first configuration cell. It may depend on NumPy, pandas, SciPy, scikit-learn, joblib, openpyxl, matplotlib, LightGBM, and CatBoost. It must fail early with a clear dependency or missing-file message.

Every run performs training from Excel. Existing joblib and CSV outputs are never used as inputs. The run uses fixed random seeds so the executed notebook, saved artifact, and replayed recommendation agree.

## Data and Validation Design

- Load the 292 chronological furnace batches and validate the six model inputs plus total-gas target.
- Mark high-gas outliers without deleting them.
- Reserve the last 44 batches as a locked audit set.
- Use the first 248 batches for three expanding chronological folds: 148/33, 181/33, and 214/34.
- Freeze the development-stage model choice before revealing the locked audit.
- Report the development Champion separately from the locked-set winner. The locked audit cannot retroactively change the frozen Champion.
- Include three-fold random cross-validation only as an explicitly labelled auxiliary reference.

## Prediction Model Matrix

Evaluate direct-total-gas and unit-gas routes for Ridge, ElasticNet, Huber, GAM, GPR, CatBoost, LightGBM, Ridge plus LightGBM residual, and Huber plus LightGBM residual. Build a nonnegative OOF ensemble from development-fold out-of-fold predictions. Compare MAE, RMSE, WAPE, R², error-over-10% rate, fold variability, worst-fold RMSE, and a stability-penalized selection score.

Tune small-data variants of LightGBM and CatBoost using development folds only. Generate 90% conformal intervals from development out-of-fold residuals. Also report chronological-fold prediction standard deviation and GPR native standard deviation where available.

## Recommendation Context and Baselines

For a default total weight of 86000 kg:

- Select historical batches within ±5% of total weight, widening to ±10% only if fewer than 20 batches are available.
- Display the number of similar batches and the actual total-gas median used as the common baseline.
- Select the lowest-gas 20% of similar batches and calculate each controllable parameter's median independently.
- Explicitly state that this median parameter vector need not be one real historical furnace.
- Use k-nearest-neighbour distance to test whether a joint candidate is historically supported.

## Optimization and Safety Tests

Compare random search, genetic search, and GPR expected-improvement Bayesian optimization with identical bounds, evaluation budgets, and seeds. Use seeds 0, 1, and 2 for optimizer stability comparison, then fixed seed 42 for the deployment recommendation.

Run two diagnostic recommendation regimes:

1. Broad historical 5th-to-95th-percentile search, used to reproduce and explain boundary-seeking or high-uncertainty failures such as the earlier LightGBM result.
2. Production-candidate search inside the intersection of the historical range and ±10% around the historical low-gas parameter medians.

Evaluate recommendations from LightGBM direct, GPR direct, Ridge plus LightGBM residual direct, Huber plus LightGBM residual direct, and OOF Ensemble. Retain one best safe recommendation per model in the summary while preserving optimizer-level comparison reports.

A production candidate passes only if all checks pass:

- conservative predicted gas is lower than the similar-history actual-gas median;
- at least two of three chronological fold models predict savings;
- normalized historical kNN distance is at most 1;
- no controllable parameter lies within 2% of a search boundary;
- the joint candidate lies inside the ±10% trust region.

The notebook must display each safety component as a separate Boolean column and show the reason for every failure. Passing is an offline A-grade trial candidate, not proof of realized factory savings.

## Artifact and Replay Verification

Fit the frozen prediction model and chosen recommendation model on all 292 batches only after the locked audit is reported. Save feature order, preprocessing, fold models, conformal radius, feasibility reference, recommendation configuration, optimizer, seed, training metadata, and the default 86000 kg recommendation in one joblib bundle.

Reload the bundle in a clean notebook cell and verify:

- artifact version and expected feature schema;
- prediction before and after serialization agree to absolute tolerance `1e-10`;
- the fixed-seed 86000 kg recommendation parameters and conservative prediction agree to `1e-10`;
- malformed input is rejected with an explanatory error.

## Notebook Test Summary

The last section runs explicit assertions and prints a compact pass/fail table covering data shape, chronology and no leakage, model-matrix completeness, frozen selection, locked-set size, uncertainty availability, equal optimizer budgets, integer door-count handling, recommendation safety components, joblib round trip, and absence of notebook execution errors. A failed required assertion stops execution rather than silently producing a completed notebook.

## Presentation

The notebook is written in Chinese, with code and concise explanations adjacent to each result. Tables identify whether they are development validation, locked audit, diagnostic broad search, or production-candidate search. The conclusion distinguishes:

- development prediction Champion;
- locked-audit winner;
- production recommendation model;
- historical fallback;
- offline predicted savings versus realized factory savings.

No API or real-time server access is included.
