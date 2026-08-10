# Self-Contained Furnace Excel Test Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute one Chinese Jupyter notebook that reads only the four-month Excel workbook, retrains the complete furnace prediction/recommendation experiment, saves isolated artifacts and reports, and proves serialization and safety-gate reproducibility.

**Architecture:** A deterministic notebook-builder script embeds a frozen standalone runtime into notebook cells; the notebook writes that embedded runtime into its own output directory and imports only that generated runtime. The notebook then executes data validation, chronological model evaluation, locked audit, broad-search diagnostics, trust-region recommendation, artifact replay, and assertion-based checks. The executed notebook is produced with `nbconvert` and inspected for errors and required outputs.

**Tech Stack:** Python 3.11, Jupyter/nbformat/nbconvert, NumPy, pandas, SciPy, scikit-learn, joblib, openpyxl, matplotlib, LightGBM, CatBoost, unittest.

## Global Constraints

- The only project input at notebook runtime is `4_month_data_2026_02_01_2026_06_25.xlsx`.
- Do not read existing joblib files, generated CSV reports, project Python modules, or other notebooks.
- Write all generated files under `advanced_furnace_ml/complete_notebook_output/`.
- Do not overwrite existing advanced artifacts/reports/notebooks or original production-test files.
- Use 44 locked batches and exact chronological folds 148/33, 181/33, and 214/34.
- Use seeds 0, 1, and 2 for optimizer comparison and seed 42 for the replayable production recommendation.
- The notebook must distinguish development Champion, locked-audit winner, and recommendation model.
- Offline safety passage is a trial-candidate label, not proof of factory savings.
- The workspace is not a Git repository, so commit steps are not applicable.

---

### Task 1: Define the notebook structural contract

**Files:**
- Create: `advanced_furnace_ml/tests/test_complete_excel_notebook.py`
- Create: `advanced_furnace_ml/tools/build_complete_excel_notebook.py`
- Create: `advanced_furnace_ml/notebooks/furnace_complete_excel_test.ipynb`

**Interfaces:**
- Consumes: the approved design specification and existing Excel filename.
- Produces: `build_notebook(destination: Path) -> Path` and an unexecuted notebook satisfying the structural contract.

- [ ] **Step 1: Write the failing structural test**

```python
class CompleteExcelNotebookTests(unittest.TestCase):
    def test_notebook_is_standalone_and_contains_required_sections(self):
        path = PROJECT / "advanced_furnace_ml/notebooks/furnace_complete_excel_test.ipynb"
        self.assertTrue(path.exists())
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        for marker in REQUIRED_MARKERS:
            self.assertIn(marker, source)
        self.assertNotIn("from advanced_furnace_ml", source)
        self.assertNotIn("reports/model_cv_summary.csv", source)
        self.assertNotIn("advanced_furnace_bundle.joblib", source)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest advanced_furnace_ml/tests/test_complete_excel_notebook.py -v`

Expected: FAIL because `furnace_complete_excel_test.ipynb` does not exist.

- [ ] **Step 3: Implement the minimal notebook builder and section skeleton**

Use `nbformat.v4.new_notebook`, `new_markdown_cell`, and `new_code_cell`. Include Chinese sections for configuration, dependency check, data audit, chronological validation, model matrix, locked audit, baseline derivation, broad diagnostic search, production recommendation, artifact replay, and final assertions. The configuration cell must define only the Excel input and isolated output root.

- [ ] **Step 4: Generate the notebook and verify GREEN**

Run:

```bash
python advanced_furnace_ml/tools/build_complete_excel_notebook.py
python -m unittest advanced_furnace_ml/tests/test_complete_excel_notebook.py -v
```

Expected: PASS for all structural assertions.

---

### Task 2: Embed the standalone runtime and validate data/chronology

**Files:**
- Modify: `advanced_furnace_ml/tests/test_complete_excel_notebook.py`
- Modify: `advanced_furnace_ml/tools/build_complete_excel_notebook.py`
- Regenerate: `advanced_furnace_ml/notebooks/furnace_complete_excel_test.ipynb`

**Interfaces:**
- Consumes: Excel path and output root from Task 1.
- Produces inside the executed notebook: `df`, `dev`, `locked`, `splits`, and generated module `complete_notebook_output/runtime/furnace_complete_runtime.py`.

- [ ] **Step 1: Add failing tests for embedded runtime and exact split assertions**

The test must require embedded source markers `load_batch_data`, `chronological_dev_lock_split`, `large_window_splits`, and notebook assertions for 292 rows, 248 development rows, 44 locked rows, and exact split tuples `[(0,148,148,181), (0,181,181,214), (0,214,214,248)]`.

- [ ] **Step 2: Verify RED**

Run the single notebook contract test and confirm the missing runtime markers cause failure.

- [ ] **Step 3: Embed data, feature, validation, and metric functions**

Embed source in the notebook rather than reading project modules at runtime. The runtime must validate the six features and target, preserve chronology, mark target outliers without deletion, and reject duplicate batch IDs or missing target values. Door-open count remains numeric during modeling and integer-constrained during optimization.

- [ ] **Step 4: Regenerate and verify GREEN**

Run the builder and structural test. Inspect the generated notebook JSON to confirm no project-module import or existing-output read was introduced.

---

### Task 3: Embed model matrix, tuning, OOF ensemble, and uncertainty

**Files:**
- Modify: `advanced_furnace_ml/tests/test_complete_excel_notebook.py`
- Modify: `advanced_furnace_ml/tools/build_complete_excel_notebook.py`
- Regenerate: `advanced_furnace_ml/notebooks/furnace_complete_excel_test.ipynb`

**Interfaces:**
- Consumes: `dev`, exact chronological splits, and feature schema.
- Produces: `experiment`, `model_cv_summary`, `chronological_fold_metrics`, `tree_tuning_summary`, `random_cv_reference`, and fold models.

- [ ] **Step 1: Add failing model-matrix content tests**

Require model names Ridge, ElasticNet, Huber, GAM, GPR, CatBoost, LightGBM, Ridge+LGBMResidual, Huber+LGBMResidual, OOFEnsemble; both direct and unit routes; nonnegative OOF weights; LightGBM/CatBoost variant tables; conformal 90% interval; fold standard deviation; and GPR native standard deviation.

- [ ] **Step 2: Verify RED**

Run the contract test and confirm it fails on missing model/uncertainty markers.

- [ ] **Step 3: Add runtime implementations and notebook result cells**

Use preprocessing pipelines with median imputation and robust scaling where appropriate. Tune tree variants on development folds only. Calculate selection score as mean chronological RMSE plus `0.25 * RMSE standard deviation`. Freeze the minimum-score development candidate before locked evaluation.

- [ ] **Step 4: Regenerate and verify GREEN**

Run the builder and tests; confirm every required model-route marker is present and no locked target participates in selection.

---

### Task 4: Add locked audit and explicit Champion interpretation

**Files:**
- Modify: `advanced_furnace_ml/tests/test_complete_excel_notebook.py`
- Modify: `advanced_furnace_ml/tools/build_complete_excel_notebook.py`
- Regenerate: `advanced_furnace_ml/notebooks/furnace_complete_excel_test.ipynb`

**Interfaces:**
- Consumes: frozen development experiment and the 44 locked rows.
- Produces: `locked_audit`, `selected_before_lock`, `locked_rmse_winner`, and interpretation table.

- [ ] **Step 1: Add failing tests for frozen selection language and audit fields**

Require the phrases `开发阶段 Champion`, `锁定集 RMSE 优胜者`, and `锁定集不能反向参与选模`, plus locked metrics MAE, RMSE, WAPE, R², >10% error rate, bootstrap RMSE limits, and interval coverage.

- [ ] **Step 2: Verify RED**

Run the contract test and confirm missing audit/interpretation content fails.

- [ ] **Step 3: Implement audit cells**

Audit every frozen candidate on the 44 rows, sort a display copy by RMSE, retain `selected_before_lock`, and print both the pre-lock Champion and locked-set winner without replacing the frozen selection.

- [ ] **Step 4: Regenerate and verify GREEN**

Run the builder and test, then statically verify locked rows are referenced only after model selection is frozen.

---

### Task 5: Add recommendation baselines, optimizers, and two safety regimes

**Files:**
- Modify: `advanced_furnace_ml/tests/test_complete_excel_notebook.py`
- Modify: `advanced_furnace_ml/tools/build_complete_excel_notebook.py`
- Regenerate: `advanced_furnace_ml/notebooks/furnace_complete_excel_test.ipynb`

**Interfaces:**
- Consumes: all 292 historical rows after audit, recommendation candidates and fold models.
- Produces: `recommendation_context`, `broad_search_diagnostics`, `optimizer_comparison`, `recommendation_summary`, and `production_recommendation`.

- [ ] **Step 1: Add failing safety-contract tests**

Require baseline fields for similar-row count, ±5/±10 tolerance, actual-gas median, low-gas 20% count, and independent controllable medians. Require random, genetic, and Bayesian optimizers with equal budget/seeds and integer door counts. Require individual Boolean columns `savings_pass`, `consensus_pass`, `feasibility_pass`, `boundary_pass`, `trust_region_pass`, and final `safety_pass`.

- [ ] **Step 2: Verify RED**

Run the contract test and confirm it fails for missing baseline/safety markers.

- [ ] **Step 3: Implement broad and production searches**

Broad diagnostics use historical 5th-to-95th-percentile bounds and display boundary/uncertainty failures without treating them as production recommendations. Production candidates use the historical-bound/trust-region intersection, conservative prediction equal to chronological-fold median plus fold standard deviation, at least 2/3 savings consensus, normalized kNN distance at most 1, no 2% boundary hit, and fixed deployment seed 42.

- [ ] **Step 4: Add model recommendation comparison and fallback**

Compare LightGBM direct, GPR direct, both residual direct models, and OOF Ensemble. If no candidate passes every gate, return the historical low-gas median parameter vector with `safety_pass=False`; never relabel fallback as a model result.

- [ ] **Step 5: Regenerate and verify GREEN**

Run the structural tests and verify all safety components and disclaimer text appear.

---

### Task 6: Add isolated reports, artifact, replay, and assertion suite

**Files:**
- Modify: `advanced_furnace_ml/tests/test_complete_excel_notebook.py`
- Modify: `advanced_furnace_ml/tools/build_complete_excel_notebook.py`
- Regenerate: `advanced_furnace_ml/notebooks/furnace_complete_excel_test.ipynb`

**Interfaces:**
- Consumes: experiment, audit, optimizer and recommendation results.
- Produces: isolated CSV/JSON files, generated runtime module, `furnace_complete_bundle.joblib`, `artifact_replay_tests`, and `final_test_summary`.

- [ ] **Step 1: Add failing output/replay contract tests**

Require isolated output paths, artifact version, feature schema validation, prediction round-trip tolerance `1e-10`, recommendation round-trip tolerance `1e-10`, malformed-input rejection, and final named assertion table.

- [ ] **Step 2: Verify RED**

Run the contract test and confirm the replay/output markers are missing.

- [ ] **Step 3: Implement report and artifact cells**

Save model CV, fold metrics, tree tuning, random CV reference, locked audit, broad diagnostics, optimizer comparison, recommendation summary, and final JSON summary. Store preprocessing/model objects, fold models, conformal radius, feasibility reference, optimizer, seed, metadata, and default recommendation in joblib.

- [ ] **Step 4: Implement reload and negative-input tests**

Unload and re-import the generated runtime module before `joblib.load`, compare predictions and recommendations exactly within tolerance, and deliberately test missing, extra, and nonnumeric input fields.

- [ ] **Step 5: Regenerate and verify GREEN**

Run the structural contract test and inspect the notebook for prohibited input reads.

---

### Task 7: Execute the notebook and perform final verification

**Files:**
- Create: `advanced_furnace_ml/notebooks/furnace_complete_excel_test_executed.ipynb`
- Create: `advanced_furnace_ml/complete_notebook_output/runtime/furnace_complete_runtime.py`
- Create: `advanced_furnace_ml/complete_notebook_output/artifacts/furnace_complete_bundle.joblib`
- Create: reports under `advanced_furnace_ml/complete_notebook_output/reports/`
- Modify only if a real defect is found: builder, notebook contract test, or generated notebook.

**Interfaces:**
- Consumes: unexecuted notebook and Excel workbook.
- Produces: fully executed notebook with no error outputs and verified artifacts/reports.

- [ ] **Step 1: Execute with an isolated Jupyter kernel**

Run:

```bash
jupyter nbconvert --to notebook --execute \
  advanced_furnace_ml/notebooks/furnace_complete_excel_test.ipynb \
  --output furnace_complete_excel_test_executed.ipynb \
  --output-dir advanced_furnace_ml/notebooks \
  --ExecutePreprocessor.timeout=1800 \
  --ExecutePreprocessor.kernel_name=python3
```

Expected: exit 0 and an executed notebook file. If local kernel socket creation is sandbox-blocked, request the required permission and rerun the same command.

- [ ] **Step 2: Inspect execution results**

Parse with `nbformat`; assert every code cell has an execution count, no output has `output_type == "error"`, the final test table reports all required checks as passed, and the offline-savings disclaimer is present.

- [ ] **Step 3: Verify generated files and joblib in a fresh Python process**

Add only `complete_notebook_output/runtime` to `PYTHONPATH`, load the new joblib, run one prediction and the fixed-seed 86000 kg recommendation, and compare against the JSON summary.

- [ ] **Step 4: Run all regression tests**

Run:

```bash
PYTHONPATH=advanced_furnace_ml/src \
python -m unittest discover -s advanced_furnace_ml/tests -v
```

Expected: all existing and new tests pass.

- [ ] **Step 5: Verify protected files remain unchanged**

Recompute SHA-256 for `artifacts/gas_champion.joblib`, `testing_project.py`, `Nanshang_three_model_standalone.ipynb`, and `Nanshang_three_model_standalone_executed.ipynb`; compare with the recorded hashes before delivery.

- [ ] **Step 6: Deliver links and evidence**

Report the unexecuted and executed notebook paths, new artifact/output directory, test counts, execution-error count, development Champion, locked winner, recommendation model, safety result, and explicit offline-savings caveat.
