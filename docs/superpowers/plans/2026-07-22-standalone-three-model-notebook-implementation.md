# Standalone Three-Model Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and execute one self-contained notebook containing the entire Excel-to-model-selection-to-three-model-recommendation experiment.

**Architecture:** Copy the tested logic into notebook code cells with no imports from the local `furnace_champion` package. Validate structure before execution, run the full notebook, verify outputs and protected-file checksums, then delete the superseded modular notebook artifacts explicitly requested by the user.

**Tech Stack:** Python 3.11, Jupyter, pandas, NumPy, scikit-learn, LightGBM, matplotlib, joblib.

## Global Constraints

- The notebook must run with only itself, the Excel workbook, and installed third-party dependencies.
- Use exactly Linear, Huber, and LightGBM.
- Use 5 chronological folds, 5000 optimizer evaluations, and seeds 0-9.
- Use total weight 86000 kg.
- Do not overwrite `artifacts/gas_champion.joblib`, `testing_project.py`, existing reports, or the user's `Nanshang_testing_diff_method*.ipynb` files.

### Task 1: Standalone structural test

**Files:**
- Create: `tests/test_standalone_notebook.py`

- [ ] Add a test that requires the source notebook to exist; rejects `furnace_champion` imports; and requires in-notebook definitions for Excel loading, model building, chronological evaluation, scoring, random search, GA, historical fallback, 86000 kg, 5000 evaluations, ten seeds, and optional joblib export.
- [ ] Run the test and confirm it fails because the notebook does not exist.

### Task 2: Build the standalone notebook

**Files:**
- Create: `Nanshang_three_model_standalone.ipynb`

- [ ] Add configuration/import cells with `EXPORT_JOBLIB = False`.
- [ ] Add complete data constants, Excel loading, transpose, numeric conversion, and outlier-marking functions.
- [ ] Add complete model pipelines, expanding chronological evaluation, metrics, ranking, and full-history fitting.
- [ ] Add complete context, P5-P95 bounds, kNN feasibility, uncertainty-penalized scoring, random search, GA, comparison, summary, and historical fallback functions.
- [ ] Add execution cells for all tables, figures, three raw recommendations, safety decisions, production candidates, and optional non-overwriting joblib export.
- [ ] Run the structural test and JSON validation.

### Task 3: Execute and verify

**Files:**
- Create: `Nanshang_three_model_standalone_executed.ipynb`

- [ ] Execute every cell with `jupyter nbconvert` and a 900-second timeout.
- [ ] Confirm every code cell has an execution number and no error/warning output.
- [ ] Confirm results contain exactly the three required models, 86000 kg, optimizer comparison, raw recommendations, historical fallback, and production candidates.
- [ ] Confirm all recommended integer and continuous parameters stay inside displayed bounds.
- [ ] Run the complete unittest suite.
- [ ] Confirm protected-file checksums are unchanged.

### Task 4: Remove superseded generated files

**Files:**
- Delete: `Nanshang_champion_workflow.ipynb`
- Delete: `Nanshang_champion_workflow_executed.ipynb`
- Delete: `Nanshang_three_model_selection_and_recommendation.ipynb`
- Delete: `Nanshang_three_model_selection_and_recommendation_executed.ipynb`
- Delete: `tests/test_three_model_notebook.py`
- Delete: `docs/superpowers/specs/2026-07-22-three-model-recommendation-notebook-design.md`
- Delete: `docs/superpowers/plans/2026-07-22-three-model-recommendation-notebook-implementation.md`

- [ ] Delete only the listed superseded generated files with `apply_patch` after the standalone executed notebook passes.
- [ ] Run the full unittest suite again after deletion.
- [ ] List final notebook files and verify the user's original `Nanshang_testing_diff_method*.ipynb` files remain.

This directory is not a Git repository, so no commit, branch, or merge steps apply.
