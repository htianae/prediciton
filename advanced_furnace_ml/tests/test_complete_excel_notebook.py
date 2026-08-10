from pathlib import Path
import unittest

import nbformat


PROJECT = Path(__file__).resolve().parents[2]
NOTEBOOK = PROJECT / "advanced_furnace_ml/notebooks/furnace_complete_excel_test.ipynb"

REQUIRED_SECTIONS = (
    "完全自包含：Excel 全量训练与安全推荐测试",
    "配置与依赖检查",
    "Excel 数据质量检查",
    "三个时间折与锁定集",
    "完整模型矩阵与开发阶段选模",
    "最后44炉锁定审计",
    "86000 kg 历史基准推导",
    "宽范围搜索诊断",
    "生产候选推荐与安全门",
    "joblib 保存与重载复现",
    "最终自动测试汇总",
)


class CompleteExcelNotebookTests(unittest.TestCase):
    def test_every_code_cell_compiles(self):
        self.assertTrue(NOTEBOOK.exists(), str(NOTEBOOK))
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            try:
                compile(cell.source, f"notebook-cell-{index}", "exec")
            except SyntaxError as error:
                self.fail(f"代码单元 {index} 不能编译: {error}")

    def test_notebook_is_standalone_and_contains_required_sections(self):
        self.assertTrue(NOTEBOOK.exists(), str(NOTEBOOK))
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        for marker in REQUIRED_SECTIONS:
            self.assertIn(marker, source)
        self.assertIn("4_month_data_2026_02_01_2026_06_25.xlsx", source)
        self.assertIn("complete_notebook_output", source)
        self.assertNotIn("from advanced_furnace_ml", source)
        self.assertNotIn("reports/model_cv_summary.csv", source)
        self.assertNotIn("advanced_furnace_bundle.joblib", source)

    def test_notebook_embeds_runtime_and_exact_chronological_contract(self):
        self.assertTrue(NOTEBOOK.exists(), str(NOTEBOOK))
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        for marker in (
            "RUNTIME_SOURCES",
            "furnace_complete_runtime",
            "def load_batch_data",
            "def chronological_dev_lock_split",
            "def large_window_splits",
            "assert len(df) == 292",
            "assert len(dev) == 248",
            "assert len(locked) == 44",
            "expected_windows = [(0, 148, 148, 181), (0, 181, 181, 214), (0, 214, 214, 248)]",
        ):
            self.assertIn(marker, source)

    def test_notebook_runs_full_model_matrix_and_uncertainty(self):
        self.assertTrue(NOTEBOOK.exists(), str(NOTEBOOK))
        source = "\n".join(
            cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells
        )
        for marker in (
            "experiment = run_model_matrix(dev, splits)",
            "selected_before_lock = experiment.selected_name",
            "experiment.freeze()",
            "model_cv_summary = experiment.summary.copy()",
            "chronological_fold_metrics",
            "tree_tuning_summary",
            "random_cv_reference_table = random_cv_reference",
            "OOFEnsemble",
            "Ridge+LGBMResidual",
            "Huber+LGBMResidual",
            "conformal_radius_90",
            "native_model_std_mean",
        ):
            self.assertIn(marker, source)

    def test_notebook_keeps_locked_audit_separate_from_selection(self):
        self.assertTrue(NOTEBOOK.exists(), str(NOTEBOOK))
        source = "\n".join(
            cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells
        )
        for marker in (
            "locked_audit = audit_frozen_models(experiment, locked)",
            "locked_rmse_winner",
            "开发阶段 Champion",
            "锁定集 RMSE 优胜者",
            "锁定集不能反向参与选模",
            "rmse_bootstrap_low95",
            "rmse_bootstrap_high95",
            "interval_coverage_90",
            "error_gt_10pct_rate",
        ):
            self.assertIn(marker, source)

    def test_notebook_compares_optimizers_and_exposes_every_safety_gate(self):
        self.assertTrue(NOTEBOOK.exists(), str(NOTEBOOK))
        source = "\n".join(
            cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells
        )
        for marker in (
            "TARGET_WEIGHT = 86000.0",
            "similar_history_rows",
            "low_gas_20pct_rows",
            "actual_baseline_gas",
            "历史低气耗参数分别取中位数，不保证来自同一炉",
            "broad_search_diagnostics",
            "historical 5%-95%",
            "compare_optimizers",
            "common_budget=SEARCH_BUDGET",
            "seeds=(0, 1, 2)",
            "DEPLOYMENT_SEED = 42",
            "savings_pass",
            "consensus_pass",
            "feasibility_pass",
            "boundary_pass",
            "trust_region_pass",
            "safety_pass",
            "random_search",
            "genetic_algorithm",
            "bayesian_optimization",
            "DOOR_COUNT_COL",
        ):
            self.assertIn(marker, source)

    def test_notebook_saves_isolated_artifact_and_replays_it(self):
        self.assertTrue(NOTEBOOK.exists(), str(NOTEBOOK))
        source = "\n".join(
            cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells
        )
        for marker in (
            "furnace_complete_bundle.joblib",
            "artifact_version",
            "furnace-complete-notebook-1.0",
            "prediction_round_trip_atol_1e-10",
            "recommendation_round_trip_atol_1e-10",
            "malformed_missing_rejected",
            "malformed_extra_rejected",
            "malformed_nonnumeric_rejected",
            "artifact_replay_tests",
            "final_test_summary",
            "run_summary.json",
            "model_cv_summary.to_csv",
            "locked_audit.to_csv",
            "recommendation_summary.to_csv",
        ):
            self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
