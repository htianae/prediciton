from pathlib import Path
import unittest

import nbformat


PROJECT = Path(__file__).resolve().parents[2]
NOTEBOOK = PROJECT / "advanced_furnace_ml/notebooks/random_cv_model_comparison.ipynb"

REQUIRED_SECTIONS = (
    "随机80/20留出与训练集5折CV模型比较",
    "配置、依赖与自包含运行时",
    "Excel数据检查：保留全部292炉",
    "随机训练集与测试集",
    "训练集内部5折交叉验证与19候选",
    "冻结Champion后的59炉测试审计",
    "随机CV与时间滚动CV比较",
    "最终单模型重训与joblib",
    "最终自动测试汇总",
)


def notebook_source(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


class RandomCVNotebookTests(unittest.TestCase):
    def test_notebook_is_self_contained_prediction_only(self):
        self.assertTrue(NOTEBOOK.exists(), str(NOTEBOOK))
        source = notebook_source(NOTEBOOK)
        for marker in REQUIRED_SECTIONS:
            self.assertIn(marker, source)
        self.assertIn("RUNTIME_SOURCES", source)
        self.assertIn("random_cv_runtime", source)
        self.assertNotIn("from advanced_furnace_ml", source)
        self.assertNotIn("recommend_bundle", source)
        self.assertNotIn("安全门", source)

    def test_every_code_cell_compiles(self):
        self.assertTrue(NOTEBOOK.exists(), str(NOTEBOOK))
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            try:
                compile(cell.source, f"random-cv-cell-{index}", "exec")
            except SyntaxError as error:
                self.fail(f"代码单元 {index} 不能编译: {error}")

    def test_random_holdout_and_cv_selection_are_training_only(self):
        self.assertTrue(NOTEBOOK.exists(), str(NOTEBOOK))
        source = notebook_source(NOTEBOOK)
        for marker in (
            "train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)",
            "KFold(n_splits=5, shuffle=True, random_state=42)",
            "random_splits = list(kfold.split(train_df))",
            "random_experiment = run_model_matrix(train_df, random_splits)",
            "selected_before_test",
            "random_experiment.freeze()",
            "assert len(df) == 292",
            "assert len(train_df) == 233",
            "assert len(test_df) == 59",
            "assert len(random_splits) == 5",
            "assert random_cv_summary['candidate'].nunique() == 19",
            "outlier_count_before_split",
            "outlier_count_after_split",
        ):
            self.assertIn(marker, source)
        before_freeze = source.split("random_experiment.freeze()", 1)[0]
        self.assertNotIn("test_df[TARGET_COL]", before_freeze)

    def test_test_audit_and_chronological_reference_are_separate(self):
        self.assertTrue(NOTEBOOK.exists(), str(NOTEBOOK))
        source = notebook_source(NOTEBOOK)
        for marker in (
            "random_test_audit = audit_frozen_models(random_experiment, test_df)",
            "test_rmse_winner",
            "CV Champion",
            "test RMSE winner",
            "测试集结果不能反向改变CV Champion",
            "rmse_bootstrap_low95",
            "rmse_bootstrap_high95",
            "interval_coverage_90",
            "interval_mean_width",
            "error_gt_10pct_rate",
            "chron_dev = df.iloc[:248].copy()",
            "chronological_experiment = run_model_matrix(chron_dev, large_window_splits(248))",
            "random_vs_chronological",
            "随机划分会混合早期和晚期炉次",
        ):
            self.assertIn(marker, source)

    def test_final_artifact_and_replay_contract(self):
        self.assertTrue(NOTEBOOK.exists(), str(NOTEBOOK))
        source = notebook_source(NOTEBOOK)
        for marker in (
            "random_cv_champion.joblib",
            "random-cv-furnace-1.0",
            "final_champion.fit(df[FEATURE_COLS], df[TARGET_COL])",
            "prediction_round_trip_atol_1e_10",
            "artifact_replay_tests",
            "missing_column_rejected",
            "extra_column_rejected",
            "nonnumeric_value_rejected",
            "random_cv_summary.csv",
            "random_cv_fold_metrics.csv",
            "random_test_audit.csv",
            "chronological_cv_summary.csv",
            "random_vs_chronological.csv",
            "tree_tuning_summary.csv",
            "ensemble_weights.csv",
            "run_summary.json",
        ):
            self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
