from pathlib import Path
import tempfile
import unittest

import numpy as np

from advanced_furnace_ml.artifacts import load_bundle, predict_bundle, recommend_bundle
from advanced_furnace_ml.data import FEATURE_COLS, load_batch_data
from advanced_furnace_ml.pipeline import run_advanced_pipeline


EXCEL = Path(__file__).parents[2] / "4_month_data_2026_02_01_2026_06_25.xlsx"


class PipelineArtifactTests(unittest.TestCase):
    def test_small_end_to_end_and_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_advanced_pipeline(
                EXCEL,
                directory,
                optimizer_budget=30,
                seeds=[0],
                fast_mode=True,
            )
            output = Path(directory)
            for relative in (
                "artifacts/advanced_furnace_bundle.joblib",
                "reports/model_cv_summary.csv",
                "reports/tree_tuning_summary.csv",
                "reports/random_cv_reference.csv",
                "reports/locked_audit.csv",
                "reports/recommendation_summary.csv",
                "reports/pipeline_summary.json",
            ):
                self.assertTrue((output / relative).exists(), relative)
            self.assertEqual(len(result["locked_audit"]), 4)
            self.assertEqual(result["selected_before_lock"], result["bundle"]["selected_model_name"])
            bundle = load_bundle(output / "artifacts/advanced_furnace_bundle.joblib")
            df = load_batch_data(EXCEL)
            rows = df[FEATURE_COLS].iloc[:3]
            before = predict_bundle(result["bundle"], rows)["predicted_gas"].to_numpy()
            after = predict_bundle(bundle, rows)["predicted_gas"].to_numpy()
            np.testing.assert_allclose(before, after, rtol=0, atol=1e-10)
            replay = recommend_bundle(bundle, total_weight=86000, budget=30, seed=42)
            expected = result["production_recommendation"]
            self.assertEqual(replay["recommendation_model_name"], expected["candidate"])
            self.assertEqual(replay["optimizer"], expected["optimizer"])
            self.assertEqual(replay["safety_pass"], expected["safety_pass"])
            for column, value in expected["recommendation"].items():
                self.assertAlmostEqual(replay["recommendation"][column], value, places=10)
            with self.assertRaisesRegex(ValueError, "缺少"):
                predict_bundle(bundle, rows.drop(columns=FEATURE_COLS[0]))


if __name__ == "__main__":
    unittest.main()
