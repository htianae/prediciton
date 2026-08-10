import json
from pathlib import Path
import unittest


class StandaloneNotebookTests(unittest.TestCase):
    def test_notebook_contains_complete_standalone_workflow(self):
        path = Path("Nanshang_three_model_standalone.ipynb")
        self.assertTrue(path.exists())
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertNotIn("from furnace_champion", source)
        self.assertNotIn("import furnace_champion", source)
        for token in (
            "def load_batch_data",
            "def mark_target_outliers",
            "def build_candidate_models",
            "def evaluate_models_time_series",
            "def score_candidates",
            "def random_search",
            "def genetic_search",
            "def compare_optimizers",
            "def historical_fallback",
            "LightGBM",
            "Linear",
            "Huber",
            "TOTAL_WEIGHT = 86000.0",
            "SEARCH_BUDGET = 5000",
            "SEEDS = range(10)",
            "EXPORT_JOBLIB = False",
            "three_model_standalone.joblib",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
