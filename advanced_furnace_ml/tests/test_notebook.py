import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class NotebookTests(unittest.TestCase):
    def test_notebook_contains_required_advanced_sections(self):
        path = ROOT / "notebooks/advanced_furnace_experiment.ipynb"
        self.assertTrue(path.exists())
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        for token in (
            "run_advanced_pipeline",
            "148/33",
            "181/33",
            "214/34",
            "最后44炉",
            "model_cv_summary",
            "locked_audit",
            "recommendation_summary",
            "optimizer_runs",
            "offline_savings_disclaimer",
            "离线预计节省不等于工厂实际节省",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
