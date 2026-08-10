import unittest

import numpy as np

from advanced_furnace_ml.validation import (
    LockedAudit,
    bootstrap_metric_interval,
    large_window_splits,
    regression_metrics,
)


class ValidationTests(unittest.TestCase):
    def test_exact_large_windows_and_no_future_leakage(self):
        splits = large_window_splits(248)
        self.assertEqual([(len(a), len(b)) for a, b in splits], [(148, 33), (181, 33), (214, 34)])
        self.assertEqual([(a[-1], b[0], b[-1]) for a, b in splits], [(147, 148, 180), (180, 181, 213), (213, 214, 247)])
        self.assertTrue(all(train.max() < test.min() for train, test in splits))

    def test_metrics_bootstrap_and_lock(self):
        y = np.array([100.0, 200.0, 300.0, 400.0])
        p = np.array([110.0, 180.0, 330.0, 390.0])
        metrics = regression_metrics(y, p)
        self.assertAlmostEqual(metrics["mae"], 17.5)
        self.assertAlmostEqual(metrics["wape"], 0.07)
        a = bootstrap_metric_interval(y, p, "mae", seed=7, n_boot=100)
        b = bootstrap_metric_interval(y, p, "mae", seed=7, n_boot=100)
        self.assertEqual(a, b)
        audit = LockedAudit(np.arange(4))
        with self.assertRaisesRegex(RuntimeError, "freeze"):
            audit.values()
        audit.freeze()
        np.testing.assert_array_equal(audit.values(), np.arange(4))


if __name__ == "__main__":
    unittest.main()
