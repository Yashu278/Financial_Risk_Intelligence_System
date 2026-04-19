import unittest
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_utils import compute_features_from_lists


class FeatureConsistencyTests(unittest.TestCase):
    def test_income_growth_rate_matches_training_formula(self):
        income = np.array([10000, 11000, 12000, 13000, 14000, 15000], dtype=float)
        expense = np.array([7000, 7600, 8200, 8800, 9400, 10000], dtype=float)

        got = compute_features_from_lists(income, expense)
        expected = float(np.polyfit(np.arange(len(income), dtype=float), income, 1)[0])

        self.assertAlmostEqual(got["income_growth_rate"], expected, places=10)

    def test_avg_irregular_amt_is_absolute_irregular_expense_mean(self):
        income = np.array([10000] * 12, dtype=float)
        expense = np.array([7000] * 10 + [13000, 14000], dtype=float)

        got = compute_features_from_lists(income, expense)

        mean_expense = float(np.mean(expense))
        irregular_mask = expense > (mean_expense * 1.15)
        expected = float(np.mean(expense[irregular_mask])) if np.any(irregular_mask) else 0.0

        self.assertAlmostEqual(got["avg_irregular_amt"], expected, places=10)

    def test_severe_overspend_freq_is_empirical_threshold_freq(self):
        income = np.array([10000] * 12, dtype=float)
        expense = np.array([7000] * 9 + [12100, 12500, 13000], dtype=float)

        got = compute_features_from_lists(income, expense)
        expected = float(np.mean(expense > income * 1.20))

        self.assertAlmostEqual(got["severe_overspend_freq"], expected, places=10)


if __name__ == "__main__":
    unittest.main()
