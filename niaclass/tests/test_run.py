from unittest import TestCase
from niaclass import NiaClass
import pandas as pd
import os


class RuleTestCase(TestCase):
    def test_fit1_works_fine(self):
        src = os.path.dirname(os.path.abspath(__file__)) + "/test_files/dataset.csv"
        data = pd.read_csv(src, header=None)
        y = data.pop(data.columns[len(data.columns) - 1])
        x = data
        nc = NiaClass(10, 100, "accuracy", "DifferentialEvolution")
        self.assertEqual(nc._NiaClass__pop_size, 10)
        self.assertEqual(nc._NiaClass__num_evals, 100)
        self.assertEqual(nc._NiaClass__score_func_name, "accuracy")
        self.assertEqual(nc._NiaClass__algo, "DifferentialEvolution")
        self.assertEqual(1, len(nc._NiaClass__algo_args))
        self.assertEqual(10, nc._NiaClass__algo_args["population_size"])
        self.assertIsNone(nc._NiaClass__rules)

        nc.fit(x, y)
        self.assertIsNotNone(nc._NiaClass__rules)

    def test_fit2_works_fine(self):
        src = os.path.dirname(os.path.abspath(__file__)) + "/test_files/dataset.csv"
        data = pd.read_csv(src, header=None)
        y = data.pop(data.columns[len(data.columns) - 1])
        x = data
        nc = NiaClass(
            10, 100, "accuracy", "FireflyAlgorithm", alpha=0.5, betamin=0.2, gamma=1.0
        )
        self.assertEqual(nc._NiaClass__pop_size, 10)
        self.assertEqual(nc._NiaClass__num_evals, 100)
        self.assertEqual(nc._NiaClass__score_func_name, "accuracy")
        self.assertEqual(nc._NiaClass__algo, "FireflyAlgorithm")
        self.assertEqual(4, len(nc._NiaClass__algo_args))
        self.assertEqual(10, nc._NiaClass__algo_args["population_size"])
        self.assertEqual(0.5, nc._NiaClass__algo_args["alpha"])
        self.assertEqual(0.2, nc._NiaClass__algo_args["betamin"])
        self.assertEqual(1.0, nc._NiaClass__algo_args["gamma"])
        self.assertIsNone(nc._NiaClass__rules)

        nc.fit(x, y)
        self.assertIsNotNone(nc._NiaClass__rules)

    def test_predict_works_fine(self):
        src = os.path.dirname(os.path.abspath(__file__)) + "/test_files/dataset.csv"
        data = pd.read_csv(src, header=None)
        y = data.pop(data.columns[len(data.columns) - 1])
        x = data
        nc = NiaClass(10, 100, "accuracy", "DifferentialEvolution")
        nc.fit(x, y)

        predictions = nc.predict(x)
        self.assertEqual(len(y), len(predictions))
