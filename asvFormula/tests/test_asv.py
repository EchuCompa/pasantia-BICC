import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from asvFormula.ASV import ASV, ApproximateASV


class TestASVCalculation(unittest.TestCase):
    def test_approximate_sampler_uses_weighted_sampler_for_polyforests(self):
        dag = nx.DiGraph([("a", "b")])
        orders = [["a", "b"]] * 3
        asv = ApproximateASV.__new__(ApproximateASV)
        asv.dag = dag
        asv.numberOfToposorts = len(orders)

        with patch("asvFormula.ASV.randomTopoSorts", return_value=orders) as sampler:
            with patch("asvFormula.ASV.mcmcTopoSorts") as mcmc:
                asv.equivalenceClasses("a")

        sampler.assert_called_once_with(dag, len(orders))
        mcmc.assert_not_called()

    def test_approximate_rejects_nonpositive_sample_count(self):
        with patch.object(ASV, "__init__", return_value=None):
            for count in (0, -1):
                with self.subTest(count=count):
                    with self.assertRaises(ValueError):
                        ApproximateASV(None, None, None, {}, "", numTopologicalOrders=count)

    def test_approximate_sampler_uses_mcmc_for_non_polyforest_dags(self):
        dag = nx.DiGraph(
            [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
        )
        orders = [
            ["a", "b", "c", "d"],
            ["a", "c", "b", "d"],
            ["a", "b", "c", "d"],
        ]
        asv = ApproximateASV.__new__(ApproximateASV)
        asv.dag = dag
        asv.numberOfToposorts = len(orders)

        with patch("asvFormula.ASV.mcmcTopoSorts", return_value=orders) as mcmc:
            with patch("asvFormula.ASV.randomTopoSorts") as tree_sampler:
                asv.equivalenceClasses("a")

        mcmc.assert_called_once_with(dag, len(orders))
        tree_sampler.assert_not_called()

    def test_approximate_normalizes_by_sample_count_not_exact_count(self):
        dag = nx.DiGraph(
            [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
        )
        asv = ApproximateASV.__new__(ApproximateASV)
        asv.dag = dag
        asv.valuesPerFeature = {"a": [0, 1]}
        asv.numberOfToposorts = 3
        asv.meanPredictionForEquivalenceClass = lambda fixed: (
            np.array([0.5, 0.5]) if not fixed else np.array([0.0, 1.0])
        )

        value = asv.asvForFeature("a", instance=None)

        np.testing.assert_allclose(value, [-0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
