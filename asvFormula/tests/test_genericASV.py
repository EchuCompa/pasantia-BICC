import unittest
from itertools import permutations

import numpy as np
import networkx as nx

from asvFormula.genericASV import asvFromTopologicalOrders


class TestGenericASV(unittest.TestCase):
    def test_generic_oracle_reproduces_shapley_and_causal_chain_values(self):
        def value(evidence):
            if "Y" in evidence:
                return 1.0
            if "X" in evidence:
                return 0.9
            return 0.5

        orders = list(permutations(("X", "Y")))
        shapley = {
            feature: asvFromTopologicalOrders(feature, orders, value).item()
            for feature in ("X", "Y")
        }
        causal_order = [("X", "Y")]
        asymmetric = {
            feature: asvFromTopologicalOrders(feature, causal_order, value).item()
            for feature in ("X", "Y")
        }

        self.assertAlmostEqual(shapley["X"], 0.2)
        self.assertAlmostEqual(shapley["Y"], 0.3)
        self.assertAlmostEqual(asymmetric["X"], 0.4)
        self.assertAlmostEqual(asymmetric["Y"], 0.1)
        self.assertAlmostEqual(sum(shapley.values()), 0.5)
        self.assertAlmostEqual(sum(asymmetric.values()), 0.5)

    def test_caches_repeated_coalition_values(self):
        calls = []

        def value(evidence):
            calls.append(evidence)
            return float(len(evidence))

        orders = list(permutations(("X", "Y", "Z")))
        asvFromTopologicalOrders("X", orders, value)

        self.assertEqual(len(calls), 8)
        self.assertEqual(
            set(calls),
            {
                frozenset(),
                frozenset({"X"}),
                frozenset({"Y"}),
                frozenset({"Z"}),
                frozenset({"X", "Y"}),
                frozenset({"X", "Z"}),
                frozenset({"Y", "Z"}),
                frozenset({"X", "Y", "Z"}),
            },
        )

    def test_rejects_empty_orders_and_missing_feature(self):
        value = lambda evidence: 0.0

        with self.assertRaises(ValueError):
            asvFromTopologicalOrders("X", [], value)
        with self.assertRaises(ValueError):
            asvFromTopologicalOrders("X", [("Y",)], value)

    def test_vector_contributions_sum_to_prediction_difference_on_a_forest(self):
        dag = nx.DiGraph([(0, 1), (2, 3)])
        orders = list(nx.all_topological_sorts(dag))

        def value(coalition):
            p = (1 + sum(node + 1 for node in coalition)) / 11
            return np.array([1 - p, p, 0.0])

        contributions = [
            asvFromTopologicalOrders(node, orders, value) for node in dag
        ]
        np.testing.assert_allclose(
            sum(contributions), value(frozenset(dag)) - value(frozenset())
        )
        self.assertTrue(all(result.shape == (3,) for result in contributions))
        self.assertLess(contributions[0][0], 0)
        self.assertGreater(contributions[0][1], 0)

    def test_rejects_inconsistent_orders_and_oracle_outputs(self):
        for orders in ([("X", "X")], [("X", "Y"), ("X", "Z")]):
            with self.assertRaises(ValueError):
                asvFromTopologicalOrders("X", orders, lambda _: 0.0)
        with self.assertRaises(ValueError):
            asvFromTopologicalOrders("X", [("X",)],
                                     lambda s: [1, 2] if s else [0])
        with self.assertRaises(ValueError):
            asvFromTopologicalOrders("X", [("X",)], lambda _: float("nan"))


if __name__ == "__main__":
    unittest.main()
