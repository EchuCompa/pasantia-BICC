import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from asvFormula.ASV import ASV, ApproximateASV
from asvFormula.genericASV import asvFromTopologicalOrders


class TestASVCalculation(unittest.TestCase):
    def test_target_projection_preserves_precedence_and_uniform_feature_orders(self):
        graph = nx.DiGraph([("X", "Y"), ("Y", "Z")])
        graph.add_node("W")
        values = {node: [0, 1] for node in graph}
        with patch("asvFormula.ASV.obtainDecisionTreeDigraph"):
            asv = ASV(graph, None, None, values, "Y")

        full_orders = list(nx.all_topological_sorts(graph))
        projected = {tuple(node for node in order if node != "Y") for order in full_orders}
        feature_orders = list(nx.all_topological_sorts(asv.dag))
        self.assertEqual({tuple(order) for order in feature_orders}, projected)
        self.assertEqual(len(feature_orders), 3)
        self.assertNotIn("Y", asv.dag)

        def value(coalition):
            return np.array([float({"X", "W"} <= coalition)])

        asv.meanPredictionForEquivalenceClass = lambda fixed: value(set(fixed))
        for feature in asv.dag:
            np.testing.assert_allclose(
                asv.asvForFeature(feature, None),
                asvFromTopologicalOrders(feature, feature_orders, value),
            )
        np.testing.assert_allclose(asv.asvForFeature("X", None), [1 / 3])

    def test_projected_non_forest_is_rejected_by_exact_asv(self):
        graph = nx.DiGraph([("A", "Y"), ("B", "Y"), ("Y", "C"), ("Y", "D")])
        values = {node: [0, 1] for node in graph}
        with patch("asvFormula.ASV.obtainDecisionTreeDigraph"):
            asv = ASV(graph, None, None, values, "Y")
            approximate = ApproximateASV(graph, None, None, values, "Y", numTopologicalOrders=1)
        self.assertEqual(
            set(asv.dag.edges()), {("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")}
        )
        with self.assertRaisesRegex(ValueError, "directed forest"):
            asv.asvForFeature("A", None)
        with patch("asvFormula.ASV.mcmcTopoSorts", return_value=[["A", "B", "C", "D"]]) as sampler:
            approximate.equivalenceClasses("A")
        sampler.assert_called_once_with(approximate.dag, 1)

    def test_projection_preserves_orders_for_small_dags(self):
        nodes = ["X", "Y", "Z", "W"]
        possible_edges = [(nodes[i], nodes[j]) for i in range(4) for j in range(i + 1, 4)]
        for mask in range(1 << len(possible_edges)):
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edge for i, edge in enumerate(possible_edges) if mask & (1 << i))
            values = {node: [0, 1] for node in nodes}
            with patch("asvFormula.ASV.obtainDecisionTreeDigraph"):
                asv = ASV(graph, None, None, values, "Y")
            expected = {
                tuple(node for node in order if node != "Y")
                for order in nx.all_topological_sorts(graph)
            }
            self.assertEqual({tuple(order) for order in nx.all_topological_sorts(asv.dag)},
                             expected, msg=f"Original edges: {list(graph.edges())}")

    def test_projection_preserves_paths_through_multiple_excluded_nodes(self):
        graph = nx.DiGraph([("X", "L"), ("L", "Y"), ("Y", "Z")])
        values = {node: [0, 1] for node in ("X", "Y", "Z")}
        with patch("asvFormula.ASV.obtainDecisionTreeDigraph"):
            asv = ASV(graph, None, None, values, "Y")
        self.assertEqual(list(nx.all_topological_sorts(asv.dag)), [["X", "Z"]])
        self.assertEqual(set(graph.nodes()), {"X", "L", "Y", "Z"})

    def test_missing_model_feature_is_rejected(self):
        graph = nx.DiGraph([("X", "Y")])
        values = {node: [0, 1] for node in ("X", "Y", "Z")}
        with self.assertRaisesRegex(ValueError, "Features missing from the causal DAG"):
            ASV(graph, None, None, values, "Y")

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
