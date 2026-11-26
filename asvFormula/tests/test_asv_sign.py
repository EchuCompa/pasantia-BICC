import unittest

import networkx as nx
import numpy as np

from asvFormula.ASV import ASV
from asvFormula.genericASV import asvFromTopologicalOrders


class FixedASV(ASV):
    def __init__(self, dag):
        self.dag = dag
        self.valuesPerFeature = {node: [0, 1] for node in dag.nodes}

    def equivalenceClasses(self, feature):
        return [(["x", "y"], 1)]

    def meanPredictionForEquivalenceClass(self, fixedFeatures):
        if fixedFeatures:
            return np.array([0.0, 1.0])
        return np.array([0.5, 0.5])


class TestASVSign(unittest.TestCase):
    def test_feature_contribution_uses_prediction_after_adding_feature(self):
        asv = FixedASV(nx.DiGraph([("x", "y")]))

        value = asv.asvForFeature("x", instance=None)

        np.testing.assert_allclose(value, [-0.5, 0.5])

    def test_output_cardinality_is_prediction_cardinality(self):
        asv = FixedASV(nx.DiGraph([("x", "y")]))
        asv.valuesPerFeature["x"] = [0, 1, 2]
        np.testing.assert_allclose(asv.asvForFeature("x", None), [-0.5, 0.5])

    def test_exact_classes_agree_with_enumerated_orders(self):
        dag = nx.DiGraph([("a", "b"), ("a", "c")])
        asv = ASV.__new__(ASV)
        asv.dag = dag
        asv.valuesPerFeature = {node: [0, 1, 2] for node in dag}

        def value(coalition):
            p = (1 + sum({"a": 1, "b": 2, "c": 3}[n] for n in coalition)) / 8
            return np.array([1 - p, p])

        asv.meanPredictionForEquivalenceClass = lambda fixed: value(frozenset(fixed))
        orders = list(nx.all_topological_sorts(dag))
        contributions = []
        for node in dag:
            actual = asv.asvForFeature(node, None)
            expected = asvFromTopologicalOrders(node, orders, value)
            np.testing.assert_allclose(actual, expected)
            contributions.append(actual)
        np.testing.assert_allclose(sum(contributions), value(frozenset(dag)) - value(frozenset()))


if __name__ == "__main__":
    unittest.main()
