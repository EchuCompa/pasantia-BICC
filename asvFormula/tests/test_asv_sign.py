import unittest

import networkx as nx
import numpy as np

from asvFormula.ASV import ASV


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


if __name__ == "__main__":
    unittest.main()
