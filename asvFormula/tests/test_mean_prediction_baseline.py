import unittest

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn.tree import DecisionTreeClassifier

from asvFormula.bayesianNetworks.bayesianNetwork import (
    DecisionTreeDigraph,
    meanPredictionForDTinBNWithEvidence,
    obtainDecisionTreeDigraph,
)
from asvFormula.bayesianNetworks.pathCondition import PathCondition


class TestMeanPredictionBaseline(unittest.TestCase):
    def test_constant_tree_with_one_feature(self):
        classifier = DecisionTreeClassifier(random_state=0).fit([[0], [1]], [1, 1])
        tree = obtainDecisionTreeDigraph(classifier, ["X"])
        self.assertEqual(len(tree), 1)
        np.testing.assert_array_equal(tree.nodePrediction(0), [1])

    def test_exact_conditional_probabilities_with_zero_probability_paths(self):
        network = BayesianNetwork([("X", "Y")])
        network.add_cpds(
            TabularCPD("X", 2, [[0.25], [0.75]]),
            TabularCPD("Y", 2, [[1.0, 0.0], [0.0, 1.0]],
                       evidence=["X"], evidence_card=[2]),
        )
        network.check_model()
        inference = VariableElimination(network)
        tree = DecisionTreeDigraph()
        tree.add_node(0, feature="Y", threshold=0.5, classesProbabilities=np.array([0.5, 0.5]))
        tree.add_node(1, classesProbabilities=np.array([1.0, 0.0]))
        tree.add_node(2, classesProbabilities=np.array([0.0, 1.0]))
        tree.add_edges_from([(0, 1), (0, 2)])
        values = {"X": [0, 1], "Y": [0, 1]}
        for evidence, expected in (({}, [0.25, 0.75]), ({"X": 0}, [1, 0]),
                                   ({"X": 1}, [0, 1]), ({"Y": 0}, [1, 0])):
            with self.subTest(evidence=evidence):
                actual = meanPredictionForDTinBNWithEvidence(
                    tree, 0, inference, PathCondition(values), evidence
                )
                np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
