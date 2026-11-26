import unittest

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

from asvFormula.ASV import ASV
from asvFormula.datasetManipulation import initializeDataAndRemoveVariable


class TestTargetMarginalization(unittest.TestCase):
    def test_target_remains_in_inference_but_not_model_or_feature_dag(self):
        network = BayesianNetwork([("X", "Y")])
        network.add_cpds(
            TabularCPD("X", 2, [[0.5], [0.5]], state_names={"X": ["x0", "x1"]}),
            TabularCPD(
                "Y",
                2,
                [[1.0, 0.0], [0.0, 1.0]],
                evidence=["X"],
                evidence_card=[2],
                state_names={"X": ["x0", "x1"], "Y": ["y0", "y1"]},
            ),
        )
        network.check_model()

        inference, values_per_feature, _, model, _ = initializeDataAndRemoveVariable(
            network, "Y", numberOfSamples=40, treeMaxDepth=2, seed=0
        )

        self.assertIn("Y", network.nodes())
        self.assertAlmostEqual(
            inference.query(variables=["Y"]).get_value(Y="y1"), 0.5
        )
        self.assertEqual(model.n_features_in_, 1)

        asv = ASV(network, model, inference, values_per_feature, "Y")
        self.assertEqual(set(asv.dag.nodes()), {"X"})
        self.assertNotIn("Y", asv.modelAsTree.nodes())


if __name__ == "__main__":
    unittest.main()
