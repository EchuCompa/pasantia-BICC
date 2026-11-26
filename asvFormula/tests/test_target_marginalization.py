import unittest
from itertools import product
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

from asvFormula.ASV import ASV
from asvFormula.datasetManipulation import initializeDataAndRemoveVariable
from asvFormula.genericASV import asvFromTopologicalOrders


class TestTargetMarginalization(unittest.TestCase):
    def test_target_remains_in_inference_but_not_model_or_feature_dag(self):
        network = BayesianNetwork([("Y", "X")])
        network.add_node("Z")
        network.add_cpds(
            TabularCPD("Y", 2, [[0.8], [0.2]], state_names={"Y": ["y0", "y1"]}),
            TabularCPD(
                "X",
                2,
                [[1.0, 0.0], [0.0, 1.0]],
                evidence=["Y"],
                evidence_card=[2],
                state_names={"X": ["x0", "x1"], "Y": ["y0", "y1"]},
            ),
            TabularCPD("Z", 2, [[0.5], [0.5]], state_names={"Z": ["z0", "z1"]}),
        )
        network.check_model()

        inference, values_per_feature, encoded_data, model, _ = initializeDataAndRemoveVariable(
            network, "Y", numberOfSamples=40, treeMaxDepth=2, seed=0
        )

        self.assertIn("Y", network.nodes())
        self.assertAlmostEqual(
            inference.query(variables=["X"]).get_value(X="x1"), 0.2
        )
        self.assertEqual(model.n_features_in_, 2)

        instance = encoded_data.drop(columns="Y").iloc[0]
        asv = ASV(network, model, inference, values_per_feature, "Y", instance=instance)
        self.assertEqual(set(asv.dag.nodes()), {"X", "Z"})
        self.assertNotIn("Y", model.feature_names_in_)
        np.testing.assert_allclose(
            asv.meanPredictionForEquivalenceClass([]), [0.8, 0.2]
        )

    def test_mediator_target_is_marginalized_and_precedence_is_preserved(self):
        network = BayesianNetwork([("X", "Y"), ("Y", "Z")])
        states = {"X": ["x0", "x1"], "Y": ["y0", "y1"], "Z": ["z0", "z1"]}
        network.add_cpds(
            TabularCPD("X", 2, [[0.6], [0.4]], state_names={"X": states["X"]}),
            TabularCPD("Y", 2, [[0.8, 0.3], [0.2, 0.7]], evidence=["X"],
                       evidence_card=[2], state_names={"X": states["X"], "Y": states["Y"]}),
            TabularCPD("Z", 2, [[0.7, 0.2], [0.3, 0.8]], evidence=["Y"],
                       evidence_card=[2], state_names={"Y": states["Y"], "Z": states["Z"]}),
        )
        network.check_model()
        inference, values, data, model, _ = initializeDataAndRemoveVariable(
            network, "Y", numberOfSamples=100, treeMaxDepth=2, seed=3
        )
        instance = data.drop(columns="Y").iloc[0]
        asv = ASV(network, model, inference, values, "Y", instance=instance)
        self.assertEqual(list(nx.all_topological_sorts(asv.dag)), [["X", "Z"]])

        joint = VariableElimination(network).query(variables=["X", "Z"])

        def exact_value(coalition):
            total = np.zeros(len(model.classes_))
            mass = 0.0
            for x, z in product(range(2), repeat=2):
                row = {"X": x, "Z": z}
                if any(row[feature] != instance[feature] for feature in coalition):
                    continue
                probability = joint.get_value(X=values["X"][x], Z=values["Z"][z])
                prediction = model.predict(pd.DataFrame([row]))[0]
                total += probability * (model.classes_ == prediction)
                mass += probability
            return total / mass

        orders = list(nx.all_topological_sorts(asv.dag))
        with patch.object(inference, "query", wraps=inference.query) as query:
            for coalition in ([], ["X"], ["Z"], ["X", "Z"]):
                np.testing.assert_allclose(asv.meanPredictionForEquivalenceClass(coalition),
                                           exact_value(coalition))
            for feature in ("X", "Z"):
                np.testing.assert_allclose(
                    asv.asvForFeature(feature, instance),
                    asvFromTopologicalOrders(feature, orders, exact_value),
                )
            for call in query.call_args_list:
                self.assertNotIn("Y", call.kwargs.get("evidence", {}))
                self.assertNotIn("Y", call.kwargs["variables"])


if __name__ == "__main__":
    unittest.main()
