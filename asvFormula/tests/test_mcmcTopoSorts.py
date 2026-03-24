import unittest
import random
from parameterized import parameterized
import networkx as nx

from asvFormula.digraph import *
from asvFormula.topoSorts.randomTopoSortsGeneration import mcmcTopoSort, mcmcTopoSorts
from asvFormula.topoSorts.utils import isTopologicalSort


class TestMcmcTopoSorts(unittest.TestCase):

    seed = 42

    # --- Validity tests ---

    @parameterized.expand([
        ("empty_graph",      emptyGraph(5)),
        ("naive_bayes",      naiveBayes(6)),
        ("naive_bayes_path", naiveBayesWithPath(4, 3)),
        ("multiple_paths",   multiplePaths(2, 3)),
        ("balanced_tree",    balancedTree(3, 2)),
    ])
    def test_mcmcSortsAreValid(self, name, graph):
        """Every sort returned by mcmcTopoSorts must be a valid topological sort."""
        random.seed(self.seed)
        nSamples = 50
        topoSorts = mcmcTopoSorts(graph, nSamples)

        self.assertEqual(len(topoSorts), nSamples)
        for order in topoSorts:
            self.assertTrue(
                isTopologicalSort(graph, order),
                f"Generated sort {order} is not a valid topological sort for graph with edges {list(graph.edges())}"
            )

    # --- Distribution tests ---

    @parameterized.expand([
        ("empty_graph",    emptyGraph(4)),
        ("naive_bayes",    naiveBayes(5)),
        ("multiple_paths", multiplePaths(2, 3)),
        ("balanced_tree",  balancedTree(3, 2)),
    ])
    def test_mcmcDistributionIsApproximatelyUniform(self, name, graph):
        """The ratio of unique MCMC samples should be close to that of true uniform sampling.

        Following the same approach as testRandomTopoSortsGeneration: we draw nSamples
        equal to the total number of linear extensions and compare the fraction of
        distinct orderings produced by MCMC against that of independent uniform sampling.
        """
        random.seed(self.seed)
        allTopoSorts = list(nx.all_topological_sorts(graph))
        nSamples = len(allTopoSorts)

        # Use 5*n^3 steps to guarantee good mixing, especially for unconstrained graphs
        n = len(graph.nodes())
        stepsPerSample = 5 * n * n * n
        mcmcOrders = mcmcTopoSorts(graph, nSamples, stepsPerSample=stepsPerSample)
        sampledToposorts = random.choices(allTopoSorts, k=nSamples)

        mcmcUniqueRatio    = len(set(tuple(o) for o in mcmcOrders))     / nSamples
        sampledUniqueRatio = len(set(tuple(o) for o in sampledToposorts)) / nSamples

        self.assertAlmostEqual(
            mcmcUniqueRatio, sampledUniqueRatio, delta=0.1,
            msg=(
                f"MCMC unique ratio {mcmcUniqueRatio:.3f} differs too much from "
                f"uniform sampling ratio {sampledUniqueRatio:.3f}"
            )
        )

    # --- Edge-case tests ---

    def test_mcmcTopoSort_returns_single_valid_sort(self):
        """mcmcTopoSort should return exactly one valid topological sort."""
        random.seed(self.seed)
        graph = naiveBayes(5)
        order = mcmcTopoSort(graph)
        self.assertIsInstance(order, list)
        self.assertEqual(len(order), len(graph.nodes()))
        self.assertTrue(isTopologicalSort(graph, order))

    def test_single_node_graph(self):
        """A single-node graph has exactly one trivial topological sort."""
        graph = nx.DiGraph()
        graph.add_node(0)
        topoSorts = mcmcTopoSorts(graph, 5)
        self.assertEqual(len(topoSorts), 5)
        for order in topoSorts:
            self.assertEqual(order, [0])

    def test_linear_chain_has_unique_toposort(self):
        """A strict linear chain a→b→c→… has exactly one valid topological sort."""
        graph = nx.DiGraph()
        for i in range(5):
            graph.add_node(i)
        for i in range(4):
            graph.add_edge(i, i + 1)

        expected = list(range(5))
        topoSorts = mcmcTopoSorts(graph, 10)
        for order in topoSorts:
            self.assertEqual(order, expected)

    def test_custom_steps_per_sample(self):
        """Passing a custom stepsPerSample should work without error."""
        random.seed(self.seed)
        graph = naiveBayes(4)
        topoSorts = mcmcTopoSorts(graph, 10, stepsPerSample=50)
        self.assertEqual(len(topoSorts), 10)
        for order in topoSorts:
            self.assertTrue(isTopologicalSort(graph, order))

    def test_all_nodes_appear_in_each_sort(self):
        """Every returned ordering must contain exactly the same nodes as the graph."""
        random.seed(self.seed)
        graph = balancedTree(3, 2)
        expected_nodes = set(graph.nodes())
        for order in mcmcTopoSorts(graph, 20):
            self.assertEqual(set(order), expected_nodes)

