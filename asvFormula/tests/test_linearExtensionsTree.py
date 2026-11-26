import random
import unittest
from collections import Counter
from itertools import product

import networkx as nx

from asvFormula.topoSorts.randomTopoSortsGeneration import (
    mcmcTopoSorts,
    obtainRandomStartNodes,
    randomTopoSorts,
    randomTopoSortsAtkinson,
    randomTopoSortsLegacy,
)
from asvFormula.topoSorts.linearExtensionsTree import (
    allPolyforestTopoSortsAtkinson,
    linearExtensionPositionCounts,
)
from asvFormula.topoSorts.topoSortsCalc import allPolyTopoSorts


class TestLinearExtensionsTree(unittest.TestCase):
    def test_position_counts_match_enumeration_on_oriented_trees(self):
        for size in range(1, 8):
            for seed in range(3):
                random.seed(size * 10 + seed)
                tree = nx.DiGraph()
                tree.add_nodes_from(range(size))
                order = list(range(size))
                random.shuffle(order)
                rank = {node: index for index, node in enumerate(order)}
                for node in range(1, size):
                    parent = random.randrange(node)
                    edge = (parent, node) if rank[parent] < rank[node] else (node, parent)
                    tree.add_edge(*edge)

                extensions = list(nx.all_topological_sorts(tree))
                for node in tree:
                    expected = [
                        sum(extension.index(node) == position for extension in extensions)
                        for position in range(size)
                    ]
                    self.assertEqual(linearExtensionPositionCounts(tree, node), expected)

    def test_count_matches_enumeration_on_disconnected_forests(self):
        forest = nx.DiGraph(
            [("a", "b"), ("c", "d"), ("e", "f"), ("e", "g")]
        )

        self.assertEqual(
            allPolyforestTopoSortsAtkinson(forest),
            len(list(nx.all_topological_sorts(forest))),
        )

    def test_atkinson_sampler_is_uniform_on_a_small_star(self):
        star = nx.DiGraph([("r", "a"), ("r", "b"), ("r", "c")])
        random.seed(42)

        samples = randomTopoSortsAtkinson(star, 6000)
        frequencies = Counter(map(tuple, samples))

        self.assertEqual(len(frequencies), 6)
        self.assertTrue(
            all(abs(count / len(samples) - 1 / 6) < 0.035 for count in frequencies.values())
        )

    def test_rejects_graphs_outside_the_forest_domain(self):
        diamond = nx.DiGraph(
            [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
        )

        with self.assertRaises(ValueError):
            allPolyforestTopoSortsAtkinson(diamond)
        with self.assertRaises(ValueError):
            allPolyTopoSorts(diamond)
        with self.assertRaises(ValueError):
            randomTopoSorts(diamond, 1)

    def test_empty_graph_has_one_linear_extension(self):
        empty = nx.DiGraph()

        self.assertEqual(allPolyTopoSorts(empty), 1)
        self.assertEqual(allPolyforestTopoSortsAtkinson(empty), 1)

    def test_all_orientations_of_small_unlabelled_trees_and_remaining_forests(self):
        for size in range(2, 6):
            for undirected in nx.nonisomorphic_trees(size):
                edges = list(undirected.edges())
                for directions in product((False, True), repeat=len(edges)):
                    tree = nx.DiGraph()
                    tree.add_nodes_from(undirected)
                    tree.add_edges_from(
                        (v, u) if reverse else (u, v)
                        for (u, v), reverse in zip(edges, directions)
                    )
                    extensions = list(nx.all_topological_sorts(tree))
                    for node in tree:
                        expected = [
                            sum(order.index(node) == position for order in extensions)
                            for position in range(size)
                        ]
                        self.assertEqual(linearExtensionPositionCounts(tree, node), expected)
                    for mask in range(1 << size):
                        remaining = tree.subgraph(
                            [node for node in tree if mask & (1 << node)]
                        )
                        self.assertEqual(
                            allPolyforestTopoSortsAtkinson(remaining),
                            len(list(nx.all_topological_sorts(remaining))),
                        )

    def test_samplers_are_reproducible_and_validate_counts(self):
        forest = nx.DiGraph([(0, 1), (2, 3)])
        for sampler in (randomTopoSorts, randomTopoSortsLegacy, mcmcTopoSorts):
            random.seed(123)
            first = sampler(forest, 30)
            random.seed(123)
            self.assertEqual(first, sampler(forest, 30))
            self.assertEqual(sampler(forest, 0), [])
            for count in (-1, 1.5, True):
                with self.assertRaises(ValueError):
                    sampler(forest, count)
        with self.assertRaises(ValueError):
            mcmcTopoSorts(nx.DiGraph([(0, 0)]), 1)
        with self.assertRaises(ValueError):
            mcmcTopoSorts(forest, 1, -1)

    def test_weighted_selection_supports_large_integer_and_zero_weights(self):
        random.seed(123)
        selections = obtainRandomStartNodes(20, [0, 10**400], ["never", "always"])
        self.assertEqual(selections, ["always"] * 20)

    def test_sampler_frequencies_match_small_forest_enumeration(self):
        graph = nx.DiGraph([(0, 2)])
        graph.add_node(1)
        exact = set(map(tuple, nx.all_topological_sorts(graph)))
        for sampler in (randomTopoSorts, randomTopoSortsLegacy, mcmcTopoSorts):
            random.seed(42)
            frequencies = Counter(map(tuple, sampler(graph, 3000)))
            self.assertEqual(set(frequencies), exact)
            for order in exact:
                self.assertLess(abs(frequencies[order] / 3000 - 1 / 3), 0.04)


if __name__ == "__main__":
    unittest.main()
