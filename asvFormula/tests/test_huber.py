import itertools
import random
import unittest
from collections import Counter, deque
from unittest.mock import patch

import networkx as nx

from asvFormula.topoSorts.huber import _BoundingChain, huberTopoSort, huberTopoSorts


class TestHuberSampler(unittest.TestCase):
    def test_bounding_chain_couples_all_small_dag_states(self):
        size = 4
        possible_edges = list(itertools.combinations(range(size), 2))
        for mask in range(1 << len(possible_edges)):
            graph = nx.DiGraph()
            graph.add_nodes_from(range(size))
            graph.add_edges_from(edge for i, edge in enumerate(possible_edges) if mask & (1 << i))
            closure = nx.transitive_closure_dag(graph)
            successors = [set(closure.successors(node)) for node in range(size)]
            extensions = list(nx.all_topological_sorts(graph))
            initial = _BoundingChain(successors)
            pending = deque([(tuple(initial.slots), initial.active)])
            seen = set(pending)

            while pending:
                slots, active = pending.popleft()
                bounds = {label: position for position, label in enumerate(slots) if label is not None}
                bounded = [
                    order for order in extensions
                    if all(order.index(label) <= position for label, position in bounds.items())
                ]
                self.assertTrue(bounded)

                for position in range(size - 1):
                    for coin in (0, 1):
                        bound = _BoundingChain(successors)
                        bound.slots, bound.active = list(slots), active
                        bound.step(position, coin)
                        state = (tuple(bound.slots), bound.active)
                        if state not in seen:
                            seen.add(state)
                            pending.append(state)
                        new_bounds = {
                            label: slot for slot, label in enumerate(bound.slots) if label is not None
                        }
                        for order in bounded:
                            actual = order.copy()
                            replay = _BoundingChain(successors)
                            replay.slots, replay.active = list(slots), active
                            replay.step(position, coin, actual)
                            self.assertEqual((tuple(replay.slots), replay.active), state)
                            self.assertTrue(all(actual.index(label) <= slot for label, slot in new_bounds.items()))
                            self.assertIn(actual, extensions)
                            if bound.coalesced():
                                self.assertEqual(actual, bound.result())

                    for order in bounded:
                        outcomes = []
                        for coin in (0, 1):
                            actual = order.copy()
                            replay = _BoundingChain(successors)
                            replay.slots, replay.active = list(slots), active
                            replay.step(position, coin, actual)
                            outcomes.append(actual)
                        swapped = order.copy()
                        if swapped[position + 1] not in successors[swapped[position]]:
                            swapped[position], swapped[position + 1] = swapped[position + 1], swapped[position]
                        self.assertCountEqual(map(tuple, outcomes), [tuple(order), tuple(swapped)])

    def test_failed_block_is_replayed_with_its_original_random_choices(self):
        graph = nx.empty_graph(2, create_using=nx.DiGraph)
        with patch("asvFormula.topoSorts.huber.random.randrange", return_value=0):
            with patch("asvFormula.topoSorts.huber.random.getrandbits", side_effect=[0, 1, 0]):
                self.assertEqual(huberTopoSort(graph), [1, 0])

    def test_small_dag_frequencies_and_consecutive_samples(self):
        graph = nx.DiGraph([("a", "c")])
        graph.add_node("b")
        expected = set(map(tuple, nx.all_topological_sorts(graph)))
        random.seed(487)

        samples = list(map(tuple, huberTopoSorts(graph, 9000)))
        counts = Counter(samples)
        pairs = Counter(zip(samples[::2], samples[1::2]))

        self.assertEqual(set(counts), expected)
        for order in expected:
            self.assertLess(abs(counts[order] / len(samples) - 1 / 3), 0.025)
        for first in expected:
            for second in expected:
                self.assertLess(abs(pairs[first, second] / 4500 - 1 / 9), 0.025)

    def test_general_dag_and_non_orderable_node_labels(self):
        graph = nx.DiGraph([(0, "b"), (0, "c"), ("b", (1,)), ("c", (1,))])
        expected = set(map(tuple, nx.all_topological_sorts(graph)))
        random.seed(7)
        samples = huberTopoSorts(graph, 100)
        self.assertEqual(set(map(tuple, samples)), expected)
        self.assertEqual(len(samples), 100)

    def test_empty_singleton_counts_and_invalid_graphs(self):
        self.assertEqual(huberTopoSorts(nx.DiGraph(), 2), [[], []])
        single = nx.DiGraph()
        single.add_node("x")
        self.assertEqual(huberTopoSorts(single, 2), [["x"], ["x"]])
        self.assertEqual(huberTopoSorts(single, 0), [])
        with self.assertRaises(ValueError):
            huberTopoSorts(single, -1)
        with self.assertRaises(TypeError):
            huberTopoSorts(single, 1.5)
        for graph in (nx.DiGraph([("a", "a")]), nx.DiGraph([("a", "b"), ("b", "a")]), nx.Graph()):
            with self.subTest(graph=graph):
                with self.assertRaises(ValueError):
                    huberTopoSorts(graph, 1)


if __name__ == "__main__":
    unittest.main()
