import unittest
from parameterized import parameterized
import unittest
import networkx as nx
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from asvFormula.digraph import *
import topoSorts.topoSortsCalc as tp
from asvFormula.topoSorts.toposPositions import naivePositionsInToposorts, positionsInToposorts


class TestTopoSorts(unittest.TestCase):

    @parameterized.expand([
        ("path_graph", multiplePaths(1,5)),
        ('naive_bayes', naiveBayesWithPath(5, 4)),
        ("simple_tree", balancedTree(2, 3)),
        ("complex_tree", balancedTree(2, 2))
    ])
    def test_allToposPositions(self, name, graph):
        nodes_to_test = self.pathToLeaf(graph, 0)
        allTopos = nx.all_topological_sorts(graph)

        for node in nodes_to_test:    
            allToposPositionsNaive = naivePositionsInToposorts(node, graph, allTopos)
            allToposPositions = positionsInToposorts(node, graph)
            self.assertEqual(allToposPositionsNaive.keys(), allToposPositions.keys())
            for pos in allToposPositionsNaive.keys():
                self.assertEqual(allToposPositionsNaive[pos], allToposPositions[pos])

    def pathToLeaf(self, tree : nx.DiGraph, start_node):
        path = [start_node]
        current = start_node
        while not isLeaf(current, tree):
            current = next(tree.successors(current))
            path.append(current)
        return path

if __name__ == '__main__':
    unittest.main()
