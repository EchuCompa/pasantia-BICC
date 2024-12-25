import unittest
from parameterized import parameterized
import networkx as nx

from asvFormula.digraph import *
from asvFormula.topoSorts import allPolyTopoSorts, allForestTopoSorts, TopoSortHasher
from asvFormula.classesSizes import equivalanceClassesSizesWithHashes, naiveEquivalenceClassesSizes

#These tests take a lot of time, they are more regression tests than unit tests.

class TestEquivalenceClasses(unittest.TestCase):

    @parameterized.expand(
        [("empty_graph", emptyGraph(8)),
        ('naive_bayes', naiveBayes(8))] + 
        [(f"naive_bayes_path_{path_length}", naiveBayesWithPath(7, path_length)) for path_length in range(1,5)] + 
        [(f"multiple_paths_{paths}", multiplePaths(paths, 4)) for paths in range(1,4)] + 
        [(f"balanced_tree_{b}", balancedTree(2, b)) for b in range(2,9)]
    )
    def test_equivalenceClassesSizesForForests(self, name, graph):
        allTopos = list(nx.all_topological_sorts(graph))

        for node in graph.nodes:
            self.assertEquivalenceClassForNode(graph, node, allTopos)


    def assertEquivalenceClassForNode(self, dag, feature_node, all_topo_sorts):
        equivalenceClasses = equivalanceClassesSizesWithHashes(dag, feature_node)
        naiveEquivalenceClasses = naiveEquivalenceClassesSizes(all_topo_sorts, feature_node, TopoSortHasher(dag, feature_node))

        self.assertEquals(len(equivalenceClasses), len(naiveEquivalenceClasses), f"The number of equivalence classes is different than expected")

        for eqClassHash, classInfo in naiveEquivalenceClasses.items():
            self.assertTrue(eqClassHash in equivalenceClasses, f"The equivalence class {eqClassHash} is not present in the recursive approach")
            self.assertEquals(classInfo[1], equivalenceClasses[eqClassHash][1], f"The sizes of the equivalence classes are not equal")


    def assertTopos(self, graph, allTopos):
        self.assertEquals(allForestTopoSorts(graph), len(allTopos))
        self.assertEquals(allPolyTopoSorts(graph), len(allTopos))

if __name__ == '__main__':
    unittest.main()
