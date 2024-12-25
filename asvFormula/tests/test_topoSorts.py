import unittest
from parameterized import parameterized
import networkx as nx

from asvFormula.digraph import *
import topoSorts.topoSortsCalc as tp
import topoSorts.topoSortsCalc_basic as tpBasic
from asvFormula.topoSorts.toposPositions import naivePositionsInToposorts, positionsInToposorts
from asvFormula.topoSorts.utils import allForestTopoSorts

class TestTopoSorts(unittest.TestCase):

    @parameterized.expand(
        [("empty_graph", emptyGraph(8)),
        ('naive_bayes', naiveBayes(8))] + 
        [(f"naive_bayes_path_{path_length}", naiveBayesWithPath(7, path_length)) for path_length in range(1,5)] + 
        [(f"multiple_paths_{paths}", multiplePaths(paths, 4)) for paths in range(1,4)] + 
        [(f"balanced_tree_{b}", balancedTree(2, b)) for b in range(2,9)]
    )
    def test_treeToposPositionsAndTopoSorts(self, name, graph):
        nodes_to_test = self.pathToLeaf(graph, 0)
        allTopos = list(nx.all_topological_sorts(graph))
        
        for node in nodes_to_test:    
            allToposPositionsNaive = naivePositionsInToposorts(node, graph, allTopos)
            allToposPositions = positionsInToposorts(node, graph)
            self.assertEqual(allToposPositionsNaive.keys(), allToposPositions.keys(),  f"Naive: {allToposPositionsNaive.keys()} and Recursive: {allToposPositions.keys()} positions are different")
            for pos in allToposPositionsNaive.keys():
                self.assertEqual(allToposPositionsNaive[pos], allToposPositions[pos],  f"Naive: {allToposPositionsNaive[pos]} and Recursive: {allToposPositions[pos]} have different values for position {pos} in node {node}")

        self.assertEqual(allForestTopoSorts(graph), len(allTopos))
    
    def pathToLeaf(self, tree : nx.DiGraph, start_node):
        path = [start_node]
        current = start_node
        while not isLeaf(current, tree):
            current = next(tree.successors(current))
            path.append(current)
        return path
    
    @parameterized.expand([ 
        (f"Level{level}_Branching{branching}_Parents{1}", level, branching, 1)
        for level in range(1, 3) for branching in range(1, 4)
    ])
    def test_polytreeOneIntersection(self, name, numLevels, branchingFactor, parentsAndChildren):

        parentTrees = [balancedTree(numLevels, branchingFactor) for _ in range(parentsAndChildren)]
        childTrees = [balancedTree(numLevels, branchingFactor) for _ in range(parentsAndChildren)]
        middleTree = multiplePaths(1, numLevels)

        union = nx.union(nx.DiGraph(), middleTree)
        allTrees = parentTrees + childTrees
        for i in range(len(allTrees)):
            union = nx.union(union, allTrees[i], rename=('', str(i+1)))

        middleTreeId = 0
        for i in range(parentsAndChildren):
            union.add_edge(f"{middleTreeId}", f"{i+1}0")
            union.add_edge(f"{len(allTrees)-i}0", f"{middleTreeId}")

        allTopos = list(nx.all_topological_sorts(union))
        self.assertEqual(tp.allPolyTopoSorts(union), len(allTopos))
        self.assertEqual(tpBasic.allPolyTopoSorts(union), len(allTopos))

    @parameterized.expand([ 
        (f"Level{level}_Branching{branching}", level, branching)
        for level in range(1, 3) for branching in range(1, 4)
    ])
    def test_polyTreeTwoIntersections(self, name, numLevels, branchingFactor):
        unevenParents = 3

        union = nx.DiGraph()
        for i in range(unevenParents):
            tree = balancedTree(numLevels, branchingFactor)
            union = nx.union(tree, union, rename=(f'{i}-', ''))

        middleTreeId = int(unevenParents/2)
        for i in range(1,middleTreeId+1):
            union.add_edge(f'{middleTreeId-i}-1', f'{middleTreeId}-1')
            union.add_edge(f'{middleTreeId+i}-1', f'{middleTreeId}-2')

        self.assertToposorts(union)

    def assertToposorts(self, union):
        allTopos = list(nx.all_topological_sorts(union))
        self.assertEqual(tp.allPolyTopoSorts(union), len(allTopos), 'The number of topological sorts is different than the expected')

    def test_polyTreeIntersectionLeafsToRoot(self):
        numLevels = 2
        branchingFactor = 3

        leftTree = balancedTree(numLevels, branchingFactor)
        rigthTree = balancedTree(numLevels, branchingFactor)
        middleTree = balancedTree(numLevels, branchingFactor)

        union = nx.union(leftTree, rigthTree, rename=('1', '2'))
        union = nx.union(union, middleTree, rename=('', '3'))

        #Connect the trees to the middle one
        union.add_edge('12', '30')
        union.add_edge('21', '30')

        self.assertToposorts(union)

    def test_polyTreeIntersectionNodesToNodes(self):
        tree = multiplePaths(1, 4)

        union = nx.union(tree, tree, rename=('1-', '2-'))
        union = nx.union(union, tree, rename=('', '3-'))

        #Connect the paths to the middle node
        union.add_edge('1-2', '3-2')
        union.add_edge('2-2', '3-2')

        self.assertToposorts(union)

    @parameterized.expand([ 
        (f"Level{level}_Branching{branching}", level, branching)
        for level in range(1, 3) for branching in range(1, 5)
    ])
    def test_polyForest(self, name, numLevels, branchingFactor):

        tree = balancedTree(numLevels, branchingFactor)
        union = nx.union(tree, tree, rename=('', 'copy'))

        self.assertToposorts(union)

    # Horrible test, but just to check some random cases
    def testRandomPolyForests(self):
        for nodes in range(1,15):
            for degree in range(1, 12):
                polyforest = createRandomPolyforest(num_nodes=nodes, max_out_degree=degree)
                if tp.allPolyTopoSorts(polyforest) < 60000: # This is to avoid taking too much time
                    self.assertToposorts(polyforest)
                

import random

def createRandomPolyforest(num_nodes, max_out_degree=3):

    # Step 1: Generate a random digraph
    polyforest = nx.DiGraph()
    polyforest.add_nodes_from(range(num_nodes))
    undirectedPolyForest = polyforest.to_undirected()
    
    # Add random edges with a constraint on out-degree
    for node in range(num_nodes):
        num_edges = random.randint(1, max_out_degree)

        possible_targets = list(set(range(num_nodes)) - {node})
        targets = random.sample(possible_targets, min(num_edges, len(possible_targets)))
        for target in targets:
            if not polyforest.has_edge(target, node) and nx.has_path(undirectedPolyForest,target, node):
                polyforest.add_edge(node, target)
                undirectedPolyForest.add_edge(node, target)

    assert isPolyforest(polyforest)
    return polyforest    


import itertools

# This can help with the debugging if some test fails
def classifyNodesByOrderAndIndexes(parentNodes, union):
    allTopos = list(nx.all_topological_sorts(union))

    permutations = list(itertools.permutations(parentNodes))
    toposPermutations = {}

    for topo in allTopos:
        for perm in permutations:
            indexes = [topo.index(node) for node in perm]
            if  indexes == sorted(indexes):
                key = f'Permutation: {perm}'
                toposPermutations[key] = toposPermutations.get(key, 0) + 1

    return toposPermutations, allTopos

if __name__ == '__main__':
    unittest.main()
