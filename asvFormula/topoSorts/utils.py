from typing import List, Dict, Any
from asvFormula.digraph import NodeState, isRoot, isLeaf, nx
from scipy.special import comb
import math
from classesSizes.equivalenceClass import EquivalenceClass

#Returns a hash that is the binary number which has 0 or 1 in the i-th position if the i-th unrelated node is before or after x_i

class TopoSortHasher:
    def __init__(self, nodes_classification: Dict[Any, NodeState]):
        self._unrelated_nodes_ids = self._get_unrelated_nodes(nodes_classification)

    def _get_unrelated_nodes(self, nodes_classification: Dict[Any, NodeState]):
        unrelated_nodes = list(filter(lambda node: nodes_classification[node] == NodeState.UNRELATED, nodes_classification.keys()))
        self._unrelated_nodes_ids = {node: i for i, node in enumerate(unrelated_nodes)}
        return self._unrelated_nodes_ids

    def hashTopoSort(self, topoSort: List[Any], x_i: Any) -> int:
        unrelated_nodes = self._unrelated_nodes_ids
        hash_val = 0
        for node in topoSort:
            if node == x_i:
                break
            if node in unrelated_nodes:
                hash_val += 2 ** unrelated_nodes[node]
        return hash_val

#Returns the size of the tree and the number of topological sorts

def sizeAndNumberOfTopoSortsTree(node, tree : nx.DiGraph) -> tuple[int, int]:
    if isLeaf(node, tree):
        return 1,1
    
    childrenSubtreeSizes = []
    children_topoSorts = []

    
    for child in tree.successors(node):
        recursionNode = child
            
        child_size, child_topos =  sizeAndNumberOfTopoSortsTree(recursionNode,tree)
        children_topoSorts.append(child_topos)
        childrenSubtreeSizes.append(child_size)
        

    topos = multinomial_coefficient(childrenSubtreeSizes) * math.prod(children_topoSorts)
    return sum(childrenSubtreeSizes)+1, topos
    
def topoSortsFrom(node, dag : nx.DiGraph):
   _, topos = sizeAndNumberOfTopoSortsTree(node, dag)
   return topos

def allTopoSorts(tree : nx.DiGraph):
    #Add a root node to the graph that is connected to all the roots of the graph.
    roots = [node for node in tree.nodes() if isRoot(node, tree)]
    

    if len(roots) == 1:
        root = roots[0]
        res = topoSortsFrom(root, tree)
    else:
        newRoot = 'Root'
        tree.add_node(newRoot)

        for root in roots:
            tree.add_edge(newRoot, root)
        res = topoSortsFrom(newRoot, tree)
        tree.remove_node(newRoot)    
    
    return res

def isTopologicalSort(G, ordering):
    position = {node: i for i, node in enumerate(ordering)}
    
    # Check that for each edge (u, v) in the graph, u appears before v in the ordering
    for u, v in G.edges():
        if position[u] >= position[v]:
            return False
    return True

def hashEquivClasses(equivClasses : List[EquivalenceClass], hasher : TopoSortHasher , feature_node, dag : nx.DiGraph):
    hashedClasses = {}
    for eqClass in equivClasses:
        topoSortForClass = eqClass.topologicalSort(feature_node)
        if not isTopologicalSort(dag, topoSortForClass):
            raise AssertionError(f"The topological sort {topoSortForClass} is not a valid topological sort for the graph, for the feature node {feature_node}")
        hash = hasher.hashTopoSort(topoSortForClass, feature_node)
        hashedClasses[hash] = [topoSortForClass,  eqClass.classSize()]
    
    return hashedClasses

def multinomial_coefficient(args) -> int:
    n = sum(args)
    coeff = 1
    for k in args:
        coeff *= comb(n, k, exact=True)
        n -= k
    return int(coeff)