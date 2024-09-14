import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
from typing import Any, Dict, List

class NodeState(Enum):
    ANCESTOR = 1
    DESCENDANT = 2
    UNRELATED = 3
    FEATURE = 4


#Classify the nodes into the category descendants, ancestors and unrelated. 

def classifyNodes(dag: nx.DiGraph, x_i : Any, nodes_classification : Dict[Any, NodeState]):
    ancestors = nx.ancestors(dag, x_i)
    descendants = nx.descendants(dag, x_i)
    unrelated_roots = []
    for node in dag.nodes():
        if node in ancestors:
            nodes_classification[node] = NodeState.ANCESTOR
        elif node in descendants:
            nodes_classification[node] = NodeState.DESCENDANT
        elif node == x_i:
            nodes_classification[node] = NodeState.FEATURE
        else:
            nodes_classification[node] = NodeState.UNRELATED
            parents =  list(dag.predecessors(node))
            parentIsAncestor = (parents[0] in ancestors) if len(parents) != 0 else False
            if isRoot(node, dag) or parentIsAncestor:
                unrelated_roots.append(node)


    return unrelated_roots

#They need to be ordered in the same way that the nodes are ordered in the graph, because in the possibleLeftOrders we are putting them assuming they have the same order as in the graph

def orderedNodes(dag : nx.DiGraph, nodesToOrder : List[Any]) -> List[Any]:
    topo_sorted_nodes = list(nx.topological_sort(dag))
    ordered_ancestors = [n for n in topo_sorted_nodes if n in nodesToOrder]
    
    return ordered_ancestors


#Digraph manipulation functions

def isLeaf(node, dag : nx.DiGraph):
    return dag.out_degree(node) == 0

def isRoot(node, dag : nx.DiGraph):
    return dag.in_degree(node) == 0

def drawGraph(dag : nx.DiGraph):
    pos = nx.spring_layout(dag)
    nx.draw(dag, pos, with_labels=True)
    plt.show()

#Digraph generation functions

def emptyGraph(numNodes):
    emptyGraph = nx.DiGraph()
    nodes = [i for i in range(numNodes)]
    emptyGraph.add_nodes_from(nodes)
    return emptyGraph
    
def naiveBayes(numNodes : int):
    naive_bayes = nx.DiGraph()
    nodes = [i for i in range(numNodes)]
    naive_bayes.add_nodes_from(nodes)
    root = list(naive_bayes.nodes)[0]
    for node in nodes:
        if node != root:
            naive_bayes.add_edge(root, node)
    return naive_bayes

def naiveBayesWithPath(numNodes : int, lengthOfPath : int):
    naive_bayes = naiveBayes(numNodes)
    for path_node in range(numNodes,numNodes+lengthOfPath):
        naive_bayes.add_node(path_node)
        naive_bayes.add_edge(path_node-1, path_node)
    return naive_bayes

def multiplePaths(numPaths, nodesPerPath) -> nx.DiGraph:
    multiplePaths = nx.DiGraph()
    multiplePaths.add_node(0)

    nodeCounter = 0
    for path in range(1,numPaths+1):
        for node in range(1,nodesPerPath+1):
            if node != 1:
                multiplePaths.add_edge(nodeCounter-1, nodeCounter)
            nodeCounter += 1
    
    return multiplePaths


def balancedTree(height: int, branchingFactor: int = 2) -> nx.DiGraph:

    balanced_tree = nx.DiGraph()
    current_level_nodes = [0]
    balanced_tree.add_node(0)
    node_counter = 1
    
    for _ in range(1, height):
        next_level_nodes = []
        for parent in current_level_nodes:
            for _ in range(branchingFactor):
                balanced_tree.add_node(node_counter)
                balanced_tree.add_edge(parent, node_counter)
                next_level_nodes.append(node_counter)
                node_counter += 1
        current_level_nodes = next_level_nodes
    
    return balanced_tree