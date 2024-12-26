from asvFormula.digraph import nx, isRoot
from asvFormula import Node
from typing import List
from topoSortsCalc import allPolyTopoSorts
import random

# We generate a random topological sort by selecting a random node from the source nodes. Defining the probability of picking it as the number of toposorts that would start with that node
def randomTopoSort(polyTree : nx.DiGraph) -> List[Node]:

    if len(polyTree.nodes) == 0:
        return []

    sourceNodes = [node for node in polyTree.nodes() if isRoot(node, polyTree)]

    topoSortsForStartNodes = []

    for startNode in sourceNodes:
        edges = list(polyTree.edges(startNode))
        polyTree.remove_node(startNode)
        topoSortsForStartNodes.append(allPolyTopoSorts(polyTree)) # All of the topological sorts we would have if we picked this node as the first one
        polyTree.add_node(startNode)
        polyTree.add_edges_from(edges)

    firstNode = random.sample(sourceNodes,1, counts=topoSortsForStartNodes)[0]
    #TODO: See if it's faster using choices or sample here. ¿https://stackoverflow.com/questions/40914862/why-is-random-sample-faster-than-numpys-random-choice#:~:text=sample%20is%20~15%25%20faster%20than,choice%20.?
    copiedTree = polyTree.copy()
    copiedTree.remove_node(firstNode)

    return [firstNode] + randomTopoSort(copiedTree)

def randomTopoSorts(polyTree : nx.DiGraph, numTopoSorts : int) -> List[List[Node]]:

    print(f'It will take ~{numTopoSorts*0.004} seconds to calculate the {numTopoSorts} topological sorts')
    return [randomTopoSort(polyTree) for _ in range(numTopoSorts)]