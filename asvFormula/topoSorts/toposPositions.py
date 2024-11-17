from typing import Dict, Any
from asvFormula.digraph import isRoot, nx
from typing import NamedTuple
from asvFormula.topoSorts.utils import sizeAndNumberOfTopoSortsTree, multinomial_coefficient

#Returns a dict with the possible positions of a node in all the toposorts and who many exists. This works for trees.
# {pos_x : number of toposorts with x in position pos_x}

def positionsInToposorts(node, tree : nx.DiGraph) -> Dict[Any, int]:
    copiedTree = tree.copy()
    positions = positionsInToposortsAndNodesBelow(node, copiedTree)
    positions = {pos: posInfo.topoSorts for pos, posInfo in positions.items()}
    return positions

def positionsInToposortsAndNodesBelow(node, tree : nx.DiGraph) -> tuple[ Dict[Any, int], int]:

    if isRoot(node, tree):
        treeSize, topoSorts =  sizeAndNumberOfTopoSortsTree(node, tree)
        nodesAfter = treeSize - 1
        return {0: PositionInfo(topoSorts, nodesAfter) }

    parent = next(tree.predecessors(node))
    tree.remove_edge(parent, node)

    parentPositions = positionsInToposortsAndNodesBelow(parent, tree)
    nodeSize, nodeTopos = sizeAndNumberOfTopoSortsTree(node, tree)
    nodesBelow = nodeSize - 1 
    
    nodePositions = {}

    for posParent, posiInfo in parentPositions.items():
        initialPosition = posParent+1
        nodesAfterParent = posiInfo.nodesAfter
        toposParent = posiInfo.topoSorts
        for nodePos in range(initialPosition, initialPosition+nodesAfterParent+1):
            topos = toposParent * nodeTopos
            parentNodesAvailable = nodesAfterParent - (nodePos-initialPosition)
            toposOrders = multinomial_coefficient([nodesBelow, parentNodesAvailable])

            positionTopos, _ = nodePositions.get(nodePos, PositionInfo(0,0))
            nodesAfter = nodesBelow + parentNodesAvailable
            nodePositions[nodePos] = PositionInfo(positionTopos + topos * toposOrders, nodesAfter) 
            

    return nodePositions

class PositionInfo(NamedTuple):
    topoSorts : int
    nodesAfter : int

def naivePositionsInToposorts(node, dag : nx.DiGraph, allTopos : list[list[Any]] = None) -> Dict[Any, int]: 
    all_topo_sorts = allTopos if allTopos is None else list(nx.all_topological_sorts(dag))
    positions = {}
    for topoSort in all_topo_sorts:
        pos = topoSort.index(node)
        positions[pos] = positions.get(pos, 0) + 1

    return positions