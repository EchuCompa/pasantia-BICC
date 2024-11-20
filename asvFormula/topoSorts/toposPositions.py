from typing import Dict, Any
from asvFormula.digraph import isRoot, nx
from typing import NamedTuple
from asvFormula.topoSorts.utils import sizeAndNumberOfTopoSortsTree, multinomial_coefficient
from functools import lru_cache

#TODO: Maybe adding a cache to sizeAndNumberOfTopoSortsTree could help with the performance of this function. We need to take into account
# that the tree is constantly changing, so for the same node we might have different results in different executions. 

class ToposortPosition:

    def __init__(self, tree : nx.DiGraph = None):
        self.tree = tree

    def setTree(self, tree : nx.DiGraph):
        self.tree = tree

    # It is not safe to call it with the same node in different trees.
    #TODO: Experiment with the cache decorator to see if it improves the performance of this function.
    #@lru_cache 
    def positionsInToposortsAndNodesBelow(self, node) -> tuple[ Dict[Any, int], int]:

        if isRoot(node, self.tree):
            treeSize, topoSorts =  sizeAndNumberOfTopoSortsTree(node, self.tree)
            nodesAfter = treeSize - 1
            return {0: PositionInfo(topoSorts, nodesAfter) }

        parent = next(self.tree.predecessors(node))
        self.tree.remove_edge(parent, node)

        parentPositions = self.positionsInToposortsAndNodesBelow(parent)
        nodeSize, nodeTopos = sizeAndNumberOfTopoSortsTree(node, self.tree)
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

#Returns a dict with the possible positions of a node in all the toposorts and who many exists. This works for trees.
# {pos_x : number of toposorts with x in position pos_x}

def positionsInToposorts(node, tree : nx.DiGraph, topoPosi : ToposortPosition = ToposortPosition() ) -> Dict[Any, int]:
    copiedTree = tree.copy()
    topoPosi.setTree(copiedTree)
    positions = topoPosi.positionsInToposortsAndNodesBelow(node)
    positions = {pos: posInfo.topoSorts for pos, posInfo in positions.items()}
    return positions

def naivePositionsInToposorts(node, dag : nx.DiGraph, allTopos : list[list[Any]] = None) -> Dict[Any, int]: 
    all_topo_sorts = allTopos if allTopos is None else list(nx.all_topological_sorts(dag))
    positions = {}
    for topoSort in all_topo_sorts:
        pos = topoSort.index(node)
        positions[pos] = positions.get(pos, 0) + 1

    return positions