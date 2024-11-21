
from asvFormula.digraph import  isLeaf, hasMultipleParents, nx, isRoot
import math
from utils import sizeAndNumberOfTopoSortsTree, multinomial_coefficient
from toposPositions import positionsInToposorts
from typing import Any
from asvFormula.classesSizes.recursiveFormula import getPossibleCombinations
from itertools import product, permutations
from typing import NamedTuple
from functools import lru_cache

#Returns a dict with the nodes with multiple parents and their parents
def removeMultipleParents(polyTree : nx.DiGraph) -> dict[Any, list[Any]]:
    disconnectedNodesAndParents = {}
    for node in polyTree.nodes():
        if hasMultipleParents(node, polyTree):
            parents = list(polyTree.predecessors(node))
            disconnectedNodesAndParents[node] = parents
            for parent in parents:
                polyTree.remove_edge(parent, node)
    return disconnectedNodesAndParents

def rootTreeID(node, polyTree : nx.DiGraph):

    return polyTree.nodes[node]['treeID']

def addTreeID(node, polyTree : nx.DiGraph, treeId : Any):

    polyTree.nodes[node]['treeID'] = treeId
    for child in polyTree.successors(node):
        addTreeID(child, polyTree, treeId)

class NodeInfo(NamedTuple):
    position: int
    node : Any
    treeSize : int
    positionTopos : int

    def __str__(self) -> str:
        return str(self.node)
    
    def __repr__(self) -> str:
        return f'Node: {str(self.node)} Position: {self.position}'

def allToposOfOrder(order : list[NodeInfo]) -> int:
    
    return math.prod([nodeInfo.positionTopos for nodeInfo in order])

def removeUsedElements(usedElements : list[int], nodesBefore : list[int], nodesAfter : list[int], nodeIndex : int):
    for i, used in enumerate(usedElements):
        actualIndex = nodeIndex + 1 + i
        if actualIndex < len(nodesBefore):
            nodesBefore[actualIndex] -= used
        else:
            nodesAfter[actualIndex - len(nodesBefore)] -= used

def addUsedElements(usedElements : list[int], nodesBefore : list[int], nodesAfter : list[int], nodeIndex : int):
    for i, used in enumerate(usedElements):
        actualIndex = nodeIndex + 1 + i
        if  actualIndex < len(nodesBefore):
            nodesBefore[actualIndex] += used
        else:
            nodesAfter[actualIndex - len(nodesBefore)] += used

#Very similar to the leftPossibleOrders function in recursiveFormula.py
@lru_cache(maxsize=None)
def allPossibleOrders(nodeIndex : int, nodesBefore : list[int] , nodesAfter : list[int], lastNode : int) -> int:
    nodesBefore = list(nodesBefore)
    nodesAfter = list(nodesAfter)

    if nodeIndex == lastNode: #You have no more nodes to place
        return multinomial_coefficient(nodesAfter)

    mustUse = nodesBefore[nodeIndex] #We need to use all of the nodes before the actual node

    usableNodes = nodesBefore[nodeIndex+1:] + nodesAfter[:nodeIndex]
    canUse = sum(usableNodes)

    totalOrders = 0
    for positionsToFill in range(0, canUse + 1):
        for comb in getPossibleCombinations(usableNodes, positionsToFill):
            removeUsedElements(comb, nodesBefore, nodesAfter, nodeIndex)
            totalOrders +=  allPossibleOrders(nodeIndex + 1 , tuple(nodesBefore), tuple(nodesAfter), lastNode) * multinomial_coefficient(comb + [mustUse])
            addUsedElements(comb, nodesBefore, nodesAfter, nodeIndex)

    return totalOrders

#Returns the results of merging the trees with the disconnected nodes and their parents
def mergeConnectedTrees(trees : dict[Any, tuple[int, int]], disconnectedNode : Any, parents : list[Any], polyTree : nx.DiGraph) -> tuple[int, int]:
 
    nodesSizesAndTopos = {parent : trees[rootTreeID(parent, polyTree)] for parent in parents + [disconnectedNode]}
    nodesPositions = {parent : positionsInToposorts(parent, polyTree) for parent in parents + [disconnectedNode]}
    
    parentsPositions = [[NodeInfo(position, parent, nodesSizesAndTopos[parent][0], toposPosition) for position, toposPosition in nodesPositions[parent].items()] for parent in parents]
    disconnectedPositions = [NodeInfo(position, disconnectedNode, nodesSizesAndTopos[disconnectedNode][0], toposPosition) for position, toposPosition in nodesPositions[disconnectedNode].items()] 

    allOrders = [list(perm) for combination in product(*parentsPositions) for perm in permutations(combination)]
    allOrders = [order + [disconnectedPosition] for order in allOrders for disconnectedPosition in disconnectedPositions]

    totalTopos = 0

    for order in allOrders:
        nodesBefore = [nodeInfo.position for nodeInfo in order]
        nodesAfter =  [nodeInfo.treeSize - 1 - nodeInfo.position for nodeInfo in order]
        totalTopos += allPossibleOrders(0, tuple(nodesBefore), tuple(nodesAfter), len(order)) * allToposOfOrder(order)
    
    totalSize = sum([size for size, _ in nodesSizesAndTopos.values()])
    return (totalSize, totalTopos)

def allPolyTopoSorts(polyTree : nx.DiGraph):
    
    #Identify the nodes with more than one parent and disconnect them from the graph. Saving the nodes and the parents. 
    copyPolyTree = polyTree.copy()
    disconnectedNodesAndParents  = removeMultipleParents(copyPolyTree)

    
    #Calculate and save the toposorts and sizes for each tree
    roots = [node for node in copyPolyTree.nodes() if isRoot(node, copyPolyTree)]
    rootsSizesAndTopos = {root: sizeAndNumberOfTopoSortsTree(root, copyPolyTree) for root in roots}

    for root in roots: #This is to identify each node with it's tree
        addTreeID(root, copyPolyTree, root)


    # Merge the results of the trees with intersections and calculate the toposorts and sizes for the new trees.
    for disconnectedNode, parents in disconnectedNodesAndParents.items():
        mergedResult = mergeConnectedTrees(rootsSizesAndTopos, disconnectedNode, parents, copyPolyTree)
        rootsSizesAndTopos[disconnectedNode] = mergedResult
        for parent in parents:
            addTreeID(parent, copyPolyTree, disconnectedNode)
            del rootsSizesAndTopos[rootTreeID(parent, copyPolyTree)]

    # Merge all of the trees and calculate the final toposort
    treesSizes, treesTopos = zip(*[tree for tree in rootsSizesAndTopos.values()])
    topos = multinomial_coefficient(list(treesSizes)) * math.prod(list(treesTopos))
    
    return topos