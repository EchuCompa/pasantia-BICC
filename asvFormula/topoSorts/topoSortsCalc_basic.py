#This is the previous implementation of the toposort calculation. It does not work if there are multiple intersections between trees. 
#But if there aren't more than two trees connected at the same time it works faster. 

from asvFormula.digraph import  hasMultipleParents, nx, isRoot
from utils import sizeAndNumberOfTopoSortsTree
from toposPositions import positionsInToposorts
from topoSortsCalc import *

#Returns a dict with the nodes with multiple parents and their parents
def removeMultipleParents(polyTree : nx.DiGraph) -> dict[Node, list[Node]]:
    disconnectedNodesAndParents = {}
    for node in polyTree.nodes():
        if hasMultipleParents(node, polyTree):
            parents = list(polyTree.predecessors(node))
            disconnectedNodesAndParents[node] = parents
            for parent in parents:
                polyTree.remove_edge(parent, node)
    return disconnectedNodesAndParents

def rootInfo(node, polyTree : nx.DiGraph):
    if isRoot(node, polyTree):
        return node
    parent = next(polyTree.predecessors(node))
    return polyTree.nodes[parent]['root']

def addRootInfo(node, polyTree : nx.DiGraph):

    polyTree.nodes[node]['root'] = rootInfo(node, polyTree)
    for child in polyTree.successors(node):
        addRootInfo(child, polyTree)

#Very similar to the possibleLeftOrders function in recursiveFormula.py
@lru_cache(maxsize=None)
def allPossibleOrdersOld(nodeIndex : int, nodesBefore : list[int] , nodesAfter : list[int], lastNode : int) -> int:
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
            totalOrders +=  allPossibleOrdersOld(nodeIndex + 1 , tuple(nodesBefore), tuple(nodesAfter), lastNode) * multinomial_coefficient(comb + [mustUse])
            addUsedElements(comb, nodesBefore, nodesAfter, nodeIndex)

    return totalOrders


#Returns the results of merging the trees with the disconnected nodes and their parents
def ordersFromIntersection(trees : dict[Node, tuple[int, int]], intersectionNode : Node, parents : list[Node], polyTree : nx.DiGraph) -> tuple[int, int]:

    nodesSizesAndTopos = {parent : trees[rootInfo(parent, polyTree)] for parent in parents + [intersectionNode]}
    nodesPositions = {parent : positionsInToposorts(parent, polyTree) for parent in parents + [intersectionNode]}

    
    parentsPositions = [[NodeInfo(position, parent, nodesSizesAndTopos[parent][0], toposPosition) for position, toposPosition in nodesPositions[parent].items()] for parent in parents]
    disconnectedPositions = [NodeInfo(position, intersectionNode, nodesSizesAndTopos[intersectionNode][0], toposPosition) for position, toposPosition in nodesPositions[intersectionNode].items()] 

    allOrders = [list(perm) for combination in product(*parentsPositions) for perm in permutations(combination)]
    allOrders = [order + [disconnectedPosition] for order in allOrders for disconnectedPosition in disconnectedPositions]

    totalTopos = 0

    for order in allOrders:
        nodesBefore = [nodeInfo.position for nodeInfo in order]
        nodesAfter =  [nodeInfo.treeSize - 1 - nodeInfo.position for nodeInfo in order]
        totalTopos += allPossibleOrdersOld(order, nodesBefore, nodesAfter, 0) * allToposOfOrder(order, nodesSizesAndTopos, polyTree)
        totalTopos += allPossibleOrdersOld(0, tuple(nodesBefore), tuple(nodesAfter), len(order)) * allToposOfOrder(order)

    totalSize = sum([size for size, _ in nodesSizesAndTopos.values()])
    return (totalSize, totalTopos)

def allPolyTopoSortsOld(polyTree : nx.DiGraph):

    #Identify the nodes with more than one parent and disconnect them from the graph. Saving the nodes and the parents. 
    copyPolyTree = polyTree.copy()
    intersectionNodesAndParents  = removeMultipleParents(copyPolyTree)


    #Calculate and save the toposorts and sizes for each tree
    roots = [node for node in copyPolyTree.nodes() if isRoot(node, copyPolyTree)]
    rootsSizesAndTopos = {root: sizeAndNumberOfTopoSortsTree(root, copyPolyTree) for root in roots}

    for root in roots: #This is to identify each node with it's tree
        addRootInfo(root, copyPolyTree)

    # Merge the results of the trees with intersections and calculate the toposorts and sizes for the new trees.
    for intersectionNode, parents in intersectionNodesAndParents.items():
        mergedResult = ordersFromIntersection(rootsSizesAndTopos, intersectionNode, parents, copyPolyTree)
        rootsSizesAndTopos[intersectionNode] = mergedResult
        for parent in parents:
            del rootsSizesAndTopos[rootInfo(parent, copyPolyTree)]

    # Merge all of the trees and calculate the final toposort
    treesSizes, treesTopos = zip(*[tree for tree in rootsSizesAndTopos.values()])
    topos = multinomial_coefficient(list(treesSizes)) * math.prod(list(treesTopos))
    
    return topos