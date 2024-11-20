
from asvFormula.digraph import  isLeaf, hasMultipleParents, nx, isRoot
import math
from utils import sizeAndNumberOfTopoSortsTree, multinomial_coefficient
from toposPositions import positionsInToposorts
from typing import Any
from asvFormula.classesSizes.recursiveFormula import getPossibleCombinations
from itertools import product, permutations
from typing import NamedTuple

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

def rootInfo(node, polyTree : nx.DiGraph):
    if isRoot(node, polyTree):
        return node
    parent = next(polyTree.predecessors(node))
    return polyTree.nodes[parent]['root']

def addRootInfo(node, polyTree : nx.DiGraph):

    polyTree.nodes[node]['root'] = rootInfo(node, polyTree)
    for child in polyTree.successors(node):
        addRootInfo(child, polyTree)



"""
Another possible approach
orders = [[rootInfo(parent, polyTree), parent] for parent in parents]
possibleOrders = interleave_lists(orders)
for order in possibleOrders:
    order.append(disconnectedNode) #The disconnected node is always the last one
totalOrders = sum([allPosibleOrders(order, polyTree, trees) for order in possibleOrders])


def interleave_lists(lists):
    if all(not lst for lst in lists):
        return [[]]
    
    result = []
    # Iterate over each list
    for i, lst in enumerate(lists):
        if lst:
            # Take the first element from the current list
            first_elem = lst[0]
            # Remaining elements in the current list
            rest_list = lst[1:]
            # Remaining lists
            rest_lists = lists[:i] + [rest_list] + lists[i+1:]
            # Recursively compute interleavings of the remaining lists
            for suffix in interleave_lists(rest_lists):
                result.append([first_elem] + suffix)
    return result

#The I should do a similar implementation to leftPossibleOrders

"""

class NodeInfo(NamedTuple):
    position: int
    node : Any
    treeSize : int

    def __str__(self) -> str:
        return str(self.node)
    
    def __repr__(self) -> str:
        return str(self.node)

def allToposOfOrder(order : list[NodeInfo], nodesSizesAndTopos : dict[Any, tuple[int, int]], polyTree : nx.DiGraph) -> int:
    
    return math.prod([nodesSizesAndTopos[nodeInfo.node][1] for nodeInfo in order])

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
def allPossibleOrders(order : list[NodeInfo], nodesBefore : list[int] , nodesAfter : list[int], nodeIndex : int) -> int:


    if nodeIndex == len(order): #You have no more nodes to place
        return multinomial_coefficient(nodesAfter)

    mustUse = nodesBefore[nodeIndex] #We need to use all of the nodes before the actual node

    usableNodes = nodesBefore[nodeIndex+1:] + nodesAfter[:nodeIndex]
    canUse = sum(usableNodes)

    totalOrders = 0
    for positionsToFill in range(0, canUse + 1):
        for comb in getPossibleCombinations(usableNodes, positionsToFill):
            removeUsedElements(comb, nodesBefore, nodesAfter, nodeIndex)
            totalOrders +=  allPossibleOrders(order, nodesBefore, nodesAfter, nodeIndex + 1) * multinomial_coefficient(comb + [mustUse])
            addUsedElements(comb, nodesBefore, nodesAfter, nodeIndex)

    return totalOrders


#Returns the results of merging the trees with the disconnected nodes and their parents
def mergeConnectedTrees(trees : dict[Any, tuple[int, int]], disconnectedNode : Any, parents : list[Any], polyTree : nx.DiGraph) -> tuple[int, int]:
 
    nodesSizesAndTopos = {parent : trees[rootInfo(parent, polyTree)] for parent in parents + [disconnectedNode]}
    nodesPositions = {parent : positionsInToposorts(parent, polyTree) for parent in parents + [disconnectedNode]}
    
    parentsPositions = [[NodeInfo(position, parent, nodesSizesAndTopos[parent][0]) for position in nodesPositions[parent].keys()] for parent in parents]
    disconnectedPositions = [NodeInfo(position, disconnectedNode, nodesSizesAndTopos[disconnectedNode][0]) for position in nodesPositions[disconnectedNode].keys()]

    allOrders = [list(perm) for combination in product(*parentsPositions) for perm in permutations(combination)]
    allOrders = [order + [disconnectedPosition] for order in allOrders for disconnectedPosition in disconnectedPositions]

    totalTopos = 0

    for order in allOrders:
        nodesBefore = [nodeInfo.position for nodeInfo in order]
        nodesAfter =  [nodeInfo.treeSize - 1 - nodeInfo.position for nodeInfo in order]
        totalTopos += allPossibleOrders(order, nodesBefore, nodesAfter, 0) * allToposOfOrder(order, nodesSizesAndTopos, polyTree)
    
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
        addRootInfo(root, copyPolyTree)

    # Merge the results of the trees with intersections and calculate the toposorts and sizes for the new trees.
    for disconnectedNode, parents in disconnectedNodesAndParents.items():
        mergedResult = mergeConnectedTrees(rootsSizesAndTopos, disconnectedNode, parents, copyPolyTree)
        rootsSizesAndTopos[disconnectedNode] = mergedResult
        for parent in parents:
            del rootsSizesAndTopos[rootInfo(parent, copyPolyTree)]

    # Merge all of the trees and calculate the final toposort
    treesSizes, treesTopos = zip(*[tree for tree in rootsSizesAndTopos.values()])
    topos = multinomial_coefficient(list(treesSizes)) * math.prod(list(treesTopos))
    
    return topos