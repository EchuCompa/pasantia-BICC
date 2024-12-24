
from asvFormula.digraph import  nx
import math
from utils import multinomial_coefficient
from typing import Any
from asvFormula.classesSizes.recursiveFormula import getPossibleCombinations
from itertools import product, permutations
from typing import NamedTuple
from functools import lru_cache

def allPolyTopoSorts(polyTree : nx.DiGraph, firstNode = None) -> int:
    
    visited = {node : False for node in polyTree.nodes()}
    
    subTreeResults = {}
    for node in polyTree.nodes():
        if not visited[node]:
            subTreeResults[node] = allPolyTopoSortsAndSizeFromNode(node, polyTree, visited)

    subtreeSizes = [result[1] for result in subTreeResults.values()]
    subtreesTopoSorts = [result[0] for result in subTreeResults.values()]

    topos = subtreesTopoSorts[0] if len(subtreesTopoSorts) == 1 else multinomial_coefficient(subtreeSizes) * math.prod(subtreesTopoSorts)
    print(f'This would take {topos/60000} seconds in the naive way') 
    return topos

def allPolyTopoSortsAndSizeFromNode(node, polyTree : nx.DiGraph, visited : dict[Any, bool]) -> int:
    
    positionsAndTopos = allPolyTopoSortsAndPositions(node, polyTree, visited)
    topos = sum([nodeInfo.positionTopos for nodeInfo in positionsAndTopos])
    size = positionsAndTopos[0].treeSize

    return topos,size

class NodeInfo(NamedTuple):
    position: int
    node : Any
    treeSize : int
    positionTopos : int

    def __str__(self) -> str:
        return str(self.node)
    
    def __repr__(self) -> str:
        return f'Node: {str(self.node)} Position: {self.position}, PositionTopos: {self.positionTopos}'

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
def allPossibleOrdersOld(nodeIndex : int, nodesBefore : list[int] , nodesAfter : list[int], lastNode : int, nodesToPutBefore : int, placedNodeIndex: int) -> int:
    
    if nodesToPutBefore < 0 and nodeIndex <= placedNodeIndex: #I still haven't placed the node and I have already passed that position
        return 0
    
    nodesBefore = list(nodesBefore)
    nodesAfter = list(nodesAfter)

    if nodeIndex == lastNode: #You have no more nodes to place 
        totalOrders = 0
        for comb in getPossibleCombinations(nodesAfter, sum(nodesAfter)): #You must put all of the remaining nodes
            totalOrders +=  multinomial_coefficient(comb) #There are no more nodes to place
        return totalOrders

    mustUse = nodesBefore[nodeIndex] #We need to use all of the nodes before the actual node

    usableNodes = nodesBefore[nodeIndex+1:] + nodesAfter[:nodeIndex]
    canUse = sum(usableNodes)

    totalOrders = 0
    possibleNodesToUse = range(0, canUse + 1) if nodeIndex != placedNodeIndex else [nodesToPutBefore] #If the actual node is the one that we need to place, we must use exactly nodesToPutBefore nodes
    for positionsToFill in possibleNodesToUse:
        for comb in getPossibleCombinations(usableNodes, positionsToFill):
            placedNodes = positionsToFill + mustUse + 1 #The actual node
            removeUsedElements(comb, nodesBefore, nodesAfter, nodeIndex)
            newNodesToPut = 0 if nodeIndex > placedNodeIndex else nodesToPutBefore - placedNodes
            totalOrders +=  allPossibleOrdersOld(nodeIndex + 1 , tuple(nodesBefore), tuple(nodesAfter), lastNode, newNodesToPut, placedNodeIndex) * multinomial_coefficient(comb + [mustUse])
            addUsedElements(comb, nodesBefore, nodesAfter, nodeIndex)

    return totalOrders

def allPolyTopoSortsAndPositions(node, polyTree : nx.DiGraph, visited : dict[Any, bool]) -> list[NodeInfo]:
    
    visited[node] = True

    notVisitedChildren = [child for child in polyTree.successors(node) if not visited[child]]
    notVisitedParents = [parent for parent in polyTree.predecessors(node) if not visited[parent]]

    for neighbour in notVisitedChildren + notVisitedParents:
        visited[neighbour] = True

    if len(notVisitedChildren + notVisitedParents) == 0: #Leaf node
        return [NodeInfo(0, node, 1, 1)]
    
    notVisitedChildrenPositions = [allPolyTopoSortsAndPositions(child, polyTree, visited) for child in notVisitedChildren]
    notVisitedParentsPositions = [allPolyTopoSortsAndPositions(parent, polyTree, visited) for parent in notVisitedParents]
    
    actualNodeInfo = NodeInfo(0, node, 1, 1) 
    allChildOrders = [list(perm) for combination in product(*notVisitedChildrenPositions) for perm in permutations(combination)]
    
    allParentOrders = [list(perm) for combination in product(*notVisitedParentsPositions) for perm in permutations(combination)]

    # The actual node will be after it's children and before it's parents
    allOrders = [parentOrder + [actualNodeInfo] + childOrder  for childOrder in allChildOrders for parentOrder in allParentOrders]
    #TODO: Maybe do this with itertools if it's faster

    toposPerPosition = {}
    totalSize = sum([nodeInfo.treeSize for nodeInfo in allOrders[0]])
    numParents = len(notVisitedParents)
    numChildren = len(notVisitedChildren)

    for order in allOrders:
        addToposortsOfOrder(order, toposPerPosition, totalSize, numParents, numChildren)
            
    results  = [NodeInfo(position, node, totalSize, positionTopos) for position, positionTopos in toposPerPosition.items()]

    return results

def addToposortsOfOrder(order, toposPerPosition : dict[Any, ], totalSize, numParents, numChildren):
    nodesBefore = [nodeInfo.position for nodeInfo in order]
    nodesAfter =  [nodeInfo.treeSize - 1 - nodeInfo.position for nodeInfo in order]
    nodesMustBeBefore = sum(nodesBefore[:numParents]) + numParents #These nodes must be put before the actual node
    nodesMustBeAfter = sum(nodesAfter[numParents:]) + numChildren #These nodes must be put after the actual node
    for actualNodePosition in range(nodesMustBeBefore , totalSize - nodesMustBeAfter  + 1):
        toposWithPosition = allPossibleOrdersOld(0, tuple(nodesBefore), tuple(nodesAfter), len(order), actualNodePosition, numParents) * allToposOfOrder(order)
        if toposWithPosition != 0:
            toposPerPosition[actualNodePosition] = toposPerPosition.get(actualNodePosition, 0) + toposWithPosition