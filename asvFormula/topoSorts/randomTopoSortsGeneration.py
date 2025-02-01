from asvFormula.digraph import nx, isRootWithVisited
from asvFormula import Node
from typing import List
from topoSortsCalc import allPolyTopoSortsWithVisited
import random, time
from collections import Counter

#TODO : Make it faster. 
global totalTime
totalTime = {}

# We generate a random topological sort by selecting a random node from the source nodes. Defining the probability of picking it as the number of toposorts that would start with that node

def randomTopoSort(polyTree : nx.DiGraph) -> List[Node]:

    return randomTopoSorts(polyTree, 1)[0]

def randomTopoSorts(polyTree : nx.DiGraph, numTopoSorts : int) -> List[List[Node]]:
    #print(f'It will take {numTopoSorts/60000} seconds in the naive way')
    return randomTopoSortsAux(polyTree, numTopoSorts, {}, {node : False for node in polyTree.nodes()})

#Using a cache and visited nodes to avoid recalculating the same thing multiple times, it's cleaner removing the nodes from the polytree but it's slower.
def randomTopoSortsAux(polyTree : nx.DiGraph, numTopoSorts : int, topoSortsCache : dict[tuple[Node], int], visited : dict[Node, bool]) -> List[List[Node]]:

    noVisitedNodes = notVisitedNodes(visited)

    if len(noVisitedNodes) <= 1:
        return [noVisitedNodes]*numTopoSorts

    sourceNodes = [node for node in noVisitedNodes if isRootWithVisited(node, polyTree, visited)]

    topoSortsForStartNodes = [0]*len(sourceNodes)
    for i, startNode in enumerate(sourceNodes):
        topoSortsForStartNodes[i] = calculateTopoSortsForNode(startNode, polyTree, visited, topoSortsCache)

    firstNodes = obtainRandomStartNodes(numTopoSorts, topoSortsForStartNodes, sourceNodes)

    topoSorts = []
    for firstNode, count in Counter(firstNodes).items():
        visited[firstNode] = True
        results = randomTopoSortsAux(polyTree, count, topoSortsCache, visited)

        topoSorts += [[firstNode] + topoSort for topoSort in results]

        visited[firstNode] = False
    
    return topoSorts

def calculateTopoSortsForNode(startNode : Node, polyTree : nx.DiGraph, visited : dict[Node, bool], topoSortsCache : dict[Node, int]) -> int:
    visitedCopy = visited.copy()        
    visitedCopy[startNode] = True


    cacheKey = frozenset(notVisitedNodes(visitedCopy))
    if not cacheKey in topoSortsCache: #I do this and not get() cause get will always call allPolyTopoSortsWithVisited
        topoSortsCache[cacheKey] = allPolyTopoSortsWithVisited(polyTree, visitedCopy)
    global totalTime

    return topoSortsCache[cacheKey]

def obtainRandomStartNodes(numTopoSorts : int, topoSortsForStartNodes : List[int], sourceNodes : List[Node]) -> List[Node]:
    
    return random.choices(sourceNodes,weights=topoSortsForStartNodes, k=numTopoSorts)

def notVisitedNodes(visited : dict[Node, bool]) -> List[Node]:
    return [node for node, isVisited in visited.items() if not isVisited]


def randomTopoSortsTime(polyTree : nx.DiGraph, numTopoSorts : int) -> List[List[Node]]:
    #print(f'It will take {numTopoSorts/60000} seconds in the naive way')
    return randomTopoSortsAuxTime(polyTree, numTopoSorts, {}, {node : False for node in polyTree.nodes()})

#Using a cache and visited nodes to avoid recalculating the same thing multiple times, it's cleaner removing the nodes from the polytree but it's slower.
def randomTopoSortsAuxTime(polyTree : nx.DiGraph, numTopoSorts : int, topoSortsCache : dict[tuple[Node], int], visited : dict[Node, bool]) -> List[List[Node]]:

    start_time = time.time()
    noVisitedNodes = notVisitedNodes(visited)
    totalTime['notVisitedNodes'] = totalTime.get('notVisitedNodes', 0) + (time.time() - start_time)

    if len(noVisitedNodes) <= 1:
        return [noVisitedNodes]*numTopoSorts

    start_time = time.time()
    sourceNodes = [node for node in noVisitedNodes if isRootWithVisited(node, polyTree, visited)]
    totalTime['isRootWithVisited'] = totalTime.get('isRootWithVisited', 0) + (time.time() - start_time)

    start_time = time.time()
    topoSortsForStartNodes = [0]*len(sourceNodes)
    for i, startNode in enumerate(sourceNodes):
        topoSortsForStartNodes[i] = calculateTopoSortsForNode(startNode, polyTree, visited, topoSortsCache)
    totalTime['calculateTopoSortsForNode'] = totalTime.get('calculateTopoSortsForNode', 0) + (time.time() - start_time)

    start_time = time.time()
    firstNodes = obtainRandomStartNodes(numTopoSorts, topoSortsForStartNodes, sourceNodes)
    totalTime['obtainRandomStartNodes'] = totalTime.get('obtainRandomStartNodes', 0) + (time.time() - start_time)

    topoSorts = []
    for firstNode, count in Counter(firstNodes).items():
        visited[firstNode] = True
        results = randomTopoSortsAux(polyTree, count, topoSortsCache, visited)
        start_time = time.time()
        topoSorts += [[firstNode] + topoSort for topoSort in results]
        totalTime['addingSourceNodesToTopos'] = totalTime.get('addingSourceNodesToTopos', 0) + (time.time() - start_time)
        visited[firstNode] = False
    

    return topoSorts