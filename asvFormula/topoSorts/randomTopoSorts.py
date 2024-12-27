from asvFormula.digraph import nx, isRootWithVisited, isRoot
from asvFormula import Node
from typing import List
from topoSortsCalc import allPolyTopoSorts, allPolyTopoSortsWithVisited
import random, itertools
from collections import Counter
import numpy as np

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
        return [noVisitedNodes]

    sourceNodes = [node for node in noVisitedNodes if isRootWithVisited(node, polyTree, visited)]
    topoSortsForStartNodes = [0]*len(sourceNodes)

    for i, startNode in enumerate(sourceNodes):
        visitedCopy = visited.copy()        
        visitedCopy[startNode] = True

        cacheKey = hash(frozenset(notVisitedNodes(visitedCopy)))
        if not cacheKey in topoSortsCache: #I do this and not get() cause get will always call allPolyTopoSortsWithVisited
            topoSortsCache[cacheKey] = allPolyTopoSortsWithVisited(polyTree, visitedCopy)

        topoSortsForStartNodes[i] = topoSortsCache[cacheKey] # All of the topological sorts we would have if we picked this node as the first one

    allTopoSorts = sum(topoSortsForStartNodes)
    probs = [topoSorts/allTopoSorts for topoSorts in topoSortsForStartNodes]
    firstNodes = random.choices(sourceNodes,weights=probs, k=numTopoSorts)
    
    topoSorts = []
    for firstNode, count in Counter(firstNodes).items():        
        visited[firstNode] = True
        
        topoSorts.extend([ [firstNode] + topoSort for topoSort in randomTopoSortsAux(polyTree, count, topoSortsCache, visited)])

        visited[firstNode] = False

    return topoSorts

def notVisitedNodes(visited : dict[Node, bool]) -> List[Node]:
    return [node for node, isVisited in visited.items() if not isVisited]