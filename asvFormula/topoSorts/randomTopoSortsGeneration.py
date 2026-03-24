from asvFormula.digraph import nx, isRootWithVisited
from asvFormula import Node
from typing import List, Optional
from topoSortsCalc import allPolyTopoSortsWithVisited
import random
from collections import Counter

#TODO : Make it faster. 
global totalTime
totalTime = {}



import time
from functools import wraps

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        totalTime[func.__name__] = totalTime.get(func.__name__, 0) + elapsed_time
        return result
    return wrapper

# We generate a random topological sort by selecting a random node from the source nodes. Defining the probability of picking it as the number of toposorts that would start with that node

def randomTopoSort(polyTree : nx.DiGraph) -> List[Node]:

    return randomTopoSorts(polyTree, 1)[0]

def randomTopoSorts(polyTree : nx.DiGraph, numTopoSorts : int) -> List[List[Node]]:
    #print(f'It will take {numTopoSorts/60000} seconds in the naive way')
    return randomTopoSortsAux(polyTree, numTopoSorts, {}, {node : False for node in polyTree.nodes()})

#Using a cache and visited nodes to avoid recalculating the same thing multiple times, it's cleaner removing the nodes from the polytree but it's slower.
def randomTopoSortsAux(polyTree : nx.DiGraph, numTopoSorts : int, topoSortsCache : dict[frozenset[Node], int], visited : dict[Node, bool]) -> List[List[Node]]:

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

        topoSorts += addFirstNodeToAllTopos(firstNode, results)

        visited[firstNode] = False
    
    return topoSorts

@time_it
def addFirstNodeToAllTopos(firstNode, results):
    return [[firstNode] + topoSort for topoSort in results]

@time_it
def calculateTopoSortsForNode(startNode : Node, polyTree : nx.DiGraph, visited : dict[Node, bool], topoSortsCache : dict[frozenset[Node], int]) -> int:
    visitedCopy = visited.copy()        
    visitedCopy[startNode] = True

    cacheKey = frozenset(notVisitedNodes(visitedCopy))
    if not cacheKey in topoSortsCache: #I do this and not get() cause get will always call allPolyTopoSortsWithVisited
        topoSortsCache[cacheKey] = allPolyTopoSortsWithVisited(polyTree, visitedCopy)
    global totalTime

    return topoSortsCache[cacheKey]

@time_it
def obtainRandomStartNodes(numTopoSorts : int, topoSortsForStartNodes : List[int], sourceNodes : List[Node]) -> List[Node]:
    
    return random.choices(sourceNodes,weights=topoSortsForStartNodes, k=numTopoSorts)

@time_it
def notVisitedNodes(visited : dict[Node, bool]) -> List[Node]:
    return [node for node, isVisited in visited.items() if not isVisited]


# MCMC-based topological sort generation (Huber 1998, "Faster random generation of linear extensions")
# At each step, a random index i is sampled. If swapping positions i and i+1 yields a valid
# linear extension (i.e. the two elements are incomparable in the DAG) and the coin flip is won, the swap is accepted.
# The transitive closure is precomputed so each comparability check is O(1).

def mcmcTopoSort(dag: nx.DiGraph, stepsPerSample: Optional[int] = None) -> List[Node]:
    return mcmcTopoSorts(dag, 1, stepsPerSample)[0]

def mcmcTopoSorts(dag: nx.DiGraph, numTopoSorts: int, stepsPerSample: Optional[int] = None) -> List[List[Node]]:
    n = len(dag.nodes())
    if n <= 1:
        return [list(dag.nodes())] * numTopoSorts

    if stepsPerSample is None:
        stepsPerSample = defaultMixingSteps(n)

    transitiveClosureDAG = nx.transitive_closure_dag(dag)
    current = list(nx.topological_sort(dag))

    # Initial burn-in before collecting any samples
    mcmcSteps(current, transitiveClosureDAG, stepsPerSample)

    topoSorts = []
    for _ in range(numTopoSorts):
        topoSorts.append(current[:])
        mcmcSteps(current, transitiveClosureDAG, stepsPerSample)

    return topoSorts

def defaultMixingSteps(n: int) -> int:
    return n * n * n

@time_it
def mcmcSteps(current: List[Node], transitiveClosureDAG: nx.DiGraph, steps: int) -> None:
    """Runs 'steps' steps of the MCMC chain on 'current' in place.
    A swap of adjacent positions i and i+1 is accepted iff there is no directed
    path from current[i] to current[i+1] in the original DAG."""
    n = len(current)
    for _ in range(steps):
        i = random.randint(0, n - 2)
        coin_flip = random.random() < 0.5
        if not transitiveClosureDAG.has_edge(current[i], current[i + 1]) and coin_flip:
            current[i], current[i + 1] = current[i + 1], current[i]