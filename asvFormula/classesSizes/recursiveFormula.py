from typing import List, Dict, Tuple
from asvFormula.digraph import nx, orderedNodes, isLeaf, isRoot, classifyNodes
from asvFormula.topoSorts.utils import TopoSortHasher, topoSortsFrom, hashEquivClasses, multinomial_coefficient
from equivalenceClass import EquivalenceClass, NodePosition
from asvFormula import Node, getPossibleCombinations
import itertools
import math
memoHits = 0


def equivalenceClassesFor(dag, feature_node) -> List[EquivalenceClass]:
   
    return list(equivalanceClassesSizesWithHashes(dag, feature_node).values())


def equivalanceClassesSizesWithHashes(dag, feature_node) -> Dict[int, tuple[List[Node], int]]:
    nodes_classification = {}
    unr_roots = classifyNodes(dag, feature_node, nodes_classification)
    hasher = TopoSortHasher(dag, feature_node)
    equivClasses = recursiveEquivalenceClassesSizes(dag, unr_roots, hasher, feature_node)

    return equivClasses


def recursiveEquivalenceClassesSizes(dag : nx.DiGraph, unr_roots : List[Node], hasher : TopoSortHasher, feature_node) -> Dict[int, tuple[Node, int]]:
    unr_classes = list(map(lambda child : unrelatedEquivalenceClassesSizes(child,dag), unr_roots))
    unr_classes = uniteClassesWithSameParent(unr_classes, dag)
    ancestors = orderedNodes(dag, nx.ancestors(dag, feature_node))
    descendants = orderedNodes(dag, nx.descendants(dag, feature_node))

    descendantsTopoSorts = topoSortsFrom(feature_node, dag)
    recursiveClassesSizes = lastUnionOf(unr_classes, ancestors, descendants, descendantsTopoSorts, dag)

    recursiveClassesSizes = hashEquivClasses(recursiveClassesSizes, hasher, feature_node, dag)
    return recursiveClassesSizes

def unrelatedEquivalenceClassesSizes(node, dag : nx.DiGraph) -> List[EquivalenceClass]:
    if isLeaf(node, dag):
        classes = []
        for x in [False, True]:
           classes.append(EquivalenceClass([NodePosition(node, x)]))
        return classes
    
    children_classes = list(map(lambda child : unrelatedEquivalenceClassesSizes(child,dag), dag.successors(node)))

    classes_combinations = list(itertools.product(*children_classes)) #Generate al the possible combinations for each eqClass of each child with the eqClass of the other children. 
    
    # All the equivalence classes will have this node in the left part. 

    classes = list(map(lambda mix : uniteChildrenAndAddParent(node, list(mix)), classes_combinations))

    allRight = unionOf(list(classes_combinations[len(classes_combinations)-1]))
    if allRight.num_nodes_before() == 0:
        allRight.addParentToRigth(node)
        classes.append(allRight)
        # If the parent is to the right, then all of the children should be after the feature node

    return classes

#TODO : Add dynamic programming so that it stores the result of the run for each node, or it stores some results so that it can reconstruct the solution.

def uniteChildrenAndAddParent(node, equivalence_classes : List[EquivalenceClass]) -> EquivalenceClass:
        union = unionOf(equivalence_classes)
        union.addParent(node)

        return union

#This function unites the unrelated class that have the same parent, so that all of the nodes that have the same dependency are in the same position in leftElementOfClasses.

def uniteClassesWithSameParent(equivalence_classes : List[List[EquivalenceClass]], dag) -> List[List[EquivalenceClass]]:
    parentToClasses = {}
    for eqClasses in equivalence_classes:
        classParent = eqClasses[0].classParent()
        parent = 'Root' if isRoot(classParent, dag) else list(dag.predecessors(classParent))[0]
        listOfClasses = parentToClasses.get(parent, [])
        listOfClasses.append(eqClasses)
        parentToClasses[parent] = listOfClasses
    
    for parent, classes in parentToClasses.items():
        if len(classes) > 1:
            classesCombinations = list(itertools.product(*classes))
            parentToClasses[parent] = list(map(lambda mix : unionOf(list(mix)), classesCombinations))
        else:
            parentToClasses[parent] = classes[0]
        
    
    return list(parentToClasses.values())

def unionOf(equivalence_classes : List[EquivalenceClass], addLeftToposOrder : bool = True) -> EquivalenceClass:
    n = len(equivalence_classes)
    positions = []
    nodes_before = [0]*n
    nodes_after = [0]*n
    left_topos = [0]*n
    right_topos = [0]*n
    for i,eq_class in enumerate(equivalence_classes):
        nodes_before[i] = eq_class.num_nodes_before()
        nodes_after[i] = eq_class.num_nodes_after()
        left_topos[i] = eq_class.left_topo
        right_topos[i] = eq_class.right_topo
        positions = positions + eq_class.allNodes()

    left_size = math.prod(left_topos)
    if addLeftToposOrder:
        left_size *= multinomial_coefficient(nodes_before)  
        
    right_size = multinomial_coefficient(nodes_after) * math.prod(right_topos)
    return EquivalenceClass(positions, left_size, right_size)

# TODO: Make some of the variables global or create a class for this method, so that I don't need to pass them as arguments

def lastUnionOf(unr_classes : List[List[EquivalenceClass]], ancestors : List[Node], descendants : List[Node], descendantsTopoSorts : int, dag : nx.DiGraph) -> List[EquivalenceClass]:
    classes_combinations = list(itertools.product(*unr_classes)) #Generate al the possible combinations for each eqClass of each child with the eqClass of the other children. 
    
    descendants_position = [NodePosition(des, True) for des in descendants]
    descendants_eqClass = EquivalenceClass(descendants_position,1, descendantsTopoSorts) 
    classes = []
    # All the descendants appear after the feature node, because all of them appear before it then it has 1 rigth_topo (the empty one). 
    if (len(ancestors) == 0):
        classes = list(map(lambda mix : unionOf(list(mix)), classes_combinations))
        
        if len(descendants) != 0:
            classes = [unionOf([descendants_eqClass, mix]) for mix in classes]

    else:
        memorization = {}
        for unr_class in classes_combinations:
            leftElements = getLeftElementsOfClasses(ancestors, dag, unr_class)
            ascendantsCombinationsWithUnrelated = possibleLeftOrders(0, leftElements, 0 , list(ancestors), memorization)

            eqClass = unionOf(unr_class, False)

            eqClass.addAncestors(ancestors)
            eqClass.addLeftTopo(ascendantsCombinationsWithUnrelated)

            if (len(descendants) != 0):
                eqClass = unionOf([eqClass, descendants_eqClass])
                
            classes.append(eqClass)
    
    return classes

def getLeftElementsOfClasses(ancestors : List[Node], dag : nx.DiGraph, unrClasses : List[EquivalenceClass]):
    leftElements = [0]*(len(ancestors)+1)

    for unrClass in unrClasses: #These trees will always be available to use, because they are not related to any ancestor. They can go before the root of the ancestors even.
            if isRoot(unrClass.classParent(), dag):
                leftElements[0] = unrClass.num_nodes_before()
            else:
                parent = list(dag.predecessors(unrClass.classParent()))[0]
                leftElements[ancestors.index(parent)+1] = unrClass.num_nodes_before()

    return leftElements

def removePutElements(putElements, leftElementsOfClasses : List[int]):
    for i, put in enumerate(putElements):
        leftElementsOfClasses[i] -= put

def addPutElements(putElements, leftElementsOfClasses : List[int]):
    for i, put in enumerate(putElements):
        leftElementsOfClasses[i] += put

def possibleLeftOrders(actualPosition : int, leftElementsOfClasses : List[int], ancestorIndex : int, ancestors : List[Node], memo: Dict[Tuple[int, Tuple[int], int, int], int]) -> int:
    global memoHits
    state = (actualPosition, tuple(leftElementsOfClasses), ancestorIndex)

    if state in memo:
        memoHits += 1
        return memo[state]

    if (sum(leftElementsOfClasses) == 0): #There are no more elements to select
        return 1
    
    totalOrders = 0
    #I just need to select all of the elements of the classes.
    if (ancestorIndex == len(ancestors)): #I have already selected all the ancestors
        totalOrders =  multinomial_coefficient(leftElementsOfClasses)
        memo[state] = totalOrders
        return totalOrders

    usableElements = leftElementsOfClasses[:ancestorIndex+1]

    for ancestorPosition in range(actualPosition, actualPosition + sum(usableElements) + 1):
        positionsToFill = ancestorPosition - actualPosition
        for comb in getPossibleCombinations(usableElements, positionsToFill):
            removePutElements(comb, leftElementsOfClasses)
            totalOrders += multinomial_coefficient(comb) * possibleLeftOrders(ancestorPosition+1, leftElementsOfClasses, ancestorIndex + 1, ancestors, memo)
            addPutElements(comb, leftElementsOfClasses)
    
    memo[state] = totalOrders
    return totalOrders

# TODO: Make some of the variables global or create a class for this method, so that I don't need to pass them as arguments
# Improve the memoization data structure. Look if there is a structure that is useful for this kind of problem. ¿Maybe use lru_cache or other structure?
