from typing import Any, Dict, List
from digraph import nx, classifyNodes, orderedNodes, NodeState
from topoSorts import allTopoSorts, TopoSortHasher, topoSortsFrom
from equivalenceClass import EquivalenceClass, numberOfEquivalenceClasses
from recursiveFormula import unrelatedEquivalenceClassesSizes, lastUnionOf, uniteClassesWithSameParent, hashEquivClasses
import time

def assertTopoSortsAndEquivalenceClasses(dag, feature_node, recursiveClassesSizes):
    sumOfAllClasses = sum(map(lambda eqClass : eqClass[1],recursiveClassesSizes.values()))
    assert (sumOfAllClasses == allTopoSorts(dag)), f"Number of topological sorts is different than the sum of all equivalence classes. \n Topological Sorts: {allTopoSorts(dag)}, Sum of all classes: {sumOfAllClasses}"
    assert (len(recursiveClassesSizes) == numberOfEquivalenceClasses(dag, feature_node)), f"Number of equivalence classes is different than the expected. \n Expected: {numberOfEquivalenceClasses(dag, feature_node)}, Actual: {len(recursiveClassesSizes)}"


def measure_graph_data(dag: nx.DiGraph, nodesToEvaluate : List[Any]) -> Dict[str, Any]:

    times_per_node = []
    equiv_classes_per_node = []

    graph_data = timeRecursiveFunctionFor(dag, nodesToEvaluate)

    for node in nodesToEvaluate:
        node_data = graph_data[node]
        times_per_node.append(node_data['Total Time'])
        equiv_classes_per_node.append(numberOfEquivalenceClasses(dag, node))


    def obtainMaxMinAvg(data):
        maxValue = max(data)
        maxValue = maxValue  # f', Index of node: {data.index(maxValue)}'
        minValue = min(data)
        minValue = minValue #+ f', Index of node: {data.index(minValue)}'
        average = sum(data) / len(data)
        return maxValue, minValue, average
   
    longest_time, shortest_time, average_time = obtainMaxMinAvg(times_per_node)
    biggest_equiv_classes, smallest_equiv_classes, average_equiv_classes = obtainMaxMinAvg(equiv_classes_per_node)

    return {
        "allTopoSortsNumber": allTopoSorts(dag),
        "recursiveLongestTime": longest_time,
        "recursiveShortestTime": shortest_time,
        "recursiveAverageTime": average_time,
        "biggestEquivClasses": biggest_equiv_classes,
        "smallestEquivClasses": smallest_equiv_classes,
        "averageEquivClasses": average_equiv_classes,
    }

def timeOfAllTopologicalSorts(dag, printTime = 500000, maxTopoSorts = 10000000):
    allTopoSorts = []
    timing_dict = {}
    start_time = time.time()
    for topoSort in nx.all_topological_sorts(dag):
        allTopoSorts.append(topoSort)
        if (len(allTopoSorts) % printTime == 0):
            timing_dict[len(allTopoSorts)]= time.time() - start_time
            print(f'Number of topological sorts: {len(allTopoSorts)}, time: {timing_dict[len(allTopoSorts)]}')
        if (len(allTopoSorts) == maxTopoSorts):
            break
    return timing_dict, allTopoSorts

def timeRecursiveFunctionParts(dag : nx.DiGraph, unr_roots : List[Any], hasher : TopoSortHasher, feature_node, nodes_classification : Dict[Any, NodeState]) -> List[EquivalenceClass]:
    run_data = {}

    start_time = time.time()
    unr_classes = list(map(lambda child : unrelatedEquivalenceClassesSizes(child,dag), unr_roots))
    unr_classes = uniteClassesWithSameParent(unr_classes, dag)
    end_time = time.time()
    run_data['Unrelated Classes Calculation'] = end_time - start_time

    ancestors = orderedNodes(dag, nx.ancestors(dag, feature_node))
    descendants = orderedNodes(dag, nx.descendants(dag, feature_node))

    descendantsTopoSorts = topoSortsFrom(feature_node, dag)
    start_time = time.time()
    recursiveClassesSizes = lastUnionOf(unr_classes, ancestors, descendants, descendantsTopoSorts, dag, nodes_classification)
    recursiveClassesSizes = hashEquivClasses(recursiveClassesSizes, hasher, feature_node, dag)
    assertTopoSortsAndEquivalenceClasses(dag, feature_node, recursiveClassesSizes)
    end_time = time.time()
    run_data['Last Union Calculation'] = end_time - start_time

    return run_data

def timeRecursiveFunctionFor(dag, nodesToEvaluate):
    global memoHits
    timing_dict = {}
    for feature_node in nodesToEvaluate:
        #(f'Running for node {feature_node} which has {numberOfEquivalenceClasses(dag, feature_node)} equivalence classes')
        memoHits = 0
        timing_dict[feature_node] = {}
        nodes_classification = {}
        unr_roots = classifyNodes(dag, feature_node, nodes_classification)
        hasher = TopoSortHasher(nodes_classification)
        # Recursive approach
        start_time = time.time()
        recursiveFunctionResult = timeRecursiveFunctionParts(dag, unr_roots, hasher, feature_node, nodes_classification)
        end_time = time.time()
        timing_dict[feature_node]['Recursive Formula'] = recursiveFunctionResult
        timing_dict[feature_node]['Total Time'] = end_time - start_time
        timing_dict[feature_node]['Memoization Hits'] = memoHits
    
        #print(f'Node {feature_node} took {timing_dict[feature_node]["Total Time"]} seconds to run')
    return timing_dict