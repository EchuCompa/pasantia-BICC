import time
from digraph import *
from equivalenceClass import numberOfEquivalenceClasses
from  topoSorts import TopoSortHasher, allTopoSorts
from algorithmTime import assertTopoSortsAndEquivalenceClasses
from  recursiveFormula import *
from  naiveFormula import *

def assertEquivalenceClassesForNode(dag: nx.DiGraph, feature_node, all_topo_sorts: List[List[Any]], timing_dict: Dict[str, Dict[str, float]]):
    
    nodes_classification = {}
    unr_roots = classifyNodes(dag, feature_node, nodes_classification)
    hasher = TopoSortHasher(nodes_classification)

    # Naive approach
    start_time = time.time()
    naiveClassesSizes = naiveEquivalenceClassesSizes(all_topo_sorts, feature_node, hasher)
    end_time = time.time()
    timing_dict[feature_node]['Naive Formula'] = end_time - start_time

    # Recursive approach
    start_time = time.time()
    recursiveClassesSizes = recursiveEquivalenceClassesSizes(dag, unr_roots, hasher, feature_node, nodes_classification)
    end_time = time.time()
    timing_dict[feature_node]['Recursive Formula'] = end_time - start_time

    timing_dict[feature_node]['Number of equivalence classes'] = len(naiveClassesSizes.keys())
    
    # Assert that each equivalence class has the same number of elements.

    naiveEqClasses = len(naiveClassesSizes.keys())
    recursiveEqClasses = len(recursiveClassesSizes.keys())
    if naiveEqClasses != recursiveEqClasses and numberOfEquivalenceClasses(dag, feature_node) != naiveEqClasses:
        raise AssertionError(f"The number of equivalence classes is different. \n Naive Approach: {naiveEqClasses}, Recursive Approach: {recursiveEqClasses} \n Feature Node: {feature_node}")

    assertTopoSortsAndEquivalenceClasses(dag, feature_node, recursiveClassesSizes)

    for eqClassHash in naiveClassesSizes.keys():
        clSize1 = naiveClassesSizes[eqClassHash][1]
        clTopo1 = naiveClassesSizes[eqClassHash][0]
        try: 
            clSize2 = recursiveClassesSizes[eqClassHash][1]
            clTopo2 = recursiveClassesSizes[eqClassHash][0]
        except KeyError:
            raise AssertionError(f"The equivalence class {eqClassHash} is not present in the recursive approach. \n Naive Approach: Topo {clTopo1}, Size {clSize1} \n Feature Node: {feature_node}")
        if (clSize1 != clSize2):
            raise AssertionError(f"The sizes of the equivalence classes are not equal. \n Naive Approach: Topo {clTopo1}, Size {clSize1} \n Recursive Approach: Topo {clTopo2}, Size {clSize2} \n Feature Node: {feature_node}")

#TODO: Find a better algorithm than all_topological_sorts, it takes too much time. In the paper they mention a dynamic programming approach, maybe implement that. This takes too much time.

def assertEquivClassesForDag(dag: nx.DiGraph, nodesToEvaluate = None, allSorts = None) -> Dict[str, float]:
    timing_dict = {}
    
    # Measure time for all topological sorts
    start_time = time.time()
    all_topo_sorts = allSorts if allSorts != None else list(nx.all_topological_sorts(dag))
    assert len(all_topo_sorts) == allTopoSorts(dag)
    end_time = time.time()
    timing_dict['Time Of Topological Sorts'] = end_time - start_time
    timing_dict['Number of Topological Sorts'] = len(all_topo_sorts)
    
    nodesToEvaluate = nodesToEvaluate if nodesToEvaluate != None else list(dag.nodes)
    for node in nodesToEvaluate:
            timing_dict[node] = {}
            assertEquivalenceClassesForNode(dag, node, all_topo_sorts, timing_dict)
    
    return timing_dict

def test_allTopos(graph):
    all_topos = allTopoSorts(graph)
    all_topo_sorts = list(nx.all_topological_sorts(graph))
    assert all_topos == len(all_topo_sorts), "allTopos and all_topological_sorts have different lengths"