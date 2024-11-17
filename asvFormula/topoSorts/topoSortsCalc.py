
from asvFormula.digraph import  isLeaf, hasMultipleParents, nx
import math
from toposPositions import multinomial_coefficient

def sizeAndNumberOfTopoSorts(node, tree : nx.DiGraph):
    if isLeaf(node, tree):
        return 1,1
    
    childrenSubtreeSizes = []
    children_topoSorts = []

    
    for child in tree.successors(node):
        recursionNode = child

        # I want to check if the child has more than one parent
        if hasMultipleParents(child, tree):
            parents = list(tree.predecessors(child))
            tree.remove_edge(node, child)
            anotherParent = next((parent for parent in parents if parent!=node), None)
            recursionNode = anotherParent
            
        child_size, child_topos =  sizeAndNumberOfTopoSorts(recursionNode,tree)
        children_topoSorts.append(child_topos)
        childrenSubtreeSizes.append(child_size)
        

    topos = multinomial_coefficient(childrenSubtreeSizes) * math.prod(children_topoSorts)
    return sum(childrenSubtreeSizes)+1, topos