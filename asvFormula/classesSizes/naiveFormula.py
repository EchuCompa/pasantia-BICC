from typing import List
from asvFormula import Node
from asvFormula.topoSorts.utils import TopoSortHasher

def naiveEquivalenceClassesSizes(all_topo_sorts : List[List[Node]], feature_node : Node, hasher : TopoSortHasher) -> dict[int, tuple[Node, int]]:
      
   result = {} 
   for topoSort in all_topo_sorts:
      hash = hasher.hashTopoSort(topoSort, feature_node)
      actual_value = result.get(hash, [topoSort, 0])
      result[hash] = [actual_value[0], actual_value[1] + 1] 
      # It has a representative of each class and the number of topological orders that are in that class.

   return result
