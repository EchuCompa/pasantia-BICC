from typing import List, Any
from asvFormula.topoSorts.utils import TopoSortHasher

def naiveEquivalenceClassesSizes(all_topo_sorts : List[List[Any]], feature_node : Any, hasher : TopoSortHasher):
      
   result = {} 
   for topoSort in all_topo_sorts:
      hash = hasher.hashTopoSort(topoSort, feature_node)
      actual_value = result.get(hash, [topoSort, 0])
      result[hash] = [actual_value[0], actual_value[1] + 1] 
      # It has a representative of each class and the number of topological orders that are in that class.

   return result
