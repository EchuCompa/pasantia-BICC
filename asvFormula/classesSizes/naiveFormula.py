from typing import List, Any
from asvFormula.topoSorts.topoSorts import TopoSortHasher

def naiveEquivalenceClassesSizes(all_topo_sorts : List[List[Any]], feature_node : Any, hasher : TopoSortHasher):
      
   result = {} 
   for topoSort in all_topo_sorts:
      hash = hasher.hashTopoSort(topoSort, feature_node)
      actual_value = result.get(hash, [topoSort, 0])
      result[hash] = [actual_value[0], actual_value[1] + 1] 
      # It has a representative of each class and the number of topological orders that are in that class.

   return result

#TODO: Here I don't need the topological orders with the descendants of feature_node, I can remove them and then multiply the number of topological orders of each class. 
# To do this I just need to calculate the "merging" of this possible topological orders as I do in the recursive approach. So when I calculate all_topo_sorts I
# can do it with only the unrelated nodes and the ascendants, removing the descendants. But to do do this I need to recalculate all_topo_sorts for each feature node.
