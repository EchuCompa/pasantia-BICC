import sys, os
from typing import List
# This is to import the modules correctly from the parent directory
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if module_path not in sys.path:
    sys.path.append(module_path)

#There must be a better way to do this, but this is good for now

#global networkSamplesPath 
asvRunResultsPath = "/results/asvRunResults"

from typing import NewType, Any
Node = NewType('Node', Any)

# The idea would be that it has the left elements of each unrelated class, ordered by the ascendant node that is their parent. 

def getPossibleCombinations(leftElementsOfClasses: List[int], sumToObtain: int) -> List[List[int]]:
    def backtrack(index, current_combination, current_sum, maximumAmount):
        # If the current sum equals the required elementsToSelect, add the combination to the result
        if current_sum == sumToObtain:
            result.append(list(current_combination))
            return
        
        if current_sum > sumToObtain or index == len(leftElementsOfClasses) or current_sum + maximumAmount < sumToObtain:
            return
        
        for value in range(leftElementsOfClasses[index] + 1):
            current_combination.append(value)
            backtrack(index + 1, current_combination, current_sum + value, maximumAmount - leftElementsOfClasses[index])
            current_combination.pop() 

    result = []
    backtrack(0, [], 0, sum(leftElementsOfClasses))
    return result