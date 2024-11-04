import itertools

class PathCondition:

    def __init__(self, featuresValues : dict[str, list]):
        self.featureLimits = {}
        # It's a dictionary of ranges, the key is the feature and the value is a tuple with  [minimum,maximum] values. Its a closed interval. 
        self.featuresValues = featuresValues
        
    def getValuesOfFeature(self, feature):
        return self.featuresValues[feature]
    
    def getFeatureLimits(self, feature) -> tuple:
        return self.featureLimits[feature]
    
    def getVariables(self):
        return list(self.featureLimits.keys())

    def getLowerLimit(self, feature):
        if feature not in self.featureLimits: #This may not be the best, maybe I should raise an exception?
            return 0
        return self.getFeatureLimits(feature)[0]
    
    def getUpperLimit(self, feature):
        if feature not in self.featureLimits:
            return self.possibleValuesFor(feature) - 1
        return self.getFeatureLimits(feature)[1]

    def possibleValuesFor(self, feature):
        return len(self.getValuesOfFeature(feature))

    def setVariableUpperLimit(self, feature : str, upperLimit : int):
        lowerLimit = self.getLowerLimit(feature)

        self.setNewLimit(feature, lowerLimit, upperLimit)

    def setVariableLowerLimit(self, feature : str, lowerLimit : int):
        upperLimit = self.getUpperLimit(feature)

        self.setNewLimit(feature, lowerLimit, upperLimit)

    def setNewLimit(self, feature : str, lowerLimit : int, upperLimit : int):
        if upperLimit+1 - lowerLimit == len(self.getValuesOfFeature(feature)): #This is the same as not having any limit
            self.removeVariable(feature)
        else:
            self.featureLimits[feature] = (lowerLimit, upperLimit)

    def removeVariable(self, feature : str):
        if feature in self.featureLimits:
            del self.featureLimits[feature]

    def allPossibleEvents(self):
        conditions = []
        for feature in self.getVariables():
            conditions.append(self.getValuesOfFeature(feature)[self.getLowerLimit(feature) : self.getUpperLimit(feature)+1])
        classes_combinations = list(itertools.product(*conditions))
        return classes_combinations

    def doesNotMatchEvidence(self, priorEvidence : dict[str, list]) -> bool:
        if priorEvidence is None:
            return False
        for feature in priorEvidence:
            if not self.inRange(priorEvidence, feature):
                return True
        return False

    def inRange(self, priorEvidence : dict[str, list], feature : str) -> bool:
        return self.getLowerLimit(feature) <= priorEvidence[feature] <= self.getUpperLimit(feature)
    
    def decodeEvidence(self, priorEvidence : dict[str, list]) -> dict[str, str]:
        if priorEvidence is None:
            return None
        
        decodeFeature = lambda encodedValue, feature : self.getValuesOfFeature(feature)[encodedValue]
        return {feature : decodeFeature(priorEvidence[feature], feature) for feature in priorEvidence}
    
    def removeEvidence(self, priorEvidence : dict[str, list]):
        if not priorEvidence is None:
            for feature in priorEvidence:
                self.removeVariable(feature)