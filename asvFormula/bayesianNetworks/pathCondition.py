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
        if feature not in self.featureLimits:
            return 0
        return self.getFeatureLimits(feature)[0]
    
    def getUpperLimit(self, feature):
        if feature not in self.featureLimits:
            return len(self.getValuesOfFeature(feature))-1
        return self.getFeatureLimits(feature)[1]

    def setVariableUpperLimit(self, feature : str, upperLimit : int):
        lowerLimit = self.getLowerLimit(feature)

        self.featureLimits[feature] = (lowerLimit, upperLimit)

    def setVariableLowerLimit(self, feature : str, lowerLimit : int):
        upperLimit = self.getUpperLimit(feature)

        self.featureLimits[feature] = (lowerLimit, upperLimit)

    def removeVariable(self, feature : str):
        if feature in self.featureLimits:
            del self.featureLimits[feature]

    def removeVariableLowerLimit(self, feature : str):
        self.featureLimits[feature] = (0, self.getUpperLimit(feature))

    def removeVariableUpperLimit(self, feature : str):
        self.featureLimits[feature] = (self.getLowerLimit(feature), len(self.getValuesOfFeature(feature))-1)

    def allPossibleEvents(self):
        conditions = []
        for feature in self.getVariables():
            conditions.append(self.getValuesOfFeature(feature)[self.getLowerLimit(feature) : self.getUpperLimit(feature)+1])
        classes_combinations = list(itertools.product(*conditions))
        return classes_combinations
