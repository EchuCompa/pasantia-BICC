#We are obtaining the bayesian networks from https://www.bnlearn.com/bnrepository

from pgmpy.inference import VariableElimination
from bayesianNetworks.bayesianNetwork import *
from datasetManipulation import *
from classesSizes.recursiveFormula import *
from asvFormula.topoSorts.utils import allForestTopoSorts
from functools import lru_cache
import shap

class ASV:
    def __init__(self, dag : nx.DiGraph, model, feature_distributions : VariableElimination, valuesPerFeature : dict[str, list], feature : str, predictionFunction : str = 'Exact'):
        self.dag = dag
        self.model = model
        self.feature_distributions = feature_distributions
        self.valuesPerFeature = valuesPerFeature
        self.predictionFunction = (lambda n, tree : tree.nodeMeanPrediction(n)) if predictionFunction == 'Mean' else lambda n, tree : tree.nodePrediction(n)

        modelFeatures = list(valuesPerFeature.keys())
        modelFeatures.remove(feature)
        self.modelAsTree = obtainDecisionTreeDigraph(model, list(modelFeatures))

    #It returns the ASV value for each possible value of the feature. 
    def asvForFeature(self, feature : str, instance : pd.Series, showProgress = False) -> list[float]: 
        
        equivalenceClasses = equivalenceClassesFor(self.dag, feature)
        asvValue = 0
        totalTopologicalOrders = 0
        allTopologicalOrders = allForestTopoSorts(self.dag)

        for i, equivalenceClass in enumerate(equivalenceClasses):
            classFeaturesOrder = equivalenceClass[0]
            classSize = equivalenceClass[1]
            totalTopologicalOrders += classSize
            asvValue += classSize * (self.meanPredictionForEquivalenceClass(classFeaturesOrder, feature, instance, True) -  
                                    self.meanPredictionForEquivalenceClass(classFeaturesOrder, feature, instance, False))
            if showProgress: print(f'Progress of classes processed: {i/len(equivalenceClass)}%')

        assert totalTopologicalOrders == allTopologicalOrders, f"The total number of topological orders for the equivalence classes is {totalTopologicalOrders} and the total number of topological orders for the graph is {allTopologicalOrders}"
        return asvValue/allTopologicalOrders #This is the normalization for the ASV value

    def meanPredictionForEquivalenceClass(self, classFeaturesOrder : List[str], feature : str, instance : pd.Series, featureFixed : bool) -> list[float]:
        fixedFeaturesFrom = classFeaturesOrder.index(feature) + (1 if featureFixed else 0)
        fixedFeatures = classFeaturesOrder[:fixedFeaturesFrom]

        priorEvidence = {feature : instance[feature] for feature in fixedFeatures}
        rootNodes = [node for node in self.modelAsTree.nodes if isRoot(node, self.modelAsTree)] 
        root = rootNodes[0]

        return meanPredictionForDTinBNWithEvidence(self.modelAsTree, root, self.feature_distributions, PathCondition(self.valuesPerFeature), priorEvidence, nodePrediction=self.predictionFunction)
    
    def oldMeanPredictionForEquivalenceClass(self, classFeaturesOrder : List[str], feature : str, instance : pd.Series, featureFixed : bool) -> float:
        
        fixedFeaturesFrom = classFeaturesOrder.index(feature) + (1 if featureFixed else 0)
        fixedFeatures = classFeaturesOrder[:fixedFeaturesFrom]
        variableFeatures = classFeaturesOrder[fixedFeaturesFrom:]

        #TODO: Optimize this section, it takes too much time
        consistentDataset = self.consistentInstances(instance, fixedFeatures, variableFeatures)

        predictions = self.model.predict(consistentDataset)

        probabilities = consistentDataset.apply(
            lambda row: self.probOfInstance(row, fixedFeatures), 
            axis=1
        )
        #print(self.cached_prob_of_instance.cache_info())
        meanPrediction = np.dot(predictions, probabilities)
        return [1-meanPrediction, meanPrediction] #Esto es cualca, es para que corra nada más

    def consistentInstances(self, instance: pd.Series, fixedFeatures: list[str], variableFeatures: list[str]) -> pd.DataFrame:
        fixed_values = {feature: instance[feature] for feature in fixedFeatures}
        fixed_part_df = pd.DataFrame([fixed_values])

        variable_values = [range(len(self.valuesPerFeature[feature])) for feature in variableFeatures] #This is because the LabelEncoder() encodes the values from [0: n_values]
        variable_combinations_df = pd.DataFrame(itertools.product(*variable_values), columns=variableFeatures)

        fixed_part_repeated = pd.concat([fixed_part_df] * len(variable_combinations_df), ignore_index=True)

        # Step 4: Concatenate fixed and variable parts and reorder columns
        consistentInstances = pd.concat([fixed_part_repeated, variable_combinations_df], axis=1)
        consistentInstances = consistentInstances[instance.index]  # Ensure column order matches the original instance

        return consistentInstances

    # TODO: See if there is a better way to cache/encode instances to generate more hits.
    @lru_cache(maxsize=None)
    def cached_prob_of_instance(self, tuple_instance: Tuple, realFeaturesTuple: Tuple[str], rowIndex : Tuple) -> float:
        # Convert tuple back to pandas Series
        matching_instance = pd.Series(data=list(tuple_instance),index=list(rowIndex) )
        return self.probOfInstance(matching_instance, list(realFeaturesTuple))

    def probOfInstance(self, matchingInstance : pd.Series, realFeatures : List[str]) -> float:
        decodedInstance = self.decodeInstance(matchingInstance)
        priorEvidence = {realFeature : decodedInstance[realFeature] for realFeature in realFeatures}
        variablesToEstimate = [feature for feature in decodedInstance.keys() if feature not in realFeatures]
        
        if len(variablesToEstimate) == 0:
            return 1 #If the evidence is complete, the probability of the instance is 1
        
        inference = self.feature_distributions.query(variables=variablesToEstimate, evidence=priorEvidence)
        
        return inference.get_value(**{var : decodedInstance[var] for var in variablesToEstimate})


    def decodeInstance(self,instance : pd.Series) -> pd.Series:
        decodedInstance = instance.copy()
        for feature in instance.keys():
            if feature in self.valuesPerFeature:
                decodedInstance[feature] = self.valuesPerFeature[feature][instance[feature]]
        return decodedInstance
    
    #TODO: Finish this so that I can compare the mean predictions of the two approaches. I need to adapt the old mean prediction output to the new one

    def assertAllMeanPredictionsAreSimilar(self, instance : pd.Series, showProgress = False) -> list[float]: 

        features = [feature for feature in self.valuesPerFeature.keys() if feature in self.dag.nodes]
        for feature in features:
            for equivalenceClass in equivalenceClassesFor(self.dag, feature):
                classFeaturesOrder = equivalenceClass[0]
                self.assertMeanPredictionsAreSimilar(classFeaturesOrder, feature, instance, True)

    def assertMeanPredictionsAreSimilar(self, classFeaturesOrder : List[str], feature : str, instance : pd.Series, featureFixed : bool):
        
        meanPredAlgorithm = self.meanPredictionForEquivalenceClass(classFeaturesOrder, feature, instance, featureFixed)
        oldMeanPredAlgorithm = self.oldMeanPredictionForEquivalenceClass(classFeaturesOrder, feature, instance, featureFixed)
        assert np.allclose(meanPredAlgorithm, oldMeanPredAlgorithm, atol=1.e-2), f"The mean prediction for the new algorithm is {meanPredAlgorithm} and the mean prediction for the old algorithm is {oldMeanPredAlgorithm}"


def showASVandShapleyFor(first_instance : pd.Series, features : list[str], dtTreeClassifier : DecisionTreeClassifier, asv : ASV , progress = False):
    sumOfAsv = 0   
    print("ASV values for the first instance:")
    for feature in features:
        asvValue = asv.asvForFeature(feature, first_instance, showProgress=progress)
        sumOfAsv += asvValue
        print(f"ASV for {feature}: {asvValue}")

    print(f"Sum of ASV values: {sumOfAsv}")

    explainer = shap.TreeExplainer(dtTreeClassifier)
    shap_values = explainer.shap_values(first_instance)

    sumOfAShapleyValues = 0
    print("Shapley values for the first instance:")
    for feature, shap_value in zip(features, shap_values):
        sumOfAShapleyValues += shap_value[1]
        print(f"Feature: {feature}, Shapley Value: {shap_value[1]}")

    print(f"Sum of Shapley values: {sumOfAShapleyValues}")

#I want to the same as the previous function but saving the results in a file

def writeASVAndShapleyIntoFile(first_instance : pd.Series, features : list[str], dtTreeClassifier : DecisionTreeClassifier, asv : ASV, file : str, valuesPerFeature : dict[str,list], variableToPredict : str, progress = False):
    # I want it in a CSV format as a table. With the features as the columns and the values as the rows
    explainer = shap.TreeExplainer(dtTreeClassifier)
    shap_values = explainer.shap_values(first_instance)
    predictionValues = valuesPerFeature[variableToPredict]
    with open(file, 'w') as f:
        f.write(f"Feature,{variableToPredict} value, ASV,Shapley\n")
        sumOfAsv = np.zeros(dtTreeClassifier.n_classes_)
        for i, feature in enumerate(features):
            asvValue = asv.asvForFeature(feature, first_instance, showProgress=progress)
            sumOfAsv += asvValue
            shapleyValue = shap_values[i]
            for i in range(len(shapleyValue)):
                f.write(f"{feature},{predictionValues[i]},{asvValue[i]},{shapleyValue[i]}\n")
        f.write(f"Sum,{sumOfAsv},{sum(shap_values)}\n")