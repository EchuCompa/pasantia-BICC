from pgmpy.inference import VariableElimination
from bayesianNetworks.bayesianNetwork import *
from datasetManipulation import *
from classesSizes.recursiveFormula import *
from asvFormula.topoSorts.utils import allForestTopoSorts
from functools import lru_cache
from asvFormula.topoSorts.randomTopoSortsGeneration import randomTopoSorts
from asvFormula.classesSizes.naiveFormula import naiveEquivalenceClassesSizes

class ASV:
    def __init__(self, bayesianNetwork : BayesianNetwork, model : DecisionTreeClassifier, featureDistributions : VariableElimination, valuesPerFeature : dict[str, list], featureToPredict : str, predictionFunction : str = 'Exact', instance : pd.Series = None):
        self.dag = bayesianNetworkToDigraph(bayesianNetwork)
        self.model = model
        self.featureDistributions = featureDistributions
        self.valuesPerFeature = valuesPerFeature
        self.predictionFunction = (lambda n, tree : tree.nodeProbabilityPrediction(n)) if predictionFunction == 'Mean' else lambda n, tree : tree.nodePrediction(n)
        self.instance = instance
        self.featureToPredict = featureToPredict

        modelFeatures = list(valuesPerFeature.keys())
        modelFeatures.remove(featureToPredict)
        self.modelAsTree = obtainDecisionTreeDigraph(model, list(modelFeatures))

    #It returns the ASV value for each possible value of the feature for that specific instance. 
    def asvForFeature(self, feature : str, instance : pd.Series, showProgress = False) -> list[float]: 
        self.instance = instance

        equivalenceClasses = self.equivalenceClasses(feature)
        asvValue = 0
        totalTopologicalOrders = 0
        allTopologicalOrders = allForestTopoSorts(self.dag)

        for i, equivalenceClass in enumerate(equivalenceClasses):
            classFeaturesOrder = equivalenceClass[0]
            classSize = equivalenceClass[1]
            totalTopologicalOrders += classSize
            featureIndex = classFeaturesOrder.index(feature)
            fixedFeatures = classFeaturesOrder[:featureIndex]
            fixedFeaturesWithFeature = classFeaturesOrder[:featureIndex+1]
            asvValue += classSize * (self.meanPredictionForEquivalenceClass(fixedFeatures) -  
                                    self.meanPredictionForEquivalenceClass(fixedFeaturesWithFeature))
            if showProgress: print(f'Progress of classes processed: {i/len(equivalenceClass)}%')

        self.asssertToposortsMatches(totalTopologicalOrders, allTopologicalOrders)
        return asvValue/totalTopologicalOrders 

    def asssertToposortsMatches(self, totalTopologicalOrders, allTopologicalOrders):
        assert totalTopologicalOrders == allTopologicalOrders, f"The total number of topological orders for the equivalence classes is {totalTopologicalOrders} and the total number of topological orders for the graph is {allTopologicalOrders}"

    def equivalenceClasses(self, feature):
        return equivalenceClassesFor(self.dag, feature)#This is the normalization for the ASV value

    def meanPredictionForEquivalenceClass(self, fixedFeatures : List[str]) -> list[float]:

        return meanPredictionForDTinBNWithEvidence(self.modelAsTree, rootNode(self.modelAsTree), self.featureDistributions, PathCondition(self.valuesPerFeature), priorEvidenceFrom(self.instance, fixedFeatures), nodePrediction=self.predictionFunction)
    
    def naiveMeanPredictionForEquivalenceClass(self, fixedFeatures : List[str], variableFeatures: List[str]) -> list[float]:
        
        #TODO: Optimize this section, it takes too much time
        consistentDataset = self.consistentInstances(fixedFeatures, variableFeatures)

        predictions = self.model.predict(consistentDataset)

        probabilities = consistentDataset.apply(
            lambda row: self.probOfInstance(row, fixedFeatures), 
            axis=1
        )
        
        # Convertimos a numpy array por comodidad (asumiendo que predictions es iterable)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        result = {}
        # Iteramos sobre cada valor único predicho
        featureToPredictValues = len(self.valuesPerFeature[self.featureToPredict])
        for value in range(featureToPredictValues):
            # Creamos un vector indicador: 1 si la predicción es igual al valor, 0 si no lo es.
            indicator = (predictions == value).astype(float)
            mean_value = np.dot(indicator, probabilities)
            result[value] = mean_value

        return list(result.values())

    def consistentInstances(self, fixedFeatures: list[str], variableFeatures: list[str]) -> pd.DataFrame:
        fixed_values = {feature: self.instance[feature] for feature in fixedFeatures}
        fixed_part_df = pd.DataFrame([fixed_values])

        variable_values = [range(len(self.valuesPerFeature[feature])) for feature in variableFeatures] #This is because the LabelEncoder() encodes the values from [0: n_values]
        variable_combinations_df = pd.DataFrame(itertools.product(*variable_values), columns=variableFeatures)

        fixed_part_repeated = pd.concat([fixed_part_df] * len(variable_combinations_df), ignore_index=True)

        # Step 4: Concatenate fixed and variable parts and reorder columns
        consistentInstances = pd.concat([fixed_part_repeated, variable_combinations_df], axis=1)
        consistentInstances = consistentInstances[self.instance.index]  # Ensure column order matches the original instance

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
        
        inference = self.featureDistributions.query(variables=variablesToEstimate, evidence=priorEvidence)
        
        return inference.get_value(**{var : decodedInstance[var] for var in variablesToEstimate})

    def decodeInstance(self, instance : pd.Series) -> pd.Series:
        decodedInstance = instance.copy()
        for feature in instance.keys():
            decodedInstance[feature] = self.valuesPerFeature[feature][instance[feature]]
        return decodedInstance
    
    #TODO: Finish this so that I can compare the mean predictions of the two approaches. I need to adapt the old mean prediction output to the new one

    def assertAllMeanPredictionsAreSimilar(self, instance : pd.Series) -> list[float]: 

        features = [feature for feature in self.valuesPerFeature.keys() if feature in self.dag.nodes]
        for feature in features:
            for equivalenceClass in equivalenceClassesFor(self.dag, feature):
                classFeaturesOrder = equivalenceClass[0]
                self.assertMeanPredictionsAreSimilar(classFeaturesOrder, feature, instance, True)

    def assertMeanPredictionsAreSimilar(self, classFeaturesOrder : List[str], feature : str, instance : pd.Series, featureFixed : bool):
        featureIndex = classFeaturesOrder.index(feature) + int(featureFixed)
        fixedFeatures = classFeaturesOrder[:featureIndex]
        variableFeatures = classFeaturesOrder[featureIndex:]

        meanPredAlgorithm = self.meanPredictionForEquivalenceClass(fixedFeatures, feature, instance)
        oldMeanPredAlgorithm = self.naiveMeanPredictionForEquivalenceClass(fixedFeatures, variableFeatures, instance)
        assert np.allclose(meanPredAlgorithm, oldMeanPredAlgorithm, atol=1.e-2), f"The mean prediction for the new algorithm is {meanPredAlgorithm} and the mean prediction for the old algorithm is {oldMeanPredAlgorithm}"

class ApproximateASV(ASV):

    def __init__(self, bayesianNetwork : BayesianNetwork, model : DecisionTreeClassifier, featureDistributions : VariableElimination, valuesPerFeature : dict[str, list], featureToPredict : str, predictionFunction : str = 'Exact', instance : pd.Series = None, numTopologicalOrders : int = 1000):
        super().__init__(bayesianNetwork, model, featureDistributions, valuesPerFeature, featureToPredict, predictionFunction, instance)
        self.numberOfToposorts = numTopologicalOrders

    def equivalenceClasses(self, feature):
        topologoicalOrders = randomTopoSorts(self.dag, self.numberOfToposorts)
        return naiveEquivalenceClassesSizes(topologoicalOrders, feature, TopoSortHasher(self.dag, feature)).values()
    
    def asssertToposortsMatches(self, totalTopologicalOrders, allTopologicalOrders):
        assert totalTopologicalOrders == self.numberOfToposorts, f"The total number of topological orders for the equivalence classes is {totalTopologicalOrders} and the total number of topological orders generated for the graph is {self.numberOfToposorts}"
