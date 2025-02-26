import shap
from exactASV import ASV
import timeit
from asvFormula.bayesianNetworks.bayesianNetwork import *

def showMeanPredictionOfModel(variableToPredict : str, completeBNInference : VariableElimination, valuesPerFeature : dict[str,list], dtTreeClassifier : DecisionTreeClassifier, asv : ASV, numberOfVariableFeatures : int):

    allFeatures = list(valuesPerFeature.keys())
    allFeatures.remove(variableToPredict)

    numVariables = min(numberOfVariableFeatures, len(allFeatures))
    fixedFeatures = allFeatures[numVariables:]
    variableFeatures = allFeatures[:numVariables]

    print(f"Mean prediction of model for the variable {variableToPredict}")
    
    def meanPredictionForDT():
        global meanPredictions, meanProbsPredictions
        meanPredictions = meanPredictionForDTinBN(dtTreeClassifier, completeBNInference, valuesPerFeature, variableToPredict, asv.instance, fixedFeatures)
        meanProbsPredictions = meanProbabilityPredictionForDTinBN(dtTreeClassifier, completeBNInference, valuesPerFeature, variableToPredict, asv.instance, fixedFeatures)

    meanPredDTTime = timeit.timeit(meanPredictionForDT, number=1)
    
    print(f"Mean prediction value for the decision tree: {meanPredictions}, it took {meanPredDTTime/2} seconds")
    print(f"Mean prediction value for the probabilities of the decision tree: {meanProbsPredictions}, it took {meanPredDTTime/2} seconds")

    def meanPredictionForData():
        global meanPredictions
        meanPredictions = asv.naiveMeanPredictionForEquivalenceClass(fixedFeatures, variableFeatures)

    meanPredDataTime = timeit.timeit(meanPredictionForData, number=1)
    print(f"Mean prediction value for possible values of the dataset: {meanPredictions}, it took {meanPredDataTime} seconds")


    explainer = shap.TreeExplainer(dtTreeClassifier)
    print(f"Estimated value for shap explainer: {explainer.expected_value}")

    variableProbability = completeBNInference.query([variableToPredict])
    meanPrediction = []
    for _, featureValue in enumerate(valuesPerFeature[variableToPredict]):
        meanPrediction.append(variableProbability.get_value(**{variableToPredict : featureValue}))
    print(f"Probabilities of the variable in the bayesian network: {meanPrediction}")
    
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