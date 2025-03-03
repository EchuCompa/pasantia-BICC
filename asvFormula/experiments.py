import shap
from exactASV import ASV
import timeit
from asvFormula.bayesianNetworks.bayesianNetwork import *
from pgmpy.readwrite import BIFReader
from asvFormula.bayesianNetworks import networkSamplesPath

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


#I want to the same as the previous function but saving the results in a file

def writeASVAndShapleyIntoFile(first_instance : pd.Series, dataSet : pd.DataFrame, dtTreeClassifier : DecisionTreeClassifier, asv : ASV, file : str, valuesPerFeature : dict[str,list], variableToPredict : str, progress = False):
    # I want it in a CSV format as a table. With the features as the columns and the values as the rows
    features = list(dataSet.columns)
    explainer = shap.TreeExplainer(dtTreeClassifier, dataSet)
    shap_values = explainer.shap_values(first_instance)
    predictionValues = valuesPerFeature[variableToPredict]
    with open(file, 'w') as f:
        f.write(f"Feature,{variableToPredict} value,ASV,Shapley\n")
        sumOfAsv = np.zeros(dtTreeClassifier.n_classes_)
        for i, feature in enumerate(features):
            asvValue = asv.asvForFeature(feature, first_instance, showProgress=progress)
            sumOfAsv += asvValue
            shapleyValue = shap_values[i]
            for i in range(len(shapleyValue)):
                f.write(f"{feature},{predictionValues[i]},{asvValue[i]},{shapleyValue[i]}\n")
        f.write(f"Sum,{sumOfAsv},{sum(shap_values)}\n")

def cancerNetworkConfig():
    cancerNetworkPath = networkSamplesPath + "/cancer.bif"
    BNmodel = BIFReader(cancerNetworkPath).get_model()
    variableToPredict = "Smoker"
    numberOfSamples = 600
    treeMaxDepth = 3
    return BNmodel,variableToPredict,numberOfSamples,treeMaxDepth


def childNetworkConfig():
    childNetworkPath = networkSamplesPath + "/child.bif"
    treeBNChild = childBNAsTree(BIFReader(childNetworkPath).get_model())
    variableToPredict = "Age"
    numberOfSamples = 10000
    treeMaxDepth = 9
    return treeBNChild,variableToPredict,numberOfSamples,treeMaxDepth