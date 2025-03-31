import shap
from asvFormula.ASV import ASV
import timeit
from asvFormula.bayesianNetworks.bayesianNetwork import *
from pgmpy.readwrite import BIFReader
from asvFormula.bayesianNetworks import networkSamplesPath
import os
import seaborn as sns
import matplotlib.pyplot as plt

def showMeanPredictionOfModel(variableToPredict : str, completeBNInference : VariableElimination, valuesPerFeature : dict[str,list], dtTreeClassifier : DecisionTreeClassifier, asv : ASV, numberOfVariableFeatures : int):

    allFeatures = list(valuesPerFeature.keys())
    allFeatures.remove(variableToPredict)

    numVariables = min(numberOfVariableFeatures, len(allFeatures))
    fixedFeatures = allFeatures[numVariables:]
    variableFeatures = allFeatures[:numVariables]

    print(f"Mean prediction of model for the variable {variableToPredict}")
    
    def meanPredictionForDT():
        global meanPredictions, meanProbsPredictions
        meanPredictions = meanPredictionForDTinBN(dtTreeClassifier, asv.featureDistributions, valuesPerFeature, variableToPredict, asv.instance, fixedFeatures)
        meanProbsPredictions = meanProbabilityPredictionForDTinBN(dtTreeClassifier, asv.featureDistributions, valuesPerFeature, variableToPredict, asv.instance, fixedFeatures)

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

def writeASVAndShapleyIntoFile(
    first_instance: pd.Series,
    dataSet: pd.DataFrame,
    dtTreeClassifier: DecisionTreeClassifier,
    asv,
    file: str,
    valuesPerFeature: dict[str, list],
    variableToPredict: str,
    seed: int,
    progress=False
):
    features = list(dataSet.columns)
    explainer = shap.TreeExplainer(dtTreeClassifier, dataSet)
    shap_values = explainer.shap_values(first_instance)
    predictionValues = valuesPerFeature[variableToPredict]

    # Check if the file exists to decide whether to write the header
    file_exists = os.path.isfile(file)

    with open(file, 'a') as f:
        if not file_exists:
            f.write(f"Feature,{variableToPredict} value,ASV,Shapley,Seed\n")

        for i, feature in enumerate(features):
            asvValue = asv.asvForFeature(feature, first_instance, showProgress=progress)
            shapleyValue = shap_values[i]
            for j in range(len(shapleyValue)):
                f.write(f"{feature},{predictionValues[j]},{asvValue[j]},{shapleyValue[j]},{seed}\n")

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

def dataframeFromCsv(path, seed=None):
    df = pd.read_csv(path)
    for column in ['ASV', 'Shapley']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    df = df[df['Seed'] == seed]
    df.name = path.split('/')[-1][:-4]
    return df

def multipleSeedsDataframes(dataPath, seeds) -> dict[int, pd.DataFrame]:
    dataframesDict = {}
    for seed in seeds:
        dataframesDict[seed] = dataframeFromCsv(dataPath, seed=seed)
    return dataframesDict

childFeatureOrder = ["Disease", "CardiacMixing", "CO2Report", "LungParench", "DuctFlow", "Grunting", "LungFlow", "LVHreport", "ChestXray", "BirthAsphyxia", "RUQO2", "Sick", "XrayReport", "LowerBodyO2", "HypoxiaInO2", "HypDistrib", "CO2", "GruntingReport", "LVH"]

def plotValuesFromDF(ax, df, valueToPlot, hueValue, seed, paletteValue='Set2'):
    bayesianNetwork = 'Cancer' if 'cancer' in df.name.lower() else 'Child'

    desired_order = ["Xray", "Pollution", "Dyspnoea", "Cancer"] if bayesianNetwork == 'Cancer' else childFeatureOrder
    sns.barplot(x="Feature", y=valueToPlot, hue=hueValue, data=df, palette=paletteValue, ax=ax, order=desired_order)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"{valueToPlot} Values, with seed: {seed} in {bayesianNetwork} Network")
    ax.set_ylabel(valueToPlot)
    ax.set_xlabel("Feature")
    if bayesianNetwork == 'Child':
        ax.tick_params(axis='x', rotation=45)
    ax.legend(title=hueValue)

def plotASVandShapFromDF(df, hueValue, seed, figPath=None, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(15, 6))
    plotValuesFromDF(axes[0], df, 'ASV', hueValue, seed)
    plotValuesFromDF(axes[1], df, 'Shapley', hueValue, seed)
    if figPath:
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(figPath)
        plt.show()