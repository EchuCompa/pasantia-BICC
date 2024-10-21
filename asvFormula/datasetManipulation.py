from sklearn.preprocessing import LabelEncoder
from pgmpy.models import BayesianNetwork
from asvFormula.bayesianNetworks.bayesianNetwork import *

def encodeCategoricalColumns(dataset):
    encodingDict = {}
    le = LabelEncoder()
    encodedDataset = dataset.copy()
    categorical_columns = dataset.select_dtypes(include=['object', 'category', 'bool']).columns
    for columnName in categorical_columns:
        encodedDataset[columnName] = le.fit_transform(encodedDataset[columnName])
        encodingDict[columnName] =  le.classes_
    return encodingDict, encodedDataset


def initializeData(BNmodel : BayesianNetwork, variableToPredict : str, numberOfSamples : int, treeMaxDepth : int):
    # Create a BNDatabaseGenerator object from the model
    dataFromBN = datasetFromBayesianNetwork(BNmodel, numberOfSamples)

    BNmodel.remove_node(variableToPredict) # We remove the variable to predict from the BN model so that we won't have this information when we are going to predict it
    BNInference = VariableElimination(BNmodel)


    valuesPerFeature, encodedDataset = encodeCategoricalColumns(dataFromBN)
    dtTreeClassifier = decisionTreeFromDataset(encodedDataset, variableToPredict , treeMaxDepth)

    featureColumns = list(dataFromBN.columns)
    featureColumns.remove(variableToPredict)  
    dtAsNetwork = obtainNetworkXTreeStructure(dtTreeClassifier, featureColumns)

    return BNInference, valuesPerFeature, encodedDataset, dtTreeClassifier, dtAsNetwork