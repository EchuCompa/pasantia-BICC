from sklearn.preprocessing import LabelEncoder
from pgmpy.models import BayesianNetwork
from asvFormula.bayesianNetworks.bayesianNetwork import *
import random

def encodeCategoricalColumns(dataset):
    encodingDict = {}
    le = LabelEncoder()
    encodedDataset = dataset.copy()
    categorical_columns = dataset.select_dtypes(include=['object', 'category', 'bool']).columns
    for columnName in categorical_columns:
        encodedDataset[columnName] = le.fit_transform(encodedDataset[columnName])
        encodingDict[columnName] =  le.classes_
    return encodingDict, encodedDataset


def initializeDataAndRemoveVariable(BNmodel : BayesianNetwork, variableToPredict : str, numberOfSamples : int, treeMaxDepth : int, seed = None):
    # Create a BNDatabaseGenerator object from the model
    np.random.seed(seed)
    random.seed(seed)
    dataFromBN = datasetFromBayesianNetwork(BNmodel, numberOfSamples, seed)

    BNmodel.remove_node(variableToPredict) # We remove the variable to predict from the BN model so that we won't have this information when we are going to predict it
    BNInference = VariableElimination(BNmodel)

    valuesPerFeature, encodedDataset = encodeCategoricalColumns(dataFromBN)
    dtTreeClassifier = decisionTreeFromDataset(encodedDataset, variableToPredict , treeMaxDepth, seed)

    featureColumns = list(dataFromBN.columns)
    featureColumns.remove(variableToPredict)  
    dtAsNetwork = obtainDecisionTreeDigraph(dtTreeClassifier, featureColumns)

    return BNInference, valuesPerFeature, encodedDataset, dtTreeClassifier, dtAsNetwork