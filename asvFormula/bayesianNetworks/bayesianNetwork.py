import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from pgmpy.inference import VariableElimination
import shap
from classesSizes.digraph import *
from pgmpy.models import BayesianNetwork


def obtainNetworkXTreeStructure(decisionTree : tree.DecisionTreeClassifier, featureNames : list[str]) -> nx.DiGraph:
    # Extract the necessary details from the tree
    G = nx.DiGraph()
    children_left = decisionTree.tree_.children_left
    children_right = decisionTree.tree_.children_right
    feature = decisionTree.tree_.feature
    threshold = decisionTree.tree_.threshold
    values = decisionTree.tree_.value

    # Stack to keep track of nodes to visit
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    # Assuming you have a list of feature names

    
    while len(stack) > 0:
        node_id, depth = stack.pop()
        # Modify the label to show the actual feature name, we asume that feature names has the same order as the original dataset
        G.add_node(node_id, label=f"{featureNames[feature[node_id]]}")

        # Check if it's a split node (non-leaf)
        is_split_node = children_left[node_id] != children_right[node_id]
        
        if is_split_node:
            # Add edges to left and right children
            G.add_edge(node_id, children_left[node_id], label=f"X[{feature[node_id]}] <= {threshold[node_id]:.2f}")
            G.add_edge(node_id, children_right[node_id], label=f"X[{feature[node_id]}] > {threshold[node_id]:.2f}")
            
            # Add children nodes to the stack
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            # Leaf node, add node with its value
            G.add_node(node_id, label=f"Leaf {node_id}: {np.around(values[node_id], 3)}")
    return G

def drawDecisionTree(decisionTree : nx.DiGraph):
    plt.figure(figsize=(12, 8))
    pos = nx.nx_agraph.graphviz_layout(decisionTree, prog="dot") #To have a tree layout
    nx.draw(decisionTree, pos, with_labels=True, labels={n: decisionTree.nodes[n].get('label', str(n)) for n in decisionTree.nodes()})
    nx.draw_networkx_edge_labels(decisionTree, pos, edge_labels=nx.get_edge_attributes(decisionTree, 'label'))
    plt.show()

def decisionTreeFromDataset(dataset : pd.DataFrame, target_feature, maximum_depth, rand_state=42):
    X = dataset.drop(target_feature, axis=1)
    y = dataset[target_feature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    rf_model = tree.DecisionTreeClassifier(max_depth=maximum_depth, random_state=rand_state)
    # These trees will always be binary ones, that's the implementation is sklearn.
    rf_model.fit(X_train, y_train)
    print(f'The model accuracy is : {rf_model.score(X_test, y_test)}')
    
    return rf_model

def datasetFromBayesianNetwork(model, n):
    return model.simulate(n_samples=n)

# TODO: Add the possibility to have a different number of values for each feature, with a variable expectedValue. 

def meanForDTinBN(dtClassifer : DecisionTreeClassifier, bayesianNetwork : VariableElimination, valuesPerFeature : dict[str, list], variableToPredict : str) -> float:
    assert variableToPredict in valuesPerFeature.keys(), "The variable to predict must be in the valuesPerFeature dictionary"

    featureNames = list(valuesPerFeature.keys())
    featureNames.remove(variableToPredict) # The variable to predict is not a feature, it was not present in the original dataset
    dtAsNetwork = obtainNetworkXTreeStructure(dtClassifer, featureNames)

    rootNodes = [node for node in dtAsNetwork.nodes if isRoot(node, dtAsNetwork)] 
    root = rootNodes[0]
    treePrediction = lambda x : list(x).index(max(x))  #The decision tree predicts the class with the highest probability
    predictionPerNode = {node : treePrediction(dtClassifer.tree_.value[node][0]) for node in dtAsNetwork.nodes if isLeaf(node, dtAsNetwork)}

    return meanForDTinBNWithEvidence(predictionPerNode, dtAsNetwork, root, bayesianNetwork, {}, valuesPerFeature, variableToPredict)


    

def meanForDTinBNWithEvidence(predictionPerNode : Dict[Any,float], decisionTreeGraph : nx.DiGraph, node, bayesianNetwork : VariableElimination, actualEvidence : dict[str, str], valuesPerFeature : dict[str, list], modelFeature : str) -> float:
    
    if isLeaf(node, decisionTreeGraph):
        modelPrediction = predictionPerNode[node]
        inferenceWithEvidence = bayesianNetwork.query(variables=[modelFeature], evidence = actualEvidence)
        probOfFeature = inferenceWithEvidence.get_value(**{modelFeature : modelPrediction})
        # TODO: Check if the model prediction is needed here. 
        return modelPrediction * probOfFeature
    
    feature = nodeLabel(node, decisionTreeGraph)
    featureValues = valuesPerFeature[feature]
    children = list(decisionTreeGraph.successors(node))
    
    actualEvidence[feature] = featureValues[0]
    leftMean = meanForDTinBNWithEvidence(predictionPerNode, decisionTreeGraph, children[0] , bayesianNetwork, actualEvidence, valuesPerFeature, modelFeature)

    actualEvidence[feature] = featureValues[1]
    rightMean = meanForDTinBNWithEvidence(predictionPerNode, decisionTreeGraph, children[1], bayesianNetwork, actualEvidence, valuesPerFeature, modelFeature)

    del actualEvidence[feature]
    return leftMean + rightMean

def showMeanPredictionOfModel(variableValue : str, variableToPredict : str, completeBNInference : VariableElimination, valuesPerFeature : dict[str,list], dtTreeClassifier : DecisionTreeClassifier):


    print(f"Predicting the value {variableValue} for the variable {variableToPredict}")

    meanValueForModel = meanForDTinBN(dtTreeClassifier, completeBNInference, valuesPerFeature, variableToPredict)
    print(f"Mean prediction value for the model: {meanValueForModel}")

    encodedValue,  = np.where(valuesPerFeature[variableToPredict] ==(variableValue))
    encodedValue = encodedValue[0]
    explainer = shap.TreeExplainer(dtTreeClassifier)
    meanValueForShap = explainer.expected_value[encodedValue]
    print(f"Estimated value for shap explainer: {meanValueForShap}")

    variableProbability = completeBNInference.query([variableToPredict])
    valueOrderInBN = variableProbability.state_names[variableToPredict].index(variableValue)
    print(f"Probability in the bayesian network: {variableProbability.values[valueOrderInBN]}")

def removeEdgeAndMarginalizeCPD(treeBNChild : BayesianNetwork, tail, head):
    treeBNChild.remove_edge(tail, head)
    treeBNChild.get_cpds(head).marginalize([tail], inplace=True)