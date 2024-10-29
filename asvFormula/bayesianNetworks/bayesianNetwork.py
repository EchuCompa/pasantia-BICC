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
from pathCondition import PathCondition
import math


def obtainNetworkXTreeStructure(decisionTree : tree.DecisionTreeClassifier, featureNames : list[str]) -> nx.DiGraph:
    # Extract the necessary details from the tree
    G = nx.DiGraph()
    children_left = decisionTree.tree_.children_left
    children_right = decisionTree.tree_.children_right
    feature = decisionTree.tree_.feature
    thresholds = decisionTree.tree_.threshold
    values = decisionTree.tree_.value

    # Stack to keep track of nodes to visit
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    
    nodeMeanPrediction = lambda x : sum([i * x[i] for i in range(len(x))])
    
    while len(stack) > 0:
        node_id, depth = stack.pop()
        meanPredictionPerNode = nodeMeanPrediction(values[node_id][0])
        # Modify the label to show the actual feature name, we asume that feature names has the same order as the original dataset
        G.add_node(node_id, label=f"{featureNames[feature[node_id]]}, node: {node_id}", threshold = thresholds[node_id], meanPrediction = meanPredictionPerNode, feature = featureNames[feature[node_id]])

        # Check if it's a split node (non-leaf)
        is_split_node = children_left[node_id] != children_right[node_id]
        
        if is_split_node:
            # Add edges to left and right children
            G.add_edge(node_id, children_left[node_id], label=f"<= {thresholds[node_id]:.2f}")
            G.add_edge(node_id, children_right[node_id], label=f"> {thresholds[node_id]:.2f}")
            
            # Add children nodes to the stack
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            #G.add_node(node_id, label=f"Leaf {node_id}: {np.around(values[node_id], 3)}")
            # Obtains the prediction of the node instead of the probabilities
            G.add_node(node_id, label=f"{node_id}: {round(meanPredictionPerNode,2)}")
    return G

def drawDecisionTree(decisionTree : nx.DiGraph):
    num_nodes = len(decisionTree.nodes())
    dynamic_size = max(8, min(num_nodes * 0.5, 20))  # Scale factor can be adjusted

    plt.figure(figsize=(dynamic_size, dynamic_size * 0.9)) 
    pos = nx.nx_agraph.graphviz_layout(decisionTree, prog="dot") #To have a tree layout
    labelForNode = lambda node : decisionTree.nodes[node].get('label', str(node))
    nx.draw(decisionTree, pos, with_labels=True, labels={n: labelForNode(n) for n in decisionTree.nodes()})
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
    
    return meanForDTinBNWithEvidence(dtAsNetwork, root, bayesianNetwork, PathCondition(valuesPerFeature), variableToPredict)

def meanForDTinBNWithEvidence(decisionTreeGraph : nx.DiGraph, node, bayesianNetwork : VariableElimination, pathCondition : PathCondition, modelFeature : str, priorEvidence : dict[str, list] = None) -> float:
    
    if isLeaf(node, decisionTreeGraph):  
        pathVariables = pathCondition.getVariables()
        inferenceWithEvidence = bayesianNetwork.query(variables=pathVariables, evidence = priorEvidence)

        probOfAllEvents = 0
        for event in pathCondition.allPossibleEvents():
            query = {pathVariables[i] : event[i] for i in range(len(pathVariables))}
            probOfAllEvents += inferenceWithEvidence.get_value(**query)
        return nodeMeanPrediction(node, decisionTreeGraph) * probOfAllEvents
    
    feature = nodeFeature(node, decisionTreeGraph)
    treshold = nodeThreshold(node, decisionTreeGraph) 
    children = list(decisionTreeGraph.successors(node))
    
    pathCondition.setVariableUpperLimit(feature, math.floor(treshold))
    leftMean = meanForDTinBNWithEvidence(decisionTreeGraph, children[0] , bayesianNetwork, pathCondition, modelFeature)
    pathCondition.removeVariable(feature)

    pathCondition.setVariableLowerLimit(feature, math.ceil(treshold))
    rightMean = meanForDTinBNWithEvidence(decisionTreeGraph, children[1], bayesianNetwork, pathCondition, modelFeature)
    pathCondition.removeVariable(feature)

    return leftMean + rightMean

def showMeanPredictionOfModel(variableToPredict : str, completeBNInference : VariableElimination, valuesPerFeature : dict[str,list], dtTreeClassifier : DecisionTreeClassifier):


    print(f"Mean prediction of model for the variable {variableToPredict}")

    meanValueForModel = meanForDTinBN(dtTreeClassifier, completeBNInference, valuesPerFeature, variableToPredict)
    print(f"Mean prediction value for the model: {meanValueForModel}")

    meanValueForShap = 0
    explainer = shap.TreeExplainer(dtTreeClassifier)
    for value, prob in enumerate(explainer.expected_value):
        meanValueForShap += value * prob
    print(f"Estimated value for shap explainer: {meanValueForShap}")

    variableProbability = completeBNInference.query([variableToPredict])
    meanPrediction = 0
    for prediction, featureValue in enumerate(valuesPerFeature[variableToPredict]):
        meanPrediction += prediction * variableProbability.get_value(**{variableToPredict : featureValue})
    print(f"Mean prediction if we only use the bayesian network: {meanPrediction}")

def removeEdgeAndMarginalizeCPD(treeBNChild : BayesianNetwork, tail, head):
    treeBNChild.remove_edge(tail, head)
    treeBNChild.get_cpds(head).marginalize([tail], inplace=True)