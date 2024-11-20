import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from pgmpy.inference import VariableElimination
import shap
from asvFormula.digraph import *
from pgmpy.models import BayesianNetwork
from pathCondition import PathCondition
import math
from typing import Callable


class DecisionTreeDigraph(nx.DiGraph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def nodeAttribute(self , node, attribute : str):
        return self.nodes[node].get(attribute, str(node)) #TODO: Refactor this, remove the default case.

    def nodeLabel(self , node):
        return self.nodeAttribute(node, 'label')

    def nodeThreshold(self , node):
        return self.nodeAttribute(node, 'threshold')

    #It asumes that is a binary classifier, so it returns the probability for each class
    def nodeMeanPrediction(self , node):
        return self.nodeAttribute(node, 'classesProbabilities') 
    
    #It asumes that is a binary classifier, so it returns 0 or 1 for each class. It will return 1 for the class with the highest probability
    def nodePrediction(self , node):
        probs = self.nodeAttribute(node, 'classesProbabilities')
        maxValues = probs == max(probs)
        return maxValues.astype(int)

    def nodeFeature(self , node):
        return self.nodeAttribute(node, 'feature')

def obtainDecisionTreeDigraph(decisionTree : tree.DecisionTreeClassifier, featureNames : list[str]) -> DecisionTreeDigraph:
    # Extract the necessary details from the tree
    G = DecisionTreeDigraph()
    children_left = decisionTree.tree_.children_left
    children_right = decisionTree.tree_.children_right
    feature = decisionTree.tree_.feature
    thresholds = decisionTree.tree_.threshold
    values = decisionTree.tree_.value

    # Stack to keep track of nodes to visit
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    
    while len(stack) > 0:
        node_id, depth = stack.pop()
        classesValues = values[node_id][0]
        # Modify the label to show the actual feature name, we asume that feature names has the same order as the original dataset
        G.add_node(node_id, label=f"{featureNames[feature[node_id]]}, node: {node_id}", threshold = thresholds[node_id], classesProbabilities = classesValues, feature = featureNames[feature[node_id]])

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
            prediction = np.argmax(classesValues)
            G.add_node(node_id, label=f"{node_id}: {prediction}")
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

#It returns the mean value for each possible value of the feature. 
def meanPredictionForDTinBN(dtClassifer : DecisionTreeClassifier, bayesianNetwork : VariableElimination, valuesPerFeature : dict[str, list], variableToPredict : str) -> list[float]:
    assert variableToPredict in valuesPerFeature.keys(), "The variable to predict must be in the valuesPerFeature dictionary"

    featureNames = list(valuesPerFeature.keys())
    featureNames.remove(variableToPredict) # The variable to predict is not a feature, it was not present in the original dataset
    dtAsNetwork = obtainDecisionTreeDigraph(dtClassifer, featureNames)

    rootNodes = [node for node in dtAsNetwork.nodes if isRoot(node, dtAsNetwork)] 
    root = rootNodes[0]
    
    return meanPredictionForDTinBNWithEvidence(dtAsNetwork, root, bayesianNetwork, PathCondition(valuesPerFeature))

def meanPredictionForDTinBNWithEvidence(decisionTreeGraph : DecisionTreeDigraph, node, bayesianNetwork : VariableElimination, pathCondition : PathCondition, priorEvidence : dict[str, list] = None, nodePrediction : Callable = lambda n, tree : tree.nodeMeanPrediction(n)) -> list[float]:
    
    if pathCondition.doesNotMatchEvidence(priorEvidence):
        possibleOutputs = len(nodePrediction(node, decisionTreeGraph))
        return np.zeros(possibleOutputs)

    if isLeaf(node, decisionTreeGraph):

        pathCondition.removeEvidence(priorEvidence)
        pathVariables = pathCondition.getVariables()
        
        if pathVariables != []: #This means that the path we took is not included in the prior evidence
            probOfAllEvents = 0
            decodedEvidence = pathCondition.decodeEvidence(priorEvidence)
            inferenceWithEvidence = bayesianNetwork.query(variables=pathVariables, evidence = decodedEvidence)
        
            for event in pathCondition.allPossibleEvents():
                query = {pathVariables[i] : event[i] for i in range(len(pathVariables))}
                probOfAllEvents += inferenceWithEvidence.get_value(**query)
        else:
            probOfAllEvents = 1 #The evidence is the same as the path condition, so the probability is 1

        return  nodePrediction(node, decisionTreeGraph) * probOfAllEvents
    
    feature = decisionTreeGraph.nodeFeature(node)
    treshold = decisionTreeGraph.nodeThreshold(node) 
    children = list(decisionTreeGraph.successors(node))
    previousUpperLimit = pathCondition.getUpperLimit(feature)
    previousLowerLimit = pathCondition.getLowerLimit(feature)
    
    pathCondition.setVariableUpperLimit(feature, math.floor(treshold))
    leftMean = meanPredictionForDTinBNWithEvidence(decisionTreeGraph, children[0] , bayesianNetwork, pathCondition, priorEvidence, nodePrediction)
    pathCondition.setVariableUpperLimit(feature, previousUpperLimit) 

    pathCondition.setVariableLowerLimit(feature, math.ceil(treshold))
    rightMean = meanPredictionForDTinBNWithEvidence(decisionTreeGraph, children[1], bayesianNetwork, pathCondition, priorEvidence, nodePrediction)
    pathCondition.setVariableLowerLimit(feature, previousLowerLimit)

    return leftMean + rightMean

def showMeanPredictionOfModel(variableToPredict : str, completeBNInference : VariableElimination, valuesPerFeature : dict[str,list], dtTreeClassifier : DecisionTreeClassifier):

    print(f"Mean prediction of model for the variable {variableToPredict}")

    meanValueForModel = 0
    meanPredictions = meanPredictionForDTinBN(dtTreeClassifier, completeBNInference, valuesPerFeature, variableToPredict)
    for value, prob in enumerate(meanPredictions):
        meanValueForModel += value * prob
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