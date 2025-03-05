import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from asvFormula.digraph import *
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

    #It returns the probability for each class
    def nodeProbabilityPrediction(self , node):
        return self.nodeAttribute(node, 'classesProbabilities') 
    
    #It asumes that is a binary classifier, so it returns 0 or 1 for each class. It will return 1 for the class with the highest probability
    def nodePrediction(self , node):
        probs = self.nodeAttribute(node, 'classesProbabilities')
        maxValues = probs == max(probs)
        return maxValues.astype(int)

    def nodeFeature(self , node):
        return self.nodeAttribute(node, 'feature')

def obtainDecisionTreeDigraph(decisionTree : tree.DecisionTreeClassifier, featureNames : list[str], meanPrediction = False) -> DecisionTreeDigraph:
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
        label = f"{featureNames[feature[node_id]]}" #Other option: f"{featureNames[feature[node_id]]}, node: {node_id}"
        G.add_node(node_id, label=label, threshold = thresholds[node_id], classesProbabilities = classesValues, feature = featureNames[feature[node_id]])

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
            # Obtains the exact prediction of the node instead of the probabilities
            prediction = np.around(values[node_id], 3) if meanPrediction else np.argmax(classesValues)
            G.add_node(node_id, label=f"Pred: {prediction}") #Other option: {node_id}: {prediction}
    return G

def drawDecisionTree(decisionTree: nx.DiGraph, filePath : str = None):
    num_nodes = len(decisionTree.nodes())
    dynamic_size = max(8, min(num_nodes * 0.5, 20))

    plt.figure(figsize=(dynamic_size, dynamic_size * 0.9))
    
    pos = nx.nx_agraph.graphviz_layout(decisionTree, prog="dot")

    label_offset = 10
    adjusted_pos = {node: (x, y + label_offset) for node, (x, y) in pos.items()}

    node_labels = {n: decisionTree.nodes[n].get('label', str(n)) for n in decisionTree.nodes()}

    nx.draw(decisionTree, pos, with_labels=False, node_color="lightblue", edge_color="gray", node_size=800)
    nx.draw_networkx_labels(decisionTree, adjusted_pos, labels=node_labels, font_size=12, font_color="black")
    nx.draw_networkx_edge_labels(decisionTree, pos, edge_labels=nx.get_edge_attributes(decisionTree, 'label'))

    if filePath:
        plt.savefig(filePath) 

    plt.show()

def decisionTreeFromDataset(dataset : pd.DataFrame, target_feature, maximum_depth, seed) -> tree.DecisionTreeClassifier:
    X = dataset.drop(target_feature, axis=1)
    y = dataset[target_feature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    rf_model = tree.DecisionTreeClassifier(max_depth=maximum_depth, random_state=seed)
    # These trees will always be binary ones, that's the implementation is sklearn.
    rf_model.fit(X_train, y_train)
    print(f'The model accuracy is : {rf_model.score(X_test, y_test)}')
    
    return rf_model

def datasetFromBayesianNetwork(model : BayesianNetwork, n, seed : int):
    return model.simulate(n_samples=n, seed = seed)

#It returns the mean value for each possible value of the feature. 
def meanPredictionForDTinBN(dtClassifer : DecisionTreeClassifier, bayesianNetwork : VariableElimination, valuesPerFeature : dict[str, list], variableToPredict : str, instance : pd.Series, fixedFeatures : list[str]) -> list[float]:
    assert variableToPredict in valuesPerFeature.keys(), "The variable to predict must be in the valuesPerFeature dictionary"

    dtAsNetwork = convertDecisionTreeToGraph(dtClassifer, valuesPerFeature, variableToPredict)
    
    return meanPredictionForDTinBNWithEvidence(dtAsNetwork, rootNode(dtAsNetwork), bayesianNetwork, PathCondition(valuesPerFeature), priorEvidence=priorEvidenceFrom(instance, fixedFeatures))

def meanProbabilityPredictionForDTinBN(dtClassifer : DecisionTreeClassifier, bayesianNetwork : VariableElimination, valuesPerFeature : dict[str, list], variableToPredict : str, instance : pd.Series, fixedFeatures : list[str]) -> list[float]:
    assert variableToPredict in valuesPerFeature.keys(), "The variable to predict must be in the valuesPerFeature dictionary"

    dtAsNetwork = convertDecisionTreeToGraph(dtClassifer, valuesPerFeature, variableToPredict)
    
    return meanPredictionForDTinBNWithEvidence(dtAsNetwork, rootNode(dtAsNetwork), bayesianNetwork, PathCondition(valuesPerFeature), priorEvidenceFrom(instance, fixedFeatures), lambda n, tree : tree.nodeProbabilityPrediction(n))

def priorEvidenceFrom(instance, fixedFeatures):
    return {featureEvidence : instance[featureEvidence] for featureEvidence in fixedFeatures}

def convertDecisionTreeToGraph(dtClassifer, valuesPerFeature, variableToPredict):
    featureNames = list(valuesPerFeature.keys())
    featureNames.remove(variableToPredict) # The variable to predict is not a feature, it was not present in the original dataset
    dtAsNetwork = obtainDecisionTreeDigraph(dtClassifer, featureNames)
    return dtAsNetwork

def meanPredictionForDTinBNWithEvidence(decisionTreeGraph : DecisionTreeDigraph, node, bayesianNetwork : VariableElimination, pathCondition : PathCondition, priorEvidence : dict[str, list] = None, nodePrediction : Callable = lambda n, tree : tree.nodePrediction(n)) -> list[float]:
    
    if pathCondition.doesNotMatchEvidence(priorEvidence): #It would be more efficient to only do this check when you add the variable to the path condition, but it's more legible this way (and the perfomance loss is not significant)
        possibleOutputs = len(nodePrediction(node, decisionTreeGraph))
        return np.zeros(possibleOutputs)

    if isLeaf(node, decisionTreeGraph):

        return  leafNodePrediction(decisionTreeGraph, node, bayesianNetwork, pathCondition, priorEvidence, nodePrediction)
    
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

def leafNodePrediction(decisionTreeGraph, node, bayesianNetwork, pathCondition, priorEvidence, nodePrediction) -> list[float]:
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
        
    nodePred = nodePrediction(node, decisionTreeGraph) * probOfAllEvents
    return nodePred

def bayesianNetworkToDigraph(bayesianNetwork : BayesianNetwork) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(bayesianNetwork.nodes())
    G.add_edges_from(bayesianNetwork.edges())
    return G

def removeEdgeAndMarginalizeCPD(treeBNChild : BayesianNetwork, tail, head):
    treeBNChild.remove_edge(tail, head)
    treeBNChild.get_cpds(head).marginalize([tail], inplace=True)

def childBNAsTree(treeBNChild):
    #I remove this edges so that it is a tree and we can work with it
    removeEdgeAndMarginalizeCPD(treeBNChild, 'LungParench', 'Grunting')
    removeEdgeAndMarginalizeCPD(treeBNChild, 'LungParench', 'HypoxiaInO2')
    removeEdgeAndMarginalizeCPD(treeBNChild, 'HypoxiaInO2', 'LowerBodyO2')
    removeEdgeAndMarginalizeCPD(treeBNChild, 'CardiacMixing', 'HypDistrib')
    removeEdgeAndMarginalizeCPD(treeBNChild, 'Sick', 'Age')
    removeEdgeAndMarginalizeCPD(treeBNChild, 'LungFlow', 'ChestXray')

    assert hasUnderlyingTree(treeBNChild)
    return treeBNChild