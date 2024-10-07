import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pgmpy
from pgmpy.inference import VariableElimination
from classesSizes.digraph import *


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
        # Modify the label to show the actual feature name
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

def meanForDTinBN(dtClassifer : DecisionTreeClassifier, bayesianNetwork : VariableElimination, valuesPerFeature : dict[str, list], variableToPredict : str, expectedValue : str, featureNames : list[str]) -> float:
    dtAsNetwork = obtainNetworkXTreeStructure(dtClassifer, featureNames)

    rootNodes = [node for node in dtAsNetwork.nodes if isRoot(node, dtAsNetwork)] 
    root = rootNodes[0]
    probPerLeafNode = {node : dtClassifer.tree_.value[node][0][0] for node in dtAsNetwork.nodes if isLeaf(node, dtAsNetwork)}

    return meanForDTinBN(probPerLeafNode, dtAsNetwork, root, bayesianNetwork, {}, valuesPerFeature, variableToPredict, expectedValue)


    

def meanForDTinBN(probPerLeafNode : Dict[Any,float], tree : nx.DiGraph, node, bayesianNetwork : VariableElimination, actualEvidence : dict[str, str], valuesPerFeature : dict[str, list], variableToPredict : str, expectedValue : str) -> float:
    feature = nodeLabel(node, tree)
    probs = bayesianNetwork.query(variables=[feature], evidence = actualEvidence)

    if isLeaf(node, tree):
        return probPerLeafNode[node]
    if variableToPredict == feature:
        return probs.get_value(**{variableToPredict : expectedValue})
    
    featureValues = valuesPerFeature[feature]
    children = list(tree.successors(node))
    
    actualEvidence[feature] = featureValues[0]
    leftMean = meanForDTinBN(probPerLeafNode, tree, children[0] , bayesianNetwork, actualEvidence, valuesPerFeature, variableToPredict)
    leftProbability = probs.get_value(**{feature : featureValues[0]})

    actualEvidence[feature] = featureValues[1]
    rightMean = meanForDTinBN(probPerLeafNode, tree, children[1], bayesianNetwork, actualEvidence, valuesPerFeature, variableToPredict)
    rightProbability = probs.get_value(**{feature : featureValues[1]})

    return leftMean * leftProbability + rightMean * rightProbability