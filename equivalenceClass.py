from typing import Any, List

class NodePosition:

    def __init__(self, node_name, appears_after_xi : bool) -> None:
        self.node_name = node_name
        self.appears_after_xi = appears_after_xi
        self.relative_position = 'After' if appears_after_xi else 'Before'

    def isBefore(self):
        return not self.appears_after_xi
    
    def nodeName(self):
        return self.node_name

    def __str__(self):
        return f"({self.node_name}, {self.relative_position})"

class EquivalenceClass:

    def __init__(self, unrelated_node_position : List[NodePosition], left_topo=1, right_topo=1):
        self.before = self.obtainNodesBefore(unrelated_node_position)
        self.after = self.obtainNodesAfter(unrelated_node_position)
        self.left_topo = left_topo
        self.right_topo = right_topo
        if len(unrelated_node_position) != 0:
            self.parent = list(unrelated_node_position)[0].nodeName()

    def topologicalSort(self, feature_node = 'Feature Node'):
        return self.before + [feature_node] + self.after

    def obtainNodesBefore(self, posi):
        positions = filter(lambda node_pos : node_pos.isBefore(), posi)
        return list(map(lambda p : p.nodeName(),  positions))
    
    def obtainNodesAfter(self, posi):
        positions = filter(lambda node_pos : not node_pos.isBefore(), posi)
        return list(map(lambda p : p.nodeName(),  positions))
    
    def allNodes(self):
        beforeAsPosition = list(map(lambda node : NodePosition(node, False), self.before))
        afterAsPosition = list(map(lambda node : NodePosition(node, True), self.after))
        return beforeAsPosition + afterAsPosition
        
    def nodes_after(self): #The nodes after x_i
        return self.after

    def nodes_before(self): #The nodes before x_i
        return self.before
    
    def num_nodes_before(self): 
        return len(self.before)
    
    def num_nodes_after(self):
        return len(self.after)
    
    def classSize(self): #Number of topological orders
        return self.left_topo * self.right_topo
    
    def __str__(self):
        
        return f"Equivalence Class (TopoSort={self.topologicalSort()}, Size={self.classSize()})"
    
    def addParent(self, parent : Any):
        self.before =  [parent] + self.before
        self.parent = parent

    def addParentToRigth(self, parent : Any):
        self.after =  [parent] + self.after
        self.parent = parent

    def addAncestors(self, nodes : List[Any]):
        self.before =  nodes + self.before

    def classParent(self):
        return self.parent

    def addLeftTopo(self, leftTopos : int):
        self.left_topo *= leftTopos
