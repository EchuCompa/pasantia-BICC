"""Perfect sampling of linear extensions using Huber's non-Markovian CFTP.

M. Huber, "Fast perfect sampling from linear extensions", Discrete Mathematics
306 (2006), sections 3 and 4: https://doi.org/10.1016/j.disc.2006.01.003
"""

import operator
import random
from typing import Optional

import networkx as nx

from asvFormula import Node


class _BoundingChain:
    def __init__(self, successors: list[set[int]]):
        self.successors = successors
        self.slots: list[Optional[int]] = [None] * len(successors)
        self.slots[-1] = 0
        self.active = 1

    def step(self, position: int, coin: int, order: Optional[list[int]] = None) -> None:
        left, right = self.slots[position:position + 2]
        unrelated = left is not None and right is not None and right not in self.successors[left]

        if order is not None:
            # Huber's coupling flips the coin when the right bracket's item is
            # already on the left; using the same coin for all states is invalid.
            actual_coin = coin
            if right is not None and order[position] == right and (left is None or unrelated):
                actual_coin = 1 - coin
            first, second = order[position:position + 2]
            if actual_coin and second not in self.successors[first]:
                order[position], order[position + 1] = second, first

        if coin:
            if unrelated:
                self.slots[position], self.slots[position + 1] = right, left
            elif left is not None and right is None:
                self.slots[position], self.slots[position + 1] = None, left
            elif left is None and right is not None:
                self.slots[position], self.slots[position + 1] = right, None

        if self.active < len(self.slots) and self.slots[-1] is None:
            self.slots[-1] = self.active
            self.active += 1

    def coalesced(self) -> bool:
        return self.active == len(self.slots)

    def result(self) -> list[int]:
        if not self.coalesced():
            raise ValueError("The bounding chain has not coalesced")
        # All slots contain distinct labels once every label is active.
        return [label for label in self.slots if label is not None]


def huberTopoSort(dag: nx.DiGraph) -> list[Node]:
    """Return one exactly uniform topological order of a DAG.

    Exactness follows Huber's algorithm under ideal independent random bits;
    Python's pseudorandom generator supplies the bits in this implementation.
    """
    return huberTopoSorts(dag, 1)[0]


def huberTopoSorts(dag: nx.DiGraph, numTopoSorts: int) -> list[list[Node]]:
    """Return independently generated perfect samples, without a mixing cutoff."""
    numTopoSorts = operator.index(numTopoSorts)
    if numTopoSorts < 0:
        raise ValueError("numTopoSorts must be nonnegative")
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("The graph must be a directed acyclic graph")

    nodes = list(nx.topological_sort(dag))
    if len(nodes) <= 1:
        return [nodes.copy() for _ in range(numTopoSorts)]

    indices = {node: index for index, node in enumerate(nodes)}
    closure = nx.transitive_closure_dag(dag)
    successors = [{indices[node] for node in closure.successors(item)} for item in nodes]
    return [[nodes[index] for index in _sample(successors)] for _ in range(numTopoSorts)]


def _sample(successors: list[set[int]]) -> list[int]:
    n = len(successors)
    blocks: list[list[tuple[int, int]]] = []
    length = 1

    # Each recursive CFTP call uses fresh randomness; retain failed blocks
    # so their same random choices can be replayed after the deeper call.
    while True:
        bound = _BoundingChain(successors)
        block = []
        for _ in range(length):
            position, coin = random.randrange(n - 1), random.getrandbits(1)
            block.append((position, coin))
            bound.step(position, coin)
        if bound.coalesced():
            order = bound.result()
            break
        blocks.append(block)
        length *= 2

    for block in reversed(blocks):
        bound = _BoundingChain(successors)
        for position, coin in block:
            bound.step(position, coin, order)
    return order
