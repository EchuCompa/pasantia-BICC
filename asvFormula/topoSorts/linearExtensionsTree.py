import math

from asvFormula import Node
from asvFormula.digraph import nx
from asvFormula.topoSorts.utils import multinomial_coefficient


def linearExtensionPositionCounts(tree: nx.DiGraph, node: Node) -> list[int]:
    if node not in tree:
        raise ValueError(f"Node {node!r} is not in the graph")
    if not nx.is_directed_acyclic_graph(tree) or not nx.is_forest(tree.to_undirected()):
        raise ValueError("The graph must be a directed forest")
    if not nx.is_connected(tree.to_undirected()):
        raise ValueError("The graph must be a connected tree")
    return _treePositionCounts(tree, node)


def allPolyforestTopoSortsAtkinson(polyforest: nx.DiGraph) -> int:
    if len(polyforest) == 0:
        return 1
    if not nx.is_directed_acyclic_graph(polyforest) or not nx.is_forest(polyforest.to_undirected()):
        raise ValueError("The graph must be a directed forest")

    sizes = []
    counts = []
    for component in nx.connected_components(polyforest.to_undirected()):
        tree = polyforest.subgraph(component).copy()
        spectrum = _treePositionCounts(tree, next(iter(component)))
        sizes.append(len(tree))
        counts.append(sum(spectrum))

    return multinomial_coefficient(tuple(sizes)) * math.prod(counts)


def allPolyforestTopoSortsAtkinsonWithVisited(
    polyforest: nx.DiGraph, visited: dict[Node, bool]
) -> int:
    remaining = [node for node in polyforest if not visited[node]]
    return allPolyforestTopoSortsAtkinson(polyforest.subgraph(remaining).copy())


def _treePositionCounts(tree: nx.DiGraph, node: Node) -> list[int]:
    binomials = [[1]]
    for size in range(1, len(tree) + 1):
        previous = binomials[-1]
        binomials.append(
            [1] + [previous[k - 1] + previous[k] for k in range(1, size)] + [1]
        )
    return _treePositionCountsWithBinomials(tree, node, binomials)


def _treePositionCountsWithBinomials(
    tree: nx.DiGraph, node: Node, binomials: list[list[int]]
) -> list[int]:
    if tree.number_of_edges() == 0:
        return [1]

    if tree.out_degree(node):
        lower, upper = node, next(iter(tree.successors(node)))
    else:
        lower, upper = next(iter(tree.predecessors(node))), node

    split = tree.copy()
    split.remove_edge(lower, upper)
    components = list(nx.connected_components(split.to_undirected()))
    first = split.subgraph(next(component for component in components if lower in component)).copy()
    second = split.subgraph(next(component for component in components if upper in component)).copy()

    lower_spectrum = _treePositionCountsWithBinomials(first, lower, binomials)
    upper_spectrum = _treePositionCountsWithBinomials(second, upper, binomials)
    return _gluePositionCounts(lower_spectrum, upper_spectrum, lower == node, binomials)


def _gluePositionCounts(
    lower_spectrum: list[int], upper_spectrum: list[int], lower_is_target: bool,
    binomials: list[list[int]],
) -> list[int]:
    if not lower_is_target:
        lower_spectrum, upper_spectrum = upper_spectrum, lower_spectrum

    lower_size = len(lower_spectrum)
    upper_size = len(upper_spectrum)
    prefix = [0]
    for count in upper_spectrum:
        prefix.append(prefix[-1] + count)
    suffix = [0] * (upper_size + 1)
    for index in range(upper_size - 1, -1, -1):
        suffix[index] = suffix[index + 1] + upper_spectrum[index]

    result = [0] * (lower_size + upper_size)
    for position in range(1, lower_size + upper_size + 1):
        lower_position_min = max(1, position - upper_size)
        lower_position_max = min(lower_size, position)
        for lower_position in range(lower_position_min, lower_position_max + 1):
            upper_before_target = position - lower_position
            compatible_upper_extensions = (
                suffix[upper_before_target]
                if lower_is_target
                else prefix[upper_before_target]
            )
            result[position - 1] += (
                lower_spectrum[lower_position - 1]
                * binomials[position - 1][lower_position - 1]
                * binomials[lower_size + upper_size - position][lower_size - lower_position]
                * compatible_upper_extensions
            )
    return result
