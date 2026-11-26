from collections.abc import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt

from asvFormula import Node


def asvFromTopologicalOrders(
    feature: Node,
    topologicalOrders: Iterable[Sequence[Node]],
    characteristicFunction: Callable[[frozenset[Node]], npt.ArrayLike],
) -> npt.NDArray[np.float64]:
    """Estimate an ASV from orders and a model/distribution-specific value oracle."""
    cache: dict[frozenset[Node], npt.NDArray[np.float64]] = {}
    total = None
    orderCount = 0
    expectedNodes = None
    expectedShape = None

    def valueFor(coalition: frozenset[Node]) -> npt.NDArray[np.float64]:
        nonlocal expectedShape
        if coalition not in cache:
            value = np.asarray(characteristicFunction(coalition), dtype=float)
            if expectedShape is None:
                expectedShape = value.shape
            elif value.shape != expectedShape:
                raise ValueError("The characteristic function must return a consistent shape")
            if not np.all(np.isfinite(value)):
                raise ValueError("The characteristic function must return finite values")
            cache[coalition] = value
        return cache[coalition]

    for order in topologicalOrders:
        nodes = frozenset(order)
        if len(nodes) != len(order):
            raise ValueError("An ordering cannot contain duplicate nodes")
        if expectedNodes is None:
            expectedNodes = nodes
        elif nodes != expectedNodes:
            raise ValueError("All orderings must contain the same nodes")
        try:
            featurePosition = order.index(feature)
        except ValueError as error:
            raise ValueError(f"Feature {feature!r} is missing from an ordering") from error

        beforeFeature = frozenset(order[:featurePosition])
        withFeature = beforeFeature | {feature}
        contribution = valueFor(withFeature) - valueFor(beforeFeature)
        if total is None:
            total = np.zeros_like(contribution, dtype=float)
        total += contribution
        orderCount += 1

    if orderCount == 0:
        raise ValueError("At least one topological ordering is required")
    return total / orderCount
