"""Benchmark standalone topological-sort counting and end-to-end sampling."""

import argparse
import csv
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asvFormula.topoSorts.linearExtensionsTree import allPolyforestTopoSortsAtkinson
from asvFormula.topoSorts.randomTopoSortsGeneration import (
    randomTopoSorts,
    randomTopoSortsLegacy,
)
from asvFormula.topoSorts.topoSortsCalc import allPolyTopoSorts, allPossibleOrders


COUNT_SIZES_BY_GRAPH = {
    "path": (6, 8, 10, 12, 14, 16, 18),
    "binary_tree": (6, 8, 10, 12, 14, 16, 18),
    # Keep stars small because the legacy counter permutes all neighbors.
    "star": (6, 8, 10),
}
SAMPLING_NODES_BY_GRAPH = {"path": 20, "binary_tree": 14, "star": 10}
SAMPLE_SIZES = (100, 1000, 2000)
RANDOM_SEED = 20261126
OUTPUT_CSV = REPO_ROOT / "results" / "toposortBenchmarks" / "count_sampling_timings.csv"
FIGURE_DIR = REPO_ROOT / "latex-tesis" / "JAIO_extension" / "img"


def make_graphs(node_count):
    path = nx.DiGraph()
    path.add_nodes_from(range(node_count))
    path.add_edges_from((node, node + 1) for node in range(node_count - 1))

    binary_tree = nx.DiGraph()
    binary_tree.add_nodes_from(range(node_count))
    binary_tree.add_edges_from(
        ((node - 1) // 2, node) for node in range(1, node_count)
    )

    star = nx.DiGraph()
    star.add_nodes_from(range(node_count))
    star.add_edges_from((0, node) for node in range(1, node_count))

    return {"path": path, "binary_tree": binary_tree, "star": star}


def clear_count_cache():
    allPossibleOrders.cache_clear()


def measure(call, repeats, before_each=None, after_each=None):
    timings = []
    result = None
    for repeat in range(repeats):
        clear_count_cache()
        if before_each is not None:
            before_each(repeat)
        start = time.perf_counter()
        result = call()
        timings.append(time.perf_counter() - start)
        if after_each is not None:
            after_each(result)
    return timings, result


def is_valid_topological_order(graph, order):
    if len(order) != graph.number_of_nodes() or set(order) != set(graph.nodes()):
        return False
    positions = {node: index for index, node in enumerate(order)}
    return all(positions[source] < positions[target] for source, target in graph.edges())


def benchmark(repeats):
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "networkx_version": nx.__version__,
        "matplotlib_version": matplotlib.__version__,
        "scipy_version": scipy.__version__,
        "seed": RANDOM_SEED,
    }
    rows = []

    for graph_name, node_counts in COUNT_SIZES_BY_GRAPH.items():
        for node_count in node_counts:
            graph = make_graphs(node_count)[graph_name]
            max_degree = max(dict(graph.to_undirected().degree()).values(), default=0)
            legacy_count = allPolyTopoSorts(graph)
            atkinson_count = allPolyforestTopoSortsAtkinson(graph)
            if legacy_count != atkinson_count:
                raise RuntimeError(
                    f"Count mismatch for {graph_name} with {node_count} nodes"
                )

            count_methods = {
                "topoSortsCalc": lambda graph=graph: allPolyTopoSorts(graph),
                "Atkinson": lambda graph=graph: allPolyforestTopoSortsAtkinson(graph),
            }
            for method_name, call in count_methods.items():
                timings, count = measure(call, repeats)
                if count != legacy_count:
                    raise RuntimeError(f"Unstable count from {method_name}")
                for repeat, elapsed in enumerate(timings, start=1):
                    rows.append(
                        {
                            "stage": "count",
                            "graph": graph_name,
                            "nodes": node_count,
                            "max_degree": max_degree,
                            "method": method_name,
                            "samples": 0,
                            "repeat": repeat,
                            "elapsed_seconds": elapsed,
                            **metadata,
                        }
                    )

    for graph_name, node_count in SAMPLING_NODES_BY_GRAPH.items():
        graph = make_graphs(node_count)[graph_name]
        max_degree = max(dict(graph.to_undirected().degree()).values(), default=0)
        for sample_count in SAMPLE_SIZES:
            samplers = {
                "legacy sampler": randomTopoSortsLegacy,
                "Atkinson sampler": randomTopoSorts,
            }
            for method_name, sampler in samplers.items():
                def set_seed(repeat):
                    random.seed(RANDOM_SEED + repeat)

                def validate_samples(samples):
                    if len(samples) != sample_count or not all(
                        is_valid_topological_order(graph, sample) for sample in samples
                    ):
                        raise RuntimeError(
                            f"Invalid sample produced by {method_name} on {graph_name}"
                        )

                timings, samples = measure(
                    lambda sampler=sampler: sampler(graph, sample_count),
                    repeats,
                    before_each=set_seed,
                    after_each=validate_samples,
                )
                for repeat, elapsed in enumerate(timings, start=1):
                    rows.append(
                        {
                            "stage": "sampling",
                            "graph": graph_name,
                            "nodes": node_count,
                            "max_degree": max_degree,
                            "method": method_name,
                            "samples": sample_count,
                            "repeat": repeat,
                            "elapsed_seconds": elapsed,
                            **metadata,
                        }
                    )
    return rows


def write_csv(rows):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "graph",
        "nodes",
        "max_degree",
        "method",
        "samples",
        "repeat",
        "elapsed_seconds",
        "python",
        "platform",
        "cpu",
        "networkx_version",
        "matplotlib_version",
        "scipy_version",
        "seed",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def median_rows(rows, stage, graph_name, method_name):
    selected = [
        row
        for row in rows
        if row["stage"] == stage
        and row["graph"] == graph_name
        and row["method"] == method_name
    ]
    grouped = {}
    for row in selected:
        key = int(row["nodes"] if stage == "count" else row["samples"])
        grouped.setdefault(key, []).append(float(row["elapsed_seconds"]))
    return sorted(
        (key, statistics.median(values)) for key, values in grouped.items()
    )


def plot_results(rows):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    graph_titles = {
        "path": "Path (maximum degree 2)",
        "binary_tree": "Binary tree (maximum degree 3)",
        "star": "Star (maximum degree n - 1)",
    }
    count_methods = ("topoSortsCalc", "Atkinson")
    sampler_methods = ("legacy sampler", "Atkinson sampler")
    colors = {"topoSortsCalc": "tab:blue", "Atkinson": "tab:orange"}
    sampler_colors = {
        "legacy sampler": "tab:blue",
        "Atkinson sampler": "tab:orange",
    }

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for axis, (graph_name, title) in zip(axes, graph_titles.items()):
        for method_name in count_methods:
            data = median_rows(rows, "count", graph_name, method_name)
            axis.plot(
                [item[0] for item in data],
                [item[1] for item in data],
                marker="o",
                color=colors[method_name],
                label=method_name,
            )
        axis.set_title(title)
        axis.set_xlabel("Number of nodes")
        axis.set_xticks(
            [
                node_count
                for node_count, _ in median_rows(
                    rows, "count", graph_name, count_methods[0]
                )
            ]
        )
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Median runtime (seconds, log scale)")
    axes[-1].legend(loc="best", fontsize="small")
    figure.suptitle("Standalone exact topological-sort counting")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "toposort_count_runtime.png", dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for axis, (graph_name, title) in zip(axes, graph_titles.items()):
        for method_name in sampler_methods:
            data = median_rows(rows, "sampling", graph_name, method_name)
            axis.plot(
                [item[0] for item in data],
                [item[1] for item in data],
                marker="o",
                color=sampler_colors[method_name],
                label=method_name,
            )
        axis.set_title(f"{title} (n={SAMPLING_NODES_BY_GRAPH[graph_name]})")
        axis.set_xlabel("Number of sampled orders")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Median end-to-end runtime (seconds, log scale)")
    axes[-1].legend(loc="best", fontsize="small")
    figure.suptitle(
        "End-to-end sampling runtime (includes internal counts)"
    )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "toposort_sampling_runtime.png", dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    rows = benchmark(args.repeats)
    write_csv(rows)
    plot_results(rows)
    print(f"Saved timings to {OUTPUT_CSV}")
    print(f"Saved figures to {FIGURE_DIR}")
    groups = {}
    for row in rows:
        key = (
            row["stage"],
            row["graph"],
            row["nodes"],
            row["max_degree"],
            row["method"],
            row["samples"],
        )
        groups.setdefault(key, []).append(row["elapsed_seconds"])
    for key, timings in groups.items():
        stage, graph, nodes, degree, method, samples = key
        print(
            f"{stage:8} {graph:11} n={nodes:2} degree={degree:2} "
            f"{method:17} samples={samples:4} "
            f"median={statistics.median(timings):.6f}s"
        )
    print(
        f"Environment: Python {rows[0]['python']}, {rows[0]['platform']}, "
        f"CPU {rows[0]['cpu']}, NetworkX {rows[0]['networkx_version']}, "
        f"Matplotlib {rows[0]['matplotlib_version']}, SciPy {rows[0]['scipy_version']}"
    )


if __name__ == "__main__":
    main()
