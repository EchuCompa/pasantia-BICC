"""Reproducible small-graph sampler comparison with exact reference values."""

import argparse
import csv
import json
import platform
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from asvFormula.classesSizes.naiveFormula import naiveEquivalenceClassesSizes
from asvFormula.genericASV import asvFromTopologicalOrders
from asvFormula.topoSorts.huber import huberTopoSorts
from asvFormula.topoSorts.randomTopoSortsGeneration import (
    mcmcTopoSorts,
    randomTopoSorts,
    randomTopoSortsLegacy,
)
from asvFormula.topoSorts.utils import TopoSortHasher
from asvFormula.topoSorts.linearExtensionsTree import allPolyforestTopoSortsAtkinson
from asvFormula.topoSorts.topoSortsCalc import allPolyTopoSorts, allPossibleOrders
from asvFormula.topoSorts.utils import multinomial_coefficient_cached


def graphs():
    return {
        "cadena": nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        "estrella": nx.DiGraph([(0, i) for i in range(1, 6)]),
        "bosque": nx.DiGraph([(0, 1), (2, 3), (2, 4), (4, 5)]),
        "diamante": nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)]),
    }


def value(coalition):
    # A nonlinear vector oracle makes predecessor-set sampling matter.
    p = (sum((node + 1) for node in coalition) + 3 * ({1, 2} <= coalition)) / 24
    return np.array([1 - p, p])


def lag_one(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or np.std(x[:-1]) == 0 or np.std(x[1:]) == 0:
        return float("nan")
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def run_case(name, dag, sampler_name, sampler, count, seed):
    exact_orders = list(nx.all_topological_sorts(dag))
    exact = np.array([asvFromTopologicalOrders(node, exact_orders, value) for node in dag])
    start = time.perf_counter()
    nx.transitive_closure_dag(dag)
    closure_seconds = time.perf_counter() - start

    random.seed(seed)
    start = time.perf_counter()
    orders = sampler(dag, count)
    generation_seconds = time.perf_counter() - start
    # The weighted implementations return groups of identical prefixes, not draw order.
    # Shuffle only those exchangeable batches before measuring lag correlation.
    if sampler_name.startswith("ponderado"):
        random.Random(seed + 100000).shuffle(orders)
    frequencies = Counter(map(tuple, orders))
    tv = sum(abs(frequencies[tuple(order)] / count - 1 / len(exact_orders))
             for order in exact_orders) / 2
    positions = [order.index(1) for order in orders]

    start = time.perf_counter()
    classes = [naiveEquivalenceClassesSizes(orders, node, TopoSortHasher(dag, node))
               for node in dag]
    grouping_seconds = time.perf_counter() - start
    start = time.perf_counter()
    estimates = []
    for node, groups in zip(dag, classes):
        result = np.zeros(2)
        for order, multiplicity in groups.values():
            index = order.index(node)
            before = frozenset(order[:index])
            result += multiplicity * (value(before | {node}) - value(before))
        estimates.append(result / count)
    prediction_seconds = time.perf_counter() - start
    total = generation_seconds + grouping_seconds + prediction_seconds
    return {
        "graph": name, "sampler": sampler_name, "samples": count, "seed": seed,
        "extensions": len(exact_orders), "closure_seconds": closure_seconds,
        "generation_seconds": generation_seconds, "grouping_seconds": grouping_seconds,
        "prediction_seconds": prediction_seconds, "total_seconds": total,
        "seconds_per_sample": generation_seconds / count,
        "total_variation": tv, "lag1_position_node1": lag_one(positions),
        "asv_rmse": float(np.sqrt(np.mean((np.array(estimates) - exact) ** 2))),
        "unique_orders": len(frequencies),
    }, [(name, sampler_name, count, seed, "|".join(map(str, order)),
         frequencies[tuple(order)], 1 / len(exact_orders)) for order in exact_orders]


def plot_results(rows, output):
    import pandas as pd

    data = pd.DataFrame(rows)
    metrics = [
        ("generation_seconds", "Tiempo de generación (s)"),
        ("grouping_seconds", "Tiempo de agrupación (s)"),
        ("prediction_seconds", "Tiempo del oráculo (s)"),
        ("total_variation", "Distancia TV a la distribución exacta"),
        ("lag1_position_node1", "Autocorrelación lag 1 de posición"),
        ("asv_rmse", "RMSE ASV vs. enumeración"),
        ("seconds_per_sample", "Tiempo de generación por muestra (s)"),
    ]
    for metric, ylabel in metrics:
        fig, axes = plt.subplots(1, len(graphs()), figsize=(15, 3.5), squeeze=False)
        for ax, (graph, group) in zip(axes[0], data.groupby("graph", sort=True)):
            for sampler, sub in group.groupby("sampler", sort=True):
                summary = sub.groupby("samples")[metric].agg(["mean", "std"])
                ax.errorbar(summary.index, summary["mean"], yerr=summary["std"].fillna(0),
                            marker="o", capsize=3, label=sampler)
            ax.set_title(graph)
            ax.set_xlabel("Muestras")
            ax.grid(alpha=0.25)
        axes[0, 0].set_ylabel(ylabel)
        axes[0, -1].legend()
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for sampler, group in data.groupby("sampler"):
        ax.scatter(group["total_seconds"], group["asv_rmse"], label=sampler, alpha=0.65)
    ax.set_xlabel("Tiempo total medido (s)")
    ax.set_ylabel("RMSE ASV")
    ax.set_title("Error vs. tiempo; puntos: grafo, tamaño y semilla")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "error_vs_tiempo.png", dpi=150)
    plt.close(fig)


def compare_counts(output):
    rows = []
    for size in range(3, 9):
        graph = nx.DiGraph([(0, node) for node in range(1, size)])
        counts = []
        for label, counter in (("anterior", allPolyTopoSorts),
                               ("atkinson", allPolyforestTopoSortsAtkinson)):
            timings = []
            for _ in range(5):
                allPossibleOrders.cache_clear()
                multinomial_coefficient_cached.cache_clear()
                start = time.perf_counter()
                count = counter(graph)
                timings.append(time.perf_counter() - start)
            rows.append({"nodes": size, "degree": size - 1, "counter": label,
                         "seconds": statistics.median(timings), "count": count})
            counts.append(count)
        if counts[0] != counts[1]:
            raise AssertionError(f"Conteos incompatibles para estrella de {size} nodos")
    with (output / "conteos.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in ("anterior", "atkinson"):
        subset = [row for row in rows if row["counter"] == label]
        ax.plot([row["degree"] for row in subset], [row["seconds"] for row in subset],
                marker="o", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Grado de la estrella")
    ax.set_ylabel("Tiempo de conteo (s)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "conteo_vs_grado.png", dpi=150)
    plt.close(fig)


def plot_frequencies(frequencies, output, samples, seed):
    import pandas as pd

    data = pd.DataFrame(frequencies, columns=[
        "graph", "sampler", "samples", "seed", "order", "frequency", "exact_probability"
    ])
    for name in graphs():
        subset = data[(data.graph == name) & (data.samples == samples) & (data.seed == seed)]
        fig, ax = plt.subplots(figsize=(10, 4))
        for sampler, group in subset.groupby("sampler", sort=True):
            ax.plot(range(len(group)), group.frequency.to_numpy() / samples,
                    marker=".", linewidth=0.8, label=sampler)
        ax.axhline(subset.exact_probability.iloc[0], color="black", linestyle="--", label="exacta")
        ax.set_title(f"{name}: frecuencias por orden, n={samples}, semilla={seed}")
        ax.set_xlabel("Índice de orden enumerado")
        ax.set_ylabel("Frecuencia relativa")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"frecuencias_{name}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results") / "sampler_baseline")
    parser.add_argument("--samples", type=int, nargs="+", default=[100, 500])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    args = parser.parse_args()
    if not args.samples or any(n <= 1 for n in args.samples) or not args.seeds:
        parser.error("Use al menos una semilla y cantidades de muestras mayores que 1")
    args.output.mkdir(parents=True, exist_ok=True)
    rows, frequencies = [], []
    for name, dag in graphs().items():
        samplers = [("mcmc", mcmcTopoSorts), ("huber", huberTopoSorts)]
        if nx.is_forest(dag.to_undirected()):
            samplers += [("ponderado_anterior", randomTopoSortsLegacy),
                         ("ponderado_atkinson", randomTopoSorts)]
        for sampler_name, sampler in samplers:
            for count in args.samples:
                for seed in args.seeds:
                    row, freqs = run_case(name, dag, sampler_name, sampler, count, seed)
                    rows.append(row)
                    frequencies.extend(freqs)
                    print(name, sampler_name, count, seed, flush=True)

    with (args.output / "mediciones.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "frecuencias.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["graph", "sampler", "samples", "seed", "order", "frequency", "exact_probability"])
        writer.writerows(frequencies)
    metadata = {"python": platform.python_version(), "networkx": nx.__version__,
                "numpy": np.__version__, "samples": args.samples, "seeds": args.seeds,
                "mcmc_steps": "n^3 burn-in and between samples",
                "oracle": "synthetic nonlinear vector; not Bayesian inference"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_results(rows, args.output)
    plot_frequencies(frequencies, args.output, max(args.samples), args.seeds[0])
    compare_counts(args.output)


if __name__ == "__main__":
    main()
