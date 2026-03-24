# ASV para Árboles de Decisión en Redes Bayesianas

Código fuente de la tesis **"Optimización de ASV para árboles de decisión"**, desarrollada en el marco de las pasantías BICC en la Licenciatura en Ciencias de la Computación de la UBA. El repositorio implementa un algoritmo exacto y uno aproximado para el cálculo del ASV (Asymmetric Shapley Value), conteo eficiente de clases de equivalencia, y sampleo de órdenes topológicos, junto con los experimentos reportados en la tesis.

---

## Tabla de contenidos

- [Estructura del repositorio](#estructura-del-repositorio)
- [Algoritmos y dónde encontrarlos](#algoritmos-y-dónde-encontrarlos)
  - [Cap. 2 – Fórmula SHAP y ASV](#cap-2--fórmula-shap-y-asv)
  - [Cap. 3 – Redes Bayesianas y predicción promedio](#cap-3--redes-bayesianas-y-predicción-promedio)
  - [Cap. 4–5 – Clases de equivalencia para dtrees](#cap-45--clases-de-equivalencia-para-dtrees)
  - [Cap. 6 – Sampleo de órdenes topológicos en polytrees](#cap-6--sampleo-de-órdenes-topológicos-en-polytrees)
  - [Cap. 7 – ASV end to end (exacto y aproximado)](#cap-7--asv-end-to-end-exacto-y-aproximado)
- [Experimentos](#experimentos)
- [Archivos de redes Bayesianas](#archivos-de-redes-bayesianas)
- [Resultados](#resultados)
- [Tests](#tests)

---

## Estructura del repositorio

```
asvFormula/               # Librería principal
│
├── ASV.py                # Clases ASV y ApproximateASV (puntos de entrada)
├── datasetManipulation.py# Generación de datos, encoding, pipeline de entrenamiento de DT
├── digraph.py            # Utilidades de DAGs y constructores de grafos para experimentos
├── experiments.py        # Harness de experimentos y helpers de plotting
├── testingFunctions.py   # Helpers de validación cruzada (naïve vs. recursivo)
│
├── bayesianNetworks/
│   ├── bayesianNetwork.py# Conversión BN ↔ dígrafo, predicción promedio para DT+BN
│   └── pathCondition.py  # Seguimiento de restricciones de camino durante travesía del DT
│
├── classesSizes/
│   ├── equivalenceClass.py   # Estructura EquivalenceClass + fórmula de conteo
│   ├── naiveFormula.py       # Baseline fuerza bruta (solo para validación)
│   ├── recursiveFormula.py   # Algoritmo recursivo eficiente (usado por ASV)
│   └── algorithmTime.py      # Utilidades de profiling para el algoritmo recursivo
│
├── topoSorts/
│   ├── topoSortsCalc.py      # Contar topo-sorts de un polytree (algoritmo actual)
│   ├── topoSortsCalc_basic.py# Algoritmo de conteo anterior (mantenido para comparación)
│   ├── randomTopoSortsGeneration.py  # Sampler exacto ponderado + sampler MCMC
│   ├── toposPositions.py     # Distribuciones de posición por nodo en topo-sorts
│   └── utils.py              # TopoSortHasher, multinomial_coefficient, helpers
│
└── tests/
    ├── test_equivalenceClasses.py  # Recursivo vs. naïve en clases de equivalencia
    ├── test_topoSorts.py           # Conteo de topo-sorts y tests de posición
    └── test_mcmcTopoSorts.py       # Correctitud y uniformidad del sampler MCMC

experiments/              # Jupyter notebooks (uno por sección de experimentos)
networksExamples/         # Archivos .bif para las BNs Cancer, Child y Student
results/                  # Salidas CSV y plots producidos por los experimentos
```

---

## Algoritmos y dónde encontrarlos

### Cap. 2 – Fórmula SHAP y ASV

La fórmula del ASV está implementada en **`asvFormula/ASV.py`**:

- `ASV.asvForFeature(feature, instance)` — calcula el ASV exacto iterando sobre las clases de equivalencia ponderadas por su tamaño.
- El baseline SHAP usado para comparación se computa con la librería `shap` dentro de `asvFormula/experiments.py` (`writeASVAndShapleyIntoFile`).

---

### Cap. 3 – Redes Bayesianas y predicción promedio

- **`asvFormula/bayesianNetworks/bayesianNetwork.py`** — convierte una red Bayesiana de `pgmpy` y un árbol de decisión de scikit-learn en dígrafos (`bayesianNetworkToDigraph`, `obtainDecisionTreeDigraph`) e implementa el algoritmo central de **predicción promedio** $E[f(x) \mid \text{evidencia}]$ (`meanPredictionForDTinBNWithEvidence`).
- **`asvFormula/bayesianNetworks/pathCondition.py`** — `PathCondition` registra las restricciones de rango sobre features acumuladas al recorrer un camino en el DT, y enumera combinaciones de evidencia válidas (`allPossibleEvents`).
- **`asvFormula/datasetManipulation.py`** — pipeline completo: samplea un dataset de la BN, lo codifica, entrena un `DecisionTreeClassifier` y construye el dígrafo correspondiente (`initializeDataAndRemoveVariable`).

---

### Cap. 4–5 – Clases de equivalencia para dtrees

La Sección 4 (conteo de topo-sorts de un DAG) y la Sección 5 (algoritmo exacto de clases de equivalencia para dtrees) están en **`asvFormula/classesSizes/`**:

| Archivo | Contenido |
|---------|-----------|
| `equivalenceClass.py` | Estructura `EquivalenceClass`; fórmula `numberOfEquivalenceClasses` (§4.1) |
| `naiveFormula.py` | Baseline fuerza bruta `naiveEquivalenceClassesSizes` (§5.1, solo validación) |
| `recursiveFormula.py` | Algoritmo recursivo eficiente `equivalenceClassesFor` (§5.2); incluye `unrelatedEquivalenceClassesSizes`, `lastUnionOf` y los merges multinomiales |
| `algorithmTime.py` | Utilities de benchmarking usadas en §8.1 |

El `recursiveFormula` es la versión invocada por `ASV.equivalenceClasses(feature)`.

---

### Cap. 6 – Sampleo de órdenes topológicos en polytrees

Todo el código de sampleo está en **`asvFormula/topoSorts/`**:

| Archivo | Contenido |
|---------|-----------|
| `topoSortsCalc.py` | `allPolyTopoSorts` — cuenta topo-sorts de un polytree (algoritmo final de §6.2) usando `allPossibleOrders` con multinomial cacheado |
| `topoSortsCalc_basic.py` | Algoritmo de conteo anterior, restringido a árboles con a lo sumo un nodo compartido (idea inicial de §6.2) |
| `randomTopoSortsGeneration.py` | **Sampler exacto ponderado** `randomTopoSorts` para polytrees (§6.1); y **sampler MCMC** `mcmcTopoSorts` (Huber 1998) para DAGs generales |
| `toposPositions.py` | `positionsInToposorts` — distribución de posiciones por nodo, usada por el algoritmo básico |
| `utils.py` | `multinomial_coefficient`, `TopoSortHasher`, `topoSortsFrom` |

El algoritmo MCMC (`mcmcTopoSorts` / `mcmcSteps`) implementa la cadena de Markov de intercambios adyacentes de Huber (1998): en cada paso se propone un intercambio de un par adyacente aleatorio; el intercambio se acepta con probabilidad 1/2 si los dos nodos son incomparables en el DAG (verificado en O(1) mediante el cierre transitivo precomputado). Tras $n^3$ pasos de burn-in la cadena está aproximadamente mezclada; las muestras se recolectan con $n^3$ pasos entre cada una.

---

### Cap. 7 – ASV end to end (exacto y aproximado)

Ambas variantes están en **`asvFormula/ASV.py`**:

- `ASV` — versión exacta: usa `equivalenceClassesFor` (fórmula recursiva) para enumerar las clases de equivalencia y las pondera por tamaño de clase multiplicado por $E[f(x) \mid \text{representante de clase}]$.
- `ApproximateASV(ASV)` — versión aproximada: sobreescribe `equivalenceClasses` para usar `mcmcTopoSorts` (sampleo aleatorio) en lugar de enumeración exacta, y luego agrupa las muestras por hash en clases de equivalencia.

---

## Experimentos

Cada notebook en `experiments/` corresponde a una sección del Capítulo 8 de la tesis:

| Notebook | Sección de la tesis | Qué mide |
|----------|---------------------|---------|
| `classSizesExperiments.ipynb` | §8.1 — Clases de equivalencia vs. Órdenes Topológicos | Tiempo de ejecución del algoritmo recursivo de clases de equivalencia vs. enumeración bruta de topo-sorts en grafos sintéticos (Naive Bayes, balanced trees, multiple-paths). Guarda datos de timing en `results/graphData/`. |
| `ASVinBayesianNetworksExperiments.ipynb` | §8.2 — ASV vs. SHAP | Calcula valores exactos de ASV y SHAP para las BNs Cancer y Child en múltiples seeds; guarda resultados en `results/asvRuns/`. |
| `ApproximateASVExperiments.ipynb` | §8.3 — ASV exacto sin EqClasses vs. ASV aproximado | Compara `ApproximateASV` (sampleo MCMC) contra ASV exacto en las redes Cancer y Child. |
| `meanPredictionExperiments.ipynb` | §8.4 — Predicción promedio para DTs binarios | Valida que `meanPredictionForDTinBNWithEvidence` coincide con la enumeración naïve del dataset y con el valor esperado de SHAP. |
| `topoSortsComparisonExperiments.ipynb` | §8 / §6 — Comparación de samplers de topo-sorts | Benchmarkea el sampler exacto ponderado (`randomTopoSorts`) contra el sampler MCMC (`mcmcTopoSorts`) en cadenas, árboles binarios balanceados y grafos Naive Bayes, variando tamaño del grafo y número de muestras. |
| `randomTopoSortsExperiments.ipynb` | §6 — Sampleo aleatorio de topo-sorts | Prueba calidad de sampleo, tiempos y uniformidad de `randomTopoSorts` en distintos tipos de grafos. |
| `bayesianNetworksExperiments.ipynb` | §3 — Redes Bayesianas | Notebook exploratorio: carga archivos `.bif`, consulta distribuciones, entrena y visualiza árboles de decisión desde muestras de la BN. |

---

## Archivos de redes Bayesianas

Los archivos `.bif` en `networksExamples/` son las redes Bayesianas de benchmark estándar usadas en los experimentos:

- `cancer.bif` — Red Cancer (5 nodos)
- `child.bif` — Red Child (20 nodos)
- `student.bif` — Red Student

---

## Resultados

Los resultados precomputados se almacenan en `results/`:

```
results/
├── asvRuns/          # CSVs con valores de ASV y SHAP (redes Cancer y Child)
├── graphData/        # CSVs de timing para el benchmark de clases de equivalencia (§8.1)
├── plots/            # Figuras generadas
└── probabilityShifts/# CSVs de probability shifts para los experimentos Cancer y Child
```

---

## Tests

Los tests unitarios y de regresión están en `asvFormula/tests/`. Para ejecutarlos:

```bash
pytest asvFormula/tests/
```

| Archivo de test | Qué cubre |
|-----------------|-----------|
| `test_equivalenceClasses.py` | Fórmula recursiva vs. naïve en múltiples familias de grafos |
| `test_topoSorts.py` | Conteo de topo-sorts (`allPolyTopoSorts`) y distribuciones de posición |
| `test_mcmcTopoSorts.py` | Validez y uniformidad aproximada del sampler MCMC |
