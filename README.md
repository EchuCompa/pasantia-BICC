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
│   ├── huber.py              # Muestreo perfecto para DAGs por Huber
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
| `topoSortsCalc.py` | `allPolyTopoSorts` — cuenta topo-sorts de un polytree (algoritmo de §6.2) usando `allPossibleOrders` con multinomial cacheado |
| `linearExtensionsTree.py` | Conteo de órdenes lineales y distribución de posiciones mediante la recurrencia de Atkinson para árboles y bosques dirigidos |
| `topoSortsCalc_basic.py` | Algoritmo de conteo anterior, restringido a árboles con a lo sumo un nodo compartido (idea inicial de §6.2) |
| `randomTopoSortsGeneration.py` | **Sampler exacto ponderado** `randomTopoSorts` para polyforests, con conteo de Atkinson; `randomTopoSortsLegacy` conserva el conteo anterior para comparación, y `mcmcTopoSorts` es el sampler MCMC para DAGs generales |
| `huber.py` | `huberTopoSort` y `huberTopoSorts` — muestreo perfecto por cadena de cotas y CFTP no markoviano para DAGs generales |
| `toposPositions.py` | `positionsInToposorts` — distribución de posiciones por nodo, usada por el algoritmo básico |
| `utils.py` | `multinomial_coefficient`, `TopoSortHasher`, `topoSortsFrom` |

El conteo por posiciones sigue la recurrencia de [Atkinson (1990)](https://doi.org/10.1007/BF00383170), con $O(n^2)$ operaciones aritméticas en un árbol; se combinan componentes de bosques con un multinomial. Esto no es una cota en operaciones de bits ni para el costo total de muestreo. Los contadores se comparan exhaustivamente con enumeración para todas las orientaciones de árboles no isomorfos de hasta cinco nodos y sus subbosques.

El algoritmo MCMC (`mcmcTopoSorts` / `mcmcSteps`) usa intercambios adyacentes: en cada paso se propone un par adyacente aleatorio; el intercambio se acepta con probabilidad 1/2 si los dos nodos son incomparables en el DAG (verificado en O(1) mediante el cierre transitivo precomputado). Los $n^3$ pasos predeterminados son una heurística, no una garantía de mezcla o independencia; las muestras consecutivas pueden estar correlacionadas. Los samplers ponderados agrupan las salidas por prefijo y no preservan el orden temporal de los sorteos; sus frecuencias son válidas, pero hay que permutar el lote antes de estudiar correlación. La selección usa pesos enteros exactos y `random.seed` permite reproducir las salidas en un mismo entorno; las secuencias con una misma semilla cambiaron respecto de la versión anterior.

El sampler de [Huber (2006)](https://doi.org/10.1016/j.disc.2006.01.003) utiliza una cadena de cotas y coupling from the past no markoviano, sin un número fijo de pasos de calentamiento. Cada orden se genera en una ejecución independiente. La garantía de uniformidad exacta supone bits aleatorios independientes ideales; aquí se usa el generador pseudoaleatorio `random` de Python. El trabajo da O(n³ log n) pasos esperados para su algoritmo; esto no es una medición ni una garantía de tiempo real para esta implementación, que también calcula el cierre transitivo y guarda bloques aleatorios. Ejemplo:

```python
import networkx as nx
from asvFormula.topoSorts import huberTopoSorts

dag = nx.DiGraph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
orders = huberTopoSorts(dag, 100)
```

Se rechazan grafos que no sean DAGs y cantidades negativas; el número de pasos y la memoria son variables y no se impone un cutoff silencioso. Si el DAG tiene un único orden, se devuelve directamente. La generación optimizada usa bits aleatorios de manera distinta a la versión anterior: la misma semilla no reproduce sus muestras históricas. `ApproximateASV` no cambia de sampler automáticamente: Huber se ofrece por separado para comparar.

---

### Cap. 7 – ASV end to end (exacto y aproximado)

Ambas variantes están en **`asvFormula/ASV.py`**:

- `ASV` — versión exacta: usa `equivalenceClassesFor` (fórmula recursiva) para enumerar las clases de equivalencia y las pondera por tamaño de clase multiplicado por $E[f(x) \mid \text{representante de clase}]$.
- `ApproximateASV(ASV)` — versión aproximada: para polyforests usa `randomTopoSorts`, y para DAGs cuyo grafo subyacente tiene ciclos usa `mcmcTopoSorts`; después agrupa las muestras por hash en clases de equivalencia. `randomTopoSortsLegacy` permite comparar el conteo anterior; el sobrecosto de Atkinson puede hacerlo más lento en grafos pequeños. El MCMC es una aproximación heurística: el número predeterminado de pasos no es una garantía de mezcla ni independencia.

La contribución es predicción **con** la feature menos predicción **sin** ella, y el vector de salida tiene la cardinalidad de las clases predichas, no la de la feature explicada.

Para modelos o distribuciones no integrados con pgmpy, **`asvFormula/genericASV.py`** ofrece `asvFromTopologicalOrders`: recibe órdenes topológicos y un oráculo de función característica. Sirve como interfaz model-agnostic y baseline correcto, pero no resuelve por sí solo el muestreo de DAGs ni acelera el cálculo del oráculo; la optimización específica de inferencia sigue en la implementación con redes bayesianas.

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
| `synthetic_asv_example.py` | Ejemplo sintético | Compara ASV y Shapley simétrico para una cadena binaria causal con distintos niveles de ruido, usando la misma función característica observacional. |
| `sampler_baseline.py` | Línea de base reproducible | Compara samplers y conteos con enumeración exacta; genera CSV y gráficos de tiempos, frecuencias, correlación y error. |
| `bayesianNetworksExperiments.ipynb` | §3 — Redes Bayesianas | Notebook exploratorio: carga archivos `.bif`, consulta distribuciones, entrena y visualiza árboles de decisión desde muestras de la BN. |

El ejemplo sintético se puede ejecutar desde la raíz del repositorio con `python -m experiments.synthetic_asv_example`.

### Línea de base reproducible

En Windows con Python **3.11.15**, desde la raíz:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest asvFormula\tests\test_asv.py asvFormula\tests\test_asv_sign.py asvFormula\tests\test_genericASV.py asvFormula\tests\test_linearExtensionsTree.py asvFormula\tests\test_mean_prediction_baseline.py
.\.venv\Scripts\python.exe -m experiments.sampler_baseline
```

`requirements.in` fija dependencias directas de la biblioteca, estas pruebas y el experimento; `requirements.txt` fija también las transitivas resueltas para Windows y Python 3.11. No incluye dependencias adicionales de todos los notebooks (como SHAP). Los resultados se escriben en `results/sampler_baseline/`; un segundo lanzamiento sobrescribe esos archivos. Se pueden cambiar `--output`, `--samples` y `--seeds`.

El experimento compara cadena, estrella, bosque y diamante de seis nodos, con 100 y 500 muestras y tres semillas. Enumera todos los órdenes para obtener frecuencias y ASV de referencia. Los CSV guardan los datos por corrida y por orden; hay gráficos de frecuencia, distancia TV, correlación lag 1 de la posición del nodo 1, error ASV, error frente a tiempo, tiempos por etapa y conteo frente al grado. La barra indica desvío estándar entre semillas; una correlación de una variable constante es indefinida. El lote ponderado se permuta antes de medir correlación. La generación incluye todo el preprocesamiento de cada sampler; el cierre transitivo se mide aparte (`closure_seconds`) y no se suma de nuevo. El conteo ponderado y su caché se intercalan con la generación y no se separan como un preprocesamiento fijo; el contador anterior tiene además una caché global que puede permanecer caliente entre corridas. La comparación de conteo usa la mediana de cinco corridas con cachés de conteo vaciadas.

La etapa de predicción del experimento usa un oráculo vectorial sintético, **no** el tiempo de inferencia de pgmpy; la prueba separada de predicción promedio comprueba una BN determinista con caminos de probabilidad cero. Las mediciones son exploratorias, no una demostración de complejidad ni de mezcla. En esta corrida, Atkinson mejora el conteo de la estrella de grado 7, pero el sampler anterior es más rápido en las instancias pequeñas; ver los CSV para valores y variabilidad. Falta perfilar la inferencia real, analizar evidencia condicionante de probabilidad cero y ampliar tamaños, familias y semillas.

[Huber (2006)](https://doi.org/10.1016/j.disc.2006.01.003) propone muestreo perfecto con cadena acotante y acoplamiento desde el pasado modificado, con costo esperado $O(n^3\log n)$ según el artículo. No es el MCMC implementado aquí. En esta comparación pequeña, Huber tarda más que MCMC, aunque no requiere elegir burn-in; no se generaliza esa diferencia a otros tamaños. [Bubley–Dyer (1999)](https://doi.org/10.1016/S0012-365X(98)00333-1) analiza otra cadena, por lo que su cota no se atribuye aquí a esta implementación.

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
