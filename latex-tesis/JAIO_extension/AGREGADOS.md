# Agregados a la extensión JAIO

Registro acumulativo de contenido nuevo incorporado al paper, sus experimentos y figuras. Todos los agentes que trabajen en este repositorio deben leer este archivo al comenzar; si agregan o modifican contenido del paper, experimentos o figuras, deben sumar una entrada fechada al terminar.

## 26 de noviembre de 2026 — Separación de conteo y muestreo

- Se agregó `experiments/benchmark_toposort_count_sampling.py` para medir por separado los algoritmos standalone de conteo (`topoSortsCalc` y Atkinson) y el tiempo end-to-end de los samplers legado y Atkinson.
- El experimento compara caminos, árboles binarios y estrellas; valida que ambos conteos coincidan y que todas las muestras sean órdenes topológicos válidos. Registra las cinco mediciones individuales por condición, además del entorno, cantidad de nodos, grado máximo y tamaño de muestra, en `results/toposortBenchmarks/count_sampling_timings.csv`; las figuras muestran medianas.
- Los gráficos `img/toposort_count_runtime.png` y `img/toposort_sampling_runtime.png` se insertaron en la sección experimental del paper. El gráfico del sampler muestra tiempo end-to-end, incluido el conteo interno; no se suman los tiempos de conteo standalone al tiempo de muestreo.
- Para repetir el experimento: `uv run --python 3.11 --no-project --with networkx==3.2.1 --with matplotlib==3.9.4 --with scipy==1.13.1 python experiments/benchmark_toposort_count_sampling.py`.
- En esta ejecución (Python 3.11.15, Windows build 22621, NetworkX 3.2.1, SciPy 1.13.1, Matplotlib 3.9.4), `topoSortsCalc` tardó 1.171 s al contar la estrella de 10 nodos y Atkinson 0.0038 s. Para 1000 muestras en la misma estrella, el sampler legado tardó 0.682 s y el Atkinson 1.914 s. En el árbol binario, los samplers tardaron 0.097 y 0.179 s; en el camino, 0.027 y 0.062 s. En estas familias pequeñas, Atkinson reduce mucho el conteo standalone de la estrella, pero el muestreador end-to-end Atkinson tarda más. Son resultados de una máquina y grafos pequeños; no deben generalizarse a todos los polytrees.

## 26 de noviembre de 2026 — Sampler perfecto de Huber

- Se agregó `asvFormula/topoSorts/huber.py`, con cadena de cotas y CFTP no markoviano para órdenes topológicos de DAGs, y se exportaron sus funciones en `asvFormula/topoSorts/__init__.py`.
- `asvFormula/tests/test_huber.py` comprueba exhaustivamente las transiciones y la condición de acotamiento en DAGs de cuatro nodos, la reproducción de bloques aleatorios, casos límite y frecuencias en un ejemplo pequeño. Esas comprobaciones apoyan la implementación; no constituyen una demostración estadística de uniformidad.
- `README.md` describe el uso, la referencia y sus límites. El sampler no cambia el comportamiento predeterminado de `ApproximateASV`.
- Sondeo exploratorio de generación: 100 órdenes, tres semillas (0, 1, 2), Python 3.14.5/NetworkX 3.7, Windows con Intel i5-7500. Medianas Huber frente a MCMC con `n³` pasos: diamond de 4 nodos 0,0520 s frente a 0,0076 s; anticadena de 5, 0,1082 s frente a 0,0203 s; cadena de 5, 0,2001 s frente a 0,0152 s. Incluye cierre transitivo; se validaron los órdenes, no la precisión ni el tiempo ASV end-to-end. Hubo alta variación y no se controló la carga de la máquina.

## 26 de noviembre de 2026 — Línea de base ASV y comparación de samplers

- Se agregaron `requirements.in` y `requirements.txt` para Windows/Python 3.11, las pruebas focalizadas y `experiments/sampler_baseline.py`; la ejecución y las limitaciones se explican en `README.md`.
- El sampler ponderado predeterminado usa Atkinson; `randomTopoSortsLegacy` conserva el conteo anterior. Se usan pesos enteros, se rechazan cantidades inválidas y se corrigió la cardinalidad de salida de ASV para que coincida con la predicción. Se validaron signo, suma de contribuciones, árboles pequeños y subbosques enumerados, frecuencias de un bosque, y una predicción en una BN con caminos de probabilidad cero.
- `python -m experiments.sampler_baseline` produjo `results/sampler_baseline/`, con CSV y trece gráficos de frecuencias, tiempos, error y correlación para cuatro grafos de seis nodos, tres semillas y 100/500 muestras. Se comparan Huber y MCMC en todos, y los dos samplers ponderados en los bosques.
- Para 500 muestras, tiempos medios de generación de Huber frente a MCMC: 0,692 s frente a 0,136 s en la estrella y 0,745 s frente a 0,170 s en el diamante. El conteo aislado de la estrella de ocho nodos dio medianas de 0,0051 s anterior y 0,0017 s Atkinson. Son mediciones locales, no cotas; la predicción cronometrada es un oráculo sintético, no inferencia BN real. La evidencia condicionante imposible y una comparación más amplia siguen pendientes.
- Pruebas ejecutadas: `python -m pytest asvFormula\tests\test_genericASV.py asvFormula\tests\test_linearExtensionsTree.py asvFormula\tests\test_mean_prediction_baseline.py asvFormula\tests\test_asv.py asvFormula\tests\test_asv_sign.py asvFormula\tests\test_mcmcTopoSorts.py -q -k "not Distribution"` (32 aprobadas, cuatro deseleccionadas); `python -m pytest asvFormula\tests\test_huber.py -q` (cinco aprobadas); además, 14 pruebas seleccionadas de conteo existente aprobaron.

## 26 de noviembre de 2026 — Optimización de Huber

- En `asvFormula/topoSorts/huber.py` se evita el sampler cuando existe un único orden, se empacan posición y moneda en un entero con rechazo de posiciones inválidas, y se evita actualizar la cota ante una moneda cero. Se mantiene la reproducción de las elecciones en los bloques fallidos. Como cambia el consumo de bits, la misma semilla ya no reproduce la secuencia de la versión anterior.
- Se ampliaron las pruebas en `asvFormula/tests/test_huber.py`: verifican el rechazo de posiciones inválidas, el atajo para orden único, la ausencia de aliasing y que una moneda cero mantiene cada estado acotante alcanzable enumerado.
- En una medición exploratoria local con Python 3.11.15 y NetworkX 3.2.1, se compararon 300 órdenes de la versión original `405a7b2` y la optimizada para siete semillas, alternando el orden de ejecución. Las medianas anterior/optimizada fueron: cadena de 6 nodos 0,67769/0,00013 s; diamante de 6, 0,68280/0,32065 s; estrella de 6, 0,65234/0,37588 s; anticadena de 8, 1,74220/0,87647 s. Se verificó la validez de todos los órdenes. Las secuencias aleatorias son distintas, hubo variación y no se controló la carga de la máquina: no son cotas ni tiempos ASV end-to-end.
- Para repetir desde PowerShell, con el commit original disponible y el código optimizado en el working tree:

```powershell
@'
import random, statistics, subprocess, time, types
import networkx as nx
from asvFormula.topoSorts.huber import huberTopoSorts
source = subprocess.check_output(['git', 'show', '405a7b2:asvFormula/topoSorts/huber.py'], text=True)
before = types.ModuleType('huber_before')
exec(compile(source, 'huber_before.py', 'exec'), before.__dict__)
cases = {
    'cadena6': nx.path_graph(6, create_using=nx.DiGraph),
    'diamante6': nx.DiGraph([(0,1),(0,2),(1,3),(2,3),(3,4),(3,5)]),
    'estrella6': nx.DiGraph([(0,i) for i in range(1,6)]),
    'anticadena8': nx.empty_graph(8, create_using=nx.DiGraph),
}
for name, graph in cases.items():
    times = {'antes': [], 'ahora': []}
    for seed in range(7):
        samplers = [('antes', before.huberTopoSorts), ('ahora', huberTopoSorts)]
        if seed % 2: samplers.reverse()
        for label, sampler in samplers:
            random.seed(seed)
            start = time.perf_counter(); orders = sampler(graph, 300); elapsed = time.perf_counter() - start
            assert len(orders) == 300
            assert all(len(o)==len(graph) and set(o)==set(graph) and all(o.index(a)<o.index(b) for a,b in graph.edges()) for o in orders)
            times[label].append(elapsed)
    print(name, {k: round(statistics.median(v), 5) for k,v in times.items()}, 'corridas', times, flush=True)
'@ | uv run --offline --no-project --python 3.11 --with networkx==3.2.1 --with matplotlib==3.9.4 --with scipy==1.13.1 python -
```
