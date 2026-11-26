# Seguimiento de la extensión JAIO

## Punto de partida

- La carpeta es una copia del estado de trabajo de `../JAIO` al iniciar la extensión; el paper original permanece separado.
- Objetivo: preparar una extensión para conferencia que mejore la implementación y la evaluación de valores de Shapley asimétricos (ASV).
- La tesis y el paper son la referencia inicial; las contribuciones, el alcance y la conferencia aún deben acordarse.

## Líneas a investigar

- Robustez, rendimiento y reproducibilidad de la implementación.
- Estrategias alternativas de muestreo y conteo de órdenes topológicos.
- Modelos causales más generales y costo de la predicción promedio.
- Ejemplos sintéticos que ilustren diferencias entre ASV y Shapley simétrico.
- Clases de equivalencia para polytrees y otras preguntas de complejidad planteadas en la tesis.

## Estado

- [x] Copiar el paper original y crear este registro.
- [x] Hacer una primera revisión de la implementación, los experimentos y la literatura.
- [ ] Acordar contribución, protocolo experimental y próximos hitos.

## Investigación inicial — 23 de noviembre de 2026

### Punto de partida y diagnóstico

El paper ya propone clases de equivalencia, conteo y muestreo para estructuras restringidas, y una aproximación conceptualmente aplicable a DAGs arbitrarios. Por lo tanto, «implementar otro sampler» o «permitir otros modelos» no es, por sí solo, una contribución teórica nueva. Una hipótesis de trabajo más fuerte es **ASV confiable y práctico: implementación validada, mejor conteo/muestreo y evaluación reproducible**, complementada con ejemplos que expliquen qué cambia al incorporar el orden causal. La novedad concreta y la conferencia quedan por decidir.

Hallazgos de la primera inspección, anteriores a los fixes descritos más abajo:

### Líneas de investigación y prioridad

1. **Robustez y línea de base — prioridad alta.** Preparar un entorno reproducible con versiones fijadas; probar contra enumeración de todos los órdenes en DAGs pequeños y expectativas calculadas exactamente. Cubrir signo, suma de atribuciones, grafos desconectados, entradas inválidas, cardinalidades, ceros de probabilidad y aleatoriedad reproducible. Perfilar por separado generación de órdenes, agrupación y predicción promedio. No cambiar resultados sin validar primero la convención de salida y los call sites.
2. **Conteo de polytrees — prioridad alta.** Atkinson (1990) da un algoritmo de **O(n²) operaciones aritméticas** para extensiones lineales de órdenes cuyo grafo de cobertura es un árbol, independientemente del grado. Un polytree dirigido satisface esa condición; al quitar fuentes aparecen bosques, que requieren combinar componentes. Es un candidato particularmente prometedor frente a la dependencia del grado actual: implementar y validar conteos por posición contra enumeración, después sustituir el conteo del sampler y medir el costo total. O(n²) cuenta operaciones aritméticas, no necesariamente operaciones de bits ni el costo total de generar muchas muestras. La sustitución no demuestra por sí sola un sampler de O(n²).
3. **Muestreo — prioridad alta después de validar la base.** Comparar el sampler ponderado actual, el mismo con conteo de Atkinson y MCMC en estructuras donde cada uno aplica. Evaluar Huber 2006 como candidato de muestreo **perfecto** para DAGs arbitrarios, con O(n³ log n) esperado según su artículo; no confundirlo con la cadena implementada. Bubley–Dyer estudia otra cadena y una cota de mezcla O(n³ log n); no adjudicar esa cota sin verificar que la transición implementada coincide. Medir preprocesamiento, tiempo por muestra, correlación y error ASV versus tiempo; contrastar frecuencias contra la distribución exacta en grafos pequeños. Elegir si vale la pena implementar Huber después de un prototipo y revisión detallada.
4. **Ejemplos sintéticos interpretables — prioridad alta.** Comparar Shapley simétrico y ASV usando *la misma* función característica condicional, modelo, distribución e instancia. Ejemplo analítico: `X → Y`, `X ~ Bernoulli(1/2)`, `Y = X`, `f(X,Y) = Y`, instancia `(1,1)`. Entonces `v(∅)=1/2`, `v({X})=v({Y})=v({X,Y})=1`; Shapley simétrico da `(1/4, 1/4)` y ASV con orden `X,Y` da `(1/2, 0)`. Ambas sumas son `f(e)-E[f]=1/2`. Es un ejemplo de reparto predictivo condicionado, no una prueba de efectos causales de intervención; además es ilustrativo, no nuevo por sí mismo. Agregar variantes con ruido, rama común y grafo vacío; revisar contra cómputo exacto y mostrar sensibilidad al grafo supuesto.
5. **Más modelos y DAGs — prioridad media.** Separar la generación de órdenes del cálculo de `v(S)=E[f(X) | X_S=e_S]` mediante una interfaz pequeña de oráculo, sin prometer inferencia eficiente universal. Primero admitir modelos predictivos y distribuciones con un oráculo explícito; luego investigar SCMs y oráculos aproximados, documentando cómo se calculan y qué significado causal tienen. Incluir la Child original solo con un método que soporte efectivamente DAGs generales; no asumir que funciona la ruta actual.
6. **Predicción promedio y evidencia parcial — prioridad media.** Medir cuántas consultas y coaliciones se repiten entre clases y features, y si la inferencia es el cuello de botella. Explorar cache por instancia/modelo/evidencia y reutilización de eliminaciones o consultas de unión; contrastar con enumeración, incluyendo evidencia imposible. Reportar memoria y controlar invalidación del cache. La inferencia observacional no se convierte en intervencional por usar un DAG causal.
7. **Clases de equivalencia para polytrees — investigación de mayor riesgo.** Derivar un algoritmo que respete padres compartidos y pesos de clases; comparar contra enumeración exhaustiva en grafos pequeños. Generalizar el conteo de órdenes no resuelve automáticamente el conteo de clases. Perseguirlo como contribución teórica solo si los primeros ejemplos y recurrencias son prometedores.
8. **Complejidad y enumeración — exploración acotada.** Jerrum–Valiant–Vazirani relacionan conteo *aproximado* y generación *casi uniforme* para relaciones autorreducibles; no es una equivalencia automática entre conteo exacto y muestreo exacto. Precisar hipótesis y parámetros antes de formular una afirmación. Inoue–Minato indexan órdenes mediante una estructura comprimida: estudiar memoria, costo de construcción y extracción aleatoria, sin suponer que enumerar o tomar prefijos deterministas produce muestras uniformes. Kangas et al. ofrecen conteo para órdenes dispersos y O(n^(t+4)) para treewidth `t`; con `t=1` resulta n⁵, frente al resultado especializado de Atkinson.

### Protocolo experimental propuesto

- **Corrección:** enumeración exhaustiva y conteo exacto en grafos pequeños; validar órdenes, distribuciones, pesos de clases, oráculos y atribuciones, incluida la propiedad de eficiencia. Usar al menos un ejemplo de referencia con solución analítica.
- **Familias:** cadenas, antichains, estrellas de alto grado, árboles balanceados, polytrees con múltiples padres, bosques y DAGs con ciclos en el grafo subyacente; variar tamaño, grado, densidad y estructura. Cancer y Child modificada para comparación histórica; Child original si la ruta general está validada.
- **Samplers:** para grafos pequeños, comparar frecuencias por orden y por coalición, incluyendo el estimador ASV; para grafos grandes, contrastar contra conteos/oráculos disponibles y medir autocorrelación y tamaño efectivo de muestra. Variar burn-in y separación para MCMC; no asumir independencia ni exactitud.
- **Rendimiento y precisión:** curvas de error absoluto y RMSE frente a tiempo total y número de muestras; informar error relativo solo con umbral explícito para valores cercanos a cero. Registrar preprocesamiento, inferencia, memoria, semillas, hardware, versiones, número de réplicas e intervalos de incertidumbre. Comparar métodos bajo igual presupuesto y misma función característica; separar comparaciones de semánticas distintas.
- **Ablaciones:** conteo original vs. Atkinson; caché activado/desactivado; exacto vs. muestreado; distribución condicional exacta vs. aproximada. No afirmar mejoras hasta medirlas.

## Resultados de pruebas e implementación inicial

- Corregido en `asvFormula/ASV.py` el signo de las contribuciones: ahora calcula `v(S ∪ {x}) - v(S)`, como en la definición del paper. Una prueba con un oráculo controlado reproduce la diferencia.
- Corregida la normalización de `ApproximateASV`: compara el peso de las clases con el número de muestras solicitado, no con el número exacto de órdenes. Para DAGs cuyo grafo subyacente no es un bosque utiliza MCMC; para polyforests conserva el sampler ponderado existente. La cadena sigue siendo aproximada y no tiene garantía de mezcla o muestras independientes.
- Contraejemplo reproducible: en el DAG diamond `a→b,a→c,b→d,c→d` hay 2 órdenes, mientras el conteo polytree antiguo daba 3 (y `allForestTopoSorts` daba 6). Se añadieron guardas a `allPolyTopoSorts` y al sampler ponderado para rechazar DAGs fuera de su dominio. **No se agregó una guarda a `allForestTopoSorts` ni a la ruta exacta de ASV**, que sigue restringida y no debe usarse con DAGs generales sin validación.
- Implementado `linearExtensionsTree.py`, con espectro de posiciones y conteo de órdenes por la recurrencia de Atkinson para árboles y bosques dirigidos. Se contrasta contra enumeración exhaustiva en árboles orientados aleatorios pequeños y bosques desconectados. Su sampler está disponible para comparar, pero **no se usa como default**: el costo de copiar/procesar subgrafos puede degradar el tiempo total de muestreo.
- Implementado `genericASV.py`: referencia independiente de pgmpy que recibe una función característica arbitraria y órdenes topológicos, y cachea consultas repetidas a `v(S)`. Esto demuestra la separación entre explicador y oráculo, pero la clase optimizada `ASV` sigue integrada a redes bayesianas y árboles de scikit-learn.
- Verificación final: `pytest asvFormula\tests\ -q` pasó **88 pruebas** (Python 3.14.5, dependencias temporales); queda un `FutureWarning` de pgmpy sobre `StructureScore`. El ejemplo sintético también se ejecutó correctamente como módulo.
- El experimento ejecutable `experiments/synthetic_asv_example.py` compara ambas atribuciones con la misma función característica observacional para `X→Y`, `Y=X XOR ruido`, `f(X,Y)=Y` y entidad `(1,1)`. La eficiencia suma `0.5` en todos los casos:

  | Ruido | SHAP(X,Y) | ASV(X,Y), orden X→Y |
  |---:|---:|---:|
  | 0.00 | (0.250, 0.250) | (0.500, 0.000) |
  | 0.10 | (0.200, 0.300) | (0.400, 0.100) |
  | 0.25 | (0.125, 0.375) | (0.250, 0.250) |

  Es un ejemplo de atribución observacional y ordenamiento causal asumido, no una comparación de efectos de intervención.

### Benchmarks preliminares

Mediciones locales exploratorias, Python 3.14.5/Windows, sin fijar afinidad ni estado de CPU. No son resultados publicables; medir de nuevo con protocolo controlado antes de citar:

| Conteo | Grafo | Conteo actual | Atkinson | Resultado |
|---|---:|---:|---:|---|
| Estrella | 8 nodos | mediana 0.0044 s | mediana 0.0040 s | iguales |
| Estrella | 10 nodos | mediana 0.652 s | mediana 0.0029 s | iguales; gran mejora en alto grado |
| Estrella | 100 nodos | — | 0.156 s | 156 dígitos en el conteo |
| Camino | 100 nodos | — | 0.190 s | una extensión |
| Árbol balanceado | 127 nodos | — | 0.056 s | 164 dígitos en el conteo |

En generación de 1000 órdenes, el sampler nuevo fue más lento que el actual en las tres familias probadas: estrella de 10 nodos 0.816 s vs. 0.496 s; camino de 10 nodos 0.034 s vs. 0.016 s; árbol balanceado de 15 nodos 3.731 s vs. 0.623 s. Por eso se conserva como alternativa experimental y no se cambia el default. La granularidad del experimento es preliminar y el generador de grafos debe fijarse antes de repetirlo.

Para el diamond, MCMC generó 10 000 órdenes válidos con frecuencias 5023/4977 entre los dos órdenes; el sampler ponderado antiguo generó 5069/4931 en este ejemplo pese a que su conteo total es erróneo. Esto **no prueba** mezcla, independencia ni uniformidad general. En una estrella de 4 nodos, ambos samplers cubrieron los seis órdenes con 12 000 muestras y frecuencias compatibles con uniformidad en una prueba de tolerancia fija.

### Próximos hitos sugeridos

- [ ] Elegir con el equipo la conferencia, extensión y contribución principal.
- [x] Encontrar un intérprete Python y ejecutar las pruebas con dependencias temporales.
- [x] Corregir el signo de ASV y la normalización/selección del sampler aproximado.
- [x] Prototipar Atkinson y comparar conteo y muestreo; mantener el sampler antiguo como default por rendimiento observado.
- [x] Implementar un oráculo genérico y el ejemplo sintético analítico.
- [ ] Fijar dependencias/semillas/hardware y repetir benchmarks con réplicas e intervalos.
- [ ] Probar la clase optimizada `ASV` de extremo a extremo con modelos/distribuciones no-BN y decidir el contrato de oráculo público.
- [ ] Ejecutar el protocolo de tiempo/error y decidir, según resultados, si profundizar en Huber, clases para polytrees o modelos generales.

### Fuentes

- [Atkinson, «On computing the number of linear extensions of a tree» (1990)](https://link.springer.com/article/10.1007/BF00383170), O(n²) operaciones aritméticas.
- [Kangas et al., «Counting Linear Extensions of Sparse Posets» (IJCAI 2016)](https://ijcai.org/proceedings/2016/92), conteo por componentes y treewidth.
- [Huber, «Fast perfect sampling from linear extensions» (2006)](https://doi.org/10.1016/j.disc.2006.01.003), muestreo perfecto.
- [Bubley y Dyer, «Faster random generation of linear extensions» (1999)](https://doi.org/10.1016/S0012-365X(98)00333-1), muestreo aproximadamente uniforme.
- [Jerrum, Valiant y Vazirani, «Random generation of combinatorial structures from a uniform distribution» (1986)](https://stat.duke.edu/~scs/Courses/Stat376/Papers/ConvergeRates/RandomizedAlgs/JerrumValiantVaziraniTCS1986.pdf), relaciones entre conteo aproximado y generación.
- [Inoue y Minato, «An Efficient Method for Indexing All Topological Orders of a Directed Graph» (2014)](https://doi.org/10.1007/978-3-319-13075-0_9), indexación de órdenes.
- [Frye, Rowat y Feige, «Asymmetric Shapley values» (NeurIPS 2020)](https://arxiv.org/abs/1910.06358), formulación original y aplicaciones.
- [Vignette de shapr: asymmetric y causal Shapley](https://cran.r-project.org/web/packages/shapr/vignettes/asymmetric_causal.html), contraste entre semánticas marginal, condicional e intervencional y baseline potencial sujeto a compatibilidad de órdenes.

## Investigación: qué mejorar y con qué benchmarks — 23 de noviembre de 2026

### Resumen para discutir con el equipo

**Recomendación:** orientar la extensión a una implementación **confiable en DAGs generales y evaluada con un protocolo reproducible**, y usar velocidad como resultado medido, no como promesa de antemano. Comparar ASV y Shapley con la misma función característica y sumar un caso interpretativo. Esto combina un avance práctico respecto del paper con una experimentación más convincente; la novedad concreta y el alcance deben acordarse con los coautores.

- **Más general:** el paper ya plantea el esquema aproximado para DAGs arbitrarios, pero su implementación y evaluación completa siguen por validar para ellos. No afirmar como novedad teórica algo que ya está en el paper. La implementación actual usa MCMC en el camino aproximado para DAGs no polyforest; es una aproximación heurística, no el sampler uniforme exacto requerido por la garantía original. La ruta exacta sigue especializada en árboles dirigidos.
- **Más confiable:** contrastar contra referencias exactas en casos pequeños, aclarar cuál es la distribución que se explica, comprobar probabilidades y dimensión del output, y dar diagnósticos si el grafo, la evidencia o la inferencia no son válidos.
- **Más rápida:** primero perfilar y luego optimizar el cuello de botella. Puede interesar disminuir tiempo total, responder muchas consultas o alcanzar cierto error antes; no son el mismo objetivo. Las mejoras aisladas de conteo no prueban una mejora de punta a punta.
- **Más útil:** mostrar de manera controlada cuándo el orden causal cambia la atribución, pero también sus limitaciones y sensibilidad a un grafo mal especificado. ASV condicional no es automáticamente una estimación de un efecto de intervención.

### Qué cubre el trabajo actual y qué falta

El [paper original](../JAIO/secciones/4_resultados_experimentales.tex) usa Cancer y Child modificada, datos simulados y árboles de decisión; también compara familias sintéticas. Child se transforma eliminando seis aristas antes de la evaluación. La tabla del paper da, para esa Child, 1,117 s para generar 1.000 órdenes y 10,8387 s para 30.000; son tiempos **de muestreo**, no de explicación end-to-end. Reporta alrededor de 8 % de error relativo promedio con 1.000 muestras, con dificultades para valores verdaderos pequeños. Parte del tiempo de enumeración se **extrapola**, no se mide por ejecución completa.

El paper sostiene que el sampler de grado acotado es más eficiente que alternativas generales, pero la sección experimental no muestra una comparación cronometrada directa con Huber. Conviene presentarlo como hipótesis hasta correr ambos métodos con el mismo protocolo, si se logra una implementación correcta de referencia.

El repositorio ya tiene [Cancer, Child y Student](../../networksExamples/) y resultados en [results](../../results/). Los CSV existentes son históricos: fueron generados antes de verificar los fixes de signo y normalización, y no deben tomarse como ground truth sin regenerarlos. [`writeASVAndShapleyIntoFile`](../../asvFormula/experiments.py) usa `shap.TreeExplainer` con datos de fondo, mientras ASV usa una esperanza condicional de la red. Además, `ASV` usa por defecto la clase predicha codificada one-hot y no necesariamente probabilidades del clasificador. Hay que alinear explícitamente distribución, función característica y output antes de interpretar una diferencia como efecto del orden. La [documentación de TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html) distingue opciones de dependencia y outputs; «interventional» en SHAP no debe confundirse sin más con aplicar `do` a nuestro SCM.

### Hallazgo: al eliminar el target puede cambiar la distribución

Antes, [`initializeDataAndRemoveVariable`](../../asvFormula/datasetManipulation.py) generaba datos con la red original y luego llamaba `BNmodel.remove_node(variableToPredict)` antes de hacer inferencia. En pgmpy 1.1.2 esto elimina la CPD del nodo y marginaliza las CPD de sus hijos **localmente**, lo que en general no equivale a marginalizar el target de la distribución conjunta original.

Lo comprobé con [`cancer.bif`](../../networksExamples/cancer.bif), el target `Smoker` usado en la configuración existente y `VariableElimination`: antes de eliminarlo, `P(Cancer=True)=0,01163`; después, `0,01745`. Esto no invalida automáticamente el algoritmo: indica que se entrenó con una distribución y se explicó usando otra, por lo que la interpretación y comparaciones requieren revisión. `Age` es hoja en Child original; allí quitarlo no modifica CPD de hijos, aunque la modificación previa de Child debe evaluarse por separado.

**Estado actual:** se conserva la red original para inferencia, sin usar el target como evidencia ni como input del predictor. El grafo de ASV contiene solo las features y conserva las precedencias a través de nodos eliminados; se consideran uniformemente los órdenes válidos de features, sin ponderarlos por su multiplicidad en la red completa. Una prueba sintética con target intermediario compara las expectativas y ASV contra enumeración. El cálculo exacto rechaza grafos que no sean bosques dirigidos con a lo sumo un padre por nodo; el aproximado usa MCMC para DAGs no polyforest, sin garantía de mezcla con los valores predeterminados. Sigue pendiente validar las redes experimentales y regenerar resultados históricos.

### Redes candidatas y forma de los grafos

Las redes discretas del [repositorio bnlearn](https://www.bnlearn.com/bnrepository/) incluyen estructura y CPD; [pgmpy](https://pgmpy.org/guides/example_models.html) también ofrece una API de modelos. Son buenas instancias estándar, pero una red bayesiana no es automáticamente un grafo causal validado. Si se usa como generador sintético, decirlo; si se hacen afirmaciones causales sobre datos reales, justificar estructura, variables y supuestos.

Medí los grafos **completos**, antes de quitar el target, con modelos `bnlearn` cargados por `pgmpy.example_models.load_model` (pgmpy 1.1.2 y NetworkX 3.7). `ciclos` es el número ciclomático del esqueleto no dirigido `m−n+componentes`; es cero exactamente cuando es bosque. `tw≤` es una **cota superior heurística**, no el treewidth exacto, calculada con `networkx.approximation.treewidth_min_fill_in` sobre el grafo moralizado. No confundirla con el treewidth del esqueleto usado en teoría de órdenes.

| Red | Nodos | Aristas | Ciclos | Máx. padres | Máx. grado esqueleto | tw moral ≤ | Utilidad |
|---|---:|---:|---:|---:|---:|---:|---|
| Cancer | 5 | 4 | 0 | 2 | 4 | 2 | Control pequeño e histórico; al quitar `Smoker` queda un árbol dirigido de 4 nodos. |
| Asia | 8 | 8 | 1 | 2 | 4 | 2 | Primer DAG no polyforest; pequeño para enumeración exacta. |
| Sachs | 11 | 17 | 8 | 3 | 7 | 3 | Caso biológico; contraste entre modelo de benchmark y estudio causal original. |
| Child original | 20 | 25 | 6 | 2 | 8 | 3 | Comparación con Child modificada; al quitar `Age` quedan 19 nodos, 23 aristas y 5 ciclos. |
| Alarm | 37 | 46 | 10 | 4 | 6 | 4 | Escala intermedia, monitorización médica. |
| Insurance | 27 | 52 | 26 | 3 | 9 | 7 | Caso más conectado y con inferencia potencialmente más exigente. |

Fuentes de los tamaños y dominios: [bnlearn: redes pequeñas](https://www.bnlearn.com/bnrepository/discrete-small.html) y [medianas](https://www.bnlearn.com/bnrepository/discrete-medium.html). Sachs se vincula al trabajo de [Sachs et al.](https://doi.org/10.1126/science.1105809), que utilizó perturbaciones experimentales. El BIF discreto de benchmark no debe confundirse sin verificación con los datos originales ni tratarse como verdad causal indiscutible.

**No encontré evidencia suficiente para decir que una forma de grafo predomina en todos los dominios causales.** Estos ejemplos son una muestra seleccionada, no representativa. Aun así, ilustran que pocos padres por nodo **no implica** ser polytree; el grado máximo, ciclos, componentes, cantidad de órdenes, ancho y la complejidad de inferencia pueden afectar costos de formas distintas. Para cada target registrar las métricas del **grafo efectivamente usado**, no solo de la red completa. Incluir también cadenas, estrellas, árboles balanceados, colliders y diamonds sintéticos para aislar los factores estructurales.

### Medición exploratoria nueva: solo generación MCMC

En Windows, Intel i5-7500 3,40 GHz, Python 3.14.5, NetworkX 3.7 y pgmpy 1.1.2, medí el MCMC existente sobre los grafos completos de arriba. Son **3 corridas** de 100 órdenes, semillas de `random` 0, 1 y 2; el tiempo con `perf_counter` incluye cierre transitivo, burn-in y generación. Cada muestra pasó la verificación de aristas del orden topológico. Los pasos de burn-in y entre muestras usan el default `n³`. Sin controlar carga del sistema; es un sondeo local, no un benchmark publicable ni tiempo total de ASV.

| Red | Tres tiempos (s) | Mediana (s) |
|---|---|---:|
| Cancer | 0,0118; 0,0132; 0,0110 | 0,0118 |
| Asia | 0,0403; 0,0404; 0,0384 | 0,0403 |
| Sachs | 0,1265; 0,1137; 0,1103 | 0,1137 |
| Child original | 0,7099; 0,7701; 0,7012 | 0,7099 |
| Alarm | 4,1131; 4,1304; 5,6146 | 4,1304 |
| Insurance | 1,4680; 1,4377; 1,4231 | 1,4377 |

La validez de los órdenes **no demuestra** mezcla ni uniformidad. No se midió error de ASV, inferencia, agrupación ni costo por todas las features; tampoco se comparó contra el sampler ponderado, que no admite los DAGs no polyforest. El costo de MCMC varía con `n³` y la cantidad de muestras por construcción; no extrapolar estos datos como tiempo final ni como comparación controlada con el paper.

Para repetir el sondeo, con versiones compatibles instaladas, ejecutar desde la raíz:

```python
import random, statistics, time
import networkx as nx
from pgmpy.example_models import load_model
from asvFormula.topoSorts.randomTopoSortsGeneration import mcmcTopoSorts
from asvFormula.topoSorts.utils import isTopologicalSort

for name in ("cancer", "asia", "sachs", "child", "alarm", "insurance"):
    model = load_model("bnlearn/" + name)
    dag = nx.DiGraph(model.edges())
    dag.add_nodes_from(model.nodes())
    times = []
    for seed in range(3):
        random.seed(seed)
        start = time.perf_counter()
        orders = mcmcTopoSorts(dag, 100)
        times.append(time.perf_counter() - start)
        assert len(orders) == 100
        assert all(isTopologicalSort(dag, order) for order in orders)
    print(name, times, statistics.median(times))
```

La API puede descargar modelos al primer uso; para un experimento final fijar versiones y guardar la versión y hash de cada BIF/CPD, ya que el contenido remoto puede cambiar.

### ¿300 segundos a 30?

Es una mejora de 10×, pero **no hay todavía un caso medido aquí que justifique adoptarla como requisito**. Para una explicación ocasional, cinco minutos podrían ser aceptables; para un panel interactivo, explicar miles de registros, repetir muchos modelos o explorar grafos, podrían no serlo. El presupuesto debe especificar por instancia, feature, output y volumen de consultas, incluyendo preparación y reutilización de caché. Sin esos datos, ni 30 ni 300 segundos son una meta universal.

Propuesta para comparar sin decidir prematuramente un umbral:

1. Medir tiempos de preparación, generación, agrupación, consultas del oráculo y total; distinguir primera ejecución y reutilización, una feature y todas las features, además de memoria.
2. Elegir instancias y salidas concretas; informar mediana y dispersión con varias semillas/réplicas, hardware, límites de tiempo y fallos. Los *timeouts* se reportan, no se descartan.
3. En grafos pequeños calcular referencia exacta; en medianos, usarla solo cuando sea factible, o una referencia aproximada independiente y más costosa, señalando su incertidumbre. Comparar **error absoluto/RMSE y tiempo**, no solo muestras por segundo; reportar errores relativos con denominador cercano a cero por separado.
4. Proponer umbrales de error y tiempo para cada caso de uso **después** de obtener la línea de base. Si un caso tarda aproximadamente 300 s, evaluar 30 s como objetivo aspiracional a igual precisión y semántica, no como resultado asumido. Para MCMC estudiar burn-in, autocorrelación y múltiples cadenas; más muestras correlacionadas no implican igual precisión que muestras independientes.

### Plan experimental concreto

1. **Corrección antes que escala:** probar Cancer, Asia, diamond y otras redes pequeñas contra enumeración de órdenes y esperanza condicional exacta para todos los subconjuntos necesarios. Controlar suma de atribuciones, clases y vectores; usar un modelo e instancia fijos. Auditar especialmente eliminación del target y valores con probabilidad cero.
2. **Ejemplos explicativos:** ejecutar el [ejemplo sintético](../../experiments/synthetic_asv_example.py) ya implementado y extender con grafo vacío, cadena con ruido, causa común y collider. Conservar modelo, distribución, instancia y `v(S)` al comparar SHAP simétrico con ASV; variar el grafo supuesto por separado y reportar diferencias y limitaciones.
3. **Escala en capas:** Cancer/Child modificada como continuidad histórica; Asia/Sachs/Child original para DAGs con ciclos en el esqueleto; Alarm/Insurance como pruebas más exigentes. Para la ruta exacta usar únicamente su dominio comprobado. Para DAGs generales usar la aproximación validada, sin prometer que todas las redes terminan dentro de un plazo.
4. **Baselines justos:** enumeración exhaustiva donde sea posible; sampler ponderado en polyforests; MCMC en todos los DAGs válidos; Atkinson como alternativa experimental solo en su dominio. Comparar el mismo modelo, output y distribución; medir coste completo y precisión. [shapr](https://cran.r-project.org/web/packages/shapr/vignettes/asymmetric_causal.html) es un candidato externo para órdenes compatibles: verificar primero que represente exactamente las mismas restricciones y pesos, además de la misma `v(S)`. [Heskes et al.](https://proceedings.neurips.cc/paper/2020/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html) es una referencia de valores causales intervencionales, no un baseline numéricamente equivalente por defecto.
5. **Optimizar según perfil:** reutilización segura de coaliciones y consultas entre features, mejoras al sampler o inferencia; medir por separado y luego end-to-end. Repetir Atkinson contra el original con muestras y precisión comparables antes de decidir si reemplazarlo.

Preguntas para acordar: ¿uso individual offline o lotes de explicaciones?, ¿qué precisión y recursos son aceptables?, ¿un caso biológico aplicado con supuestos sustentados o solo proof of concept sintético?, ¿la contribución principal será robustez/escala o un resultado algorítmico nuevo? No se asumieron respuestas en este documento.
