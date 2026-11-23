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

Hallazgos del código actual, a confirmar con pruebas ejecutables antes de comparar resultados:

- [`ApproximateASV`](../../asvFormula/ASV.py) llama al sampler ponderado `randomTopoSorts`, no a `mcmcTopoSorts` como indica el README. Su método `asssertToposortsMatches` tiene una `s` de más y no sobrescribe `assertToposortsMatches`; la clase base compara el número de muestras contra el conteo del grafo. Además, `allForestTopoSorts` está diseñado para bosques dirigidos, no DAGs arbitrarios. Es una limitación práctica, aunque la idea matemática sea más general.
- La acumulación en `asvForFeature` usa `v(S) - v(S ∪ {x})`, al revés de la definición del paper. Hay que validar el signo, la dimensión del vector de salida y la propiedad de eficiencia contra enumeración exacta antes de usar los CSV existentes como verdad de referencia.
- [`randomTopoSortsGeneration.py`](../../asvFormula/topoSorts/randomTopoSortsGeneration.py) contiene tanto muestreo ponderado por conteos para polytrees como una cadena MCMC de intercambios adyacentes para DAGs. La cadena no implementa el muestreo perfecto de Huber 2006. El valor `n³` predeterminado es una elección de implementación, no una garantía demostrada de error o independencia para esta cadena concreta. La prueba existente de proporción de órdenes distintos no alcanza para demostrar uniformidad.
- [`topoSortsCalc.py`](../../asvFormula/topoSorts/topoSortsCalc.py) enumera productos y permutaciones de vecinos; el grado puede afectar fuertemente su costo. El conteo y el sampler deben medirse por separado.
- Los experimentos del [paper original](../JAIO/secciones/4_resultados_experimentales.tex) usan Cancer y una Child modificada para que sea polytree, y parte del tiempo de enumeración es extrapolado. El código de comparación usa `shap.TreeExplainer` con datos de fondo, mientras ASV usa expectativas condicionales de una red bayesiana: hay que verificar la semántica y el output de cada baseline antes de atribuir diferencias exclusivamente al orden causal.
- El enunciado y la demostración de la cota de sampleo en el paper y el suplemento usan parámetros distintos. Además, Hoeffding requiere las hipótesis de independencia apropiadas; no se puede trasladar automáticamente a muestras consecutivas correlacionadas de MCMC. Revisar también las hipótesis de acotación y la precisión del oráculo.

La suite no se pudo recolectar en este entorno: `python -m pytest asvFormula\tests\ --collect-only -q` invoca el alias de Microsoft Store y no hay un intérprete de Python disponible en PATH. No se verificaron los hallazgos de código mediante ejecución ni se ejecutaron nuevos experimentos.

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

### Próximos hitos sugeridos

- [ ] Elegir con el equipo la conferencia, extensión y contribución principal.
- [ ] Configurar Python y dependencias; establecer una suite reproducible y un baseline exacto pequeño.
- [ ] Corregir y probar las discrepancias detectadas en ASV y muestreo, sin modificar silenciosamente la interpretación de resultados previos.
- [ ] Prototipar Atkinson y benchmarkear conteo por grado y tamaño; decidir si integrarlo al sampler.
- [ ] Implementar el ejemplo sintético analítico y variantes, comparando con la misma función característica.
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
