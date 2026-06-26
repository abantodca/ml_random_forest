# Sustento estadístico — Ruteo de variedades a anclas

> **Documento de respaldo metodológico** para el experimento
> [`experiment_variety_anchor_routing.ipynb`](experiment_variety_anchor_routing.ipynb).
> Audiencia: gerencia + revisión técnica. Explica **qué se decide, por qué la
> decisión es estadísticamente defendible, y cuáles son sus límites honestos.**
>
> Cifras: corrida del **2026-06-25** sobre `data/training/DB-HISTORICA.xlsx`
> (23 variedades, ~22.000 filas). Reproducibles con los artefactos en `data/`.

---

## 1. Resumen ejecutivo

**El problema.** Tenemos ~23 variedades de arándano, pero la data está muy
concentrada: **3 variedades (POP, BEAUTY, VENTURA) reúnen ~46% de todas las filas**
y la cola larga tiene variedades con apenas 35–80 registros. Una variedad con tan
poca data **no puede entrenar un modelo de pronóstico propio confiable** — el modelo
memoriza ruido en lugar de aprender la relación real.

**La decisión.** Para cada variedad respondemos una sola pregunta operativa:

> ¿Entrena su **modelo propio** (es "ancla") o **hereda** el modelo de otra variedad
> que la pronostica bien?

**El resultado (data de hoy):** **11 anclas** (modelo propio) + **12 variedades
ruteadas** a un ancla. **20 de 23 (87%)** quedan en una categoría confiable
(*ancla / bueno / aceptable*); **2** se marcan para revisión y **1** es un caso
atípico (ATLAS, ver §7).

**El nivel de confianza, honesto.** La **metodología es sólida y conservadora**: usa
tamaño de efecto en vez de p-valores, valida fuera de muestra, corrige por
comparaciones múltiples y es explícita sobre lo que *no* puede afirmar. Pero la
corrida actual tiene **tres asuntos de calibración** (§7) que conviene cerrar antes
de tomar las categorías al pie de la letra. **La dirección de la decisión es
correcta; los umbrales finos requieren un ajuste.**

---

## 2. El problema de negocio y por qué importa estadísticamente

El sistema pronostica **productividad de cosecha** (`KG/JR_H`, kg por jornal-hora)
por variedad. Cada modelo necesita data suficiente para estimar de forma estable la
relación `features → productividad`.

| Grupo | Variedades | Data | Qué pueden soportar |
|---|---|---|---|
| Cabeza | POP, BEAUTY, VENTURA, BIANCA, ATLAS… | cientos a miles de filas | Modelo propio con validación cruzada completa |
| Cola larga | STELLA (35), MAGNUS (37), BONITA (39)… | decenas de filas | **No** alcanzan para un modelo propio fiable |

**Por qué importa:** entrenar un modelo sobre 35 filas produce pronósticos con
varianza enorme — parecen buenos en la data que vieron y fallan en producción
(*sobreajuste*). La estadística clásica de esto es directa: la incertidumbre de una
estimación escala con `1/√n`. Pasar de n=35 a n=2.000 reduce el error de estimación
~8×. Por eso la cola larga **hereda** un modelo entrenado con suficiente data.

---

## 3. La decisión metodológica clave: esto **no es clustering**

Es el corazón del documento y la pregunta más probable de un revisor técnico.

La tentación natural es "agrupar variedades parecidas" (clustering). **Lo
descartamos a propósito, con evidencia.** Son dos preguntas estadísticas distintas:

| | **Clustering** (lo que NO hacemos) | **Ruteo a donante** (lo que SÍ hacemos) |
|---|---|---|
| Pregunta | ¿Hay grupos *naturales* de variedades? | ¿Qué modelo **pronostica** mejor a cada variedad chica? |
| Etiqueta | No existe; se descubre | Existe: cada variedad ya tiene su `KG/JR_H` |
| Validación correcta | Silhouette, estabilidad de grupos | **Error de pronóstico fuera de muestra (MAPE OOS)** |

**Por qué la distinción decide todo:** dos variedades pueden tener **distribuciones
distintas** y aun así la **misma relación features → productividad**. En ese caso
comparten modelo sin problema, aunque un clustering las separaría. Lo único que
importa para compartir modelo es si el modelo del ancla **pronostica bien** la otra
variedad — no si "se parecen".

**Y la evidencia confirma que no hay clusters naturales:** corrimos la batería
estándar de validez de clusters sobre las ~16.000 observaciones escaladas:

- **Barrido de k (silhouette):** la única estructura con señal positiva está en
  **k=2–3** (~+0.33). Con las 11 anclas el silhouette es **negativo** → forzar 11
  grupos sobre-particiona un continuo.
- **HDBSCAN (densidad):** clasifica ~100% como ruido → no hay grumos discretos, es
  un **continuo de productividad**.

> **Conclusión presentable a gerencia:** "Probamos si las variedades formaban grupos
> naturales y **estadísticamente no los forman** — son un continuo. Por eso no
> inventamos grupos artificiales; en su lugar medimos, variedad por variedad, **qué
> modelo la pronostica mejor**. Es una decisión basada en error de pronóstico real,
> no en una apariencia de similitud."

Esta honestidad es una **fortaleza**: resistimos la tentación de presentar 11
"clusters verdes" que serían estadísticamente falsos.

---

## 4. La metodología paso a paso y su sustento

El pipeline tiene dos capas. Las secciones 2–5 del notebook son un **diagnóstico de
soporte** (¿tiene sentido la cercanía?); la sección **6/7 es la decisión accionable**
(¿quién pronostica mejor?).

### 4.1 Decisión final — Error predictivo fuera de muestra (MAPE OOS)

Es **la única capa que decide**. Para cada variedad chica:

1. Se entrena un modelo en **cada** ancla (HistGradientBoosting sobre 4 features).
2. Se mide el **MAPE** (error porcentual absoluto medio) de cada modelo-ancla al
   pronosticar la variedad chica. El ancla **nunca vio** esa data → el error es
   **fuera de muestra (honesto)**.
3. La variedad se rutea al ancla de **menor MAPE OOS**.

**Por qué es la prueba correcta:** mide exactamente lo que se quiere lograr —
pronosticar bien — en vez de un proxy de similitud. Es el principio de oro del ML:
**validar contra el objetivo real, fuera de muestra.**

La columna **Ganancia** = `MAPE_propio − MAPE_OOS` cuantifica, en puntos de error,
cuánto se gana (o pierde) al heredar en vez de entrenar propio. Es el número que
traduce la decisión a impacto medible.

### 4.2 Diagnóstico de soporte — por qué cada técnica está bien elegida

| Técnica | Qué mide | Por qué es la correcta aquí |
|---|---|---|
| **RobustScaler** (mediana/IQR) | Normaliza features | Resistente a outliers, a diferencia de estandarizar por media/desv (que un outlier distorsiona) |
| **IsolationForest** | Filtra outliers por variedad | Wasserstein es sensible a extremos; se limpia 2–5% por variedad |
| **Distancia de Wasserstein-1** | Diferencia entre *distribuciones completas* | Captura forma y varianza, no solo la media; estable con pocos datos; en unidades originales (interpretable) |
| **Cliff's delta** (tamaño de efecto) | *Cuánto* difieren dos variedades | No paramétrico, robusto; mide magnitud, no solo significancia |
| **Bootstrap (100×)** | Fragilidad de cada asignación | Convierte una etiqueta ("ancla X") en una **probabilidad** ("85% de las veces cae en X") |
| **Holm-Bonferroni** | Corrige comparaciones múltiples | Con 5 features, el riesgo de un falso positivo sube a ~23%; Holm lo controla al 5% |
| **Silhouette sobre todas las obs** | ¿Hay estructura real? | Más potente y **no circular** vs. el silhouette sobre 1 centroide por variedad |

### 4.3 La decisión estadística más importante: **efecto, no p-valor**

> Con n grande (miles de filas), **cualquier** test sale "significativo" (p < 0.05).
> Es la *trampa de la gran muestra*. Si decidiéramos por p-valores, **todo** se vería
> "estadísticamente diferente" y la herramienta sería inútil.

Por eso la decisión usa **tamaño de efecto** (Cliff's delta) y **error de pronóstico
real** (MAPE), que miden *magnitud accionable*, no mera significancia. El p-valor
(Mann-Whitney) se reporta como **informativo**, no decide. Esto es exactamente lo que
recomienda la literatura estadística moderna (ASA, 2016, sobre el mal uso del
p-valor) y es un punto fuerte defendible ante cualquier revisor.

---

## 5. Resultados sobre la data actual

**Mapping generado** (`variety_predictive_routing.yaml`):

| Decisión | Criterio | # | Variedades |
|---|---|---|---|
| 🔵 **ancla** | modelo propio | 11 | POP, BEAUTY, VENTURA, BIANCA, ATLAS, JUPITER, MAGICA, KIRRA, EMERALD, ROSITA, BILOXI |
| 🟢 **bueno** | MAPE ≤ 1.5× baseline | 2 | STELLA→POP, MALIBU→POP |
| 🟢 **aceptable** | ≤ 2.0× baseline | 8 | MASIRAH→VENTURA, BELLA→JUPITER, MADEIRA→BEAUTY, RAYMI→POP, FCM17-132→KIRRA, AZRA→POP, ARANA→JUPITER, BONITA→BILOXI |
| 🟡 **revisar** | > 2.0× | 2 | MAGNUS→MAGICA, TERRAPIN→JUPITER |

**KPIs:** 20/23 confiables · MAPE típico de un ancla (baseline) ≈ **14.5%** · ganancia
media al heredar ≈ **+8 pp** de MAPE en las variedades chicas con poca data.

**Casos donde heredar claramente gana** (la tesis del proyecto, confirmada):

| Variedad | n | MAPE propio | MAPE heredado | Ganancia |
|---|---|---|---|---|
| TERRAPIN | 60 | 97.2% | 29.7% | **+67.5 pp** |
| STELLA | 35 | 43.0% | 21.3% | **+21.8 pp** |
| MAGNUS | 37 | 41.8% | 29.2% | **+12.7 pp** |
| FCM17-132 | 48 | 35.2% | 26.0% | **+9.2 pp** |

Estas variedades con su propio modelo pronostican pésimo (MAPE 35–97%); heredando, el
error cae a la mitad o menos. **Ese es el valor concreto del enfoque.**

---

## 6. Fortalezas estadísticas (lo que está bien hecho)

1. **Valida contra el objetivo real, fuera de muestra.** La decisión es MAPE OOS, no
   un proxy. Es metodológicamente impecable.
2. **Efecto > p-valor.** Evita la trampa de la gran muestra; alineado con la guía
   estadística moderna.
3. **Honestidad sobre la estructura.** Demuestra con datos que **no** hay clusters y
   se niega a fabricarlos. Resiste circularidad: descarta los tests (Ratio, PERMANOVA)
   que serían tautológicos y solo cuenta el silhouette no circular.
4. **Corrección por comparaciones múltiples** (Holm-Bonferroni) y **cuantificación de
   fragilidad** (bootstrap) — rigor que rara vez se ve en un experimento interno.
5. **Reproducible y auditable:** semillas fijas, lógica en módulo testeable
   (`variety_routing.py`), artefactos versionables.

---

## 7. Limitaciones y riesgos conocidos (lectura crítica honesta)

Un documento para gerencia debe declarar lo que **aún no** está cerrado. Estos tres
puntos **no invalidan la dirección** de la decisión, pero **sí afectan las categorías
finas** y deben resolverse antes de "congelar" el mapping.

### 7.1 El baseline está contaminado por ATLAS (impacto alto)

El umbral de todas las categorías es `ratio = MAPE_OOS / baseline`, donde el baseline
es el **MAPE promedio de las 11 anclas**. Pero **ATLAS se autopronostica con MAPE
75.6%** — un caso anómalo frente a las otras 10 anclas (4–14%).

- Baseline **con** ATLAS ≈ **14.5%**. Baseline **sin** ATLAS ≈ **8.4%**.
- El outlier **infla el baseline ~6 pp**, lo que **abarata artificialmente todos los
  ratios** y hace que más variedades caigan en "aceptable" en vez de "revisar".

> **Riesgo:** con un baseline honesto (~8.4%), TERRAPIN pasaría de `2.05×` a `~3.5×`
> y varias "aceptable" se reclasificarían. **Las categorías actuales son
> optimistas.** Acción: tratar ATLAS como caso aparte (no es un buen donante de
> nadie) y recalcular el baseline robusto (mediana o promedio recortado).

### 7.2 El baseline es *dentro* de muestra; el ratio compara peras con manzanas (impacto medio)

El MAPE OOS de la variedad heredada es **fuera de muestra** (honesto), pero el MAPE
propio del ancla (`anchor_own`, que forma el baseline) es **dentro de muestra** (el
ancla se evalúa sobre la misma data con que se entrenó → optimista). El ratio divide
un error honesto entre un baseline optimista. Acción: medir el baseline del ancla
también por **validación cruzada**, para comparar lo comparable.

### 7.3 El "prior" valida una partición distinta de la que decide (impacto medio)

Las secciones 2–5 (Wasserstein, Cliff, bootstrap, Holm, silhouette) se corren con un
set de anclas (`config.anchor_varieties`) que **incluye BELLA y ARANA y excluye KIRRA
y BILOXI**. Pero la decisión final (sección 7) usa el set inverso: **BELLA y ARANA se
rutean, y KIRRA y BILOXI son anclas.**

> Consecuencia: todo el diagnóstico de soporte (estabilidad bootstrap, efecto)
> describe **una partición que la decisión no usa**. No es un error de cálculo, pero
> rompe la narrativa "las secciones 2–5 son el *prior* de la 7". Acción: unificar un
> único set de anclas y correr ambas capas sobre él.

### 7.4 Señales que el propio resultado ya delata

- **Ganancias negativas:** BELLA (−3.1 pp), RAYMI (−2.5 pp), MALIBU (−2.8 pp), ARANA
  (−0.8 pp) pronostican **mejor con su propio modelo** que heredando. Con n=195–311
  (BELLA, ARANA, RAYMI) **tienen data suficiente para ser anclas** — coherente con que
  el `config` original las trataba como anclas. **Recomendación: BELLA y ARANA
  deberían entrenar modelo propio**, no heredar.
- **Pronóstico sobre pocos puntos:** STELLA (35), MAGNUS (37), BONITA (39) deciden su
  ruteo con un MAPE OOS medido sobre decenas de filas → estimación ruidosa. La
  dirección (heredar) es correcta, pero *a qué ancla* puede variar.

### 7.5 Features proxy (limitación declarada, impacto a verificar)

El ruteo usa 4 features y **no** las *lag features* del pipeline real de producción.
El notebook lo declara ("con features reales el MAPE baja"), pero **no se ha
verificado que el *ranking* de anclas (quién gana) se mantenga** bajo el modelo real.
La decisión es un ranking; su estabilidad bajo las features de producción está
**pendiente de validar**.

---

## 8. Recomendaciones

| Prioridad | Acción | Cierra |
|---|---|---|
| **Alta** | Excluir ATLAS del cálculo de baseline (o usar mediana/promedio recortado) y recategorizar | §7.1 |
| **Alta** | Unificar un único set de anclas para prior y decisión | §7.3 |
| **Alta** | Reclasificar BELLA y ARANA como anclas (ganancia negativa + n suficiente) | §7.4 |
| Media | Medir el baseline del ancla por validación cruzada (comparar OOS con OOS) | §7.2 |
| Media | Reportar bandas de incertidumbre (CV / bootstrap) sobre el MAPE OOS de cada ruteo | §7.4 |
| Media | Re-correr el ruteo con las *lag features* reales y verificar que el ranking se mantiene | §7.5 |

Ninguna recomendación cambia la **tesis**: las variedades de cola larga deben heredar.
Cambian **cuáles** son anclas y **qué tan estricto** es el corte de categorías.

---

## 9. Glosario para no-estadísticos

- **MAPE (error porcentual absoluto medio):** en promedio, ¿en qué % se equivoca el
  pronóstico? MAPE 10% = el pronóstico erra ~10% del valor real. Menos es mejor.
- **Fuera de muestra (OOS):** evaluar un modelo con datos que **no** usó para
  entrenar. Es la única forma honesta de saber si funcionará en producción.
- **Tamaño de efecto vs. p-valor:** el p-valor dice "¿hay *alguna* diferencia?"; el
  tamaño de efecto dice "¿*cuánta*?". Con mucha data, lo primero siempre dice "sí",
  por eso usamos lo segundo.
- **Ancla:** variedad con data suficiente que entrena su propio modelo y puede
  prestárselo a otras.
- **Bootstrap:** repetir el cálculo con remuestreos de los datos para ver qué tan
  estable es el resultado (lo convierte de "sí/no" en "% de confianza").

---

## 10. Apéndice técnico

- **Umbrales de Cliff's delta** (Romano et al.): |d|<0.147 nulo · <0.33 pequeño ·
  <0.474 mediano · resto grande.
- **FWER de Mann-Whitney sin corregir:** 1 − 0.95⁵ ≈ 0.226 (23%) → Holm-Bonferroni
  step-down lo devuelve al 5%.
- **Reducción efectiva de muestra** (split + nested-CV + outliers): ver
  `ExperimentConfig.reduction_factor` — define los umbrales de viabilidad por filas.
- **Modelo del ruteo:** `HistGradientBoostingRegressor`, 200 iter (anclas) / 120 iter
  (CV propio), `random_state=123`, KFold 5-fold para el MAPE propio.
- **Código fuente auditable:** [`variety_routing.py`](variety_routing.py) — toda la
  lógica estadística, importable y con tests P0.
- **Referencia metodológica:** Wasserman, *All of Statistics*; ASA Statement on
  p-Values (2016); Hennig, *Cluster-wise assessment of cluster stability* (2007).
