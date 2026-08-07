# Auditoría ML y plan de reducción de error — 2026-08-07

Auditoría de `src/`, de las guías y del estado del repo. Cada hallazgo trae **evidencia
verificable** (archivo:línea o comando), su **efecto esperado sobre el error**, y una
**acción**. Nada de lo implementado cambia el comportamiento por defecto: todas las
palancas nuevas son opt-in, como el resto del proyecto.

Complementa —no reemplaza— a `docs/04-guia-validacion-estadistica.md`, que fija la
política de validación. Esta guía dice *qué está mal hoy y en qué orden atacarlo*.

---

## 1. Estado del repo (corregido en esta auditoría)

| # | Hallazgo | Evidencia | Estado |
|---|---|---|---|
| 1.1 | `docs/adr/` **no existía**: los 9 ADR + su README se borraron en `b7bf8b1`, dejando 56 citas colgadas —incluidas `src/config.py` y `src/step_06_track/mlflow_registry.py`— y una guía que afirmaba *"los ADR **ya existen** en `docs/adr/`"* | `git show --stat b7bf8b1`; `docs/02-produccion-aws.md` #7958 | **Restaurados** desde `b7bf8b1^` |
| 1.2 | Las 3 guías se renombraron a `*-mejorado.md`, rompiendo **30 referencias** (README, CLAUDE.md, CONTRIBUTING, Taskfile, `docker-compose.yml`, `src/config.py`, `tasks/*.yml`) y **todos** los links relativos internos entre guías | `git log --diff-filter=R -- docs/` | **Renombradas** a los nombres canónicos (`01-local.md`, `02-produccion-aws.md`, `03-arquitectura.md`); 0 links rotos |
| 1.3 | `CLAUDE.md` y `CONTRIBUTING.md` afirmaban *"no hay suite de tests committeada"* cuando sí la había (10 archivos / 791 líneas, `testpaths` declarado) | `git ls-files tests/` | **Resuelto por decisión del owner (2026-08-07)**: se retiraron `tests/` y `api/tests/`, más `pytest` de `pyproject.toml` y `requirements-dev.txt`. Las pruebas se pegan desde la guía. La afirmación de las guías vuelve a ser cierta y ADR-008 queda vigente |
| 1.4 | `tasks/` (10 archivos con los includes del Taskfile) | decisión del owner | **Eliminado.** ⚠ `Taskfile.yml` conserva su bloque `includes:` apuntando a `./tasks/*.yml`, así que **todo comando `task` falla** hasta que se pegue de vuelta. Las guías mencionan esos archivos pero **no contienen su contenido**: la única copia está en el historial git (`git show HEAD:tasks/infra.yml`) |
| 1.5 | Código de producción fuera de lugar en la raíz | — | **No existe.** La raíz solo tiene entrypoints legítimos (`main.py` = CLI del trainer, `Dockerfile` = imagen del trainer, `Taskfile.yml`/`docker-compose.yml` = orquestación). No hay `.tf` ni `infra/`: el Terraform es pegable desde `docs/02-produccion-aws.md` #3.11, como está diseñado. Solo se limpiaron cachés regenerables (`__pycache__/`, `.pytest_cache/`) y un `scratchpad/` vacío |

| 1.6 | 7 citas colgadas a `docs/PLAN_REFACTOR_2026-06-12.md` y `docs/REFACTOR_ARQUITECTURA_2026-06-23.md`, borrados en `c7e61a4`. Seis viven dentro de ADRs —registros inmutables, no se reescriben— y una en `src/variety_config.py:1` | `git show --stat c7e61a4` | **Restaurados** desde `c7e61a4^` con banner *DOCUMENTO HISTÓRICO* que remite a las guías vigentes |
| 1.7 | `config.py` presentaba `ENABLE_EXTRA_CATEGORICALS` como si bastara encender la flag, cuando los datos no traen las columnas (ver #2.4) | lectura de ambos workbooks | **Comentario corregido** con el diagnóstico y los pasos para desbloquearlo |
| 1.8 | El `.venv` del proyecto apuntaba a un intérprete inexistente y `.ruff_cache/0.15.13` era `root:root` (escrito por el contenedor) → `task lint` fallaba con *Permission denied* | `cat .venv/pyvenv.cfg`; `stat .ruff_cache/*` | **`.venv` recreado** (py3.12 + deps pineadas + `api/requirements.txt`); dir root apartado por rename, `task lint` operativo |

**Verificación de que nada se rompió.** Se comparó un entrenamiento real (nested CV sobre
ROSITA, ambos backends) entre `HEAD` prístino y el árbol modificado: `best_params`, todas
las métricas, el hash de las predicciones OOF y el del modelo final salen **idénticos byte
a byte**. Con los defaults el sistema entrena exactamente igual que antes.

---

## 2. Hallazgos de modelado

Ordenados por **efecto esperado sobre el error / coste de implementar**.

### 2.1 El modelo de producción nunca entrena con el ~10% de datos más recientes

**El hallazgo.** `data_loader` deja el dataset ordenado cronológicamente
(`data_loader.py`, `sort_values(DATE_COLUMN)`). El early stopping recorta como holdout el
**tramo final** de ese orden (`validation_split.py::temporal_tail_holdout_indices`) y
**nunca lo devuelve al fit**: no hay ningún refit posterior en todo `src/`
(`grep -rn "best_iteration" src/` lo confirma). Como los K pipelines del
`OOFEnsembleRegressor` heredan cada uno ese recorte, el modelo que se registra en MLflow
jamás ha visto las observaciones más recientes.

**Por qué importa estadísticamente.** Esas filas no son un holdout honesto de evaluación
—la evaluación honesta ya vive en el outer CV— sino **dato de entrenamiento descartado**.
Y no es dato cualquiera: en un problema de pronóstico agronómico con drift documentado
(`temporal_mape_oof` 22-34% vs 13-15% estratificado, ver `config.py`
`CHAMPION_WARN_TEMPORAL_MAPE`) las observaciones recientes son las de mayor peso
informativo sobre la campaña siguiente. Se está pagando dos veces: menos datos **y** los
datos que más importan.

**Acción — implementada, opt-in.** `EARLY_STOPPING_REFIT_FULL=1` (default `0`) convierte
el fit en dos etapas, que es la práctica estándar:

1. fit con `eval_set` sobre la cola temporal → descubre `best_iteration`;
2. refit sobre el **100%** de las filas con `n_estimators = best_iteration × n/(n−n_val)`,
   capado a `N_ESTIMATORS_MAX`, sin `eval_set`.

El escalado `n/(n−n_val)` reconoce que más datos admiten algo más de árboles antes de
memorizar. Coste: ~2× el tiempo por fit. `src/step_04_train/early_stopping.py`.

> Esta es la palanca con mejor relación efecto/riesgo de la lista. Empezar por acá.

### 2.2 Optuna optimiza MAE; el campeón se decide y se bloquea por MAPE

**El hallazgo.** `_objective` minimiza MAE en espacio original
(`tuning.py`, `mean_absolute_error(yv, ...)`), pero:

- `champion.select_champion` desempata por `oof_mape` (`champion.py`, `_decision_key`);
- el quality gate registra o rechaza según `CHAMPION_MAX_MAPE` sobre `oof_mape`.

**Por qué no es lo mismo.** Sobre un target derecho-sesgado —el que justifica el `log1p`
del TTR, el cap p99.5 y todo el bloque `SKEW_*` de `config.py`— los dos óptimos divergen
sistemáticamente:

| | domina | un error de 2 kg sobre… |
|---|---|---|
| **MAE** | filas de `y` **alto** | `y=30` pesa igual que sobre `y=3` |
| **MAPE** | filas de `y` **bajo** | `y=30` → 6.7%; `y=3` → **67%** |

Es decir: el TPE está eligiendo hiperparámetros contra una función objetivo distinta de la
que después juzga —y descarta— al modelo. Cualquier presupuesto extra de trials refina la
respuesta a la pregunta equivocada.

**Acción — implementada, opt-in.** `OPTUNA_OBJECTIVE_METRIC` (default `mae`,
bit-idéntico):

- `mape` — alineación directa con el gate. Usa piso **relativo** de denominador
  (`TEMPORAL_MAPE_REL_FLOOR × mediana|y|` del fold), la misma protección anti-artefacto de
  `_temporal_honesty_check` (el piso físico `MAPE_MIN_DENOM=1.0` está calibrado para KG/JR,
  no para KG/JR_H que vive en escala ~3).
- `mae_log` — MAE sobre `log1p(y)`. Es **exactamente la loss que los árboles ya minimizan
  dentro del TTR**, penaliza error relativo y no se desestabiliza cuando `y→0`. La opción
  robusta si el MAPE resulta ruidoso entre folds.

`src/config.py` + `tuning.py::_fold_score`. El gap de la penalización opcional usa la misma
métrica (restar MAE de MAPE no tiene sentido dimensional).

> Ortogonal a `SAMPLE_WEIGHT_INV_Y`, que ataca lo mismo por el lado de los pesos. **A/B por
> separado** para poder atribuir el efecto.

### 2.3 `final_holdout_*` no es un holdout virgen

**El hallazgo.** El comentario decía *"holdout final intocable: no participa en Optuna,
selección de params ni calibración"*, y `docs/04` #2.5 lo repetía. Es falso:
`_temporal_honesty_check` recibe los `best_params` de `_pick_final_params`, cuya ronda
final de Optuna optimiza sobre `X_final` **completo** —ese año incluido— y encima puede
venir sembrada por warm-start del campeón registrado, entrenado sobre datos que solapan.

**Qué mide en realidad.** Con hiperparámetros dados, cuánto error da refitear solo con el
pasado y predecir el último año: aísla **drift temporal**, no generalización virgen.
Leerlo como lo segundo lo vuelve optimista, y es justo el número que uno miraría para
decidir si promover.

**Acción — implementada (documental).** Corregidos el comentario, el log
(`"Ultimo anio (drift, NO holdout virgen)"`) y `docs/04` #2.5. Un holdout de verdad exige
excluir el último año **antes** de tunear; eso es un cambio estructural del orquestador y
queda propuesto, no implementado.

### 2.4 La mejora documentada más grande está bloqueada por datos, no por código

`config.py` presenta `ENABLE_EXTRA_CATEGORICALS` (CALIBRE + TIPO DE COSECHA) con evidencia
fuerte: **−1.58 pp de MAPE** en A/B sobre POP, ICC residual ~11% / ~8%. Está OFF y el
comentario sugiere que es cuestión de encender la flag.

**No lo es.** Verificado leyendo los workbooks:

```
data/training/DB-HISTORICA.xlsx  → 23 hojas, 15 columnas
data/BD_HISTORICO_ACUMULADO.xlsx → 1 hoja 'acumulado', 34.486 filas, 15 columnas
columnas: FECHA VARIEDAD FORMATO FUNDO HA KG/JR H-EF P/BAYA %INDUS
          DIA_SEM DPC MES KG/HA DIA_COSECHA KG/JR_H
CALIBRE → AUSENTE · TIPO DE COSECHA → AUSENTE · DESCRIPCION LAB → AUSENTE
```

Ni el archivo de training **ni la fuente acumulada** traen esas columnas. Encender la flag
hoy hace **fallar** el `data_loader` (valida `RAW_FEATURE_COLUMNS` y levanta `ValueError`).

**Acción — tuya, no de código.** Pedir al sistema de origen un extract que incluya
`CALIBRE` y `TIPO DE COSECHA` (y `DESCRIPCION LAB`, ICC ~6%, como tercera palanca),
regenerar `BD_HISTORICO_ACUMULADO.xlsx` → `task data:split` → recién ahí
`ENABLE_EXTRA_CATEGORICALS=1`. **Es la mayor mejora de MAPE con evidencia previa que tiene
el proyecto documentada.** Convendría corregir el comentario de `config.py` para que no
sugiera que basta la flag.

### 2.5 Lo que NO está roto (verificado)

Vale registrarlo para no perder tiempo re-auditando:

- **Sin leakage de lags.** Todo va por `shift(1)` dentro del `Pipeline`
  (`LagFeatureTransformer` como step 0), los flags se hornean en `flags_` al fit, y el
  historial se serializa. Invariante #9 de `CLAUDE.md` respetado.
- **Loss alineada con la métrica** dentro del modelo: `reg:absoluteerror` (XGB) /
  `regression_l1` (LGB) + `eval_metric` MAE explícito.
- **Cap del target CV-safe**: el p99.5 se calcula dentro de `TTR.fit`, o sea con el
  `y_train` del fold. Correcto.
- **Limpieza de datos sólida**: target ≤0 descartado en el filtro canónico (evita el
  `NaN` de `log1p`), dedup defensivo, fechas inválidas fuera, piso relativo de MAPE contra
  artefactos casi-cero, colapso de categorías raras adaptativo a `n`.
- **Adaptatividad por `n`** bien pensada (`caps_for_n`, `_adapt_folds_to_n`,
  `ADAPT_RARE_MIN_COUNT`): solo *aprieta* en variedades chicas, deja POP bit-idéntico.

---

## 3. Protocolo de experimentos

Con 23 variedades y métricas ruidosas, **una corrida no es evidencia**. Reglas:

1. **Una palanca por experimento.** Si mueves dos, no puedes atribuir el efecto.
2. **Sin registrar**: `--no-register` + `REGISTER_ENABLED=0` mientras experimentas.
3. **Perfil mínimo `prod`.** En `smoke`/`dev` la varianza entre folds tapa cualquier
   efecto real (`smoke` además no registra por diseño).
4. **Semillas.** El efecto de una palanca debe sobrevivir a ≥3 valores de `SEED`. Si
   cambia de signo entre semillas, es ruido.
5. **Múltiples variedades.** Mínimo POP (n≈10k) + una mediana (MAGICA n≈1.1k) + una chica
   (ROSITA n≈588). Una palanca que solo ayuda a POP no es una mejora del sistema.
6. **Métrica de decisión**: `temporal_mape_oof` y `final_holdout_mape` (extrapolación),
   no solo el MAPE OOF estratificado (interpolación, optimista).
7. **Umbral de adopción.** `OOF_MAPE_TIE_TOLERANCE = 0.5 pp` ya declara que menos de eso
   es ruido entre semillas. Exigir una mejora **mediana** > 0.5 pp sostenida, y revisar el
   `gap_rel` para no comprar MAPE con sobreajuste.

### Orden sugerido

| Orden | Experimento | Comando |
|---|---|---|
| 1 | Refit tras early stopping (#2.1) | `EARLY_STOPPING_REFIT_FULL=1` |
| 2 | Objetivo = MAPE (#2.2) | `OPTUNA_OBJECTIVE_METRIC=mape` |
| 2b | Objetivo = MAE log — alternativa robusta | `OPTUNA_OBJECTIVE_METRIC=mae_log` |
| 3 | Pesos ∝ 1/y — misma diana que #2, otra vía | `SAMPLE_WEIGHT_INV_Y=1` |
| 4 | Lags simples: PACF lag1 = +0.71, lag2 = +0.37 | `ENABLE_SIMPLE_LAGS=1` |
| 5 | Lags de features: ACF de P/BAYA 0.74, DPC 0.61; además imputa implícitamente el 39% de NaN de P/BAYA | `ENABLE_FEATURE_LAGS=1` |
| 6 | Mediana jerárquica para el fallback del imputer | `IMPUTER_GROUP_MEDIAN=1` |
| 7 | Colas de derivadas: ratio kurt=383, slope kurt=393, days_since kurt=836 | `LAG_LOG_DERIVED=1` |
| 8 | Cascade de outliers por (FUNDO, FORMATO) | `ENABLE_OUTLIER_CASCADE_FF=1` |
| 9 | Quitar la raw cuando existe su versión skew-mitigada | `SKEW_DROP_RAW=1` |

Ejemplo:

```bash
docker compose run --rm \
  -e EARLY_STOPPING_REFIT_FULL=1 \
  -e REGISTER_ENABLED=0 \
  trainer --varieties POP,MAGICA,ROSITA --tuning prod --no-register
```

**Los #4-#7 son los candidatos de feature engineering con evidencia estadística previa ya
anotada en `config.py`.** No inventar features nuevas antes de agotar estas: están
implementadas, son CV-safe y su justificación (ACF/PACF, kurtosis, % de NaN) ya está hecha.

---

## 4. Lo que NO recomiendo tocar

- **Subir trials sin arreglar #2.2.** Más presupuesto refina la respuesta a la pregunta
  equivocada mientras el objetivo esté desalineado del gate.
- **`OPTUNA_OBJECTIVE_GAP_PENALTY` / `STD_PENALTY`.** Ya se evaluaron y descartaron el
  2026-06-25 con razón documentada: desvían el objetivo de minimizar error. El anti-overfit
  vive en la capacidad de la grilla y en el gate del campeón.
- **Añadir un backend nuevo.** El cuello de botella hoy no es la familia de modelos: son
  la desalineación objetivo/gate (#2.2), el dato descartado (#2.1) y las columnas que no
  llegan del origen (#2.4).
- **`REQUIRE_TEMPORAL_GATE=1` en producción sin calibrar.** `docs/04` #2 ya lo advierte y
  tiene razón: falla cerrado y tumbaría promociones vigentes.
