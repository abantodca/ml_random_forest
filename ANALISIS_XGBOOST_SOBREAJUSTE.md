# Análisis: por qué XGBoost sobreajusta y queda descalificado por el gate

> Documento de análisis (2026-06-19). Variedad de referencia: **POP**.
> Basado en el **entrenamiento ya registrado** (artifacts en S3) + sondas
> empíricas. **No requiere re-entrenar**. Fixes en la rama `experiment/xgb-fair-fight`.
> Complemento de `ANALISIS_GPBOOST_SUBAJUSTE.md` (mismo método, hallazgo distinto).

## TL;DR

XGB no pierde por MAPE: con **14.58% OOF era competitivo** con `lgb` (14.36%).
Pierde porque **SOBREAJUSTA** y **falla el gate de overfitting** (gap_rel=0.42 >
0.40). Es el problema **opuesto** al de GPB (que subajustaba).

**A diferencia de GPB, XGB NO tiene un bug.** Se verificó empíricamente que su
early stopping ya corta por MAE. El sobreajuste es el **tradeoff deliberado de
rev.7** (capacidad de grilla abierta, control delegado al gate) **+** que el
objetivo de Optuna no penaliza el gap train→test. **El gate está funcionando
correctamente: `lgb` gana de forma legítima.** No había que re-entrenar.

## 1. La evidencia: XGB sobreajusta (firma opuesta a GPB)

De `champion_POP.json` (entrenamiento del 2026-06-15):

| Modelo | MAE **train** | MAE **test** | gap (overfit) | MAPE OOF | Tiempo |
|---|---|---|---|---|---|
| **XGB** | **0.38** ← menor | 0.655 | **0.274** ← mayor | 14.58% | 9711s ← más lento |
| **LGB 🏆** | 0.48 | 0.650 | 0.167 | **14.36%** | 5762s |
| GPB | 0.60 | 0.72 | 0.122 | 16.13% | 2941s |

XGB tiene el **menor error de train** y el **mayor gap**: firma textbook de
**sobreajuste**. El campeón JSON lo dice explícito: *"XGB descartado por gate de
overfitting: gap_rel=0.42 supera el maximo 0.4 (campeon: 0.26)"*. Su MAPE era
competitivo — el problema es puramente la generalización (gap).

## 2. NO es un bug: el early stopping ya corta por MAE (verificado)

Sospecha inicial (análoga a GPB): ¿XGB corta el boosting por la métrica
equivocada? `EarlyStoppingXGBRegressor.fit` no seteaba `eval_metric`
(`src/step_04_train/early_stopping.py`), a diferencia de lgb (`eval_metric="l1"`).

**Sonda directa (xgboost 3.2.0)** con `objective="reg:absoluteerror"`:

| eval_metric | métrica usada | best_iteration |
|---|---|---|
| (default, sin setear) | **mae** | 143 |
| `"mae"` explícito | mae | 143 (idéntico) |
| `"rmse"` | rmse | 259 (más árboles → más overfit) |

→ Con `reg:absoluteerror`, el default de XGBoost **ya es `mae`**. El early
stopping **ya cortaba por MAE correctamente**; el docstring era cierto. **No hay
bug.** (Se fija `eval_metric="mae"` explícito igualmente, por robustez/paridad —
ver fix #1, sin cambio de comportamiento.)

## 3. La causa real del sobreajuste (dos factores, ambos por diseño)

1. **Grilla de capacidad abierta (rev.7, deliberada).** El docstring de
   `suggest_xgb_params` (`search_spaces.py:62-85`) documenta que rev.6.x apretó
   la grilla "para pasar el gate" y eso causó **subajuste** estructural; rev.7
   revirtió: pisos de regularización ~0, `max_depth` 3-8, `lossguide` +
   `max_leaves` hasta 64, y el control de overfit se delega a (a) early stopping,
   (b) CV temporal, (c) el gate. Con esa capacidad, XGB puede memorizar el train
   (MAE_train 0.38). **Apretar la grilla otra vez sería repetir el error de
   rev.6.x** → no se hace.

2. **El objetivo de Optuna no penaliza el gap.** `_objective`
   (`src/step_04_train/tuning.py:139-185`) optimiza `mean(MAE_val) + λ·std(MAE_val)`:
   **solo mide MAE de validación, no el gap train→test**. Así el TPE elige para
   XGB la config de menor MAE_val, que cae en una región de **alto gap** (porque
   XGB con capacidad abierta logra buen MAE_val *memorizando*). El gate del
   campeón (`select_champion`) la descalifica después. El gate es una
   **restricción post-hoc**, no parte del objetivo de tuning (ADR-002).

## 4. Qué NO se hace (sería trampa o regresión)

- **Relajar el gate** (subir el umbral de gap_rel) para dejar entrar a XGB:
  sería *gaming* — el gate existe justamente para vetar modelos que no
  generalizan. No se toca.
- **Apretar la grilla de XGB**: rev.7 ya demostró que causa subajuste. No se hace.

## 5. Fixes preparados en `experiment/xgb-fair-fight` (uno por commit)

| # | Fix | Archivo | Riesgo | Efecto |
|---|-----|---------|--------|--------|
| 1 | `eval_metric="mae"` explícito | `model_xgb.py` | Nulo (sin cambio hoy) | Robustez/paridad con lgb |
| 2 | **Penalización opcional por gap** en el objetivo Optuna (`OPTUNA_OBJECTIVE_GAP_PENALTY`, default 0.0) | `config.py`, `tuning.py` | Nulo con default 0.0 | **El lever real**: opt-in para tunear hacia generalización |
| 3 | Instrumentación de overfit (best_iter + gap) DEBUG | `early_stopping.py` | Nulo | Observabilidad del overfit |
| 4 | Este doc | — | — | — |

### El lever real (fix #2), en detalle

`OPTUNA_OBJECTIVE_GAP_PENALTY` añade al objetivo:
`score = mean(MAE_val) + std_penalty + λ·mean(max(0, MAE_val − MAE_train))`.

Con `λ>0`, el TPE **evita las configs que memorizan el train** — las mismas que
luego falla el gate. Es el camino **honesto** para que XGB pase el gate: no se
relaja el gate, se hace que el tuning lo **respete por construcción**, buscando
generalización en vez de solo MAE_val. Default 0.0 = comportamiento histórico
bit-idéntico (ni siquiera calcula el gap). Aplica a los 3 backends.

> Nota de diseño: esto convierte el control de overfit de "restricción post-hoc"
> (gate) a "objetivo blando" (penalización) — un cambio de filosofía respecto a
> ADR-002. Por eso es **opt-in y default off**: habilitarlo en producción es una
> decisión de ADR, no un default que se cambie sin medir.

## 6. Conclusión y cómo medir

- **No re-entrenar fue correcto.** `lgb` gana legítimamente; el gate descalifica
  a XGB porque genuinamente sobreajusta, no por un defecto del pipeline.
- A diferencia de GPB, **XGB no tenía un bug** — el early stopping ya estaba bien.
  Su overfit es el costo de la capacidad abierta de rev.7, no corregible con un
  one-liner sin gaming.
- **Cómo probar el lever** (cuando se quiera, sin compromiso): correr
  `OPTUNA_OBJECTIVE_GAP_PENALTY=0.5 task train VARIETIES=POP TUNING=dev` en esta
  rama y revisar si XGB baja su gap_rel por debajo de 0.40 (pasaría el gate) y a
  qué costo de MAPE. Con `LOG_LEVEL=debug`, la instrumentación (fix #3) muestra
  el gap por fold en vivo.
- **Predicción**: con el gap-penalty, XGB pasaría el gate pero su MAPE subiría
  hacia ~14.6-15% (al renunciar a parte del ajuste fino), así que **seguiría sin
  superar a `lgb`** — confirmando, igual que con GPB, que el campeón actual es el
  correcto.

## 7. Revisión 2026-06-22: causa raíz en la grilla (estudio multi-agente)

Un segundo análisis código-a-código (un agente por backend, comparando contra
LGB como referencia) matizó la conclusión de #6. Hallazgo central: **LGB es la
plantilla de la que se clonaron las grillas de XGB (rev.7) y GPB (rev.1)** — no
tres diseños independientes. Y apareció una asimetría estructural concreta:

- **`grow_policy='depthwise'` sin tope de hojas** (`search_spaces.py`, grilla
  rev.7-8): en ~50% de los trials del TPE el árbol crecía a `2^depth` hojas
  (hasta ~256 en depth 8) **sin control de ancho**, mientras LGB **siempre**
  tunea `num_leaves` acotado a `min(2^depth-1, 64)`. Es la asimetría que mejor
  explica el síntoma (XGB: menor train MAE, mayor gap).
- El tuning, el early stopping (mismo holdout seed 42, misma métrica MAE) y la
  limpieza/feature-eng son **simétricos entre XGB y LGB** (verificado línea a
  línea) → **no** son la causa. XGB no necesita feature engineering distinto.

**Fix aplicado (rev.9, commit `fix(xgb): max_leaves siempre acotado`)**: tunear
`max_leaves` SIEMPRE, acoplado a depth (`8 .. min(2^depth, 64)`), espejando a
`suggest_lgb_params`. En XGBoost ≥2.0 `max_leaves` acota el ancho también con
`depthwise` (sonda 3.2.0: depthwise+max_leaves=16 → 16 hojas; sin él → 72).

> Matiz vs #6: esto **no es gaming del gate**. No relaja la restricción ni
> recorta la capacidad "anti-gap" (el error de rev.6 que causó subajuste);
> solo pone el control de ANCHO de XGB a la par del de LGB. Es el lever de
> causa raíz; el `OPTUNA_OBJECTIVE_GAP_PENALTY` de #5 trata el síntoma. Ambos
> son ortogonales y robustos para cualquier variedad (el cap se acopla a depth,
> no a constantes ajustadas a POP).
