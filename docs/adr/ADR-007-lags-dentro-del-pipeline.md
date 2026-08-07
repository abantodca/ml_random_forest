# ADR-007 — Los lags se computan dentro del sklearn Pipeline

**Estado:** accepted
**Número asignado:** 2026-07-20, en la reorganización documental. El invariante y su rationale ya
estaban en `CLAUDE.md:127-132` y `CONTRIBUTING.md:63-64`.

## Contexto

El modelo usa features de lag (valores de períodos anteriores por grupo). Calcularlas en el
`data_loader` es lo cómodo: se hace una vez, sobre el dataframe completo, y todo lo de abajo recibe
las columnas ya listas.

**Eso filtra información del futuro entre folds de validación cruzada.** Si los lags se computan
sobre el dataset entero *antes* del split, cada fold de train contiene valores derivados de filas que
están en su fold de test. El MAPE de CV sale optimista y el modelo elegido es el que mejor explotó la
fuga, no el que mejor predice.

`CLAUDE.md:127-132`:

> "Lag features are computed INSIDE the sklearn Pipeline (`LagFeatureTransformer`, step 0 of
> `build_preprocessing_pipeline`) so each CV fold computes lags only from its own train slice —
> `data_loader.py` returns the 9 raw columns only. Do not move lag computation back to the loader:
> computing lags over the full dataset before the CV split **leaks future information across folds**.
> The transformer bakes its feature flags into `flags_` at fit time (self-contained serialization);
> never read env flags at transform time."

## Decisión

`LagFeatureTransformer` es el **paso 0** de `build_preprocessing_pipeline`. `data_loader.py` devuelve
únicamente las 9 columnas crudas. Los flags de features se **hornean en `flags_` en el `fit`**; el
`transform` nunca lee variables de entorno.

## Consecuencias

**Se gana:**

- El MAPE OOF es honesto, y por lo tanto [ADR-002](ADR-002-campeon-automatico.md) —que elige el
  campeón por esa métrica— decide sobre algo real.
- El pipeline serializado es **autocontenido**: un `.joblib` cargado en otra máquina, sin el `.env`
  del entrenamiento, produce exactamente las mismas columnas.

**Se pierde:**

- Los lags se recalculan en cada fold. Es trabajo repetido y el CV es más lento. Es el precio de no
  filtrar.

**Historia que respalda la segunda mitad de la decisión:** el horneado de flags no es teórico. El plan
de refactor registra el incidente como caso de prueba prioritario — *"`LagFeatureTransformer`:
fit→pickle→unpickle→transform produce las mismas columnas sin env vars (**el bug de serialización que
existió**)"*.

## Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| Calcular lags en `data_loader.py` | Filtra información del futuro entre folds de CV. Es la razón de existir de este ADR |
| Leer los flags de entorno en `transform()` | Un pipeline deserializado en otro entorno produciría columnas distintas — el bug que efectivamente ocurrió |
| Mover `LagFeatureTransformer` a otro módulo en un refactor | Su path está horneado en los `.joblib` ([ADR-005](ADR-005-nombres-step-xx-contrato-serializacion.md)) |

## ⚠️ Deriva documental detectada

`README.md:343` afirma que los lags se calculan *"en `data_loader.py` antes del CV split"*, que es
**exactamente lo contrario** de este ADR y de `CLAUDE.md:129`. Es documentación obsoleta que
sobrevivió al cambio: describe el diseño que este ADR reemplazó. Corregir esa línea, porque hoy
instruye a hacer justo lo que rompe el anti-leakage.

## Dónde vive en el código

- `src/step_03_features/lag_features.py::LagFeatureTransformer` — el transformer y su `flags_`
- `src/pipeline/build_pipeline.py::build_preprocessing_pipeline` — paso 0
- `src/step_01_load/data_loader.py` — devuelve solo las 9 columnas crudas
- `src/step_04_train/temporal_cv.py`, `cv_strategy.py` — los folds que dependen de esto
