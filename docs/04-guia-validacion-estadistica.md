# Guía de validación estadística y mejora del modelo

Esta guía conecta el comportamiento real de `src/` con una política segura de
entrenamiento, tuning y promoción. No reemplaza el flujo existente: distingue
dos problemas que requieren validaciones diferentes.

## 1. Definir qué se predice

- **Nowcasting/interpolación**: al predecir ya se conocen variables del evento
  como `KG/HA` y `%INDUS`. La CV estratificada por `FUNDO+FORMATO` mide este
  caso, pero mezcla años.
- **Forecast ex-ante**: al predecir todavía no se conocen variables del día de
  cosecha. Debe usarse `EXANTE_MODE=1`, lags construidos solo con pasado y CV
  temporal. Este modo continúa siendo experimental y no se registra.

No se deben comparar directamente sus MAPE: responden preguntas distintas.

## 2. Validación recomendada

El pipeline implementa validación dual:

1. Nested CV estratificada para tuning y comparación histórica.
2. `TemporalYearSplit` expanding-window como backtest de años no vistos.
3. `PurgedDateSplit` en el inner CV temporal, manteniendo fechas iguales en un
   solo lado y dejando un gap entre train y validación.
4. Baseline jerárquico `FUNDO+FORMATO → FUNDO → FORMATO → global`, ajustado
   únicamente con el train de cada fold.
5. El último año del expanding-window se publica además como
   `final_holdout_*`. No interviene en Optuna ni en la elección de parámetros.

Para investigación se conserva el default:

```bash
DUAL_CV_REPORT=1 python main.py --tuning dev --varieties POP
```

Para que un modelo de forecast no llegue al Registry solo por interpolar bien:

```bash
REQUIRE_TEMPORAL_GATE=1 \
MIN_TEMPORAL_FOLDS=2 \
CHAMPION_MAX_TEMPORAL_MAPE=35 \
CHAMPION_MIN_TEMPORAL_BASELINE_SKILL=0 \
python main.py --tuning prod --varieties POP
```

El gate falla cerrado si faltan folds o métricas temporales. Antes de activarlo
en producción, calibrar `CHAMPION_MAX_TEMPORAL_MAPE` por variedad usando varios
backtests; no elegir el umbral mirando una sola corrida.

## 3. Métricas que deben leerse juntas

- **MAE**: error en unidad física y métrica principal para el skill frente al
  baseline.
- **WAPE**: error relativo agregado, estable cuando existen targets pequeños.
- **MAPE con piso físico**: interpretable por fila; reporta cuántas observaciones
  casi cero excluye.
- **sMAPE**: vista porcentual simétrica complementaria.
- **R²**: capacidad explicativa; no sustituye una métrica de error.
- **Bias**: signo y magnitud del sesgo sistemático.
- **Skill MAE**: `1 - MAE_modelo / MAE_baseline`; debe ser mayor que cero para
  justificar el modelo complejo.

Reportar media y dispersión por fold, además de resultados por año, `FUNDO` y
`FORMATO`. Un promedio global puede ocultar un segmento inaceptable.

## 4. Tuning sin fuga

- Optuna solo puede observar folds del inner CV.
- El outer fold estima generalización y no participa en la elección de
  hiperparámetros.
- Imputación, categorías raras, outliers, features y lags permanecen dentro del
  `Pipeline`; moverlos antes del split produciría leakage.
- El holdout de early stopping debe ser la cola temporal del train.
- La ronda final puede reutilizar la región de parámetros de un campeón previo,
  pero nunca sembrar los outer folds.

Más trials no corrigen una validación incorrecta. Primero se estabilizan folds,
baseline y definición ex-ante; después se aumenta `dev → prod → prod_xl`.

## 5. Series de tiempo + machine learning

La combinación apropiada ya es un modelo global de árboles con:

- calendario cíclico y temporadas;
- lags y estadísticas móviles por jerarquía;
- expanding-window para medir extrapolación;
- ventana reciente opcional por variedad cuando el backtest demuestra drift;
- intervalos conformales calibrados con residuos OOF y énfasis reciente.

Un modelo mixto estadístico puede añadirse como candidato, no como reemplazo
directo: pronóstico de nivel/tendencia estacional por grupo más un booster sobre
los residuos. Debe usar exactamente los mismos outer folds y superar tanto al
baseline jerárquico como a XGB/LGB en backtest temporal. No conviene incorporarlo
sin evidencia de mejora consistente en varios años y grupos.

El repositorio incluye `ResidualHybridRegressor` como candidato experimental:
Ridge captura la estructura suave y `HistGradientBoostingRegressor` corrige sus
residuos. Está fuera del flujo normal y solo se agrega al registry al definir
`ENABLE_EXPERIMENTAL_MIXED_BACKEND=1`. Durante evaluación debe combinarse con
`--no-register` y `REGISTER_ENABLED=0`.

## 6. Criterio de modelo ganador

La selección actual conserva el contrato:

1. pasa el gate de gap relativo;
2. minimiza MAPE OOF de negocio;
3. ante empate técnico, minimiza tiempo.

Luego, el quality gate exige MAPE operativo y skill frente al baseline. Con
`REQUIRE_TEMPORAL_GATE=1` añade evidencia temporal mínima, MAPE temporal y skill
temporal. Esto separa correctamente selección entre candidatos de la decisión
más exigente de promover a producción.

## 7. Secuencia segura de adopción

1. Ejecutar `dev` con validación dual y revisar diferencias entre CV
   estratificada y temporal.
2. Calibrar umbrales por variedad con backtests históricos, nunca con el test
   más reciente solamente.
3. Activar el gate temporal en staging y observar al menos un ciclo con etiquetas
   futuras.
4. Probar ventanas recientes y flags de features mediante ablación: una sola
   modificación por experimento.
5. Promover solo mejoras repetibles en MAE/WAPE, skill temporal y sesgo, no una
   ganancia aislada de MAPE.

## 8. Compatibilidad y producción

Los defaults conservan el sistema previo:

- `REQUIRE_TEMPORAL_GATE=0`: el nuevo gate no cambia promociones existentes.
- `ENABLE_EXPERIMENTAL_MIXED_BACKEND=0`: solo se entrenan XGB y LGB.
- Los umbrales por variedad son `None`: se usan los globales.

No activar flags experimentales directamente en producción. Primero ejecutar
local o staging con `--no-register`, conservar resultados por fold y observar
etiquetas futuras. La promoción productiva queda como una acción separada.
