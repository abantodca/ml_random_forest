# Refactor de arquitectura — 2026-06-23

Reorganización para dejar el repo "listo para un experto": tooling moderno,
documentación de arquitectura, CI/CD, y modularización de archivos grandes
**preservando comportamiento e invariantes** (`CLAUDE.md` #1–#10). Trabajo en la
rama `refactor/arquitectura-experto`, commits atómicos por fase.

## Hecho en esta tanda

| Fase | Entregable | Validación |
|---|---|---|
| 1 · Tooling | `.pre-commit-config.yaml` (ruff+higiene); `pyproject` con `[tool.pytest]`/`[tool.mypy]`; `pre-commit`+`mypy` en dev-deps | TOML válido |
| — · Formato | `ruff format` repo-wide (130 archivos, solo estilo) | check+format limpios; P0 verde |
| 2 · Docs | `ARCHITECTURE.md` (Mermaid C4/secuencia/deploy) + `CONTRIBUTING.md` (incl. checklist de invariantes) | — |
| 3 · CI/CD | `.github/workflows/ci.yml` (lint+P0+build) + `dependabot.yml` | YAML válido |
| 4 · Auditoría | Dead-code **0**, TODOs reales **0** (el "107" era la palabra "todo" en español). Fix de drift de esquema (abajo). Inventario stale de `notebooks/README.md` corregido | ruff |
| 5 · Split (parcial) | `dashboard_index.py` 692→429 (CSS/JS → `_dashboard_assets.py`); `single_run.py` 619→290 (helpers → `_run_mlflow_logging.py` + `_run_outputs.py`) | render funcional + import + P0 |

**Fix de bug (invariante #10):** la UI descartaba en silencio las bandas
`kghora_std/lo/hi/confidence` que el API sí emite. Añadidos a `ForecastRecord`
y `PredictionResult` (`ui/app/schemas/models.py`) y poblados en
`forecast_service.predict_dry`. **Pendiente:** renderizarlas en
`views/forecast.py` (feature de UI, no cleanup).

## Pendiente — splits de archivos >500 LOC (plan por archivo)

Patrón de seguridad (aplica a todos): **el símbolo público nunca cambia de
path**; solo se mueven privados/helpers, re-exportados desde el módulo o
`__init__.py` que ya importa el call-site. Validar `ruff` + P0 (+ stack si toca
runtime) tras CADA split.

| Archivo | LOC | Acción | Riesgo |
|---|---|---|---|
| `step_05_evaluate/html/sections.py` | 991 | → sub-paquete `sections/` (hero, kpis, groups, backends, diagnostics, errors_detail, links_actions); `__init__` re-exporta los 16 `build_*` que importa `winner_dashboard.py` | bajo (preservar laziness de plotly) |
| `diagnostics/eda.py` | 539 | extraer `eda_sidecar.py` (`_write_eda_sidecar`, `find_latest_eda_sidecar`, `extract_drift_summary`); re-exportar las 2 públicas desde `eda.py` | bajo |
| `orchestration/variety_runner.py` | 756 | helpers MLflow → `_variety_mlflow.py`; outputs → `_variety_outputs.py`. `train_variety` y `_apply_quality_gate` (lo toca un test) **quedan** | bajo |
| `api/app/services/drift_service.py` | 667 | estadística pura (PSI/KS/Chi²) → `_drift_stats.py`; `DriftService` queda en su path (fachada `services/__init__.py`) | bajo (ojo tests sobre staticmethods) |
| `step_03_features/lag_features.py` | 913 | cómputo puro → `_lag_compute.py`; **`LagFeatureTransformer` NO se mueve** (su path está horneado en `.joblib`, inv. #4). Re-exportar símbolos legacy | **alto** — solo con `test_lag_transformer.py` (pickle) verde antes/después |

**NO partir** (cohesión real alta / riesgo > beneficio, confirmado en auditoría):
`config.py` (contrato de import global + orden), `tuning.py` (invariante #2:
"tuning.py untouched"), `api/.../mlflow_service.py` (`import src` side-effect +
disciplina de lock/cache).

Orden sugerido por rédito/riesgo: `html/sections` → `eda` → `variety_runner` →
`drift_service` → (`lag_features` al final, con red de pickle).

## Pendiente — limpiezas menores (de la auditoría)

- **Duplicación de helpers de fecha en la UI** (consolidar en un util):
  `iso_week`/`_iso_week_label` en `presenters/tracking.py:42` y
  `services/tracking_service.py:32`; closure `_iso` en `presenters/forecast.py:71`
  y `presenters/tracking.py:122`.
- **Huérfanos en `scripts/experiments/`** (one-shots fechados ya ejecutados:
  `backfill_mlflow_reports.py`, `mlflow_cleanup.py`, `lgb_l1_vs_tweedie.py`) —
  decidir borrar o mover a `notebooks/`; sus docstrings de uso apuntan a rutas
  viejas (`artifacts/...`). **No se borraron aquí** (son del usuario).
- **Doc suelto** `ANALISIS_GPBOOST_RETIRO.md` (raíz, 0 refs) → mover a `docs/` o
  enlazar desde README. (`ANALISIS_XGBOOST_SOBREAJUSTE.md` **se queda**: lo
  referencia `src/config.py:291,356`.)
- **Opcional:** que `api/.../feature_pipeline.py` importe los nombres de columnas
  desde `src` en vez del literal `MODEL_INPUT_COLUMNS` (contrato sincronizado a
  mano).
