# Plan de refactor priorizado — 2026-06-12

Origen: revisión exhaustiva con 5 agentes expertos (arquitectura, ML/leakage,
API/serving, MLOps/infra, reportes) + hallazgos propios de la fase de
ejecución. **Restricción rectora: el repo no tiene suite de tests**, así que
todo cambio estructural va secuenciado en pasos chicos verificables con el
stack (`task build` → smoke → UI), nunca como big-bang.

Los fixes críticos de la revisión YA están aplicados y probados
(2026-06-11/12): serialización self-contained de flags, gate de gap RELATIVO
multi-variedad, bandas conformal por fundo + cold-start en la API, rediseño
del Winner HTML (criterio real del campeón, plotly gzip offline), bucket S3 +
mount de credenciales del MLflow server. Lo de abajo es lo que queda.

## P0 — antes de escalar a más variedades

1. **Suite de tests mínima (pytest) — HECHO (2026-06-13).** `tests/` con 17
   tests verdes en contenedor (trainer 16+1 skip, api 4): lag transformer
   (round-trip, flags horneados, same-day ex-ante), select_champion,
   conformal bands, guard de registro, API halfwidths. Comando en CLAUDE.md.
   Falta: integrarla a CI cuando exista workflow. Detalle original:
   Sin esto, cada refactor es fe. Empezar por los puntos donde ya nos quemamos:
   - `LagFeatureTransformer`: fit→pickle→unpickle→transform produce las mismas
     columnas sin env vars (el bug de serialización que existió).
   - `select_champion`: gate relativo en ambos extremos de escala de target.
   - `build_conformal_metadata`: cascade fundo→global, cold-start, <20 residuos.
   - API `_conformal_halfwidths`: fundo conocido / desconocido / legacy sin
     `conformal_`.
   Esfuerzo: 1-2 días. Riesgo: cero (solo agrega).

2. **Config por variedad — HECHO (2026-06-13).** `src/variety_config.py`:
   `VarietyConfig` (frozen dataclass, None = default global) + `VARIETY_OVERRIDES`
   + `for_variety()`. Knobs conectados con paso explícito (no env): meses de
   temporada (FeatureGenerator, serializados en el pickle), umbral KNN
   (CustomKNNImputer vía factory), meses del boost de sample weights (tuning)
   y rare_min_count (data_loader, mismo valor en training y reportes).
   Validado: 6 tests (`tests/test_variety_config.py`) + smoke POP con métricas
   BIT-IDÉNTICAS al pre-refactor. Para una variedad nueva: agregar entrada en
   `VARIETY_OVERRIDES` con evidencia propia (no extrapolar umbrales de POP).

## P0.5 — guard de registro para runs experimentales (incidente 2026-06-13)

Un run dev con `EXANTE_MODE=1` (experimento #11) pasó el quality gate
(20.8% < 25%) y registró su campeón como `rnd-forest-POP` v2 — la API
sirve siempre la última versión, así que habría servido un modelo
experimental degradado en producción local (se eliminó la v2 a mano).
HECHO (2026-06-13): `REGISTER_ENABLED` en config.py + bloqueo automático
con `EXANTE_MODE=1` en `_apply_quality_gate` (variety_runner), cubierto
por `tests/test_register_guard.py`.

## P1 — deuda que ya molesta

3. **Partir `api/app/services/mlflow_service.py` — HECHO (2026-06-13).**
   Extraído `api/app/services/uncertainty.py` (funciones puras
   `conformal_halfwidths` + `predict_with_halfwidths`); el servicio delega.
   Tests actualizados (`tests/test_api_conformal.py` apunta al módulo nuevo).

4. **MLflow y reports server sin auth.** Local es tolerable; en AWS el ALB
   expone MLflow con `--serve-artifacts` (lectura/escritura de modelos).
   Mínimo: basic auth en el ALB o `mlflow server --app-name basic-auth`.
   Coordinar con `GUIA_MLOPS_AWS_V2.md` (#3.x) y los healthchecks.

5. **`step_05_evaluate/html/sections.py` creció a ~1000 líneas.** Partir en
   `sections/` (hero, comparativo, errores, fundo_formato) cuando se toque de
   nuevo; no como churn aislado.

6. **Unificar la doble fuente de "criterio del campeón".** El texto vive en
   `champion.py::decision_criteria` y se re-narra en `sections.py`. Mover los
   umbrales mostrados (ya se hace con `CHAMPION_MAX_GAP_REL`) y generar la
   narrativa desde `decision` siempre (nunca hardcodear en HTML).

## P2 — oportunista (tocar solo si se pasa por ahí)

7. `variety_runner.py` maneja registro + gate + summary + reportes: extraer
   `registry.py` (registro MLflow) cuando se agregue la 2da variedad.
8. Tipado: `mypy --strict` en `src/step_05_evaluate/` y `api/app/services/`
   primero (donde hay más None-juggling).
9. `mlruns/` legacy en el árbol: borrar tras confirmar que nada lo lee.
10. Renombrar repo/registry `rnd-forest-*`: NO hacerlo — el prefijo es
    contrato trainer↔API (invariante #8 de CLAUDE.md); documentado y barato
    de vivir con él.

## Validación estándar de cada paso

```bash
task lint
docker compose build trainer api
docker compose run --rm trainer --varieties POP --tuning smoke   # end-to-end
# + pytest cuando exista (P0.1)
```
