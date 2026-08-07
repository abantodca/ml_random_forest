# ADR-008 — CI sin job `test` mientras no exista `tests/`

**Estado:** accepted (vigente). Supersede una posición anterior no escrita (la suite existía y
fue retirada).

**Nota de vigencia (2026-08-07):** entre el 2026-07-23 y el 2026-07-24 se reintrodujo una suite
nueva, lo que invalidó temporalmente la premisa de este ADR. El 2026-08-07 el owner la retiró
otra vez —junto con `pytest` de `pyproject.toml` y `requirements-dev.txt`—, de modo que la
premisa ("el repo no tiene `tests/`") vuelve a ser cierta y este ADR queda vigente tal cual.
Las pruebas se pegan desde la guía cuando se necesitan.

Este es el único ADR cuyo **nombre de archivo ya estaba fijado** en la documentación antes de
existir: se lo referencia como `docs/adr/ADR-008-ci-sin-tests-todavia.md` desde tres lugares de la
guía de producción.

## Contexto

El repo **tuvo** una suite de tests y **ya no la tiene**. El commit `4650e7d`
(*Wed Jul 1 18:15:46 2026*, "feat(train): adaptividad por variedad (n-aware) + pruner Optuna +
limpieza") dice en el cuerpo:

> "Elimina la suite `tests/` por decision del owner (Taskfile/CLAUDE.md/CONTRIBUTING/pyproject
> actualizados)."

Archivos borrados: `tests/conftest.py`, `test_api_conformal.py`, `test_conformal_bands.py`,
`test_lag_transformer.py`, `test_register_guard.py`, `test_select_champion.py`,
`test_variety_config.py`.

Lo que había —inventariado en `docs/PLAN_REFACTOR_2026-06-12.md`— eran **17 tests verdes en
contenedor** (trainer 16+1 skip, api 4): lag transformer (round-trip, flags horneados, same-day
ex-ante), `select_champion`, conformal bands, guard de registro, halfwidths de la API. Ese mismo
documento justificaba su existencia con una frase que conviene no perder: **"Sin esto, cada refactor
es fe."**

Con la suite fuera, la pregunta de CI pasó a ser: ¿se deja un job `test` que saltee cuando no hay
tests, o no se pone job?

## Decisión

**No hay job `test` en CI.** El workflow de deploy corre `lint` (`task lint` + `task infra:validate`)
y nada más. El razonamiento, literal:

> "Un job `test` con `pytest tests/` que 'saltea si no hay tests' es código aspiracional que oculta
> cobertura cero. Mejor reflejar el estado real y agregar el job cuando exista `tests/`."

y, dicho de otra forma:

> "**No hay job `test` en CI** porque no hay `tests/`: un task que pasa trivialmente es deuda
> encubierta."

La validación real es: `ruff check` + smoke run (`task train VARIETIES=POP TUNING=smoke`, ~1 min) +
los gates de modelo (`CHAMPION_MAX_MAPE`, `CHAMPION_MAX_GAP`).

## Consecuencias

**Se gana:** el CI no miente. Un check verde significa "el lint pasó", no "el sistema está probado".

**Se pierde —y esto hay que asumirlo explícitamente:**

- Todo cambio estructural es manual y secuenciado. `docs/PLAN_REFACTOR_2026-06-12.md` abre con la
  restricción: *"el repo no tiene suite de tests, así que todo cambio estructural va secuenciado en
  pasos chicos verificables con el stack (`task build` → smoke → UI), **nunca como big-bang**."*
- **Hay al menos un refactor planificado que quedó inejecutable.**
  `docs/REFACTOR_ARQUITECTURA_2026-06-23.md` condiciona el split de `lag_features.py` a
  *"`test_lag_transformer.py` (pickle) verde antes/después"* — un test que ya no existe. Ver
  [ADR-007](ADR-007-lags-dentro-del-pipeline.md), que es justamente el invariante que ese test
  protegía.
- El contrato de branch protection queda pendiente: *"El context `Deploy / lint` corresponde al par
  (`name: Deploy`, job id `lint`). **Cuando agregues `tests/`, actualizar a
  `["Deploy / lint", "Deploy / test"]`**."*

## Cuándo revisar este ADR

Cuando exista `tests/`. Los candidatos mínimos ya están identificados y son los que más barato
compran:

1. **`select_champion` con empate** — protege [ADR-002](ADR-002-campeon-automatico.md)
2. **Shape contract de `prepare_data`**
3. **Roundtrip de joblib** — protege [ADR-005](ADR-005-nombres-step-xx-contrato-serializacion.md) y
   [ADR-007](ADR-007-lags-dentro-del-pipeline.md)

Al agregarlos: sumar el job `test` al workflow, actualizar branch protection, y escalar a
`mypy --strict src/` + `bandit -r src/` + `pytest --cov-fail-under=N`.

## Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| Job `test` que saltea si no hay `tests/` | *"código aspiracional que oculta cobertura cero"* / *"un task que pasa trivialmente es deuda encubierta"* |
| Mantener la suite | Decisión del owner, 2026-07-01 |
| Bloquear todo refactor hasta tener tests | Paralizaría el repo; en su lugar se adoptó la regla de pasos chicos verificables |

## Deriva documental detectada

`docs/REFACTOR_ARQUITECTURA_2026-06-23.md` afirma que la Fase 3 entregó
`.github/workflows/ci.yml (lint+P0+build)`. Ese archivo **no existe** — el directorio
`.github/workflows/` no está en el repo. Es una entrada obsoleta, escrita cuando el plan se daba por
ejecutado.
