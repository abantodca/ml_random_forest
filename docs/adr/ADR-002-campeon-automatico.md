# ADR-002 — El sistema elige el campeón; no hay flag para forzar un backend

**Estado:** accepted

## Contexto

El pipeline entrena varios backends por variedad (hoy XGBoost y LightGBM). Había que decidir **quién
elige el ganador**: el operador con un flag, o el sistema con un criterio fijo.

La decisión y su historia están en el docstring de `src/step_05_evaluate/champion.py:1-26`, que es el
bloque de rationale más denso del repo:

> "Cada modelo (XGB, LGB, ...) entrena de forma INDEPENDIENTE: su propio Optuna study, su propio
> search space, su propio MLflow run. Cuando todos terminan, comparamos sus metricas con un criterio
> LEX-ORDER (prioridad estricta) que refleja el contrato de MLOps:
>  1. GATE DE OVERFITTING: `gap_rel = |gap|/MAE_test <= CHAMPION_MAX_GAP_REL`. El gap es una
>     RESTRICCION (descalifica modelos rotos), NO un objetivo a minimizar. **Minimizar gap como
>     criterio primario premiaba al modelo mas subajustado, no al que mejor predice (revision
>     2026-06-10).** Relativo desde 2026-06-11: comparable entre variedades de escala distinta (el
>     `|gap|*100` viejo eran kilos disfrazados de pp).
>  2. GENERALIZACION: menor MAPE OOF de negocio (cada fila predicha por un modelo que NO la vio en
>     train). Es la metrica honesta de produccion.
>  3. EFICIENCIA: menor tiempo de entrenamiento ante empate practico (`OOF_MAPE_TIE_TOLERANCE`)."

## Decisión

**Todos los backends del registry entrenan, siempre.** `select_champion` elige por lex-order estricto:
gate de overfitting (restricción, no objetivo) → MAPE OOF de negocio → wall time ante empate.

**No hay flag para forzar un backend.** `README.md:136-137`:

> `# Siempre entrena XGB y LGB y select_champion elige el ganador`
> `# (ADR-002: el sistema decide, no hay flag para forzar un modelo).`

## Consecuencias

**Se gana:**

- La elección es auditable y reproducible: dada la misma data y el mismo seed, el campeón es el
  mismo. No depende de qué recordó el operador.
- El punto de extensión es barato y fijo: *"Adding a backend = one new file in `src/step_04_train/`
  + its entry in `registry.py` and `search_spaces.py`; `tuning.py` is untouched"* (`CLAUDE.md:101`).
  Eso está además como checklist de PR en `CONTRIBUTING.md:69-70`.
- Protege a `tuning.py` de refactors: la tabla "NO partir" de
  `docs/REFACTOR_ARQUITECTURA_2026-06-23.md` lo lista con el motivo *"invariante #2: tuning.py
  untouched"*.

**Se pierde:**

- Pagás siempre el tiempo de entrenar **todos** los backends, aunque sospeches cuál va a ganar. Es el
  costo explícito de no confiar en la corazonada del operador.

**Guard:** las corridas con `--tuning smoke` **nunca registran modelos** (`CLAUDE.md:102`).

## Alternativas descartadas

| Alternativa | Por qué no | Cuándo se descartó |
|---|---|---|
| Minimizar el gap como criterio primario | *"premiaba al modelo mas subajustado, no al que mejor predice"* | revisión 2026-06-10 |
| Gap absoluto (`\|gap\|*100`) | *"eran kilos disfrazados de pp"* — no comparable entre variedades de escala distinta | 2026-06-11 |
| `full_mape` (in-sample) como métrica de decisión | *"optimista por construccion y premiaba memorizar el train"*. Se conserva como informativa en dashboards | — |
| `composite_score` | Degradado a tag de MLflow; *"el campeon NO lo usa para la decision"* | — |
| Stacking | *"Stacking (eliminado, no existe)"* | — |
| GPBoost / mixed-effects | Evaluado y removido: *"the data has no group structure that random effects capture beyond the per-group lag features"* (`CLAUDE.md:17-18`) | — |
| Un flag `--force-backend` | Haría la elección dependiente del operador y no auditable | — |

## Dónde vive en el código

- `src/step_05_evaluate/champion.py:1-26` (rationale), `:50-70` (`select_champion`)
- `src/step_04_train/registry.py`, `src/step_04_train/search_spaces.py`
- `src/orchestration/variety_runner.py`, `src/orchestration/quality_gate.py`
- Gates en `src/config.py`: `CHAMPION_MAX_MAPE`, `CHAMPION_MAX_GAP`, `CHAMPION_MAX_GAP_REL`,
  `OOF_MAPE_TIE_TOLERANCE`
