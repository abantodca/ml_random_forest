# ADR-004 — Diagnostics, EDA y drift son standalone y no bloquean el training

**Estado:** accepted

> **Nota sobre el alcance.** Este ADR se citaba desde un único lugar del repo —`README.md:227`,
> `diagnostics/ # EDA standalone (eda.py) + dashboards / drift (ADR-004)`— sin documento detrás. El
> contenido de abajo reconstruye la decisión a partir del rationale que **sí** está escrito en el
> repo, pero sin etiquetar como ADR-004. Si el alcance original era más angosto, este documento lo
> ensancha: revisalo antes de tratarlo como ratificado.

## Contexto

El sistema necesita tres cosas que no son entrenar: explorar los datos (EDA), medir drift entre train
y test, y publicar dashboards de resultados. La pregunta era si eso vive **dentro** del pipeline de
entrenamiento —y por lo tanto puede bloquearlo— o al costado.

Sobre el EDA, la guía es explícita:

> "El EDA es una necesidad **aparte y opcional** del entrenamiento: lo corres cuando querés
> inspeccionar la calidad/drift de los datos de una variedad, tantas veces como haga falta. **No
> forma parte del pipeline de `batch:train` — es standalone y repetible.**"

Sobre el drift, el mecanismo y su umbral:

> "`task eda` calcula `psi_train_test` por feature numérica y escribe `artifacts/eda_<variety>.json`.
> `task train` lee ese JSON: `psi > 0.25` en cualquier feature **warn-loud** en stdout y se tagea el
> run con `psi_warn=true`. Bloqueante a futuro si pasa a ser política. — **Drift severo entre
> train/test indica que el split de validación no representa training; un campeón sobre data
> drifteada es modelo sobre ruido.**"

## Decisión

- **EDA y diagnostics viven en `src/diagnostics/`**, se corren por separado (`task eda`) y son
  repetibles.
- **El drift se mide como PSI** y se comunica por un JSON sidecar (`artifacts/eda_<variety>.json`)
  que el training lee. Con `psi > 0.25` el run **avisa fuerte y se tagea** con `psi_warn=true`, pero
  **no falla**.
- **Los índices de reportes se generan server-side** como HTML estático autocontenido.

## Consecuencias

**Se gana:**

- El training no depende de que el EDA se haya corrido. Un pipeline de producción no se cae porque
  falte un análisis exploratorio.
- El drift queda registrado en el run de MLflow (`psi_warn=true`), así que la evidencia sobrevive
  aunque nadie haya mirado el stdout.
- Los dashboards son autocontenidos: no dependen de un CDN ni del autoindex del servidor.

**Se pierde:**

- Un modelo entrenado sobre datos drifteados **se registra igual**. El PSI avisa, no frena. Esa
  decisión está tomada a conciencia y marcada como revisable (*"Bloqueante a futuro si pasa a ser
  política"*).

## Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| PSI como gate bloqueante | Diferido, no adoptado: frenaría corridas legítimas mientras se calibra el umbral. La escalera propuesta es `psi > 0.10` warn temprano + `psi > 0.25` bloqueante |
| EDA dentro del pipeline de training | Lo haría obligatorio y lento en cada corrida, para algo que se consulta ocasionalmente |
| `index.html` dinámico con JS sobre autoindex de nginx | Tenía un bug de loop: *"nginx servia el `index.html` a `fetch('./')` en vez del listing del directorio"* (`src/diagnostics/dashboard_index.py:1-6`) |
| Plotly desde CDN | Rompe la autocontención del reporte. `REPORT_PLOTLY_OFFLINE=1` (default) embebe plotly.js gzip |

## Dónde vive en el código

- `src/diagnostics/` — `eda.py`, `dashboard_index.py`, `_dashboard_assets.py`,
  `statistical_tests.py`, `multivariate.py`, `residuals.py`, `temporal.py`
- `src/orchestration/variety_runner.py` — lectura del JSON y tagging `psi_warn`
- `api/app/services/drift_service.py`, `drift_baseline.py` — drift del lado del serving
- `src/config.py` — `REPORT_PLOTLY_OFFLINE`
