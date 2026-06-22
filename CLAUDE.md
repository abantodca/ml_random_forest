# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Monorepo for an MLOps system that forecasts harvest productivity (`KG/JR_H`, kg per
jornal-hour) per crop **variety**. Three deployables share one ML codebase (`src/`):

| Piece | Location | Role |
|---|---|---|
| Trainer | `src/` + `main.py` | trains XGB + LGB per variety, picks a champion, registers `rnd-forest-<variety>` in MLflow |
| API | `api/` (FastAPI) | serves the registered models + persists forecasts to Postgres |
| UI | `ui/` (Streamlit) | management dashboard that consumes the API |

The repo is historically named `ml_random_forest` but trains **XGBoost + LightGBM**, not Random Forest.
(A third backend, GPBoost/mixed-effects, was evaluated and removed — see #N in README: the data
has no group structure that random effects capture beyond the per-group lag features.)

Code comments and docs are written in **Spanish** — match that when editing. In docs/comments
reference sections as `#N`, not `§N`.

## Authoritative docs (read before deep work — do not duplicate them here)

- **`README.md`** — the full ML design: pipeline flow, feature schema, champion selection,
  nested-CV, anti-overfitting, MLflow conventions, outputs. This is the deepest source.
- **`GUIA_MLOPS_AWS_V2.md`** — step-by-step runbook for local + AWS stand-up, and the **ADRs**
  (ADR-001/002/003/004 referenced throughout the code).
- **`REPORTE_COSTOS_GERENCIAL.md`** — cost analysis (scheduler on/off economics).

## Commands

`Taskfile.yml` (go-task) is the command hub; it loads `.env` and includes namespaced taskfiles
(`infra:`, `ecr:`, `batch:`, `ops:`, `local:`). Run `task` (no args) for the full menu,
`task --list` for every task.

### Local dev (docker compose: postgres + mlflow + reports + api + ui + trainer)

```bash
task build                                  # build trainer+api+ui images + bring up the whole stack
task up                                     # bring up stack without rebuild
task data:split                             # generate data/training/DB-HISTORICA.xlsx (runs in trainer container)
task train VARIETIES=POP TUNING=smoke       # train (smoke ~1min); also dev/prod/prod_xl, VARIETIES=all, PARALLEL=N
task eda VARIETIES=POP                       # standalone EDA (no training)
task logs                                   # tail trainer + mlflow
task down                                   # stop services (preserves the pg-data volume)
```

Local URLs once up: UI `:8501`, API Swagger `:8000/docs`, MLflow `:5000`, reports/artifacts `:8080`.
There is **no `:80`** locally — that's the prod ALB.

### Lint

```bash
task lint          # == ruff check src/ main.py scripts/ tests/ api/ ui/   (ruff config in pyproject.toml; py312, 100 col)
```

`ruff` is the only formatter/linter (replaces black+isort+flake8+pyupgrade). Install dev deps with
`pip install -r requirements.txt -r requirements-dev.txt`.

### Tests

Suite P0 mínima en `tests/` (pytest, desde 2026-06-13): LagFeatureTransformer (pickle
round-trip, flags horneados, same-day ex-ante), select_champion (gate relativo),
conformal bands (cascade/cold-start), guard de registro y `_conformal_halfwidths` del API.
Correr DENTRO de los contenedores (mismas versiones que producción):

```bash
docker compose run --rm -v "$(pwd)/tests:/app/tests" --entrypoint sh trainer \
  -c "pip install -q pytest; PYTHONPATH=/app python -m pytest tests/ -q"
# el test del API corre en su contenedor (en trainer se salta solo):
docker compose run --rm --user root -v "$(pwd)/tests:/app/tests" --entrypoint sh api \
  -c "/opt/venv/bin/pip install -q pytest; cd /app && /opt/venv/bin/python -m pytest tests/test_api_conformal.py -q"
```

No committed `.github/workflows`. CI/CD infra (GHA OIDC → ECR/ECS deploy) vive como Terraform
pegable en `GUIA_MLOPS_AWS_V2.md` #3.10 (`modules/cicd/`); no hay carpeta `infra/` en este repo.
Cambios estructurales se validan además con el stack (`task build` → `task train …` → UI).

### Running the apps standalone (rarely needed; compose is the norm)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000   # from api/  (or: fastapi dev app/main.py)
streamlit run app/app.py                           # from ui/
python main.py --tuning dev --varieties POP        # trainer CLI (needs a venv + a reachable MLflow server)
```

### AWS (high-level shortcuts; granular tasks live in the namespaces)

```bash
task deploy        # full stand-up: storage -> 5 ECR images -> rest of infra
task smoke         # deploy + one Batch POP smoke job
task wake / sleep  # power the prod stack on/off (RDS+MLflow+reports+api+ui as a block; scheduler-driven cost model)
task status        # terraform outputs + cluster state + public URLs
task destroy / nuke  # DESTRUCTIVE (nuke also removes tfstate + OIDC)
```

## Non-obvious invariants (learned by reading across files — keep these intact)

1. **`src/` is the single source of truth for ML code.** `api/Dockerfile`'s build context is the
   **repo root** (not `api/`) specifically so it can `COPY src/` — the trainer and the API load the
   exact same pipeline code. Never vendor/copy ML logic into `api/`; it will silently drift.

2. **The pipeline always trains every backend in the registry; there is no flag to force one model**
   (ADR-002). `src/step_05_evaluate/champion.py::select_champion` decides the winner per variety via a
   strict lex-order: gap gate (`|gap|*100 <= CHAMPION_MAX_GAP`, a constraint — not minimized) →
   OOF business MAPE (honest generalization) → wall time. `full_mape` (in-sample) is informational
   only. Adding a backend = one new file in `src/step_04_train/` + its entry in `registry.py` and
   `search_spaces.py`; `tuning.py` is untouched. `--tuning smoke` runs never register models.

3. **The MLflow backend is ALWAYS Postgres + S3** (ADR-001/003). Never introduce `file://mlruns`,
   sqlite, or LocalStack. `mlruns/` in the tree is legacy.

4. **`step_XX_verbo/` module names encode pipeline order and must not be renamed.** Renaming breaks
   every `from src.step_X import …` *and* the paths baked into already-serialized `.joblib` pipelines.

5. **`ui/app/app.py` shadows the `app/` package under `streamlit run`.** The `sys.path.insert` shim at
   the top of that file fixes "`'app' is not a package'" — do not remove it. (`AppTest` masks the bug
   because it already runs with the project root on the path.) In the UI, **`views/` are the real pages**
   (registered via `st.navigation`); there is no `pages/` directory.

6. **In production the API is routed by specific path prefixes** (`/api/health*`, `/api/forecasts*`,
   `/api/varieties*`, `/api/history*`, `/docs`, …), **never a generic `/api/*`** — MLflow with
   `--serve-artifacts` owns `/api/2.0/mlflow-artifacts/*` and a wildcard would steal it. The UI calls the
   API by internal service discovery (`http://api.ml-training.local:8000`), not via the ALB.

7. **The `forecasts` Postgres DB is separate from the MLflow DB.** Locally both live in one Postgres
   (initdb creates `forecasts`). In prod the API reuses MLflow's RDS and auto-creates `forecasts` on first
   boot (`api/app/models/database.py::ensure_database`, idempotent).

8. **`MODEL_REGISTRY_PREFIX` default `rnd-forest-` is a contract** between trainer and API: the API loads
   models as `rnd-forest-<variety>`. Changing it requires coordinating both sides.

9. **Lag features are computed INSIDE the sklearn Pipeline** (`LagFeatureTransformer`, step 0 of
   `build_preprocessing_pipeline`) so each CV fold computes lags only from its own train slice —
   `data_loader.py` returns the 9 raw columns only. Do not move lag computation back to the loader:
   computing lags over the full dataset before the CV split leaks future information across folds.
   The transformer bakes its feature flags into `flags_` at fit time (self-contained serialization);
   never read env flags at transform time.

10. Both API and UI apps follow a layered structure under `app/` (`core/`, `services/`, `routers|views/`,
    `schemas/`, `client/` in the UI). The UI's `client/` layer mirrors the API's surface; keep them in sync.
