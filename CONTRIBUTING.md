# Contribuir a ml_training

Guía de onboarding para desarrolladores. El **qué/por qué** del ML vive en
`README.md`; las **invariantes** en `CLAUDE.md`; el **runbook AWS** en
`GUIA_MLOPS_AWS_V2.md`; la **vista visual** en `ARCHITECTURE.md`. Esta guía es
el **cómo trabajar** en el repo.

## 1. Setup local (una vez)

```bash
# Dev deps (linter, hooks, tests, type-checker)
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install                 # instala los git hooks (ruff + higiene)

# Stack completo (postgres + mlflow + reports + api + ui + trainer)
task build                         # build de imágenes + levanta todo
task data:split                    # genera data/training/DB-HISTORICA.xlsx
task train VARIETIES=POP TUNING=smoke   # sanity ~1 min
```

URLs: UI `:8501` · API `:8000/docs` · MLflow `:5000` · reports `:8080`.
`task` (sin args) lista el menú; `task --list` muestra todo.

## 2. Ciclo de trabajo

1. **Rama** desde `main` (`git switch -c tipo/descripcion`).
2. Edita. Los hooks de `pre-commit` formatean/lint en cada commit.
3. **Lint** manual: `task lint` (== `ruff check …`). Formato: `ruff format .`.
4. **Tests P0** (DENTRO del contenedor, mismas versiones que prod):
   ```bash
   docker compose run --rm -v "$(pwd)/tests:/app/tests" --entrypoint sh trainer \
     -c "pip install -q pytest; PYTHONPATH=/app python -m pytest tests/ -q"
   # El test del API corre en su propio contenedor (en trainer se salta solo):
   docker compose run --rm --user root -v "$(pwd)/tests:/app/tests" --entrypoint sh api \
     -c "/opt/venv/bin/pip install -q pytest; cd /app && /opt/venv/bin/python -m pytest tests/test_api_conformal.py -q"
   ```
5. Cambios estructurales: validar además con el stack
   (`task build` → `task train …` → revisar UI/API/reports).
6. **Commit** en español, `tipo(scope): resumen` (ver §4).

## 3. Convenciones de código

- **Idioma:** comentarios y docs en **español**. Referencias a secciones como
  `#N`, no `§N`.
- **Estilo:** `ruff` es el único formatter+linter (reemplaza black+isort+flake8+
  pyupgrade). Python 3.12, 100 columnas, comillas dobles. Config en `pyproject.toml`.
- **Typing:** gradual y opt-in (`mypy`, ver `pyproject.toml [tool.mypy]`). Aún no
  es gate. Al anotar un módulo, sube su rigor con un `override`.
- **Lee como el código vecino:** densidad de comentarios, naming e idioma del
  módulo que tocas.

## 4. Mensajes de commit

`tipo(scope): resumen en imperativo`. Tipos: `feat fix refactor perf chore docs
test`. Scope = área (`tuning`, `api`, `ui`, `taskfile`, `tooling`…). El cuerpo
explica el **porqué**, no el qué. Ejemplos reales del historial:
`perf(tuning): piso LR 1e-2 …`, `fix(taskfile): TUNING default prod_xl …`.

## 5. ⚠️ Checklist de invariantes (antes de un PR estructural)

Romper uno de estos invalida modelos en producción o despliegues. Detalle en
`CLAUDE.md` #1–#10.

- [ ] **No vendoricé lógica ML fuera de `src/`** — es la única fuente de verdad;
      la API la `COPY`a desde la raíz (#1).
- [ ] **No renombré ni moví paquetes `step_XX_verbo/`** — sus paths están
      horneados en los `.joblib` serializados y en los imports (#4).
- [ ] **No introduje `file://mlruns`, sqlite ni LocalStack** — MLflow es
      **siempre** Postgres + S3 (#3, ADR-001/003).
- [ ] **Los lags siguen DENTRO del Pipeline** (`LagFeatureTransformer`), no en el
      loader — mover rompe el anti-leakage de CV (#9).
- [ ] **No cambié `MODEL_REGISTRY_PREFIX` (`rnd-forest-`)** sin coordinar
      trainer **y** API (#8).
- [ ] **API ruteada por prefijos específicos**, nunca `/api/*` genérico — colisiona
      con `--serve-artifacts` de MLflow (#6).
- [ ] **UI: no removí el shim `sys.path.insert` de `ui/app/app.py`** (#5); las
      páginas viven en `views/`, no en `pages/`.
- [ ] **Si toqué la superficie de la API, sincronicé `ui/app/client/`** (#10).
- [ ] **¿Nuevo backend de modelo?** Un archivo en `src/step_04_train/` + entrada
      en `registry.py` y `search_spaces.py`. `tuning.py` queda intacto (#2).

## 6. Estructura del repo (mapa rápido)

| Carpeta | Qué |
|---|---|
| `src/` | pipeline ML (única fuente de verdad) — `step_01..06`, `orchestration/`, `pipeline/`, `diagnostics/` |
| `api/app/` | FastAPI en capas (`routers/ services/ models/ crud/ schemas/`) |
| `ui/app/` | Streamlit en capas (`views/ client/`) |
| `infra/` | Terraform por módulo (`network storage mlflow api ui reports batch …`) |
| `scripts/` | módulos importables desde `main.py` + tasks (`prepare_data`, `s3_sync`) |
| `tests/` | suite P0 (correr en contenedor) |
| `tasks/` | taskfiles namespaced incluidos por `Taskfile.yml` |
| `docs/` | planes y notas de diseño |

El árbol detallado y el flujo del pipeline: `README.md` #197 y #264.
Los diagramas visuales: `ARCHITECTURE.md`.
