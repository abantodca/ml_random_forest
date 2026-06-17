"""
Servicio MLflow para gestión de modelos de ML
==============================================
Maneja la carga, predicción y administración de modelos desde MLflow

Ciclo de vida de un modelo:
1. En startup: si preload=True, carga modelos disponibles
2. En predicción: si no está cargado, carga bajo demanda via _get_or_load
3. En reload: verifica nuevas versiones y actualiza solo las que cambiaron
4. En shutdown: limpia caches
"""

import asyncio
import logging
import threading
from typing import Any, TypedDict

import httpx
import mlflow
import mlflow.exceptions
import mlflow.pyfunc
import pandas as pd

# `src.*` es un paquete a la raiz del backend (sibling de `app/`) que espeja
# los transformers de `ml_training`. Los modelos serializados en MLflow
# referencian rutas tipo `src.step_03_features.lag_features.LagFeatureTransformer`
# y pickle las resuelve via `import src.step_03_features.lag_features` al
# cargar. La importacion explicita aqui asegura el side-effect de registro
# en sys.modules antes de cualquier `mlflow.pyfunc.load_model`.
import src  # noqa: F401
from app.core import (
    ModelNotAvailableError,
    PredictionError,
    settings,
)
from app.services.uncertainty import predict_with_halfwidths

logger = logging.getLogger(__name__)


class ModelVersionInfo(TypedDict):
    """Información de una versión de modelo obtenida desde MLflow."""

    version: str
    run_id: str
    params: dict


class MLflowService:
    """
    Servicio para interactuar con MLflow y gestionar modelos de ML.

    Características:
    - Carga y cache de modelos en memoria
    - Predicciones individuales y batch
    - Recarga inteligente de modelos (solo si hay nueva versión)
    - Verificación de conectividad con MLflow
    """

    _MAX_REGISTERED_MODELS: int = settings.mlflow_max_registered_models
    _LATEST_VERSION_ONLY: int = 1

    def __init__(
        self,
        tracking_uri: str,
        experiment_prefix: str = "rnd-forest",
        preload: bool = True,
    ):
        """
        Inicializa el servicio MLflow.

        Args:
            tracking_uri: URL del servidor MLflow
            experiment_prefix: Prefijo de los nombres de modelos
            preload: Si precarga todos los modelos disponibles
        """
        self.tracking_uri = tracking_uri
        self.experiment_prefix = experiment_prefix
        self.preload = preload

        # Cache de modelos y versiones
        self._model_cache: dict[str, Any] = {}
        self._model_versions: dict[str, str] = {}
        self._info_cache: dict[str, dict] = {}
        # Cache de Winner_<VARIETY>.html: variety -> (run_id, html_content)
        self._dashboard_cache: dict[str, tuple[str, str]] = {}

        # Las mutaciones de los caches anteriores ocurren cross-thread: el
        # reload (`reload_models`) y el lazy-load (`predict` -> `_get_or_load`)
        # corren en el threadpool de Starlette (`run_in_executor`), así que un
        # reload concurrente con una predicción puede entrelazar el
        # read-modify-write de `_model_cache`/`_model_versions`/`_info_cache`.
        # Un RLock serializa esas secciones críticas (carga + actualización de
        # versión + invalidación de info) manteniéndolas atómicas entre sí.
        # Es reentrante porque `_load_model_if_new_version` -> `_load_version_if_needed`
        # anidan tomas del mismo lock.
        self._cache_lock = threading.RLock()

        # Configurar MLflow
        mlflow.set_tracking_uri(tracking_uri)

    # ========================================================================
    # Inicialización y Shutdown
    # ========================================================================

    def startup(self) -> None:
        """Precarga modelos si está habilitado."""
        if not self.preload:
            logger.info("⏸️  Precarga desactivada — modelos se cargarán bajo demanda")
            return

        available = self.get_available_models()

        if not available:
            logger.warning("⚠️  No se encontraron modelos en MLflow")
            logger.warning("   Los modelos se cargarán bajo demanda")
            return

        logger.info("📦 Modelos disponibles en MLflow: %d", len(available))
        logger.info("⏳ Precargando modelos...")

        loaded = 0
        failed = 0

        for variety in available:
            try:
                if self._load_model_if_new_version(variety):
                    loaded += 1
            except Exception as exc:
                failed += 1
                logger.debug(
                    "No se pudo cargar modelo '%s': %s", variety, exc, exc_info=True
                )

        if loaded > 0:
            logger.info("✅ Precarga completada: %d modelos cargados", loaded)
        if failed > 0:
            logger.warning("⚠️  %d modelos fallaron (se cargarán bajo demanda)", failed)

    def shutdown(self) -> None:
        """Limpia caches al detener el servicio."""
        with self._cache_lock:
            self._model_cache.clear()
            self._model_versions.clear()
            self._info_cache.clear()
            self._dashboard_cache.clear()
        logger.info("MLflow service shut down")

    # ========================================================================
    # Gestión de Modelos
    # ========================================================================

    def get_available_models(self) -> list[str]:
        """
        Obtiene la lista de modelos registrados en MLflow.

        Returns:
            Lista de nombres de variedades disponibles
        """
        try:
            client = mlflow.tracking.MlflowClient()
            registered_models = client.search_registered_models(
                max_results=self._MAX_REGISTERED_MODELS
            )

            available = []
            for model in registered_models:
                # `experiment_prefix` debe incluir el separador final.
                # Ej: prefix="productivity_" → "productivity_POP" -> "POP"
                if model.name.startswith(self.experiment_prefix):
                    variety = model.name[len(self.experiment_prefix):]
                    available.append(variety)

            return sorted(available)
        except Exception as exc:
            logger.error(
                "Error consultando modelos disponibles: %s", exc, exc_info=True
            )
            return []

    def is_loaded(self, variety: str) -> bool:
        """Verifica si un modelo está cargado en memoria."""
        return variety in self._model_cache

    @property
    def models_loaded(self) -> int:
        """Cantidad de modelos cargados en memoria."""
        return len(self._model_cache)

    async def check_connection(self) -> bool:
        """Verifica que el servidor MLflow responde (async para no bloquear)."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.tracking_uri}/health", timeout=2.0)
                return response.status_code == 200
        except Exception:
            return False

    async def reload_models(self) -> dict:
        """
        Recarga modelos que tengan nueva versión en MLflow.

        Cada variedad hace una sola llamada a MLflow (search_model_versions +
        get_run) en lugar de las tres anteriores.

        Returns:
            Diccionario con información del reload:
            {
                'models_loaded': int,
                'models_available': int,
                'summary': {'loaded', 'updated', 'unchanged', 'failed'},
                'models': [{'variety', 'version', 'status', 'message'}]
            }
        """
        loop = asyncio.get_running_loop()
        available = await loop.run_in_executor(None, self.get_available_models)
        logger.info("♻️  Verificando versiones de %d modelos...", len(available))

        models_info = []
        summary = {"loaded": 0, "updated": 0, "unchanged": 0, "failed": 0}

        for variety in available:
            try:
                info = await loop.run_in_executor(
                    None, self._reload_single_variety, variety
                )
                models_info.append(info)
                summary[info["status"]] += 1
            except Exception as exc:
                logger.error(
                    "Error recargando '%s': %s", variety, exc, exc_info=True
                )
                models_info.append({
                    "variety": variety,
                    "version": "N/A",
                    "status": "failed",
                    "message": str(exc),
                })
                summary["failed"] += 1

        self._log_reload_summary(summary)

        return {
            "models_loaded": len(self._model_cache),
            "models_available": len(available),
            "summary": summary,
            "models": sorted(models_info, key=lambda x: x["variety"]),
        }

    # ========================================================================
    # Predicciones
    # ========================================================================

    async def predict(
        self,
        variety: str,
        features_df: pd.DataFrame,
    ) -> list[float]:
        """Predice KGHORA dado el DataFrame ya enriquecido (40 columnas).

        El feature engineering (lags) NO se hace aqui: el caller
        construye el DataFrame con `FeaturePipeline.build_features` y
        lo pasa listo. Esa separacion mantiene a `MLflowService` enfocado
        en su responsabilidad (carga + prediccion contra MLflow) y deja
        el feature engineering en su propio servicio testeable.

        Raises:
            ModelNotAvailableError: si el modelo no esta cargado.
            PredictionError: si la inferencia falla.
        """
        return await self._predict_df(variety, features_df)

    async def predict_with_std(
        self,
        variety: str,
        features_df: pd.DataFrame,
    ) -> tuple[list[float], list[float] | None]:
        """Como `predict` pero devuelve ademas la std por fila.

        El pipeline de produccion es un `OOFEnsembleRegressor` (K=5 pipelines
        internos); la dispersion entre sus K predicciones es un proxy directo
        de incertidumbre del modelo (ver oof_ensemble.predict_with_std). La
        autopsia de errores 2026-06-11 mostro que el peor 5% del OOF son
        picos de temporada (sep/oct) impredecibles con las features actuales:
        esos errores no se pueden eliminar, pero SI senalizar — std alta =>
        revisar manualmente.

        Devuelve (preds, stds). `stds` es None si el modelo cargado no expone
        `predict_with_std` (p.ej. un pickle legacy pre-ensemble): los callers
        degradan a la respuesta sin banda, sin romper.
        """
        loop = asyncio.get_running_loop()
        try:
            preds, stds = await loop.run_in_executor(
                None, self._get_and_predict_with_std, variety, features_df
            )
            return (
                [float(p) for p in preds],
                [float(s) for s in stds] if stds is not None else None,
            )
        except (ModelNotAvailableError, PredictionError):
            raise
        except Exception as exc:
            raise PredictionError(variety, str(exc)) from exc

    def _get_and_predict_with_std(self, variety: str, df: pd.DataFrame):
        """Version con incertidumbre de `_get_and_predict` (threadpool).

        El modelo cacheado es un pyfunc wrapper; el sklearn real (el
        OOFEnsembleRegressor) vive en `_model_impl.sklearn_model`.

        Prioridad de la incertidumbre:
          1. `conformal_` (metadata calibrada con residuos OOF, por fundo,
             con deteccion de cold-start) — cobertura estadistica real.
          2. `predict_with_std` (±std del ensemble) — heuristica legacy,
             cobertura << nominal; solo para modelos sin conformal_.
          3. stds=None (pickles pre-ensemble).
        """
        model = self._get_or_load(variety)
        impl = getattr(model, "_model_impl", None)
        skl = getattr(impl, "sklearn_model", None) if impl is not None else None
        target = skl if skl is not None else model
        try:
            # Logica de incertidumbre extraida a uncertainty.py (P1.3):
            # conformal por fundo+cold-start -> ±1.96·std legacy -> None.
            return predict_with_halfwidths(target, df)
        except Exception as exc:
            raise PredictionError(variety, str(exc)) from exc

    # ========================================================================
    # Información de Modelos
    # ========================================================================

    def get_model_info(self, variety: str) -> dict:
        """Obtiene información y métricas del modelo.

        El resultado se cachea por variedad y se invalida automáticamente
        cuando `_load_version_if_needed` carga una versión nueva (vía reload o
        lazy-load), así que tras un `reload_models` que actualice la variedad
        el dashboard `/varieties/{v}` deja de servir métricas viejas. El fetch
        (red a MLflow) se hace FUERA del lock para no serializar round-trips de
        variedades distintas; el lock solo protege el read-modify-write del
        cache.
        """
        with self._cache_lock:
            cached = self._info_cache.get(variety)
        if cached is not None:
            return cached

        info = self._fetch_model_info(variety)
        with self._cache_lock:
            # Otro thread pudo poblarlo mientras hacíamos el fetch; respetamos
            # el primero que llegó para no pisar una versión más nueva.
            return self._info_cache.setdefault(variety, info)

    # ========================================================================
    # Winner dashboard (artifact HTML)
    # ========================================================================

    def get_winner_dashboard_html(self, variety: str) -> str:
        """Devuelve el HTML del reporte gerencial `Winner_<VARIETY>.html`.

        Localiza el run_id de la última versión del registered model en
        MLflow y descarga el artifact `winner_dashboard/Winner_<VARIETY>.html`
        a un directorio temporal del proceso. Cachea en memoria por (variety, run_id)
        para no rebajar el archivo en cada request.

        Raises:
            ModelNotAvailableError: si no hay versión registrada en MLflow.
            FileNotFoundError: si el run no contiene el artifact esperado.
        """
        with self._cache_lock:
            cache = self._dashboard_cache.get(variety)
        version_info = self.get_latest_version_info(variety)
        if not version_info:
            raise ModelNotAvailableError(variety)

        run_id = version_info["run_id"]
        if cache and cache[0] == run_id:
            return cache[1]

        # El nombre del HTML incluye un timestamp (Winner_<VAR>_<fecha>.html),
        # así que NO se puede asumir un nombre fijo: listamos el directorio
        # `winner_dashboard` del run y tomamos el .html (preferimos el que
        # contiene la variedad). Robusto a cualquier sufijo del trainer.
        try:
            client = mlflow.tracking.MlflowClient()
            entries = client.list_artifacts(run_id, "winner_dashboard")
            htmls = [e.path for e in entries if e.path.lower().endswith(".html")]
            if not htmls:
                raise FileNotFoundError(
                    f"Sin .html en winner_dashboard del run {run_id}"
                )
            artifact_path = next(
                (p for p in htmls if variety.upper() in p.upper()), htmls[0]
            )
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=artifact_path,
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise FileNotFoundError(
                f"No se pudo obtener el reporte HTML del run {run_id}: {exc}"
            ) from exc

        with open(local, encoding="utf-8") as fh:
            html = fh.read()

        with self._cache_lock:
            self._dashboard_cache[variety] = (run_id, html)
        return html

    # ========================================================================
    # Métodos Privados
    # ========================================================================

    async def _predict_df(self, variety: str, df: pd.DataFrame) -> list[float]:
        """
        Ejecuta predicción sobre un DataFrame en un thread pool.

        `_get_or_load` puede invocar `mlflow.pyfunc.load_model` (descarga de
        red + deserialización pickle) si el modelo aún no está en cache. Esa
        operación es bloqueante, así que se delega al threadpool junto con la
        inferencia — ambos pasos comparten el mismo `run_in_executor` para
        evitar un segundo round-trip de scheduling.

        En el caso caliente (modelo ya cacheado) `_get_or_load` solo hace un
        dict lookup bajo el RLock, que es insignificante.

        Raises:
            ModelNotAvailableError: Si el modelo no está disponible
            PredictionError: Si falla la predicción
        """
        loop = asyncio.get_running_loop()
        try:
            predictions = await loop.run_in_executor(
                None, self._get_and_predict, variety, df
            )
            return [float(p) for p in predictions]
        except (ModelNotAvailableError, PredictionError):
            raise
        except Exception as exc:
            raise PredictionError(variety, str(exc)) from exc

    def _get_and_predict(self, variety: str, df: pd.DataFrame):
        """Obtiene (o carga) el modelo y predice. Diseñado para correr en el
        threadpool de `_predict_df`: combina la carga potencialmente bloqueante
        con la inferencia en un solo `run_in_executor`, evitando que cualquiera
        de los dos congele el event loop.
        """
        model = self._get_or_load(variety)
        try:
            return model.predict(df)
        except Exception as exc:
            raise PredictionError(variety, str(exc)) from exc

    def _get_or_load(self, variety: str) -> Any:
        """Obtiene modelo del cache o lo carga si no existe.

        El lookup + carga condicional + relectura van bajo el lock para que un
        reload concurrente (otro thread) no deje ver un cache a medio
        actualizar. La consulta de versión a MLflow dentro de
        `_load_model_if_new_version` ocurre con el lock tomado; es aceptable
        porque el costo dominante (descarga del modelo) ya estaba serializado
        de facto y evita doble descarga de la misma variedad.
        """
        with self._cache_lock:
            if variety not in self._model_cache:
                self._load_model_if_new_version(variety)

            model = self._model_cache.get(variety)
            if model is None:
                raise ModelNotAvailableError(variety)

            return model

    def _fetch_latest_run(self, variety: str) -> tuple[Any, Any] | None:
        """Una sola consulta a MLflow: última versión registrada + su run.

        Centraliza el round-trip (`search_model_versions` + `get_run`) que
        comparten `get_latest_version_info` y `_fetch_model_info`, evitando
        dos llamadas distintas que pedían lo mismo. Devuelve `(version, run)`
        o `None` si la variedad no tiene versión registrada. No atrapa
        excepciones: cada caller decide cómo degradar.
        """
        client = mlflow.tracking.MlflowClient()
        model_name = f"{self.experiment_prefix}{variety}"
        versions = client.search_model_versions(
            f"name='{model_name}'",
            max_results=self._LATEST_VERSION_ONLY,
            order_by=["version_number DESC"],
        )
        if not versions:
            return None
        latest = versions[0]
        run = client.get_run(latest.run_id)
        return latest, run

    def get_latest_version_info(
        self, variety: str
    ) -> ModelVersionInfo | None:
        """Versión + run_id + params en una sola consulta a MLflow.

        Público porque colaboradores externos (p. ej. `DriftService`)
        necesitan el run_id de la última versión para invalidar su cache
        por (variety, run_id); exponerlo evita que alcancen un método
        privado de esta clase.

        Returns:
            ModelVersionInfo o None si no hay versión registrada.
        """
        try:
            fetched = self._fetch_latest_run(variety)
            if fetched is None:
                return None

            latest, run = fetched
            run_metrics = run.data.metrics
            run_params = run.data.params

            # Hiperparametros del pipeline entrenado: cualquier param con
            # prefix `regressor__` o `preprocessor__` (formato sklearn) o
            # con prefijo legacy `best_`. Se preserva el nombre completo
            # para que el consumidor sepa a que step pertenece.
            best_params = {
                k.replace("best_", ""): v
                for k, v in run_params.items()
                if k.startswith(("regressor__", "preprocessor__", "best_"))
            }

            # Mapeo de metricas al schema del frontend. Origenes:
            #   - mae/r2 modelo (KG/JR_H, target original):
            #     test  -> nested_cv_*_mean (out-of-fold honesto)
            #     train -> nested_cv_mae_train_mean / full_model_r2
            #   - mape (KG/JR, escala de negocio):
            #     test  -> business_oof_mape
            #     train -> business_insample_mape
            params = {
                "version": latest.version,
                "run_id": run.info.run_id,
                "model_type": run_params.get("model_type", "unknown"),
                "best_params": best_params,
                "metrics": {
                    "test_mae": run_metrics.get("nested_cv_mae_mean"),
                    "test_r2": run_metrics.get("nested_cv_r2_mean"),
                    "test_mape": run_metrics.get("business_oof_mape"),
                    "train_mae": run_metrics.get("nested_cv_mae_train_mean"),
                    "train_r2": run_metrics.get("full_model_r2"),
                    "train_mape": run_metrics.get("business_insample_mape"),
                },
            }

            return ModelVersionInfo(
                version=latest.version,
                run_id=latest.run_id,
                params=params,
            )
        except Exception as exc:
            logger.warning(
                "Error consultando versión/params para '%s': %s",
                variety,
                exc,
                exc_info=True,
            )
            return None

    def _load_model_if_new_version(self, variety: str) -> bool:
        """
        Carga el modelo solo si hay una nueva versión.

        Usa get_latest_version_info para obtener la versión con una
        única llamada a MLflow.

        Raises:
            ModelNotAvailableError: Si no se encuentra versión registrada
        """
        version_info = self.get_latest_version_info(variety)
        if not version_info:
            raise ModelNotAvailableError(variety)

        return self._load_version_if_needed(variety, version_info["version"])

    def _load_version_if_needed(self, variety: str, latest_version: str) -> bool:
        """
        Carga la versión indicada solo si difiere de la cacheada.

        Returns:
            True si se cargó una nueva versión, False si ya estaba al día.

        Raises:
            ModelNotAvailableError: Si falla la carga desde MLflow
        """
        # RLock reentrante: `_get_or_load` ya puede tenerlo tomado. Serializa
        # la comparación de versión + carga + escritura de los tres caches
        # para que queden consistentes entre sí frente a un reload concurrente.
        with self._cache_lock:
            cached_version = self._model_versions.get(variety)

            if cached_version == latest_version:
                return False

            try:
                model_name = f"{self.experiment_prefix}{variety}"
                uri = f"models:/{model_name}/{latest_version}"

                if cached_version:
                    logger.info(
                        "🔄 Actualizando '%s': v%s → v%s",
                        variety,
                        cached_version,
                        latest_version,
                    )
                else:
                    logger.info("📥 Cargando '%s' v%s", variety, latest_version)

                model = mlflow.pyfunc.load_model(uri)
                self._model_cache[variety] = model
                self._model_versions[variety] = latest_version
                # La info cacheada (métricas/best_params/version) corresponde a
                # la versión anterior: se invalida para que el próximo
                # `get_model_info` la recompute desde el registry. Sin esto,
                # `/varieties/{v}` serviría métricas obsoletas tras un reload.
                self._info_cache.pop(variety, None)

                return True

            except mlflow.exceptions.MlflowException as exc:
                logger.error("❌ No se pudo cargar '%s': %s", variety, exc)
                raise ModelNotAvailableError(variety) from exc

    def _reload_single_variety(self, variety: str) -> dict:
        """
        Verifica y recarga el modelo de una variedad si hay nueva versión.

        Returns:
            Dict con variety, version, status y message.

        Raises:
            ModelNotAvailableError: Si falla la carga desde MLflow
        """
        version_info = self.get_latest_version_info(variety)
        if not version_info:
            return {
                "variety": variety,
                "version": "N/A",
                "status": "failed",
                "message": "No se encontró versión en MLflow",
            }

        latest_version = version_info["version"]
        params = version_info["params"]
        cached_version = self._model_versions.get(variety)
        loaded = self._load_version_if_needed(variety, latest_version)

        if loaded:
            if cached_version:
                return {
                    "variety": variety,
                    "version": latest_version,
                    "status": "updated",
                    "message": f"Actualizado de v{cached_version} a v{latest_version}",
                    "training_params": params,
                }
            return {
                "variety": variety,
                "version": latest_version,
                "status": "loaded",
                "message": "Cargado exitosamente",
                "training_params": params,
            }

        return {
            "variety": variety,
            "version": latest_version,
            "status": "unchanged",
            "message": "Ya está en la última versión",
            "training_params": params,
        }

    def _log_reload_summary(self, summary: dict) -> None:
        """Registra el resumen de una operación de recarga."""
        if summary["loaded"] > 0 or summary["updated"] > 0:
            logger.info(
                "✅ %d nuevos, %d actualizados", summary["loaded"], summary["updated"]
            )
        if summary["unchanged"] > 0:
            logger.info("⏭️  %d sin cambios", summary["unchanged"])
        if summary["failed"] > 0:
            logger.warning("❌ %d fallaron", summary["failed"])

    def _fetch_model_info(self, variety: str) -> dict:
        """Info del modelo desde el REGISTRY (no in-memory): version, model_type,
        métricas curadas y best_params. Reusa `get_latest_version_info` (una sola
        consulta) y alimenta el dashboard MLOps de `GET /varieties/{variety}`.

        Devuelve `{}` si no hay versión registrada.
        """
        info = self.get_latest_version_info(variety)
        if info is None:
            return {}
        params = info["params"]
        try:
            version = int(params["version"])
        except (TypeError, ValueError):
            version = None
        return {
            "version": version,
            "model_type": params.get("model_type"),
            "metrics": {k: v for k, v in params["metrics"].items() if v is not None},
            "best_params": params.get("best_params", {}),
        }
