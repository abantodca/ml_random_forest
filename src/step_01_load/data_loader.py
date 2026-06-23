"""Carga del dataset de entrenamiento desde Excel.

Devuelve un DataFrame con las columnas RAW (incluyendo la columna de fecha).
La derivacion de features ciclicas y el one-hot ocurren mas adelante, en el
preprocesador, para que `fit` solo vea train fold y no haya fugas.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    DATE_COLUMN,
    DEFAULT_VARIETIES,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    RARE_GROUP_COLS,
    RARE_MIN_COUNT,
    RAW_FEATURE_COLUMNS,
    TARGET,
    TRAINING_FILE,
    USELESS_COLUMNS,
)
from src.step_01_load.validation import validate_dataset

# Logger inerte hasta que el caller (main.py) configure handlers via
# `setup_logging()`. Usar `getLogger(__name__)` evita side-effects al
# importar este modulo desde tests u otros entrypoints.
logger = logging.getLogger(__name__)


def list_varieties(path: str | Path | None = None) -> list[str]:
    """Devuelve los nombres de hoja (variedades) presentes en el Excel."""
    file_path = Path(path) if path else TRAINING_FILE
    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {file_path}")
    return pd.ExcelFile(file_path).sheet_names


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Quita espacios en blanco a los nombres de columna (defensivo)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _resolve_default_sheet() -> str:
    """Toma la PRIMERA variedad de DEFAULT_VARIETIES como fallback.

    DEFAULT_VARIETIES puede ser CSV ("POP,JUPITER") o "all" para uso CLI.
    Como sheet de Excel no acepta esos formatos, el fallback se queda con
    el primer nombre concreto. En produccion casi nunca aplica: los
    callers (single_run, variety_runner) pasan sheet=variety explicito.
    """
    raw = (DEFAULT_VARIETIES or "").strip()
    if not raw or raw.lower() == "all":
        # Sin info concreta: el caller debe pasar sheet=. Si no, pandas
        # explotara con un mensaje claro al intentar leer la hoja vacia.
        return raw
    return raw.split(",", 1)[0].strip()


def _load_sheet_aligned(
    path: str | Path | None,
    sheet: str | None,
) -> tuple[pd.DataFrame, Path, str]:
    """Lee Excel + normaliza columnas + dropna(TARGET) + reset_index.

    Devuelve (df, file_path, sheet_name) con la misma alineacion fila a
    fila que `load_data` y `load_business_columns`. Centralizar este paso
    aqui evita que cualquier consumidor cambie el filtro en uno y olvide
    el otro -> KG/JR quedaria desalineado con (X, y) silenciosamente.

    `sheet` siempre lo pasan los callers reales (single_run, variety_runner)
    con la variedad concreta. El fallback solo aplica en uso interactivo.
    """
    file_path = Path(path) if path else TRAINING_FILE
    sheet_name = sheet or _resolve_default_sheet()

    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {file_path}")

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = _normalize_columns(df)
    if TARGET not in df.columns:
        raise ValueError(
            f"Columna target '{TARGET}' faltante en {file_path.name}/{sheet_name}. "
            f"Disponibles: {list(df.columns)}"
        )
    n_before = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    n_dropped_target = n_before - len(df)
    if n_dropped_target:
        logger.warning(f"{n_dropped_target} filas descartadas por target NaN ({sheet_name})")
    # EDA POP 2026-05-09: duplicados estructurales (mismas filas) inflaban CV
    # metrics. Detecta filas identicas sobre TODAS las columnas; mantiene el
    # PRIMERO (orden Excel = orden temporal en POP). Desde 2026-06-15 el dedup
    # CANONICO vive en scripts/prepare_data.split_workbook (1x al generar
    # DB-HISTORICA.xlsx); este queda DEFENSIVO: no-op si la data entra limpia,
    # pero protege recargas directas del Excel crudo. Si dispara el WARNING de
    # abajo, es senal de que alguien salteo el split.
    n_pre_dedup = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    n_dropped_dup = n_pre_dedup - len(df)
    if n_dropped_dup:
        logger.warning(f"{n_dropped_dup} duplicados estructurales descartados ({sheet_name})")
    return df, file_path, sheet_name


def load_data(
    path: str | Path | None = None,
    sheet: str | None = None,
    rare_min_count: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Carga el dataset y devuelve (X_raw, y).

    - Lee Excel y normaliza nombres de columna.
    - Valida presencia de target + features.
    - Elimina filas con target NaN (no se pueden usar para entrenar).
    - Castea categoricas a string y numericas a float.
    - Mantiene la columna de fecha como datetime para derivadas posteriores.
    - Loggea explicitamente las columnas excluidas por leakage o nula info.

    `rare_min_count` (P0.2, VarietyConfig): umbral por VARIEDAD para
    colapsar categorias raras en 'OTROS'. None = config.RARE_MIN_COUNT.
    OJO: el training y la recarga para reportes (variety_runner) deben
    usar el MISMO valor o las categorias no calzan con el pipeline.
    """
    df, file_path, sheet_name = _load_sheet_aligned(path, sheet)

    logger.info(f"Leyendo {file_path.name} | hoja={sheet_name}")

    needed = [TARGET, *RAW_FEATURE_COLUMNS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas faltantes en {file_path.name}/{sheet_name}: {missing}. "
            f"Disponibles: {list(df.columns)}"
        )

    leakage_present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if leakage_present:
        logger.info(f"Descartadas por LEAKAGE (no van al modelo): {leakage_present}")

    useless_present = [c for c in USELESS_COLUMNS if c in df.columns]
    if useless_present:
        # MES se descarta como columna raw; su informacion se reinyecta como
        # ciclicas (MES_SIN/COS, MES_SIN2/COS2, MES_SIN3/COS3) y SEMANA_SIN/COS
        # que FeatureGenerator deriva de FECHA durante el fit del Pipeline.
        # DIA_SEM se descarta sin reemplazo (auditoria 2026-05-05: corr ~0
        # con target, η²=0.0014). VARIEDAD se elimina del todo: ya entrenamos
        # un modelo por variedad.
        logger.info(
            f"Descartadas como raw {useless_present} "
            f"(MES se reinyecta como ciclicas desde FECHA en FeatureGenerator)"
        )

    df = df.loc[:, needed].copy()

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()

    # Agrupa categorias raras en 'OTROS' para evitar one-hot con n insuficiente.
    # En POP, CLAMSHELL 6 OZ tiene n=27 y MAPE=40.7%: el modelo no puede
    # aprender un dummy con tan poca muestra y solo introduce ruido.
    # Umbrales en config.py (RARE_MIN_COUNT, RARE_GROUP_COLS) salvo
    # override por variedad (parametro rare_min_count, P0.2).
    min_count = rare_min_count if rare_min_count is not None else RARE_MIN_COUNT
    for col in RARE_GROUP_COLS:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        rare = counts[counts < min_count].index.tolist()
        if rare:
            df[col] = df[col].where(~df[col].isin(rare), other="OTROS")
            logger.info(
                f"Agrupados en 'OTROS' por n<{min_count} en {col}: "
                f"{rare} (filas afectadas={int(df[col].eq('OTROS').sum())})"
            )

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    n_no_date = df[DATE_COLUMN].isna().sum()
    if n_no_date:
        logger.warning(f"{n_no_date} filas sin fecha valida (se imputaran al transformar)")

    # Validacion de schema antes de pasar al Pipeline. Detecta tipos
    # incorrectos, rangos imposibles, duplicados estructurales, etc.
    # Solo loguea warnings (las columnas requeridas ya se validaron
    # arriba con `needed`/`missing`).
    validate_dataset(df)

    # NOTA: las lag features se calculan AHORA dentro del Pipeline via
    # `LagFeatureTransformer` (step 0 de `build_preprocessing_pipeline`).
    # Esto elimina el leakage moderado que existia cuando los lags se
    # calculaban aqui sobre el dataset completo antes del CV split.
    # `data_loader` deja X con las 9 columnas raw; el transformer las
    # expande a ~38 dentro del fold de cada modelo.

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)

    logger.info(
        f"Dataset cargado | filas={len(df)} | features_raw={X.shape[1]} | "
        f"target='{TARGET}' (mean={y.mean():.3f}, std={y.std():.3f}, "
        f"p95={y.quantile(0.95):.3f})"
    )
    logger.info(f"X (features raw) = {list(X.columns)}")
    logger.info(
        f"y (target) = '{TARGET}' | lags FF/F/FMT y ciclicas de FECHA se "
        f"computan dentro del Pipeline (LagFeatureTransformer + FeatureGenerator)"
    )
    return X, y


def load_business_columns(
    path: str | Path | None = None,
    sheet: str | None = None,
) -> pd.DataFrame:
    """Carga las columnas KG/JR y H-EF alineadas por indice con load_data().

    Comparte el helper `_load_sheet_aligned` con `load_data` para garantizar
    alineacion 1:1 fila-a-fila con el (X, y) que devuelve load_data: si el
    filtro de target cambia en un sitio cambia en ambos.

    Devuelve un DataFrame con columnas ['KG/JR', 'H-EF']; las columnas que no
    existan en el Excel salen como NaN (defensivo: el caller debe chequear).

    Why: KG/JR y H-EF son LEAKAGE para entrenar (config.LEAKAGE_COLUMNS), pero
    necesarias para validar el modelo en la unidad de negocio:
        KG/JR_estimado = pred(KG/JR_H) * H-EF       (mide step_06)
    """
    df, _, _ = _load_sheet_aligned(path, sheet)

    out = pd.DataFrame(index=df.index)
    for col in LEAKAGE_COLUMNS:  # ['KG/JR', 'H-EF']
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else float("nan")
    return out
