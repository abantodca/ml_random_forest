"""Entrypoint del modulo EDA.

Uso programatico:
    from src.diagnostics import run_eda
    out_path = run_eda("POP")
    print(out_path)  # reports/EDA_POP_2026-05-09_12-34.html

Uso CLI (dentro del container):
    docker compose run --rm --no-deps --entrypoint python trainer \\
      -m src.diagnostics.eda --variety POP

El proceso:
    1. Carga la data raw via `step_01_load.data_loader.load_data(sheet=variety)`.
    2. Genera profile univariado por cada NUMERIC_FEATURE + TARGET.
    3. Genera profile temporal por TARGET (y opcionalmente top features).
    4. Calcula multivariado (correlation, VIF, MI con target).
    5. Calcula drift PSI por anio para todas las numericas.
    6. Sintetiza hallazgos en regla-based: top 5 con severity.
    7. Renderiza HTML en `reports/EDA_<variety>_<ts>.html`.
    8. Si hay un MLflow run activo, sube el HTML como artifact.
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    CORRELATION_HIGH_THRESHOLD,
    DATE_COLUMN,
    EDA_KURT_HIGH,
    EDA_KURT_WARN,
    EDA_SKEW_HIGH,
    NUMERIC_FEATURES,
    OUTLIER_FRACTION_WARN,
    REPORTS_DIR,
    SKEW_THRESHOLD,
    TARGET,
    init_dirs,
)
from src.diagnostics.categorical import (
    CategoricalReport,
    build_categorical_report,
)
from src.diagnostics.distributions import VariableProfile, profile_all_numeric
from src.diagnostics.html_renderer import render_eda_html, write_eda_html
from src.diagnostics.multivariate import (
    compute_mutual_information,
    compute_vif,
    correlation_matrix,
)
from src.diagnostics.plots import (
    acf_pacf_bars,
    boxplot_by_group,
    correlation_heatmap,
    histogram_with_kde,
    mi_bars,
    psi_heatmap,
    qq_plot,
    vif_bars,
)
from src.diagnostics.temporal import (
    DriftReport,
    TemporalProfile,
    drift_report,
    profile_temporal,
)
from src.step_01_load.data_loader import load_data

logger = logging.getLogger(__name__)


def _write_eda_sidecar(
    path: Path,
    *,
    variety: str,
    quality: dict,
    var_profiles,
    target_temporal,
    corr,
    vif_results,
    mi_results,
    drift_reports,
    cat_report,
    findings,
) -> None:
    """Emite JSON con todos los stats crudos del EDA.

    Pensado para que el feature engineering posterior consuma datos reales
    (skew, missing %, lags significativos, Cramer's V por categorica, etc)
    sin parsear el HTML. Convierte dataclasses con `asdict`; numpy arrays /
    Series se convierten a listas / floats Python primitivos.
    """
    import json
    from dataclasses import asdict

    def _safe(obj):
        # Recursivo: maneja dataclasses, dicts, lists, np types y pandas
        try:
            import numpy as np  # local import
        except Exception:
            np = None  # type: ignore[assignment]
        if obj is None:
            return None
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _safe(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            # JSON solo soporta keys string/int/float/bool/None: tuples (ej.
            # DriftReport.psi_values con (year_a, year_b)) se serializan como
            # "year_a-year_b". Aplicamos lo mismo a otros tipos no triviales.
            def _safe_key(k):
                if isinstance(k, tuple):
                    return "-".join(str(_safe(x)) for x in k)
                if isinstance(k, (str, int, float, bool)) or k is None:
                    return k
                return str(k)

            return {_safe_key(k): _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        if np is not None:
            if isinstance(obj, (np.floating,)):
                v = float(obj)
                return None if (v != v or v in (float("inf"), float("-inf"))) else v
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return _safe(obj.tolist())
        if isinstance(obj, float):
            return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
        return obj

    payload = {
        "variety": variety,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "quality": _safe(quality),
        "numeric_profiles": [_safe(p) for p in var_profiles],
        "target_temporal": _safe(target_temporal),
        "categorical": _safe(cat_report) if cat_report is not None else None,
        "multivariate": {
            "correlation_method": corr.method,
            "high_corr_pairs": [{"a": a, "b": b, "r": float(r)} for a, b, r in corr.high_pairs],
            "vif": [_safe(v) for v in vif_results],
            "mutual_info": [_safe(m) for m in mi_results],
        },
        "drift": [_safe(d) for d in drift_reports],
        "findings": [{"severity": s, "message": m} for s, m in findings],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def find_latest_eda_sidecar(variety: str, reports_dir: Path | None = None) -> Path | None:
    """Devuelve el path del JSON sidecar mas reciente para una variedad.

    Busca en `reports/EDA_<variety>_<ts>.json`. None si no hay ninguno.
    El sidecar tiene drift PSI, findings, stats que MLflow puede tagear.
    """
    rdir = Path(reports_dir) if reports_dir else REPORTS_DIR
    if not rdir.exists():
        return None
    candidates = sorted(
        rdir.glob(f"EDA_{variety}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def extract_drift_summary(sidecar_path: Path) -> dict:
    """Lee el JSON sidecar y resume drift + findings para tags MLflow.

    Devuelve dict con (todos string para que MLflow lo acepte):
      drift_psi_max         : float "X.XX" — peor PSI entre anios.
      drift_severe_count    : int "N" — vars con PSI>0.25.
      drift_severe_vars     : str "var1,var2" — nombres separados por coma.
      eda_findings_high     : int — cantidad de findings severity=high.
      eda_generated_at      : timestamp del EDA.
      eda_n_rows            : int del dataset al momento del EDA.
    """
    import json

    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    drift = data.get("drift", []) or []
    severe = [d for d in drift if d.get("drift_severity") == "severo"]
    psi_vals = [
        v
        for v in (d.get("max_psi") for d in drift)
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    max_psi = max(psi_vals, default=0.0)

    findings = data.get("findings", []) or []
    n_high = sum(1 for f in findings if f.get("severity") == "high")

    quality = data.get("quality", {}) or {}

    return {
        "drift_psi_max": f"{max_psi:.4f}",
        "drift_severe_count": str(len(severe)),
        "drift_severe_vars": ",".join(d["variable"] for d in severe),
        "eda_findings_high": str(n_high),
        "eda_generated_at": str(data.get("generated_at", "")),
        "eda_n_rows": str(quality.get("n_rows", "")),
    }


def _quality_metrics(df: pd.DataFrame) -> dict:
    """Metricas top-level de calidad de datos."""
    return {
        "n_rows": len(df),
        "n_cols_raw": df.shape[1],
        "n_duplicates": int(df.duplicated().sum()),
        "n_missing_total": int(df.isna().sum().sum()),
        "missing_ratio_global": f"{df.isna().mean().mean():.2%}",
        "n_unique_FUNDO": int(df["FUNDO"].nunique()) if "FUNDO" in df.columns else 0,
        "n_unique_FORMATO": int(df["FORMATO"].nunique()) if "FORMATO" in df.columns else 0,
        "fecha_min": str(df[DATE_COLUMN].min()) if DATE_COLUMN in df.columns else "—",
        "fecha_max": str(df[DATE_COLUMN].max()) if DATE_COLUMN in df.columns else "—",
    }


def _synthesize_findings(
    var_profiles: list[VariableProfile],
    target_temporal: TemporalProfile,
    high_corr_pairs: list[tuple],
    vif_results,
    drift_reports: list[DriftReport],
) -> list[tuple[str, str]]:
    """Reglas heuristicas que producen los hallazgos top.

    Devuelve [(severity, message)] con severity in {"high", "medium", "low", "good"}.
    """
    findings: list[tuple[str, str]] = []

    # Heteroscedasticidad latente: variables muy skewed o con kurt alta
    for p in var_profiles:
        if abs(p.skew) > SKEW_THRESHOLD or abs(p.kurtosis) > EDA_KURT_WARN:
            sev = (
                "high"
                if abs(p.skew) > EDA_SKEW_HIGH or abs(p.kurtosis) > EDA_KURT_HIGH
                else "medium"
            )
            findings.append(
                (
                    sev,
                    f"{p.name} muestra distribucion fuertemente sesgada "
                    f"(skew={p.skew:+.2f}, kurt={p.kurtosis:+.2f}). "
                    f"Recomendacion: {p.boxcox_recommendation}.",
                )
            )

    # Outliers altos
    for p in var_profiles:
        if p.n > 0 and p.n_outliers_iqr / max(p.n, 1) > OUTLIER_FRACTION_WARN:
            findings.append(
                (
                    "medium",
                    f"{p.name} tiene {p.n_outliers_iqr} outliers IQR "
                    f"({p.n_outliers_iqr / p.n:.1%}). "
                    f"OutlierCapper actual deberia manejarlos; revisar bounds por FUNDO.",
                )
            )

    # Autocorrelacion del target sin remover (estacional)
    if target_temporal.durbin_watson.statistic is not None:
        dw = target_temporal.durbin_watson.statistic
        if dw < 1.5:
            findings.append(
                (
                    "high",
                    f"Target {TARGET} presenta autocorrelacion positiva fuerte (DW={dw:.2f}). "
                    "Confirma necesidad de lag features y CV temporal honesto.",
                )
            )
        elif dw > 2.5:
            findings.append(
                (
                    "medium",
                    f"Target {TARGET} presenta autocorrelacion negativa (DW={dw:.2f}). "
                    "Posible sobre-dispersion; revisar transformacion del target.",
                )
            )

    # Estacionariedad del target
    adf = target_temporal.adf
    kpss = target_temporal.kpss
    if adf.rejects_h0 and kpss.rejects_h0 is False:
        findings.append(
            (
                "good",
                f"Target {TARGET} estacionaria (ADF rechaza, KPSS no rechaza).",
            )
        )
    elif adf.rejects_h0 is False and kpss.rejects_h0:
        findings.append(
            (
                "high",
                f"Target {TARGET} NO estacionaria (ADF no rechaza, KPSS rechaza). "
                "Considerar diferenciacion o detrending si los lag features no alcanzan.",
            )
        )

    # Multicolinealidad alta
    high_vif = [r for r in vif_results if r.severity == "high"]
    if high_vif:
        top = ", ".join(r.feature for r in high_vif[:5])
        findings.append(
            (
                "high",
                f"{len(high_vif)} features con VIF>10 (multicolinealidad severa). Top: {top}.",
            )
        )

    # Correlation pairs >0.95 (redundancia casi perfecta)
    very_high = [(a, b, r) for a, b, r in high_corr_pairs if abs(r) >= 0.95]
    if very_high:
        findings.append(
            (
                "medium",
                f"{len(very_high)} pares de features con |corr| ≥ 0.95 — redundancia casi total.",
            )
        )

    # Drift severo
    severe_drift = [r for r in drift_reports if r.drift_severity == "severo"]
    if severe_drift:
        names = ", ".join(r.variable for r in severe_drift[:5])
        findings.append(
            (
                "high",
                f"{len(severe_drift)} variables con PSI > 0.25 entre anios consecutivos: {names}. "
                "Inestabilidad temporal — revisar drift / cambio de regimen.",
            )
        )

    # Si no hay hallazgos, mensaje positivo
    if not findings:
        findings.append(
            (
                "good",
                "Sin hallazgos criticos automaticos. Revisar tarjetas detalladas para nuances.",
            )
        )

    # Cap a top 8 ordenados por severidad
    severity_order = {"high": 0, "medium": 1, "low": 2, "good": 3}
    findings.sort(key=lambda f: severity_order.get(f[0], 9))
    return findings[:8]


def run_eda(variety: str, out_dir: Path | None = None, *, log_to_mlflow: bool = True) -> Path:
    """Ejecuta el EDA completo para una variedad y devuelve el path al HTML."""
    init_dirs()
    out_dir = out_dir or REPORTS_DIR
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = out_dir / f"EDA_{variety}_{ts}.html"

    # ---- 1. Carga raw ----
    logger.info(f"[EDA/{variety}] cargando data raw...")
    X, y = load_data(sheet=variety)
    df = X.copy()
    df[TARGET] = y.values
    if DATE_COLUMN in df.columns:
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # ---- 2. Calidad ----
    quality = _quality_metrics(df)

    # ---- 3. Univariado ----
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + [TARGET]
    logger.info(f"[EDA/{variety}] univariado sobre {len(numeric_cols)} variables...")
    var_profiles = profile_all_numeric(df, numeric_cols)
    var_profiles_with_figs = [
        (
            p,
            histogram_with_kde(df[p.name], p.name),
            qq_plot(df[p.name], p.name),
            boxplot_by_group(df, p.name, "FUNDO", p.name)
            if "FUNDO" in df.columns
            else boxplot_by_group(df, p.name, CATEGORICAL_FEATURES[0], p.name),
        )
        for p in var_profiles
    ]

    # ---- 4. Temporal del target ----
    logger.info(f"[EDA/{variety}] temporal sobre target...")
    target_series = df[TARGET]
    target_temporal = profile_temporal(TARGET, target_series, period=12)
    temporal_profiles_with_figs = [
        (
            target_temporal,
            acf_pacf_bars(
                target_temporal.acf,
                target_temporal.pacf,
                n=len(target_series.dropna()),
                name=TARGET,
            ),
        )
    ]
    # Tambien temporal de top-3 numericas con mayor varianza
    top_var_cols = sorted(
        [(c, df[c].var()) for c in NUMERIC_FEATURES if c in df.columns],
        key=lambda t: t[1] or 0,
        reverse=True,
    )[:3]
    for c, _ in top_var_cols:
        prof = profile_temporal(c, df[c], period=12)
        temporal_profiles_with_figs.append(
            (
                prof,
                acf_pacf_bars(prof.acf, prof.pacf, n=len(df[c].dropna()), name=c),
            )
        )

    # ---- 5. Multivariado ----
    logger.info(f"[EDA/{variety}] multivariado...")
    numeric_only = df[[c for c in numeric_cols if c != TARGET]]
    corr = correlation_matrix(
        numeric_only,
        method="spearman",
        high_threshold=CORRELATION_HIGH_THRESHOLD,
    )
    vif_results = compute_vif(numeric_only)
    mi_results = compute_mutual_information(numeric_only, df[TARGET])

    # ---- 5-bis. Categoricas (FORMATO, FUNDO, ...) ----
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    cat_report: CategoricalReport | None = None
    if cat_cols:
        logger.info(f"[EDA/{variety}] categoricas sobre {len(cat_cols)} columnas: {cat_cols}...")
        cat_report = build_categorical_report(df, cat_cols, df[TARGET], top_n=15)

    # ---- 6. Drift por anio ----
    logger.info(f"[EDA/{variety}] drift entre anios...")
    drift_reports = [
        drift_report(df, c, year_col="ANIO", date_col=DATE_COLUMN)
        for c in numeric_cols
        if c != TARGET
    ]
    drift_reports = [r for r in drift_reports if r.year_pairs]

    # ---- 7. Findings ----
    findings = _synthesize_findings(
        var_profiles=var_profiles,
        target_temporal=target_temporal,
        high_corr_pairs=corr.high_pairs,
        vif_results=vif_results,
        drift_reports=drift_reports,
    )

    # ---- 8. Render ----
    html = render_eda_html(
        variety=variety,
        n_rows=len(df),
        n_cols=df.shape[1],
        quality_metrics=quality,
        findings=findings,
        var_profiles_with_figs=var_profiles_with_figs,
        temporal_profiles_with_figs=temporal_profiles_with_figs,
        corr_fig=correlation_heatmap(corr.columns, corr.matrix, corr.method),
        vif_fig=vif_bars(vif_results),
        mi_fig=mi_bars(mi_results),
        psi_fig=psi_heatmap(drift_reports),
        high_corr_pairs=corr.high_pairs,
        categorical_report=cat_report,
        out_path=out_path,
    )

    write_eda_html(html, out_path)
    logger.info(f"[EDA/{variety}] HTML generado: {out_path}")

    # ---- 8-bis. JSON sidecar con stats crudos ----
    # Sirve para que el feature engineering / LLM lea datos reales sin
    # parsear HTML. Mismo basename que el HTML, extension .json.
    json_path = out_path.with_suffix(".json")
    _write_eda_sidecar(
        json_path,
        variety=variety,
        quality=quality,
        var_profiles=var_profiles,
        target_temporal=target_temporal,
        corr=corr,
        vif_results=vif_results,
        mi_results=mi_results,
        drift_reports=drift_reports,
        cat_report=cat_report,
        findings=findings,
    )
    logger.info(f"[EDA/{variety}] JSON sidecar: {json_path}")

    # ---- 9. MLflow artifact ----
    # Fix 2026-06-13: el modo standalone (`task eda` / main --eda) NUNCA
    # abre un run, asi que el `if active_run()` anterior se saltaba el log
    # en silencio y el EDA jamas aparecia en MLflow UI (queja del usuario).
    # Ahora: sin run activo se crea un run dedicado `eda_<variedad>` en el
    # experimento de la variedad. Se sube tambien el sidecar JSON (mismo
    # contrato que el HTML: datos crudos sin parsear HTML).
    if log_to_mlflow:
        try:
            import mlflow

            from src.config import MLFLOW_EXPERIMENT_PREFIX

            artefactos = [str(out_path), str(json_path)]
            if mlflow.active_run() is not None:
                for p in artefactos:
                    mlflow.log_artifact(p, artifact_path="eda")
            else:
                mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_PREFIX}{variety}")
                with mlflow.start_run(run_name=f"eda_{variety}_{ts}"):
                    mlflow.set_tag("run_type", "eda")
                    for p in artefactos:
                        mlflow.log_artifact(p, artifact_path="eda")
            logger.info(f"[EDA/{variety}] HTML+JSON subidos como artifacts a MLflow")
        except Exception as exc:
            logger.warning(f"[EDA/{variety}] log_artifact MLflow fallo: {exc}")

    # ---- 10. Regenerar dashboard global (reports/index.html) ----
    # Sin esto el sidebar de http://localhost:8080/reports/ no muestra la
    # variedad nueva hasta el siguiente training. variety_runner ya hace
    # esto post-training; replicamos aqui para que `task eda VARIETIES=X`
    # tambien aparezca de inmediato — con solo el bloque EDA mientras no
    # exista Winner/Residuals para X.
    try:
        from src.diagnostics.dashboard_index import write_dashboard

        write_dashboard(out_dir)
    except Exception:
        logger.exception(f"[EDA/{variety}] no se pudo regenerar reports/index.html")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA diagnostico standalone")
    p.add_argument("--variety", required=True, help="Variedad (hoja del Excel)")
    p.add_argument("--out-dir", default=None, help="Override del directorio de salida")
    p.add_argument(
        "--no-mlflow", action="store_true", help="No intenta subir como artifact a MLflow"
    )
    return p.parse_args()


def _main() -> int:
    from src.utils.logger import setup_logging

    setup_logging()
    args = _parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else None
    out_path = run_eda(
        args.variety,
        out_dir=out_dir,
        log_to_mlflow=not args.no_mlflow,
    )
    print(f"\nEDA report: {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
