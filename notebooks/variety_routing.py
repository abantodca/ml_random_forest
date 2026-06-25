"""Core del experimento de ruteo de variedades a anclas (donante).

Lógica estable y testeable extraída del notebook `experiment_variety_anchor_routing.ipynb`
(el notebook conserva solo orquestación + visualización). Estructura:

    Configuración        ExperimentConfig, constantes de viabilidad y de efecto
    Carga de datos       normalize_columns · load_variety_data · normalize_features · filter_outliers
    Distancias           mean_wasserstein(_arrays) · assign_to_anchors
    Validación robusta    cliffs_delta · validate_effect_size · apply_holm
                          validate_mann_whitney · mann_whitney_holm · bootstrap_stability
    Estructura global     silhouette_over_observations · tests 1-4 · run_all_validations
    Resultado/Export     build_final_result · compute_group_summary · export_results

Decisión de diseño (estadística robusta): la asignación se valida por **tamaño de
efecto** (Cliff's delta), no por p-valores, que saturan a n grande. Los 4 tests
globales clásicos se conservan como *informativos* — el veredicto honesto se basa
solo en el silhouette sobre observaciones (evidencia no circular).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binom, kruskal, mannwhitneyu, wasserstein_distance
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

try:
    from statsmodels.stats.multitest import multipletests

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover - statsmodels es opcional
    _HAS_STATSMODELS = False

RANDOM_STATE = 123

# Umbrales de |Cliff's delta| (Romano et al.): efecto pequeño / mediano.
EFFECT_SMALL = 0.33
EFFECT_MEDIUM = 0.474
# Mínimo de observaciones para validar una distribución de forma fiable.
MIN_N_VALIDATE = 20


# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════
@dataclass
class ExperimentConfig:
    """Configuración centralizada del experimento de agrupación."""

    features: list = field(
        default_factory=lambda: [
            "KGHECT",
            "INDUSTRIAL",
            "DPC",
            "PesoBayaFIMPRO",
            "KGHORA",
        ]
    )
    anchor_varieties: list = field(
        # Nombres = hojas del Excel. "POP"/"BEAUTY" son SEKOYA POP / SEKOYA
        # BEAUTY (las 2 variedades con más datos); con el nombre largo NO
        # matcheaban la hoja y caían como no-anclas.
        default_factory=lambda: [
            "POP",
            "VENTURA",
            "BEAUTY",
            "BIANCA",
            "ATLAS",
            "JUPITER",
            "ROSITA",
            "BELLA",
            "ARANA",
            "EMERALD",
            "MAGICA",
        ]
    )
    excel_path: str = "../data/training/DB-HISTORICA.xlsx"
    output_dir: str = "../data"
    test_size: float = 0.2
    outer_folds: int = 5
    inner_folds: int = 5
    contamination: float = 0.02
    alpha: float = 0.05
    min_similar_pct: float = 60.0
    n_permutations: int = 999
    # ── Análisis reforzado ──
    outlier_contamination: float = 0.05
    bootstrap_iterations: int = 100
    bootstrap_seed: int = 42
    mw_holm_null_percentile: float = 90.0
    k_sweep_min: int = 3
    k_sweep_max: int = 15

    @property
    def reduction_factor(self) -> float:
        """Fracción de filas que sobrevive al split + nested CV + outliers."""
        return (
            (1 - self.test_size)
            * (self.outer_folds - 1)
            / self.outer_folds
            * (self.inner_folds - 1)
            / self.inner_folds
            * (1 - self.contamination)
        )

    @property
    def thresholds(self) -> dict:
        """Umbrales de filas mínimas por zona de viabilidad."""
        rf = self.reduction_factor
        return {
            "conservative": int(np.ceil((20 * 2) / rf)),
            "moderate": int(np.ceil((10 * 8) / rf)),
            "robust": int(np.ceil((20 * 16) / rf)),
        }


# Zonas de viabilidad de entrenamiento (orden descendente de robustez).
VIABILITY_ZONES = [
    {"label": "✓ Robusto", "zone": "🟢 Individual", "color": "#2ecc71",
     "note": "Modelo individual con nested CV completo"},
    {"label": "~ Básico", "zone": "🟡 Individual", "color": "#f39c12",
     "note": "Modelo individual (árboles básicos)"},
    {"label": "⚠ Mínimo", "zone": "🔴 Analizar", "color": "#e67e22",
     "note": "Ancla con pocos datos — revisar"},
    {"label": "✗ Insuficiente", "zone": "⬛ Agrupar", "color": "#e74c3c",
     "note": "Ancla con datos insuficientes"},
]


def classify_viability(n_filas: int, thresholds: dict) -> dict:
    """Clasifica una variedad según su zona de viabilidad del pipeline."""
    if n_filas >= thresholds["robust"]:
        return VIABILITY_ZONES[0]
    if n_filas >= thresholds["moderate"]:
        return VIABILITY_ZONES[1]
    if n_filas >= thresholds["conservative"]:
        return VIABILITY_ZONES[2]
    return VIABILITY_ZONES[3]


# ══════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════
# Mapeo de columnas crudas del export histórico → esquema de features.
# El Excel trae nombres crudos (a veces con espacios); el resto del pipeline
# usa los nombres canónicos. El mapeo es idempotente.
RAW_TO_FEATURE = {
    "KG/HA": "KGHECT",
    "%INDUS": "INDUSTRIAL",
    "DPC": "DPC",
    "P/BAYA": "PesoBayaFIMPRO",
    "KG/JR_H": "KGHORA",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza nombres de columnas crudas al esquema de features.

    Limpia espacios y aplica `RAW_TO_FEATURE`. Si las columnas ya vienen
    renombradas (export antiguo), no las altera.
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out.rename(columns=RAW_TO_FEATURE)


def load_variety_data(cfg: ExperimentConfig) -> tuple:
    """Carga el Excel (una hoja por variedad) y separa anclas vs no-anclas."""
    import openpyxl

    wb = openpyxl.load_workbook(cfg.excel_path, read_only=True)
    sheets = wb.sheetnames
    wb.close()

    all_data = {
        s: normalize_columns(pd.read_excel(cfg.excel_path, sheet_name=s))
        for s in sheets
    }
    anchors = [v for v in cfg.anchor_varieties if v in all_data]
    non_anchors = sorted(v for v in all_data if v not in cfg.anchor_varieties)

    summary = pd.DataFrame(
        [
            {
                "variedad": name,
                "n_filas": len(df),
                "tipo": "⚓ ANCLA" if name in cfg.anchor_varieties else "a asignar",
            }
            for name, df in all_data.items()
        ]
    ).sort_values("n_filas", ascending=False)

    return all_data, anchors, non_anchors, summary


def normalize_features(all_data: dict, features: list) -> dict:
    """Normaliza features globalmente con RobustScaler (resistente a outliers)."""
    combined = pd.concat(
        [df.assign(variedad=name) for name, df in all_data.items()], ignore_index=True
    )
    scaler = RobustScaler().fit(combined[features].dropna())

    scaled = {}
    for name, df in all_data.items():
        subset = df[[f for f in features if f in df.columns]].dropna()
        if len(subset) > 0:
            scaled[name] = pd.DataFrame(
                scaler.transform(subset.reindex(columns=features, fill_value=0)),
                columns=features,
            )
    return scaled


def filter_outliers(
    scaled_data: dict,
    features: list,
    contamination: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Remueve outliers por variedad con IsolationForest.

    Wasserstein es sensible a extremos: pocos outliers inflan distancias y
    degradan silhouette / efecto. Se filtra por variedad porque cada
    distribución tiene sus propios outliers.
    """
    cleaned = {}
    for name, df in scaled_data.items():
        X = df[features].values
        if len(X) < 5:
            cleaned[name] = df.copy()
            continue
        mask = IsolationForest(
            contamination=contamination, random_state=random_state
        ).fit_predict(X) == 1  # +1 inlier, -1 outlier
        cleaned[name] = df[mask].reset_index(drop=True)
    return cleaned


# ══════════════════════════════════════════════════════════════════
# DISTANCIA DE WASSERSTEIN
# ══════════════════════════════════════════════════════════════════
def mean_wasserstein_arrays(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 promedio por columna entre dos arrays (n×f) y (m×f)."""
    if a.shape[1] == 0:
        return np.inf
    return float(np.mean([wasserstein_distance(a[:, i], b[:, i]) for i in range(a.shape[1])]))


def mean_wasserstein(data_a: pd.DataFrame, data_b: pd.DataFrame, features: list) -> float:
    """Wasserstein-1 promedio por feature entre dos distribuciones multivariadas."""
    dists = [
        wasserstein_distance(data_a[f].values, data_b[f].values)
        for f in features
        if len(data_a[f].values) > 0 and len(data_b[f].values) > 0
    ]
    return float(np.mean(dists)) if dists else np.inf


# Submuestreo para que el cálculo de Cliff's delta (O(n·m)) sea viable con
# variedades grandes. Estima bien la distribución y es determinista (seed fijo).
_CLIFF_CAP = 400


def _subsample(X: np.ndarray, cap: int = _CLIFF_CAP) -> np.ndarray:
    if len(X) <= cap:
        return X
    return X[np.random.default_rng(RANDOM_STATE).choice(len(X), cap, replace=False)]


def max_cliff_arrays(a: np.ndarray, b: np.ndarray) -> float:
    """max sobre features de |Cliff's delta| entre arrays (n×f) y (m×f)."""
    return max(
        abs(
            ((a[:, i][:, None] > b[:, i][None, :]).sum() - (a[:, i][:, None] < b[:, i][None, :]).sum())
            / (len(a) * len(b))
        )
        for i in range(a.shape[1])
    )


def assign_to_anchors(
    non_anchors: list,
    anchors: list,
    all_data: dict,
    scaled_data: dict,
    features: list,
    *,
    by: str = "effect",
) -> pd.DataFrame:
    """Asigna cada variedad no-ancla al ancla más parecida.

    `by="effect"` (recomendado): minimiza max|Cliff's delta| — **coherente con el
    criterio de decisión** (validate_effect_size), así no se asigna por una métrica
    y se juzga por otra. `by="wasserstein"`: métrica original (solo distribución).
    La columna `distancia_wass` siempre reporta el Wasserstein al ancla elegida.
    """
    rows = []
    for variety in sorted(non_anchors):
        if variety not in scaled_data:
            continue
        cand = [a for a in anchors if a in scaled_data]
        X_var = _subsample(scaled_data[variety][features].values)

        def _wass(a):
            return mean_wasserstein(scaled_data[variety], scaled_data[a], features)

        def _eff(a):
            return max_cliff_arrays(X_var, _subsample(scaled_data[a][features].values))

        if by == "effect":
            ranked = sorted(cand, key=lambda a: (_eff(a), _wass(a)))
        else:
            ranked = sorted(cand, key=_wass)

        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else "-"
        rows.append(
            {
                "variedad": variety,
                "n_filas": len(all_data[variety]),
                "ancla_asignada": best,
                "distancia_wass": round(_wass(best), 4),
                "segunda_opcion": second,
                "dist_segunda": round(_wass(second), 4) if second != "-" else np.inf,
            }
        )
    return pd.DataFrame(rows).sort_values("ancla_asignada")


# ══════════════════════════════════════════════════════════════════
# CORRECCIÓN DE COMPARACIONES MÚLTIPLES (Holm-Bonferroni)
# ══════════════════════════════════════════════════════════════════
def apply_holm(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Holm-Bonferroni step-down. Usa statsmodels si está; si no, fallback manual."""
    if _HAS_STATSMODELS:
        rejected, _, _, _ = multipletests(pvals, alpha=alpha, method="holm")
        return rejected
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    rejected = np.zeros(n, dtype=bool)
    for i, idx in enumerate(np.argsort(pvals)):
        if pvals[idx] <= alpha / (n - i):
            rejected[idx] = True
        else:
            break  # step-down: al primer no-rechazo, paramos
    return rejected


# ══════════════════════════════════════════════════════════════════
# VALIDACIÓN MANN-WHITNEY U (informativa)
# ══════════════════════════════════════════════════════════════════
def validate_mann_whitney(
    assignment_df: pd.DataFrame, all_data: dict, features: list, alpha: float = 0.05
) -> pd.DataFrame:
    """Valida cada asignación con Mann-Whitney U por feature (sin corrección)."""
    rows = []
    for _, arow in assignment_df.iterrows():
        variety, anchor = arow["variedad"], arow["ancla_asignada"]
        sims = []
        for feat in features:
            d1 = all_data[variety][feat].dropna().values
            d2 = all_data[anchor][feat].dropna().values
            if len(d1) >= 3 and len(d2) >= 3:
                _, p = mannwhitneyu(d1, d2, alternative="two-sided")
                sims.append(p > alpha)
        n_sim, n_tot = int(np.sum(sims)), len(sims)
        pct = n_sim / n_tot * 100 if n_tot else 0
        rows.append(
            {
                "variedad": variety,
                "ancla": anchor,
                "features_similares": f"{n_sim}/{n_tot}",
                "pct_similar": pct,
                "status": "✓" if pct >= 60 else "⚠",
            }
        )
    return pd.DataFrame(rows).sort_values("pct_similar")


def mann_whitney_holm(
    validation_df: pd.DataFrame,
    all_data: dict,
    anchors: list,
    features: list,
    *,
    alpha: float = 0.05,
    null_percentile: float = 90.0,
    fallback_pct: float = 60.0,
) -> tuple[pd.DataFrame, float]:
    """Mann-Whitney U con corrección Holm-Bonferroni + umbral calibrado.

    Devuelve `(mw_holm_df, threshold)`. El umbral se calibra con el null de
    parejas variedad-ancla **no asignadas** (percentil `null_percentile`):
    refleja cuán 'similar' se ve un ancla equivocada por azar.
    """
    feat_cache = {
        name: {f: all_data[name][f].dropna().values for f in features}
        for name in all_data
    }

    def pct_similar(v_feats: dict, a_feats: dict) -> float:
        pvals = []
        for f in features:
            v_d, a_d = v_feats.get(f), a_feats.get(f)
            if v_d is None or a_d is None or len(v_d) < 3 or len(a_d) < 3:
                continue
            pvals.append(float(mannwhitneyu(v_d, a_d, alternative="two-sided")[1]))
        if not pvals:
            return 0.0
        rejected = apply_holm(np.array(pvals), alpha)
        return (~rejected).sum() / len(pvals) * 100

    null_pcts = [
        pct_similar(feat_cache[r["variedad"]], feat_cache[a])
        for _, r in validation_df.iterrows()
        for a in anchors
        if a != r["ancla"]
    ]
    threshold = (
        float(np.percentile(null_pcts, null_percentile)) if null_pcts else float(fallback_pct)
    )

    rows = []
    for _, r in validation_df.iterrows():
        variety, anchor = r["variedad"], r["ancla"]
        pvals, n_valid = [], 0
        for f in features:
            v_d = all_data[variety][f].dropna().values
            a_d = all_data[anchor][f].dropna().values
            if len(v_d) < 3 or len(a_d) < 3:
                continue
            pvals.append(float(mannwhitneyu(v_d, a_d, alternative="two-sided")[1]))
            n_valid += 1
        if pvals:
            n_sim = int((~apply_holm(np.array(pvals), alpha)).sum())
            pct = n_sim / len(pvals) * 100
        else:
            n_sim, pct = 0, 0.0
        rows.append(
            {
                "variedad": variety,
                "ancla": anchor,
                "n_similar_holm": n_sim,
                "n_total_valid": n_valid,
                "pct_similar_holm": pct,
            }
        )
    return pd.DataFrame(rows), threshold


# ══════════════════════════════════════════════════════════════════
# VALIDACIÓN ROBUSTA POR TAMAÑO DE EFECTO (decide la asignación)
# ══════════════════════════════════════════════════════════════════
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: tamaño de efecto no-paramétrico en [-1, 1].

    |d| < 0.147 negligible · < 0.33 pequeño · < 0.474 mediano · resto grande.
    Robusto a outliers y a no-normalidad (solo usa el orden de los valores).
    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def validate_effect_size(
    assignment_df: pd.DataFrame,
    scaled_data: dict,
    features: list,
    min_n: int = MIN_N_VALIDATE,
) -> pd.DataFrame:
    """Valida cada asignación por tamaño de efecto en vez de por p-valor.

    Una asignación es 'similar' si el mayor |Cliff's delta| entre sus features
    es < `EFFECT_SMALL` (difieren, a lo sumo, de forma pequeña). Variedades con
    n < `min_n` se marcan 'insuficiente'. Corre sobre la representación escalada
    + filtrada — la MISMA que usa la asignación.
    """
    rows = []
    for _, arow in assignment_df.iterrows():
        variety, anchor = arow["variedad"], arow["ancla_asignada"]
        if variety not in scaled_data or anchor not in scaled_data:
            continue
        n = len(scaled_data[variety])
        if n < min_n:
            rows.append({"variedad": variety, "ancla": anchor, "n": n,
                         "max_cliff": float("nan"), "mean_cliff": float("nan"),
                         "veredicto": "insuficiente"})
            continue
        deltas = [
            abs(cliffs_delta(scaled_data[variety][f].values, scaled_data[anchor][f].values))
            for f in features
        ]
        max_d, mean_d = float(np.max(deltas)), float(np.mean(deltas))
        if max_d < EFFECT_SMALL:
            veredicto = "similar"
        elif max_d < EFFECT_MEDIUM:
            veredicto = "moderado"
        else:
            veredicto = "diferente"
        rows.append({"variedad": variety, "ancla": anchor, "n": n,
                     "max_cliff": max_d, "mean_cliff": mean_d, "veredicto": veredicto})
    return pd.DataFrame(rows).sort_values("max_cliff", na_position="last")


def bootstrap_stability(
    non_anchors: list,
    anchors: list,
    scaled_data: dict,
    anchor_of: dict,
    features: list,
    *,
    iterations: int = 100,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Estabilidad bootstrap de la asignación + significancia Holm vs azar.

    Para cada variedad resamplea sus filas `iterations` veces y recomputa el
    ancla nearest; `stability_pct` = % de veces que cae en el ancla canónica.
    `boot_significant` rechaza H0 (asignación al azar, p=1/k) tras Holm.
    """
    rng = np.random.default_rng(seed)
    anchors_arr = {a: scaled_data[a][features].values for a in anchors if a in scaled_data}

    rows = []
    for variety in non_anchors:
        if variety not in scaled_data:
            continue
        X = scaled_data[variety][features].values
        n = len(X)
        canon = anchor_of[variety]
        if n < 2:
            rows.append({"variedad": variety, "ancla_canonica": canon,
                         "stability_pct": 0.0, "top_alternative": "—",
                         "top_alt_pct": 0.0, "note": f"n={n} (insuficiente para bootstrap)"})
            continue

        counts = {a: 0 for a in anchors_arr}
        for _ in range(iterations):
            sample = X[rng.integers(0, n, size=n)]
            best = min(anchors_arr, key=lambda a: mean_wasserstein_arrays(sample, anchors_arr[a]))
            counts[best] += 1

        alternatives = sorted(
            [(a, c) for a, c in counts.items() if a != canon], key=lambda x: -x[1]
        )
        top_alt = alternatives[0] if alternatives else ("—", 0)
        rows.append({"variedad": variety, "ancla_canonica": canon,
                     "stability_pct": counts[canon] / iterations * 100,
                     "top_alternative": top_alt[0],
                     "top_alt_pct": top_alt[1] / iterations * 100, "note": ""})

    df = pd.DataFrame(rows).sort_values("stability_pct")

    # H0: la variedad cae en su ancla por azar → counts ~ Binomial(iter, 1/k).
    k = len(anchors_arr)
    p_null = 1.0 / k
    pvals = [
        float(1 - binom.cdf(int(round(s / 100 * iterations)) - 1, iterations, p_null))
        for s in df["stability_pct"]
    ]
    return df.assign(p_null_holm=pvals, boot_significant=apply_holm(np.array(pvals), alpha))


# ══════════════════════════════════════════════════════════════════
# ESTRUCTURA GLOBAL (silhouette honesto + 4 tests informativos)
# ══════════════════════════════════════════════════════════════════
def silhouette_over_observations(
    scaled_data: dict, anchor_of: dict, features: list
) -> tuple[float, np.ndarray, np.ndarray]:
    """Silhouette sobre TODAS las observaciones (no centroides) → (score, X, y).

    Test estructural honesto y de alta potencia: a diferencia del silhouette
    sobre 1 centroide por variedad, no diluye la señal ni es circular.
    """
    obs, labels = [], []
    for variety, df_scaled in scaled_data.items():
        obs.append(df_scaled[features].values)
        labels.extend([anchor_of.get(variety, variety)] * len(df_scaled))
    X, y = np.vstack(obs), np.array(labels)
    score = (
        float(silhouette_score(X, y, metric="euclidean")) if len(set(labels)) > 1 else float("nan")
    )
    return score, X, y


def compute_variety_centroids(
    all_data: dict, scaled_data: dict, final_result: pd.DataFrame, features: list
):
    """Centroides normalizados y etiquetas (ancla) por variedad."""
    centroids, labels = [], []
    for name in sorted(all_data):
        if name not in scaled_data:
            continue
        centroids.append(scaled_data[name][features].mean().values)
        match = final_result[final_result["variedad"] == name]
        labels.append(match.iloc[0]["entrena_con"] if len(match) > 0 else name)
    return np.array(centroids), np.array(labels)


def test_silhouette(centroids: np.ndarray, labels: np.ndarray) -> dict:
    """Test 1 (informativo): Silhouette sobre centroides de variedades."""
    unique, counts = np.unique(labels, return_counts=True)
    multi = unique[counts > 1]
    if len(multi) < 2:
        return {"name": "Silhouette Score", "passed": False, "detail": "Insuficientes clusters"}

    mask = np.isin(labels, multi)
    score = silhouette_score(centroids[mask], labels[mask], metric="euclidean")
    passed = score > 0.25
    quality = "Excelente" if score > 0.5 else ("Aceptable" if passed else "Bajo")

    samples = silhouette_samples(centroids[mask], labels[mask], metric="euclidean")
    per_group = {grp: float(samples[labels[mask] == grp].mean()) for grp in sorted(multi)}
    return {"name": "Silhouette Score", "passed": passed, "score": score,
            "quality": quality, "per_group": per_group}


def test_intra_inter_ratio(
    assignment_df: pd.DataFrame, anchors: list, scaled_data: dict, features: list
) -> dict:
    """Test 2 (informativo, circular): ratio distancia intra vs inter-grupo."""
    intra, inter = [], []
    for _, row in assignment_df.iterrows():
        variety = row["variedad"]
        if variety not in scaled_data:
            continue
        for anchor in anchors:
            if anchor not in scaled_data:
                continue
            d = mean_wasserstein(scaled_data[variety], scaled_data[anchor], features)
            (intra if anchor == row["ancla_asignada"] else inter).append(d)

    m_intra, m_inter = np.mean(intra), np.mean(inter)
    ratio = m_intra / m_inter
    passed = ratio < 1.0
    quality = (
        "Excelente" if ratio < 0.5
        else "Bueno" if ratio < 0.8
        else "Aceptable" if passed
        else "Malo"
    )
    return {"name": "Ratio intra/inter", "passed": passed, "ratio": ratio,
            "mean_intra": m_intra, "mean_inter": m_inter, "quality": quality}


def test_kruskal_wallis(final_result: pd.DataFrame, all_data: dict, features: list) -> dict:
    """Test 3 (informativo, trivial a n grande): Kruskal-Wallis por feature."""
    tagged = pd.concat(
        [
            all_data[row["variedad"]].assign(grupo_ancla=row["entrena_con"])
            for _, row in final_result.iterrows()
            if row["variedad"] in all_data
        ],
        ignore_index=True,
    )
    results = []
    for feat in features:
        groups = [
            g[feat].dropna().values
            for _, g in tagged.groupby("grupo_ancla")
            if len(g[feat].dropna()) >= 3
        ]
        if len(groups) >= 2:
            h, p = kruskal(*groups)
            results.append({"feature": feat, "H": h, "p_value": p, "sig": p < 0.05})
    n_sig = sum(r["sig"] for r in results)
    return {"name": "Kruskal-Wallis", "passed": n_sig >= 3, "results": results,
            "n_sig": n_sig, "n_total": len(results)}


def test_permanova(centroids: np.ndarray, labels: np.ndarray, n_perms: int = 999) -> dict:
    """Test 4 (informativo, circular): PERMANOVA permutacional sobre centroides."""
    unique_groups = np.unique(labels)
    k, n = len(unique_groups), len(centroids)
    grand = centroids.mean(axis=0)
    ss_total = np.sum((centroids - grand) ** 2)

    def ss_within(labs):
        return sum(
            np.sum((centroids[labs == grp] - centroids[labs == grp].mean(axis=0)) ** 2)
            for grp in unique_groups
            if len(centroids[labs == grp]) > 0
        )

    ss_w = ss_within(labels)
    ss_b = ss_total - ss_w
    f_obs = (ss_b / max(k - 1, 1)) / (ss_w / max(n - k, 1))
    f_perms = np.array(
        [
            ((ss_total - (ssw := ss_within(np.random.permutation(labels)))) / max(k - 1, 1))
            / (ssw / max(n - k, 1))
            for _ in range(n_perms)
        ]
    )
    p_val = (np.sum(f_perms >= f_obs) + 1) / (n_perms + 1)
    return {"name": "PERMANOVA", "passed": p_val < 0.05, "f_obs": f_obs,
            "p_value": p_val, "r2": ss_b / ss_total}


def run_all_validations(
    final_result, assignment_df, anchors, all_data, scaled_data, features, n_perms=999
):
    """Corre los 4 tests (informativos) y emite un veredicto honesto.

    El veredicto se basa SOLO en el silhouette sobre observaciones (evidencia
    no circular). Ratio y PERMANOVA son casi tautológicos (la etiqueta es el
    ancla más cercana por la misma distancia) y Kruskal-Wallis es trivial a n
    grande, por eso se reportan pero no votan.
    """
    centroids, labels = compute_variety_centroids(all_data, scaled_data, final_result, features)
    tests = [
        test_silhouette(centroids, labels),
        test_intra_inter_ratio(assignment_df, anchors, scaled_data, features),
        test_kruskal_wallis(final_result, all_data, features),
        test_permanova(centroids, labels, n_perms),
    ]

    for t in tests:
        print(f"\n{'='*70}")
        print(f"{'✓' if t['passed'] else '✗'} {t['name']}")
        print(f"{'='*70}")
        if t["name"] == "Silhouette Score":
            if "score" in t:
                print(f"   Score global: {t['score']:.4f} — {t['quality']}")
                for grp, val in t.get("per_group", {}).items():
                    print(f"   ⚓ {grp:20s}: {val:+.3f}")
            else:
                print(f"   {t['detail']}")
        elif t["name"] == "Ratio intra/inter":
            print(f"   Intra-grupo: {t['mean_intra']:.4f}  |  Inter-grupo: {t['mean_inter']:.4f}")
            print(f"   Ratio: {t['ratio']:.4f} — {t['quality']}")
        elif t["name"] == "Kruskal-Wallis":
            for r in t["results"]:
                flag = "✓ Sig." if r["sig"] else "✗ No sig."
                print(f"   {r['feature']:20s}  H={r['H']:10.2f}  p={r['p_value']:.2e}  {flag}")
            print(f"   → {t['n_sig']}/{t['n_total']} features significativas")
        elif t["name"] == "PERMANOVA":
            print(f"   F={t['f_obs']:.4f}  p={t['p_value']:.4f}  R²={t['r2']:.4f} ({t['r2']*100:.1f}%)")

    anchor_of = {r["variedad"]: r["entrena_con"] for _, r in final_result.iterrows()}
    sil_obs, X_obs, _ = silhouette_over_observations(scaled_data, anchor_of, features)
    tests[0]["score_all_obs"] = sil_obs  # lo consume la sección 6

    print(f"\n{'='*70}")
    print("VEREDICTO HONESTO — solo evidencia no-circular")
    print(f"{'='*70}")
    print("   Informativos (NO votan): Ratio y PERMANOVA son circulares;")
    print("   Kruskal-Wallis es trivial a n grande.")
    print(f"   Silhouette sobre {len(X_obs):,} observaciones: {sil_obs:+.4f}")
    if sil_obs > 0.25:
        verdict = "✅ ESTRUCTURA DE CLUSTERS REAL"
    elif sil_obs > 0.0:
        verdict = "⚠ ESTRUCTURA DÉBIL — decidir por estabilidad + tamaño de efecto (5.1)"
    else:
        verdict = (
            "❌ SIN ESTRUCTURA NATURAL — esto es 'modelo donante más cercano', "
            "no clusters. La decisión por variedad la dan bootstrap + Cliff's delta (5.1)."
        )
    print(f"\n   → {verdict}")
    return tests


# ══════════════════════════════════════════════════════════════════
# RESULTADO FINAL + EXPORTACIÓN
# ══════════════════════════════════════════════════════════════════
def build_final_result(
    anchors: list,
    assignment_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    all_data: dict,
    thresholds: dict,
) -> pd.DataFrame:
    """Tabla final: anclas (datos propios) + no-anclas (con su ancla asignada)."""
    rows = []
    for anchor in sorted(anchors):
        n = len(all_data[anchor])
        viab = classify_viability(n, thresholds)
        rows.append({"variedad": anchor, "n_filas": n, "zona": viab["zone"],
                     "entrena_con": anchor, "ancla_asignada": anchor,
                     "distancia_wass": 0.0, "nota": viab["note"]})
    for _, arow in assignment_df.iterrows():
        variety, n = arow["variedad"], arow["n_filas"]
        anchor, dist = arow["ancla_asignada"], arow["distancia_wass"]
        val = validation_df[validation_df["variedad"] == variety]
        pct = val["pct_similar"].values[0] if len(val) > 0 else 0
        rows.append({"variedad": variety, "n_filas": n, "zona": "⬛ Agrupar",
                     "entrena_con": anchor, "ancla_asignada": anchor,
                     "distancia_wass": dist,
                     "nota": f"Vecino más cercano: {anchor} (dist={dist:.3f}, similitud={pct:.0f}%)"})
    return pd.DataFrame(rows).sort_values(
        ["entrena_con", "zona", "n_filas"], ascending=[True, True, False]
    )


def compute_group_summary(anchors: list, assignment_df: pd.DataFrame, all_data: dict) -> pd.DataFrame:
    """Resumen por grupo ancla: filas propias + asignadas + total."""
    rows = []
    for anchor in sorted(anchors):
        assigned = assignment_df[assignment_df["ancla_asignada"] == anchor]
        propias = len(all_data[anchor])
        rows.append({"ancla": anchor, "filas_propias": propias,
                     "filas_asignadas": int(assigned["n_filas"].sum()),
                     "n_asignadas": len(assigned),
                     "total": propias + int(assigned["n_filas"].sum())})
    return pd.DataFrame(rows).sort_values("total", ascending=False)


# ══════════════════════════════════════════════════════════════════
# RUTEO PREDICTIVO (la decisión accionable — transferencia, no clustering)
# ══════════════════════════════════════════════════════════════════
def route_to_anchors_predictive(
    all_data: dict,
    anchors: list,
    predictors: list,
    target: str,
    *,
    random_state: int = RANDOM_STATE,
    max_iter: int = 200,
    min_own_n: int = 25,
) -> tuple[pd.DataFrame, float]:
    """Rutea cada variedad al ancla cuyo MODELO la pronostica mejor (MAPE OOS).

    Decisión por error predictivo, no por distribución. Devuelve
    `(routing_df, baseline)` donde `baseline` es el MAPE típico de un ancla.
    Columnas de `routing_df`:
      - mape_oos    : MAPE del modelo del ancla sobre la variedad (out-of-sample;
                      el ancla nunca vio esa data → honesto).
      - mape_propio : MAPE del modelo propio de la variedad (5-fold CV);
                      NaN si n < `min_own_n` (no hay data para uno fiable).
      - ganancia    : mape_propio − mape_oos = puntos % que se GANAN al heredar
                      en vez de entrenar propio (>0 ⇒ heredar reduce el error).
      - ratio       : mape_oos / baseline.
      - decision    : ancla · bueno (≤1.5×) · aceptable (≤2.0×) · revisar (>2.0×).

    El modelo es HistGradientBoosting sobre `predictors` (proxy sin lag features);
    fija el RUTEO, no el MAPE de producción.
    """
    def _clean(v):
        df = all_data[v][predictors + [target]].dropna()
        return df[df[target] > 0]

    def _mape(y, y_hat):
        return float(np.mean(np.abs((y - y_hat) / y)) * 100)

    def _fit(df, it=max_iter):
        m = HistGradientBoostingRegressor(random_state=random_state, max_iter=it)
        m.fit(df[predictors].values, df[target].values)
        return m

    anchor_models, anchor_own = {}, {}
    for a in anchors:
        d = _clean(a)
        anchor_models[a] = _fit(d)
        anchor_own[a] = _mape(d[target].values, anchor_models[a].predict(d[predictors].values))
    baseline = float(np.mean(list(anchor_own.values())))

    kf = KFold(5, shuffle=True, random_state=random_state)
    rows = []
    for v in sorted(all_data):
        d = _clean(v)
        if len(d) == 0:
            continue
        if v in anchors:
            rows.append({"variedad": v, "entrena_con": v, "n": len(d),
                         "mape_oos": anchor_own[v], "mape_propio": anchor_own[v],
                         "ganancia": 0.0, "ratio": 1.0, "decision": "ancla"})
            continue
        y, X = d[target].values, d[predictors].values
        errs = {a: _mape(y, anchor_models[a].predict(X)) for a in anchors}
        best = min(errs, key=errs.get)
        mape_oos, ratio = errs[best], errs[best] / baseline
        if len(d) >= min_own_n:
            own = float(np.mean([
                _mape(d.iloc[te][target].values,
                      _fit(d.iloc[tr], it=120).predict(d.iloc[te][predictors].values))
                for tr, te in kf.split(d)
            ]))
            ganancia = own - mape_oos
        else:
            own, ganancia = float("nan"), float("nan")
        decision = "bueno" if ratio <= 1.5 else "aceptable" if ratio <= 2.0 else "revisar"
        rows.append({"variedad": v, "entrena_con": best, "n": len(d),
                     "mape_oos": mape_oos, "mape_propio": own, "ganancia": ganancia,
                     "ratio": ratio, "decision": decision})
    return pd.DataFrame(rows).sort_values(["decision", "mape_oos"]), baseline
