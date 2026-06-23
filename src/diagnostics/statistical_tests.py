"""Wrappers tipados sobre statsmodels/scipy.

Cada funcion devuelve un dataclass `TestResult` con:
    name      : nombre del test
    statistic : valor de la estadistica
    p_value   : p-value (None si el test no produce p)
    rejects_h0 : True si rechaza H0 al nivel alpha (default 0.05)
    h0_meaning : descripcion legible de H0 ("residuos son homocedasticos", ...)
    notes      : info adicional (parametros, advertencias)

Patron defensivo: TODOS los tests devuelven `TestResult` aunque fallen.
Si scipy/statsmodels lanza por n insuficiente o singularidad, se captura
y se devuelve un resultado con `statistic=None` y un nota explicativa.
Esto permite que el HTML report siempre tenga datos que mostrar, marcando
los tests fallidos sin que el script entero abort.

Convención de p-value:
    Tests donde RECHAZAR significa "hay problema" (heterocedasticidad,
    autocorrelacion, no-normalidad): rejects_h0=True -> 🔴 hallazgo.
    Tests donde RECHAZAR significa "OK" (estacionariedad ADF):
    rejects_h0=True -> 🟢 OK.
    El campo `is_finding` lo declara cada wrapper para que el renderer
    pinte el badge correcto sin tener que conocer cada test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TestResult:
    name: str
    statistic: float | None = None
    p_value: float | None = None
    rejects_h0: bool | None = None
    h0_meaning: str = ""
    is_finding: bool = False  # True si rechazar H0 = "problema a remediar"
    notes: str = ""
    extra: dict = field(default_factory=dict)
    # Campos opcionales para compatibilidad con consumers que esperaban
    # `HeteroscedasticityTest` (HTML report del step_05). `note` (sin 's')
    # almacena la interpretacion en lenguaje natural; `is_heteroscedastic`
    # es un alias semantico de `rejects_h0` para tests de heterocedasticidad.
    note: str | None = None
    is_heteroscedastic: bool | None = None

    def status_emoji(self) -> str:
        """Devuelve 🟢/🔴/⚪ segun el resultado del test.

        ⚪ = test fallo o n insuficiente.
        🔴 = test arrojo hallazgo accionable (segun is_finding semantics).
        🟢 = test paso sin hallazgo.
        """
        if self.statistic is None or self.rejects_h0 is None:
            return "⚪"
        if self.is_finding:
            return "🔴" if self.rejects_h0 else "🟢"
        return "🟢" if self.rejects_h0 else "🔴"


def _safe_test(name: str, h0: str, is_finding: bool) -> TestResult:
    """Constructor de fallback para tests fallidos."""
    return TestResult(
        name=name,
        h0_meaning=h0,
        is_finding=is_finding,
        notes="test no aplicable o fallo (n insuficiente / singular)",
    )


# ---------------------------------------------------------------------------
# Normalidad (univariado)
# ---------------------------------------------------------------------------
def shapiro_wilk(x: pd.Series, alpha: float = 0.05) -> TestResult:
    """Shapiro-Wilk normality test. H0: sample viene de distribucion normal.

    Solo aplicable para n entre 3 y 5000. Si fuera de rango, devuelve
    fallback (preferir Anderson-Darling para n>5000).
    """
    from scipy.stats import shapiro

    name = "Shapiro-Wilk"
    h0 = "la muestra viene de una distribucion normal"
    x_clean = x.dropna()
    n = len(x_clean)
    if n < 3 or n > 5000:
        out = _safe_test(name, h0, is_finding=True)
        out.notes = f"n={n} fuera de rango [3, 5000]; usar Anderson-Darling"
        return out
    try:
        stat, p = shapiro(x_clean)
        return TestResult(
            name=name,
            statistic=float(stat),
            p_value=float(p),
            rejects_h0=p < alpha,
            h0_meaning=h0,
            is_finding=True,
            notes=f"n={n}",
        )
    except Exception as e:
        out = _safe_test(name, h0, is_finding=True)
        out.notes = f"error: {e}"
        return out


def anderson_darling(x: pd.Series, alpha: float = 0.05) -> TestResult:
    """Anderson-Darling normality test. Devuelve estadistica + p-value.

    scipy 1.15+ requiere `method=` explicito (FutureWarning sin el).
    Intentamos primero la API nueva (`method='interpolate'` da p-value
    directo); si scipy es viejo, caemos al legacy de critical_values.
    """
    from scipy.stats import anderson

    name = "Anderson-Darling"
    h0 = "la muestra viene de una distribucion normal"
    x_clean = x.dropna()
    if len(x_clean) < 8:
        return _safe_test(name, h0, is_finding=True)
    try:
        # API nueva (scipy >= 1.15): da pvalue directo, mas preciso
        try:
            result = anderson(x_clean, dist="norm", method="interpolate")
            stat = float(result.statistic)
            p = float(result.pvalue) if hasattr(result, "pvalue") else None
            if p is not None:
                return TestResult(
                    name=name,
                    statistic=stat,
                    p_value=p,
                    rejects_h0=p < alpha,
                    h0_meaning=h0,
                    is_finding=True,
                    notes=f"n={len(x_clean)}",
                )
        except TypeError:
            # scipy < 1.15: method param no existe, cae al legacy abajo
            pass

        # API legacy (scipy < 1.15): critical_values + significance_level
        result = anderson(x_clean, dist="norm")
        sig_levels = list(result.significance_level)  # [15.0, 10.0, 5.0, 2.5, 1.0]
        crit_at_alpha = result.critical_values[sig_levels.index(alpha * 100)]
        rejects = result.statistic > crit_at_alpha
        return TestResult(
            name=name,
            statistic=float(result.statistic),
            p_value=None,
            rejects_h0=bool(rejects),
            h0_meaning=h0,
            is_finding=True,
            notes=f"critical(α={alpha})={crit_at_alpha:.3f}",
            extra={
                "critical_values": list(result.critical_values),
                "significance_level": sig_levels,
            },
        )
    except Exception as e:
        out = _safe_test(name, h0, is_finding=True)
        out.notes = f"error: {e}"
        return out


def jarque_bera(x: pd.Series, alpha: float = 0.05) -> TestResult:
    """Jarque-Bera test. H0: skew=0 y kurtosis=3 (normal)."""
    from scipy.stats import jarque_bera as jb

    name = "Jarque-Bera"
    h0 = "skew=0 y excess_kurtosis=0 (normal)"
    x_clean = x.dropna()
    if len(x_clean) < 20:
        return _safe_test(name, h0, is_finding=True)
    try:
        stat, p = jb(x_clean)
        return TestResult(
            name=name,
            statistic=float(stat),
            p_value=float(p),
            rejects_h0=p < alpha,
            h0_meaning=h0,
            is_finding=True,
            notes=f"n={len(x_clean)} | skew={x_clean.skew():.3f} kurt={x_clean.kurtosis():.3f}",
        )
    except Exception as e:
        out = _safe_test(name, h0, is_finding=True)
        out.notes = f"error: {e}"
        return out


# ---------------------------------------------------------------------------
# Heterocedasticidad
# ---------------------------------------------------------------------------
def breusch_pagan(y: pd.Series, X: pd.DataFrame, alpha: float = 0.05) -> TestResult:
    """Breusch-Pagan test. H0: var(residuos|X) = constante (homocedasticidad).

    Ajusta OLS y(X) y testea heterocedasticidad sobre los residuos.
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.tools import add_constant

    name = "Breusch-Pagan"
    h0 = "varianza de residuos es constante (homocedasticidad)"
    df = pd.concat([y, X], axis=1).dropna()
    if len(df) < 30:
        return _safe_test(name, h0, is_finding=True)
    try:
        y_clean = df[y.name]
        X_clean = add_constant(df[X.columns], has_constant="add")
        ols = OLS(y_clean.values, X_clean.values).fit()
        lm, lm_p, f, f_p = het_breuschpagan(ols.resid, X_clean.values)
        return TestResult(
            name=name,
            statistic=float(lm),
            p_value=float(lm_p),
            rejects_h0=lm_p < alpha,
            h0_meaning=h0,
            is_finding=True,
            notes=f"n={len(df)}, F={f:.2f}, F_p={f_p:.4f}",
            extra={
                "lm_statistic": float(lm),
                "lm_pvalue": float(lm_p),
                "f_statistic": float(f),
                "f_pvalue": float(f_p),
            },
        )
    except Exception as e:
        out = _safe_test(name, h0, is_finding=True)
        out.notes = f"error: {e}"
        return out


def breusch_pagan_residuals(
    residuals,
    predictions,
    alpha: float = 0.05,
) -> TestResult:
    """Variante de Breusch-Pagan que parte de residuos y predicciones ya calculadas.

    Pensada para diagnostico post-hoc de un modelo arbitrario (no necesariamente
    OLS): ajusta una OLS auxiliar `residuos ~ predictions` y aplica el test LM
    de heterocedasticidad sobre los residuos de esa auxiliar.

    H0: var(residual) constante (homoscedastico).
    H1: var(residual) varia con la magnitud predicha.

    Si p < alpha, los intervalos simetricos del modelo (mean +/- k*std) NO son
    validos uniformemente; recomendado usar Conformal Prediction (que no asume
    homoscedasticidad).

    Devuelve `TestResult` con los campos extra `note` (interpretacion en
    espanol) y `is_heteroscedastic` (alias de `rejects_h0`) para mantener
    compatibilidad con consumers que originalmente esperaban
    `HeteroscedasticityTest`.
    """
    import numpy as np

    name = "Breusch-Pagan (residuals)"
    h0 = "varianza de residuos es constante (homocedasticidad)"

    residuals = np.asarray(residuals, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    n = len(residuals)
    if n < 10:
        out = _safe_test(name, h0, is_finding=True)
        out.notes = f"n={n} insuficiente (n<10)"
        out.note = "Muestra insuficiente para test BP (n<10)."
        out.is_heteroscedastic = False
        return out

    try:
        from statsmodels.regression.linear_model import OLS
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.tools.tools import add_constant
    except ImportError:
        out = _safe_test(name, h0, is_finding=True)
        out.note = "statsmodels no disponible para BP test."
        out.is_heteroscedastic = False
        return out

    try:
        # OLS auxiliar: residuals ~ predictions (1 regresor)
        exog = add_constant(predictions)
        model = OLS(residuals, exog).fit()
        lm_stat, p_val, _, _ = het_breuschpagan(model.resid, exog)
        is_hetero = bool(p_val < alpha)
        if is_hetero:
            note = (
                f"Heteroscedasticidad detectada (p={p_val:.4f} < {alpha}). "
                "Los intervalos simetricos del modelo no son validos uniformemente. "
                "Recomendado: Conformal Prediction para bandas garantizadas."
            )
        else:
            note = (
                f"Sin evidencia de heteroscedasticidad (p={p_val:.4f}). "
                "Bandas simetricas (mean +/- 1.96*std) son razonables."
            )
        return TestResult(
            name=name,
            statistic=float(lm_stat),
            p_value=float(p_val),
            rejects_h0=is_hetero,
            h0_meaning=h0,
            is_finding=True,
            notes=f"n={n}",
            extra={"lm_statistic": float(lm_stat), "lm_pvalue": float(p_val)},
            note=note,
            is_heteroscedastic=is_hetero,
        )
    except Exception as e:
        out = _safe_test(name, h0, is_finding=True)
        out.notes = f"error: {e}"
        out.note = f"Test fallido: {e}"
        out.is_heteroscedastic = False
        return out


def white_test(y: pd.Series, X: pd.DataFrame, alpha: float = 0.05) -> TestResult:
    """White test. Version mas robusta de heterocedasticidad (no asume forma)."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.diagnostic import het_white
    from statsmodels.tools import add_constant

    name = "White (heteroscedasticity)"
    h0 = "varianza de residuos es constante (homocedasticidad)"
    df = pd.concat([y, X], axis=1).dropna()
    # White requiere n > k(k+3)/2 + 1, conservador: n >= 50
    if len(df) < 50:
        return _safe_test(name, h0, is_finding=True)
    try:
        y_clean = df[y.name]
        X_clean = add_constant(df[X.columns], has_constant="add")
        ols = OLS(y_clean.values, X_clean.values).fit()
        lm, lm_p, f, f_p = het_white(ols.resid, X_clean.values)
        return TestResult(
            name=name,
            statistic=float(lm),
            p_value=float(lm_p),
            rejects_h0=lm_p < alpha,
            h0_meaning=h0,
            is_finding=True,
            notes=f"n={len(df)}",
            extra={"lm_statistic": float(lm), "lm_pvalue": float(lm_p)},
        )
    except Exception as e:
        out = _safe_test(name, h0, is_finding=True)
        out.notes = f"error: {e}"
        return out


# ---------------------------------------------------------------------------
# Autocorrelacion
# ---------------------------------------------------------------------------
def durbin_watson(residuals: pd.Series) -> TestResult:
    """Durbin-Watson sobre residuos.

    DW ∈ [0, 4]. ≈2 => no autocorr. <1.5 => positiva. >2.5 => negativa.
    No tiene p-value canónico; reportamos la estadistica y un veredicto
    cualitativo en `notes`.
    """
    from statsmodels.stats.stattools import durbin_watson as dw

    name = "Durbin-Watson"
    h0 = "no hay autocorrelacion de primer orden en residuos"
    r = residuals.dropna()
    if len(r) < 10:
        return _safe_test(name, h0, is_finding=True)
    try:
        d = float(dw(r.values))
        if d < 1.5:
            verdict = "POSITIVA (modelo deja patron temporal)"
            rejects = True
        elif d > 2.5:
            verdict = "NEGATIVA (rara, posible overfitting)"
            rejects = True
        else:
            verdict = "no concluyente / OK"
            rejects = False
        return TestResult(
            name=name,
            statistic=d,
            p_value=None,
            rejects_h0=rejects,
            h0_meaning=h0,
            is_finding=True,
            notes=f"DW={d:.3f} → {verdict}",
        )
    except Exception as e:
        out = _safe_test(name, h0, is_finding=True)
        out.notes = f"error: {e}"
        return out


def ljung_box(residuals: pd.Series, lags: int = 10, alpha: float = 0.05) -> TestResult:
    """Ljung-Box (Q-test) sobre residuos. H0: no autocorrelacion hasta lag k."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    name = f"Ljung-Box (Q@{lags})"
    h0 = f"no hay autocorrelacion en residuos hasta lag {lags}"
    r = residuals.dropna()
    if len(r) < lags + 5:
        return _safe_test(name, h0, is_finding=True)
    try:
        result = acorr_ljungbox(r, lags=[lags], return_df=True)
        stat = float(result["lb_stat"].iloc[-1])
        p = float(result["lb_pvalue"].iloc[-1])
        return TestResult(
            name=name,
            statistic=stat,
            p_value=p,
            rejects_h0=p < alpha,
            h0_meaning=h0,
            is_finding=True,
            notes=f"n={len(r)}, lags={lags}",
        )
    except Exception as e:
        out = _safe_test(name, h0, is_finding=True)
        out.notes = f"error: {e}"
        return out


# ---------------------------------------------------------------------------
# Estacionariedad
# ---------------------------------------------------------------------------
def adf_test(x: pd.Series, alpha: float = 0.05) -> TestResult:
    """Augmented Dickey-Fuller. H0: existe raiz unitaria (NO estacionaria).

    Como rechazar H0 = "es estacionaria" = bueno, is_finding=False.
    """
    from statsmodels.tsa.stattools import adfuller

    name = "ADF (stationarity)"
    h0 = "la serie tiene raiz unitaria (NO estacionaria)"
    x_clean = x.dropna()
    if len(x_clean) < 30:
        return _safe_test(name, h0, is_finding=False)
    try:
        result = adfuller(x_clean.values, autolag="AIC")
        stat, p = float(result[0]), float(result[1])
        return TestResult(
            name=name,
            statistic=stat,
            p_value=p,
            rejects_h0=p < alpha,
            h0_meaning=h0,
            is_finding=False,
            notes=f"n={len(x_clean)}, lag_used={result[2]}",
            extra={"critical_values": dict(result[4])},
        )
    except Exception as e:
        out = _safe_test(name, h0, is_finding=False)
        out.notes = f"error: {e}"
        return out


def kpss_test(x: pd.Series, alpha: float = 0.05) -> TestResult:
    """KPSS. H0: la serie es estacionaria. Inverso al ADF, util como corroboracion."""
    from statsmodels.tsa.stattools import kpss

    name = "KPSS (stationarity)"
    h0 = "la serie es estacionaria"
    x_clean = x.dropna()
    if len(x_clean) < 30:
        return _safe_test(name, h0, is_finding=True)
    try:
        # warning suppression: KPSS ruidoso si p fuera de tabla
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, lag, crit = kpss(x_clean.values, regression="c", nlags="auto")
        return TestResult(
            name=name,
            statistic=float(stat),
            p_value=float(p),
            rejects_h0=p < alpha,
            h0_meaning=h0,
            is_finding=True,
            notes=f"n={len(x_clean)}, lag={lag}",
            extra={"critical_values": dict(crit)},
        )
    except Exception as e:
        out = _safe_test(name, h0, is_finding=True)
        out.notes = f"error: {e}"
        return out
