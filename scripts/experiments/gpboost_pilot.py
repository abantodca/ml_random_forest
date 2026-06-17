# Piloto GPBoost (tarea #13, 2026-06-13) — ¿efectos mixtos por FUNDO×FORMATO
# mejoran al campeón LGB (14.27% business MAPE OOF / run 1ed787be)?
#
# Decisión de diseño (pedido del usuario): PRIMERO medir si vale la pena en
# un script standalone; SOLO si mejora se considera backend formal en
# registry.py. No toca código del trainer.
#
# Protocolo (máxima comparabilidad con el campeón):
#   - Mismos datos: load_data(POP) + load_business_columns (alineados 1:1).
#   - Mismos folds: _build_cv_splitters(X, outer=5, inner=3, seed=42) ==
#     perfil prod con CV_OUTER_STRATEGY=stratified (default).
#   - Mismo preprocesamiento: create_preprocessing_pipeline() fit POR fold
#     (lags sin leakage cross-fold).
#   - Mismo target transform: _log1p_cap en fit, _expm1 al predecir.
#   - Misma métrica: target-unit (KG/JR_H) + business-unit (pred × H-EF vs
#     KG/JR) vía _align_and_clean + calculate_regression_metrics.
#
# Simplificaciones documentadas (pilot-grade, no certificación):
#   - Tuning Optuna 20 trials SOLO en el train del fold 0 (inner 3-fold
#     sobre la matriz ya transformada del fold; el preproc no se re-ajusta
#     por inner fold). El campeón tuvo nested tuning completo de 60 trials:
#     si GPBoost queda a <0.5pp del campeón con este presupuesto, merece
#     el run con protocolo completo.
#   - num_boost_round final = mediana de best_iteration de los inner folds.
#
# Brazos:
#   C1: RE = FUNDO×FORMATO (1 factor), SIN dummies de grupo en X.
#   C2: RE = FUNDO + FORMATO cruzados (2 factores), SIN dummies.
#   D : RE = FUNDO×FORMATO, CON dummies (los árboles también ven el grupo).
# Anclas: champion prod (npz OOF) target-unit y business-unit.
#
# Uso:
#   docker compose run --rm --entrypoint sh trainer -c \
#     "pip install -q gpboost && python artifacts/gpboost_pilot.py"
import json
import warnings

import gpboost as gpb
import numpy as np
import optuna

from src.config import RANDOM_STATE
from src.pipeline.build_pipeline import create_preprocessing_pipeline
from src.step_01_load.data_loader import load_business_columns, load_data
from src.step_04_train.target_transform import _expm1, _log1p_cap
from src.step_04_train.tuning import _build_cv_splitters
from src.step_05_evaluate.metrics import calculate_regression_metrics
from src.step_06_track.business_validation import _align_and_clean

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTER_FOLDS, INNER_FOLDS = 5, 3  # perfil prod
N_TRIALS = 20
MAX_ROUNDS, EARLY_STOP = 2000, 50
GROUP_DUMMY_PREFIXES = ("FUNDO__", "FORMATO__", "FUNDO_FORMATO__")

print("=== Piloto GPBoost POP ===")
X, y = load_data(sheet="POP")
business = load_business_columns(sheet="POP")
h_ef = business["H-EF"].to_numpy(dtype=float)
kg_jr = business["KG/JR"].to_numpy(dtype=float)
print(f"filas: {len(X)}")

# ---- Ancla: OOF del campeón prod (lgb v6) ----
npz = np.load("artifacts/oof_POP_lgb_v6.npz")
assert np.allclose(y.to_numpy(), npz["y_true"]), "npz desalineado con load_data"
champ_oof = npz["y_pred"]
m = np.isfinite(champ_oof)
champ_target = calculate_regression_metrics(y.to_numpy()[m], champ_oof[m])
cp, cr, _, _ = _align_and_clean(champ_oof, h_ef, kg_jr)
champ_biz = calculate_regression_metrics(cr, cp)
print(f"ANCLA campeón: target MAPE {champ_target['mape']:.2f}% | "
      f"business MAPE {champ_biz['mape']:.2f}% R2 {champ_biz['r2']:.3f}")

# ---- Folds idénticos al prod ----
outer_cv, inner_cv, strat_label, strategy = _build_cv_splitters(
    X, OUTER_FOLDS, INNER_FOLDS, RANDOM_STATE
)
print(f"estratificación: {strategy}")
folds = list(outer_cv.split(X, strat_label))

# ---- Preproc por fold (cacheado, compartido por todos los brazos) ----
ff_all = X["FUNDO"].astype(str) + "___" + X["FORMATO"].astype(str)
fold_data = []
for k, (tr_i, te_i) in enumerate(folds):
    pre = create_preprocessing_pipeline()
    Xt_tr = pre.fit_transform(X.iloc[tr_i], y.iloc[tr_i])
    Xt_te = pre.transform(X.iloc[te_i])
    fold_data.append({
        "tr_i": tr_i, "te_i": te_i,
        "Xt_tr": Xt_tr, "Xt_te": Xt_te,
        "y_tr": y.iloc[tr_i].to_numpy(), "y_te": y.iloc[te_i].to_numpy(),
        "ff_tr": ff_all.iloc[tr_i].to_numpy(), "ff_te": ff_all.iloc[te_i].to_numpy(),
        "fundo_tr": X["FUNDO"].astype(str).iloc[tr_i].to_numpy(),
        "fundo_te": X["FUNDO"].astype(str).iloc[te_i].to_numpy(),
        "fmt_tr": X["FORMATO"].astype(str).iloc[tr_i].to_numpy(),
        "fmt_te": X["FORMATO"].astype(str).iloc[te_i].to_numpy(),
    })
    print(f"fold {k}: preproc OK | train {len(tr_i)} test {len(te_i)} "
          f"cols {Xt_tr.shape[1]}")


def _matrix(df, drop_dummies):
    cols = [c for c in df.columns
            if not (drop_dummies and c.startswith(GROUP_DUMMY_PREFIXES))]
    return df[cols].to_numpy(dtype=np.float64)


def _make_gp(fd, kind, idx=None, te=False):
    """GPModel del brazo: 'FF' (1 factor) o 'CROSSED' (FUNDO + FORMATO)."""
    side = "te" if te else "tr"
    sel = slice(None) if idx is None else idx
    if kind == "CROSSED":
        g = np.column_stack([fd[f"fundo_{side}"][sel], fd[f"fmt_{side}"][sel]])
    else:
        g = fd[f"ff_{side}"][sel]
    return g


def _train_one(params, nbr, X_tr, y_tr_log, g_tr, X_va=None, y_va_log=None,
               g_va=None):
    """Entrena un booster GPBoost; con valid -> early stopping."""
    gp_model = gpb.GPModel(group_data=g_tr, likelihood="gaussian")
    ds = gpb.Dataset(X_tr, y_tr_log)
    kwargs = {}
    if X_va is not None:
        gp_model.set_prediction_data(group_data_pred=g_va)
        kwargs = {
            "valid_sets": gpb.Dataset(X_va, y_va_log, reference=ds),
            "early_stopping_rounds": EARLY_STOP,
            "use_gp_model_for_validation": True,
        }
    bst = gpb.train(params=params, train_set=ds, gp_model=gp_model,
                    num_boost_round=nbr, **kwargs)
    return bst


def _predict(bst, X_te, g_te):
    pred = bst.predict(data=X_te, group_data_pred=g_te,
                       predict_var=False, pred_latent=False)
    out = pred["response_mean"] if isinstance(pred, dict) else pred
    return np.clip(_expm1(np.asarray(out, dtype=float)), 0.0, None)


# ---- Tuning dual (árboles) en fold 0, RE=FF sin dummies ----
fd0 = fold_data[0]
X0 = _matrix(fd0["Xt_tr"], drop_dummies=True)
y0_log = _log1p_cap(fd0["y_tr"])
strat0 = strat_label.iloc[fd0["tr_i"]] if strat_label is not None else None
inner_splits = list(inner_cv.split(X0, strat0))


def objective(trial):
    max_depth = trial.suggest_int("max_depth", 3, 8)
    params = {
        "objective": "regression",
        "verbose": -1,
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int(
            "num_leaves", 7, max(7, min(2 ** max_depth - 1, 64))),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 60, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        # GPBoost no soporta bagging ("Bagging cannot be applied for the
        # GPBoost algorithm") — solo feature_fraction como regularizador
        # estocastico.
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
    }
    maes, iters = [], []
    for itr, iva in inner_splits:
        bst = _train_one(
            params, MAX_ROUNDS, X0[itr], y0_log[itr],
            _make_gp(fd0, "FF", idx=itr),
            X0[iva], y0_log[iva], _make_gp(fd0, "FF", idx=iva),
        )
        yp = _predict(bst, X0[iva], _make_gp(fd0, "FF", idx=iva))
        maes.append(float(np.mean(np.abs(fd0["y_tr"][iva] - yp))))
        iters.append(bst.best_iteration or MAX_ROUNDS)
    trial.set_user_attr("nbr", int(np.median(iters)))
    return float(np.mean(maes))


study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
best = study.best_trial
nbr_final = max(50, best.user_attrs["nbr"])
best_params = {
    "objective": "regression", "verbose": -1,
    **{k: v for k, v in best.params.items()},
}
print(f"tuning OK: inner MAE {best.value:.4f} | nbr {nbr_final} | "
      f"params {best.params}")

# ---- OOF de los 3 brazos con params tuneados ----
ARMS = [
    ("C1_FF_sin_dummies", "FF", True),
    ("C2_crossed_sin_dummies", "CROSSED", True),
    ("D_FF_con_dummies", "FF", False),
]
results = {}
for name, kind, drop in ARMS:
    oof = np.full(len(X), np.nan)
    for fd in fold_data:
        bst = _train_one(
            best_params, nbr_final,
            _matrix(fd["Xt_tr"], drop), _log1p_cap(fd["y_tr"]),
            _make_gp(fd, kind),
        )
        oof[fd["te_i"]] = _predict(bst, _matrix(fd["Xt_te"], drop),
                                   _make_gp(fd, kind, te=True))
    m = np.isfinite(oof)
    tgt = calculate_regression_metrics(y.to_numpy()[m], oof[m])
    bp, br, _, _ = _align_and_clean(oof, h_ef, kg_jr)
    biz = calculate_regression_metrics(br, bp)
    results[name] = {
        "target_mape": tgt["mape"], "target_r2": tgt["r2"],
        "business_mape": biz["mape"], "business_r2": biz["r2"],
    }
    print(f"{name}: target MAPE {tgt['mape']:.2f}% | "
          f"business MAPE {biz['mape']:.2f}% R2 {biz['r2']:.3f}")

out = {
    "fecha": "2026-06-13",
    "protocolo": "5-fold stratified seed42, preproc por fold, log1p+cap, "
                 "tuning 20 trials fold0 (simplificado)",
    "ancla_champion": {
        "target_mape": champ_target["mape"], "target_r2": champ_target["r2"],
        "business_mape": champ_biz["mape"], "business_r2": champ_biz["r2"],
    },
    "best_params": best.params, "num_boost_round": nbr_final,
    "brazos": results,
}
with open("artifacts/GPBOOST_PILOT_2026-06-13.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nguardado: artifacts/GPBOOST_PILOT_2026-06-13.json")
