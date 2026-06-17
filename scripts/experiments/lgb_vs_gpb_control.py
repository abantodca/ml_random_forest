# Control de presupuesto justo (2026-06-13) — respuesta a la objeción del
# usuario: "el 17.92% de GPBoost fue un entrenamiento light vs el campeón
# de 3+ horas". Aquí TODO corre con el MISMO harness del piloto:
#   - mismos 5 folds stratified seed42, mismo preproc por fold,
#   - mismo target transform (log1p+cap), misma métrica,
#   - mismo presupuesto: Optuna 20 trials en fold 0 para CADA modelo.
# Brazos:
#   B  : LightGBM plano (con dummies) — control mismo presupuesto.
#   C2 : GPBoost RE cruzados FUNDO+FORMATO (sin dummies) — params YA
#        tuneados del piloto (artifacts/GPBOOST_PILOT_2026-06-13.json).
#   S  : stacking Ridge sobre los OOF de B y C2 (cross_val_predict para
#        no evaluar el meta-modelo sobre sus propios datos de ajuste).
# Además reporta overfitting por brazo: MAE train (in-sample del fold) vs
# MAE OOF -> gap_rel comparable al gate del campeón.
#
# Uso:
#   docker compose run --rm --entrypoint sh trainer -c \
#     "pip install -q gpboost && PYTHONPATH=/app python artifacts/lgb_vs_gpb_control.py"
import json
import warnings

import gpboost as gpb
import lightgbm as lgbm
import numpy as np
import optuna
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

from src.config import RANDOM_STATE
from src.pipeline.build_pipeline import create_preprocessing_pipeline
from src.step_01_load.data_loader import load_business_columns, load_data
from src.step_04_train.target_transform import _expm1, _log1p_cap
from src.step_04_train.tuning import _build_cv_splitters
from src.step_05_evaluate.metrics import calculate_regression_metrics
from src.step_06_track.business_validation import _align_and_clean

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTER_FOLDS, INNER_FOLDS = 5, 3
N_TRIALS = 20
MAX_ROUNDS, EARLY_STOP = 2000, 50
GROUP_DUMMY_PREFIXES = ("FUNDO__", "FORMATO__", "FUNDO_FORMATO__")

print("=== Control mismo presupuesto: LGB vs GPBoost ===")
X, y = load_data(sheet="POP")
business = load_business_columns(sheet="POP")
h_ef = business["H-EF"].to_numpy(dtype=float)
kg_jr = business["KG/JR"].to_numpy(dtype=float)

with open("artifacts/GPBOOST_PILOT_2026-06-13.json") as _f:
    pilot = json.load(_f)
gpb_params = {"objective": "regression", "verbose": -1, **pilot["best_params"]}
gpb_nbr = pilot["num_boost_round"]
print(f"params GPBoost (del piloto): {pilot['best_params']} | nbr {gpb_nbr}")

outer_cv, inner_cv, strat_label, strategy = _build_cv_splitters(
    X, OUTER_FOLDS, INNER_FOLDS, RANDOM_STATE
)
folds = list(outer_cv.split(X, strat_label))

fold_data = []
for k, (tr_i, te_i) in enumerate(folds):
    pre = create_preprocessing_pipeline()
    Xt_tr = pre.fit_transform(X.iloc[tr_i], y.iloc[tr_i])
    Xt_te = pre.transform(X.iloc[te_i])
    fold_data.append({
        "tr_i": tr_i, "te_i": te_i, "Xt_tr": Xt_tr, "Xt_te": Xt_te,
        "y_tr": y.iloc[tr_i].to_numpy(), "y_te": y.iloc[te_i].to_numpy(),
        "fundo_tr": X["FUNDO"].astype(str).iloc[tr_i].to_numpy(),
        "fundo_te": X["FUNDO"].astype(str).iloc[te_i].to_numpy(),
        "fmt_tr": X["FORMATO"].astype(str).iloc[tr_i].to_numpy(),
        "fmt_te": X["FORMATO"].astype(str).iloc[te_i].to_numpy(),
    })
    print(f"fold {k}: preproc OK ({Xt_tr.shape[1]} cols)")


def _matrix(df, drop_dummies):
    cols = [c for c in df.columns
            if not (drop_dummies and c.startswith(GROUP_DUMMY_PREFIXES))]
    return df[cols].to_numpy(dtype=np.float64)


def _crossed(fd, te=False, idx=None):
    side = "te" if te else "tr"
    sel = slice(None) if idx is None else idx
    return np.column_stack([fd[f"fundo_{side}"][sel], fd[f"fmt_{side}"][sel]])


# ---- Tuning LGB plano: mismo presupuesto (20 trials, fold 0) ----
fd0 = fold_data[0]
X0 = _matrix(fd0["Xt_tr"], drop_dummies=False)
y0_log = _log1p_cap(fd0["y_tr"])
strat0 = strat_label.iloc[fd0["tr_i"]] if strat_label is not None else None
inner_splits = list(inner_cv.split(X0, strat0))


def objective_lgb(trial):
    max_depth = trial.suggest_int("max_depth", 3, 8)
    params = {
        "objective": "regression", "verbosity": -1,
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int(
            "num_leaves", 7, max(7, min(2 ** max_depth - 1, 64))),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 60, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 1,
    }
    maes, iters = [], []
    for itr, iva in inner_splits:
        ds = lgbm.Dataset(X0[itr], y0_log[itr])
        bst = lgbm.train(
            params, ds, num_boost_round=MAX_ROUNDS,
            valid_sets=[lgbm.Dataset(X0[iva], y0_log[iva], reference=ds)],
            callbacks=[lgbm.early_stopping(EARLY_STOP, verbose=False)],
        )
        yp = np.clip(_expm1(bst.predict(X0[iva])), 0.0, None)
        maes.append(float(np.mean(np.abs(fd0["y_tr"][iva] - yp))))
        iters.append(bst.best_iteration or MAX_ROUNDS)
    trial.set_user_attr("nbr", int(np.median(iters)))
    return float(np.mean(maes))


study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective_lgb, n_trials=N_TRIALS, show_progress_bar=False)
lgb_params = {"objective": "regression", "verbosity": -1, "bagging_freq": 1,
              **study.best_trial.params}
lgb_nbr = max(50, study.best_trial.user_attrs["nbr"])
print(f"tuning LGB OK: inner MAE {study.best_trial.value:.4f} | nbr {lgb_nbr} | "
      f"{study.best_trial.params}")


def _evaluate(oof, train_maes):
    m = np.isfinite(oof)
    tgt = calculate_regression_metrics(y.to_numpy()[m], oof[m])
    bp, br, _, _ = _align_and_clean(oof, h_ef, kg_jr)
    biz = calculate_regression_metrics(br, bp)
    mae_train = float(np.mean(train_maes))
    gap = tgt["mae"] - mae_train
    return {
        "target_mape": tgt["mape"], "target_r2": tgt["r2"],
        "target_mae_oof": tgt["mae"], "mae_train": mae_train,
        "gap": gap, "gap_rel": gap / tgt["mae"] if tgt["mae"] else None,
        "business_mape": biz["mape"], "business_r2": biz["r2"],
    }


# ---- OOF brazo B: LGB plano ----
oof_lgb = np.full(len(X), np.nan)
lgb_train_maes = []
for fd in fold_data:
    Xm_tr = _matrix(fd["Xt_tr"], False)
    ds = lgbm.Dataset(Xm_tr, _log1p_cap(fd["y_tr"]))
    bst = lgbm.train(lgb_params, ds, num_boost_round=lgb_nbr)
    oof_lgb[fd["te_i"]] = np.clip(
        _expm1(bst.predict(_matrix(fd["Xt_te"], False))), 0.0, None)
    yp_tr = np.clip(_expm1(bst.predict(Xm_tr)), 0.0, None)
    lgb_train_maes.append(float(np.mean(np.abs(fd["y_tr"] - yp_tr))))
res_lgb = _evaluate(oof_lgb, lgb_train_maes)
print(f"B_lgb_mismo_presupuesto: biz MAPE {res_lgb['business_mape']:.2f}% "
      f"R2 {res_lgb['business_r2']:.3f} gap_rel {res_lgb['gap_rel']:.3f}")

# ---- OOF brazo C2: GPBoost cruzado (params del piloto) ----
oof_gpb = np.full(len(X), np.nan)
gpb_train_maes = []
for fd in fold_data:
    Xm_tr = _matrix(fd["Xt_tr"], True)
    gp_model = gpb.GPModel(group_data=_crossed(fd), likelihood="gaussian")
    ds = gpb.Dataset(Xm_tr, _log1p_cap(fd["y_tr"]))
    bst = gpb.train(params=gpb_params, train_set=ds, gp_model=gp_model,
                    num_boost_round=gpb_nbr)
    def _pred(Xm, g, bst=bst):  # bind del loop var: cada fold usa SU booster
        p = bst.predict(data=Xm, group_data_pred=g,
                        predict_var=False, pred_latent=False)
        out = p["response_mean"] if isinstance(p, dict) else p
        return np.clip(_expm1(np.asarray(out, dtype=float)), 0.0, None)
    oof_gpb[fd["te_i"]] = _pred(_matrix(fd["Xt_te"], True), _crossed(fd, te=True))
    yp_tr = _pred(Xm_tr, _crossed(fd))
    gpb_train_maes.append(float(np.mean(np.abs(fd["y_tr"] - yp_tr))))
res_gpb = _evaluate(oof_gpb, gpb_train_maes)
print(f"C2_gpboost_crossed: biz MAPE {res_gpb['business_mape']:.2f}% "
      f"R2 {res_gpb['business_r2']:.3f} gap_rel {res_gpb['gap_rel']:.3f}")

# ---- Brazo S: stacking Ridge sobre los dos OOF ----
mask = np.isfinite(oof_lgb) & np.isfinite(oof_gpb)
Z = np.column_stack([oof_lgb[mask], oof_gpb[mask]])
yv = y.to_numpy()[mask]
stack_pred = cross_val_predict(Ridge(alpha=1.0), Z, yv, cv=5)
oof_stack = np.full(len(X), np.nan)
oof_stack[mask] = np.clip(stack_pred, 0.0, None)
res_stack = _evaluate(oof_stack, [np.nan])
for _k in ("mae_train", "gap", "gap_rel"):
    res_stack.pop(_k)
w = Ridge(alpha=1.0).fit(Z, yv).coef_
print(f"S_stack_ridge: biz MAPE {res_stack['business_mape']:.2f}% "
      f"R2 {res_stack['business_r2']:.3f} | pesos lgb={w[0]:.2f} gpb={w[1]:.2f}")

out = {
    "fecha": "2026-06-13",
    "protocolo": "harness del piloto; 20 trials/modelo; folds y preproc identicos",
    "B_lgb_mismo_presupuesto": res_lgb,
    "C2_gpboost_crossed": res_gpb,
    "S_stack_ridge": {**res_stack, "pesos": {"lgb": float(w[0]), "gpb": float(w[1])}},
    "lgb_params": study.best_trial.params, "lgb_nbr": lgb_nbr,
}
np.savez("artifacts/oof_control_lgb_gpb.npz", oof_lgb=oof_lgb, oof_gpb=oof_gpb,
         y=y.to_numpy())
with open("artifacts/LGB_VS_GPB_CONTROL_2026-06-13.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nguardado: artifacts/LGB_VS_GPB_CONTROL_2026-06-13.json")
