# A/B (2026-06-15): ¿una loss que modela varianza-proporcional-a-la-media
# (Tweedie/Gamma) bate al campeon actual (L1 + log1p) en la cola de alta
# magnitud (GRANEL/A9)? El analisis de residuales mostro heterocedasticidad:
# el error absoluto escala con el target. Tweedie (1<p<2) y Gamma (p=2)
# modelan justamente Var[y]∝mean^p, sobre el target CRUDO (sin log1p).
#
# Harness: 5 folds stratified seed42, mismo
# preproc por fold, mismo presupuesto (20 trials inner en fold 0), decision
# por business MAPE (la metrica del campeon). Brazos:
#   A_l1_log   : LightGBM regression_l1 + log1p+cap  (config del campeon)
#   B_tweedie  : LightGBM tweedie (var_power tuneado) sobre y CRUDO
#   C_gamma    : LightGBM gamma sobre y CRUDO
#
# Uso:
#   docker compose run --rm --entrypoint sh trainer -c \
#     "PYTHONPATH=/app python scripts/experiments/lgb_l1_vs_tweedie.py"
import json
import warnings

import lightgbm as lgbm
import numpy as np
import optuna

from src.config import RANDOM_STATE
from src.pipeline.build_pipeline import create_preprocessing_pipeline
from src.step_01_load.data_loader import load_business_columns, load_data
from src.step_04_train.cv_strategy import build_cv_splitters
from src.step_04_train.target_transform import _expm1, _log1p_cap
from src.step_05_evaluate.metrics import calculate_regression_metrics
from src.step_06_track.business_validation import _align_and_clean

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Presupuesto reducido a proposito: este A/B compara LOSS FUNCTIONS, no busca
# el optimo absoluto de hiperparametros. 12 trials + 700 rondas max bastan para
# un contraste justo entre brazos (todos comparten el mismo presupuesto) y lo
# mantienen en ~25 min en vez de horas.
OUTER_FOLDS, INNER_FOLDS = 5, 3
N_TRIALS = 12
MAX_ROUNDS, EARLY_STOP = 700, 50

print("=== A/B: L1+log1p (campeon) vs Tweedie vs Gamma | POP ===")
X, y = load_data(sheet="POP")
business = load_business_columns(sheet="POP")
h_ef = business["H-EF"].to_numpy(dtype=float)
kg_jr = business["KG/JR"].to_numpy(dtype=float)

outer_cv, inner_cv, strat_label, _ = build_cv_splitters(X, OUTER_FOLDS, INNER_FOLDS, RANDOM_STATE)
folds = list(outer_cv.split(X, strat_label))

fold_data = []
for _k, (tr_i, te_i) in enumerate(folds):
    pre = create_preprocessing_pipeline()
    Xt_tr = pre.fit_transform(X.iloc[tr_i], y.iloc[tr_i])
    Xt_te = pre.transform(X.iloc[te_i])
    fold_data.append(
        {
            "te_i": te_i,
            "tr_i": tr_i,
            "Xtr": Xt_tr.to_numpy(dtype=np.float64),
            "Xte": Xt_te.to_numpy(dtype=np.float64),
            "y_tr": y.iloc[tr_i].to_numpy(dtype=float),
        }
    )
print(f"folds OK | {fold_data[0]['Xtr'].shape[1]} cols | n={len(X)}")

fd0 = fold_data[0]
strat0 = strat_label.iloc[folds[0][0]] if strat_label is not None else None
inner_splits = list(inner_cv.split(fd0["Xtr"], strat0))
y0 = fd0["y_tr"]
kgjr0 = kg_jr[folds[0][0]]  # KG/JR alineado al train del fold 0 (pesos financieros)
COMMON = {"verbosity": -1, "bagging_freq": 1}


def fin_weight(kg):
    """Peso financiero ∝ KG/JR (impacto en presupuesto). Normalizado a media 1
    sobre las filas positivas, cap 5.0 (misma filosofia que el pipeline) y piso
    0.1 (las filas chicas pesan menos, no cero)."""
    kg = np.nan_to_num(np.asarray(kg, dtype=float), nan=0.0)
    m = kg[kg > 0].mean() if np.any(kg > 0) else 1.0
    return np.clip(kg / m, 0.1, 5.0)


def _grid(trial):
    md = trial.suggest_int("max_depth", 3, 8)
    return {
        "max_depth": md,
        "num_leaves": trial.suggest_int("num_leaves", 7, max(7, min(2**md - 1, 64))),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 25.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
    }


# Funcion de tuning generica por brazo. `use_log` controla el target transform.
def make_objective(objective_name, use_log, tweedie=False, weighted=False):
    def obj(trial):
        params = {**COMMON, "objective": objective_name, **_grid(trial)}
        if tweedie:
            params["tweedie_variance_power"] = trial.suggest_float(
                "tweedie_variance_power", 1.1, 1.9
            )
        maes, iters = [], []
        for itr, iva in inner_splits:
            ytr = _log1p_cap(y0[itr]) if use_log else y0[itr]
            w = fin_weight(kgjr0[itr]) if weighted else None
            ds = lgbm.Dataset(fd0["Xtr"][itr], ytr, weight=w)
            yva = _log1p_cap(y0[iva]) if use_log else y0[iva]
            bst = lgbm.train(
                params,
                ds,
                num_boost_round=MAX_ROUNDS,
                valid_sets=[lgbm.Dataset(fd0["Xtr"][iva], yva, reference=ds)],
                callbacks=[lgbm.early_stopping(EARLY_STOP, verbose=False)],
            )
            raw = bst.predict(fd0["Xtr"][iva])
            yp = np.clip(_expm1(raw) if use_log else raw, 0.0, None)
            maes.append(float(np.mean(np.abs(y0[iva] - yp))))
            iters.append(bst.best_iteration or MAX_ROUNDS)
        trial.set_user_attr("nbr", int(np.median(iters)))
        return float(np.mean(maes))

    return obj


def _evaluate(oof, train_maes):
    m = np.isfinite(oof)
    tgt = calculate_regression_metrics(y.to_numpy()[m], oof[m])
    bp, br, _, _ = _align_and_clean(oof, h_ef, kg_jr)
    biz = calculate_regression_metrics(br, bp)
    mae_train = float(np.mean(train_maes))
    gap = tgt["mae"] - mae_train
    return {
        "target_mape": tgt["mape"],
        "target_mae_oof": tgt["mae"],
        "mae_train": mae_train,
        "gap_rel": gap / tgt["mae"] if tgt["mae"] else None,
        "business_mape": biz["mape"],
        "business_r2": biz["r2"],
    }


def run_arm(name, objective_name, use_log, tweedie=False, weighted=False):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True),
    )
    study.optimize(
        make_objective(objective_name, use_log, tweedie, weighted),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )
    bp = study.best_trial.params
    nbr = max(50, study.best_trial.user_attrs["nbr"])
    params = {**COMMON, "objective": objective_name, **{k: v for k, v in bp.items()}}
    oof = np.full(len(X), np.nan)
    tmaes = []
    for fd in fold_data:
        ytr = _log1p_cap(fd["y_tr"]) if use_log else fd["y_tr"]
        w = fin_weight(kg_jr[fd["tr_i"]]) if weighted else None
        bst = lgbm.train(params, lgbm.Dataset(fd["Xtr"], ytr, weight=w), num_boost_round=nbr)
        raw_te = bst.predict(fd["Xte"])
        oof[fd["te_i"]] = np.clip(_expm1(raw_te) if use_log else raw_te, 0.0, None)
        raw_tr = bst.predict(fd["Xtr"])
        yp_tr = np.clip(_expm1(raw_tr) if use_log else raw_tr, 0.0, None)
        tmaes.append(float(np.mean(np.abs(fd["y_tr"] - yp_tr))))
    res = _evaluate(oof, tmaes)
    res["best_params"] = bp
    res["nbr"] = nbr
    print(
        f"{name:<12}: biz MAPE {res['business_mape']:.2f}% | R2 {res['business_r2']:.3f} "
        f"| gap_rel {res['gap_rel']:.3f} | tgt MAPE {res['target_mape']:.2f}%"
    )
    return res, oof


results = {}
oofs = {}
results["A_l1_log"], oofs["A"] = run_arm("A_l1_log", "regression_l1", use_log=True)
results["B_tweedie"], oofs["B"] = run_arm("B_tweedie", "tweedie", use_log=False, tweedie=True)
results["C_gamma"], oofs["C"] = run_arm("C_gamma", "gamma", use_log=False)
results["D_l1_finw"], oofs["D"] = run_arm("D_l1_finw", "regression_l1", use_log=True, weighted=True)

base = results["A_l1_log"]["business_mape"]
print("\n=== Veredicto (business MAPE; campeon A = referencia) ===")
for k, r in results.items():
    d = r["business_mape"] - base
    flag = "  <-- mejora" if d < -0.1 else ("  (peor)" if d > 0.1 else "  (empate)")
    print(f"{k:<12}: {r['business_mape']:.2f}%  ({d:+.2f}pp){flag}")

# Desglose por segmento: ¿el brazo financiero (D) baja el error en A9/GRANEL
# aunque suba el MAPE global? Ese es el trade-off absoluto vs relativo.
try:
    fundo = X["FUNDO"].astype(str).to_numpy()
    fmt = X["FORMATO"].astype(str).to_numpy()
    m_gran = fmt == "GRANEL"
    m_a9g = m_gran & (fundo == "A9")

    def seg_mape(oof, mask):
        bp, br, _, _ = _align_and_clean(oof[mask], h_ef[mask], kg_jr[mask])
        return calculate_regression_metrics(br, bp)["mape"]

    def abs_err_kg(oof):
        bp, br, _, _ = _align_and_clean(oof, h_ef, kg_jr)
        return float(np.abs(br - bp).sum())

    print("\n=== Desglose por segmento (business MAPE %) + error abs total (kg) ===")
    print(f"{'arm':<12}{'GLOBAL':>8}{'GRANEL':>9}{'A9/GRAN':>9}{'absERRkg':>11}")
    for k, key in [("A_l1_log", "A"), ("B_tweedie", "B"), ("C_gamma", "C"), ("D_l1_finw", "D")]:
        o = oofs[key]
        print(
            f"{k:<12}{results[k]['business_mape']:>8.2f}{seg_mape(o, m_gran):>9.2f}"
            f"{seg_mape(o, m_a9g):>9.2f}{abs_err_kg(o):>11.0f}"
        )
except Exception as exc:
    print(f"(desglose por segmento omitido: {exc})")

np.savez(
    "artifacts/oof_ab_tweedie.npz",
    oof_A=oofs["A"],
    oof_B=oofs["B"],
    oof_C=oofs["C"],
    oof_D=oofs["D"],
    y=y.to_numpy(),
)
with open("artifacts/AB_TWEEDIE_2026-06-15.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nguardado: artifacts/AB_TWEEDIE_2026-06-15.json")
