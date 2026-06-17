# Test de estructura residual (2026-06-13) — responde la pregunta del
# usuario: "¿puedo forzar residuos descorrelacionados entre dos modelos?".
#
# Idea: la unica forma de que un 2do modelo aporte a un stack es que los
# residuos del campeon tengan ESTRUCTURA aprendible (patron que otro
# modelo pueda capturar). Test directo: entrenar un LGB para PREDECIR el
# residuo OOF del campeon con las mismas features, evaluado OOF.
#   - R2 ~ 0  -> el residuo es ruido dado el feature set actual: ningun
#               segundo modelo sobre estas features puede descorrelacionar
#               nada util; la unica salida es INFORMACION NUEVA (clima).
#   - R2 >> 0 -> hay senal sin extraer -> un stack/boost de residuos si
#               tendria espacio.
# Evidencia previa consistente: AR(1) de residuos rho=+0.047 (serial ~0),
# sesgo por FUNDO×FORMATO ~0 en todos los grupos.
#
# Uso:
#   docker compose run --rm --entrypoint sh trainer -c \
#     "PYTHONPATH=/app python artifacts/residual_structure_test.py"
import warnings

import lightgbm as lgbm
import numpy as np

from src.config import RANDOM_STATE
from src.pipeline.build_pipeline import create_preprocessing_pipeline
from src.step_01_load.data_loader import load_data
from src.step_04_train.tuning import _build_cv_splitters

warnings.filterwarnings("ignore")

print("=== Test de estructura en residuos del campeon (LGB control) ===")
X, y = load_data(sheet="POP")
d = np.load("artifacts/oof_control_lgb_gpb.npz")
assert np.allclose(y.to_numpy(), d["y"]), "npz desalineado"
resid = y.to_numpy() - d["oof_lgb"]  # residuo OOF del LGB control (14.23%)

outer_cv, _, strat_label, _ = _build_cv_splitters(X, 5, 3, RANDOM_STATE)

oof_resid_pred = np.full(len(X), np.nan)
for tr_i, te_i in outer_cv.split(X, strat_label):
    pre = create_preprocessing_pipeline()
    Xt_tr = pre.fit_transform(X.iloc[tr_i], y.iloc[tr_i]).to_numpy(dtype=float)
    Xt_te = pre.transform(X.iloc[te_i]).to_numpy(dtype=float)
    bst = lgbm.train(
        {"objective": "regression", "verbosity": -1, "max_depth": 6,
         "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 20},
        lgbm.Dataset(Xt_tr, resid[tr_i]), num_boost_round=300,
    )
    oof_resid_pred[te_i] = bst.predict(Xt_te)

m = np.isfinite(oof_resid_pred) & np.isfinite(resid)
ss_res = float(np.sum((resid[m] - oof_resid_pred[m]) ** 2))
ss_tot = float(np.sum((resid[m] - resid[m].mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot
corr = float(np.corrcoef(resid[m], oof_resid_pred[m])[0, 1])
print(f"R2 OOF del modelo-de-residuos: {r2:+.4f}")
print(f"corr(residuo real, residuo predicho): {corr:+.4f}")
print("Interpretacion: R2~0 => el residuo no tiene estructura aprendible "
      "con las features actuales; informacion nueva (clima) es la unica via.")
