"""P0.1b — select_champion: gate de gap RELATIVO en ambos extremos de escala.

El gate del campeon es gap_rel = |gap| / MAE_test <= CHAMPION_MAX_GAP_REL
(adimensional, comparable entre variedades con targets de escala distinta).
Lex-order: gate -> menor MAPE OOF (tolerancia 0.5pp) -> menor tiempo.
"""
from __future__ import annotations

from src.config import CHAMPION_MAX_GAP_REL
from src.step_05_evaluate.champion import ModelResult, select_champion


def _resultado(model_type, gap, mae_test, mape_oof, elapsed=100.0):
    return ModelResult(
        model_type=model_type,
        metrics={
            "nested_cv_gap_mean": gap,
            "nested_cv_mae_mean": mae_test,
        },
        best_params={},
        mlflow_run_id=f"run_{model_type}",
        pipeline_path="",
        elapsed_seconds=elapsed,
        business_metrics_oof={"mape": mape_oof},
    )


def test_gate_descarta_mejor_mape_con_overfit():
    # B tiene mejor MAPE pero gap_rel = 0.9/1.0 > umbral -> gana A.
    a = _resultado("a", gap=0.10, mae_test=1.0, mape_oof=15.0)
    b = _resultado("b", gap=0.90, mae_test=1.0, mape_oof=10.0)
    assert b.gap_rel > CHAMPION_MAX_GAP_REL
    assert select_champion([a, b]) is a


def test_gate_es_invariante_a_escala_del_target():
    # Mismo gap RELATIVO en un target 100x mas grande: misma decision.
    # (El gate absoluto viejo en pp aprobaba/rechazaba segun la escala.)
    chico = _resultado("chico", gap=0.10, mae_test=1.0, mape_oof=15.0)
    grande = _resultado("grande", gap=10.0, mae_test=100.0, mape_oof=15.0)
    assert abs(chico.gap_rel - grande.gap_rel) < 1e-12
    assert (chico.gap_rel <= CHAMPION_MAX_GAP_REL) == (
        grande.gap_rel <= CHAMPION_MAX_GAP_REL
    )


def test_menor_mape_oof_gana_si_ambos_pasan_gate():
    a = _resultado("a", gap=0.10, mae_test=1.0, mape_oof=16.0)
    b = _resultado("b", gap=0.10, mae_test=1.0, mape_oof=14.0)
    assert select_champion([a, b]) is b


def test_empate_practico_de_mape_lo_decide_el_tiempo():
    # Diferencia 0.3pp (< tolerancia 0.5pp) -> gana el mas rapido.
    lento = _resultado("lento", gap=0.10, mae_test=1.0, mape_oof=14.0,
                       elapsed=900.0)
    rapido = _resultado("rapido", gap=0.10, mae_test=1.0, mape_oof=14.3,
                        elapsed=100.0)
    assert select_champion([lento, rapido]) is rapido
