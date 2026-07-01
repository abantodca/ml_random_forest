"""Promueve un modelo MLflow a Production aplicando quality gates.

Gates:
  1. Absoluto: MAPE del candidato <= MAX_MAPE.
  2. A/B: MAPE del candidato < MAPE de la version Production actual (si existe).

Uso:
  python scripts/promote_model.py rnd-forest-POP 3 --max-mape 20

La URI de MLflow se resuelve desde:
  1. --uri (flag explicito), o
  2. env MLFLOW_ALB_DNS  (http://$MLFLOW_ALB_DNS).

Nota: el fallback historico `terraform output -raw alb_dns` se removio cuando
la carpeta `infra/` migro a codigo pegable en GUIA_MLOPS_AWS_V2.md. Para
obtener el DNS sin tener Terraform a mano: `aws elbv2 describe-load-balancers
--query 'LoadBalancers[?contains(LoadBalancerName,\\`mlflow\\`)].DNSName' --output text`.
"""

from __future__ import annotations

import argparse
import os
import sys

from mlflow.tracking import MlflowClient

# Nombre real que loguea el trainer: `business_oof_mape` (ver
# BusinessValidation.to_mlflow_metrics, prefijo business_oof_). Los otros dos
# quedan como fallback para runs antiguos con la convencion vieja.
METRIC_KEYS = ("business_oof_mape", "mape_oof", "mape")


def resolve_uri(uri: str | None) -> str:
    if uri:
        return uri
    alb = os.environ.get("MLFLOW_ALB_DNS")
    if alb:
        return f"http://{alb}"
    sys.exit(
        "ERROR no se pudo resolver MLflow URI. Provee --uri <url> o exporta "
        "MLFLOW_ALB_DNS=<dns-del-alb>."
    )


def get_mape(client: MlflowClient, run_id: str) -> float:
    metrics = client.get_run(run_id).data.metrics
    for key in METRIC_KEYS:
        if key in metrics:
            return metrics[key]
    sys.exit(f"ERROR el run {run_id} no tiene ninguna metrica MAPE ({', '.join(METRIC_KEYS)})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_name")
    p.add_argument("version")
    p.add_argument("--max-mape", type=float, default=20.0)
    p.add_argument("--uri", default=None)
    args = p.parse_args()

    import mlflow

    mlflow.set_tracking_uri(resolve_uri(args.uri))
    client = MlflowClient()

    print(f">>> {args.model_name} v{args.version}  (gate MAPE <= {args.max_mape})")

    try:
        candidate = client.get_model_version(args.model_name, args.version)
    except Exception:
        sys.exit(f"ERROR no se encontro {args.model_name} v{args.version}")

    mape_new = get_mape(client, candidate.run_id)
    print(f"    candidato mape={mape_new}")
    if mape_new > args.max_mape:
        sys.exit(f"GATE FAIL absoluto  MAPE={mape_new} > {args.max_mape}")
    print("GATE absoluto OK")

    prod = client.get_latest_versions(args.model_name, stages=["Production"])
    if not prod:
        print("Sin Production previo, skip A/B")
    else:
        mape_prod = get_mape(client, prod[0].run_id)
        print(f"    Production v{prod[0].version} mape={mape_prod}")
        if mape_new >= mape_prod:
            sys.exit(
                f"GATE FAIL A/B  candidato MAPE={mape_new} no mejora vs "
                f"Production v{prod[0].version} MAPE={mape_prod}"
            )
        print("GATE A/B OK")

    print(">>> Transicionando a Production (archive existing)...")
    new = client.transition_model_version_stage(
        name=args.model_name,
        version=args.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"OK {new.name} v{new.version} en {new.current_stage}")


if __name__ == "__main__":
    main()
