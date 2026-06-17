"""Metadata por run para trazabilidad / reproducibilidad.

Tags que tiene sentido loggear como `mlflow.set_tags`:
    git_commit         : SHA del HEAD si estamos en un repo git
    git_dirty          : "true" si hay cambios no commiteados
    docker_image_tag   : si el container expone IMAGE_TAG via env
    dataset_sha256     : hash SHA-256 del Excel de entrada
    dataset_n_rows     : filas del subset usado en el training
    dataset_n_cols     : columnas raw

Estos tags hacen posible:
    - Reproducir un training: clonar el commit + bajar el dataset hash + correr
    - Detectar drift: si dataset_sha256 cambia entre runs, la data se actualizo
    - Auditar: que codigo + que data produjo este model en Registry
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _git_info(repo_root: Path | None = None) -> dict[str, str]:
    """SHA del HEAD + flag de dirty.

    Resolucion en orden:
      1. Env vars GIT_SHA / GIT_DIRTY -> tiene prioridad (inyectado en
         docker run via Taskfile, ya que el container no monta .git).
      2. `git -C <repo>` binary -> funciona en host con repo git real.
      3. Fallback "unknown".

    Esto resuelve el caso "container sin .git montado" sin necesidad de
    bind-mount del .git (ahorra I/O en Windows + Docker Desktop).
    """
    # 1) Env var explicita gana
    env_sha = os.environ.get("GIT_SHA", "").strip()
    if env_sha and env_sha != "unknown":
        return {
            "git_commit": env_sha[:12],
            "git_dirty": os.environ.get("GIT_DIRTY", "unknown"),
        }

    # 2) git binary contra repo en disco
    cwd = repo_root or Path.cwd()
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "-C", str(cwd), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        dirty = "true" if status.strip() else "false"
        return {"git_commit": sha[:12], "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def _dataset_hash(file_path: Path, chunk: int = 1024 * 1024) -> str:
    """SHA-256 del archivo (streamed para no cargar 100MB en memoria)."""
    if not file_path.exists():
        return "missing"
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]  # 16 chars suficientes para identificar


def collect_run_metadata(
    *,
    training_file: Path,
    n_rows: int,
    n_cols: int,
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Recolecta tags de trazabilidad para un MLflow run.

    Devuelve un dict listo para pasar a `mlflow.set_tags(...)`.
    """
    tags: dict[str, str] = {}
    tags.update(_git_info(repo_root))
    tags["dataset_sha256"] = _dataset_hash(training_file)
    tags["dataset_path"] = str(training_file.name)
    tags["dataset_n_rows"] = str(n_rows)
    tags["dataset_n_cols"] = str(n_cols)
    # Image tag desde build args (Dockerfile expone GIT_SHA / VERSION)
    if "IMAGE_TAG" in os.environ:
        tags["docker_image_tag"] = os.environ["IMAGE_TAG"]
    if "GIT_SHA" in os.environ:
        tags["docker_git_sha"] = os.environ["GIT_SHA"][:12]
    return tags
