"""Divide BD_HISTORICO_ACUMULADO.xlsx en una hoja por VARIEDAD.

Entrada : Excel con una unica hoja `acumulado` (todas las variedades).
Salida  : Excel con una hoja por variedad (matching el formato historico
          que consume `src.step_01_load.data_loader`).

Diseno:
    - Las columnas del acumulado vienen con espacios (` HA`, ` KG/JR`, ...).
      Aqui se normalizan UNA SOLA VEZ para que el data_loader downstream
      no tenga que hacerlo cada vez.
    - Variedades con muy pocas filas se descartan (`--min-rows`, default 100):
      no hay datos suficientes para tunear con CV anidado.
    - Dedup estructural: filas 100% identicas se eliminan por variedad
      (keep="first"), UNA sola vez. El data_loader mantiene un dedup
      defensivo que queda en no-op con el dato ya limpio.
    - Nombres de hoja saneados (Excel: max 31 chars, sin /\\?*[]:).

CLI:
    python -m scripts.prepare_data \\
        --input  data/BD_HISTORICO_ACUMULADO.xlsx \\
        --output data/training/DB-HISTORICA.xlsx \\
        --min-rows 100
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

_INVALID_SHEET_CHARS = re.compile(r"[\\/\?\*\[\]:]+")


def _sanitize_sheet_name(name: str) -> str:
    """Excel exige nombres de hoja <=31 chars y sin \\/?*[]:."""
    cleaned = _INVALID_SHEET_CHARS.sub("_", str(name)).strip()
    return cleaned[:31] if cleaned else "SIN_NOMBRE"


def _split_by_variety(
    df: pd.DataFrame,
    min_rows: int,
    variety_col: str = "VARIEDAD",
) -> Iterable[tuple[str, pd.DataFrame]]:
    """Itera (sheet_name, sub_df) por cada variedad con suficientes filas."""
    if variety_col not in df.columns:
        raise ValueError(f"La columna '{variety_col}' no existe. Disponibles: {list(df.columns)}")

    df = df.dropna(subset=[variety_col]).copy()
    df[variety_col] = df[variety_col].astype(str).str.strip()

    grouped = df.groupby(variety_col, sort=False)
    seen: set[str] = set()
    for variety, sub in grouped:
        if len(sub) < min_rows:
            continue
        sheet = _sanitize_sheet_name(variety)
        # evita colisiones tras sanitizar (extremadamente raro)
        suffix = 1
        base = sheet
        while sheet in seen:
            suffix += 1
            sheet = f"{base[:28]}_{suffix}"
        seen.add(sheet)
        yield sheet, sub.reset_index(drop=True)


def split_workbook(
    input_path: str | Path,
    output_path: str | Path,
    min_rows: int = 100,
    source_sheet: str = "acumulado",
    variety_col: str = "VARIEDAD",
) -> dict:
    """Funcion programatica reusable. Devuelve el resumen del split."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path, sheet_name=source_sheet)
    df.columns = [str(c).strip() for c in df.columns]

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "min_rows": min_rows,
        "varieties_total": int(df[variety_col].nunique()),
        "varieties_written": 0,
        "varieties_skipped": 0,
        "rows_written": 0,
        "duplicates_dropped": 0,
        "sheets": [],
    }

    skipped: list[tuple[str, int]] = []
    written: list[dict] = []
    duplicates_dropped_total = 0

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet, sub in _split_by_variety(df, min_rows=min_rows, variety_col=variety_col):
            # Dedup estructural AGUAS ARRIBA (2026-06-15): elimina filas 100%
            # identicas una sola vez, para que DB-HISTORICA.xlsx quede canonico
            # y limpio. Antes el unico dedup vivia en data_loader._load_sheet_aligned
            # y corria 1x por backend (lgb/xgb releen la misma hoja) -> el
            # warning "N duplicados estructurales" se repetia 3-4x por training.
            # keep="first": orden Excel = orden temporal (mismo criterio que el
            # loader). El min_rows ya se evaluo en _split_by_variety sobre el
            # conteo crudo, asi que la seleccion de variedades NO cambia.
            n_pre = len(sub)
            sub = sub.drop_duplicates(keep="first").reset_index(drop=True)
            n_dup = n_pre - len(sub)
            duplicates_dropped_total += n_dup
            sub.to_excel(writer, sheet_name=sheet, index=False)
            written.append(
                {"sheet": sheet, "rows": int(len(sub)), "duplicates_dropped": int(n_dup)}
            )

    counts = df[variety_col].value_counts()
    skipped = [(v, int(n)) for v, n in counts.items() if n < min_rows]

    summary["varieties_written"] = len(written)
    summary["varieties_skipped"] = len(skipped)
    summary["rows_written"] = sum(r["rows"] for r in written)
    summary["duplicates_dropped"] = int(duplicates_dropped_total)
    summary["sheets"] = written
    summary["skipped"] = skipped
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split del Excel acumulado en una hoja por VARIEDAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Ruta del Excel acumulado")
    p.add_argument("--output", required=True, help="Ruta del Excel resultante")
    p.add_argument(
        "--min-rows", type=int, default=100, help="Filas minimas por variedad para incluirla"
    )
    p.add_argument(
        "--source-sheet", default="acumulado", help="Hoja de entrada en el Excel acumulado"
    )
    p.add_argument("--variety-col", default="VARIEDAD", help="Columna que identifica la variedad")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = split_workbook(
        input_path=args.input,
        output_path=args.output,
        min_rows=args.min_rows,
        source_sheet=args.source_sheet,
        variety_col=args.variety_col,
    )

    print(
        f"OK | escritas {summary['varieties_written']} hojas "
        f"({summary['rows_written']} filas) en {summary['output']} | "
        f"descartadas {summary['varieties_skipped']} variedades por <{summary['min_rows']} filas | "
        f"duplicados estructurales eliminados: {summary['duplicates_dropped']}"
    )
    if summary["sheets"]:
        print("Variedades escritas:")
        for s in summary["sheets"]:
            print(f"  - {s['sheet']:32s}  {s['rows']:>6d} filas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
