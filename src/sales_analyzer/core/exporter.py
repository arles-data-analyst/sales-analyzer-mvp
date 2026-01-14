from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class ExportPack:
    filename: str
    bytes_data: bytes
    mime: str


def to_excel_bytes(sheets: Dict[str, pd.DataFrame], filename: str = "export.xlsx") -> ExportPack:
    """
    Exporte plusieurs DataFrames en un fichier Excel (bytes).
    Nécessite openpyxl installé.
    """
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = str(name)[:31]  # limite Excel
            df.to_excel(writer, index=False, sheet_name=safe_name)
    return ExportPack(filename=filename, bytes_data=bio.getvalue(), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def to_csv_bytes(df: pd.DataFrame, filename: str = "export.csv") -> ExportPack:
    b = df.to_csv(index=False).encode("utf-8")
    return ExportPack(filename=filename, bytes_data=b, mime="text/csv")
