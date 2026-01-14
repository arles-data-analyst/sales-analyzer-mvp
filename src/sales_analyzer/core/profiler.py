from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
    is_object_dtype,
)


@dataclass
class ProfileSummary:
    n_rows: int
    n_cols: int
    duplicate_rows: int
    duplicate_rows_pct: float
    missing_cells: int
    missing_cells_pct: float
    columns_with_missing: int
    columns_with_missing_pct: float
    dtypes: Dict[str, int]
    suspected_numeric: List[str]
    suspected_date: List[str]
    high_cardinality: List[str]
    alerts: List[str]


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convertit une série (souvent string/object) en numérique de manière tolérante.
    """
    if is_numeric_dtype(s):
        return s
    # Remplace virgule décimale -> point (cas fréquent FR)
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _is_probably_numeric(s: pd.Series) -> bool:
    """
    Heuristique: numérique natif OU conversion numérique réussie sur un échantillon.
    """
    if is_numeric_dtype(s):
        return True

    if is_string_dtype(s) or is_object_dtype(s):
        sample = s.dropna().astype(str).head(200)
        if sample.empty:
            return False
        coerced = pd.to_numeric(sample.str.replace(",", ".", regex=False), errors="coerce")
        return coerced.notna().mean() > 0.8

    return False


def _is_probably_date(s: pd.Series) -> bool:
    """
    Heuristique: datetime natif OU conversion datetime réussie sur un échantillon.
    Compatible avec dtype Pandas 'string[python]'.
    """
    if is_datetime64_any_dtype(s):
        return True

    if is_string_dtype(s) or is_object_dtype(s):
        sample = s.dropna().astype(str).head(200)
        if sample.empty:
            return False
        coerced = pd.to_datetime(sample, errors="coerce", dayfirst=True)
        return coerced.notna().mean() > 0.8

    return False


def build_profile_summary(df: pd.DataFrame) -> ProfileSummary:
    n_rows, n_cols = df.shape

    dup_rows = int(df.duplicated().sum())
    dup_rows_pct = (dup_rows / n_rows) * 100 if n_rows else 0.0

    missing_cells = int(df.isna().sum().sum())
    total_cells = n_rows * n_cols if n_rows and n_cols else 0
    missing_cells_pct = (missing_cells / total_cells) * 100 if total_cells else 0.0

    missing_by_col = df.isna().mean().sort_values(ascending=False)
    cols_with_missing = int((missing_by_col > 0).sum())
    cols_with_missing_pct = (cols_with_missing / n_cols) * 100 if n_cols else 0.0

    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()

    suspected_numeric = [c for c in df.columns if _is_probably_numeric(df[c])]
    suspected_date = [c for c in df.columns if _is_probably_date(df[c])]

    high_cardinality: List[str] = []
    for c in df.columns:
        nunique = df[c].nunique(dropna=True)
        if n_rows >= 50 and nunique > max(50, int(0.5 * n_rows)):
            high_cardinality.append(c)

    alerts: List[str] = []
    if dup_rows_pct >= 1.0:
        alerts.append(f"Doublons lignes élevés : {dup_rows_pct:.1f}% ({dup_rows} lignes).")
    if missing_cells_pct >= 5.0:
        alerts.append(f"Beaucoup de valeurs manquantes : {missing_cells_pct:.1f}% des cellules.")
    if cols_with_missing_pct >= 30.0:
        alerts.append(f"Nombre important de colonnes incomplètes : {cols_with_missing} / {n_cols}.")

    for c, rate in missing_by_col.head(5).items():
        if rate >= 0.2:
            alerts.append(f"Colonne '{c}' très incomplète : {rate*100:.1f}% de valeurs manquantes.")

    return ProfileSummary(
        n_rows=n_rows,
        n_cols=n_cols,
        duplicate_rows=dup_rows,
        duplicate_rows_pct=dup_rows_pct,
        missing_cells=missing_cells,
        missing_cells_pct=missing_cells_pct,
        columns_with_missing=cols_with_missing,
        columns_with_missing_pct=cols_with_missing_pct,
        dtypes=dtype_counts,
        suspected_numeric=suspected_numeric,
        suspected_date=suspected_date,
        high_cardinality=high_cardinality,
        alerts=alerts,
    )


def quality_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "Colonne": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Taux_null_%": (df.isna().mean() * 100).round(2).values,
            "Nb_null": df.isna().sum().values,
            "N_unique": df.nunique(dropna=True).values,
            "Exemples": [", ".join(df[c].dropna().astype(str).head(3).tolist()) for c in df.columns],
        }
    )
    return out.sort_values("Taux_null_%", ascending=False).reset_index(drop=True)


def numeric_describe(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return pd.DataFrame()

    tmp = df[cols].copy()
    for c in cols:
        tmp[c] = _coerce_numeric_series(tmp[c])

    desc = tmp.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    desc = desc.rename(
        columns={
            "count": "count_non_null",
            "mean": "mean",
            "std": "std",
            "min": "min",
            "5%": "p05",
            "25%": "p25",
            "50%": "median",
            "75%": "p75",
            "95%": "p95",
            "max": "max",
        }
    )
    return desc.reset_index(names="Colonne")
