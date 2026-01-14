from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class CleanOptions:
    remove_duplicates: bool = True

    # Missing values
    drop_rows_with_any_null: bool = False
    fill_numeric_nulls_with_median: bool = True
    fill_text_nulls_with_placeholder: bool = False
    text_placeholder: str = "Inconnu"

    # Type coercion (based on profiling suggestions)
    coerce_numeric: bool = True
    coerce_dates: bool = True
    date_dayfirst: bool = True

    # Business sanity checks (sales)
    remove_negative_amounts: bool = True
    amount_column_name: Optional[str] = None  # if None, will try to auto-detect


@dataclass
class ImpactReport:
    rows_before: int
    rows_after: int
    rows_removed_duplicates: int
    rows_removed_nulls: int
    rows_removed_negative_amounts: int
    nulls_filled_numeric: int
    nulls_filled_text: int
    coerced_numeric_cols: List[str]
    coerced_date_cols: List[str]
    notes: List[str]


def _coerce_numeric_col(s: pd.Series) -> pd.Series:
    if is_numeric_dtype(s):
        return s
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _coerce_date_col(s: pd.Series, dayfirst: bool) -> pd.Series:
    return pd.to_datetime(s.astype(str), errors="coerce", dayfirst=dayfirst)


def _auto_detect_amount_col(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristique simple: cherche une colonne qui ressemble à un montant (montant, total, amount, revenue, ca...)
    """
    candidates = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if any(k in cl for k in ["montant", "amount", "total", "revenue", "ca", "chiffre"]):
            candidates.append(c)
    return candidates[0] if candidates else None


def simulate_impact(
    df: pd.DataFrame,
    options: CleanOptions,
    suspected_numeric_cols: List[str],
    suspected_date_cols: List[str],
) -> ImpactReport:
    """
    Simule l’impact sans modifier df : on applique sur une copie et on compare.
    """
    cleaned_df, report = apply_cleaning(df, options, suspected_numeric_cols, suspected_date_cols)
    return report


def apply_cleaning(
    df: pd.DataFrame,
    options: CleanOptions,
    suspected_numeric_cols: List[str],
    suspected_date_cols: List[str],
) -> Tuple[pd.DataFrame, ImpactReport]:
    notes: List[str] = []
    work = df.copy()

    rows_before = len(work)

    # 1) Remove duplicate rows
    rows_removed_duplicates = 0
    if options.remove_duplicates:
        dup_count = int(work.duplicated().sum())
        work = work.drop_duplicates()
        rows_removed_duplicates = dup_count

    # 2) Coerce numeric
    coerced_numeric_cols: List[str] = []
    if options.coerce_numeric and suspected_numeric_cols:
        for c in suspected_numeric_cols:
            if c in work.columns:
                before_dtype = str(work[c].dtype)
                work[c] = _coerce_numeric_col(work[c])
                after_dtype = str(work[c].dtype)
                if before_dtype != after_dtype:
                    coerced_numeric_cols.append(c)

    # 3) Coerce dates
    coerced_date_cols: List[str] = []
    if options.coerce_dates and suspected_date_cols:
        for c in suspected_date_cols:
            if c in work.columns:
                before_dtype = str(work[c].dtype)
                work[c] = _coerce_date_col(work[c], dayfirst=options.date_dayfirst)
                after_dtype = str(work[c].dtype)
                if before_dtype != after_dtype:
                    coerced_date_cols.append(c)

    # 4) Missing values handling
    rows_removed_nulls = 0
    nulls_filled_numeric = 0
    nulls_filled_text = 0

    if options.drop_rows_with_any_null:
        before = len(work)
        work = work.dropna(axis=0, how="any")
        rows_removed_nulls = before - len(work)
    else:
        # fill numeric nulls with median (for numeric suspected cols)
        if options.fill_numeric_nulls_with_median and suspected_numeric_cols:
            for c in suspected_numeric_cols:
                if c in work.columns:
                    # ensure numeric if possible
                    work[c] = _coerce_numeric_col(work[c])
                    n_null = int(work[c].isna().sum())
                    if n_null > 0:
                        med = work[c].median(skipna=True)
                        # if all null -> keep as is
                        if pd.notna(med):
                            work[c] = work[c].fillna(med)
                            nulls_filled_numeric += n_null

        # fill text nulls with placeholder (for object/string columns)
        if options.fill_text_nulls_with_placeholder:
            for c in work.columns:
                if c in suspected_numeric_cols or c in suspected_date_cols:
                    continue
                if str(work[c].dtype).startswith(("string", "object")):
                    n_null = int(work[c].isna().sum())
                    if n_null > 0:
                        work[c] = work[c].fillna(options.text_placeholder)
                        nulls_filled_text += n_null

    # 5) Remove negative amounts (business sanity)
    rows_removed_negative_amounts = 0
    if options.remove_negative_amounts:
        amount_col = options.amount_column_name or _auto_detect_amount_col(work)
        if amount_col is None:
            notes.append("Colonne Montant non détectée : règle 'montants négatifs' ignorée.")
        else:
            # try coerce numeric
            work[amount_col] = _coerce_numeric_col(work[amount_col])
            before = len(work)
            work = work.loc[~(work[amount_col].notna() & (work[amount_col] < 0))].copy()
            rows_removed_negative_amounts = before - len(work)
            notes.append(f"Colonne Montant utilisée : '{amount_col}'.")

    rows_after = len(work)

    report = ImpactReport(
        rows_before=rows_before,
        rows_after=rows_after,
        rows_removed_duplicates=rows_removed_duplicates,
        rows_removed_nulls=rows_removed_nulls,
        rows_removed_negative_amounts=rows_removed_negative_amounts,
        nulls_filled_numeric=nulls_filled_numeric,
        nulls_filled_text=nulls_filled_text,
        coerced_numeric_cols=coerced_numeric_cols,
        coerced_date_cols=coerced_date_cols,
        notes=notes,
    )
    return work, report
