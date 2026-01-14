from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


@dataclass
class Mapping:
    date_col: str
    amount_col: str
    product_col: Optional[str] = None
    client_col: Optional[str] = None
    channel_col: Optional[str] = None


def _norm(s: str) -> str:
    return str(s).strip().lower()


def suggest_columns(
    df: pd.DataFrame,
    suspected_numeric: List[str],
    suspected_date: List[str],
) -> Dict[str, List[str]]:
    """
    Retourne des suggestions ordonnées pour chaque rôle.
    """
    cols = list(df.columns)

    # Date candidates: suspected_date first, then any datetime dtype, then name heuristics
    date_candidates = []
    for c in cols:
        if c in suspected_date:
            date_candidates.append(c)
    for c in cols:
        if c not in date_candidates and is_datetime64_any_dtype(df[c]):
            date_candidates.append(c)
    for c in cols:
        cl = _norm(c)
        if c not in date_candidates and any(k in cl for k in ["date", "jour", "day", "dt"]):
            date_candidates.append(c)

    # Amount candidates: name heuristics first then numeric columns
    amount_candidates = []
    for c in cols:
        cl = _norm(c)
        if any(k in cl for k in ["montant", "amount", "total", "revenue", "ca", "chiffre", "sales"]):
            amount_candidates.append(c)
    for c in suspected_numeric:
        if c in cols and c not in amount_candidates:
            amount_candidates.append(c)
    for c in cols:
        if c not in amount_candidates and is_numeric_dtype(df[c]):
            amount_candidates.append(c)

    # Product candidates: name heuristics
    product_candidates = []
    for c in cols:
        cl = _norm(c)
        if any(k in cl for k in ["produit", "product", "sku", "item", "article", "libelle"]):
            product_candidates.append(c)

    # Client candidates: name heuristics
    client_candidates = []
    for c in cols:
        cl = _norm(c)
        if any(k in cl for k in ["client", "customer", "buyer", "account"]):
            client_candidates.append(c)

    # Channel candidates: name heuristics
    channel_candidates = []
    for c in cols:
        cl = _norm(c)
        if any(k in cl for k in ["canal", "channel", "source", "store", "magasin", "shop"]):
            channel_candidates.append(c)

    # Ensure uniqueness + fallback with remaining columns
    def _unique_keep_order(lst: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return {
        "date": _unique_keep_order(date_candidates),
        "amount": _unique_keep_order(amount_candidates),
        "product": _unique_keep_order(product_candidates),
        "client": _unique_keep_order(client_candidates),
        "channel": _unique_keep_order(channel_candidates),
        "all": cols,
    }


def validate_mapping(df: pd.DataFrame, mapping: Mapping) -> Optional[str]:
    """
    Retourne None si OK, sinon un message d'erreur.
    """
    if mapping.date_col not in df.columns:
        return f"Colonne date introuvable : {mapping.date_col}"
    if mapping.amount_col not in df.columns:
        return f"Colonne montant introuvable : {mapping.amount_col}"
    return None


def standardize_dataset(df: pd.DataFrame, mapping: Mapping) -> pd.DataFrame:
    """
    Retourne un dataset standardisé avec colonnes :
    - Date (datetime si possible)
    - Montant (float)
    - Produit, Client, Canal (optionnels)
    """
    out = df.copy()

    rename_map = {
        mapping.date_col: "Date",
        mapping.amount_col: "Montant",
    }
    if mapping.product_col:
        rename_map[mapping.product_col] = "Produit"
    if mapping.client_col:
        rename_map[mapping.client_col] = "Client"
    if mapping.channel_col:
        rename_map[mapping.channel_col] = "Canal"

    out = out.rename(columns=rename_map)

    # Coerce Date & Montant
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=True)
    out["Montant"] = pd.to_numeric(out["Montant"].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    return out
