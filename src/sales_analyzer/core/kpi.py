from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass
class KpiPack:
    ca_total: float
    nb_lignes: int
    nb_jours: int
    nb_produits: Optional[int]
    nb_clients: Optional[int]
    ca_moyen_par_ligne: float


def ensure_columns(std_df: pd.DataFrame) -> None:
    required = {"Date", "Montant"}
    missing = required - set(std_df.columns)
    if missing:
        raise ValueError(f"Dataset standardisé invalide. Colonnes manquantes: {missing}")


def filter_std_df(
    std_df: pd.DataFrame,
    date_min: Optional[pd.Timestamp],
    date_max: Optional[pd.Timestamp],
    produit: Optional[str],
    client: Optional[str],
    canal: Optional[str],
) -> pd.DataFrame:
    df = std_df.copy()

    if date_min is not None:
        df = df[df["Date"] >= date_min]
    if date_max is not None:
        df = df[df["Date"] <= date_max]

    if produit and "Produit" in df.columns:
        df = df[df["Produit"] == produit]
    if client and "Client" in df.columns:
        df = df[df["Client"] == client]
    if canal and "Canal" in df.columns:
        df = df[df["Canal"] == canal]

    return df


def compute_kpis(df: pd.DataFrame) -> KpiPack:
    # sécuriser types
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Montant"] = pd.to_numeric(df["Montant"], errors="coerce")

    ca_total = float(df["Montant"].sum(skipna=True))
    nb_lignes = int(len(df))
    nb_jours = int(df["Date"].dropna().dt.date.nunique())

    nb_produits = int(df["Produit"].nunique()) if "Produit" in df.columns else None
    nb_clients = int(df["Client"].nunique()) if "Client" in df.columns else None

    ca_moyen_par_ligne = float(df["Montant"].mean(skipna=True)) if nb_lignes > 0 else 0.0

    return KpiPack(
        ca_total=ca_total,
        nb_lignes=nb_lignes,
        nb_jours=nb_jours,
        nb_produits=nb_produits,
        nb_clients=nb_clients,
        ca_moyen_par_ligne=ca_moyen_par_ligne,
    )


def ca_timeseries(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """
    freq:
      - 'D' jour
      - 'W' semaine
      - 'M' mois
    """
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Montant"] = pd.to_numeric(out["Montant"], errors="coerce")

    out = out.dropna(subset=["Date"])
    if out.empty:
        return pd.DataFrame(columns=["Periode", "CA"])

    # normaliser période
    if freq == "D":
        out["Periode"] = out["Date"].dt.to_period("D").dt.to_timestamp()
    elif freq == "W":
        out["Periode"] = out["Date"].dt.to_period("W").dt.start_time
    else:  # "M"
        out["Periode"] = out["Date"].dt.to_period("M").dt.to_timestamp()

    ts = out.groupby("Periode", as_index=False)["Montant"].sum()
    ts = ts.rename(columns={"Montant": "CA"}).sort_values("Periode")
    return ts


def top_dimension(df: pd.DataFrame, dim: str, top_n: int = 10) -> pd.DataFrame:
    if dim not in df.columns:
        return pd.DataFrame(columns=[dim, "CA"])

    out = df.copy()
    out["Montant"] = pd.to_numeric(out["Montant"], errors="coerce")
    g = out.groupby(dim, as_index=False)["Montant"].sum().rename(columns={"Montant": "CA"})
    g = g.sort_values("CA", ascending=False).head(top_n)
    return g


def pareto_dimension(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    """
    Retourne une table Pareto triée décroissante + cumul.
    """
    if dim not in df.columns:
        return pd.DataFrame(columns=[dim, "CA", "Part_CA", "Cumul_CA"])

    out = df.copy()
    out["Montant"] = pd.to_numeric(out["Montant"], errors="coerce")
    g = out.groupby(dim, as_index=False)["Montant"].sum().rename(columns={"Montant": "CA"})
    g = g.sort_values("CA", ascending=False).reset_index(drop=True)

    total = g["CA"].sum()
    if total == 0:
        g["Part_CA"] = 0.0
        g["Cumul_CA"] = 0.0
        return g

    g["Part_CA"] = g["CA"] / total
    g["Cumul_CA"] = g["Part_CA"].cumsum()
    return g


def pareto_80_20_stats(pareto_df: pd.DataFrame) -> Tuple[int, float]:
    """
    Retourne:
      - nb_items nécessaires pour atteindre 80%
      - part des items (nb_items / total_items)
    """
    if pareto_df.empty:
        return 0, 0.0
    idx = (pareto_df["Cumul_CA"] >= 0.8).idxmax()
    nb_items = int(idx + 1)
    part_items = nb_items / len(pareto_df) if len(pareto_df) else 0.0
    return nb_items, float(part_items)
