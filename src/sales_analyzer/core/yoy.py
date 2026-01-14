from __future__ import annotations

from typing import Tuple
import pandas as pd


def yoy_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare N et N-1 sur la base de l'année max présente.
    """
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Annee"] = d["Date"].dt.year

    year_n = int(d["Annee"].max())
    year_n1 = year_n - 1

    df_n = d[d["Annee"] == year_n]
    df_n1 = d[d["Annee"] == year_n1]

    return df_n, df_n1


def yoy_kpis(df_n: pd.DataFrame, df_n1: pd.DataFrame) -> dict:
    def _sum(df: pd.DataFrame) -> float:
        return float(pd.to_numeric(df["Montant"], errors="coerce").sum())

    ca_n = _sum(df_n)
    ca_n1 = _sum(df_n1)

    delta = ca_n - ca_n1
    pct = (delta / ca_n1 * 100) if ca_n1 != 0 else None

    return {
        "CA_N": ca_n,
        "CA_N1": ca_n1,
        "Delta": delta,
        "Pct": pct,
    }


def yoy_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    CA mensuel par année pour comparaison YoY.
    """
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Annee"] = d["Date"].dt.year
    d["Mois"] = d["Date"].dt.month

    g = (
        d.groupby(["Annee", "Mois"], as_index=False)["Montant"]
        .sum()
        .rename(columns={"Montant": "CA"})
    )
    return g.sort_values(["Annee", "Mois"])
