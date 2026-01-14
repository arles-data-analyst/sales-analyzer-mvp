from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .kpi import ca_timeseries, pareto_dimension, pareto_80_20_stats


def generate_insights(df: pd.DataFrame) -> List[str]:
    """
    Insights simples, fiables, actionnables.
    (Pas d'IA ici : règles déterministes = confiance.)
    """
    insights: List[str] = []

    if df.empty:
        return ["Aucune donnée après filtres : élargis la période ou supprime des filtres."]

    # CA total
    ca_total = float(pd.to_numeric(df["Montant"], errors="coerce").sum(skipna=True))
    insights.append(f"Chiffre d’affaires filtré : {ca_total:,.2f}".replace(",", " ").replace(".", ","))

    # Trend (dernier vs précédent) sur série mensuelle
    ts = ca_timeseries(df, freq="M")
    if len(ts) >= 2:
        last = float(ts.iloc[-1]["CA"])
        prev = float(ts.iloc[-2]["CA"])
        if prev != 0:
            pct = (last - prev) / prev * 100
            direction = "hausse" if pct >= 0 else "baisse"
            insights.append(f"Évolution mensuelle : {direction} de {abs(pct):.1f}% sur la dernière période.")
        else:
            insights.append("Évolution mensuelle : période précédente à 0, variation non calculable.")
    else:
        insights.append("Évolution mensuelle : historique insuffisant (moins de 2 mois).")

    # Pareto produit si dispo
    if "Produit" in df.columns:
        p = pareto_dimension(df, "Produit")
        nb, part_items = pareto_80_20_stats(p)
        if len(p) > 0:
            insights.append(
                f"Concentration produits : {nb} produit(s) génèrent ~80% du CA "
                f"({part_items*100:.1f}% des produits)."
            )

    # Mix canal si dispo
    if "Canal" in df.columns:
        mix = df.groupby("Canal", as_index=False)["Montant"].sum()
        mix["Montant"] = pd.to_numeric(mix["Montant"], errors="coerce")
        total = float(mix["Montant"].sum())
        if total > 0 and len(mix) >= 2:
            mix = mix.sort_values("Montant", ascending=False)
            top_canal = mix.iloc[0]["Canal"]
            share = float(mix.iloc[0]["Montant"] / total * 100)
            insights.append(f"Mix canal : '{top_canal}' représente {share:.1f}% du CA.")

    # Qualité: dates nulles / montants nuls
    dates_nulles = int(pd.to_datetime(df["Date"], errors="coerce").isna().sum())
    montants_nulls = int(pd.to_numeric(df["Montant"], errors="coerce").isna().sum())
    if dates_nulles > 0:
        insights.append(f"Qualité : {dates_nulles} ligne(s) avec Date invalide (à corriger/filtrer).")
    if montants_nulls > 0:
        insights.append(f"Qualité : {montants_nulls} ligne(s) avec Montant invalide (à corriger/filtrer).")

    return insights
