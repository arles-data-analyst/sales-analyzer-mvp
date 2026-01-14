from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


@dataclass
class PdfPack:
    filename: str
    bytes_data: bytes
    mime: str = "application/pdf"


def _fmt_money(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ").replace(".", ",")


def _fmt_int(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def build_comex_pdf(
    title: str,
    period_label: str,
    kpi: dict,
    insights: List[str],
    ts_mensuel: pd.DataFrame,
    top_produits: Optional[pd.DataFrame] = None,
    top_clients: Optional[pd.DataFrame] = None,
    filename: str = "comex_one_pager.pdf",
) -> PdfPack:
    """
    PDF compact A4:
      - Header (titre + période)
      - KPIs (ligne)
      - Mini tableau CA mensuel (6 derniers)
      - Top Produits / Clients (5)
      - Insights (bullets)
    """
    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=A4)
    w, h = A4

    # Marges
    left = 18 * mm
    right = w - 18 * mm
    y = h - 18 * mm

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, y, title)
    y -= 7 * mm

    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Période : {period_label}")
    y -= 10 * mm

    # KPIs line
    c.setFont("Helvetica-Bold", 11)
    kpi_items = [
        ("CA", _fmt_money(float(kpi.get("CA", 0.0)))),
        ("Transactions", _fmt_int(int(kpi.get("Transactions", 0)))),
        ("Panier moyen", _fmt_money(float(kpi.get("Panier_moyen", 0.0)))),
        ("Produits", str(kpi.get("Nb_produits", "—"))),
        ("Clients", str(kpi.get("Nb_clients", "—"))),
    ]

    x = left
    for label, val in kpi_items:
        c.drawString(x, y, f"{label}: {val}")
        x += 38 * mm  # espacement
    y -= 10 * mm

    # Section CA mensuel (mini-table)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "CA mensuel (6 dernières périodes)")
    y -= 6 * mm

    c.setFont("Helvetica", 10)
    ts = ts_mensuel.copy()
    if not ts.empty:
        ts = ts.tail(6)
        # Expect columns: Periode, CA
        for _, row in ts.iterrows():
            per = pd.to_datetime(row["Periode"], errors="coerce")
            per_label = per.strftime("%Y-%m") if pd.notna(per) else str(row["Periode"])
            ca = _fmt_money(float(row["CA"]))
            c.drawString(left, y, f"{per_label}")
            c.drawRightString(right, y, f"{ca}")
            y -= 5 * mm
    else:
        c.drawString(left, y, "Données insuffisantes.")
        y -= 5 * mm

    y -= 4 * mm

    # Top contributeurs
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Top contributeurs")
    y -= 7 * mm

    col_mid = left + (right - left) / 2

    def _draw_top_table(x0: float, y0: float, title_: str, df_: Optional[pd.DataFrame], dim_col: str) -> float:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x0, y0, title_)
        y1 = y0 - 5 * mm
        c.setFont("Helvetica", 9)

        if df_ is None or df_.empty:
            c.drawString(x0, y1, "Indisponible")
            return y1 - 5 * mm

        tmp = df_.head(5)
        for _, r in tmp.iterrows():
            name = str(r[dim_col])
            ca = _fmt_money(float(r["CA"]))
            c.drawString(x0, y1, name[:28])
            c.drawRightString(x0 + 80 * mm, y1, ca)
            y1 -= 5 * mm
        return y1 - 2 * mm

    y_left_end = _draw_top_table(left, y, "Top Produits (CA)", top_produits, "Produit") if top_produits is not None else (y - 12 * mm)
    y_right_end = _draw_top_table(col_mid, y, "Top Clients (CA)", top_clients, "Client") if top_clients is not None else (y - 12 * mm)

    y = min(y_left_end, y_right_end) - 4 * mm

    # Insights
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Points clés (insights)")
    y -= 7 * mm

    c.setFont("Helvetica", 10)
    max_lines = 7
    for line in insights[:max_lines]:
        c.drawString(left, y, f"• {line[:120]}")
        y -= 5 * mm

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(left, 12 * mm, "Généré par Sales Analyzer — Synthèse COMEX")

    c.showPage()
    c.save()

    return PdfPack(filename=filename, bytes_data=bio.getvalue())
