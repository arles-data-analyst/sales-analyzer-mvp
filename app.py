from __future__ import annotations

import pandas as pd

import streamlit as st

import plotly.express as px

from src.sales_analyzer.core.kpi import (
    ensure_columns,
    filter_std_df,
    compute_kpis,
    ca_timeseries,
    top_dimension,
    pareto_dimension,
    pareto_80_20_stats,
)
from src.sales_analyzer.core.insights import generate_insights

from src.sales_analyzer.core.loader import load_csv, basic_validation
from src.sales_analyzer.core.profiler import build_profile_summary, quality_table, numeric_describe
from src.sales_analyzer.core.cleaner import CleanOptions, simulate_impact, apply_cleaning
from src.sales_analyzer.core.mapper import Mapping, suggest_columns, validate_mapping, standardize_dataset
from src.sales_analyzer.core.exporter import to_excel_bytes, to_csv_bytes
from src.sales_analyzer.core.pdf_report import build_comex_pdf
from src.sales_analyzer.core.kpi import ca_timeseries
from src.sales_analyzer.core.yoy import yoy_split, yoy_kpis, yoy_timeseries






# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Sales Analyzer (MVP)", layout="wide")

st.title("Sales Analyzer — MVP (Ventes CSV)")
st.caption("Écrans 1 à 3 — Ingestion + Profiling (Audit) + Nettoyage assisté")


def _intfmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _reset_state_for_new_file() -> None:
    for k in ["raw_df", "sep", "encoding", "upload_filename", "data_loaded", "cleaned_df", "clean_report"]:
        st.session_state.pop(k, None)


# Init flags
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False


# -----------------------------
# UI — Écran 1: Upload (stateful)
# -----------------------------
st.subheader("Écran 1 — Charger un fichier CSV")

colA, colB = st.columns([1, 1])

with colA:
    with st.form("upload_form", clear_on_submit=False):
        uploaded = st.file_uploader("Choisis un fichier", type=["csv"], accept_multiple_files=False)

        st.markdown("**Recommandations**")
        st.markdown("- CSV exporté depuis Excel / ERP")
        st.markdown("- Une première ligne d’en-têtes")
        st.markdown("- Éviter les cellules fusionnées (Excel)")

        submitted = st.form_submit_button("Charger et valider", type="primary", use_container_width=True)

    # Bouton de reset (si déjà chargé)
    if st.session_state["data_loaded"]:
        if st.button("Charger un autre fichier", use_container_width=True):
            _reset_state_for_new_file()
            st.rerun()

with colB:
    st.subheader("Résultat de lecture & aperçu")

    # Si on vient de soumettre, on charge et on persiste
    if submitted:
        if uploaded is None:
            st.error("Aucun fichier sélectionné.")
        else:
            result = load_csv(uploaded)
            if result.error:
                st.error(result.error)
            else:
                err = basic_validation(result.df)
                if err:
                    st.error(err)
                else:
                    st.session_state["raw_df"] = result.df.copy()
                    st.session_state["sep"] = result.sep
                    st.session_state["encoding"] = result.encoding
                    st.session_state["upload_filename"] = uploaded.name
                    st.session_state["data_loaded"] = True

                    # Reset downstream states each time a file is loaded
                    st.session_state.pop("cleaned_df", None)
                    st.session_state.pop("clean_report", None)

                    st.success("Fichier chargé et validé.")

    # Affichage si data déjà chargée (persistent)
    if st.session_state["data_loaded"]:
        df0 = st.session_state["raw_df"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Lignes", _intfmt(df0.shape[0]))
        c2.metric("Colonnes", _intfmt(df0.shape[1]))
        c3.metric("Encodage / Séparateur", f"{st.session_state['encoding']} / {repr(st.session_state['sep'])}")

        st.markdown(f"**Fichier :** `{st.session_state.get('upload_filename','')}`")
        st.markdown("**Aperçu (10 premières lignes)**")
        st.dataframe(df0.head(10), use_container_width=True)
    else:
        st.info("Charge un fichier puis clique sur « Charger et valider ».")


# Si pas de données, on s’arrête ici (mais sans casser les reruns une fois chargé)
if not st.session_state["data_loaded"]:
    st.stop()


# -----------------------------
# UI — Écran 2: Profiling (Audit)
# -----------------------------
st.divider()
st.subheader("Écran 2 — Profiling automatique (Audit Data Analyst)")

df = st.session_state["raw_df"]
summary = build_profile_summary(df)

a1, a2, a3, a4 = st.columns(4)
a1.metric("Doublons (lignes)", _intfmt(summary.duplicate_rows))
a2.metric("Doublons (%)", f"{summary.duplicate_rows_pct:.2f}%")
a3.metric("Cellules manquantes", _intfmt(summary.missing_cells))
a4.metric("Manquants (%)", f"{summary.missing_cells_pct:.2f}%")

b1, b2, b3 = st.columns(3)
b1.metric("Colonnes avec null", f"{summary.columns_with_missing} / {summary.n_cols}")
b2.metric("Colonnes numériques (suspectées)", f"{len(summary.suspected_numeric)}")
b3.metric("Colonnes dates (suspectées)", f"{len(summary.suspected_date)}")

if summary.alerts:
    st.warning("Alertes qualité détectées :")
    for al in summary.alerts:
        st.write(f"- {al}")
else:
    st.success("Aucune alerte majeure détectée sur la qualité des données.")

st.markdown("### Qualité par colonne")
st.dataframe(quality_table(df), use_container_width=True)

st.markdown("### Statistiques descriptives (numériques suspectées)")
nd = numeric_describe(df, summary.suspected_numeric)
if nd.empty:
    st.info("Aucune colonne numérique détectée.")
else:
    st.dataframe(nd, use_container_width=True)

with st.expander("Détails — Répartition des types (JSON)"):
    st.json(summary.dtypes)


# -----------------------------
# UI — Écran 3: Nettoyage assisté (simulation + application)
# -----------------------------
st.divider()
st.subheader("Écran 3 — Nettoyage assisté (contrôlé)")
st.caption("Principe : tu sélectionnes les règles, on simule l’impact, puis tu appliques.")

c_left, c_right = st.columns([1, 1])

with c_left:
    st.markdown("### Règles de nettoyage")

    remove_duplicates = st.checkbox("Supprimer les doublons (lignes identiques)", value=True)

    st.markdown("**Valeurs manquantes**")
    drop_rows_with_any_null = st.checkbox("Supprimer les lignes avec au moins 1 valeur manquante", value=False)
    fill_numeric_nulls_with_median = st.checkbox("Imputer les valeurs manquantes numériques par la médiane", value=True)
    fill_text_nulls_with_placeholder = st.checkbox("Imputer les valeurs manquantes texte par un libellé", value=False)

    text_placeholder = "Inconnu"
    if fill_text_nulls_with_placeholder:
        text_placeholder = st.text_input("Libellé de remplacement (texte)", value="Inconnu")

    st.markdown("**Typage**")
    coerce_numeric = st.checkbox("Forcer les colonnes numériques détectées (conversion)", value=True)
    coerce_dates = st.checkbox("Forcer les colonnes dates détectées (conversion)", value=True)

    st.markdown("**Règles métier (ventes)**")
    remove_negative_amounts = st.checkbox("Supprimer les lignes où le Montant est négatif", value=True)
    amount_column_name = st.text_input("Nom colonne Montant (optionnel)", value="").strip() or None

    options = CleanOptions(
        remove_duplicates=remove_duplicates,
        drop_rows_with_any_null=drop_rows_with_any_null,
        fill_numeric_nulls_with_median=fill_numeric_nulls_with_median,
        fill_text_nulls_with_placeholder=fill_text_nulls_with_placeholder,
        text_placeholder=text_placeholder,
        coerce_numeric=coerce_numeric,
        coerce_dates=coerce_dates,
        remove_negative_amounts=remove_negative_amounts,
        amount_column_name=amount_column_name,
    )

    simulate_btn = st.button("Simuler l’impact", use_container_width=True)
    apply_btn = st.button("Appliquer le nettoyage", type="primary", use_container_width=True)

with c_right:
    st.markdown("### Impact & comparaison")

    # Simulation (persistée)
    if simulate_btn:
        report = simulate_impact(df, options, summary.suspected_numeric, summary.suspected_date)
        st.session_state["clean_report"] = report

    report = st.session_state.get("clean_report")
    if report is None:
        st.info("Clique sur « Simuler l’impact » pour estimer les changements.")
    else:
        r = report
        m1, m2, m3 = st.columns(3)
        m1.metric("Lignes avant", _intfmt(r.rows_before))
        m2.metric("Lignes après (estimé)", _intfmt(r.rows_after))
        m3.metric("Lignes supprimées", _intfmt(r.rows_before - r.rows_after))

        st.markdown("**Détail de l’impact**")
        st.write(f"- Doublons supprimés : {_intfmt(r.rows_removed_duplicates)}")
        st.write(f"- Lignes supprimées (nulls) : {_intfmt(r.rows_removed_nulls)}")
        st.write(f"- Lignes supprimées (montant négatif) : {_intfmt(r.rows_removed_negative_amounts)}")
        st.write(f"- Nulls imputés (numériques) : {_intfmt(r.nulls_filled_numeric)}")
        st.write(f"- Nulls imputés (texte) : {_intfmt(r.nulls_filled_text)}")

        if r.coerced_numeric_cols:
            st.write(f"- Colonnes converties en numérique : {', '.join(r.coerced_numeric_cols)}")
        if r.coerced_date_cols:
            st.write(f"- Colonnes converties en date : {', '.join(r.coerced_date_cols)}")
        if r.notes:
            st.markdown("**Notes**")
            for n in r.notes:
                st.write(f"- {n}")

    # Application (persistée)
    if apply_btn:
        cleaned_df, r2 = apply_cleaning(df, options, summary.suspected_numeric, summary.suspected_date)
        st.session_state["cleaned_df"] = cleaned_df
        st.session_state["clean_report"] = r2

        st.success("Nettoyage appliqué. Dataset nettoyé stocké pour les écrans suivants.")
        st.markdown("### Aperçu après nettoyage (10 premières lignes)")
        st.dataframe(cleaned_df.head(10), use_container_width=True)

        st.markdown("### Comparaison rapide (avant / après)")
        k1, k2, k3 = st.columns(3)
        k1.metric("Lignes (avant)", _intfmt(len(df)))
        k2.metric("Lignes (après)", _intfmt(len(cleaned_df)))
        k3.metric("Colonnes", _intfmt(cleaned_df.shape[1]))


# -----------------------------
# UI — Écran 4: Mapping sémantique
# -----------------------------
st.divider()
st.subheader("Écran 4 — Mapping sémantique (colonnes clés)")
st.caption("Objectif : confirmer les colonnes Date / Montant / Produit / Client / Canal pour standardiser les KPI.")

# Base de travail: si nettoyage appliqué, on prend cleaned_df, sinon raw_df
base_df = st.session_state.get("cleaned_df", st.session_state["raw_df"])

# Reprofiler léger pour récupérer suspected cols sur la base courante
summary2 = build_profile_summary(base_df)
suggest = suggest_columns(base_df, summary2.suspected_numeric, summary2.suspected_date)

m_left, m_right = st.columns([1, 1])

with m_left:
    st.markdown("### Sélection des colonnes")

    date_default = suggest["date"][0] if suggest["date"] else None
    amount_default = suggest["amount"][0] if suggest["amount"] else None

    date_col = st.selectbox(
        "Colonne Date (obligatoire)",
        options=suggest["all"],
        index=suggest["all"].index(date_default) if date_default in suggest["all"] else 0,
    )

    amount_col = st.selectbox(
        "Colonne Montant (obligatoire)",
        options=suggest["all"],
        index=suggest["all"].index(amount_default) if amount_default in suggest["all"] else 0,
    )

    product_options = ["(Aucune)"] + suggest["all"]
    client_options = ["(Aucune)"] + suggest["all"]
    channel_options = ["(Aucune)"] + suggest["all"]

    product_default = suggest["product"][0] if suggest["product"] else "(Aucune)"
    client_default = suggest["client"][0] if suggest["client"] else "(Aucune)"
    channel_default = suggest["channel"][0] if suggest["channel"] else "(Aucune)"

    product_col = st.selectbox(
        "Colonne Produit (optionnel)",
        options=product_options,
        index=product_options.index(product_default) if product_default in product_options else 0,
    )
    client_col = st.selectbox(
        "Colonne Client (optionnel)",
        options=client_options,
        index=client_options.index(client_default) if client_default in client_options else 0,
    )
    channel_col = st.selectbox(
        "Colonne Canal (optionnel)",
        options=channel_options,
        index=channel_options.index(channel_default) if channel_default in channel_options else 0,
    )

    mapping = Mapping(
        date_col=date_col,
        amount_col=amount_col,
        product_col=None if product_col == "(Aucune)" else product_col,
        client_col=None if client_col == "(Aucune)" else client_col,
        channel_col=None if channel_col == "(Aucune)" else channel_col,
    )

    validate_btn = st.button("Valider le mapping", type="primary", use_container_width=True)

with m_right:
    st.markdown("### Prévisualisation standardisée")

    if validate_btn:
        err = validate_mapping(base_df, mapping)
        if err:
            st.error(err)
        else:
            std_df = standardize_dataset(base_df, mapping)

            # Persist mapping + standardized dataset
            st.session_state["mapping"] = mapping
            st.session_state["std_df"] = std_df

            st.success("Mapping validé. Dataset standardisé prêt pour les KPI.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Lignes", _intfmt(len(std_df)))
            c2.metric("Dates nulles", _intfmt(int(std_df["Date"].isna().sum())))
            c3.metric("Montants nulls", _intfmt(int(std_df["Montant"].isna().sum())))

            st.markdown("**Aperçu dataset standardisé (10 lignes)**")
            show_cols = [c for c in ["Date", "Montant", "Produit", "Client", "Canal"] if c in std_df.columns]
            st.dataframe(std_df[show_cols].head(10), use_container_width=True)

    else:
        st.info("Sélectionne les colonnes, puis clique sur « Valider le mapping ».")


# -----------------------------
# UI — Écran 5: KPIs + Insights
# -----------------------------
st.divider()
st.subheader("Écran 5 — KPIs automatiques & Insights (niveau direction)")
st.caption("Dashboard filtrable + analyses automatiques basées sur le dataset standardisé.")

std_df = st.session_state.get("std_df")
if std_df is None:
    st.warning("Mapping non validé : retourne à l’Écran 4 et clique « Valider le mapping ».")
    st.stop()

ensure_columns(std_df)

# Filtres (pro)
with st.expander("Filtres", expanded=True):
    f1, f2, f3 = st.columns([1, 1, 1])

    dmin = pd.to_datetime(std_df["Date"], errors="coerce").min()
    dmax = pd.to_datetime(std_df["Date"], errors="coerce").max()

    with f1:
        date_range = st.date_input(
            "Période",
            value=(dmin.date() if pd.notna(dmin) else None, dmax.date() if pd.notna(dmax) else None),
        )

    produit = None
    client = None
    canal = None

    with f2:
        if "Produit" in std_df.columns:
            produit = st.selectbox("Produit", options=["(Tous)"] + sorted(std_df["Produit"].dropna().unique().tolist()))
            if produit == "(Tous)":
                produit = None

        if "Client" in std_df.columns:
            client = st.selectbox("Client", options=["(Tous)"] + sorted(std_df["Client"].dropna().unique().tolist()))
            if client == "(Tous)":
                client = None

    with f3:
        if "Canal" in std_df.columns:
            canal = st.selectbox("Canal", options=["(Tous)"] + sorted(std_df["Canal"].dropna().unique().tolist()))
            if canal == "(Tous)":
                canal = None

    gran = st.selectbox("Granularité temporelle", options=["Jour", "Semaine", "Mois"], index=2)

# Convert date range
date_min = pd.to_datetime(date_range[0]) if isinstance(date_range, (tuple, list)) and date_range[0] else None
date_max = pd.to_datetime(date_range[1]) if isinstance(date_range, (tuple, list)) and date_range[1] else None

df_f = filter_std_df(std_df, date_min, date_max, produit, client, canal)

if df_f.empty:
    st.warning("Aucune donnée après application des filtres. Ajuste la période ou supprime un filtre.")
    st.stop()

# KPIs
k = compute_kpis(df_f)
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("CA", f"{k.ca_total:,.2f}".replace(",", " ").replace(".", ","))
k2.metric("Transactions (lignes)", _intfmt(k.nb_lignes))
k3.metric("Jours couverts", _intfmt(k.nb_jours))
k4.metric("Produits", _intfmt(k.nb_produits) if k.nb_produits is not None else "—")
k5.metric("Clients", _intfmt(k.nb_clients) if k.nb_clients is not None else "—")

st.markdown("### Évolution du CA")
freq = {"Jour": "D", "Semaine": "W", "Mois": "M"}[gran]
ts = ca_timeseries(df_f, freq=freq)

fig_ts = px.line(ts, x="Periode", y="CA", markers=True)
st.plotly_chart(fig_ts, use_container_width=True)

# Top dimensions
t1, t2 = st.columns([1, 1])

with t1:
    st.markdown("### Top Produits (CA)")
    if "Produit" in df_f.columns:
        top_p = top_dimension(df_f, "Produit", top_n=10)
        fig_top_p = px.bar(top_p, x="CA", y="Produit", orientation="h")
        st.plotly_chart(fig_top_p, use_container_width=True)
        st.dataframe(top_p, use_container_width=True)
    else:
        st.info("Colonne Produit absente (mapping).")

with t2:
    st.markdown("### Top Clients (CA)")
    if "Client" in df_f.columns:
        top_c = top_dimension(df_f, "Client", top_n=10)
        fig_top_c = px.bar(top_c, x="CA", y="Client", orientation="h")
        st.plotly_chart(fig_top_c, use_container_width=True)
        st.dataframe(top_c, use_container_width=True)
    else:
        st.info("Colonne Client absente (mapping).")

# Pareto
st.markdown("### Pareto 80/20 (Produits)")
if "Produit" in df_f.columns:
    pareto = pareto_dimension(df_f, "Produit")
    nb_80, part_items = pareto_80_20_stats(pareto)

    st.write(
        f"**Résultat :** {nb_80} produit(s) représentent ~80% du CA "
        f"({part_items*100:.1f}% des produits)."
    )

    pareto_show = pareto.copy()
    pareto_show["Part_CA_%"] = (pareto_show["Part_CA"] * 100).round(2)
    pareto_show["Cumul_CA_%"] = (pareto_show["Cumul_CA"] * 100).round(2)

    fig_p = px.line(pareto_show, x=pareto_show.index + 1, y="Cumul_CA_%", markers=True)
    st.plotly_chart(fig_p, use_container_width=True)

    st.dataframe(pareto_show[[ "Produit", "CA", "Part_CA_%", "Cumul_CA_%"]].head(30), use_container_width=True)
else:
    st.info("Colonne Produit absente (mapping).")

# Insights
st.markdown("### Insights automatiques")
ins = generate_insights(df_f)
for i in ins:
    st.write(f"- {i}")



# -----------------------------
# UI — Écran 6: Synthèse Direction (COMEX)
# -----------------------------
st.divider()
st.subheader("Écran 6 — Synthèse Direction (COMEX)")
st.caption("One-pager décisionnel : KPIs clés + tendance + top contributeurs + exports (CSV/Excel/PDF).")

std_df = st.session_state.get("std_df")
if std_df is None:
    st.warning("Dataset standardisé absent. Valide le mapping (Écran 4) puis reviens ici.")
    st.stop()

# Reprendre les filtres de l’Écran 5 si df_f existe (sinon fallback = tout le dataset)
try:
    df_f = locals().get("df_f", None)
except Exception:
    df_f = None

if df_f is None:
    df_f = std_df.copy()

if df_f.empty:
    st.warning("Aucune donnée exploitable pour la synthèse.")
    st.stop()

# KPIs
k = compute_kpis(df_f)

# Label période (si Écran 5 a défini date_range, on l'affiche, sinon générique)
period_label = "Filtré (voir Écran 5)"
try:
    dr = locals().get("date_range", None)
    if isinstance(dr, (tuple, list)) and len(dr) == 2 and dr[0] and dr[1]:
        period_label = f"{dr[0]} → {dr[1]}"
except Exception:
    pass

# Layout COMEX : KPIs en ligne (compact)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("CA", f"{k.ca_total:,.2f}".replace(",", " ").replace(".", ","))
m2.metric("Transactions", _intfmt(k.nb_lignes))
m3.metric("Panier moyen", f"{k.ca_moyen_par_ligne:,.2f}".replace(",", " ").replace(".", ","))
m4.metric("Produits", _intfmt(k.nb_produits) if k.nb_produits is not None else "—")
m5.metric("Clients", _intfmt(k.nb_clients) if k.nb_clients is not None else "—")

# Trend + Top (2 colonnes)
left, right = st.columns([1.6, 1])

with left:
    st.markdown("### CA mensuel (tendance)")
    ts_m = ca_timeseries(df_f, freq="M")

    fig = px.line(ts_m, x="Periode", y="CA", markers=True)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Top contributeurs")

    top_p = None
    top_c = None

    if "Produit" in df_f.columns:
        top_p = top_dimension(df_f, "Produit", top_n=5)
        fig_p = px.bar(top_p, x="CA", y="Produit", orientation="h")
        fig_p.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.info("Top Produits indisponible (colonne Produit absente).")

    if "Client" in df_f.columns:
        top_c = top_dimension(df_f, "Client", top_n=5)
        fig_c = px.bar(top_c, x="CA", y="Client", orientation="h")
        fig_c.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("Top Clients indisponible (colonne Client absente).")

# Insights
st.markdown("### Points clés (insights)")
insights = generate_insights(df_f)
for i in insights:
    st.write(f"- {i}")

# Exports
st.markdown("### Export")
e1, e2, e3, e4 = st.columns([1, 1, 1, 2])

# CSV
with e1:
    pack_csv = to_csv_bytes(df_f, filename="dataset_filtre.csv")
    st.download_button(
        "Dataset (CSV)",
        data=pack_csv.bytes_data,
        file_name=pack_csv.filename,
        mime=pack_csv.mime,
        use_container_width=True,
    )

# Excel
with e2:
    kpi_df = pd.DataFrame([{
        "Periode": period_label,
        "CA": k.ca_total,
        "Transactions": k.nb_lignes,
        "Jours_couverts": k.nb_jours,
        "Panier_moyen": k.ca_moyen_par_ligne,
        "Nb_produits": k.nb_produits,
        "Nb_clients": k.nb_clients,
    }])

    sheets = {
        "KPI_COMEX": kpi_df,
        "CA_Mensuel": ts_m,
        "Dataset_Filtre": df_f.head(5000),  # garde-fou taille
    }

    if "Produit" in df_f.columns:
        sheets["Top_Produits"] = top_dimension(df_f, "Produit", top_n=20)
    if "Client" in df_f.columns:
        sheets["Top_Clients"] = top_dimension(df_f, "Client", top_n=20)

    try:
        pack_xlsx = to_excel_bytes(sheets, filename="pack_comex.xlsx")
        st.download_button(
            "Pack (Excel)",
            data=pack_xlsx.bytes_data,
            file_name=pack_xlsx.filename,
            mime=pack_xlsx.mime,
            use_container_width=True,
        )
    except Exception as ex:
        st.error(f"Export Excel indisponible : {ex}")

# PDF
with e3:
    # Recalcule Top en version 20 pour alimenter le PDF (mais affichage top 5 déjà fait)
    tp_pdf = top_dimension(df_f, "Produit", top_n=20) if "Produit" in df_f.columns else None
    tc_pdf = top_dimension(df_f, "Client", top_n=20) if "Client" in df_f.columns else None

    kpi_dict = {
        "CA": k.ca_total,
        "Transactions": k.nb_lignes,
        "Panier_moyen": k.ca_moyen_par_ligne,
        "Nb_produits": k.nb_produits if k.nb_produits is not None else "—",
        "Nb_clients": k.nb_clients if k.nb_clients is not None else "—",
    }

    try:
        pdf_pack = build_comex_pdf(
            title="Synthèse Direction (COMEX)",
            period_label=period_label,
            kpi=kpi_dict,
            insights=insights,
            ts_mensuel=ts_m,
            top_produits=tp_pdf,
            top_clients=tc_pdf,
            filename="comex_one_pager.pdf",
        )

        st.download_button(
            "Synthèse (PDF)",
            data=pdf_pack.bytes_data,
            file_name=pdf_pack.filename,
            mime=pdf_pack.mime,
            use_container_width=True,
        )
    except Exception as ex:
        st.error(f"Export PDF indisponible : {ex}")

# Info export
with e4:
    st.info(
        "Exports : CSV (dataset filtré), Excel (KPI + CA mensuel + tops + extrait dataset), PDF (one-pager A4 COMEX)."
    )



# -----------------------------
# UI — Écran 7: Comparatif N vs N-1 (YoY) + fallback pro
# -----------------------------
st.divider()
st.subheader("Écran 7 — Comparatif N vs N-1 (YoY)")
st.caption("Analyse direction : évolution annuelle du chiffre d’affaires (avec fallback automatique).")

std_df = st.session_state.get("std_df")
if std_df is None:
    st.warning("Dataset standardisé absent.")
    st.stop()

# Reprendre les filtres s’ils existent (df_f), sinon fallback sur std_df
df_base = None
try:
    df_base = locals().get("df_f", None)
except Exception:
    df_base = None

if df_base is None or df_base.empty:
    df_base = std_df.copy()

# Split N / N-1
df_n, df_n1 = yoy_split(df_base)

# Fallback pro si historique insuffisant
if df_n.empty or df_n1.empty:
    st.info(
        "Historique annuel insuffisant pour un comparatif N vs N-1. "
        "Affichage de la tendance mensuelle sur l’année courante."
    )

    ts_single = ca_timeseries(df_base, freq="M")
    fig_single = px.line(ts_single, x="Periode", y="CA", markers=True)
    fig_single.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))

    # ✅ clé unique pour éviter StreamlitDuplicateElementId
    st.plotly_chart(fig_single, use_container_width=True, key="yoy_fallback_ts")

    st.markdown("### Lecture rapide")
    st.write("- Une seule année est disponible dans les données.")
    st.write("- Le comparatif YoY s’activera automatiquement dès qu’une année supplémentaire sera présente.")

    st.stop()

# KPIs YoY
yk = yoy_kpis(df_n, df_n1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("CA N", f"{yk['CA_N']:,.2f}".replace(",", " ").replace(".", ","))
c2.metric("CA N-1", f"{yk['CA_N1']:,.2f}".replace(",", " ").replace(".", ","))
c3.metric("Δ CA", f"{yk['Delta']:,.2f}".replace(",", " ").replace(".", ","))
c4.metric("Δ %", f"{yk['Pct']:.1f}%" if yk["Pct"] is not None else "—")

# Courbe YoY (par mois)
ts_yoy = yoy_timeseries(df_base)
fig_yoy = px.line(ts_yoy, x="Mois", y="CA", color="Annee", markers=True)
fig_yoy.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))

# ✅ clé unique
st.plotly_chart(fig_yoy, use_container_width=True, key="yoy_compare_ts")

# Insight automatique
st.markdown("### Lecture rapide")
if yk["Pct"] is None:
    st.write("- Année N-1 sans CA : variation non calculable.")
elif yk["Pct"] >= 0:
    st.write(f"- Croissance annuelle de **{yk['Pct']:.1f}%** par rapport à N-1.")
else:
    st.write(f"- Repli annuel de **{abs(yk['Pct']):.1f}%** par rapport à N-1.")
