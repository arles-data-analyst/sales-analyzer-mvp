from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass
class LoadResult:
    df: Optional[pd.DataFrame]
    sep: Optional[str]
    encoding: Optional[str]
    error: Optional[str]


def detect_separator(sample_text: str) -> str:
    """
    Détecte le séparateur le plus probable parmi: ',', ';', '\\t', '|'.
    Stratégie: stabilité du nombre de colonnes sur plusieurs lignes.
    """
    candidates = [",", ";", "\t", "|"]
    best_sep = ";"
    best_score = -1.0

    lines = [ln for ln in sample_text.splitlines() if ln.strip()][:20]
    if len(lines) < 2:
        return best_sep

    for sep in candidates:
        counts = [len(line.split(sep)) for line in lines]
        if not counts:
            continue
        # score = (beaucoup de colonnes) - (variance)
        s = pd.Series(counts, dtype="float")
        score = float(s.max() - (s.std() if len(s) > 1 else 0.0))
        if score > best_score:
            best_score = score
            best_sep = sep

    return best_sep


def _try_read_csv(raw_bytes: bytes, sep: str, encoding: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep=sep,
            encoding=encoding,
            engine="python",
            dtype_backend="numpy_nullable",
            on_bad_lines="skip",
        )
        return df, None
    except UnicodeDecodeError:
        return None, f"Encodage incompatible ({encoding})."
    except Exception as e:
        return None, f"Erreur de lecture CSV : {e}"


def load_csv(uploaded_file) -> LoadResult:
    """
    Charge un CSV Streamlit (UploadedFile) avec détection séparateur/encodage.
    """
    if uploaded_file is None:
        return LoadResult(df=None, sep=None, encoding=None, error="Aucun fichier chargé.")

    raw = uploaded_file.getvalue()
    if not raw or len(raw) < 10:
        return LoadResult(df=None, sep=None, encoding=None, error="Fichier vide ou illisible.")

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    # échantillon pour détecter séparateur
    sample_text = None
    used_encoding_for_sample = None
    for enc in encodings_to_try:
        try:
            sample_text = raw[:50_000].decode(enc, errors="strict")
            used_encoding_for_sample = enc
            break
        except Exception:
            continue

    if sample_text is None:
        sample_text = raw[:50_000].decode("latin-1", errors="replace")
        used_encoding_for_sample = "latin-1"

    sep = detect_separator(sample_text)

    last_error = None
    for enc in encodings_to_try:
        df, err = _try_read_csv(raw, sep=sep, encoding=enc)
        if df is not None:
            return LoadResult(df=df, sep=sep, encoding=enc, error=None)
        last_error = err

    return LoadResult(df=None, sep=sep, encoding=used_encoding_for_sample, error=last_error or "Impossible de lire le CSV.")


def basic_validation(df: pd.DataFrame) -> Optional[str]:
    """
    Contrôles minimum d’ingestion.
    """
    if df is None:
        return "DataFrame absent."
    if df.shape[0] == 0:
        return "Le fichier ne contient aucune ligne exploitable."
    if df.shape[1] < 2:
        return "Le fichier doit contenir au moins 2 colonnes."
    if df.columns.duplicated().any():
        return "Certaines colonnes ont le même nom (doublons)."
    return None
