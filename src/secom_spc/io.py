"""Lecture des données SECOM et persistance en base SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SENSOR_PREFIX = "capteur_"


def load_secom(raw_dir: str | Path) -> pd.DataFrame:
    """Charge ``secom.data`` et ``secom_labels.data`` en un seul DataFrame.

    Le fichier de mesures contient une ligne par lot et une colonne par capteur
    (valeurs séparées par des espaces, ``NaN`` pour les mesures manquantes).
    Le fichier d'étiquettes contient le verdict du test final (``-1`` = conforme,
    ``1`` = rejeté) et l'horodatage du lot.

    Returns
    -------
    DataFrame trié par horodatage, avec les colonnes ``horodatage``, ``defaut``
    (0/1) puis ``capteur_000`` … ``capteur_589``.
    """
    raw_dir = Path(raw_dir)
    X = pd.read_csv(raw_dir / "secom.data", sep=r"\s+", header=None)
    X.columns = [f"{SENSOR_PREFIX}{i:03d}" for i in range(X.shape[1])]

    labels = pd.read_csv(
        raw_dir / "secom_labels.data", sep=r"\s+", header=None, names=["verdict", "horodatage"]
    )
    labels["horodatage"] = pd.to_datetime(labels["horodatage"], format="%d/%m/%Y %H:%M:%S")
    labels["defaut"] = (labels["verdict"] == 1).astype(int)

    df = pd.concat([labels[["horodatage", "defaut"]], X], axis=1)
    return df.sort_values("horodatage", kind="stable").reset_index(drop=True)


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """Liste des colonnes de mesures (celles qui commencent par ``capteur_``)."""
    return [c for c in df.columns if c.startswith(SENSOR_PREFIX)]


def to_sqlite(df: pd.DataFrame, db_path: str | Path, table: str = "mesures_testeurs") -> None:
    """Écrit le DataFrame dans une base SQLite (table remplacée si elle existe)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql(table, con, if_exists="replace", index=False)
