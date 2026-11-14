"""Nettoyage des mesures issues des testeurs automatiques.

Trois défauts classiques des données de test sont traités :

1. capteurs trop souvent absents (taux de valeurs manquantes au-dessus d'un seuil) ;
2. capteurs constants, qui ne portent aucune information ;
3. valeurs extrêmes isolées (erreurs de mesure), écrêtées par winsorisation.

Les valeurs manquantes restantes sont remplacées par la médiane du capteur, plus
robuste que la moyenne en présence d'outliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .io import sensor_columns


@dataclass
class CleaningReport:
    """Trace de ce que le nettoyage a retiré ou modifié."""

    n_sensors_in: int = 0
    n_sensors_out: int = 0
    dropped_missing: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)
    missing_rate_before: float = 0.0

    def as_dict(self) -> dict:
        return {
            "capteurs_en_entree": self.n_sensors_in,
            "capteurs_conserves": self.n_sensors_out,
            "capteurs_retires_trop_de_manquants": len(self.dropped_missing),
            "capteurs_retires_constants": len(self.dropped_constant),
            "taux_valeurs_manquantes_avant": round(self.missing_rate_before, 4),
        }


def clean_sensors(
    df: pd.DataFrame,
    max_missing_rate: float = 0.30,
    winsor_quantile: float = 0.005,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Nettoie les colonnes capteurs et retourne (DataFrame nettoyé, rapport).

    Parameters
    ----------
    max_missing_rate : part maximale de valeurs manquantes tolérée par capteur.
    winsor_quantile  : quantile d'écrêtage ; 0,005 écrête les 0,5 % les plus bas
                       et les 0,5 % les plus hauts de chaque capteur.
    """
    cols = sensor_columns(df)
    report = CleaningReport(n_sensors_in=len(cols))
    X = df[cols]
    report.missing_rate_before = float(X.isna().mean().mean())

    missing_rate = X.isna().mean()
    report.dropped_missing = sorted(missing_rate.index[missing_rate > max_missing_rate])
    X = X.drop(columns=report.dropped_missing)

    std = X.std()
    report.dropped_constant = sorted(std.index[(std == 0) | std.isna()])
    X = X.drop(columns=report.dropped_constant)

    X = X.fillna(X.median())
    low, high = X.quantile(winsor_quantile), X.quantile(1 - winsor_quantile)
    X = X.clip(lower=low, upper=high, axis=1)

    report.n_sensors_out = X.shape[1]
    other = [c for c in df.columns if c not in cols]
    return pd.concat([df[other], X], axis=1), report
