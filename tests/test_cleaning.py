import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secom_spc.cleaning import clean_sensors  # noqa: E402
from secom_spc.io import sensor_columns  # noqa: E402


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "horodatage": pd.date_range("2024-01-01", periods=n, freq="h"),
            "defaut": rng.integers(0, 2, n),
            "capteur_000": rng.normal(size=n),
            "capteur_001": 3.0,  # constant
            "capteur_002": rng.normal(size=n),
            "capteur_003": rng.normal(size=n),
        }
    )
    df.loc[: int(n * 0.5), "capteur_002"] = np.nan  # 50 % de valeurs manquantes
    df.loc[10, "capteur_003"] = 1e6  # outlier isolé
    df.loc[20, "capteur_000"] = np.nan
    return df


def test_retire_capteurs_constants_et_trop_incomplets():
    cleaned, report = clean_sensors(_frame(), max_missing_rate=0.30)
    assert report.dropped_constant == ["capteur_001"]
    assert report.dropped_missing == ["capteur_002"]
    assert sensor_columns(cleaned) == ["capteur_000", "capteur_003"]
    assert report.n_sensors_in == 4 and report.n_sensors_out == 2


def test_impute_et_ecrete():
    cleaned, _ = clean_sensors(_frame(), winsor_quantile=0.01)
    assert not cleaned["capteur_000"].isna().any()
    assert cleaned["capteur_003"].max() < 100  # l'outlier a été écrêté


def test_conserve_les_colonnes_non_capteurs_et_l_ordre_des_lignes():
    df = _frame()
    cleaned, _ = clean_sensors(df)
    assert list(cleaned.columns[:2]) == ["horodatage", "defaut"]
    assert (cleaned["horodatage"] == df["horodatage"]).all()
