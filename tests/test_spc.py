import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secom_spc.spc import (  # noqa: E402
    D2_N2,
    capability,
    episodes,
    imr_limits,
    spc_report,
    western_electric,
)


def test_imr_limits_sur_bruit_gaussien():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(10.0, 2.0, 5000))
    lim = imr_limits(x)
    assert lim.center == pytest.approx(10.0, abs=0.1)
    assert lim.sigma == pytest.approx(2.0, rel=0.05)
    assert lim.ucl == pytest.approx(lim.center + 3 * lim.sigma)
    assert lim.lcl == pytest.approx(lim.center - 3 * lim.sigma)


def test_sigma_estime_par_etendue_mobile():
    x = pd.Series([1.0, 3.0, 1.0, 3.0])  # MR = 2 partout
    assert imr_limits(x).sigma == pytest.approx(2.0 / D2_N2)


def test_capability_procede_centre_et_decentre():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0.0, 1.0, 20000))
    cp, cpk = capability(x, lsl=-4, usl=4)
    assert cp == pytest.approx(1.333, abs=0.05)
    assert cpk == pytest.approx(cp, abs=0.05)
    cp2, cpk2 = capability(x + 2.0, lsl=-4, usl=4)  # décentrage de 2 sigma
    assert cp2 == pytest.approx(cp, abs=0.05)
    assert cpk2 == pytest.approx(0.667, abs=0.05)


def test_capability_sigma_nul():
    cp, cpk = capability(pd.Series([5.0] * 10), lsl=4, usl=6)
    assert np.isnan(cp) and np.isnan(cpk)


def test_regle_hors_limites():
    x = pd.Series([0.0] * 20 + [10.0] + [0.0] * 20)
    lim = imr_limits(pd.Series([0.0, 0.1] * 10))
    rules = western_electric(x, lim)
    assert rules["hors_limites"].sum() == 1
    assert rules.loc[20, "hors_limites"]


def test_regle_run_8():
    x = pd.Series([0.0, 1.0] * 5 + [1.0] * 7)  # 8 points consécutifs au-dessus du centre (indices 9 à 16)
    lim = imr_limits(pd.Series([0.0, 1.0] * 10))
    rules = western_electric(x, lim)
    assert rules["run_8"].sum() == 1
    assert rules["run_8"].iloc[-1]


def test_regle_tendance_6():
    x = pd.Series([5.0, 4.0, 5.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    lim = imr_limits(x)
    rules = western_electric(x, lim)
    assert rules["tendance_6"].sum() == 2  # fin de la montée : positions 9 et 10


def test_episodes_regroupe_les_points_consecutifs():
    flags = pd.Series([False, True, True, False, False, True, False, True, True, True])
    assert episodes(flags) == [(1, 2), (5, 5), (7, 9)]
    assert episodes(pd.Series([False, False])) == []


def test_spc_report_detecte_une_derive_en_phase_2():
    rng = np.random.default_rng(2)
    n = 400
    stable = rng.normal(0.0, 1.0, n)
    derive = stable.copy()
    derive[300:] += 5.0  # décalage de 5 sigma sur les 100 derniers lots
    df = pd.DataFrame(
        {
            "horodatage": pd.date_range("2024-01-01", periods=n, freq="h"),
            "defaut": 0,
            "capteur_000": stable,
            "capteur_001": derive,
        }
    )
    resume, ep = spc_report(df, ["capteur_000", "capteur_001"], ref_frac=0.3)
    stable_row = resume.set_index("capteur").loc["capteur_000"]
    derive_row = resume.set_index("capteur").loc["capteur_001"]
    assert stable_row["cpk"] > 1.0
    assert derive_row["cpk"] < stable_row["cpk"]
    assert derive_row["n_episodes_hors_limites"] >= 1
    assert (ep["capteur"] == "capteur_001").sum() > (ep["capteur"] == "capteur_000").sum()
    assert set(ep["phase"]) <= {"I", "II"}
