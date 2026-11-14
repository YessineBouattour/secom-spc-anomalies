"""Maîtrise statistique des procédés (SPC) sur des mesures individuelles.

Démarche en deux phases, comme sur une ligne réelle :

- **Phase I (référence)** : les premiers lots servent à estimer le centrage et la
  dispersion du procédé « sous contrôle ». On en déduit les limites de contrôle
  de la carte I et, faute de tolérances publiées pour SECOM, des limites de
  spécification ``mu_ref ± spec_sigma · sigma_ref``.
- **Phase II (surveillance)** : les lots suivants sont comparés à ces références.
  Les indices de capabilité y sont recalculés : un Cpk qui chute traduit une
  dérive du centrage ou une augmentation de la dispersion.

Chaque lot ne fournit qu'une mesure par capteur : on utilise la carte des
valeurs individuelles et des étendues mobiles (I-MR). L'écart-type est estimé
par ``sigma = MR_moy / d2`` avec ``d2 = 1,128`` (sous-groupes de taille 2).

Indices de capabilité :

- ``Cp  = (USL - LSL) / (6 sigma)``             : capabilité potentielle ;
- ``Cpk = min(USL - mu, mu - LSL) / (3 sigma)`` : capabilité réelle (centrage inclus).

Un procédé est généralement jugé capable à partir de Cpk = 1,33.

Les règles de détection (dérivées des règles Western Electric) sont évaluées
point par point puis regroupées en **épisodes** : une suite ininterrompue de
points en alerte pour une même règle compte pour un seul événement, ce qui
correspond à ce qu'un opérateur traite réellement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

D2_N2 = 1.128  # constante d2 des tables SPC pour des sous-groupes de taille 2
CPK_TARGET = 1.33
RULES = ("hors_limites", "run_8", "tendance_6")


@dataclass(frozen=True)
class ControlLimits:
    center: float
    ucl: float
    lcl: float
    sigma: float


def estimate_sigma(x: pd.Series) -> float:
    """Écart-type du procédé estimé par l'étendue mobile moyenne (carte I-MR)."""
    mr = x.diff().abs().dropna()
    return float(mr.mean() / D2_N2) if len(mr) else float("nan")


def imr_limits(x_ref: pd.Series, k: float = 3.0) -> ControlLimits:
    """Limites de contrôle de la carte I calculées sur la période de référence."""
    sigma = estimate_sigma(x_ref)
    center = float(x_ref.mean())
    return ControlLimits(center=center, ucl=center + k * sigma, lcl=center - k * sigma, sigma=sigma)


def capability(x: pd.Series, lsl: float, usl: float) -> tuple[float, float]:
    """Cp et Cpk de la série ``x`` par rapport aux tolérances ``[lsl, usl]``."""
    sigma = estimate_sigma(x)
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan"), float("nan")
    mu = float(x.mean())
    cp = (usl - lsl) / (6 * sigma)
    cpk = min(usl - mu, mu - lsl) / (3 * sigma)
    return float(cp), float(cpk)


def western_electric(x: pd.Series, limits: ControlLimits) -> pd.DataFrame:
    """Applique trois règles de détection à une série de mesures.

    - ``hors_limites`` : point au-delà de ± 3 sigma ;
    - ``run_8``        : 8 points consécutifs du même côté de la moyenne (décalage) ;
    - ``tendance_6``   : 6 points consécutifs strictement monotones (dérive).

    Retourne un DataFrame booléen aligné sur ``x`` (une colonne par règle plus
    ``alerte``).
    """
    side = np.sign(x - limits.center)
    hors_limites = (x > limits.ucl) | (x < limits.lcl)
    run_8 = side.rolling(8).sum().abs().eq(8)
    tendance_6 = np.sign(x.diff()).rolling(5).sum().abs().eq(5)  # 6 points = 5 variations de même signe
    out = pd.DataFrame(
        {"hors_limites": hors_limites, "run_8": run_8, "tendance_6": tendance_6}, index=x.index
    ).fillna(False).astype(bool)
    out["alerte"] = out[list(RULES)].any(axis=1)
    return out


def episodes(flags: pd.Series) -> list[tuple[int, int]]:
    """Regroupe une série booléenne en épisodes (positions de début et de fin incluses)."""
    f = flags.to_numpy(dtype=bool)
    if not f.any():
        return []
    edges = np.diff(np.concatenate([[0], f.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def spc_report(
    df: pd.DataFrame,
    sensors: list[str],
    ref_frac: float = 0.30,
    spec_sigma: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limites, capabilité phase II et épisodes d'alerte pour chaque capteur.

    Parameters
    ----------
    ref_frac   : part des lots (les premiers, chronologiquement) formant la phase I.
    spec_sigma : demi-largeur des tolérances, en sigma de référence.

    Returns
    -------
    resume   : une ligne par capteur, triée par Cpk croissant.
    episodes : une ligne par épisode d'alerte (capteur, règle, début, fin, durée,
               nombre de lots rejetés pendant l'épisode).
    """
    n_ref = int(round(len(df) * ref_frac))
    ref_idx, mon_idx = df.index[:n_ref], df.index[n_ref:]
    rows, eps = [], []

    for c in sensors:
        x = df[c]
        lim = imr_limits(x.loc[ref_idx])
        lsl, usl = lim.center - spec_sigma * lim.sigma, lim.center + spec_sigma * lim.sigma
        cp_ref, cpk_ref = capability(x.loc[ref_idx], lsl, usl)
        cp, cpk = capability(x.loc[mon_idx], lsl, usl)
        rules = western_electric(x, lim)

        row = {
            "capteur": c,
            "centre_ref": lim.center,
            "sigma_ref": lim.sigma,
            "lcl": lim.lcl,
            "ucl": lim.ucl,
            "lsl": lsl,
            "usl": usl,
            "cp_ref": cp_ref,
            "cpk_ref": cpk_ref,
            "centre_phase2": float(x.loc[mon_idx].mean()),
            "sigma_phase2": estimate_sigma(x.loc[mon_idx]),
            "cp": cp,
            "cpk": cpk,
            "n_points_alerte": int(rules["alerte"].sum()),
        }
        for rule in RULES:
            ep = episodes(rules[rule])
            row[f"n_episodes_{rule}"] = len(ep)
            for s, e in ep:
                seg = df.iloc[s : e + 1]
                eps.append(
                    {
                        "capteur": c,
                        "regle": rule,
                        "debut": seg["horodatage"].iloc[0],
                        "fin": seg["horodatage"].iloc[-1],
                        "n_lots": int(e - s + 1),
                        "n_lots_rejetes": int(seg["defaut"].sum()),
                        "phase": "I" if e < n_ref else "II",
                    }
                )
        row["n_episodes"] = sum(row[f"n_episodes_{r}"] for r in RULES)
        rows.append(row)

    resume = pd.DataFrame(rows).sort_values("cpk").reset_index(drop=True)
    cols = ["capteur", "regle", "debut", "fin", "n_lots", "n_lots_rejetes", "phase"]
    ep_df = pd.DataFrame(eps, columns=cols).sort_values(["debut", "capteur"]).reset_index(drop=True)
    return resume, ep_df
