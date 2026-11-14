"""Figures statiques pour le README (matplotlib), générées à partir des sorties du pipeline.

Usage : python scripts/figures.py [--out outputs] [--docs docs]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secom_spc.cleaning import clean_sensors  # noqa: E402
from secom_spc.io import load_secom  # noqa: E402
from secom_spc.spc import CPK_TARGET, imr_limits, western_electric  # noqa: E402

BLEU, ROUGE, VERT, ORANGE = "#1f3a6e", "#c0392b", "#16a34a", "#f59e0b"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--ref-frac", type=float, default=0.30)
    a = ap.parse_args()
    out, docs = Path(a.out), Path(a.docs)
    docs.mkdir(exist_ok=True)

    df, _ = clean_sensors(load_secom(a.raw))
    resume = pd.read_csv(out / "spc_capabilite.csv")
    proba = pd.read_csv(out / "probabilites_rejet.csv", parse_dates=["horodatage"])
    imp = pd.read_csv(out / "importance_capteurs.csv", index_col=0).iloc[:, 0]
    n_ref = int(round(len(df) * a.ref_frac))

    # 1. Carte de contrôle du capteur au Cpk le plus faible
    c = resume["capteur"].iloc[0]
    x = df[c]
    lim = imr_limits(x.iloc[:n_ref])
    al = western_electric(x, lim)["alerte"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(df["horodatage"], x, color="#94a3b8", lw=0.8)
    ax.scatter(df["horodatage"][~al], x[~al], s=8, color=BLEU, label="sous contrôle")
    ax.scatter(df["horodatage"][al], x[al], s=10, color=ROUGE, label="alerte")
    for y, lab, col in [(lim.center, "centre (phase I)", VERT), (lim.ucl, "UCL +3σ", ROUGE), (lim.lcl, "LCL −3σ", ROUGE)]:
        ax.axhline(y, ls="--", lw=1, color=col, label=lab)
    ax.axvline(df["horodatage"].iloc[n_ref - 1], color=ORANGE, lw=1, label="fin phase I")
    row = resume.iloc[0]
    ax.set_title(f"Carte I – {c} · Cpk phase II = {row['cpk']:.2f} · {int(row['n_episodes'])} épisodes d'alerte", fontsize=11)
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.tight_layout()
    fig.savefig(docs / "carte_controle.png", dpi=130)

    # 2. Cpk phase II des 25 capteurs les moins capables
    top = resume.head(25)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(top["capteur"], top["cpk"], color=[ROUGE if v < CPK_TARGET else BLEU for v in top["cpk"]])
    ax.axhline(CPK_TARGET, ls="--", color=VERT, lw=1)
    ax.text(len(top) - 0.5, CPK_TARGET + 0.03, "cible 1,33", color=VERT, ha="right", fontsize=9)
    ax.set_ylabel("Cpk (phase II)")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.set_title("Capteurs les moins capables en phase de surveillance", fontsize=11)
    fig.tight_layout()
    fig.savefig(docs / "cpk.png", dpi=130)

    # 3. Probabilité de rejet prédite dans le temps
    seuil = float(pd.read_json(out / "metrics.json")["foret_aleatoire"]["seuil"])
    ok, ko = proba[proba["defaut"] == 0], proba[proba["defaut"] == 1]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.scatter(ok["horodatage"], ok["proba_rejet"], s=8, color="#9ca3af", label="lot conforme")
    ax.scatter(ko["horodatage"], ko["proba_rejet"], s=16, color=ROUGE, label="lot rejeté")
    ax.axhline(seuil, ls="--", color=ORANGE, lw=1, label=f"seuil d'alerte ({seuil:.2f})")
    ax.set_ylabel("probabilité de rejet (hors pli)")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.set_title("Forêt aléatoire – probabilité de rejet par lot", fontsize=11)
    fig.tight_layout()
    fig.savefig(docs / "probabilite_rejet.png", dpi=130)

    # 4. Importance des capteurs
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top.index, top.values, color=BLEU)
    ax.set_xlabel("importance (diminution moyenne d'impureté)")
    ax.set_title("Capteurs les plus explicatifs du rejet", fontsize=11)
    fig.tight_layout()
    fig.savefig(docs / "importance_capteurs.png", dpi=130)
    print(f"figures écrites dans {docs.resolve()}")


if __name__ == "__main__":
    main()
