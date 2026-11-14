"""Point d'entrée : exécute le pipeline complet et écrit les livrables dans ``outputs/``.

Usage :
    python run_pipeline.py                      # data/raw -> outputs
    python run_pipeline.py --raw chemin/vers/secom --out resultats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from secom_spc.anomaly import isolation_forest_scores, spe_scores, train_supervised  # noqa: E402
from secom_spc.cleaning import clean_sensors  # noqa: E402
from secom_spc.dashboard import build_dashboard  # noqa: E402
from secom_spc.io import load_secom, sensor_columns, to_sqlite  # noqa: E402
from secom_spc.spc import CPK_TARGET, spc_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervision SPC et détection d'anomalies – SECOM")
    parser.add_argument("--raw", default="data/raw", help="dossier contenant secom.data et secom_labels.data")
    parser.add_argument("--out", default="outputs", help="dossier de sortie")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Chargement et nettoyage
    df = load_secom(args.raw)
    print(f"[1/5] {len(df)} lots, {len(sensor_columns(df))} capteurs, taux de rebut {df['defaut'].mean():.1%}")
    df, cleaning = clean_sensors(df)
    sensors = sensor_columns(df)
    print(f"      nettoyage : {cleaning.n_sensors_out} capteurs conservés "
          f"({len(cleaning.dropped_missing)} trop incomplets, {len(cleaning.dropped_constant)} constants)")
    to_sqlite(df, out / "secom.db")

    # 2. SPC (phase I de référence, phase II de surveillance)
    resume, episodes = spc_report(df, sensors)
    resume.to_csv(out / "spc_capabilite.csv", index=False)
    episodes.to_csv(out / "episodes_alerte_spc.csv", index=False)
    n_faible = int((resume["cpk"] < CPK_TARGET).sum())
    print(f"[2/5] SPC : {len(episodes)} épisodes d'alerte, {n_faible} capteurs avec Cpk < {CPK_TARGET} en phase II")

    # 3. Scores non supervisés
    from sklearn.metrics import roc_auc_score
    iso = isolation_forest_scores(df[sensors])
    spe = spe_scores(df[sensors])
    auc_iso, auc_spe = float(roc_auc_score(df["defaut"], iso)), float(roc_auc_score(df["defaut"], spe))
    pd.concat([df[["horodatage", "defaut"]], iso, spe], axis=1).to_csv(out / "scores_non_supervises.csv", index=False)
    print(f"[3/5] Non supervisé : AUC Isolation Forest {auc_iso:.3f}, AUC SPE {auc_spe:.3f}")

    # 4. Prédiction supervisée
    _, metrics, proba, importances = train_supervised(df, sensors)
    importances.to_csv(out / "importance_capteurs.csv", header=True)
    pd.concat([df[["horodatage", "defaut"]], proba], axis=1).to_csv(out / "probabilites_rejet.csv", index=False)
    m = metrics.as_dict()
    print(f"[4/5] Forêt aléatoire : AUC ROC {m['auc_roc_cv']:.3f} (CV) / {m['auc_roc_temporel']:.3f} (temporel), "
          f"AP {m['average_precision_cv']:.3f} vs taux de base {m['taux_base']:.3f} ; "
          f"seuil {m['seuil']:.3f} -> rappel {m['rappel']:.0%}, précision {m['precision']:.0%}")

    # 5. Dashboard et métriques
    build_dashboard(df, resume, episodes, proba, importances, m, out / "dashboard.html")
    summary = {
        "donnees": {"lots": int(len(df)), "periode": [str(df["horodatage"].min()), str(df["horodatage"].max())],
                    "taux_rebut": round(float(df["defaut"].mean()), 4)},
        "nettoyage": cleaning.as_dict(),
        "spc": {"episodes_alerte": int(len(episodes)), "capteurs_cpk_faible_phase2": n_faible,
                "cpk_median_phase2": round(float(resume["cpk"].median()), 3),
                "episodes_par_regle": {k: int(v) for k, v in episodes["regle"].value_counts().items()},
                "capteurs_cpk_plus_faible": resume[["capteur", "cpk"]].head(5).round(3).to_dict("records")},
        "non_supervise": {"auc_isolation_forest": round(auc_iso, 4), "auc_spe": round(auc_spe, 4)},
        "foret_aleatoire": m,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[5/5] Livrables écrits dans {out.resolve()}")


if __name__ == "__main__":
    main()
