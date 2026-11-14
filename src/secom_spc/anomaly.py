"""Détection d'anomalies et prédiction des lots défaillants.

Deux familles d'approches, volontairement comparées :

- **non supervisées** : elles ne connaissent pas le verdict du test final et
  repèrent les lots au profil capteurs atypique. Deux scores sont calculés :
  Isolation Forest, et la statistique SPE (*squared prediction error*, résidu de
  reconstruction d'une ACP), classique en surveillance multivariée de procédés ;
- **supervisée** : une forêt aléatoire apprend à partir des verdicts passés. Le
  jeu est très déséquilibré (≈ 7 % de rejets), d'où la pondération des classes
  et un seuil de décision choisi explicitement plutôt que 0,5.

Le modèle supervisé est évalué de deux façons :

1. validation croisée stratifiée (5 plis) : mesure stable, car elle exploite
   l'ensemble des 104 lots rejetés ;
2. découpage temporel (apprentissage sur les premiers 75 % des lots, test sur
   les derniers 25 %) : plus proche d'un déploiement réel. L'écart entre les
   deux mesure la sensibilité du modèle à la dérive du procédé.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


@dataclass
class SupervisedMetrics:
    auc_roc_cv: float
    average_precision_cv: float
    auc_roc_temporel: float
    average_precision_temporel: float
    seuil: float
    precision: float
    rappel: float
    taux_base: float

    def as_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


# ---------------------------------------------------------------- non supervisé


def isolation_forest_scores(X: pd.DataFrame, random_state: int = 0) -> pd.Series:
    """Score d'anomalie Isolation Forest (plus il est élevé, plus le lot est atypique)."""
    model = IsolationForest(n_estimators=500, contamination="auto", random_state=random_state, n_jobs=-1)
    model.fit(X)
    return pd.Series(-model.score_samples(X), index=X.index, name="score_isolation_forest")


def spe_scores(X: pd.DataFrame, n_components: int = 30, random_state: int = 0) -> pd.Series:
    """Statistique SPE : erreur de reconstruction après projection sur une ACP.

    Un lot dont les capteurs ne respectent plus les corrélations habituelles du
    procédé se reconstruit mal et obtient un SPE élevé.
    """
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state).fit(Z)
    recon = pca.inverse_transform(pca.transform(Z))
    spe = ((Z - recon) ** 2).sum(axis=1)
    return pd.Series(spe, index=X.index, name="score_spe")


# -------------------------------------------------------------------- supervisé


def make_classifier(random_state: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def best_threshold(y_true: np.ndarray, proba: np.ndarray, min_recall: float = 0.6) -> float:
    """Seuil qui maximise la précision sous contrainte de rappel minimal.

    En contrôle qualité, laisser passer un lot défaillant coûte plus cher qu'une
    fausse alerte : on impose un rappel plancher, puis on retient le seuil le
    plus précis parmi ceux qui le respectent.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    ok = recall >= min_recall
    if not ok.any():
        return 0.5
    return float(thresholds[ok][np.argmax(precision[ok])])


def train_supervised(
    df: pd.DataFrame, sensors: list[str], test_size: float = 0.25, random_state: int = 0
) -> tuple[RandomForestClassifier, SupervisedMetrics, pd.Series, pd.Series]:
    """Entraîne, évalue (validation croisée + découpage temporel) et explique le modèle.

    Returns
    -------
    model       : forêt entraînée sur l'ensemble des lots (prête à scorer de
                  nouveaux lots) ;
    metrics     : métriques des deux protocoles d'évaluation ;
    proba_oof   : probabilité de rejet hors pli (*out-of-fold*) pour chaque lot ;
    importances : importance moyenne des capteurs (diminution d'impureté).
    """
    X, y = df[sensors], df["defaut"]

    # 1. Validation croisée stratifiée : probabilités hors pli
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    proba_oof = pd.Series(
        cross_val_predict(make_classifier(random_state), X, y, cv=skf, method="predict_proba")[:, 1],
        index=df.index,
        name="proba_rejet",
    )
    seuil = best_threshold(y.to_numpy(), proba_oof.to_numpy())
    pred = proba_oof >= seuil
    tp = int((pred & (y == 1)).sum())

    # 2. Découpage temporel
    n_test = int(round(len(df) * test_size))
    tr, te = df.index[:-n_test], df.index[-n_test:]
    tmp = make_classifier(random_state).fit(X.loc[tr], y.loc[tr])
    proba_te = tmp.predict_proba(X.loc[te])[:, 1]

    metrics = SupervisedMetrics(
        auc_roc_cv=float(roc_auc_score(y, proba_oof)),
        average_precision_cv=float(average_precision_score(y, proba_oof)),
        auc_roc_temporel=float(roc_auc_score(y.loc[te], proba_te)),
        average_precision_temporel=float(average_precision_score(y.loc[te], proba_te)),
        seuil=seuil,
        precision=tp / max(int(pred.sum()), 1),
        rappel=tp / max(int(y.sum()), 1),
        taux_base=float(y.mean()),
    )

    # 3. Modèle final et importances
    model = make_classifier(random_state).fit(X, y)
    importances = pd.Series(model.feature_importances_, index=sensors, name="importance").sort_values(
        ascending=False
    )
    return model, metrics, proba_oof, importances
