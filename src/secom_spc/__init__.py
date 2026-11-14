"""Supervision statistique (SPC) et détection d'anomalies sur des données de tests industriels.

Le package est découpé en quatre étapes indépendantes :

- ``io``        : lecture des fichiers SECOM et écriture en base SQLite ;
- ``cleaning``  : nettoyage des mesures capteurs ;
- ``spc``       : cartes de contrôle I-MR, indices de capabilité et règles d'alerte ;
- ``anomaly``   : détection non supervisée et prédiction supervisée des lots défaillants ;
- ``dashboard`` : génération d'un tableau de bord HTML interactif.
"""

__version__ = "1.0.0"
