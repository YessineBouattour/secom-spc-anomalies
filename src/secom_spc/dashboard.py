"""Tableau de bord HTML interactif (Plotly.js chargé côté navigateur).

Le fichier produit est autonome : les données sont embarquées en JSON, seule la
bibliothèque Plotly.js est récupérée depuis un CDN à l'ouverture. Aucune
dépendance Python de visualisation n'est nécessaire.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .spc import CPK_TARGET, imr_limits, western_electric

_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Supervision SPC – ligne de test</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { --bleu: #1f3a6e; --rouge: #c0392b; --gris: #6b7280; }
  body { font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin: 0; background: #f5f6f8; color: #1a1a1a; }
  header { background: var(--bleu); color: #fff; padding: 18px 28px; }
  header h1 { margin: 0; font-size: 20px; font-weight: 600; }
  header p { margin: 4px 0 0; opacity: .85; font-size: 13px; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; padding: 20px 28px 0; }
  .kpi { background: #fff; border-radius: 8px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .kpi .label { color: var(--gris); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
  .kpi .value { font-size: 26px; font-weight: 600; margin-top: 4px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 20px 28px 28px; }
  .card { background: #fff; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .card h2 { font-size: 14px; margin: 4px 8px 8px; color: var(--bleu); font-weight: 600; }
  select { font-size: 13px; padding: 4px 8px; margin-left: 8px; }
  @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>Supervision statistique et détection d'anomalies – ligne de test</h1>
  <p>Données SECOM (UCI) · __N_LOTS__ lots · __N_CAPTEURS__ capteurs après nettoyage · période __PERIODE__</p>
</header>

<div class="kpis">
  <div class="kpi"><div class="label">Taux de rebut</div><div class="value">__TAUX_REBUT__</div></div>
  <div class="kpi"><div class="label">Capteurs Cpk &lt; 1,33</div><div class="value">__N_CPK_FAIBLE__</div></div>
  <div class="kpi"><div class="label">Épisodes d'alerte SPC</div><div class="value">__N_ALERTES__</div></div>
  <div class="kpi"><div class="label">AUC ROC – prédiction du rejet</div><div class="value">__AUC__</div></div>
  <div class="kpi"><div class="label">Rappel / précision au seuil</div><div class="value">__RAPPEL__ / __PRECISION__</div></div>
</div>

<div class="grid">
  <div class="card">
    <h2>Carte de contrôle I – capteur
      <select id="sel"></select>
    </h2>
    <div id="cc" style="height:360px"></div>
  </div>
  <div class="card"><h2>Cpk en phase de surveillance – 20 capteurs les moins capables</h2><div id="cpk" style="height:360px"></div></div>
  <div class="card"><h2>Probabilité de rejet prédite (hors pli) dans le temps</h2><div id="iso" style="height:360px"></div></div>
  <div class="card"><h2>Capteurs les plus explicatifs du rejet (forêt aléatoire)</h2><div id="imp" style="height:360px"></div></div>
</div>

<script>
const D = __DATA__;
const bleu = "#1f3a6e", rouge = "#c0392b", gris = "#9ca3af";

function carteControle(name) {
  const s = D.cartes[name];
  const colors = s.alerte.map(a => a ? rouge : bleu);
  const traces = [
    { x: D.t, y: s.y, mode: "lines+markers", name: name, line: { color: "#94a3b8", width: 1 },
      marker: { color: colors, size: 5 } },
    { x: [D.t[0], D.t.at(-1)], y: [s.cl, s.cl], mode: "lines", name: "Centre (phase I)", line: { color: "#16a34a", dash: "dash" } },
    { x: [D.t_ref, D.t_ref], y: [s.lcl, s.ucl], mode: "lines", name: "Fin phase I", line: { color: "#f59e0b", width: 1 } },
    { x: [D.t[0], D.t.at(-1)], y: [s.ucl, s.ucl], mode: "lines", name: "UCL (+3σ)", line: { color: rouge, dash: "dot" } },
    { x: [D.t[0], D.t.at(-1)], y: [s.lcl, s.lcl], mode: "lines", name: "LCL (−3σ)", line: { color: rouge, dash: "dot" } },
  ];
  Plotly.react("cc", traces, { margin: { t: 10, r: 10, l: 50, b: 40 }, legend: { orientation: "h" },
    yaxis: { title: "mesure" }, title: { text: `Cpk phase II = ${s.cpk.toFixed(2)} · ${s.n_episodes} épisodes d'alerte`, x: 0.02, y: 0.98, font: { size: 12 } } },
    { responsive: true, displaylogo: false });
}

const sel = document.getElementById("sel");
Object.keys(D.cartes).forEach(n => { const o = document.createElement("option"); o.value = n; o.textContent = n; sel.appendChild(o); });
sel.onchange = () => carteControle(sel.value);
carteControle(sel.value);

Plotly.newPlot("cpk", [{ x: D.cpk.capteur, y: D.cpk.cpk, type: "bar",
  marker: { color: D.cpk.cpk.map(v => v < __CPK_TARGET__ ? rouge : bleu) } }],
  { margin: { t: 10, r: 10, l: 50, b: 90 }, yaxis: { title: "Cpk" },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: __CPK_TARGET__, y1: __CPK_TARGET__, line: { color: "#16a34a", dash: "dash" } }],
    annotations: [{ xref: "paper", x: 1, y: __CPK_TARGET__, text: "cible 1,33", showarrow: false, yshift: 10, font: { color: "#16a34a" } }] },
  { responsive: true, displaylogo: false });

Plotly.newPlot("iso", [
  { x: D.t.filter((_, i) => !D.defaut[i]), y: D.score.filter((_, i) => !D.defaut[i]), mode: "markers", name: "Lot conforme", marker: { color: gris, size: 5 } },
  { x: D.t.filter((_, i) => D.defaut[i]), y: D.score.filter((_, i) => D.defaut[i]), mode: "markers", name: "Lot rejeté", marker: { color: rouge, size: 7 } },
], { margin: { t: 10, r: 10, l: 50, b: 40 }, legend: { orientation: "h" }, yaxis: { title: "probabilité de rejet" }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: D.seuil, y1: D.seuil, line: { color: "#f59e0b", dash: "dash" } }], annotations: [{ xref: "paper", x: 1, y: D.seuil, text: "seuil d'alerte", showarrow: false, yshift: 10, font: { color: "#f59e0b" } }] },
  { responsive: true, displaylogo: false });

Plotly.newPlot("imp", [{ x: D.importance.valeur.slice().reverse(), y: D.importance.capteur.slice().reverse(),
  type: "bar", orientation: "h", marker: { color: bleu } }],
  { margin: { t: 10, r: 10, l: 90, b: 40 }, xaxis: { title: "importance (diminution moyenne d'impureté)" } },
  { responsive: true, displaylogo: false });
</script>
</body>
</html>
"""


def build_dashboard(
    df: pd.DataFrame,
    resume: pd.DataFrame,
    episodes: pd.DataFrame,
    proba: pd.Series,
    importances: pd.Series,
    metrics: dict,
    out_path: str | Path,
    ref_frac: float = 0.30,
    n_cartes: int = 15,
    n_cpk: int = 20,
    n_importance: int = 15,
) -> Path:
    """Assemble le HTML du tableau de bord et l'écrit dans ``out_path``."""
    t = df["horodatage"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    n_ref = int(round(len(df) * ref_frac))

    cartes = {}
    for c in resume["capteur"].head(n_cartes):
        x = df[c]
        lim = imr_limits(x.iloc[:n_ref])
        rules = western_electric(x, lim)
        row = resume.loc[resume["capteur"] == c].iloc[0]
        cartes[c] = {
            "y": [round(float(v), 5) for v in x],
            "alerte": rules["alerte"].astype(int).tolist(),
            "cl": lim.center,
            "ucl": lim.ucl,
            "lcl": lim.lcl,
            "cpk": float(row["cpk"]),
            "n_episodes": int(row["n_episodes"]),
        }

    top_cpk = resume.head(n_cpk)
    top_imp = importances.head(n_importance)
    data = {
        "t": t,
        "t_ref": t[n_ref - 1],
        "defaut": df["defaut"].astype(int).tolist(),
        "score": [round(float(v), 5) for v in proba.loc[df.index]],
        "seuil": round(float(metrics["seuil"]), 4),
        "cartes": cartes,
        "cpk": {"capteur": top_cpk["capteur"].tolist(), "cpk": [round(float(v), 4) for v in top_cpk["cpk"]]},
        "importance": {"capteur": top_imp.index.tolist(), "valeur": [round(float(v), 5) for v in top_imp]},
    }

    periode = f"{df['horodatage'].min():%d/%m/%Y} → {df['horodatage'].max():%d/%m/%Y}"
    html = (
        _TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
        .replace("__N_LOTS__", f"{len(df):,}".replace(",", " "))
        .replace("__N_CAPTEURS__", str(len([c for c in df.columns if c.startswith("capteur_")])))
        .replace("__PERIODE__", periode)
        .replace("__TAUX_REBUT__", f"{df['defaut'].mean():.1%}".replace(".", ","))
        .replace("__N_CPK_FAIBLE__", str(int((resume["cpk"] < CPK_TARGET).sum())))
        .replace("__N_ALERTES__", f"{len(episodes):,}".replace(",", " "))
        .replace("__AUC__", f"{metrics['auc_roc_cv']:.2f}".replace(".", ","))
        .replace("__RAPPEL__", f"{metrics['rappel']:.0%}")
        .replace("__PRECISION__", f"{metrics['precision']:.0%}")
        .replace("__CPK_TARGET__", str(CPK_TARGET))
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
