"""Evaluationsmetriken für Uplift-Modelle: Qini-Kurve, Uplift-Kurve, Uplift@k,
Bootstrap-Konfidenzintervalle und Kalibrierung."""
import numpy as np


def _order(score, seed=0):
    """Indizes absteigend nach Score; Gleichstände zufällig (reproduzierbar) aufgelöst.
    Ohne das Mischen entscheidet bei gleichem Score die Zeilenposition – und die
    Zeilenreihenfolge im Criteo-Datensatz hängt stark mit treatment und visit zusammen."""
    score = np.asarray(score)
    perm = np.random.default_rng(seed).permutation(len(score))
    return perm[np.argsort(-score[perm], kind="stable")]


def _cumulative(y, w, score, n_points=1000, weight=None, order=None):
    """Absteigend nach Score sortieren, kumulierte Größen an n_points Stellen zurückgeben.
    weight: optionale Gewichte je Zeile (Bootstrap); order: vorab berechnete Sortierung."""
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    v = np.ones_like(y) if weight is None else np.asarray(weight, dtype=np.float64)
    order = _order(score) if order is None else order
    y, w, v = y[order], w[order], v[order]
    n_t, n_c = np.cumsum(v * w), np.cumsum(v * (1 - w))
    y_t, y_c = np.cumsum(v * y * w), np.cumsum(v * y * (1 - w))
    # Auswertung nur an gleichmäßig verteilten Anteilen (spart Speicher beim Plotten)
    k = np.unique(np.linspace(0, len(y), n_points + 1).astype(int))[1:] - 1
    frac = np.concatenate([[0.0], np.cumsum(v)[k] / v.sum()])
    pad = lambda a: np.concatenate([[0.0], a[k]])
    return frac, pad(n_t), pad(n_c), pad(y_t), pad(y_c)


def qini_curve(y, w, score, n_points=1000, weight=None, order=None):
    """Qini: zusätzliche Visits unter den Behandelten in den Top-k;
    Kontrollgruppe auf die Größe der Treatment-Gruppe skaliert."""
    frac, n_t, n_c, y_t, y_c = _cumulative(y, w, score, n_points, weight, order)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = y_t - np.where(n_c > 0, y_c * n_t / n_c, 0.0)
    return frac, q


def uplift_curve(y, w, score, n_points=1000):
    """Uplift-Kurve: (Rate_T − Rate_C) in den Top-k × Größe der Top-k."""
    frac, n_t, n_c, y_t, y_c = _cumulative(y, w, score, n_points)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where((n_t > 0) & (n_c > 0), (y_t / n_t - y_c / n_c) * (n_t + n_c), 0.0)
    return frac, u


def _area_vs_random(frac, curve):
    """Fläche zwischen Kurve und Zufallsgerade, normiert mit dem Gesamtwert.
    0 = zufällig, > 0 besser, < 0 schlechter als zufällig."""
    random = curve[-1] * frac
    return np.trapezoid(curve - random, frac) / curve[-1]


def qini_auc(y, w, score, n_points=1000, weight=None, order=None):
    return _area_vs_random(*qini_curve(y, w, score, n_points, weight, order))


def auuc(y, w, score, n_points=1000):
    return _area_vs_random(*uplift_curve(y, w, score, n_points))


def uplift_at_k(y, w, score, k=0.3):
    """Rate_T − Rate_C unter den Top-k (Anteil) nach Score."""
    y, w, score = np.asarray(y), np.asarray(w), np.asarray(score)
    top = _order(score)[: int(np.ceil(k * len(y)))]
    yt, wt = y[top], w[top]
    return yt[wt == 1].mean() - yt[wt == 0].mean()


def uplift_summary(y, w, score, ks=(0.1, 0.2, 0.3)):
    """Kennzahlen eines Modells auf einen Blick: Qini-AUC, AUUC und Uplift@k."""
    row = {"qini_auc": qini_auc(y, w, score), "auuc": auuc(y, w, score)}
    for k in ks:
        row[f"uplift@{int(k * 100)}%"] = uplift_at_k(y, w, score, k)
    return row


def bootstrap_qini(y, w, scores, n_boot=200, seed=0):
    """Qini-AUC für mehrere Modelle auf n_boot Bootstrap-Stichproben des Testsets.

    Poisson-Bootstrap: statt Zeilen neu zu ziehen bekommt jede Zeile ein Gewicht ~ Poisson(1)
    (≈ wie oft sie in einer Ziehung mit Zurücklegen vorkäme). So muss jedes Modell nur
    einmal sortiert werden. Alle Modelle bekommen in einer Runde dieselben Gewichte
    → gepaarter Bootstrap, Differenzen zwischen Modellen sind direkt auswertbar.
    Rückgabe: (DataFrame n_boot × Modelle mit Qini-AUC,
               dict Modell → Array n_boot × Kurvenpunkte mit den Qini-Kurven, Anteile in `frac`)."""
    import pandas as pd
    orders = {m: _order(s) for m, s in scores.items()}
    rng = np.random.default_rng(seed)
    aucs = np.empty((n_boot, len(scores)))
    curves = {m: [] for m in scores}
    for b in range(n_boot):
        v = rng.poisson(1.0, size=len(y))
        for j, m in enumerate(scores):
            frac, q = qini_curve(y, w, None, weight=v, order=orders[m])
            aucs[b, j] = _area_vs_random(frac, q)
            # auf das ungewichtete Raster interpolieren, damit alle Runden vergleichbar sind
            curves[m].append(np.interp(np.linspace(0, 1, 1001), frac, q))
    return pd.DataFrame(aucs, columns=list(scores)), {m: np.array(c) for m, c in curves.items()}


def calibration_table(y, w, score, n_bins=10):
    """Nach vorhergesagtem Uplift in n_bins gleich große Gruppen teilen (0 = höchster Score)
    und je Gruppe den mittleren vorhergesagten mit dem beobachteten Uplift vergleichen.
    Beobachtet = Rate_T − Rate_C, 95%-KI über den Standardfehler einer Anteilsdifferenz."""
    import pandas as pd
    y, w, score = np.asarray(y), np.asarray(w), np.asarray(score, dtype=np.float64)
    order = _order(score)
    group = np.empty(len(y), dtype=np.int64)
    group[order] = np.arange(len(y)) * n_bins // len(y)
    df = pd.DataFrame({"y": y, "w": w, "score": score, "group": group})
    g = df.groupby("group")
    n1, n0 = g["w"].sum(), g["w"].count() - g["w"].sum()
    p1 = df[df.w == 1].groupby("group")["y"].mean()
    p0 = df[df.w == 0].groupby("group")["y"].mean()
    tab = pd.DataFrame({"predicted": g["score"].mean(), "observed": p1 - p0,
                        "se": np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0),
                        "share_treated": n1 / (n1 + n0)})
    tab["ci_low"] = tab.observed - 1.96 * tab.se
    tab["ci_high"] = tab.observed + 1.96 * tab.se
    return tab


def plot_qini(curves, ax=None):
    """curves: dict {Name: (frac, q)}; zeichnet zusätzlich die Zufallsgerade."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(7, 5))[1]
    for name, (frac, q) in curves.items():
        ax.plot(frac, q, label=name)
    frac, q = next(iter(curves.values()))
    ax.plot([0, 1], [0, q[-1]], "k--", label="random")
    ax.set_xlabel("Anteil der Zielgruppe (nach Score sortiert)")
    ax.set_ylabel("zusätzliche Visits (Qini)")
    ax.legend()
    return ax