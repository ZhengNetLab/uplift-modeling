"""Evaluationsmetriken für Uplift-Modelle: Qini-Kurve, Uplift-Kurve, Uplift@k."""
import numpy as np


def _cumulative(y, w, score, n_points=1000):
    """Absteigend nach Score sortieren, kumulierte Größen an n_points Stellen zurückgeben."""
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    order = np.argsort(-np.asarray(score), kind="stable")
    y, w = y[order], w[order]
    n_t, n_c = np.cumsum(w), np.cumsum(1 - w)
    y_t, y_c = np.cumsum(y * w), np.cumsum(y * (1 - w))
    # Auswertung nur an gleichmäßig verteilten Anteilen (spart Speicher beim Plotten)
    k = np.unique(np.linspace(0, len(y), n_points + 1).astype(int))[1:] - 1
    frac = np.concatenate([[0.0], (k + 1) / len(y)])
    pad = lambda a: np.concatenate([[0.0], a[k]])
    return frac, pad(n_t), pad(n_c), pad(y_t), pad(y_c)


def qini_curve(y, w, score, n_points=1000):
    """Qini: zusätzliche Visits unter den Behandelten in den Top-k;
    Kontrollgruppe auf die Größe der Treatment-Gruppe skaliert."""
    frac, n_t, n_c, y_t, y_c = _cumulative(y, w, score, n_points)
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


def qini_auc(y, w, score, n_points=1000):
    return _area_vs_random(*qini_curve(y, w, score, n_points))


def auuc(y, w, score, n_points=1000):
    return _area_vs_random(*uplift_curve(y, w, score, n_points))


def uplift_at_k(y, w, score, k=0.3):
    """Rate_T − Rate_C unter den Top-k (Anteil) nach Score."""
    y, w, score = np.asarray(y), np.asarray(w), np.asarray(score)
    top = np.argsort(-score, kind="stable")[: int(np.ceil(k * len(y)))]
    yt, wt = y[top], w[top]
    return yt[wt == 1].mean() - yt[wt == 0].mean()


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