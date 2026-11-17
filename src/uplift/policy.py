"""Ausspielungsregeln: zusätzliche Visits und Gewinn, wenn nur ein Teil der Nutzer angesprochen wird.

Alle Schätzungen sind mit dem Propensity Score e(x) invers gewichtet (IPW, Hajek-Form):
Im Criteo-Datensatz schwankt der Treatment-Anteil leicht mit den Merkmalen, die einfache
Differenz der Raten würde den Effekt deshalb überschätzen (Notebook 04, Tag 11).
"""
import numpy as np
import pandas as pd

from .metrics import _order


def _ipw_uplift(y, w, e):
    """IPW-Differenz der Visit-Raten (Treatment − Kontrolle) in einer Gruppe."""
    a1, a0 = w / e, (1 - w) / (1 - e)
    return (a1 * y).sum() / a1.sum() - (a0 * y).sum() / a0.sum()


def incremental_visits(y, w, e, target):
    """Geschätzte zusätzliche Visits, wenn genau die Nutzer mit target=True angesprochen werden
    (im Vergleich dazu, niemanden anzusprechen)."""
    y, w, e, target = (np.asarray(a) for a in (y, w, e, target))
    if target.sum() == 0:
        return 0.0
    m = target.astype(bool)
    return _ipw_uplift(y[m].astype(np.float64), w[m].astype(np.float64), e[m]) * m.sum()


def incremental_curve(y, w, score, e, n_points=1000):
    """Budgetkurve: zusätzliche Visits, wenn die Top-k nach Score angesprochen werden,
    für k auf einem Raster von n_points Anteilen. Rückgabe: DataFrame share, n_targeted, incremental."""
    order = _order(score)
    y, w, e = (np.asarray(a, dtype=np.float64)[order] for a in (y, w, e))
    a1, a0 = w / e, (1 - w) / (1 - e)
    k = np.unique(np.linspace(0, len(y), n_points + 1).astype(int))[1:] - 1   # Rasterpunkte
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = (np.cumsum(a1 * y) / np.cumsum(a1))[k]
        r0 = (np.cumsum(a0 * y) / np.cumsum(a0))[k]
    n = k + 1
    inc = np.nan_to_num((r1 - r0) * n)   # ganz oben kann eine Gruppe noch leer sein
    return pd.DataFrame({"share": np.r_[0.0, n / len(y)],
                         "n_targeted": np.r_[0, n],
                         "incremental": np.r_[0.0, inc]})
