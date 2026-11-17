"""Modell-Bausteine für Notebook 03: LightGBM mit Early Stopping, Meta-Learner, EconML-Modelle.

Alle Modelle lernen P(visit) bzw. den Uplift aus den 12 Merkmalen; Hyperparameter werden
ausschließlich auf dem Validierungsset gewählt.
"""
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping

from .metrics import qini_auc

LGB_PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                  min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, verbose=-1)


def fit_lgb(X_tr, y_tr, X_va, y_va, regression=False, verbose=True):
    """LightGBM mit Early Stopping auf dem Validierungsset
    (Klassifikation: Logloss, Regression: L2). Höchstens 2000 Bäume."""
    m = (LGBMRegressor if regression else LGBMClassifier)(**LGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[early_stopping(50, verbose=False)])
    if verbose:
        print(f"  beste Iteration: {m.best_iteration_}")
    return m


def p_visit(model, X):
    """Vorhergesagte Visit-Wahrscheinlichkeit eines Klassifikators."""
    return model.predict_proba(X)[:, 1]


# --- S-, T-, X-Learner -------------------------------------------------------------------------

def fit_s_learner(tr, va, features, outcome="visit", treatment="treatment"):
    """Ein Modell für P(visit | X, W) mit dem Treatment als zusätzlichem Merkmal."""
    cols = features + [treatment]
    return fit_lgb(tr[cols], tr[outcome], va[cols], va[outcome])


def predict_s_learner(model, X, treatment="treatment"):
    """Uplift = Vorhersage mit Werbung − Vorhersage ohne Werbung."""
    return p_visit(model, X.assign(**{treatment: 1})) - p_visit(model, X.assign(**{treatment: 0}))


def fit_t_learner(tr, va, features, outcome="visit", treatment="treatment"):
    """Je ein Modell für Treatment (mu1) und Kontrolle (mu0). Rückgabe: (mu0, mu1)."""
    t, c = tr[treatment] == 1, tr[treatment] == 0
    vt, vc = va[treatment] == 1, va[treatment] == 0
    mu1 = fit_lgb(tr.loc[t, features], tr.loc[t, outcome], va.loc[vt, features], va.loc[vt, outcome])
    mu0 = fit_lgb(tr.loc[c, features], tr.loc[c, outcome], va.loc[vc, features], va.loc[vc, outcome])
    return mu0, mu1


def predict_t_learner(mu0, mu1, X):
    return p_visit(mu1, X) - p_visit(mu0, X)


def x_learner_imputed_effects(d, mu0, mu1, features, outcome="visit", treatment="treatment"):
    """Imputierte individuelle Effekte: Behandelte Y − mu0(x), Kontrolle mu1(x) − Y.
    Rückgabe: (D1 für die Behandelten, D0 für die Kontrolle) als Series."""
    t, c = d[d[treatment] == 1], d[d[treatment] == 0]
    return t[outcome] - p_visit(mu0, t[features]), p_visit(mu1, c[features]) - c[outcome]


def fit_x_learner(tr, va, mu0, mu1, features, outcome="visit", treatment="treatment"):
    """Stufe 2 des X-Learners: je ein Regressionsmodell für D1 (tau1) und D0 (tau0).
    Rückgabe: (tau0, tau1)."""
    d1_tr, d0_tr = x_learner_imputed_effects(tr, mu0, mu1, features, outcome, treatment)
    d1_va, d0_va = x_learner_imputed_effects(va, mu0, mu1, features, outcome, treatment)
    tau1 = fit_lgb(tr.loc[d1_tr.index, features], d1_tr, va.loc[d1_va.index, features], d1_va, regression=True)
    tau0 = fit_lgb(tr.loc[d0_tr.index, features], d0_tr, va.loc[d0_va.index, features], d0_va, regression=True)
    return tau0, tau1


def predict_x_learner(tau0, tau1, e, X):
    """Stufe 3: e(x)·tau0 + (1 − e(x))·tau1 – das Modell der großen Gruppe bekommt mehr Gewicht."""
    return e * tau0.predict(X) + (1 - e) * tau1.predict(X)


# --- EconML ------------------------------------------------------------------------------------

def make_dr_learner(prop_trees, outcome_trees=300, random_state=42):
    """DR-Learner mit LightGBM als Nuisance- und Endmodell. EconML hat kein Early Stopping →
    feste Baumzahlen; das Endmodell ist stark regularisiert, weil das Pseudo-Outcome verrauscht ist."""
    from econml.dr import DRLearner
    return DRLearner(
        model_propensity=LGBMClassifier(**{**LGB_PARAMS, "n_estimators": prop_trees}),
        model_regression=LGBMClassifier(**{**LGB_PARAMS, "n_estimators": outcome_trees}),
        model_final=LGBMRegressor(**{**LGB_PARAMS, "n_estimators": 600, "learning_rate": 0.02,
                                     "num_leaves": 15, "min_child_samples": 2000}),
        discrete_outcome=True, cv=3, random_state=random_state)


def make_causal_forest(prop_trees, outcome_trees=300, min_samples_leaf=500, random_state=42):
    """CausalForestDML mit LightGBM für die Residualisierung von Y und W."""
    from econml.dml import CausalForestDML
    return CausalForestDML(
        model_y=LGBMClassifier(**{**LGB_PARAMS, "n_estimators": outcome_trees}),
        model_t=LGBMClassifier(**{**LGB_PARAMS, "n_estimators": prop_trees}),
        discrete_outcome=True, discrete_treatment=True, cv=3,
        n_estimators=200, min_samples_leaf=min_samples_leaf, max_samples=0.45,
        random_state=random_state)


def select_by_qini(predict, candidates, y_va, w_va):
    """Wählt die Einstellung mit der höchsten Qini-AUC auf dem Validierungsset.
    predict(c) liefert für Kandidat c die Uplift-Vorhersagen auf dem Validierungsset.
    Rückgabe: (bester Kandidat, dict Kandidat → Qini)."""
    scores = {c: qini_auc(y_va, w_va, predict(c)) for c in candidates}
    return max(scores, key=scores.get), scores
