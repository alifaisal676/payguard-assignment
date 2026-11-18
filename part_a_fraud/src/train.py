"""Train the three models on the time-based train split.

- always_legit : predicts "legit" for everything -> shows the accuracy trap
- logreg       : Logistic Regression, class_weight="balanced", log(Amount) + scaling
- lightgbm     : gradient-boosted trees, early stopping on validation PR-AUC
                 (no class reweighting; see train_lightgbm for why)

Deliberately no grid search, ensembles or deep learning: with ~340 training frauds,
heavy tuning mostly fits noise in the validation set.

Usage (from the repo root):
    python -m part_a_fraud.src.train        # trains and saves models to part_a_fraud/models/
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from part_a_fraud.src.data import load, time_split, xy
from part_a_fraud.src.features import lr_preprocessor

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
SEED = 42


def train_always_legit(X_tr, y_tr):
    return DummyClassifier(strategy="constant", constant=0).fit(X_tr, y_tr)


def train_logreg(X_tr, y_tr):
    model = Pipeline([
        ("prep", lr_preprocessor()),
        # "balanced" re-weights each fraud by ~#legit/#fraud, so the loss isn't dominated by legit rows
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    return model.fit(X_tr, y_tr)


def train_lightgbm(X_tr, y_tr, X_val, y_val, scale_pos_weight: float = 1.0):
    """scale_pos_weight=1 (no reweighting) is a deliberate, tested choice.

    Validation PR-AUC: spw=1 -> 0.777, spw=sqrt(ratio)~22 -> 0.779 (same within noise),
    spw=ratio~497 -> ~0.31 (and with `Time` included it collapsed to ~0.01 after 3 trees):
    weighting every fraud ~500x lets a handful of rows dominate. Unweighted scores are also
    closer to real probabilities, which the cost-based threshold needs. Notebook, section 2.1.
    """
    model = lgb.LGBMClassifier(
        n_estimators=3000,            # upper bound; early stopping picks the real number
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,         # don't build leaves around a handful of frauds
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        metric="average_precision",   # early-stop on PR-AUC, the metric we care about
        random_state=SEED,
        verbose=-1,
    )
    # Patience 200: val PR-AUC on 57 frauds is noisy, so don't stop on a small dip.
    model.fit(X_tr, y_tr, eval_X=(X_val,), eval_y=(y_val,),
              callbacks=[lgb.early_stopping(200, verbose=False)])
    return model


def train_all(train, val) -> dict:
    X_tr, y_tr = xy(train)
    X_val, y_val = xy(val)
    return {
        "always_legit": train_always_legit(X_tr, y_tr),
        "logreg": train_logreg(X_tr, y_tr),
        "lightgbm": train_lightgbm(X_tr, y_tr, X_val, y_val),
    }


def score(model, X) -> np.ndarray:
    """Fraud score in [0, 1] (probability of class 1)."""
    return model.predict_proba(X)[:, 1]


def save(models: dict) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    for name, m in models.items():
        joblib.dump(m, MODEL_DIR / f"{name}.joblib")


def load_models() -> dict:
    return {p.stem: joblib.load(p) for p in sorted(MODEL_DIR.glob("*.joblib"))}


if __name__ == "__main__":
    train, val, _ = time_split(load())
    models = train_all(train, val)
    save(models)
    print(f"Saved {list(models)} to {MODEL_DIR}")
    print(f"LightGBM stopped at {models['lightgbm'].best_iteration_} trees")
