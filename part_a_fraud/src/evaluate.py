"""Evaluation: PR-AUC and operating metrics with bootstrap confidence intervals.

Metrics (and why):
- PR-AUC (average precision): primary. Focuses on the rare class; ROC-AUC stays high
  even when most alerts are false, because it is dominated by the huge legit class.
- ROC-AUC: reported only for comparison with the literature.
- Alert budget: the review team can check ALERT_BUDGET of all transactions. We flag the
  top-scoring ones and report recall, precision and *fraud amount* caught.
- Precision target: threshold chosen on VALIDATION to reach TARGET_PRECISION, then
  applied unchanged to test (test is never used to pick anything).
- Accuracy @0.5: shown only to demonstrate the accuracy trap.

Uncertainty: the test set has only ~74 frauds, so every metric gets a 95% bootstrap CI
(resample test rows with replacement). The same resamples are used for every model
("paired"), so model differences can be compared directly.

Usage (from the repo root):
    python -m part_a_fraud.src.evaluate     # trains if needed, writes results table + PR curves
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from part_a_fraud.src.data import load, time_split, xy
from part_a_fraud.src.train import load_models, save, score, train_all

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ALERT_BUDGET = 0.002        # review team can look at 0.2% of transactions (~113 alerts on test)
TARGET_PRECISION = 0.80     # "at least 4 in 5 alerts are real fraud"
N_BOOT = 1000
SEED = 0


# ---------- point metrics: f(y, s, amount) -> float ---------------------------------
def pr_auc(y, s, a=None):
    return average_precision_score(y, s)


def roc_auc(y, s, a=None):
    return roc_auc_score(y, s)


def _top_k(s, budget):
    k = max(1, int(round(len(s) * budget)))
    flagged = np.zeros(len(s), dtype=bool)
    flagged[np.argsort(-s, kind="stable")[:k]] = True
    return flagged


def _recall(y, flagged):
    return (flagged & (y == 1)).sum() / max((y == 1).sum(), 1)


def _precision(y, flagged):
    return (flagged & (y == 1)).sum() / flagged.sum() if flagged.any() else 0.0


def _amount_caught(y, flagged, a):
    return a[flagged & (y == 1)].sum() / a[y == 1].sum()


def threshold_for_precision(y_val, s_val, target=TARGET_PRECISION) -> float:
    """Lowest threshold whose validation precision >= target (i.e. the most recall at that precision)."""
    precision, _, thresholds = precision_recall_curve(y_val, s_val)
    ok = np.where(precision[:-1] >= target)[0]
    return float(thresholds[ok[0]]) if len(ok) else float("inf")   # inf = target unreachable -> flag nothing


def metric_fns(threshold: float) -> dict:
    """All reported metrics as f(y, s, a). `threshold` is the validation-chosen one for this model."""
    return {
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        f"recall @{ALERT_BUDGET:.1%} alerts": lambda y, s, a: _recall(y, _top_k(s, ALERT_BUDGET)),
        f"precision @{ALERT_BUDGET:.1%} alerts": lambda y, s, a: _precision(y, _top_k(s, ALERT_BUDGET)),
        f"fraud amount caught @{ALERT_BUDGET:.1%} alerts": lambda y, s, a: _amount_caught(y, _top_k(s, ALERT_BUDGET), a),
        f"recall @val-thr (P>={TARGET_PRECISION:.0%})": lambda y, s, a: _recall(y, s >= threshold),
        f"precision @val-thr (P>={TARGET_PRECISION:.0%})": lambda y, s, a: _precision(y, s >= threshold),
        "accuracy @0.5": lambda y, s, a: ((s >= 0.5) == y).mean(),
    }


# ---------- bootstrap ------------------------------------------------------------------
def bootstrap_indices(n, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def evaluate(y, scores: dict, amount, thresholds: dict, n_boot=N_BOOT) -> pd.DataFrame:
    """Long table: model, metric, point estimate, 95% CI. Same bootstrap resamples for all models."""
    y, amount = np.asarray(y), np.asarray(amount)
    idx = bootstrap_indices(len(y), n_boot)
    rows = []
    for model, s in scores.items():
        s = np.asarray(s)
        for metric, f in metric_fns(thresholds.get(model, np.inf)).items():
            point = f(y, s, amount)
            if np.unique(s).size == 1:   # constant score (always_legit): nothing to resample, no CI
                lo = hi = np.nan
            else:
                lo, hi = np.percentile([f(y[i], s[i], amount[i]) for i in idx], [2.5, 97.5])
            rows.append({"model": model, "metric": metric, "value": point, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def paired_difference(y, s_a, s_b, f=pr_auc, n_boot=N_BOOT):
    """CI for metric(model A) - metric(model B) on the same resamples. CI excluding 0 => real difference."""
    y, s_a, s_b = map(np.asarray, (y, s_a, s_b))
    diffs = [f(y[i], s_a[i]) - f(y[i], s_b[i]) for i in bootstrap_indices(len(y), n_boot)]
    return f(y, s_a) - f(y, s_b), *np.percentile(diffs, [2.5, 97.5])


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids an extra dependency on `tabulate`)."""
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in row]) + " |" for i, row in zip(df.index, df.values)]
    return "\n".join(lines)


def results_table(long: pd.DataFrame) -> pd.DataFrame:
    """Wide, human-readable table: 'point [low, high]'."""
    def cell(r):
        d = 4 if r.metric.startswith("accuracy") else 3     # accuracy needs 4 decimals to show the trap
        return f"{r.value:.{d}f} [{r.ci_low:.{d}f}, {r.ci_high:.{d}f}]" if np.isfinite(r.ci_low) else f"{r.value:.{d}f}"
    fmt = long.assign(cell=long.apply(cell, axis=1))
    return fmt.pivot(index="metric", columns="model", values="cell").reindex(long["metric"].unique())


# ---------- plots ----------------------------------------------------------------------
def plot_pr_curves(y, scores: dict, ax=None, title="Precision-recall curves (test)"):
    ax = ax or plt.subplots(figsize=(6, 4.5))[1]
    for name, s in scores.items():
        if np.unique(s).size == 1:
            ax.axhline(np.mean(y), ls=":", color="gray", label=f"{name} (= fraud rate {np.mean(y):.2%})")
            continue
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=f"{name} (PR-AUC {average_precision_score(y, s):.3f})")
    ax.set(xlabel="recall (share of frauds caught)", ylabel="precision (share of alerts that are fraud)",
           xlim=(0, 1), ylim=(0, 1.02), title=title)
    ax.legend(loc="lower left")
    return ax


# ---------- end-to-end -----------------------------------------------------------------
def run(retrain: bool = False):
    train, val, test = time_split(load())
    models = load_models()
    if retrain or not models:
        models = train_all(train, val)
        save(models)
    X_val, y_val = xy(val)
    X_te, y_te = xy(test)
    val_scores = {n: score(m, X_val) for n, m in models.items()}
    test_scores = {n: score(m, X_te) for n, m in models.items()}
    thresholds = {n: threshold_for_precision(y_val, s) for n, s in val_scores.items()}
    long = evaluate(y_te, test_scores, test["Amount"], thresholds)
    return dict(models=models, val=val, test=test, val_scores=val_scores, test_scores=test_scores,
                thresholds=thresholds, long=long, table=results_table(long))


if __name__ == "__main__":
    out = run(retrain=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    table = out["table"]
    print(table.to_string())
    (RESULTS_DIR / "results_test.md").write_text(
        f"Test set ({len(out['test']):,} rows, {int(out['test']['Class'].sum())} frauds). "
        f"Point estimate [95% bootstrap CI, {N_BOOT} resamples].\n\n" + to_markdown(table) + "\n", encoding="utf-8")
    y_te = out["test"]["Class"]
    d, lo, hi = paired_difference(y_te, out["test_scores"]["lightgbm"], out["test_scores"]["logreg"])
    print(f"\nPR-AUC lightgbm - logreg: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    ax = plot_pr_curves(y_te, out["test_scores"])
    ax.figure.tight_layout(); ax.figure.savefig(RESULTS_DIR / "pr_curves_test.png", dpi=120)
    print(f"Wrote {RESULTS_DIR / 'results_test.md'} and pr_curves_test.png")
