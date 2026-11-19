# PayGuard: feasibility build

Two prototypes and a production memo for PayGuard (fictional digital payments company):

- **Part A: card fraud detection.** Is ML-based fraud detection viable, and what would production need?
- **Part B: LLM complaint triage.** Can an LLM route complaints with a summary, without hallucinating or misrouting sensitive cases?
- **Part C: [memo.md](memo.md).** Production design for the complaint triage pipeline, to the CTO.

How AI tools were used: [AI_USAGE.md](AI_USAGE.md).

## Setup (Python 3.12)

```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                  # add GROQ_API_KEY (free: console.groq.com/keys)
```

Run everything from the repo root:

| Step | Command |
|---|---|
| Part A data | Download `creditcard.csv` from [Kaggle (mlg-ulb/creditcardfraud)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into `part_a_fraud/data/`, **or** let the code fetch it from OpenML (ID 1597, slower): `python -m part_a_fraud.src.data` |
| Part A models + results | `python -m part_a_fraud.src.train` then `python -m part_a_fraud.src.evaluate` (and `python -m part_a_fraud.src.cost`) |
| Part A full analysis | open `part_a_fraud/notebooks/fraud_analysis.ipynb` (outputs are saved; re-run: *Run All*) |
| Part B data | already committed (`part_b_triage/data/complaints_raw.json`); to re-create: `python -m part_b_triage.src.fetch_data` |
| Part B triage | `python -m part_b_triage.src.pipeline --set eval` (also `--set injection`, `--set all`, `--model ...`) |
| Part B labels | `python -m part_b_triage.src.labeling label` (and `... summaries`) |
| Part B evaluation | `python -m part_b_triage.src.evaluate` → `part_b_triage/results/triage_eval.md` |

---

## Part A: Fraud detection

**Data.** ULB credit card dataset: 284,807 transactions over 2 days, 492 frauds (0.17%). `V1`–`V28` are anonymised PCA components; only `Amount` and `Time` are raw. The ~150 MB file is not committed. `data.py` checks row count, fraud count and columns on every load.

### Method

| Decision | Why |
|---|---|
| **Drop 1,081 exact duplicates before splitting** | A shuffled split would have spread 494 of the 773 duplicate groups across train and test, scoring the model on rows it trained on. Our time split happens to keep them together, so dedup is a safeguard. |
| **Chronological split 60/20/20** (train / validation / test), never shuffled | A model must never learn from the future. The EDA also showed that both high-fraud nights fall in training, while validation and test are daytime. That regime shift is real and a shuffle would hide it. |
| **PR-AUC as primary metric**, plus recall at an alert budget, recall at a validation-chosen precision, and **money caught** | 99.87% accuracy is achievable by flagging nothing. ROC-AUC ranked LogReg ≥ LightGBM (0.982 vs. 0.978) while PR-AUC separated them (0.744 vs. 0.802). |
| **95% bootstrap CIs** (paired across models) | Test has only 74 frauds. |
| Models: always-legit → Logistic Regression (`class_weight="balanced"`, log-Amount, scaling in a Pipeline) → **LightGBM** (early stopping on validation PR-AUC) | No grid search, ensembles or deep learning: ~340 training frauds don't support it. |
| **LightGBM `scale_pos_weight=1`**, not the planned ≈497 | Full reweighting made training unstable (stopped at 3 trees; validation PR-AUC ~0.31 vs ~0.78). Unweighted scores are also roughly calibrated, which the cost step needs. |
| **`Time` removed as a feature** | SHAP showed the model using `Time` as "later = safer", pushing every test row towards legit. That's an artefact of this 2-day file, and in production `Time` keeps growing beyond the training range. |

### Results (test set: 56,746 transactions, 74 frauds)

| Metric | Always legit | Logistic Regression | **LightGBM** |
|---|---|---|---|
| **PR-AUC** | 0.001 | 0.744 [0.638, 0.855] | **0.802** [0.715, 0.883] |
| ROC-AUC | 0.500 | 0.982 [0.966, 0.994] | 0.978 [0.960, 0.992] |
| Recall when reviewing the top 0.2% (113 alerts) | 0 | 0.797 [0.708, 0.887] | 0.784 [0.696, 0.877] |
| Precision of those 113 alerts | 0 | 0.522 | 0.513 |
| Share of fraud **money** caught at that budget | 0 | 0.690 [0.417, 0.948] | 0.691 [0.417, 0.948] |
| Recall at the threshold chosen on val for ≥80% precision | 0 | 0.676 (precision 0.943) | **0.757** (precision 0.875) |
| Accuracy @0.5 | **0.9987** | 0.9717 | 0.9995 |

LightGBM minus LogReg PR-AUC: **+0.058, 95% CI [−0.010, +0.123]**. LightGBM is *probably* better, not conclusively. Full table: [results_test.md](part_a_fraud/results/results_test.md); PR curves: [pr_curves_test.png](part_a_fraud/results/pr_curves_test.png).

### Threshold: chosen by cost, on validation

Assumed costs (EUR, replaceable in `cost.py`): missed fraud = amount + €15 chargeback fee + €10 handling; every alert = €4 review; false alarm = +€10 customer friction.

- **Calibration first:** LightGBM's scores are roughly calibrated (validation: 46 expected vs 57 actual frauds). LogReg's are not (≈5,000 expected), because balanced class weights inflate them, so only LightGBM's scores are fed into cost formulas.
- **Cost-optimal threshold on validation** flags 108 test transactions and catches 58 of 74 frauds. Total test cost goes from **€9,578 with no model to €3,721**. This lands close to the "review team checks 0.2%" budget, so two independent framings agree.
- **Amount-aware rule** (flag when p·(amount+€25) > €4 + (1−p)·€10) reaches €3,436 with 7 false alarms instead of 50. The saving is **not statistically reliable** (95% CI −€700 to +€360), so it's worth testing in shadow mode, not adopting yet.
- **Recommended tiers:**

  | Tier | Rule (set on validation) | Test transactions | Frauds | Share of fraud money |
  |---|---|---|---|---|
  | Block | score ≥ 0.861 (≥95% precision on val) | 53 | 51 (96%) | 52% |
  | Step-up (OTP) / review | 2e-5 ≤ score < 0.861 | 55 | 7 | 17% |
  | Approve | below | 56,638 | 16 | 31% |

- **Sensitivity:** because LightGBM's scores are nearly all-or-nothing, the threshold jumps between plateaus (163 → 108 → 62 → 49 test alerts) as the false-alarm cost goes €1 → €250. PayGuard only needs to know which band its real cost falls in (<€10, €25–50, ≥€100), not a precise figure.

### Real-time inference

LightGBM scores one transaction in **0.6 ms (p50), 1.1 ms (p99)** on a laptop CPU, a tiny fraction of a ~100 ms authorisation budget. The real work in production is elsewhere:
- **Features:** behavioural signals (velocity in the last 1 h/24 h, amount vs. the customer's norm, new device/merchant/country, time since last transaction) served from a feature store with streaming aggregates.
- **Training/serving skew:** features must be identical and point-in-time correct offline and online, which is why online feature values are logged with each decision.
- **Label delay:** chargebacks arrive weeks later, so retrain on matured windows and monitor proxy signals in the meantime.
- **PCA features don't transfer:** `V1`–`V28` can't be reproduced live and can't be explained to a regulator. A production model should use raw, interpretable features.

### Limitations

- **Data:** only 2 days, and anonymised features.
- **Small test set:** 74 frauds, so the CIs are wide.
- **Regime shift:** the nights are in training only.
- **PCA leak:** the dataset creators fitted the PCA on all data.
- **Validation reused:** it serves both early stopping and threshold selection.
- **Assumed costs:** the cost figures are placeholders.
- **"Caught = prevented" is assumed.**

---

## Part B: LLM complaint triage

_Section completed after the evaluation run; see below._

---

## What I cut and why

_Completed at the end._
