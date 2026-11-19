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

Full, regenerated report: **[part_b_triage/results/triage_eval.md](part_b_triage/results/triage_eval.md)**.

### Data: and a change of plan

The CFPB **stopped publishing complaint narratives on 14 Aug 2026**; its API now returns metadata only and rejects the narrative field. So `fetch_data.py` samples the narratives (published before that date) from the Hugging Face mirror [`Mouwiya/cfpb-consumer-complaints`](https://huggingface.co/datasets/Mouwiya/cfpb-consumer-complaints) (revision pinned in the output) through the row API, and **verifies a random sample of complaint IDs against the official CFPB API**: 39 of 39 matched on product, issue and date (one lookup failed on the network).

- **Slice:** 500 complaints received 2024–2026, ≥100 characters, stratified by product. Seeds and quotas are in `fetch_data.py`.
- **Committed:** `part_b_triage/data/complaints_raw.json` (~0.8 MB). Narratives can't be re-downloaded from CFPB any more, so committing them keeps the work reproducible.
- **Templated complaints:** 83 of 500 belong to 24 groups of near-identical text. One Cash App template appears 25 times; several credit-repair templates appear 4 times. Real inflow looks like this too.

### Design

```
complaint ─► mask PII ─► LLM (JSON mode, temperature 0, prompt v1) ─► Pydantic schema ─► deterministic guards ─► route + log
                              ▲      invalid? send the error back, retry (max 3) │            keyword backstop can only ADD escalation
                              └─────────────────────────────────────────────────┘            grounding check on summary numbers/dates
                                                    still invalid? ─► safe fallback: "Other / unclear", requires_human = true
```

| Piece | File | Key choices |
|---|---|---|
| Taxonomy & policies | [`prompts.py`](part_b_triage/src/prompts.py), [`LABELING_GUIDE.md`](part_b_triage/labels/LABELING_GUIDE.md) | 7 routing categories with definitions and a tie-break order; urgency rubric; `requires_human` policy. The prompt and the labeling guide share one taxonomy. |
| Schema | [`schema.py`](part_b_triage/src/schema.py) | `Literal` types reject hallucinated categories and urgencies; extra keys are forbidden; summary length is limited |
| Client | [`llm_client.py`](part_b_triage/src/llm_client.py) | Groq / Gemini / Ollama via one OpenAI-compatible path (`LLM_PROVIDER`); tenacity backoff honouring `retry-after`; validation-error feedback; safe fallback; every call logged |
| Prompt-injection defences | `prompts.py` + `guards.py` | Complaint in `<complaint>` delimiters that can't be closed from inside; "data, not instructions" rule; deterministic backstop |
| Guards | [`guards.py`](part_b_triage/src/guards.py) | Regex rules (fraud, identity theft, legal, regulator/police, vulnerable, self-harm) force `requires_human`; summary grounding check; PII masking |
| Pipeline | [`pipeline.py`](part_b_triage/src/pipeline.py) | Resumable (skips done IDs); paces itself under the free tier's tokens-per-minute limit; stops cleanly at the daily quota |
| Evaluation | [`evaluate.py`](part_b_triage/src/evaluate.py), [`labeling.py`](part_b_triage/src/labeling.py) | See below |

### Evaluation design

Rigour matters more than the headline number, so each claim has its own test:

1. **Hand labels (100 complaints, stratified):** category, urgency, requires_human, sensitive.
   - Labeled **by me**, following a guide written first, in a terminal tool that hides the model's output and CFPB's product/issue fields. Model-made labels would make the evaluation circular.
   - Reports accuracy, macro-F1 and the confusion matrix; **urgency under-triage**; **sensitive-case recall** (target 100%, every miss listed); and the **escalation rate**, because escalating everything would be useless.
2. **Run-to-run stability:** 30 complaints triaged twice with identical input. This is the noise floor every other comparison needs.
3. **Prompt injection:** 8 complaints with planted instructions (four attack styles, including a fake `</complaint>` + "SYSTEM:"), compared with both the clean run and the clean re-run.
4. **Templated complaints:** identical text should get an identical route.
5. **Weak labels:** CFPB's own product/issue fields mapped to our categories, on every processed complaint.
6. **Robustness, latency and cost** from the call logs; **summary faithfulness** by human rating of 30 summaries, plus an automatic grounding check.
7. **Two models:** `gpt-oss-120b` (main) vs. `gpt-oss-20b`.

All proportions carry 95% Wilson intervals.

### Results

> **Hand-label metrics (accuracy, under-triage, sensitive recall) are filled in by `evaluate.py` once `labels/eval_labels.csv` exists.** Everything below doesn't depend on labels.

| | gpt-oss-120b | gpt-oss-20b |
|---|---|---|
| Schema failures / retries / fallbacks (100 eval complaints) | 0 / 0 / 0 | 0 / 0 / 0 |
| Summaries with a number/date not in the complaint | 0 | 0 |
| Same category on an identical re-run | 27/30 (90%) | 29/30 (97%) |
| Same **final** escalation on re-run (with guards) | **30/30** | **30/30** |
| Identical template (7 copies) routed consistently | no (Customer ×4, Fraud ×3) | yes (Fraud ×7) |
| Escalation rate, LLM alone → with guards | 30% → 51% | 49% → 63% |
| Prompt injection moved the LLM | 1 clear + 1 within noise, of 8 | 1 of 8 |
| **Prompt injection success after guards** | **0 of 8** | **0 of 8** |
| Agreement with CFPB weak labels | 46% | 48% |
| Tokens per complaint (in / out) | 992 / 129 | 989 / 110 |
| Latency p50 / p95 | 0.9 s / 1.4 s | 0.8 s / 1.6 s |

**What these say**
- **The schema plus validation retries make output failures a non-issue.** The real risks are semantic: wrong route, under-triage, an inconsistent decision.
- **The LLM alone isn't reliable enough for safety decisions.** It changes its escalation decision on 1 in 10 identical re-runs and can be pushed by a fake-delimiter attack. **The deterministic guards make the final escalation decision fully stable and resisted every injection.** That's the argument for keeping safety rules outside the model.
- **The cost of that safety is escalation volume.** With guards, about half of complaints go to a human, mainly because "fraud" words are common, for example in credit-report disputes. The hand labels measure whether that is over-escalation. The likely next step is a cheaper "light review" queue for keyword-only escalations.
- **The weak-label disagreement is mostly the weak labels' fault.** Complaints filed under CFPB's "debt collection" product are often credit-report disputes ("remove this collection from my report"), which the LLM routes correctly. Weak labels are good for spotting drift, not for measuring accuracy.
- **The smaller model was more stable and more conservative.** Its accuracy is decided by the hand labels.

### Cost (paid list prices, measured token counts)

| Model | Per complaint | Per 50,000 complaints/month |
|---|---|---|
| gpt-oss-120b (Groq, $0.15/$0.60 per 1M) | $0.00023 | **≈ $11** |
| gpt-oss-20b (Groq, $0.075/$0.30) | $0.00011 | ≈ $6 |
| Gemini 2.5 Flash ($0.30/$2.50) | ≈ $0.0006 | ≈ $31 |
| Claude Haiku 4.5 ($1/$5) | ≈ $0.0016 | ≈ $82 |

The model is **not** the cost driver. A self-hosted open model on an in-country GPU is a fixed monthly cost, plausibly in the low thousands of dollars (my rough estimate; it depends on hardware and hosting, and needs a real quote). Human review of ~50% escalations dominates everything. The in-country hosting requirement (Part C) rules out these foreign APIs anyway. The free tier's limits (8k tokens/min, a daily quota) were the practical bottleneck of this prototype, which is why the pipeline paces itself and resumes.

### Failure modes and mitigations

| Failure | Seen here? | Mitigation |
|---|---|---|
| Invalid / hallucinated output format | No (0/100) | Pydantic `Literal` schema, error-feedback retry, safe fallback |
| Inconsistent route for the same text | Yes (90% re-run agreement; template split 4/3) | Deterministic guards for safety; cluster templated complaints and route them once; v2 prompt with explicit "fraud > customer service" examples |
| Prompt injection | Yes (fake delimiter worked once) | Delimiters + instruction; guards can only add escalation; log injection patterns |
| Summary hallucination | Not found by the automatic check | Grounding check; human rating; show original next to summary in the agent UI |
| Over-escalation | 51–63% escalated | Measure precision with hand labels; split keyword-only escalations into a lighter queue |
| Provider silently changes the model | n/a | Pin versions; re-run this evaluation on every change (Part C) |

---

## What I cut and why

- **Ollama / local model comparison.** It needs a local model download (several GB on a slow connection). The provider switch is built and is one line in `.env`. I compared two hosted model sizes instead.
- **SMOTE.** It isn't installed and isn't needed: the class-weighting experiment showed reweighting hurts LightGBM here, so oversampling is unlikely to help, and imbalance is handled by the metric and threshold.
- **Hyperparameter tuning, ensembles, deep learning (Part A).** ~340 training frauds can't support them without overfitting the validation set.
- **Hour-of-day features (Part A).** Validation and test contain no night hours, so they couldn't be validated.
- **Full 500-complaint batch on both models.** Free-tier daily token quotas allow roughly 150–200 triages per model per day. The pipeline resumes, so the remainder is a re-run of the same command.
- **Prompt v2.** The template inconsistency points to a fix, but re-running the whole evaluation within the quota would have taken another day. It's listed as the first next step.
- **LLM-as-judge for summaries.** Without human ratings to calibrate it, it would just be another unvalidated model. Human ratings come first.
