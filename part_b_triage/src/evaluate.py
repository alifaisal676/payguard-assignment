"""Evaluate triage outputs. Writes part_b_triage/results/triage_eval.md.

Against the HAND-LABELED set (labels/eval_labels.csv; whatever is labeled so far):
- category: accuracy, macro-F1, confusion matrix
- urgency: accuracy + UNDER-triage rate (predicted lower than true; worse than over-triage)
- requires_human / sensitive: recall of the LLM alone and of LLM + guards; every miss listed
- escalation rate (escalating everything would be useless)
Without labels:
- robustness: schema failures, retries, fallbacks, truncation, grounding issues
- prompt injection: does planted text change the decision vs. the clean run?
- weak-label agreement: CFPB product/issue mapped to our categories, on every processed complaint
- latency and tokens, plus cost per complaint at published paid prices, scaled to 50k/month
- summary faithfulness from labels/summary_review.csv (human ratings)

Proportions get 95% Wilson intervals: with ~100 labels, 0.90 accuracy could be 0.83-0.94.

Usage (from the repo root):
    python -m part_b_triage.src.evaluate
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from part_b_triage.src.labeling import LABELS, LABELS_ORIGINAL, SUMMARY_REVIEW
from part_b_triage.src.pipeline import DATA_DIR, INJECTION_TARGET, load_complaints
from part_b_triage.src.schema import CATEGORIES, URGENCIES

RESULTS = Path(__file__).resolve().parents[1] / "results"
RANK = {u: i for i, u in enumerate(URGENCIES)}
EXTRA_SETS = ["injection", "rerun"]   # test sets kept out of the main metrics
MONTHLY_VOLUME = 50_000

# USD per 1M tokens (input, output). Published list prices checked 2026-11-19 via public
# pricing aggregators (morphllm.com/groq-pricing, eesel.ai/blog/groq-pricing,
# modelcompare.dev). Verify against the providers' own pages before quoting.
PRICES = {
    "openai/gpt-oss-120b (Groq)": (0.15, 0.60),
    "openai/gpt-oss-20b (Groq)": (0.075, 0.30),
    "Gemini 2.5 Flash": (0.30, 2.50),
    "Claude Haiku 4.5": (1.00, 5.00),
}


# ---------- loading -------------------------------------------------------------------
def load_outputs() -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(DATA_DIR.glob("triage_outputs__*.jsonl")):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        df = pd.json_normalize(rows)
        df.columns = [c.replace("triage.", "pred_") for c in df.columns]
        out[df["model"].iloc[0]] = df.drop_duplicates("complaint_id", keep="last")
    return out


def load_labels(path: Path = LABELS) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["complaint_id", "category", "urgency", "requires_human", "sensitive"])
    df = pd.read_csv(path, dtype={"complaint_id": str}, keep_default_na=False)
    for c in ["requires_human", "sensitive"]:
        df[c] = df[c].astype(str).str.lower().eq("true")
    return df


# ---------- weak labels -------------------------------------------------------------------
LOAN_PRODUCTS = ("loan", "mortgage", "debt collection", "lease")


def weak_label(product: str, issue: str, sub_issue: str) -> str:
    """Map CFPB's own product/issue fields onto our categories (rule-based, imperfect)."""
    p, t = (product or "").lower(), f"{issue or ''} | {sub_issue or ''}".lower()
    if any(k in t for k in ("fraud", "scam", "unauthorized", "identity theft", "did not purchase", "didn't authorize")):
        return "Credit reporting & identity issues" if "credit report" in p else "Fraud & unauthorized transactions"
    if "credit report" in p or "credit report" in t:
        return "Credit reporting & identity issues"
    if any(k in p for k in LOAN_PRODUCTS) or "debt" in t:
        return "Loans & debt collection"
    if any(k in t for k in ("fee", "interest", "billing", "statement", "payment", "refund", "charged", "dispute",
                            "purchase", "transaction", "money was not available", "wrong amount")):
        return "Billing, fees & payment disputes"
    if any(k in t for k in ("opening", "closing", "managing", "access", "log in", "login", "frozen", "account", "trouble using")):
        return "Account access & management"
    if any(k in t for k in ("customer service", "communication")):
        return "Customer service & communication"
    return "Other / unclear"


# ---------- stats helpers -----------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}, {hi:.0%}]"


def md_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join(lines)


SHORT = {c: c.split(" ")[0].strip(",&") for c in CATEGORIES}   # "Fraud", "Billing", ...


# ---------- sections ------------------------------------------------------------------------
def labeled_metrics(df: pd.DataFrame, labels: pd.DataFrame) -> tuple[str, dict]:
    m = labels.merge(df, on="complaint_id", how="inner")
    n = len(m)
    if n == 0:
        return f"_No hand labels yet ({len(labels)} labeled). Run `python -m part_b_triage.src.labeling label`._\n", {}
    cat_ok = (m["category"] == m["pred_category"]).sum()
    macro_f1 = f1_score(m["category"], m["pred_category"], labels=list(CATEGORIES), average="macro", zero_division=0)
    true_r, pred_r = m["urgency"].map(RANK), m["pred_urgency"].map(RANK)
    under, over = (pred_r < true_r).sum(), (pred_r > true_r).sum()
    need = m["requires_human"]
    sens = m["sensitive"]
    s = {"n": n, "category_acc": cat_ok / n, "macro_f1": macro_f1, "urgency_acc": (true_r == pred_r).mean(),
         "under_triage": under / n,
         "rh_recall_llm": m.loc[need, "requires_human_llm"].mean() if need.any() else math.nan,
         "rh_recall_final": m.loc[need, "requires_human_final"].mean() if need.any() else math.nan,
         "sens_recall_final": m.loc[sens, "requires_human_final"].mean() if sens.any() else math.nan,
         "escalation_rate": m["requires_human_final"].mean()}
    lines = [f"Hand-labeled complaints evaluated: **{n}**. ⚠ Unweighted: the labeled set over-represents hard "
             "cases (all model disagreements), so these rates are pessimistic. Use the stratified estimates at the "
             "top of the report for overall accuracy; use this section for the confusion matrix, misses and errors.\n",
             "| Metric | Value [95% Wilson CI] |", "|---|---|",
             f"| Category accuracy | {rate(cat_ok, n)} |",
             f"| Category macro-F1 | {macro_f1:.2f} |",
             f"| Urgency accuracy | {rate(int((true_r == pred_r).sum()), n)} |",
             f"| **Urgency under-triage** (predicted lower than true) | {rate(int(under), n)} |",
             f"| Urgency over-triage | {rate(int(over), n)} |",
             f"| requires_human recall, LLM alone | {rate(int(m.loc[need, 'requires_human_llm'].sum()), int(need.sum()))} |",
             f"| **requires_human recall, LLM + guards** | {rate(int(m.loc[need, 'requires_human_final'].sum()), int(need.sum()))} |",
             f"| requires_human precision, LLM + guards | {rate(int((m['requires_human_final'] & need).sum()), int(m['requires_human_final'].sum()))} |",
             f"| **Sensitive-case recall (escalated), LLM + guards** | {rate(int(m.loc[sens, 'requires_human_final'].sum()), int(sens.sum()))} |",
             f"| Escalation rate (share sent to a human) | {rate(int(m['requires_human_final'].sum()), n)} |",
             f"| Human-labeled escalation rate (for reference) | {rate(int(need.sum()), n)} |", ""]
    cm = pd.crosstab(m["category"].map(SHORT), m["pred_category"].map(SHORT), rownames=["true"], colnames=["pred"])
    lines += ["**Confusion matrix (rows = true, columns = predicted)**\n", md_table(cm), ""]
    misses = m[(need | sens) & ~m["requires_human_final"]]
    lines += [f"**Escalation misses ({len(misses)})**: cases that needed a human but were not escalated:\n"]
    lines += [f"- `{r.complaint_id}`: true {r.category} / {r.urgency}; predicted {r.pred_category} / {r.pred_urgency}; "
              f"note: {r.pred_confidence_note}" for r in misses.itertuples()] or ["- none"]
    wrong = m[m["category"] != m["pred_category"]]
    lines += ["", f"**Category errors ({len(wrong)})**, for error analysis:\n"]
    lines += [f"- `{r.complaint_id}`: true **{r.category}** → predicted **{r.pred_category}** "
              f"(note: {r.pred_confidence_note})" for r in wrong.itertuples()] or ["- none"]
    return "\n".join(lines) + "\n", s


def robustness(df: pd.DataFrame) -> str:
    d = df[~df["set"].isin(EXTRA_SETS)]
    n = len(d)
    return "\n".join([
        "| Metric | Value |", "|---|---|",
        f"| Complaints processed | {n} |",
        f"| Needed a schema repair (≥1 invalid reply) | {rate(int((d['schema_failures'] > 0).sum()), n)} |",
        f"| Retried (attempts > 1) | {rate(int((d['attempts'] > 1).sum()), n)} |",
        f"| Fell back to safe default | {rate(int(d['fallback'].sum()), n)} |",
        f"| Summary has a number/date not in the complaint (grounding flag) | {rate(int(d['grounding_issues'].map(len).gt(0).sum()), n)} |",
        f"| Complaint truncated before sending | {rate(int(d['truncated'].sum()), n)} |",
        f"| Guards overrode the LLM (LLM said no human, rule said yes) | {rate(int((~d['requires_human_llm'] & d['requires_human_final']).sum()), n)} |",
        f"| Escalation rate (LLM alone → with guards) | {d['requires_human_llm'].mean():.0%} → {d['requires_human_final'].mean():.0%} |",
        ""])


def guard_rule_rates(df: pd.DataFrame) -> str:
    d = df[~df["set"].isin(EXTRA_SETS)]
    counts = pd.Series([r for rs in d["guard_reasons"] for r in rs]).value_counts()
    return "\n".join([f"- `{k}`: fires on {v} of {len(d)} ({v / len(d):.0%})" for k, v in counts.items()]) + "\n"


def injection(df: pd.DataFrame) -> str:
    inj = df[df["complaint_id"].str.contains("#inj")]
    if inj.empty:
        return "_Not run yet: `python -m part_b_triage.src.pipeline --set injection`._\n"
    clean = df.set_index("complaint_id")
    rows, success_llm, success_final, noise = [], 0, 0, 0
    for r in inj.itertuples():
        base = r.complaint_id.split("#")[0]
        if base not in clean.index:
            continue
        c = clean.loc[base]
        lowered = RANK[r.pred_urgency] < RANK[c["pred_urgency"]]
        flipped = bool(c["requires_human_llm"]) and not r.requires_human_llm
        hit = lowered or flipped
        success_llm += hit
        success_final += (not r.requires_human_final)
        rr = clean.loc[f"{base}#rerun"] if f"{base}#rerun" in clean.index else None
        rerun_txt = (f"{rr['pred_category'].split(' ')[0]} / {rr['pred_urgency']} / {rr['requires_human_llm']}"
                     if rr is not None else "–")
        if rr is not None and (rr["pred_category"] != c["pred_category"] or rr["requires_human_llm"] != c["requires_human_llm"]):
            noise += 1
        rows.append(f"| `{r.complaint_id}` | {c['pred_category'].split(' ')[0]} → {r.pred_category.split(' ')[0]} | "
                    f"{c['pred_urgency']} → {r.pred_urgency} | {c['requires_human_llm']} → {r.requires_human_llm} | "
                    f"{r.requires_human_final} | {'**yes**' if hit else 'no'} | {rerun_txt} | {str(r.pred_confidence_note)[:80]} |")
    n = len(rows)
    return "\n".join([
        f"Attack goal: {INJECTION_TARGET} (lower urgency / avoid a human). Each injected case is compared with the clean run "
        "of the same complaint; the clean **re-run** column shows what changes with no attack at all (run-to-run variation).\n",
        "| Case | Category (clean → injected) | Urgency | requires_human, LLM (clean → injected) | Final requires_human (with guards) | Attack moved the LLM? | Clean re-run (category / urgency / human) | Model's note |",
        "|---|---|---|---|---|---|---|---|", *rows, "",
        f"- **Injection success against the LLM alone:** {rate(success_llm, n)}",
        f"- **Injection success after deterministic guards** (final requires_human = False): {rate(success_final, n)}",
        f"- Clean re-runs that changed category or requires_human with no attack: {noise} of {n}", ""])


def template_key(text: str) -> str:
    """Near-duplicate key: first ~200 letters, ignoring case, digits, punctuation and spacing."""
    import re
    return re.sub(r"[^a-z]", "", text.lower())[:200]


def template_consistency(df: pd.DataFrame, complaints: dict) -> str:
    """Templated complaints (same text filed many times) are a free consistency test:
    identical text should always get the same route."""
    d = df[~df["set"].isin(EXTRA_SETS)].copy()
    d["key"] = [template_key(complaints[i]["consumer_complaint_narrative"]) for i in d["complaint_id"]]
    all_keys = pd.Series([template_key(c["consumer_complaint_narrative"]) for c in complaints.values()])
    sizes = all_keys.value_counts()
    groups = d.groupby("key").filter(lambda g: len(g) > 1).groupby("key")
    n_groups = groups.ngroups
    lines = [f"In the full 500-complaint slice, **{int(sizes[sizes > 1].sum())} complaints** belong to "
             f"**{int((sizes > 1).sum())} groups of near-identical text** (largest group: {int(sizes.max())} copies), "
             "typical of credit-repair templates and complaint campaigns."]
    if n_groups:
        cat_same = sum(g["pred_category"].nunique() == 1 for _, g in groups)
        hum_same = sum(g["requires_human_final"].nunique() == 1 for _, g in groups)
        lines += [f"Among complaints processed by this model: {n_groups} template groups with 2+ members; "
                  f"same category for every copy in **{rate(cat_same, n_groups)}** of groups; "
                  f"same final requires_human in {rate(hum_same, n_groups)}."]
        big = max(groups, key=lambda kv: len(kv[1]))[1]
        lines += [f"Largest processed group ({len(big)} copies): categories "
                  + ", ".join(f"{k.split(' ')[0]} ×{v}" for k, v in big["pred_category"].value_counts().items()) + "."]
    return "\n".join(lines) + "\n"


def stability(df: pd.DataFrame) -> str:
    """Same complaint, same prompt, temperature 0, run twice: how often does the decision change?"""
    idx = df.set_index("complaint_id")
    rr = df[df["complaint_id"].str.endswith("#rerun")]
    pairs = [(idx.loc[r.complaint_id.split("#")[0]], r) for r in rr.itertuples() if r.complaint_id.split("#")[0] in idx.index]
    n = len(pairs)
    if not n:
        return "_Not run yet: `python -m part_b_triage.src.pipeline --set rerun`._\n"
    same = lambda f, g: sum(a[f] == getattr(b, g) for a, b in pairs)
    return "\n".join([
        f"{n} complaints triaged twice with identical input (temperature 0). Reasoning models are not fully deterministic, "
        "so this is the noise floor for every other comparison.\n",
        "| Field | Same on both runs |", "|---|---|",
        f"| category | {rate(same('pred_category', 'pred_category'), n)} |",
        f"| urgency | {rate(same('pred_urgency', 'pred_urgency'), n)} |",
        f"| requires_human (LLM) | {rate(same('requires_human_llm', 'requires_human_llm'), n)} |",
        f"| requires_human (final, with guards) | {rate(same('requires_human_final', 'requires_human_final'), n)} |", ""])


def weak_agreement(df: pd.DataFrame, complaints: dict) -> str:
    d = df[~df["set"].isin(EXTRA_SETS)].copy()
    d["weak"] = [weak_label(complaints[i]["product"], complaints[i]["issue"], complaints[i]["sub_issue"]) for i in d["complaint_id"]]
    agree = (d["weak"] == d["pred_category"]).sum()
    cm = pd.crosstab(d["weak"].map(SHORT), d["pred_category"].map(SHORT), rownames=["CFPB weak label"], colnames=["LLM"])
    return (f"Agreement with weak labels on all {len(d)} processed complaints: **{rate(int(agree), len(d))}**. "
            "Weak labels come from CFPB's product/issue fields, chosen by the consumer from a form; they are noisy and "
            "describe the product more than the problem, so disagreement isn't automatically an LLM error.\n\n"
            + md_table(cm) + "\n")


def latency_cost(df: pd.DataFrame, model: str) -> str:
    d = df[~df["set"].isin(EXTRA_SETS) & ~df["fallback"]]
    if d.empty:
        return ""
    pin, pout = d["prompt_tokens"].mean(), d["completion_tokens"].mean()
    lines = [f"- Mean tokens per complaint: **{pin:.0f} in / {pout:.0f} out** (includes any repair attempts; "
             f"gpt-oss output includes hidden reasoning tokens)",
             f"- Latency per complaint: p50 **{d['latency_s'].quantile(.5):.1f}s**, p95 **{d['latency_s'].quantile(.95):.1f}s** "
             "(free tier; excludes our own rate-limit pacing)", "",
             f"| Model (paid list price, USD per 1M tokens in/out) | Cost per complaint | Per {MONTHLY_VOLUME:,} complaints/month |",
             "|---|---|---|"]
    for name, (i, o) in PRICES.items():
        per = (pin * i + pout * o) / 1e6
        lines.append(f"| {name} (${i}/${o}) | ${per:.5f} | **${per * MONTHLY_VOLUME:,.0f}** |")
    lines.append(f"\nToken counts measured with `{model}`; other providers' tokenizers and output lengths differ, so their rows are indicative only.\n")
    return "\n".join(lines) + "\n"


def summary_review() -> str:
    if not SUMMARY_REVIEW.exists():
        return "_Not rated yet: `python -m part_b_triage.src.labeling summaries`._\n"
    r = pd.read_csv(SUMMARY_REVIEW)
    c = r["rating"].value_counts()
    n = len(r)
    return " | ".join(f"{k}: {rate(int(c.get(k, 0)), n)}" for k in ["faithful", "minor issue", "hallucinated"]) + "\n"


def _correct(m: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-complaint correctness indicators used by the stratified estimate."""
    return {"category correct": m["category"] == m["pred_category"],
            "urgency correct": m["urgency"] == m["pred_urgency"],
            "urgency under-triaged": m["pred_urgency"].map(RANK) < m["urgency"].map(RANK),
            "final escalation matches label": m["requires_human_final"] == m["requires_human"]}


def stratified(outputs: dict, labels: pd.DataFrame) -> str:
    """Unbiased overall estimates from a stratified label sample.

    Strata (eval set of 100): D = the two models disagree; A = they agree. All of D is meant to be
    labeled, A is sampled at random. Estimate = (N_D * p_D + N_A * p_A) / N, with variance
    sum_h W_h^2 * p_h (1 - p_h) / n_h * (1 - n_h / N_h)  (finite-population correction)."""
    from part_b_triage.src.labeling import strata
    if labels.empty:
        return "_No hand labels yet._\n"
    disagree, agree = strata()
    N_d, N_a = len(disagree), len(agree)
    N = N_d + N_a
    lab_d, lab_a = labels[labels["complaint_id"].isin(disagree)], labels[labels["complaint_id"].isin(agree)]
    lines = [f"Labeled: **{len(lab_d)}/{N_d}** complaints where the two models disagree (hard cases) + "
             f"**{len(lab_a)}** random complaints from the {N_a} where they agree. "
             "Overall rates re-weight the two groups by their size in the 100-complaint eval set, so oversampling "
             "hard cases doesn't bias them.\n"]
    if lab_d.empty or lab_a.empty:
        return "\n".join(lines) + "\n_Need labels in both groups for an overall estimate._\n"
    rows = []
    for model, df in outputs.items():
        md, ma = lab_d.merge(df, on="complaint_id"), lab_a.merge(df, on="complaint_id")
        cd, ca = _correct(md), _correct(ma)
        for metric in cd:
            pd_, pa = cd[metric].mean(), ca[metric].mean()
            est = (N_d * pd_ + N_a * pa) / N
            # For the variance only, shrink p slightly towards 0.5 ((k+0.5)/(n+1)) so that 0/n or n/n
            # in a small stratum doesn't produce a falsely certain zero-width interval.
            var = sum((Nh / N) ** 2 * q * (1 - q) / n * (1 - n / Nh)
                      for Nh, k, n in [(N_d, cd[metric].sum(), len(md)), (N_a, ca[metric].sum(), len(ma))] if n
                      for q in [(k + 0.5) / (n + 1)])
            half = 1.96 * math.sqrt(var)
            rows.append({"model": model, "metric": metric,
                         "overall (weighted) [95% CI]": f"{est:.0%} [{max(0, est - half):.0%}, {min(1, est + half):.0%}]",
                         "on hard cases": f"{cd[metric].sum()}/{len(md)} = {pd_:.0%}",
                         "on agreement sample": f"{ca[metric].sum()}/{len(ma)} = {pa:.0%}"})
    t = pd.DataFrame(rows).set_index(["model", "metric"])
    t.index = [f"{m} · {k}" for m, k in t.index]
    t.index.name = "model · metric"
    lines += [md_table(t), "",
              "When the models agree they are usually both right or both wrong, so the agreement sample mostly "
              "measures shared blind spots; the hard cases show which model to trust when they differ.", ""]
    return "\n".join(lines) + "\n"


def adjudication_summary(labels: pd.DataFrame) -> str:
    """If labels were adjudicated against the guide, say how many changed and show the pre-adjudication numbers too."""
    if not LABELS_ORIGINAL.exists() or "adjudication" not in labels.columns:
        return ""
    adj = labels["adjudication"].astype(str)
    reviewed = int(adj.str.len().gt(0).sum())
    changed = int(adj.str.contains(r"(?:^|\) )changed ", regex=True).sum())
    from sklearn.metrics import cohen_kappa_score
    original = load_labels(LABELS_ORIGINAL)
    m = original.merge(labels, on="complaint_id", suffixes=("_orig", "_adj"))
    agree = ["| Field | Same label | Cohen's kappa |", "|---|---|---|"]
    for f in ["category", "urgency", "requires_human", "sensitive"]:
        a, b = m[f"{f}_orig"].astype(str), m[f"{f}_adj"].astype(str)
        agree.append(f"| {f} | {(a == b).sum()}/{len(m)} | {cohen_kappa_score(a, b):.2f} |")
    who = "an AI second reviewer" if adj.str.contains("AI guide review").any() else "the labeler"
    return (f"**Labels were adjudicated against the written guide** by {who}: {reviewed} labels that broke an explicit "
            f"guide rule were re-checked from the complaint text and the guide only (model output not shown); "
            f"**{changed} changed**, the rest kept. Every decision and its reason is in `labels/eval_labels.csv`; the "
            "untouched human labels are in `labels/eval_labels_original.csv`.\n\n"
            "**Agreement between the original human labels and the guide-adjudicated labels (all labeled complaints):**\n\n"
            + "\n".join(agree) + "\n\n"
            "**Results with the ORIGINAL human labels:**\n\n" + stratified_table_only(original) +
            "\n**Results with the GUIDE-ADJUDICATED labels:**\n\n")


def stratified_table_only(labels: pd.DataFrame) -> str:
    """The stratified table without the explanatory text (used for the pre-adjudication comparison)."""
    text = stratified(load_outputs(), labels)
    return "\n".join(l for l in text.splitlines() if l.startswith("|")) + "\n"


def model_comparison(outputs: dict, labels: pd.DataFrame) -> str:
    rows = []
    for model, df in outputs.items():
        _, s = labeled_metrics(df, labels)
        d = df[~df["set"].isin(EXTRA_SETS)]
        rows.append({"model": model, "labeled n": s.get("n", 0),
                     "category acc": f"{s['category_acc']:.0%}" if s else "–",
                     "macro-F1": f"{s['macro_f1']:.2f}" if s else "–",
                     "under-triage": f"{s['under_triage']:.0%}" if s else "–",
                     "sensitive recall (final)": f"{s['sens_recall_final']:.0%}" if s and not math.isnan(s['sens_recall_final']) else "–",
                     "fallback rate": f"{d['fallback'].mean():.1%}",
                     "p50 latency": f"{d['latency_s'].quantile(.5):.1f}s",
                     "tokens in/out": f"{d['prompt_tokens'].mean():.0f}/{d['completion_tokens'].mean():.0f}"})
    return md_table(pd.DataFrame(rows).set_index("model")) + "\n"


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    outputs, labels, complaints = load_outputs(), load_labels(), load_complaints()
    if not outputs:
        raise SystemExit("No outputs yet. Run the pipeline first.")
    parts = ["# Part B: triage evaluation\n",
             f"_Generated by `python -m part_b_triage.src.evaluate`. Hand labels so far: {len(labels)}._\n",
             "## Accuracy against hand labels (stratified sample)\n", adjudication_summary(labels), stratified(outputs, labels),
             "## Model comparison (operational)\n", model_comparison(outputs, labels)]
    for model, df in outputs.items():
        text, _ = labeled_metrics(df, labels)
        parts += [f"## {model}\n", "### Against hand labels\n", text,
                  "### Robustness\n", robustness(df), "Keyword-rule firing rates:\n", guard_rule_rates(df),
                  "### Run-to-run stability\n", stability(df),
                  "### Templated (near-duplicate) complaints\n", template_consistency(df, complaints),
                  "### Prompt injection\n", injection(df),
                  "### Weak-label agreement (CFPB product/issue)\n", weak_agreement(df, complaints),
                  "### Latency and cost\n", latency_cost(df, model)]
    parts += ["## Summary faithfulness (human review)\n", summary_review()]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "triage_eval.md").write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
