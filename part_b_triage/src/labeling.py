"""Build the evaluation set and label it by hand, in the terminal.

The labels must come from a human. If an LLM wrote the labels and an LLM made the
predictions, the evaluation would be circular.

Which complaints get labeled (stratified sampling, as in active learning):
- ALL eval complaints where gpt-oss-120b and gpt-oss-20b disagree (category, urgency or
  final escalation): the hard cases, where labels are most informative;
- plus a RANDOM sample of AGREE_SAMPLE complaints where they agree.
Labeling only the hard cases would make accuracy look far worse than it is; evaluate.py
re-weights the two strata by their size to give an unbiased overall estimate.

This tool deliberately hides:
- the model's output, and
- CFPB's own product/issue fields (used separately as *weak labels*; showing them would bias
  the hand labels towards them and inflate weak-label agreement).

Commands (from the repo root):
    python -m part_b_triage.src.labeling make-set        # pick 100 complaints -> labels/eval_set.csv
    python -m part_b_triage.src.labeling label           # label them (resumable) -> labels/eval_labels.csv
    python -m part_b_triage.src.labeling adjudicate      # re-check labels that break a guide rule
    python -m part_b_triage.src.labeling summaries       # rate 30 model summaries -> labels/summary_review.csv

Read labels/LABELING_GUIDE.md before labeling.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

from part_b_triage.src.pipeline import EVAL_SET, LABEL_DIR, load_complaints, output_path
from part_b_triage.src.schema import CATEGORIES

LABELS = LABEL_DIR / "eval_labels.csv"
SUMMARY_REVIEW = LABEL_DIR / "summary_review.csv"
EVAL_SIZE, SEED = 100, 7
LABEL_FIELDS = ["complaint_id", "category", "urgency", "requires_human", "sensitive", "notes"]
URG = {"l": "low", "m": "medium", "h": "high"}
MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")   # disagreement between these defines the hard stratum
AGREE_SAMPLE = 20                                          # random cases labeled from the agreement stratum


def make_set():
    if EVAL_SET.exists():
        sys.exit(f"{EVAL_SET} already exists; delete it first if you really want a new sample.")
    complaints = load_complaints()
    by_group = defaultdict(list)
    for cid, c in sorted(complaints.items()):
        by_group[c["product_group"]].append(cid)
    rng, picked = random.Random(SEED), []
    per_group = EVAL_SIZE // len(by_group)
    for g in sorted(by_group):
        picked += rng.sample(by_group[g], min(per_group, len(by_group[g])))
    rest = [c for g in by_group for c in by_group[g] if c not in picked]
    picked += rng.sample(rest, EVAL_SIZE - len(picked))
    rng.shuffle(picked)   # label in random order, not grouped by product
    with EVAL_SET.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["complaint_id", "product_group"])
        for cid in picked:
            w.writerow([cid, complaints[cid]["product_group"]])
    print(f"Wrote {len(picked)} IDs to {EVAL_SET}")


def _read(path, key="complaint_id"):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r[key]: r for r in csv.DictReader(f)}


def _append(path, fields, row):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def _show(text):
    print("\n" + "\n".join(textwrap.fill(p, 100) for p in text.split("\n") if p.strip()))


def _ask(prompt, valid):
    while True:
        a = input(prompt).strip().lower()
        if a in valid:
            return a
        print(f"  please type one of: {', '.join(sorted(valid))}")


def eval_order() -> list[str]:
    with EVAL_SET.open(encoding="utf-8") as f:
        return [r["complaint_id"] for r in csv.DictReader(f)]   # already in random order


def _eval_outputs(model: str) -> dict:
    p = output_path(model)
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["complaint_id"]: r for r in rows if r["set"] == "eval"}


def strata() -> tuple[list[str], list[str]]:
    """Split the eval set into (disagree, agree): the two models differ on category, urgency
    or final escalation, or they agree on all three. Order inside each stratum stays random."""
    a, b = _eval_outputs(MODELS[0]), _eval_outputs(MODELS[1])
    def differs(i):
        return (a[i]["triage"]["category"] != b[i]["triage"]["category"]
                or a[i]["triage"]["urgency"] != b[i]["triage"]["urgency"]
                or a[i]["requires_human_final"] != b[i]["requires_human_final"])
    ids = [i for i in eval_order() if i in a and i in b]
    return [i for i in ids if differs(i)], [i for i in ids if not differs(i)]


def label():
    """Label ALL disagreement cases, then a random sample of AGREE_SAMPLE agreement cases.

    This is stratified sampling: hard cases are fully covered, easy cases are sampled, and
    evaluate.py re-weights both strata so overall accuracy stays unbiased."""
    complaints = load_complaints()
    disagree, agree = strata()
    done = _read(LABELS)
    n_agree_done = sum(i in done for i in agree)
    todo = [i for i in disagree if i not in done]
    todo += [i for i in agree if i not in done][: max(0, AGREE_SAMPLE - n_agree_done)]
    print(f"Plan: all {len(disagree)} cases where the two models disagree + {AGREE_SAMPLE} random cases where they agree.\n"
          f"Done: {sum(i in done for i in disagree)}/{len(disagree)} disagreement, {n_agree_done}/{AGREE_SAMPLE} agreement. "
          f"{len(todo)} to go. Type q at any prompt to stop (progress is saved).")
    cat_menu = "  ".join(f"{k}={c}" for k, c in enumerate(CATEGORIES, 1))
    for n, cid in enumerate(todo, 1):
        group = "disagreement case" if cid in disagree else "random agreement case"
        print("\n" + "=" * 100 + f"\n[{n}/{len(todo)}] complaint {cid}  ({group})")
        _show(complaints[cid]["consumer_complaint_narrative"])
        print("\n" + textwrap.fill(cat_menu, 100))
        c = _ask("category 1-7: ", {str(k) for k in range(1, 8)} | {"q"})
        if c == "q": break
        u = _ask("urgency l/m/h: ", set(URG) | {"q"})
        if u == "q": break
        h = _ask("requires_human y/n: ", {"y", "n", "q"})
        if h == "q": break
        s = _ask("sensitive (fraud/legal/regulator/vulnerable/self-harm) y/n: ", {"y", "n", "q"})
        if s == "q": break
        notes = input("notes (optional, Enter to skip): ").strip()
        _append(LABELS, LABEL_FIELDS, {"complaint_id": cid, "category": CATEGORIES[int(c) - 1], "urgency": URG[u],
                                       "requires_human": h == "y", "sensitive": s == "y", "notes": notes})
    done = _read(LABELS)
    print(f"\nSaved to {LABELS}: {sum(i in done for i in disagree)}/{len(disagree)} disagreement, "
          f"{sum(i in done for i in agree)}/{AGREE_SAMPLE} agreement cases labeled.")


LABELS_ORIGINAL = LABEL_DIR / "eval_labels_original.csv"
GUIDE_RULES = {
    "sensitive_without_trigger": ("You marked SENSITIVE, but no fraud/identity-theft, legal, regulator/police, vulnerable-customer "
                                  "or self-harm wording was detected.",
                                  "Guide §4: sensitive = fraud/identity theft, legal threats, regulator/police, vulnerable customer or self-harm."),
    "trigger_but_not_sensitive": ("You marked NOT sensitive, but the text contains a trigger: {reasons}.",
                                  "Guide §4 (same definition). A trigger word isn't proof: decide whether the complaint really alleges it."),
    "human_without_reason": ("You marked REQUIRES_HUMAN, but not sensitive and no trigger was detected.",
                             "Guide §3: yes only for fraud/scam, identity theft, legal, regulator/police, vulnerable, self-harm, "
                             "or if YOU found the category genuinely ambiguous."),
    "sensitive_but_no_human": ("You marked SENSITIVE but NOT requires_human.",
                               "Guide §3/§4: every sensitive case requires a human."),
    "low_urgency_money_issue": ("You marked LOW urgency for a {category} complaint.",
                                "Guide §2: medium = unresolved financial impact (wrong charge, fee, credit-report error); "
                                "low = informational, service quality only, or already resolved."),
    "template_inconsistent": ("This exact complaint text appears {n} times in your labels with different labels: {labels}.",
                              "Identical text should get identical labels: pick the one the guide supports."),
}


def _flags(cid: str, lab: dict, text: str, template_labels: dict) -> list[tuple[str, dict]]:
    """Guide-rule checks on ONE human label. Uses only the complaint text and the label, never model output."""
    from part_b_triage.src.guards import keyword_backstop
    reasons = keyword_backstop(text)
    sens, human = lab["sensitive"] == "True", lab["requires_human"] == "True"
    out = []
    if sens and not reasons:
        out.append(("sensitive_without_trigger", {}))
    if not sens and reasons:
        out.append(("trigger_but_not_sensitive", {"reasons": ", ".join(reasons)}))
    if human and not sens and not reasons:
        out.append(("human_without_reason", {}))
    if sens and not human:
        out.append(("sensitive_but_no_human", {}))
    if lab["urgency"] == "low" and lab["category"] in ("Credit reporting & identity issues", "Billing, fees & payment disputes",
                                                       "Fraud & unauthorized transactions"):
        out.append(("low_urgency_money_issue", {"category": lab["category"]}))
    group = template_labels.get(cid)
    if group and len(set(group.values())) > 1:
        out.append(("template_inconsistent", {"n": len(group), "labels": "; ".join(sorted(set(group.values())))}))
    return out


def adjudicate():
    """Re-check labels that break an explicit guide rule. Model outputs are never shown.
    The first run saves the untouched labels to eval_labels_original.csv (both are reported)."""
    import shutil
    from part_b_triage.src.evaluate import template_key
    complaints = load_complaints()
    if not LABELS_ORIGINAL.exists():
        shutil.copy(LABELS, LABELS_ORIGINAL)
    with LABELS.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = LABEL_FIELDS + ["adjudication"]
    by_key = defaultdict(dict)
    for r in rows:
        by_key[template_key(complaints[r["complaint_id"]]["consumer_complaint_narrative"])][r["complaint_id"]] = \
            f"{r['category']} / {r['urgency']}"
    template_labels = {cid: g for g in by_key.values() if len(g) > 1 for cid in g}
    todo = [(r, _flags(r["complaint_id"], r, complaints[r["complaint_id"]]["consumer_complaint_narrative"], template_labels))
            for r in rows if not r.get("adjudication")]
    todo = [(r, f) for r, f in todo if f]
    print(f"{len(todo)} labels break a guide rule and need a second look. You won't see any model output.\n"
          "For each: k = keep your label, c = change it. q = stop (progress is saved).")
    cat_menu = "  ".join(f"{k}={c}" for k, c in enumerate(CATEGORIES, 1))
    for n, (r, flags) in enumerate(todo, 1):
        cid = r["complaint_id"]
        print("\n" + "=" * 100 + f"\n[{n}/{len(todo)}] complaint {cid}")
        _show(complaints[cid]["consumer_complaint_narrative"])
        print(f"\nYOUR LABEL: category={r['category']} | urgency={r['urgency']} | requires_human={r['requires_human']} | "
              f"sensitive={r['sensitive']}" + (f" | notes: {r['notes']}" if r.get("notes") else ""))
        for name, ctx in flags:
            msg, guide = GUIDE_RULES[name]
            print(f"  ⚑ {msg.format(**ctx)}\n    {guide}")
        a = _ask("keep or change? k/c: ", {"k", "c", "q"})
        if a == "q":
            break
        if a == "c":
            print(textwrap.fill(cat_menu, 100))
            c = _ask(f"category 1-7 (Enter = keep '{r['category']}'): ", {str(k) for k in range(1, 8)} | {""})
            u = _ask(f"urgency l/m/h (Enter = keep '{r['urgency']}'): ", set(URG) | {""})
            h = _ask(f"requires_human y/n (Enter = keep '{r['requires_human']}'): ", {"y", "n", ""})
            s = _ask(f"sensitive y/n (Enter = keep '{r['sensitive']}'): ", {"y", "n", ""})
            before = f"{r['category']}/{r['urgency']}/{r['requires_human']}/{r['sensitive']}"
            r["category"] = CATEGORIES[int(c) - 1] if c else r["category"]
            r["urgency"] = URG[u] if u else r["urgency"]
            r["requires_human"] = str(h == "y") if h else r["requires_human"]
            r["sensitive"] = str(s == "y") if s else r["sensitive"]
            why = input("why (one line): ").strip()
            r["adjudication"] = f"changed from {before}: {why}"
        else:
            r["adjudication"] = "kept: " + input("why keep (optional, one line): ").strip()
        with LABELS.open("w", newline="", encoding="utf-8") as f:     # save after every decision
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    changed = sum(r.get("adjudication", "").startswith("changed") for r in rows)
    print(f"\nDone for now. {changed} labels changed; originals kept in {LABELS_ORIGINAL.name}.")


def summaries(model: str = "openai/gpt-oss-120b", n: int = 30):
    complaints = load_complaints()
    outs = [json.loads(l) for l in output_path(model).read_text(encoding="utf-8").splitlines() if l.strip()]
    outs = [o for o in outs if o["set"] == "eval" and not o["fallback"]]
    sample = random.Random(SEED).sample(outs, min(n, len(outs)))
    done = _read(SUMMARY_REVIEW)
    print("Rate each summary: f = faithful, m = minor issue (vague/missing key fact), h = hallucinated (states something not in the complaint).")
    for k, o in enumerate([o for o in sample if o["complaint_id"] not in done], len(done) + 1):
        print("\n" + "=" * 100 + f"\n[{k}/{len(sample)}] complaint {o['complaint_id']}")
        _show(complaints[o["complaint_id"]]["consumer_complaint_narrative"])
        print("\nMODEL SUMMARY: " + o["triage"]["summary"])
        r = _ask("f/m/h: ", {"f", "m", "h", "q"})
        if r == "q": break
        _append(SUMMARY_REVIEW, ["complaint_id", "model", "rating", "notes"],
                {"complaint_id": o["complaint_id"], "model": model, "rating": {"f": "faithful", "m": "minor issue", "h": "hallucinated"}[r],
                 "notes": input("notes (optional): ").strip()})
    print(f"Saved to {SUMMARY_REVIEW}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "label"
    {"make-set": make_set, "label": label, "adjudicate": adjudicate, "summaries": summaries}[cmd]()
