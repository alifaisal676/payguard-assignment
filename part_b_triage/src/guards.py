"""Deterministic safety guards that run outside the LLM.

The LLM can be wrong, manipulated by prompt injection, or silently changed by its provider.
These checks don't depend on it:

1. keyword_backstop: regex rules that FORCE requires_human=True, whatever the LLM said.
   High recall by design; the cost is a higher escalation rate, which evaluate.py measures.
2. grounding_issues: numbers, amounts and dates in the summary that don't appear in the
   complaint, which flags possible hallucination.
3. mask_pii: masks emails, phone numbers, card/account numbers and SSNs before text leaves
   the company. CFPB data is already redacted ("XXXX"), but production data won't be.

To add a rule: add a (name, pattern) pair to RULES. Patterns are case-insensitive.
"""
from __future__ import annotations

import re

RULES: list[tuple[str, str]] = [
    ("fraud", r"\b(fraud\w*|scam\w*|stolen|steal\w*|unauthori[sz]ed|hack(ed|er|ing)?|phishing|"
              r"compromised|did not (make|authori[sz]e)|didn'?t (make|authori[sz]e)|never (made|authori[sz]ed))\b"),
    ("identity_theft", r"\b(identity theft|stole my identity|identity was stolen|opened in my name)\b"),
    ("legal", r"\b(lawyer|attorney(?! general)|lawsuit|sue|suing|sued|court|legal action|litigation|small claims)\b"),
    # Every complaint in this dataset was filed with the CFPB, so "CFPB" itself is not a signal
    # here; other regulators and law enforcement are. In production, add the local regulator.
    ("regulator_or_police", r"\b(attorney general|FTC|federal trade commission|OCC|FDIC|federal reserve|"
                            r"regulator\w*|police|law enforcement|FBI|secret service|police report)\b"),
    ("vulnerable", r"\b(elderly|senior citizen|disabled|disability|dementia|alzheimer\w*|cancer|chemo\w*|"
                   r"terminal\w*|hospital\w*|homeless|evict\w*|foreclos\w*|repossess\w*|domestic (violence|abuse))\b"),
    ("self_harm", r"\b(suicid\w*|kill (myself|me)|end (my|it) (life|all)|self[- ]harm|want to die|"
                  r"don'?t want to live|no reason to live)\b"),
]
_COMPILED = [(name, re.compile(p, re.IGNORECASE)) for name, p in RULES]


def keyword_backstop(text: str) -> list[str]:
    """Names of the rules that fire on this complaint (empty list = no forced escalation)."""
    return [name for name, rx in _COMPILED if rx.search(text)]


def apply_guards(text: str, llm_requires_human: bool) -> tuple[bool, list[str]]:
    """Final requires_human = LLM decision OR any rule firing. Returns (final, reasons)."""
    reasons = keyword_backstop(text)
    return (llm_requires_human or bool(reasons)), reasons


# ---------- summary grounding ---------------------------------------------------------
_NUM = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}|\$?\d[\d,]*(?:\.\d+)?")   # whole dates first, then amounts/numbers


def _norm_number(s: str) -> str:
    s = s.replace("$", "").replace(",", "")
    return s[:-3] if s.endswith(".00") else s


def grounding_issues(summary: str, complaint: str) -> list[str]:
    """Numbers/amounts/dates in the summary that are not in the complaint text."""
    in_text = {_norm_number(m) for m in _NUM.findall(complaint)}
    issues = []
    for m in _NUM.findall(summary):
        n = _norm_number(m)
        if n and n not in in_text:
            issues.append(m)
    return issues


# ---------- PII masking -----------------------------------------------------------------
PII_PATTERNS = [   # order matters: more specific patterns first
    ("EMAIL", r"[\w.+-]+@[\w-]+\.[\w.]+"),
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("PHONE", r"(?:\+?1[ .-]?)?\(?\b\d{3}\)?[ .-]?\d{3}[ .-]\d{4}\b"),
    ("CARD_OR_ACCOUNT", r"\b\d(?:[ -]?\d){7,18}\b"),
]
_PII = [(label, re.compile(p)) for label, p in PII_PATTERNS]


def mask_pii(text: str) -> tuple[str, dict]:
    counts = {}
    for label, rx in _PII:
        text, n = rx.subn(f"[{label}]", text)
        if n:
            counts[label] = n
    return text, counts


if __name__ == "__main__":
    demo = ("I never authorized these charges. My attorney says we will sue. Call me at 555-123-4567 "
            "or j.doe@mail.com; card 4111 1111 1111 1111.")
    print(keyword_backstop(demo))
    print(mask_pii(demo))
    print(grounding_issues("Customer lost $1,250 on 03/12/2025.", "they took {$1250.00} on XX/XX/XXXX"))
