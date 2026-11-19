"""Pydantic output schema for a triage decision.

`Literal` types make the schema the single source of truth: a hallucinated category
("Mortgage servicing") or urgency ("critical") fails validation instead of silently
flowing into the routing queue.
"""
from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

Category = Literal[
    "Fraud & unauthorized transactions",
    "Billing, fees & payment disputes",
    "Account access & management",
    "Credit reporting & identity issues",
    "Loans & debt collection",
    "Customer service & communication",
    "Other / unclear",
]
Urgency = Literal["low", "medium", "high"]

CATEGORIES: tuple[str, ...] = get_args(Category)
URGENCIES: tuple[str, ...] = get_args(Urgency)
FALLBACK_CATEGORY = "Other / unclear"


class Triage(BaseModel):
    """What the LLM must return. Extra keys are rejected."""
    model_config = ConfigDict(extra="forbid")

    category: Category
    urgency: Urgency
    summary: str = Field(min_length=10, max_length=400, description="1-2 sentences, facts from the complaint only")
    requires_human: bool
    confidence_note: str = Field(max_length=300, description="what is uncertain, or 'clear'")

    @field_validator("summary")
    @classmethod
    def at_most_two_sentences(cls, v: str) -> str:
        # Count sentence ends, ignoring decimals like "$1.50".
        ends = sum(1 for i, ch in enumerate(v) if ch in ".!?" and (i + 1 == len(v) or v[i + 1] == " "))
        if ends > 3:   # small tolerance for abbreviations ("U.S. Bank")
            raise ValueError("summary must be 1-2 sentences")
        return v.strip()


def fallback(reason: str) -> Triage:
    """Safe default when the model can't produce a valid answer: route to a human, never guess."""
    return Triage(category=FALLBACK_CATEGORY, urgency="high", summary="Automatic triage failed; needs manual review.",
                  requires_human=True, confidence_note=f"fallback: {reason}"[:300])
