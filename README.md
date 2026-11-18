# PayGuard Assignment

Feasibility build for PayGuard: **(A)** card fraud detection, **(B)** LLM complaint triage, **(C)** production design memo ([memo.md](memo.md)).

> Work in progress. Sections below are filled in as each part is completed.

## Setup (Python 3.12)

```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                  # then add your GROQ_API_KEY
```

## Repository layout

```
part_a_fraud/    fraud detection: notebook + src (data, features, train, evaluate)
part_b_triage/   complaint triage: src (fetch, schema, prompts, client, guards, pipeline, evaluate), data/, labels/
memo.md          production rollout memo to the CTO
AI_USAGE.md      how AI tools were used
```

## Part A: Fraud detection
_TBD_

## Part B: Complaint triage
_TBD_

## What I cut and why
_TBD_
