# AI Usage Log

How I used AI tools on this assignment, kept as I go rather than written from memory at the end.

## Tools

| Tool | Used for |
|---|---|
| Claude Code (in VS Code) | Planning, scaffolding, code generation, review |
| Groq-hosted open models | Part B: the LLM that does the complaint triage (product component, not a coding aid) |

## Log

| Date | Step | What the AI did | What I did / checked |
|---|---|---|---|
| 2026-11-18 | Step 0 – setup | Created repo structure, `.gitignore`, `requirements.txt`, `.env.example`, this file; created GitHub repo and pushed | Reviewed every file; chose the structure and tooling in `PayGuard-Plan.md` |
| 2026-11-18 | Groq check | Listed available Groq models; found the planned default (`llama-3.3-70b-versatile`) no longer offered, switched to `openai/gpt-oss-120b` | Created the API key myself |
| 2026-11-18 | A1 – data | Wrote `part_a_fraud/src/data.py` (download, dataset checks, dedup, time split) | Downloaded the Kaggle copy manually (OpenML download was too slow); checked split summary |
| 2026-11-18 | A2 – EDA | Generated and executed `fraud_analysis.ipynb` (imbalance, amount, time, duplicates, PCA features) | Reviewed every chart and conclusion against the outputs |

## Things I did not delegate to AI

- **Hand labels for the Part B evaluation set.** Labeled by me, following the labeling guide written beforehand, so the evaluation isn't an AI grading an AI.
- _(add as you go)_

## Where AI output was wrong or needed changes

- **EDA, first draft:** conclusions were written before the outputs were checked. The amount histogram used density on log-width bins, which made large amounts look negligible; it was replaced with a "% of class" chart. Feature names in the text didn't match the computed ranking, and "almost all" duplicates kept together was actually "all" (0 of 773 groups). All fixed after checking the outputs.
- _(add as you go)_

## Rough share of AI-generated code

_To fill in at the end: approximate % of code AI-generated, and how it was reviewed._
