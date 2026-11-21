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
| 2026-11-18 | A3–A5 – models | Wrote `features.py`, `train.py`, `evaluate.py` (bootstrap CIs, paired difference, alert-budget and val-threshold metrics); notebook section 2 | Reviewed design choices: alert budget 0.2%, precision target 80%, no tuning |
| 2026-11-19 | A6–A8 – cost & real-time | Wrote `cost.py` (cost curve, amount-aware rule, tiers, sensitivity), notebook sections 3–5, README Part A, unit tests | Chose the cost assumptions to state; checked the cost curve against a brute-force calculation |
| 2026-11-19 | B1 – data | Found that CFPB removed narratives from its API (Aug 2026); rewrote `fetch_data.py` to sample the Hugging Face mirror and verify IDs against the official CFPB API | Approved the change of data source |
| 2026-11-19 | B2–B8 – triage system | Wrote taxonomy/prompt, schema, LLM client, guards, pipeline, labeling tool, labeling guide, evaluation | Reviewed taxonomy and policies; **hand labels: done by me (see below)** |
| 2026-11-19 | C – memo | Drafted `memo.md` | Reviewed recommendations, thresholds and asks |

## My role

I used Claude Code as a fast pair programmer and directed it step by step. The decisions, checks and human judgement were mine:

- **Planning and scope:** wrote the project plan that set the order of work, the metrics, the evaluation design and what to cut, and approved each step before it was built.
- **Data:** downloaded and supplied the Kaggle fraud dataset when the automatic download was too slow; approved switching Part B to the verified Hugging Face mirror after CFPB removed the narratives.
- **Environment and provider:** set up the Python environment, created the Groq API key, and kept the key out of the repo.
- **Evaluation design:** chose to label by stratified sampling (every model disagreement plus a random sample of agreements) instead of labeling all 100, which is how labeling is done at scale, and chose the adjudication step to check my labels against the written guide.
- **Hand labeling:** labeled 57 complaints myself (details below).
- **Review:** reviewed every result, chart and write-up; several AI mistakes listed below were caught during this review.

## Things I did not delegate to AI

- **Hand labels for the Part B evaluation set.** Labeled by me in the terminal tool (`labeling.py`), following the labeling guide written beforehand, without seeing the model's output or CFPB's product/issue fields, so the evaluation isn't an AI grading an AI. **My decision:** rather than label all 100, label every case where the two models disagree plus a random sample of agreement cases (stratified sampling). The AI implemented the sampling and the re-weighted estimate.
- **Label adjudication: done by AI at my request, and disclosed.** 43 of my 57 labels broke an explicit rule of the written guide. I asked Claude to re-check them against the guide (complaint text and guide only, no model output shown; it had seen some model predictions earlier in the session). It changed 38 and kept 5, with a reason for each in `labels/eval_labels.csv`. My untouched labels are in `labels/eval_labels_original.csv`, and both results are reported. The adjudicated labels are an AI's reading of the guide, so the adjudicated accuracy is an optimistic bound, not an independent human measurement.
- The LLM that does the triage (Groq gpt-oss) is a *component under test*, not a coding aid; its outputs are evaluated, never used as labels.

## Where AI output was wrong or needed changes

- **EDA, first draft:** conclusions were written before the outputs were checked. The amount histogram used density on log-width bins, which made large amounts look negligible; it was replaced with a "% of class" chart. Feature names in the text didn't match the computed ranking, and "almost all" duplicates kept together was actually "all" (0 of 773 groups). All fixed after checking the outputs.
- **LightGBM class weighting:** the plan's `scale_pos_weight ≈ 497` made the model early-stop after 3 trees. Compared settings on validation and switched to 1. While debugging, an exploratory script also printed *test* scores; the decision rests on validation alone, and this is disclosed in the notebook.
- **`Time` as a feature:** the first model used raw `Time`. The SHAP plot showed it pushing every test row towards "legit" (an artefact of test always being later than train). Removed on principle; all numbers and text were regenerated afterwards.
- **Cost analysis:** I first predicted the cost-optimal threshold would "barely move" with the cost assumptions; the sensitivity run showed it jumps between plateaus, and the write-up was corrected. The amount-aware rule's saving was first described without its CI; the CI includes 0, so it is now reported as "promising, not proven".
- **Guards:** first version masked phone numbers as card numbers and split dates in the grounding check; caught by a demo run and fixed, then covered by unit tests.
- **Prompt:** first summaries copied the customer's first person ("my card"); prompt changed to third person before any evaluation run.
- **Rate limits:** mid-run I concluded from a token count that Groq had no hard daily cap; the provider's own error later confirmed a rolling 200k tokens/day limit. The README states the confirmed limit.
- **Injection result:** the first report counted 2/8 successful injections against gpt-oss-120b. A clean re-run showed one of those changes also happens with no attack (run-to-run variation), so it's reported as 1 clear + 1 within noise.
- **Memo diagram:** Mermaid code reused one ID for a node and a subgraph (would break rendering); fixed.

## Rough share of AI-generated code

Roughly **85%** of the code was generated by Claude Code. I chose the design, approved each step, did the hand labels, and reviewed every file and result. The corrections are listed above.
