# Labeling guide: complaint triage evaluation set

Written **before** labeling, so the labels follow fixed rules rather than whatever the model happens to output. The same definitions are in the system prompt (`src/prompts.py`); the model and the labeler use one taxonomy.

## Process

- 100 complaints, stratified across six product groups (`labels/eval_set.csv`, seed 7), shown in random order.
- Labeled by a human (the author) with `python -m part_b_triage.src.labeling label`.
- The labeler sees **only the complaint text**: not the model's output, and not CFPB's product/issue fields. Those are used separately as weak labels, so hiding them keeps the two comparisons independent.
- Label the complaint as **the customer describes it**. Don't judge whether they are right.
- If unsure between two categories, apply the tie-break order below and write a note.

## 1. Category: which team must act first?

| # | Category | Use when | Not when |
|---|---|---|---|
| 1 | **Fraud & unauthorized transactions** | Charges/transfers the customer didn't make or authorise; stolen or compromised card/account; scam where money left the account | Customer made the purchase but disputes it (→ 2); identity theft that only shows up on a credit report (→ 4) |
| 2 | **Billing, fees & payment disputes** | Wrong amount, unexpected fee/interest, refund not received, payment not applied, dispute of a purchase they did make | |
| 3 | **Account access & management** | Can't log in; account frozen, closed or restricted; funds held; problems opening/closing an account | |
| 4 | **Credit reporting & identity issues** | Wrong items on a credit report; credit-report disputes; identity theft that created accounts or inquiries in their name | Fraudulent *transactions* on an existing account (→ 1) |
| 5 | **Loans & debt collection** | Loan servicing (personal, auto, student, mortgage); collection calls/letters; debt validation | Collection item on credit report is the main complaint (→ 4) |
| 6 | **Customer service & communication** | The main problem is the service itself: no reply, rude staff, endless transfers, broken promises | A concrete money/account problem is also present (→ that category) |
| 7 | **Other / unclear** | Doesn't fit, or too unclear to route | |

**Tie-break when several apply:** fraud > account access > billing > credit reporting > loans > customer service.

## 2. Urgency

- **high:** money is actively being lost or the customer can't access their funds now; fraud in progress; account locked/frozen; legal deadline, foreclosure, eviction, repossession; vulnerable customer (elderly, serious illness, disability, stated financial hardship); any self-harm language.
- **medium:** unresolved financial impact (wrong charge, fee, credit-report error) without immediate harm.
- **low:** informational, service quality only, or already resolved.

When torn between two levels, pick the **higher** one: under-triage is worse than over-triage.

## 3. requires_human (should a person handle this before any automated action?)

**Yes** if any of: fraud/scam allegation; identity theft; lawyer, attorney, lawsuit, court; regulator or police involvement (beyond the CFPB complaint itself); vulnerable customer; self-harm language; or you yourself found the category genuinely ambiguous. Otherwise **no**.

## 4. sensitive

**Yes** if the complaint involves fraud/identity theft, legal threats, regulator/police, a vulnerable customer or self-harm, i.e. a case where a misroute could cause real harm or legal exposure. Mostly overlaps with requires_human, but excludes "ambiguous category" cases. This is the set where we target **100% recall**.

## 5. Summary review (`labeling summaries`, 30 summaries)

- **faithful:** every statement is supported by the complaint.
- **minor issue:** vague or misses a key fact, but states nothing false.
- **hallucinated:** states something not in the complaint (a number, date, party or event).

## Known ambiguities (decisions made up front)

- "Bank closed my account and kept my money" → 3 (access), urgency high.
- "Charged twice, merchant won't refund, bank denied dispute" → 2.
- "Accounts opened in my name appear on my credit report" → 4, requires_human yes (identity theft).
- "Debt collector reporting a debt I don't owe" → 4 if the complaint is mainly about the report, 5 if mainly about collection behaviour.
- "Zelle/Cash App scam: I sent the money myself after being tricked" → 1 (scam where money was taken).
