# Customer Intelligence & Support Copilot

## Overview

This repository implements the four-part synthetic Instamart capstone: (A) clean and store raw customer/order/product data, (B) segment customers and predict lapse, (C) redact and classify support tickets with guardrails, and (D) build a TF-IDF policy assistant and a resumable ReAct refund agent. All records and policies are synthetic. All model calls use the supplied offline `mock_llm.py`; no network, API key or paid service is used.

## Reproduce

Run from the repository root with Python 3.11 or newer. `generate_data.py` and `mock_llm.py` are the supplied files and are committed unchanged.

1. `python -m venv .venv`
2. Windows: `.venv\Scripts\activate` (macOS/Linux: `source .venv/bin/activate`)
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_all.py`

The generator prints exactly `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. `run_all.py` executes Parts A–D in dependency order and runs `src/validate.py`; the validation script prints `ALL CHECKPOINTS PASSED`. Numerical outputs are deterministic under the fixed seeds in the scripts; ledger timestamps are generated at execution time.

## Part A

### Raw-data audit

| Raw table | Shape (rows × columns) | Column data types |
|---|---:|---|
| customers | 1200 × 5 | customer_id: object; city: object; signup_date: object; age: object; gender: object |
| products | 40 × 4 | product_id: object; product_name: object; category: object; price_inr: int64 |
| orders | 13084 × 9 | order_id: object; customer_id: object; product_id: object; quantity: int64; order_ts: object; status: object; discount_pct: int64; order_amount: object; delivery_minutes: object |

| Table | Column | Missing/blank count | Missing/blank % |
|---|---|---:|---:|
| customers | customer_id | 0 | 0.00% |
| customers | city | 0 | 0.00% |
| customers | signup_date | 0 | 0.00% |
| customers | age | 61 | 5.08% |
| customers | gender | 30 | 2.50% |
| products | product_id | 0 | 0.00% |
| products | product_name | 0 | 0.00% |
| products | category | 0 | 0.00% |
| products | price_inr | 0 | 0.00% |
| orders | order_id | 0 | 0.00% |
| orders | customer_id | 0 | 0.00% |
| orders | product_id | 0 | 0.00% |
| orders | quantity | 0 | 0.00% |
| orders | order_ts | 0 | 0.00% |
| orders | status | 0 | 0.00% |
| orders | discount_pct | 0 | 0.00% |
| orders | order_amount | 0 | 0.00% |
| orders | delivery_minutes | 2541 | 19.42% |

| Raw audit item | Count |
|---|---:|
| Raw orders rows | 13084 |
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct order IDs | 12309 |
| Unparseable order amounts | 120 |
| Negative delivery minutes | 106 |
| Blank delivery minutes | 2541 |
| Blank customer ages | 61 |
| Numeric ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank gender spellings | 10 |
| Blank gender values | 30 |

### Cleaning results

Customer cleaning normalised 476 city values and mapped 950 gender values to their canonical form or `unknown`; 61 blank ages were filled and 6 out-of-range numeric ages were replaced before city-median imputation. Signup dates were parsed to datetime. The clean city counts were: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Gender counts: female 610, male 560, unknown 30. All 1,200 ages are present after cleaning.

Order cleaning reduced the raw 13,084 rows to 12,309 unique orders: 11,110 completed and 1,199 cancelled. The pipeline removed 119 exact duplicate rows and a further 656 earlier order-status versions, retaining the latest timestamp for each of 12,309 order IDs. It repaired 120 unparseable amounts using catalogue price, quantity and discount; converted 106 negative delivery durations to missing; and imputed 774 missing completed-order delivery times by product-category median. Completed orders have 0 missing delivery times; all 1,199 cancelled orders correctly retain missing delivery times. Completed-order revenue is INR 2,998,668.80.

### SQL analysis

| Query | Result |
|---|---|
| Q1 | 30 customers after WHERE and HAVING |
| Q2 | 30 city/category rows; grand total INR 2,998,668.80; largest cell Bengaluru / Electronics = INR 233,909.45 |
| Q3 | 11 products above average catalogue price |
| Q4 | 25 customers without a completed order |
| Q5 | 550 customers above average spend; mean customer spend INR 2,552.06 |
| Q6 | 8 rows have RANK different from ROW_NUMBER |
| Q7 | 183 daily rows; final cumulative revenue INR 2,998,668.80 |

Q6 city leaders:

| City | Customer | Completed spend |
|---|---|---:|
| Bengaluru | C0082 | INR 12,856.20 |
| Chennai | C0930 | INR 10,151.40 |
| Delhi | C0291 | INR 13,742.60 |
| Hyderabad | C0743 | INR 12,730.05 |
| Mumbai | C0693 | INR 14,766.55 |
| Pune | C0426 | INR 10,377.10 |

The `RANK()` result can differ from `ROW_NUMBER()` when two customers tie on spend: RANK assigns the same position to tied values and leaves a gap, while ROW_NUMBER always assigns a unique sequential number.

Pandas cross-check: `pivot_table(..., margins=True)` returns grand total INR 2,998,668.80, matching SQL Q2.

| City | Completed revenue | Mean order amount | Order count |
|---|---:|---:|---:|
| Bengaluru | INR 523,643.40 | INR 276.47 | 1,894 |
| Chennai | INR 463,629.15 | INR 272.72 | 1,700 |
| Delhi | INR 502,829.80 | INR 243.62 | 2,064 |
| Hyderabad | INR 528,589.90 | INR 284.95 | 1,855 |
| Mumbai | INR 479,058.50 | INR 297.00 | 1,613 |
| Pune | INR 500,918.05 | INR 252.48 | 1,984 |

### Exploratory analysis

Completed order amount: mean INR 269.91; median INR 135.00; skewness 4.61; Q1 INR 64.80; Q3 INR 297.00; lower fence INR -283.50; upper fence INR 645.30; outliers below/above = 0/897.

This dataset describes completed grocery and small-electronics orders across six cities. Order amounts are strongly right-skewed: the mean is much higher than the median, and 897 values exceed the upper IQR fence, reflecting a smaller number of high-value baskets. The cleaning process also identifies duplicated exports, pending status versions, malformed amounts and impossible delivery durations. In the completed-order correlation matrix, order amount and discount percentage have the largest absolute correlation (about −0.095); order amount and quantity are about +0.087. Both relationships are weak, and correlation of this size does not establish causation.

## Part B

### RFM segmentation

The 2025-10-01 snapshot contains 1,175 customers with at least one completed order. Recency median/min/max = 19/0/180 days; frequency mean/max = 9.46/33; monetary mean/max = INR 2,552.06/INR 14,766.55.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 was selected because its silhouette score is the strongest of the tested choices and it gives four understandable groups. Inertia always falls as K rises because more centroids can fit the data more closely; that mathematical decrease alone does not justify choosing K=8, which could over-fragment customers and make campaigns less actionable.

| Segment ID | Business name | Customers | Mean recency (days) | Mean frequency | Mean monetary | Premium? |
|---:|---|---:|---:|---:|---:|---|
| 0 | Regular Value | 436 | 28.86 | 3.77 | INR 1,057.62 | No |
| 1 | Premium Value | 138 | 24.12 | 6.77 | INR 7,129.70 | Yes |
| 2 | Frequent Essentials | 419 | 8.77 | 19.40 | INR 3,410.02 | No |
| 3 | At-Risk Occasional | 182 | 109.04 | 2.20 | INR 685.99 | No |

Recommended actions: Premium Value — VIP service and retention offers; Frequent Essentials — replenishment reminders and bundles; Regular Value — relevant cross-sell offers; At-Risk Occasional — win-back incentives. Cluster IDs are arbitrary labels, not an ordered scale. PCA explained variance ratios are 0.634, 0.218 (total 0.852).

### Lapse prediction

The pre-cutoff cohort contains 1,156 customers; 443 lapsed (38.32%). The stratified split is 924 training and 232 test rows. Always predicting not-lapsed gives test accuracy 61.6% and lapsed-class F1 0. No feature uses orders dated 2025-09-01 or later, and the segment labels from RFM were not used as features. The categorical encoder and numeric scaler were fitted only on training rows; test rows were transformed only. No encoder, scaler or resampler saw test rows. A missing-feature assertion passed.

The fitted training scaler's frequency mean is 8.115, versus 8.029 for the full cohort; the values differ because the scaler was fitted on the training subset only, not on the full cohort.

| Model | Best parameters | CV F1 | Test confusion matrix [[TN, FP], [FN, TP]] | Precision | Recall | F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | `{'model__C': 100}` | 0.651 | `[[111, 32], [31, 58]]` | 0.644 | 0.652 | 0.648 |
| Random forest | `{'model__max_depth': 5, 'model__min_samples_leaf': 5}` | 0.681 | `[[114, 29], [33, 56]]` | 0.659 | 0.629 | 0.644 |

The selected model is Logistic Regression because its untouched-test lapsed-class F1 is 0.648, slightly higher than the Random Forest value of 0.644. Missing a lapsing customer means losing future revenue, while wrongly targeting a loyal customer consumes campaign budget and may annoy them. The model balances these costs; the retention team can choose a more recall-heavy threshold if missed lapses are especially expensive.

Training-only random oversampling of the logistic model changes test precision/recall/F1 to 0.589/0.854/0.697. Recall increases the most (from 0.652 to 0.854), meaning more genuinely lapsing customers are captured, at the cost of more false positives.

Top five Random Forest importances: frequency (0.305), monetary (0.279), recency (0.113), cancellation_rate (0.072), avg_discount (0.048). Logistic regression coefficient signs: frequency -1.697, favourite_category_Electronics -1.135, favourite_category_Grocery -0.449. Negative frequency indicates that more frequent customers are less likely to lapse; the negative electronics/grocery coefficients are relative to the dropped reference category.

The selected model is saved to `outputs/lapse_model.joblib`; reloading it reproduces test predictions exactly. The Premium-segment retention list contains 10 test customers; 3 truly lapsed. The list is illustrative and not a leakage-free evaluation because Premium is a post-hoc whole-window RFM label.

## Part C

### Privacy, guardrails and validation

Redaction substitutions over the 164 tickets: 55 email addresses → `[EMAIL]`, 91 phone numbers → `[PHONE]`, and 132 order IDs → `[ORDER_ID]`, for 278 total; reapplying all three patterns to the redacted text found zero remaining matches. Four empty tickets were skipped and four long tickets were flagged; the long tickets estimate to 734 tokens each and are not sent to the model. The user message to the mock model is exactly the redacted ticket text.

| Full run | Classified | Accuracy | Correct / classified | Total tokens | Tokens / classified | Retry errors recovered |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% | 126/147 | 11,584 | 78.80 | 19 |
| Few-shot | 147 | 100.0% | 147/147 | 18,916 | 128.68 | 19 |

Status counts for each full run: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 `flagged_long`, and 4 `skipped_empty`. All 19 transient failures were recovered by retry. A keyword reply is accepted only if regex extraction finds exactly one of the four allowed category words; valid JSON with an unsupported category is not accepted.

Zero-shot confusion table (rows = actual category, columns = predicted category):

| Actual \ Predicted | billing | delivery | technical | fraud |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

Few-shot costs 49.88 additional tokens per classified ticket (63.3% more) but raises accuracy from 85.7% to 100.0%, recovering 21 additional correct classifications out of 147. For a support workflow where fraud and technical misrouting can have a high operational cost, this extra prompt cost is justified; a high-volume low-risk queue could prefer zero-shot and send uncertain cases to a human.

The zero-shot budget demonstration uses TOKEN_BUDGET=6,000 and ends with 75 tickets marked `budget_exhausted`; the run stops before another model call once accumulated usage reaches the budget. The final few-shot routing run uses TOKEN_BUDGET=100,000 (18,916 tokens used; 0 tickets exhausted) and its routing output has 164 rows, with guardrail/invalid/budget cases routed to human review, fraud to the security team, Premium customers to the premium desk, and the remainder to the support queue.

## Part D

### Policy RAG

The 5 policy files produce 15 chunks with 40 words per chunk and 10-word overlap: refund 4, delivery 3, escalation 3, payment 3, privacy 2. For the 10-word-overlap evaluation, expected document retrieval is correct for 12/12 answerable questions, the answer contains the expected fact for 12/12, and the two unanswerable questions receive the `I could not find` fallback (2/2). With zero overlap, expected document retrieval remains 12/12 but only 10/12 answers contain the expected fact; both unanswerable questions still fall back (2/2). The questions “What delivery time makes a completed order eligible for a full refund?” and “How soon is money returned when a payment fails but is deducted?” break in the zero-overlap run because the key fact is split across a chunk boundary and the model sees only incomplete context.

### Refund agent

| Scenario | Status | Expected refund | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | INR 128.25 | 5 | 932 |
| S2 | REFUND | INR 135.00 | 5 | 932 |
| S3 | ESCALATE | INR 1,999.00 | 5 | 919 |
| S4 | DENIED | INR 0.00 | 4 | 633 |
| S5 | DENIED | INR 0.00 | 3 | 358 |
| S6 | NOT_FOUND | INR 0.00 | 2 | 86 |
| S7 | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 | 3,042 |

Only completed orders delivered in more than 45 minutes are eligible for a full order-amount refund; 45 minutes or less yields zero. Amounts above INR 1,000 are escalated for human approval. S1 and S2 were refunded INR 128.25 and INR 135.00; S3 was escalated at INR 1,999.00; S4 and S5 were denied; S6 returned NOT_FOUND; S7 reached `FALLBACK_MAX_ITERATIONS` after all six calls.

Stop-safely demonstrations: S1 with `token_limit=200` returned `FALLBACK_TOKEN_LIMIT` (3 model calls, 375 accumulated tokens; the third tool request was not executed); S1 with `time_limit_seconds=0` returned `FALLBACK_TIMEOUT` before the first model call (0 calls, 0 tokens).

Crash/resume: the simulated crash occurred after the second tool observation was checkpointed. The exact tool execution sequence across the initial run and resume was `get_order` → `search_policy` → `calculate_refund` → `issue_refund`; no tool was repeated. The resumed run returned `REFUND: INR 128.25 issued.` and the checkpoint records the run ID, scenario ID, complete messages, token count, step, tools executed and step observations.

**ReAct versus plan-and-execute (under 250 words).** ReAct made 5 model calls for S1 (932 tokens), 3 for S5 (358 tokens), and 2 for S6 (86 tokens), because it requests the next tool only after observing the previous result. A plan-and-execute design could ask for the full sequence once and save calls and prompt tokens on a straightforward successful order. However, it risks following a stale plan when `get_order` returns NOT_FOUND or when an order is cancelled; it would need replanning or additional deterministic checks before any refund action. For this workflow I prefer ReAct because refund decisions depend on intermediate database and policy observations, and the extra calls are acceptable for a safe internal workflow. The deterministic tool rules and human escalation remain essential regardless of orchestration style.

## Limitations and Next Steps

All data and policies are synthetic and must not be interpreted as actual company information. TF-IDF is lexical rather than semantic; a production system would need stronger retrieval evaluation, document versioning and access controls. The offline mock LLM is deterministic and does not represent real-world language-model quality. The lapse model is trained on a synthetic six-month history and should be monitored for drift before operational use. No orders dated 2025-09-01 onward entered model features, and the segment labels were excluded from the feature set. The retention list is illustrative and not a leakage-free evaluation because its Premium flag is assigned from whole-window RFM, which includes the September label window. A production version should compute Premium strictly before the lapse cutoff, validate on future time periods, and require human approval for refunds above INR 1,000.
