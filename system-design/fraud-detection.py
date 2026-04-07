import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def header(mo):
    mo.md("""
    # ML System Design: Fraud Detection Pipeline

    | Field  | Value |
    |--------|-------|
    | Date   | 2026-04-07 |
    | Track  | System Design |
    | Time   | 60 min |
    | Topics | Feature Store · Training Pipeline · Real-Time Serving · Drift Monitoring · Class Imbalance |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 1: Problem Framing

    **The prompt:**
    > "Design a fraud detection system for a payment processing platform handling 10M transactions per day."
    > The system must flag potentially fraudulent transactions in real-time, before they are approved.

    ---

    ### Clarifying Questions & Assumptions

    | Question | Reasonable Assumption |
    |---|---|
    | What types of fraud? | Card-not-present (online), stolen credentials, account takeover, friendly fraud. Assume all types. |
    | What's the fraud rate? | ~0.1–0.5% of transactions. Massive class imbalance — 200:1 to 1000:1 ratio. |
    | Latency budget? | **<100ms per transaction.** Hard requirement — user is waiting at checkout. |
    | Cost asymmetry? | False negative (missed fraud): ~$500 avg chargeback + reputation damage. False positive (blocking legit): customer friction + lost revenue. Both expensive, but **missed fraud is worse**. |
    | What actions can the system take? | Approve, decline, send to manual review queue, step-up authentication (2FA). |
    | Is there a human review team? | Yes — ~50 analysts reviewing ~500 flagged transactions per day. |

    ---

    ### Success Metrics

    **Online metrics** (what the business tracks):
    - Fraud loss rate: $ lost to fraud / total $ processed
    - False positive rate on legitimate transactions (customer friction proxy)
    - Manual review queue utilization (capacity of the human team)

    **Offline metrics** (what we use to evaluate models):
    - **Precision@K**: of the top K flagged transactions, how many are actual fraud?
    - **Recall**: what % of fraud do we catch?
    - **AUC-PR** (precision-recall curve) — **NOT AUC-ROC**

    **Business metric** (the north star):
    - Net fraud savings = (fraud prevented) − (lost revenue from false declines) − (manual review cost)

    ---

    ### Key Insight to State Upfront

    > "We optimize for **AUC-PR**, not AUC-ROC. With a 0.1% fraud rate, a model that predicts
    > 'not fraud' for everything achieves 99.9% accuracy and excellent AUC-ROC — but catches
    > **zero fraud**. Precision-recall curves are robust to class imbalance because they ignore
    > true negatives entirely."
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 2: Data

    ### Available Data Sources

    | Source | Examples |
    |---|---|
    | **Transaction data** | Amount, merchant, category, timestamp, payment method, currency, billing/shipping address |
    | **User profile** | Account age, historical transaction patterns, devices used, login history |
    | **Device/session** | IP address, device fingerprint, geolocation, browser/app info |
    | **External signals** | Card issuer risk scores, IP reputation databases, known fraud blacklists |
    | **Historical labels** | Chargeback data (delayed 30–90 days), manual review decisions, rule-based flags |

    ---

    ### Labeling Challenges — This Is Where Fraud Gets Hard

    **1. Labels are DELAYED.**
    Chargebacks arrive 30–90 days after the transaction. You are always training on stale labels.
    Your model learns from what was fraudulent a month ago, not what's fraudulent today.

    **2. Labels are INCOMPLETE.**
    Not all fraud results in a chargeback. Some fraud goes completely undetected.
    Your "not fraud" class contains mislabeled fraud examples.

    **3. Feedback loops create selection bias.**
    You only observe outcomes for **approved** transactions.
    Declined transactions have no ground truth — was it actually fraud or a false positive?
    This is a survivorship bias problem baked into the training data.

    **4. Friendly fraud.**
    Customer claims fraud but actually made the purchase.
    Deliberately or accidentally mislabeled — poisons the training set.

    **Solution strategy:**
    Combine chargeback labels + manual review labels + rule-based flags. Accept that labels are noisy.
    Use semi-supervised techniques for the unlabeled declined transactions.
    (And later: random exploration — more on this in Monitoring.)

    ---

    ### Data Pipeline

    - **Real-time stream**: Kafka/Kinesis ingesting transactions as they happen
    - **Batch ETL**: daily pull of historical transactions, chargebacks, user profiles into data lake
    - **Label pipeline**: join transactions with chargebacks (30–90 day delay), manual review outcomes,
      and rule-based flags to produce training labels
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 3: Feature Engineering

    ### Feature Categories

    **Transaction features** *(available at inference time, no lookup needed)*:
    - Raw: `amount`, `merchant_category`, `payment_method`, `hour_of_day`, `day_of_week`
    - Derived: `is_round_amount`, `amount_vs_merchant_avg`, `is_international`, `is_new_merchant_for_user`

    **User behavioral features** *(from feature store — pre-computed)*:
    - Velocity: `num_transactions_last_1h`, `num_transactions_last_24h`, `num_distinct_merchants_last_7d`
    - Spending patterns: `avg_transaction_amount_30d`, `std_transaction_amount_30d`, `max_transaction_last_24h`
    - Deviation: `amount_zscore_vs_user_history` — how unusual is THIS transaction for THIS user?
    - Account: `account_age_days`, `days_since_password_change`, `num_devices_used_last_30d`

    **Device/session features** *(real-time signals)*:
    - `is_new_device`, `is_new_ip`, `ip_country_matches_billing`
    - `distance_from_last_transaction_km`
    - `time_since_last_transaction_seconds` — rapid-fire transactions are highly suspicious

    **Graph/network features** *(expensive but powerful)*:
    - `shared_device_with_known_fraud_account`
    - `merchant_fraud_rate_30d`
    - `payment_network_clustering_coefficient` — fraud rings share devices, IPs, and addresses

    ---

    ### Feature Store Architecture

    **Offline store** (batch):
    User behavioral aggregates computed daily, merchant statistics, graph features.
    Stored in a columnar format (Parquet / Delta Lake). Used for **training**.

    **Online store** (real-time):
    Low-latency key-value store (Redis / DynamoDB).
    User features keyed by `user_id`, merchant features by `merchant_id`.
    Updated by a streaming pipeline (Flink / Spark Streaming) consuming the Kafka transaction stream.
    Used for **serving**.

    **Feature freshness tradeoff**:
    - Velocity features (last 1h) → need near-real-time updates from the streaming pipeline
    - 30-day aggregates → batch recompute daily is fine
    - The feature store bridges this gap so the serving layer never needs to compute aggregates inline

    ---

    ### Training-Serving Skew — The Most Common Production Bug

    Features are computed differently in batch (training) vs real-time (serving).
    Your model trains on clean, perfectly-computed features and then serves on approximate ones.

    **Solution**: use the same feature computation code for both paths,
    or use a managed feature store (Feast, Tecton) that enforces consistency.
    This is worth calling out explicitly in an interview.

    ---

    ### Architecture Diagram

    ```
    [Transaction Stream (Kafka)]
        → [Streaming Feature Engine (Flink)]
            → [Online Feature Store (Redis)]  ← serving reads here (<10ms)
        → [Batch Feature Engine (Spark)]
            → [Offline Feature Store (Delta Lake)]  ← training reads here
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 4: Model

    ### Model Progression — Always Start Simple

    **Baseline: Rules Engine** (already exists at most companies)
    - Hard rules: block if `amount > $10K AND new_device AND international`
    - Pros: interpretable, zero training required, fast, responds in minutes to new fraud patterns
    - Cons: rigid, easy for fraudsters to learn and evade, high false positive rate
    - **Keep this layer permanently** — it is the fast-response circuit breaker while the model catches up

    ---

    **V1: Gradient Boosted Trees (XGBoost / LightGBM)**

    Why start here, not deep learning:
    - Handles class imbalance well with `scale_pos_weight`
    - Fast inference: <5ms per prediction
    - Interpretable feature importance (SHAP values)
    - Works well on tabular data — this is the textbook use case
    - Smaller labeled dataset (fraud is rare) — GBTs generalize better with less data

    Handling imbalance:
    - `scale_pos_weight = num_negatives / num_positives` in XGBoost
    - SMOTE oversampling on the minority class in training data
    - Focal loss to down-weight easy negatives during training
    - Evaluation: **always precision-recall, never accuracy**

    Training cadence: daily retrain on a rolling 90-day window of labeled data.

    ---

    **V2: Two-Stage Model**

    - **Stage 1**: Fast, lightweight scorer (logistic regression or small GBT) scores ALL 10M transactions
      in <10ms. Routes the top 5% riskiest to Stage 2.
    - **Stage 2**: Heavier model (larger GBT or small neural net) with more features —
      graph features, deeper user history, richer device signals — on the 500K flagged transactions.

    Why: you cannot afford to compute graph features for 10M transactions/day in <100ms.
    You can afford it for 500K. Two-stage decouples recall coverage from precision refinement.

    ---

    **V3 (Future): Sequence Model**
    - LSTM or transformer over user's recent transaction sequence
    - Captures temporal behavioral patterns: normal user has coffee → commute → lunch;
      fraudster has rapid burst of purchases across unrelated merchants
    - Only worth building once V1/V2 plateau on the precision-recall curve

    ---

    ### Threshold Tuning

    The model outputs a fraud probability `p ∈ [0, 1]`.
    Different thresholds map to different business actions:

    | Probability | Action |
    |---|---|
    | p > 0.9 | Auto-decline |
    | 0.5 < p ≤ 0.9 | Send to manual review queue |
    | 0.3 < p ≤ 0.5 | Step-up authentication (require 2FA) |
    | p ≤ 0.3 | Auto-approve |

    Thresholds are tuned on **net fraud savings** (the business metric), not purely on ML metrics.
    The review queue threshold is also constrained by analyst capacity (~500/day).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 5: Serving Architecture

    ### Real-Time Inference Path

    ```
    [Transaction Request]
        → [API Gateway]
        → [Feature Lookup]        (online feature store: user + merchant features, <10ms)
        → [Rules Engine]          (hard rules: amount, velocity, blacklists, <1ms)
        → [ML Model Inference]    (XGBoost, <5ms)
        → [Decision Logic]        (threshold → approve / decline / review / 2FA)
        → [Response to client]    (total: <100ms)
        → [Async log]             (prediction + features → monitoring pipeline)
    ```

    ---

    ### Architecture Decisions

    **Model hosting**: Pre-load the XGBoost model directly in the inference service process
    (not a separate model server). XGBoost models are small enough to fit in memory.
    Avoids a network hop on the critical path — each hop adds latency variance.

    **Feature store reads**: Batch-read user + merchant features in one round-trip.
    Pre-fetch user features on session start when possible (user just logged in → warm the cache).

    **Fallback strategy**: If the ML service is unhealthy → fall back to rules engine only.
    Never block ALL transactions — that's worse than missing some fraud.
    Rules engine is always running in parallel as a safety net.

    **Shadow mode for new models**: New model runs alongside production and logs its predictions,
    but decisions are not acted on. Compare shadow predictions vs actual outcomes offline
    before promoting to champion. Zero-risk rollout.

    ---

    ### Scaling

    - **Inference service**: stateless, horizontally scaled behind a load balancer
    - **Feature store**: Redis cluster with read replicas per region
    - **Throughput**: 10M transactions/day = ~115 TPS average, ~500+ TPS at peak
      (holiday season, flash sales). Moderate scale — horizontal scaling is straightforward.
    - **Latency budget breakdown**: feature lookup 10ms + rules 1ms + model 5ms +
      network/overhead ~50ms = well within 100ms target
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 6: Monitoring & Drift

    ### What to Monitor — Where Most Candidates Fall Short

    **Model performance monitoring**:
    - Daily `precision@K` on manual review outcomes — fast feedback signal (hours, not days)
    - Weekly AUC-PR on confirmed chargeback labels — delayed but definitive ground truth
    - False positive rate tracked via customer complaint volume
    - Alert if `precision@100` drops below threshold → triggers investigation or retrain

    **Data/feature drift** (PSI — Population Stability Index):
    - Monitor feature distributions daily
    - Transaction amount distributions shift (holiday season, economic conditions)
    - New merchant categories appearing
    - Device fingerprint distribution changes (new browser versions, OS updates)
    - Alert if `PSI > 0.2` for any critical feature

    **Concept drift** — fraud patterns actively evolve:
    - Fraudsters learn what triggers blocks and change tactics
    - Seasonal patterns: holiday fraud spikes, tax season identity theft, back-to-school scams
    - Monitor: prediction score distribution over time — are scores creeping higher or lower?
    - This is why retraining on a rolling window matters: not just for fresh labels, but
      because the fraud distribution itself is non-stationary

    ---

    ### Feedback Loop Awareness — The Interview Gold Insight

    The model's own decisions corrupt future training data.
    Declined transactions never get labels — you never know if they were actually fraud.
    Over time, the model trains only on the approved-transaction distribution,
    which drifts further and further from the true transaction distribution.

    **Solution: random exploration (intentional label collection)**
    - Approve a random sample (~0.1%) of transactions that the model would have declined
    - Observe their outcomes (chargeback or not)
    - Use these as unbiased training labels for the declined-transaction region of feature space
    - Cost: you are intentionally approving some fraud. This is real money.
    - But without it, your model will silently degrade over months.

    > Mentioning this unprompted is a strong signal of production ML experience.

    ---

    ### Retraining Triggers

    - **Scheduled**: daily retrain on a rolling 90-day labeled window
    - **Event-driven**: retrain immediately if `precision@100` drops >10% or `PSI > 0.25`
    - **Champion/challenger**: new model must beat current champion on a held-out eval set
      before promotion. Promotion goes through shadow mode first.

    ---

    ### Monitoring Architecture

    ```
    [Prediction Logs]
        → [Feature + Prediction Store]
        → [Daily Eval Job]  (join predictions with chargeback labels, 30-90d delay)
            → [Dashboard]         (precision, recall, false positive rate, queue utilization)
            → [Drift Detection]   (PSI per feature, prediction score distribution)
            → [Alerts]
                → [On-call page]    (immediate issues)
                → [Retrain Trigger] (scheduled or event-driven)
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Full System Architecture

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                      REAL-TIME PATH                          │
    │                                                              │
    │  [Transaction]                                               │
    │      → [API Gateway]                                         │
    │      → [Feature Lookup]   (Redis: user + merchant features)  │
    │      → [Rules Engine]     (hard rules, blacklists)           │
    │      → [ML Model]         (XGBoost, pre-loaded in-process)   │
    │      → [Decision Logic]   (threshold → approve/decline/2FA)  │
    │      → [Response]         (<100ms total)                     │
    └──────────────────────────────┬──────────────────────────────┘
                                   │ async prediction logs
    ┌──────────────────────────────▼──────────────────────────────┐
    │                    OFFLINE PIPELINE                          │
    │                                                              │
    │  [Data Lake]                                                 │
    │      → [Batch Feature Engine (Spark)]                        │
    │          → [Offline Feature Store (Delta Lake)]              │
    │      → [Streaming Feature Engine (Flink)]                    │
    │          → [Online Feature Store (Redis)]                    │
    │      → [Label Pipeline]   (join transactions + chargebacks)  │
    │      → [Training Job]     (daily, rolling 90-day window)     │
    │          → [Model Registry]                                  │
    │          → [Shadow Deploy] → [Champion/Challenger Eval]      │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
    ┌──────────────────────────────▼──────────────────────────────┐
    │                      MONITORING                              │
    │                                                              │
    │  [Drift Detection]     (PSI on features, score distribution) │
    │  [Performance Tracking] (precision@K, AUC-PR, FPR)          │
    │  → [Alerts]            (on-call + retrain trigger)           │
    │  → [Retrain Trigger]   → [Shadow Deploy] → [Promotion]      │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Common Interviewer Follow-Ups

    **"How do you handle cold-start for new users?"**
    New users have no behavioral history, so user feature vectors are empty or defaulted.
    Rely more heavily on transaction and device features.
    Use population-level baselines (e.g., average fraud rate for this merchant category and payment method).
    Apply a higher step-up authentication rate for accounts < 30 days old as a blanket policy.

    ---

    **"How do you handle cold-start for new merchants?"**
    Use merchant category averages as the initial prior.
    Flag transactions at new merchants as slightly higher risk until enough volume accumulates.
    Gradually build merchant-specific features as transaction history grows.

    ---

    **"What if fraudsters learn to evade your model?"**
    This is exactly why the rules engine persists as a layer — new fraud patterns can be captured
    with a hard rule (minutes to deploy) while the model catches up (hours to retrain and promote).
    It's also why we monitor prediction distribution shifts: if fraudsters successfully evade,
    scores on fraudulent transactions will start dropping, which we'd detect before losses spike.

    ---

    **"How do you explain a decline to a customer?"**
    XGBoost produces feature importances per prediction via SHAP values.
    Map these to human-readable reason codes: "unusual location," "rapid successive purchases,"
    "new device not seen before." This is also a regulatory requirement in some jurisdictions
    (GDPR right to explanation) — another reason to prefer XGBoost over black-box deep learning.

    ---

    **"What about privacy and regulatory compliance?"**
    PCI-DSS compliance for any system handling card data (encryption, access controls, audit logs).
    GDPR right to explanation — SHAP-based reason codes satisfy this.
    Data retention policies: raw transaction data has a required deletion schedule.
    Anonymization / pseudonymization of training data where possible.

    ---

    **"How would you handle a sudden spike in fraud?"**
    1. **Immediate** (minutes): deploy a new hard rule in the rules engine targeting the spike pattern
    2. **Short-term** (hours): trigger an emergency retrain with recent labeled data
    3. **Circuit breaker**: temporarily lower the auto-approve threshold globally to increase
       the fraction of transactions going to review or step-up auth
    4. Investigate root cause in parallel: compromised merchant, stolen card batch, new fraud vector

    ---

    **"Why XGBoost over deep learning for V1?"**
    - Tabular data — GBTs dominate on tabular features
    - Hard latency constraint: <5ms inference; XGBoost is faster than any neural net at comparable quality
    - Interpretability requirement: SHAP values, reason codes, regulatory explanations
    - Smaller labeled dataset (fraud is rare at 0.1%): deep learning needs more data to generalize
    - Deep learning is the right call for sequence modeling (V3) but not for the initial tabular model
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interview Talking Points

    ### 60-Second System Overview

    > "I'd design a two-layer system. The real-time path scores every transaction in under 100ms:
    > it fetches pre-computed user and merchant features from a Redis feature store,
    > runs them through a rules engine for obvious fraud, then through an XGBoost classifier
    > to get a fraud probability, and maps that to an action — approve, decline, review, or 2FA.
    > The offline pipeline handles training, labeling from delayed chargebacks,
    > and monitoring for drift. The key challenges are class imbalance, feedback loops
    > from our own decline decisions corrupting training data, and the fact that fraud patterns
    > evolve continuously."

    ---

    ### The One Thing That Impresses Interviewers

    > "I'd implement **random exploration** — intentionally approving a small random sample (~0.1%)
    > of transactions the model would otherwise decline. This is expensive short-term because
    > we're knowingly approving some fraud. But without it, we never get ground truth labels
    > for the declined-transaction region of feature space, and the model silently degrades
    > as the training distribution drifts away from the serving distribution."

    ---

    ### Connection to My Own Work

    The event-driven architecture in my backtesting engine mirrors this design:
    - The **real-time inference path** maps directly to my event loop — each transaction is like
      a market event that triggers a scoring function
    - The **offline training pipeline** maps to my backtesting analysis loop — both consume
      historical data to learn patterns and evaluate strategies
    - The **feature store** pattern is exactly how I pre-compute rolling technical indicators
      (20-day moving average, RSI, Bollinger Bands) so the live strategy loop never
      recomputes them inline

    The difference: in fraud detection, features decay in hours (velocity windows).
    In finance, some features decay in seconds (order book state).
    The architectural pattern is identical — the freshness SLA changes.
    """)
    return


if __name__ == "__main__":
    app.run()
