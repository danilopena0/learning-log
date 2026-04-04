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
    # ML System Design Template + Recommendation System Design

    | Field  | Value |
    |--------|-------|
    | Date   | 2026-04-01 |
    | Track  | System Design |
    | Time   | 60 min |
    | Topics | ML System Design Framework · Two-Stage RecSys · Feature Stores · Serving Architecture |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## The 5-Stage ML System Design Template

    This is the reusable framework to apply in every ML system design interview.
    Walk through each stage in order — interviewers are listening for structured thinking,
    not just the right answer.

    ---

    ### Stage 1: Problem Framing *(~5 min in interview)*

    **What to cover:**
    - Clarify the business objective — what metric are we optimizing?
    - Define the ML task type: classification, ranking, regression, or generation
    - Identify success metrics: **online** (A/B test CTR, revenue, engagement, session length)
      vs **offline** (precision, recall, NDCG, AUC)
    - State constraints explicitly: latency budget, serving cost, fairness requirements
    - Scope the problem: single-model or multi-stage? real-time or batch?

    **What interviewers are listening for:**
    > "Does this candidate understand that we're building a *product*, not a model?
    > Can they connect ML metrics to business outcomes?"

    **Key question to always ask:** *"Who is the user and what action are we trying to drive?"*

    ---

    ### Stage 2: Data *(~5–8 min)*

    **What to cover:**
    - What data is available? User behavior logs, item metadata, explicit feedback (ratings),
      implicit signals (clicks, dwell time, skips)
    - Data collection pipeline: batch ingestion vs streaming (Kafka/Kinesis)
    - Labeling strategy: how do you define positive/negative examples?
    - Data quality issues: missing data, class imbalance, selection bias, feedback loops
    - **Train/val/test split strategy** — use time-based splits for anything with temporal
      patterns (leakage risk is real if you split randomly)
    - Scale: how much data, how fast does it grow, what are the storage/compute implications?

    **What interviewers are listening for:**
    > "Does this candidate think carefully about *how* labels are defined and *what biases*
    > live in the data before they jump to modeling?"

    ---

    ### Stage 3: Feature Engineering *(~5–8 min)*

    **What to cover:**
    - **User features:** history, preferences, demographics, behavioral patterns
    - **Item features:** metadata, content embeddings, popularity, freshness
    - **Context features:** time of day, device, location, current session
    - **Cross features:** user×item interactions, affinity scores
    - Feature stores: **offline** (batch-computed, lower latency SLA) vs
      **online** (real-time lookups, must be fast)
    - Feature freshness: which features are static vs need real-time updates?
    - Embeddings: when to use learned embeddings (rich unstructured data) vs
      hand-crafted (interpretable, low-data regimes)

    **What interviewers are listening for:**
    > "Can this candidate distinguish between features that require real-time computation
    > and those that can be precomputed? Do they mention feature leakage?"

    **Common pitfall to mention:** *Feature leakage from future data* — e.g., using
    post-event signals as training features.

    ---

    ### Stage 4: Model *(~5–8 min)*

    **What to cover:**
    - Start with a **simple baseline** (popularity, logistic regression), then add complexity
      with explicit justification for each step
    - Training pipeline: offline batch training, retraining frequency, triggers for retraining
    - Model selection rationale — don't say "deep learning", say *why* (non-linear interactions,
      rich embeddings, scale of data justify the complexity cost)
    - Experiment tracking (MLflow, W&B), hyperparameter tuning strategy
    - Offline evaluation: held-out test set, cross-validation, metrics aligned with Stage 1

    **What interviewers are listening for:**
    > "Does this candidate understand the cost/benefit of model complexity?
    > Can they justify each architectural decision with a concrete reason?"

    ---

    ### Stage 5: Serving + Monitoring *(~5–8 min)*

    **What to cover:**
    - Serving architecture: **batch precomputation** (low latency, stale) vs
      **real-time inference** (fresh, higher latency/cost)
    - Latency requirements drive architecture choices — always state the budget
    - A/B testing framework for online evaluation: traffic split, holdout period, metrics
    - **Monitoring:**
        - Data drift: input feature distribution has shifted
        - Model drift (concept drift): the relationship between features and labels has changed
        - Prediction distribution shift: outputs look different than expected
    - Alerting: what triggers retraining? what triggers rollback?
    - **Feedback loops:** recommendations influence what users see → what they click →
      what becomes training data. This is a critical system property to call out.

    **What interviewers are listening for:**
    > "Does this candidate think about the system *after* deployment?
    > Do they understand that an ML system is never 'done' once trained?"
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Template Cheat Sheet

    *Review this in 2 minutes before any ML system design interview.*

    | Stage | Time | Key Deliverable | Interviewer Wants to Hear |
    |-------|------|-----------------|---------------------------|
    | **1. Problem Framing** | 5 min | Business objective + ML task type + online/offline metrics + constraints | "I connect the ML problem to a business outcome before touching data or models." |
    | **2. Data** | 5–8 min | Data sources + labeling strategy + quality issues + time-based split | "I know where the data comes from, how labels are defined, and what biases might be lurking." |
    | **3. Feature Engineering** | 5–8 min | User/item/context/cross features + feature store design + freshness requirements | "I can distinguish real-time from batch features and I know what feature leakage looks like." |
    | **4. Model** | 5–8 min | Baseline → V1 → V2 progression with justified complexity + offline eval | "I start simple and add complexity only when I can justify it with a concrete metric hypothesis." |
    | **5. Serving + Monitoring** | 5–8 min | Serving architecture + latency budget + A/B test + drift monitoring + feedback loops | "I think about the system after deployment — drift, retraining triggers, and feedback loops." |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Practice: Design a Recommendation System

    > **Scenario:** "Design the recommendation system for a large content platform
    > (think YouTube / Netflix / Spotify home feed)."

    Apply the 5-stage template end-to-end.
    """)
    return


@app.cell
def recsys_problem_framing(mo):
    mo.md("""
    ### Stage 1: Problem Framing

    **Business objective:** Increase user engagement — specifically watch time and
    return visits.  We are not just maximizing clicks (that invites clickbait).

    **ML task:** Ranking — given a user in context, score and rank a set of candidate
    items to determine what to surface on the home feed.

    **Online metrics (production A/B tests):**
    - Click-through rate (CTR) on recommended items
    - Watch time / listen time per session
    - Session length and return rate (next-day retention)

    **Offline metrics (model evaluation):**
    - NDCG@K — normalized discounted cumulative gain, rewards putting the best items first
    - Recall@K — fraction of items the user would have engaged with that appear in top K
    - AUC — discrimination power of the ranking model

    **Constraints:**
    - **Latency:** <200ms end-to-end for serving (user is waiting on a page load)
    - **Scale:** 100M+ users, millions of items in the catalog
    - **Fairness/diversity:** avoid filter bubbles; surface a variety of content types and
      creators, not just the highest-confidence predictions
    - **Cold start:** must handle new users (no history) and new items (no engagement data)
    """)
    return


@app.cell
def recsys_data(mo):
    mo.md("""
    ### Stage 2: Data

    **Implicit signals** (no user effort, high volume, noisy):
    - Views, clicks, watch duration, skip rate, replays, shares, saves
    - *Note:* a 2-second click vs a 20-minute watch are very different signals — duration
      matters more than binary click

    **Explicit signals** (less noisy, lower volume):
    - Ratings, likes, thumbs up/down, explicit "not interested"

    **Item metadata:**
    - Title, description, category, tags, creator identity, upload date, duration,
      content embeddings (from title/thumbnail/audio)

    **User metadata:**
    - Demographic signals (age bucket, region), subscription tier, device type,
      long-term watch history, genre preference profile

    **Negative sampling strategy:**
    Implicit feedback has no explicit negatives — a user who didn't click an item may
    not have seen it, or may have seen it and ignored it.  Common approaches:
    - **Exposure-based negatives:** items shown to the user but not clicked
    - **Random negatives:** sample from the catalog (fast but noisy)
    - **Hard negatives:** items almost clicked — harder for the model to discriminate
      (improves embedding quality)

    **Train/val/test split:**
    - Time-based split — train on months 1–6, validate on month 7, test on month 8
    - *Why not random?* Temporal leakage — future behavior (month 8) cannot cause
      past behavior (month 1), so a random split would give optimistic offline metrics
      that don't reflect real deployment performance
    """)
    return


@app.cell
def recsys_features(mo):
    mo.md("""
    ### Stage 3: Feature Engineering

    **User features:**
    - Watch history embedding (learned, updated daily via batch job)
    - Average session length, time-of-day usage patterns
    - Genre/topic affinity scores (e.g., user watches 60% music, 30% news)
    - Long-term vs short-term interest: separate embeddings for "historical taste"
      vs "what the user has been watching this week"

    **Item features:**
    - Content embedding from title + description (pre-trained language model or
      fine-tuned on platform data)
    - Popularity score (views in last 7 days, normalized by catalog position)
    - Freshness score (recency decay function of upload date)
    - Creator features: creator embedding, subscriber count, historical engagement rate

    **Context features:**
    - Time of day, day of week (commute vs weekend browsing behavior differs)
    - Device type (mobile vs TV vs desktop → different UX and content length preferences)
    - Current session context: what the user has already watched in this session

    **Cross features:**
    - User–item interaction count (has user engaged with this creator before?)
    - User–genre affinity score (how aligned is this item's genre with user's preference?)
    - User–freshness affinity (does this user tend to watch new vs classic content?)

    **Feature store design:**

    | Feature Group | Update Frequency | Store Type |
    |---------------|-----------------|------------|
    | User history embeddings | Daily batch job | Offline (Redis / Feast) |
    | Item content embeddings | On item upload + daily refresh | Offline |
    | Item popularity scores | Hourly | Online (low-latency key-value) |
    | Session context | Real-time (per request) | Computed inline at serving time |
    | User long-term affinity | Weekly batch | Offline |

    **Feature leakage risk to call out:** Never include post-engagement signals
    (e.g., total lifetime views of an item) in training features — these are influenced
    by the model's own past recommendations and create a feedback loop that corrupts
    offline evaluation.
    """)
    return


@app.cell
def recsys_model(mo):
    mo.md("""
    ### Stage 4: Model

    **Two-stage architecture** (standard for large-scale RecSys):

    The core tension: we have millions of items but only 200ms. We can't score all items
    with an expensive model. Solution: decompose into retrieval (fast, approximate) then
    ranking (accurate, expensive on a small candidate set).

    ---

    **Stage 4a: Candidate Generation (Retrieval)**

    Goal: reduce millions of items to ~500 candidates quickly.

    - **Baseline:** popularity-based retrieval — top N items by global engagement score.
      No personalization, but a strong baseline (most users click popular items).
    - **V1: Collaborative filtering** — matrix factorization (ALS/SVD).  Learn user and item
      latent vectors from interaction matrix.  Approximate nearest neighbor (ANN) at serving
      time.  *Why:* proven approach, handles implicit feedback well, interpretable embeddings.
    - **V2: Two-tower model** — separate neural encoder for users and items, trained with
      contrastive loss (positive pairs: user+item they engaged with; negative pairs: random
      or hard negatives).  Dot product similarity for retrieval.  *Why:* handles rich
      heterogeneous features (text, metadata, context) that matrix factorization ignores.
    - Serve retrieval via ANN index (FAISS, ScaNN) over precomputed item embeddings.

    ---

    **Stage 4b: Ranking**

    Goal: score top-500 candidates precisely, rank → top-20 shown to user.

    - **V1:** Logistic regression on handcrafted features (fast, interpretable, strong baseline)
    - **V2:** Gradient boosted trees (XGBoost/LightGBM) — handles feature interactions well,
      robust to missing values, fast to train
    - **V3: Multi-task learning** — single model predicts both click probability AND watch time
      jointly.  *Why:* optimizing for clicks alone incentivizes clickbait; joint training forces
      the model to balance engagement quality with engagement quantity.  Each task shares a
      backbone with separate prediction heads.

    **Training pipeline:**
    - Daily retraining on last 30 days of data (recency-weighted — recent behavior matters more)
    - Triggered by: scheduled daily job OR drift alert (NDCG@10 drops >5% on daily eval set)
    - Experiment tracking in MLflow: log features used, hyperparameters, offline metrics
    - Shadow mode deployment: new model runs alongside production, logs predictions but
      doesn't serve them — compare distributions before A/B test

    **Offline evaluation:**
    - Held-out test set (month 8 in the time-based split)
    - Primary metric: NDCG@10 (matches the ranking objective)
    - Supporting: Recall@50 (did we retrieve items the user would have engaged with?),
      AUC (discrimination), coverage (diversity of items recommended across all users)
    """)
    return


@app.cell
def recsys_serving_monitoring(mo):
    mo.md("""
    ### Stage 5: Serving + Monitoring

    **Serving pipeline (per request, <200ms budget):**

    1. **Feature lookup** (~10ms): retrieve precomputed user embedding + session context
       from online feature store
    2. **Candidate generation** (~50ms): ANN query against item embedding index → top 500
    3. **Ranking model inference** (~100ms): score top 500 with multi-task ranking model
       → top 20 ranked items
    4. **Business rules** (~40ms): apply diversity constraints (no more than 3 items from
       same creator), freshness boost, content policy filtering → final top 10 shown

    Total budget: 10 + 50 + 100 + 40 = **200ms** ✓

    **Latency vs freshness tradeoff:**
    - User embeddings updated daily (batch) — acceptable staleness; user taste changes slowly
    - Item popularity scores updated hourly — viral content needs to surface quickly
    - Session context computed inline — critical for "don't show what they just watched"

    **A/B testing framework:**
    - New model ships as challenger, production stays as control
    - Start with 5% traffic on challenger (low blast radius)
    - Measurement window: 2 weeks minimum (captures weekly seasonality)
    - Success criteria: statistically significant lift in watch time per session (primary),
      no regression in session length or return rate (guardrail metrics)
    - If guardrail metrics regress, auto-rollback to control

    **Monitoring:**

    | What to Monitor | Signal | Alert Threshold |
    |-----------------|--------|-----------------|
    | Data drift | KL divergence of input feature distributions vs training baseline | > 0.1 on any key feature |
    | Prediction drift | Mean predicted click probability over last 24h | ±15% from 7-day rolling average |
    | NDCG@10 on daily eval set | Offline metric on held-out recent data | Drop > 5% triggers retraining |
    | Cold-start performance | Engagement rate for users with <5 interactions | Tracked separately — can degrade silently |
    | Popularity bias | % of impressions going to top-1000 items | > 80% triggers diversity review |

    **Feedback loop awareness:**
    The system's own recommendations determine what users watch, which becomes new
    training data.  This creates a self-reinforcing loop that can amplify early biases:

    - Popular items get recommended → more views → even more popular → dominate the catalog
    - Users in a "taste cluster" get shown similar content → taste narrows → diversity collapses

    **Mitigations:**
    - **Exploration traffic:** reserve ε% (e.g., 5%) of recommendations for exploration —
      Thompson sampling or ε-greedy to surface non-obvious items
    - **Diversity constraints** at business rules layer
    - **Serendipity metrics** tracked alongside engagement (do users discover new content?)
    - **Counterfactual logging:** log what *would have been* recommended to enable unbiased
      offline evaluation of future models
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Full System Architecture

    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        ONLINE SERVING PATH                          │
    │                                                                     │
    │  [User Request]                                                     │
    │      │                                                              │
    │      ▼                                                              │
    │  [Feature Store (online)]  ←── user embedding, session context     │
    │      │                                                              │
    │      ▼                                                              │
    │  [Candidate Generation]    ←── ANN index over item embeddings      │
    │      │  top ~500 candidates                                         │
    │      ▼                                                              │
    │  [Ranking Model]           ←── multi-task: click + watch time      │
    │      │  top 20 scored items                                         │
    │      ▼                                                              │
    │  [Business Rules Layer]    ←── diversity, freshness, policy filter │
    │      │  top 10 shown to user                                        │
    │      ▼                                                              │
    │  [Response]                                                         │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │                        OFFLINE TRAINING PATH                        │
    │                                                                     │
    │  [User Events (Kafka)]                                              │
    │      │                                                              │
    │      ▼                                                              │
    │  [Data Lake]  ──►  [Feature Engineering]  ──►  [Feature Store]     │
    │                                                        │            │
    │                    [Model Training]  ◄─────────────────┘           │
    │                         │                                          │
    │                         ▼                                          │
    │                    [Model Registry]  ──►  [A/B Test Config]        │
    │                         │                                          │
    │                         ▼                                          │
    │                    [Shadow Mode]  ──►  [Challenger Promotion]      │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │                           MONITORING                                │
    │                                                                     │
    │  [Prediction Logs]                                                  │
    │      │                                                              │
    │      ▼                                                              │
    │  [Drift Detection]  ──►  [Alerts]  ──►  [Retrain Trigger]         │
    │                                     ──►  [Rollback Trigger]        │
    │                                                                     │
    │  [Daily Eval Set]   ──►  [NDCG@10 Tracking]  ──►  [Dashboard]     │
    └─────────────────────────────────────────────────────────────────────┘
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Common Interviewer Follow-Ups

    ---

    **"How do you handle cold start for new users?"**

    > No interaction history means no personalized embedding.  Fallback strategy is layered:
    > 1. Use content-based features available at signup: stated interests, demographics if available
    > 2. Show globally popular content filtered by stated preferences
    > 3. Onboarding quiz / explicit preference collection (ask what genres they like)
    > 4. As soon as they have ≥1 interaction, start building a sparse user embedding
    > 5. "Explore-first" policy: weight exploration higher for new users to gather signal faster

    ---

    **"How do you handle cold start for new items?"**

    > New items have no engagement history, so popularity and collaborative signals don't exist.
    > 1. Use content features immediately: title/description embedding, category, creator identity
    > 2. Creator-based similarity: if this creator's past items performed well with segment X,
    >    bootstrap the new item's embedding from the creator embedding
    > 3. Exploration boost: artificially increase the item's probability of being sampled in
    >    candidate generation for the first 48 hours to gather engagement signal quickly
    > 4. Once item accumulates ≥100 interactions, transition to standard collaborative signals

    ---

    **"How do you ensure diversity in recommendations?"**

    > Several approaches at different system layers:
    > - **MMR (Maximal Marginal Relevance):** re-rank candidates to maximize relevance minus
    >   similarity to already-selected items — balances relevance with novelty
    > - **Category-level quotas:** at business rules layer, enforce that no more than N items
    >   from the same topic cluster appear in one feed
    > - **Determinantal point processes (DPP):** principled probabilistic method for selecting
    >   a diverse subset; computationally heavier but produces high-quality diverse slates
    > - **Serendipity metric:** track what fraction of engaged items were "surprising" (outside
    >   user's historical cluster) — monitor over time

    ---

    **"What about position bias in training data?"**

    > Items shown at the top of the feed get more clicks simply because of their position,
    > not because they're better.  If you train on raw click data, the model learns to predict
    > "was this in position 1?" not "is this a good item?"
    >
    > **Inverse propensity weighting (IPW):** weight each training example by 1/(probability
    > it was shown at that position).  Items shown in unfavorable positions get up-weighted.
    >
    > **Position feature:** include the item's serving position as a feature during training,
    > but set it to a constant neutral value at inference — model learns to debias itself.

    ---

    **"How would you scale this to 1B users?"**

    > - **Shard user embeddings:** distribute across a cluster of key-value stores by user ID
    > - **Precompute candidate sets:** for popular user segments, pre-generate candidate lists
    >   in batch to offload work from the real-time path
    > - **Edge caching:** cache recommendation responses for users with identical context
    >   (same region + time-of-day bucket + cold-start cluster)
    > - **Approximate retrieval:** use FAISS/ScaNN with product quantization — trades a small
    >   amount of recall for a 10–100× speedup in ANN search
    > - **Two-tier serving:** separate retrieval service (stateless, horizontally scalable)
    >   from ranking service (stateful model server, GPU-backed)

    ---

    **"How do you prevent filter bubbles?"**

    > - **Exploration traffic:** 5–10% of recommendations are drawn from outside the user's
    >   inferred taste cluster (Thompson sampling or ε-greedy)
    > - **Serendipity constraints:** hard rule — at least 2 of the top 10 recommendations
    >   must be from a category the user has not engaged with in the last 30 days
    > - **Diversity reward in multi-task training:** add a diversity objective to the multi-task
    >   loss that penalizes over-concentration in a single topic cluster
    > - **Long-term engagement monitoring:** track whether users' interest diversity narrows
    >   over time; if so, increase exploration rate for that cohort
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    ### 60-Second Elevator Pitch

    > "I'd approach this in five stages: first, I nail down the business objective and
    > define what 'good' looks like — both online metrics I'd measure in an A/B test and
    > offline metrics I can evaluate during development.  Second, I map out the data
    > available and how I'd structure the labeling.  Third, I define the feature set and
    > decide what needs real-time freshness vs what can be precomputed.  Fourth, I design
    > a model progression from a simple baseline up to the target architecture, justifying
    > each complexity step.  And fifth, I think through the full serving architecture,
    > latency budget, and monitoring setup — because the system doesn't stop being interesting
    > once the model is trained."

    ---

    ### The One Thing Interviewers Love

    > *"I'd start with the simplest baseline that actually works, then justify each complexity
    > addition with a clear metric improvement hypothesis."*

    This signals engineering judgment, not just ML knowledge.  It shows you understand
    that a more complex model is a *cost* (training time, debugging, serving infrastructure,
    explainability) that requires justification in terms of metric improvement.

    ---

    ### Connecting to My Experience

    **Event-driven serving pipeline → backtesting engine:**
    The recommendation serving pipeline (user event → feature lookup → model → response)
    mirrors the pub/sub, event-driven architecture in my backtesting engine.  In both cases,
    events trigger a downstream pipeline with latency and throughput constraints.  I can
    speak to the tradeoffs: fan-out patterns, message ordering, at-least-once vs exactly-once
    delivery semantics.

    **Two-stage retrieve-then-rank → Canopy RAG pipeline:**
    The candidate generation (ANN retrieval) → ranking model architecture directly parallels
    the retrieval-then-reranking pattern in Canopy's RAG pipeline.  In both cases, stage 1
    maximizes recall cheaply (ANN / BM25 + embedding retrieval) and stage 2 maximizes
    precision expensively (cross-encoder reranker / ranking model).  The key tradeoff is
    the same: recall@K in stage 1 sets the ceiling for final precision.

    ---

    ### Red Flags to Avoid

    - Jumping to a complex model before defining the problem ← most common mistake
    - Forgetting to mention the feedback loop (self-referential training data)
    - Saying "we'd use BERT" or "we'd use a transformer" without explaining *why* the
      problem justifies that complexity
    - Not separating offline metrics (what you measure during training) from online metrics
      (what you measure after deployment) — they are different things
    - Ignoring cold start — interviewers always ask about it
    """)
    return


if __name__ == "__main__":
    app.run()
