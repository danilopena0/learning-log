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
    # MLOps Pipeline — CI/CD for Models, A/B Testing, Canary Deploys, Rollback

    | Field  | Value |
    |--------|-------|
    | Date   | 2026-04-23 |
    | Track  | System Design |
    | Time   | 60 min |
    | Topics | MLOps Maturity · Model Versioning · CI/CD for ML · A/B Testing · Canary Deploys · Rollback |
    """)
    return


@app.cell
def why_mlops_exists(mo):
    mo.md("""
    ## Why MLOps Exists

    **The gap:** Data scientists build models in notebooks. Production needs reproducibility,
    versioning, testing, deployment, monitoring, and rollback. MLOps is the discipline that
    bridges that gap.

    **MLOps = DevOps principles applied to ML systems. But harder because:**

    - Code AND data AND model artifacts all need versioning — not just code
    - "Deploying" means deploying code + model weights + feature pipelines + configs together
    - Testing is harder — correctness is probabilistic, not deterministic. A bad model doesn't
      throw an exception, it just silently produces worse predictions
    - Models degrade without explicit errors — data drift and concept drift are invisible
      to traditional monitoring

    ---

    ### The MLOps Maturity Ladder

    | Level | Description |
    |-------|-------------|
    | **Level 0** | Manual everything. Train in notebook, copy weights to server, pray. No versioning, no tests, no monitoring. Where most teams start. |
    | **Level 1** | Automated training pipeline, manual deployment. Retraining is scripted, but promoting a model to production requires human action. |
    | **Level 2** | Automated training + deployment, manual monitoring. Model goes from training to serving automatically if eval passes, but drift detection is still human-driven. |
    | **Level 3** | Fully automated: train → eval → deploy → monitor → retrain loop. Drift triggers retraining without human intervention. |

    > **"Most companies are at Level 1. Knowing Level 3 is what gets you hired to build it."**
    """)
    return


@app.cell
def stage1_versioning_concept(mo):
    mo.md("""
    ## Stage 1: Model Versioning & Experiment Tracking

    ### The ML Trinity — What to Version

    Software has one artifact to version: code. ML has three:

    1. **Code** — git. Same as software. This problem is solved.
    2. **Data** — DVC (Data Version Control), Delta Lake, or hash-based snapshots.
       Track WHICH dataset trained WHICH model. Without this, you can't reproduce a run.
    3. **Model artifacts** — MLflow Model Registry, Weights & Biases, or S3 with
       structured naming conventions. Track weights, hyperparameters, metrics, and lineage.

    **Why all three matter:** "Model v3.2 is broken" is useless without knowing which code,
    which data, and which hyperparameters produced it. Full lineage = reproducibility.

    When an incident happens at 2am, you need to answer:
    - What model is serving?
    - What data trained it?
    - What was its eval score before deployment?
    - What's the previous stable version I can roll back to?

    All of those answers come from versioning the trinity together.
    """)
    return


@app.cell
def stage1_experiment_tracking(mo):
    mo.md("""
    ### Experiment Tracking

    Every training run should log:
    - Hyperparameters (learning rate, batch size, architecture choices)
    - Metrics (train loss, val loss, AUC, F1 — whatever's relevant)
    - Training duration and compute cost
    - Data snapshot hash or version tag
    - Git commit SHA
    - Environment (Python version, package versions, hardware)

    **Tools:**
    | Tool | Type | Notes |
    |------|------|-------|
    | MLflow | Open source, self-hosted | Industry standard, integrates with most frameworks |
    | W&B | SaaS (free tier) | Best UI, great for comparing runs visually |
    | Neptune | SaaS | Strong metadata management |
    | Comet | SaaS | Good for NLP/CV teams |

    **The "experiment" abstraction:** a named group of runs exploring different configs.
    You compare runs side by side, pick the best, and promote it.

    ---

    **My project mapping — backtesting engine:**

    > "In my backtesting engine, every strategy backtest IS an experiment run — parameters
    > (lookback window, threshold, position sizing), metrics (Sharpe, drawdown, win rate),
    > data window, and results. The pattern is identical to MLflow experiment tracking.
    > I've already built the mental model; I just haven't put the MLflow wrapper around it."
    """)
    return


@app.cell
def stage1_model_registry(mo):
    mo.md("""
    ### Model Registry

    A model registry is the central catalog of trained models with lifecycle stages:

    ```
    Training → [Staging] → [Production] → [Archived]
    ```

    Each registry entry contains:
    - Model version number
    - Evaluation metrics (offline)
    - Reference to training data version
    - Git commit of training code
    - Approval status and approver
    - Deployment history

    **Promotion flow:**
    ```
    train → register as "staging"
          → eval passes thresholds
          → promote to "production"
          → previous production → "archived"
    ```

    > **"The model registry is for models what a container registry (Docker Hub) is for services."**

    Both are immutable artifact stores with versioned entries and promotion workflows.
    The concept transfers directly.
    """)
    return


@app.cell
def stage2_ml_cicd_vs_software(mo):
    mo.md("""
    ## Stage 2: CI/CD for ML

    ### How ML CI/CD Differs from Software CI/CD

    **Software CI pipeline:**
    ```
    code change → lint → unit tests → integration tests → deploy
    ```

    **ML CI pipeline** — all of the above, PLUS:
    - **Data validation**: schema, distributions, completeness. Does the new data
      match what the model was trained on?
    - **Training pipeline smoke test**: does it converge? Does it crash?
      (On a small sample — full training is too slow for CI)
    - **Model evaluation**: metrics on held-out test set, must meet thresholds
    - **Model comparison**: does the candidate beat the current production model?
    - **Bias/fairness checks**: do any demographic slices regress beyond threshold?

    **ML CD** — not rebuilding a Docker image with model baked in. The model is too
    large and changes more frequently than code. Instead:
    - Promote a model artifact (update which weights the serving layer loads)
    - Update serving config (a config change, not a code deploy)
    - Route traffic gradually (canary, then full rollout)

    **Key insight:** In ML, the model is configuration, not code.
    Deploying a new model = updating a config pointer, not rebuilding a service.
    """)
    return


@app.cell
def stage2_ci_pipeline_detail(mo):
    mo.md("""
    ### The CI Pipeline in Detail

    **Trigger:** PR to main that touches model code, training config, feature definitions,
    or data pipeline.

    ```
    [PR Opened]
      │
      ├── [Lint + Type Check]
      │     ruff, mypy — same as any software PR
      │
      ├── [Unit Tests]
      │     Pure functions: feature engineering logic, data transforms,
      │     preprocessing steps. Deterministic inputs → deterministic outputs.
      │
      ├── [Data Validation]
      │     Great Expectations or Pandera
      │     - Schema check: expected columns, types, value ranges
      │     - Distribution check: no sudden shifts vs baseline statistics
      │     - Completeness: no unexpected nulls in required fields
      │     Fails the PR if data looks wrong — catch upstream issues here.
      │
      ├── [Training Pipeline Smoke Test]
      │     - Train on 1% of production data (enough to verify the pipeline runs)
      │     - Verify: model converges, metrics in expected range, artifacts saved correctly
      │     - NOT a full training run — that's too slow. This just proves the code works.
      │
      ├── [Model Evaluation]
      │     - Load current production model + candidate model
      │     - Run both on held-out eval set
      │     - Candidate must: meet minimum absolute thresholds AND not regress vs production
      │     - Generate comparison report as PR comment (show the diff, not just pass/fail)
      │
      ├── [Bias/Fairness Checks] (if applicable)
      │     - Evaluate metrics across demographic slices
      │     - Flag if any slice degrades beyond threshold
      │
      └── [Approve / Block PR]
            Green on all steps → auto-approve. Any red → block with report.
    ```

    The comparison report as a PR comment is key — it makes the tradeoff visible to
    reviewers without requiring them to run anything.
    """)
    return


@app.cell
def stage2_cd_pipeline(mo):
    mo.md("""
    ### The CD Pipeline

    **Trigger:** merge to main + eval passed + (optional) manual approval gate.

    ```
    [Merge to Main]
      │
      ├── [Full Training Run]
      │     Production data, full epochs, production hyperparameters
      │
      ├── [Final Evaluation]
      │     Held-out test set. Must beat production baseline by ≥ threshold.
      │     Hard stop if it doesn't — do not proceed.
      │
      ├── [Register Model]
      │     MLflow/W&B registry, stage="staging"
      │     Links artifact to git commit + data version + metrics
      │
      ├── [Shadow Deployment]
      │     Run new model alongside production.
      │     Log its predictions. Do NOT serve to users.
      │     Validates the serving infrastructure without user exposure.
      │
      ├── [Canary Deployment — 5% traffic]
      │     Route small slice of real traffic to new model.
      │     Monitor for 1–6 hours: error rate, latency, output distribution.
      │
      ├── [A/B Test — 50/50 split]
      │     Statistically rigorous measurement of business impact.
      │     Run for 1–2 weeks minimum.
      │
      ├── [Full Rollout — 100%]
      │     Promote model to production in registry.
      │     Update serving config to route all traffic.
      │
      └── [Archive Old Model]
            Keep old weights loaded for 24h (instant rollback window).
            Then archive in registry.
    ```
    """)
    return


@app.cell
def stage2_project_mapping(mo):
    mo.md("""
    ### My Project Mapping — CI/CD

    **Daily Briefing Agent:**
    > Already runs regression evals in GitHub Actions on prompt changes. This IS ML CI —
    > checking that prompt modifications don't regress summarization quality before merge.
    > It's the LLM equivalent of the model evaluation gate in a traditional ML CI pipeline.

    **Canopy:**
    > Scoring prompts are versioned in the repo. Natural extension: add regression evals
    > on scoring accuracy as a CI check before merging prompt changes.
    > The prompt is the model. Treat it like one.

    ---

    **Interview framing:**

    > "I've implemented a lightweight version of ML CI in my briefing agent — GitHub Actions
    > runs regression evals on prompt changes before merge. For a traditional ML system,
    > I'd extend this with data validation via Pandera, model training smoke tests, and
    > a model comparison gate that blocks the PR if the candidate regresses vs production."
    """)
    return


@app.cell
def stage3_ab_testing_concept(mo):
    mo.md("""
    ## Stage 3: A/B Testing for Models

    ### Why A/B Testing Exists

    A/B testing is the gold standard for evaluating whether a new model ACTUALLY improves
    business metrics in production.

    **Why offline eval isn't enough:**
    - Offline metrics (AUC, F1, NDCG) don't always correlate with business metrics
      (revenue, engagement, conversion, retention)
    - Users behave differently in production than test data distributions suggest
    - Interaction effects with other systems: recommendations affect what users see,
      which affects their behavior, which changes future training data (feedback loops)
    - A model can improve AUC by 2% and decrease revenue. It happens.

    **A/B test setup:**
    - **Control group**: current production model (Model A)
    - **Treatment group**: candidate model (Model B)
    - **Random assignment** of users/requests to groups — must be sticky
      (same user always gets same model, or results are noisy)
    - **Measure business metric**: CTR, revenue, churn reduction, session length, etc.
    - **Run duration**: statistically significant window — usually 1–2 weeks minimum
      to capture weekly behavioral patterns
    """)
    return


@app.cell
def stage3_statistical_rigor(mo):
    mo.md("""
    ### Statistical Rigor

    **Before you run the test, decide:**

    - **Sample size**: calculate via power analysis. Need enough observations to detect
      the minimum effect size you care about at the desired power (typically 80%).
    - **Significance threshold**: p < 0.05 (5% false positive rate). Pre-commit to this —
      don't adjust after seeing results.
    - **Minimum Detectable Effect (MDE)**: the smallest improvement worth deploying for.
      If MDE = 1% CTR lift, you need ~10K observations per group. Smaller MDE = larger sample needed.
    - **Duration**: at least 1 full week (ideally 2) to capture weekly patterns.

    **Common pitfalls:**

    | Pitfall | What goes wrong | Fix |
    |---------|----------------|-----|
    | Peeking early | Stop when results look good → inflated false positive rate | Pre-commit to duration, don't look until it's over |
    | Network effects | Users influence each other (social platforms) → treatment leaks into control | Cluster randomization (assign by friend group, not individual) |
    | Novelty effects | New model gets more clicks initially just because it's different | Wait for stabilization (run 2+ weeks) |
    | Multiple testing | Running 5 variants at once → 1 will look significant by chance | Bonferroni correction or sequential testing |

    > **"This connects to my ML theory notebook on A/B testing and causal inference —
    > same statistical foundations. Power analysis and MDE sizing are the most commonly
    > skipped steps and the most commonly asked follow-up questions."**
    """)
    return


@app.cell
def stage3_shadow_canary_ab_comparison(mo):
    mo.md("""
    ### Shadow vs Canary vs A/B vs Full Rollout

    | Method | Users see new model? | Measures business impact? | Risk | Duration |
    |--------|---------------------|--------------------------|------|----------|
    | **Shadow deploy** | No (log-only) | No — no user exposure | Zero | Hours |
    | **Canary deploy** | Yes (1–5%) | Partially — limited statistical power | Low | Hours to days |
    | **A/B test** | Yes (50/50 or configurable) | Yes — full statistical test | Medium | 1–2 weeks |
    | **Full rollout** | Yes (100%) | Before/after comparison (confounded) | High | N/A |

    **Typical progression:** shadow → canary → A/B → full rollout.
    Each stage gates the next. Earlier stages catch failures cheaply; A/B provides the
    statistical evidence needed to justify the business decision to roll out fully.

    **Canary vs A/B — the key distinction:**
    - Canary: "Is this broken?" (safety gate, looking for failures, errors, regressions)
    - A/B: "Is this better?" (scientific experiment, measuring business improvement with rigor)

    They answer different questions and you typically run both in sequence.
    """)
    return


@app.cell
def stage4_canary_concept(mo):
    mo.md("""
    ## Stage 4: Canary Deployments

    ### What Canary Is

    Canary = deploy the new model to a small fraction of traffic, monitor closely,
    expand gradually. Named after the "canary in a coal mine" — if something is wrong,
    the small group catches it before everyone is affected.

    **Expansion schedule (typical):**

    ```
    1% traffic  →  1 hour   →  check: error rate, latency, prediction distribution
    5% traffic  →  6 hours  →  check: business metrics starting to form
    25% traffic →  24 hours →  statistical confidence building
    50% traffic →  24 hours →  near full validation, run A/B analysis
    100% traffic →  full rollout
    ```

    At ANY stage: if metrics degrade beyond threshold → **automatic rollback** to previous model.
    The rollback must be faster than the detection — usually a config update that takes seconds.

    **Why 1% first?**
    If the new model has a catastrophic bug (crashes on certain inputs, returns null, has
    infinite latency), it affects 1% of users for 1 hour. That's recoverable. 100% for
    1 hour is an incident.
    """)
    return


@app.cell
def stage4_implementation_patterns(mo):
    mo.md("""
    ### Implementation Patterns

    **Traffic splitting — three approaches:**

    1. **Load balancer routing**: route X% of requests to new model endpoint, (100-X)% to old.
       Simple. Works at the infrastructure level. No application code changes.

    2. **Feature flag**: model version as a feature flag. Toggle per user/group/percentage.
       More flexible — can target specific user segments. Requires a feature flag system.

    3. **Model serving layer**: dedicated infrastructure (Seldon, KServe, or custom router)
       routes requests based on config. Update config = switch model instantly.
       No code deploy needed.

    **The key architectural principle:**

    > Model weights are CONFIGURATION, not code. Deploying a new model = updating which
    > weights file the serving layer loads, not rebuilding and redeploying the service.

    This is why "baking the model into the Docker image" is an anti-pattern. It couples
    model releases to code releases, makes rollback slow, and inflates image sizes.

    Instead: serve loads weights from S3/GCS at startup (or dynamically). Switching models
    = updating the config that points to which S3 path to load. Takes seconds, not minutes.
    """)
    return


@app.cell
def stage4_canary_monitoring(mo):
    mo.md("""
    ### Canary Monitoring Dashboard

    What to watch during a canary, and what each signal means:

    | Signal | What it tells you | Rollback threshold |
    |--------|------------------|--------------------|
    | **Error rate** | Hard failures — model crashing, returning nulls, timeout | Any spike → rollback immediately |
    | **Latency (p99)** | New model slower? Larger model or inefficient preprocessing | > 2x production baseline |
    | **Prediction distribution** | Output distribution should be similar to old (unless intentionally changed). Big shift = something's wrong. | KL divergence > threshold |
    | **Business metrics** | CTR, conversion, revenue — the outcomes you actually care about | Statistically significant decline |
    | **Input data quality** | Are inputs to the new model matching expected distributions? | Schema violations, distribution shift |

    **Monitoring tooling:**
    - Error rate + latency: Prometheus + Grafana, Datadog, CloudWatch
    - Prediction distribution + drift: Evidently, WhyLabs, Arize
    - Business metrics: your analytics platform (Amplitude, Mixpanel, BigQuery dashboards)

    **Alert → action mapping:**
    - Error rate spike → automated rollback, page on-call
    - Latency spike → investigate before rolling back (may be fixable with scaling)
    - Prediction distribution shift → investigate, likely rollback
    - Business metric decline → human decision (needs context), but set a threshold for auto-rollback
    """)
    return


@app.cell
def stage5_rollback_concept(mo):
    mo.md("""
    ## Stage 5: Rollback Strategies

    ### Types of Rollback

    **1. Instant model rollback** (the only acceptable default):
    - Switch traffic back to old model. Takes seconds if model weights are separate from code.
    - Old model stays loaded and warm during the canary period — do NOT evict it.
    - Rollback = config change. Not a deployment. Not a new Docker image. Seconds, not minutes.

    **2. Data rollback** (harder):
    - Revert to previous feature pipeline version.
    - May require reprocessing historical data.
    - Usually reserved for bugs in the feature pipeline, not the model itself.

    **3. Full rollback** (nuclear option):
    - Revert code + model + features.
    - For catastrophic failures that span multiple systems.
    - Takes time, requires coordination, use sparingly.

    ---

    ### Why Instant Rollback Must Be Possible

    - Model serving must support multiple loaded models with traffic routing via config
    - Old model stays warm during canary (you have not committed to the new one yet)
    - Rollback = config change (`model_version: v3.1` → `model_version: v3.0`), not a deployment

    If rollback requires redeploying code, it will take 5–15 minutes. That's 5–15 minutes
    of degraded service for your users. Instant rollback (seconds) is a design requirement,
    not a nice-to-have.
    """)
    return


@app.cell
def stage5_rollback_triggers(mo):
    mo.md("""
    ### Rollback Triggers

    **Automatic triggers** (no human needed — these fire immediately):
    - Error rate > 2x baseline for 5 minutes
    - Latency p99 > 2x baseline sustained
    - Prediction distribution KL divergence > threshold
    - Model output null rate > 1% (serving failure)

    **Manual triggers** (human decision required):
    - Business metrics declining (may need hours or days to detect — too slow for automation)
    - User complaint spike in support tickets
    - Downstream system failures traced to model output
    - Stakeholder escalation ("something is wrong with the recommendations")

    **The philosophy:**

    > "The best rollback is one that fires before users notice. Automated triggers on error
    > rate and latency catch 80% of bad deployments within minutes. Business metric triggers
    > catch the other 20% — but they take hours to accumulate enough signal, so humans
    > need to be in the loop."

    **Post-rollback process:**
    1. Rollback immediately (stop the bleeding)
    2. Root cause analysis (what caused the regression?)
    3. Fix the issue in a new model version
    4. Re-run the full CI/CD pipeline before re-deploying
    Never fast-path a model back to production after a rollback without understanding why it failed.
    """)
    return


@app.cell
def full_architecture(mo):
    mo.md("""
    ## Full MLOps Architecture

    ```
    ┌─────────────────────────────────────────────────────────┐
    │                    DEVELOPMENT                          │
    │                                                         │
    │  [Feature Branch] → [CI Pipeline]                       │
    │      Code + Data + Config changes                       │
    │      Lint → Tests → Data Validation → Smoke Train       │
    │      → Eval vs Production → PR Report                   │
    └───────────────────────┬─────────────────────────────────┘
                            │ merge
    ┌───────────────────────▼─────────────────────────────────┐
    │                    TRAINING                             │
    │                                                         │
    │  [Full Training] → [Evaluation] → [Model Registry]     │
    │      Experiment tracking (MLflow/W&B)                   │
    │      Data + code + model versioned together             │
    │      Staging → Production lifecycle in registry         │
    └───────────────────────┬─────────────────────────────────┘
                            │ promote to staging
    ┌───────────────────────▼─────────────────────────────────┐
    │                    DEPLOYMENT                           │
    │                                                         │
    │  [Shadow] → [Canary 1%] → [Canary 5%] → [A/B 50%]     │
    │          → [Full Rollout 100%]                          │
    │      Automated rollback at each gate                    │
    │      Business metrics validated before expansion        │
    │      Old model stays warm until full rollout confirmed  │
    └───────────────────────┬─────────────────────────────────┘
                            │ serving
    ┌───────────────────────▼─────────────────────────────────┐
    │                    MONITORING                           │
    │                                                         │
    │  [Prediction Logs] → [Drift Detection] → [Alerts]      │
    │  [Business Metrics] → [Dashboards]                      │
    │  [Retrain Trigger] ──────────────────► back to TRAINING │
    └─────────────────────────────────────────────────────────┘
    ```

    **The feedback loop is the whole point.** Monitoring feeds back into training.
    Drift triggers retraining. Retraining produces a new candidate. The candidate runs
    through the full pipeline again. This is the Level 3 maturity loop.
    """)
    return


@app.cell
def llm_mlops(mo):
    mo.md("""
    ## MLOps for LLM Applications

    LLM apps (Canopy, Daily Briefing Agent) have different MLOps needs than traditional ML.

    ### What Changes

    | Traditional ML | LLM Apps |
    |---------------|----------|
    | Train model weights | Prompt a hosted model — no training step |
    | Version model artifacts | Version prompts in git |
    | Eval on held-out test set | Eval with LLM-as-judge or human raters |
    | Cost = compute (training + serving) | Cost = API tokens (every call costs money) |
    | Latency = model inference time | Latency = API call + output token generation (variable) |

    **Prompt versioning replaces model versioning:**
    - The prompt is your "model." Version it in git.
    - Run evals on every prompt change before merge — same gate as model eval in CI.
    - Canary prompt deployments: route % of traffic to new prompt version, monitor quality + cost.

    **Cost monitoring matters more in LLM apps:**
    - A verbose prompt that generates longer outputs = higher per-request cost.
    - A prompt change that increases average output length by 20% = 20% cost increase.
    - Add token count and estimated cost to your eval metrics.

    ### What Stays the Same

    - CI/CD pipeline structure (lint → test → eval → deploy → monitor)
    - Canary/shadow deployment (route % of traffic to new prompt version)
    - Rollback (revert to previous prompt version — it's a git revert)
    - Monitoring (drift in output quality, cost per request, latency)

    ---

    **My project mapping:**

    > **Daily Briefing Agent**: already doing lightweight LLM CI — GitHub Actions runs
    > regression evals on prompt changes. This is the right pattern. Extending it to
    > canary-style prompt deployment is the natural next step: deploy new prompt to 5%
    > of briefing runs, compare quality scores, expand if better.

    > **Canopy**: scoring prompts are the model. Adding eval CI before merging prompt
    > changes would close the loop on CI/CD for this project.
    """)
    return


@app.cell
def tools_landscape(mo):
    mo.md("""
    ## Tools Landscape

    | Function | Open Source | Managed / SaaS |
    |----------|------------|----------------|
    | **Experiment tracking** | MLflow, W&B (free tier) | W&B Teams, Neptune, Comet |
    | **Data versioning** | DVC, LakeFS | Delta Lake (Databricks) |
    | **Pipeline orchestration** | Prefect, Airflow, Dagster | Vertex AI Pipelines, SageMaker Pipelines |
    | **Model serving** | Seldon, KServe, BentoML, FastAPI | SageMaker Endpoints, Vertex AI |
    | **Feature store** | Feast | Tecton, Databricks Feature Store |
    | **Monitoring / drift** | Evidently, WhyLabs (free tier) | Arize, Fiddler, WhyLabs |
    | **CI/CD** | GitHub Actions, GitLab CI | — |

    ---

    ### My Stack Choice

    **GitHub Actions** (CI) + **MLflow** (experiment tracking) + **DVC** (data versioning)
    + **Prefect** (orchestration) + **FastAPI** (serving) + **Evidently** (monitoring)

    **Why this stack:**
    - All open source — no vendor lock-in, no per-seat licensing
    - All tools I've used or studied across my projects
    - GitHub Actions is already in use for Daily Briefing Agent CI
    - FastAPI is already in use for my RAG app serving layer
    - The stack is cohesive: everything integrates through Python APIs

    **Where I'd pay for managed:**
    - W&B for a team (the UI and collaboration features are genuinely better)
    - Databricks if the data volume justifies it (Delta Lake is excellent at scale)
    """)
    return


@app.cell
def interviewer_followups(mo):
    mo.md("""
    ## Common Interviewer Follow-ups

    **"How do you test ML code?"**
    > Three layers: (1) unit tests for feature engineering and data transforms — deterministic
    > inputs, deterministic outputs, easy to test. (2) Integration tests for pipeline
    > end-to-end — does a training run produce a model artifact? (3) Eval tests for model
    > quality — metrics above threshold on held-out data. (4) Diff tests for model comparison
    > — new candidate vs current production, block if regression.

    **"How do you handle a model that degrades slowly?"**
    > Monitoring with drift detection — Population Stability Index (PSI) and KL divergence
    > measure distribution shifts in inputs and predictions. Set automated retrain triggers
    > when metrics cross threshold. But also: scheduled retraining (weekly or monthly)
    > as a backstop — don't rely solely on detection catching drift. The detection might
    > miss gradual concept drift that never crosses a hard threshold.

    **"What's the difference between canary and A/B?"**
    > Canary is a safety mechanism: small traffic slice, short duration, looking for hard
    > failures — errors, latency spikes, crashes. It answers "is this broken?"
    > A/B is a scientific experiment: controlled 50/50 split, 1-2 week duration,
    > measuring business impact with statistical rigor. It answers "is this better?"
    > You typically run canary first (safety gate), then A/B (business validation).

    **"How do you version ML models?"**
    > Model registry (MLflow). Each entry links to: git commit SHA (code), data snapshot
    > hash (data), hyperparameters, offline metrics, training timestamp, and deployment
    > history. Full lineage means you can reproduce any production model from the registry
    > entry alone. And you can answer "what was live on this date?" from the deployment log.

    **"What's the hardest part of MLOps?"**
    > Not the tools — it's organizational discipline. Getting data scientists to version
    > their data, log their experiments, write eval tests, and not skip the staging step
    > when they're excited about a model. The tooling is a solved problem. Changing the
    > culture around rigor in the ML development process is the actual challenge.
    """)
    return


@app.cell
def interview_talking_points(mo):
    mo.md("""
    ## Interview Talking Points

    ### "Walk me through how you'd deploy an ML model"

    > Start with shadow deployment — run the new model alongside production, log its
    > predictions, zero user exposure. Validates the serving infrastructure without risk.
    > Then canary to 1-5% — monitor error rate, latency, prediction distribution for
    > a few hours. Then canary to 25%, then 50%, each with a monitoring window.
    > After enough signal, run A/B at 50/50 for 1-2 weeks to get statistical confidence
    > on business metrics. Then full rollout. Automated rollback at every gate — rollback
    > is a config change that takes seconds. Old model stays loaded until the new one is
    > fully validated.

    ---

    ### "Have you implemented CI/CD for ML?"

    > Yes, for my LLM applications. My Daily Briefing Agent runs regression evals via
    > GitHub Actions on every prompt change — that's the LLM equivalent of model evaluation
    > in CI. The PR is blocked if the new prompt regresses on summarization quality vs
    > the baseline. For a traditional ML system, I'd extend this with data validation
    > via Pandera, a training smoke test on a sample of production data, and a model
    > comparison gate that posts the eval diff as a PR comment.

    ---

    ### "How does MLOps connect to your other work?"

    > My backtesting engine IS an evaluation framework — it tests trading strategies
    > against historical data before deployment. That's exactly what an ML eval pipeline
    > does: test a candidate against held-out data before promoting to production.
    > The walk-forward validation pattern I use prevents look-ahead bias — the same
    > temporal split concept that prevents data leakage in ML train/test splits.
    > The mental model transfers directly; I just need to wrap it in the MLOps tooling
    > and deployment infrastructure.

    ---

    ### One-sentence definitions for speed rounds

    - **MLflow**: experiment tracking + model registry, open source
    - **DVC**: git for data — version datasets alongside code
    - **Shadow deploy**: run new model in production without serving its output to users
    - **Canary**: route small % of real traffic to new model, monitor, expand gradually
    - **A/B test**: controlled experiment measuring business impact of model change
    - **Model registry**: versioned catalog of trained models with staging → production lifecycle
    - **Data drift**: input distribution shifts away from training distribution
    - **Concept drift**: relationship between inputs and labels changes over time
    """)
    return


if __name__ == "__main__":
    app.run()
