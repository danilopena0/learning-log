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
    # Design a Model Training Platform

    | Field  | Value |
    |--------|-------|
    | Date   | 2026-05-06 |
    | Track  | System Design |
    | Time   | 60 min |
    | Topics | Distributed Training · Experiment Tracking · Model Registry · Reproducibility · HPO |
    """)
    return


@app.cell
def interview_framing(mo):
    mo.md("""
    ## Interview Framing

    "Design a model training platform" is asked at every ML-focused company: Google, Meta,
    Airbnb, Uber, Netflix, and increasingly at mid-stage companies building internal ML infra.

    ---

    **What interviewers want to hear:** you understand that training a model in production is
    NOT running a notebook. It's orchestrating data, compute, experiments, artifacts, and team
    workflows at scale.

    **The mental model:** a training platform is a CI/CD system specialized for ML. Code changes
    trigger builds (training runs), tests (evaluation), and deployments (model promotion). Every
    concept from software CI/CD has an ML analogue:

    | CI/CD Concept | ML Training Platform Analogue |
    |---------------|-------------------------------|
    | Code commit | Experiment config change |
    | Build | Training run |
    | Test suite | Evaluation / offline metrics |
    | Artifact (binary) | Model weights |
    | Staging environment | Model registry: Staging |
    | Production deploy | Model promotion → serving |
    | Build cache | Checkpoint resumption |
    | Build matrix | Hyperparameter sweep |

    **Personal connection:** *"My backtesting engine is a domain-specific training platform:
    it orchestrates strategy experiments, tracks parameters and results, manages data versioning
    with walk-forward splits, and produces evaluated artifacts. Same pattern, different domain."*
    """)
    return


@app.cell
def stage1_problem_framing(mo):
    mo.md("""
    ## Stage 1: Problem Framing

    > **Scenario:** "Design a model training platform for a company with 50 ML engineers
    > training 100+ models across recommendation, fraud detection, NLP, and computer vision."
    >
    > The platform must support the full cycle:
    > **data access → experimentation → training → evaluation → artifact management → deployment handoff**

    ---

    ### Clarifying Questions + Assumptions

    | Question | Assumption |
    |----------|------------|
    | Team size? | 50 ML engineers, 10 ML platform engineers. Users are sophisticated but shouldn't manage infrastructure. |
    | Scale? | Largest models: 8 GPUs × 12 hours. Most models: 1 GPU, <1 hour. ~200 training jobs/day. |
    | Frameworks? | PyTorch primary, some TensorFlow legacy, occasional JAX. Must be framework-agnostic. |
    | Data? | Tabular (Spark/Parquet), text (S3 files), images (S3 buckets). 10TB total data lake. Feature store exists. |
    | Cloud? | AWS assumed (adaptable to GCP/Azure). Mix of on-demand and spot instances for cost. |
    | Compliance? | Model lineage and reproducibility required for audit. Every production model traceable to data + code + config. |

    ---

    ### Success Metrics

    **Platform health:**
    - Time from "I have an idea" to "model is trained and evaluated" < 30 min for standard jobs
    - GPU utilization > 70%
    - Platform uptime > 99.5%

    **User satisfaction:**
    - NPS from ML engineers > 8/10
    - No one is SSH-ing into raw instances

    **Operational:**
    - <5% job failure rate from platform issues (not user bugs)
    - Full reproducibility for any past experiment
    """)
    return


@app.cell
def stage2_architecture(mo):
    mo.md("""
    ## Stage 2: Architecture Overview

    Three planes: control, compute, data+storage.

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                      USER INTERFACE                         │
    │                                                             │
    │  [CLI] ──→ submit jobs, check status, pull artifacts        │
    │  [Web UI] ──→ experiment dashboard, model registry browser  │
    │  [SDK (Python)] ──→ programmatic job submission, logging    │
    └───────────────────────────┬─────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────┐
    │                    CONTROL PLANE                             │
    │                                                             │
    │  [API Server] ──→ authentication, job management, querying  │
    │  [Scheduler] ──→ job queue, priority, resource allocation   │
    │  [Orchestrator] ──→ spin up compute, mount data, run job    │
    └───────────────────────────┬─────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────┐
    │                    COMPUTE PLANE                             │
    │                                                             │
    │  [GPU Cluster] ──→ training jobs (spot + on-demand mix)     │
    │  [CPU Pool] ──→ data preprocessing, evaluation, HPO         │
    │  [Distributed] ──→ multi-GPU, multi-node (PyTorch DDP)      │
    └───────────────────────────┬─────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────┐
    │                    DATA + STORAGE PLANE                      │
    │                                                             │
    │  [Data Lake (S3)] ──→ raw data, processed datasets          │
    │  [Feature Store] ──→ pre-computed features (Feast/Tecton)   │
    │  [Artifact Store (S3)] ──→ model weights, configs, logs     │
    │  [Experiment DB (Postgres)] ──→ run metadata, metrics       │
    │  [Model Registry] ──→ versioned models with lineage         │
    └─────────────────────────────────────────────────────────────┘
    ```

    ---

    **Why three planes?** Separation of concerns + independent scaling:
    - Control plane: scales with number of users / jobs, not compute
    - Compute plane: scales with job volume and GPU demand — auto-scales
    - Data plane: scales with data volume and number of experiments — append-only writes, heavy reads

    **Connection to my fraud detection design:** the feature store in the data plane is the
    same feature store from Week 3. The training platform is the operational layer that sits
    *above* the feature store — it reads pre-computed features, trains models, and hands
    trained artifacts to the serving infrastructure from my MLOps notebook (Week 4).
    """)
    return


@app.cell
def stage3_experiment_tracking(mo):
    mo.md("""
    ## Stage 3: Experiment Tracking — The Brain of the Platform

    > *"If I can't recreate any historical experiment from its metadata alone, the
    > tracking system has failed."*

    ---

    ### What to Track for Every Run

    **Inputs** (what produced this model):
    - Git commit hash + branch (exact code state)
    - Config / hyperparameters (every value, not just the interesting ones)
    - Data snapshot ID (hash of the training dataset, not just a path)
    - Feature set version (which feature generation pipeline produced this data)
    - Environment: Docker image hash (reproducible compute environment)

    **Outputs** (what the model produced):
    - Metrics: train loss, val loss, eval metrics per epoch (time series, not just final value)
    - Artifacts: model weights, plots, logs, confusion matrices
    - Timing: start time, end time, GPU hours consumed (for cost attribution)

    **Lineage** (the chain of custody):
    - Which data + which code + which config = which model
    - Must be queryable: "show me all models trained on fraud data v2.3"
    - Must be reproducible: "rerun experiment abc123 exactly"

    ---

    ### Tracking Architecture

    ```
    [Training Job]
        → [SDK Logger] (in-process, lightweight)
            → log_param("learning_rate", 0.001)
            → log_metric("val_loss", 0.34, step=100)
            → log_artifact("model.pt", "/path/to/weights")
        → [Tracking Server] (centralized)
            → Postgres (structured metadata: params, metrics, run info)
            → S3 (large artifacts: model weights, datasets, plots)
        → [Web UI]
            → Compare runs side-by-side
            → Filter/sort by metric
            → Reproduce: one-click "rerun this experiment"
    ```

    ---

    ### SDK Design — What the User Writes

    ```python
    import mlplatform as ml

    with ml.start_run(experiment="fraud-detection-v3") as run:
        run.log_params({"lr": 0.001, "epochs": 50, "model": "xgboost"})
        run.log_dataset("s3://data/fraud/v2.3", split="train")

        model = train(...)

        run.log_metrics({"auc_pr": 0.87, "precision_at_100": 0.92})
        run.log_artifact(model, "model.pt")
        run.log_artifact(confusion_matrix_plot, "confusion_matrix.png")

    # Everything is now tracked, versioned, and reproducible
    ```

    **Key design principles:**
    - **Zero config:** logging works with one import. No setup, no server URL config in user code.
    - **Automatic capture:** git hash, Docker image, start time captured at `start_run()` — user doesn't need to remember.
    - **Idempotent:** logging the same metric twice at the same step is a no-op, not a corruption.
    - **Offline-capable:** logs buffer locally, sync when server is available. A network blip doesn't kill a 12-hour training job.
    - **Context manager pattern:** `with` block ensures the run is closed + synced even if training crashes.

    ---

    ### Tool Comparison

    | Tool | Type | Strengths | Weaknesses |
    |------|------|-----------|------------|
    | MLflow | Open source | Full lifecycle, model registry, broad adoption | UI is basic, scaling requires work |
    | W&B (Weights & Biases) | SaaS | Beautiful UI, collaboration, sweeps | Vendor lock-in, cost at scale |
    | Neptune | SaaS | Great for team collaboration | Less model serving integration |
    | ClearML | Open source | Agent-based, auto-logging | Smaller community |
    | Custom (Postgres + S3) | DIY | Full control, exact requirements | Build and maintain everything |

    **My recommendation:** MLflow for most teams. W&B if budget allows and team collaboration
    is priority. Custom only for FAANG-scale with a dedicated platform team.
    """)
    return


@app.cell
def stage4_distributed_training(mo):
    mo.md("""
    ## Stage 4: Distributed Training

    ---

    ### Why Distribute?

    1. **Model doesn't fit in one GPU memory** — LLMs: 7B params ≈ 14GB in fp16. One A100 = 80GB. Anything beyond ~40B params needs multi-GPU.
    2. **Training is too slow** — dataset is huge or model is large. 1M images × 100 epochs on a single GPU might take weeks.
    3. **Maximize expensive GPU utilization** — at $3–8/hr per GPU, idle capacity is money burning.

    ---

    ### Three Parallelism Strategies

    **1. Data Parallelism (DDP — DistributedDataParallel)**
    - Same model on every GPU; split the data. Each GPU computes gradients on its batch, gradients
      are averaged (all-reduce via NCCL), weights synchronized.
    - PyTorch DDP is the standard. Each process has a full model copy.
    - Scales linearly for most models up to ~64 GPUs. Beyond that, communication overhead dominates.
    - **When to use:** model fits on one GPU but training is slow. This is the most common case.

    **2. Model Parallelism**
    - Split the MODEL across GPUs. Layers 1–12 on GPU 0, layers 13–24 on GPU 1.
    - *Pipeline parallelism:* split by layers, use micro-batches to keep all GPUs busy (hide
      pipeline bubbles). Reduces inter-GPU communication to activations at boundaries.
    - *Tensor parallelism:* split individual layers across GPUs (each GPU has a subset of attention
      heads). More communication but finer granularity.
    - **When to use:** model doesn't fit on one GPU. Primarily for LLM training.

    **3. Fully Sharded Data Parallelism (FSDP)**
    - Shard model parameters, gradients, AND optimizer state across GPUs. Each GPU stores only
      1/N of the full model. Parameters are gathered on-the-fly for each forward/backward pass.
    - PyTorch FSDP: combines benefits of data and model parallelism in a unified abstraction.
    - **When to use:** model almost doesn't fit on one GPU, or maximum memory efficiency needed.
      Simpler to implement than custom pipeline parallelism.

    ---

    ### Decision Framework

    ```
    Does the model fit on one GPU?
    ├── YES → Data Parallelism (DDP)
    │   └── How many GPUs?
    │       ├── 2–8 → single-node DDP (fastest, no network overhead)
    │       └── 8+ → multi-node DDP (need fast interconnect: NVLink or InfiniBand)
    └── NO → Model Parallelism or FSDP
        ├── Slightly too large → FSDP (simplest to implement correctly)
        └── Much too large (100B+) → Pipeline + Tensor + Data parallelism combined
            └── This is Megatron-LM / DeepSpeed territory
    ```

    ---

    ### What the Platform Must Abstract Away

    The user should never configure distributed training manually.

    **User writes:**
    ```
    ml submit --gpus 8 --nodes 2 train.py
    ```

    **Platform handles:**
    - Allocating 2 nodes with 4 GPUs each from the cluster
    - Setting `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK` env vars
    - Installing NCCL, setting up the process group
    - Mounting the training data at the same path on all nodes
    - Launching 8 processes (one per GPU) with `torchrun`
    - Collecting logs from all processes, deduplicating worker logs
    - Reporting aggregate metrics to the experiment tracker

    **Environment:** Docker containers with CUDA, PyTorch, NCCL pre-installed. User specifies a
    base image + `requirements.txt`. Platform builds the final image and caches it.

    **Fault tolerance:** if one node dies during multi-node training:
    - Auto-restart from the latest checkpoint (requires periodic checkpointing)
    - NOT restart from scratch (would waste hours of GPU time)
    - Alert the user if the job fails after N retries
    - For spot instances: checkpoint every 30 min + handle SIGTERM (AWS spot interruption signal)
      by saving state before the instance is reclaimed
    """)
    return


@app.cell
def stage5_hpo(mo):
    mo.md("""
    ## Stage 5: Hyperparameter Optimization (HPO)

    > The platform already manages job submission and experiment tracking.
    > HPO is a natural extension — it's just many training runs with different configs,
    > intelligently scheduled.

    ---

    ### Strategies

    **Grid search:** try all combinations. Exhaustive but exponential — 5 params with 5 values
    each = 3125 runs. Only for small spaces (2–3 params, few values each). Rarely justified.

    **Random search:** sample randomly from distributions. Surprisingly effective — often beats
    grid search because it explores the high-dimensional space more uniformly (Bergstra & Bengio,
    2012). If only one hyperparameter matters, random search finds it faster than grid search.
    *Use this as the baseline HPO strategy.*

    **Bayesian optimization:** model the objective function with a surrogate (Gaussian process
    or tree-based model). Use the surrogate to select the most promising next config to evaluate.
    More efficient than random for expensive evaluations (>30 min runs).
    - Tools: **Optuna** (best UX), Ray Tune, W&B Sweeps

    **Hyperband / successive halving:** start many configs with small budgets (few epochs),
    kill underperformers early, give more budget to survivors. Treats HPO as a resource
    allocation problem — don't waste GPU on configs that are clearly bad.

    **BOHB (Bayesian Optimization + Hyperband):** combine Bayesian search with Hyperband's
    early stopping. State of the art for automated HPO on expensive models.

    ---

    ### Platform Integration

    ```python
    sweep = ml.Sweep(
        experiment="fraud-detection-hpo",
        method="bayesian",
        metric={"name": "auc_pr", "goal": "maximize"},
        parameters={
            "learning_rate": {"distribution": "log_uniform", "min": 1e-4, "max": 1e-1},
            "max_depth": {"values": [3, 5, 7, 9]},
            "subsample": {"distribution": "uniform", "min": 0.6, "max": 1.0},
            "n_estimators": {"distribution": "int_uniform", "min": 100, "max": 1000},
        },
        max_runs=50,
        early_terminate={"type": "hyperband", "min_iter": 10},
    )

    # Platform manages: scheduling runs, tracking results, suggesting next config
    sweep.run(train_function, gpus_per_run=1)
    ```

    The platform schedules runs one at a time (for Bayesian — needs previous results) or in
    parallel batches (for random/grid — fully independent). All runs are tracked in the
    experiment tracker under the sweep ID. Results visualized as a parallel coordinates plot.
    """)
    return


@app.cell
def stage6_model_registry(mo):
    mo.md("""
    ## Stage 6: Model Registry — From Experiment to Production

    > The model registry is the handoff point between "training" and "serving."
    > It is NOT just an artifact store — it manages versioning, lifecycle, and lineage.

    ---

    ### What the Registry Stores

    - **Model artifact:** weights file, serialized model, ONNX export (for framework-agnostic serving)
    - **Metadata:** training run ID (links to experiment tracker → full lineage), metrics, hyperparameters
    - **Lineage:** data version, code commit, feature set version, Docker image hash
    - **Stage:** `Staging` → `Production` → `Archived`
    - **Input/output signature:** what features does the model expect? what types does it output?
      Prevents serving mismatches at deployment time.

    ---

    ### Model Lifecycle

    ```
    [Experiment Tracking]
        → ML engineer finds the best run (filter by auc_pr > 0.87)
        → "Register this model" (one click or one API call)

    [Model Registry: Staging]
        → Model artifact stored in versioned artifact store (S3 + immutable versioning)
        → Linked to training run (full lineage — code, data, config, environment)
        → Automated validation gates:
            - Schema check: input/output shapes match the expected signature
            - Performance check: metrics above minimum thresholds (no regression vs current prod)
            - Bias/fairness check: demographic parity metrics within acceptable bounds
            - Model size check: artifact fits within serving infrastructure constraints
            - Smoke test: model loads, runs inference on 10 sample inputs without error

    [Model Registry: Production]
        → Approved (manual sign-off or automated after all validation gates pass)
        → Deployment config generated (serving resource requirements, autoscaling settings)
        → Handed off to serving infrastructure (CD pipeline — see MLOps notebook, Week 4)

    [Model Registry: Archived]
        → Previous production model moved here when new model takes over
        → Retained for rollback capability and audit trail
        → Artifacts may be moved to cold storage after retention period (e.g., 90 days)
    ```

    ---

    ### Registry API

    ```python
    # Register a model from a completed training run
    ml.register_model(
        run_id="abc123",
        model_name="fraud-detector",
        model_version="3.2",
        artifacts={"model": "model.pt", "config": "config.yaml"},
        signature=ModelSignature(
            inputs=Schema([("amount", "float"), ("merchant_id", "int"), ("hour_of_day", "int")]),
            outputs=Schema([("fraud_prob", "float"), ("reason_codes", "list[str]")]),
        ),
    )

    # Promote to production (after validation gates pass)
    ml.transition_model_stage("fraud-detector", version="3.2", stage="production")

    # Query production models
    production_models = ml.list_models(stage="production")

    # Rollback: promote the previous version back (< 1 minute to execute)
    ml.transition_model_stage("fraud-detector", version="3.1", stage="production")
    ```

    **Connection to fraud detection design (Week 3):** when I said "every declined transaction
    needs a traceable reason code" — this is how that becomes technically possible. The model
    registry ties each prediction to a specific model version, which ties to a training run,
    which ties to the data and features used. Auditors can trace any decision back through the
    full lineage chain.
    """)
    return


@app.cell
def stage7_reproducibility(mo):
    mo.md("""
    ## Stage 7: Reproducibility & Governance

    > Reproducibility is not a nice-to-have. For regulated industries (finance, healthcare),
    > it is a legal requirement. For any company, it is what separates "we think this model
    > is good" from "we can prove this model is good."

    ---

    ### Reproducibility Requirements

    **The standard:** given any model in the registry, I can recreate it exactly:

    > Same code (git commit) + same data (data snapshot hash) + same config (logged params)
    > + same environment (Docker image hash) = same model (within floating-point tolerance)

    **Platform enforcements:**
    - Auto-log git hash and Docker image at job start (before the user's code runs)
    - Require data versioning: reject job submissions without a data snapshot ID
    - Store the full config as an immutable artifact alongside the model weights
    - One-click "reproduce this run" from the UI: platform checks out the exact commit,
      pulls the exact Docker image, retrieves the exact dataset snapshot, resubmits the job

    ---

    ### Governance for Regulated Industries

    **Model cards:** auto-generated documentation for each registered model.
    Contents: performance metrics, fairness metrics, intended use, known limitations, training
    data summary, evaluation methodology. Generated from experiment tracker metadata — no
    manual documentation overhead.

    **Approval workflows:**
    - Production promotion requires sign-off from model owner + a second reviewer
    - Changes to high-risk models (credit scoring, fraud detection) require compliance sign-off
    - Approval state stored in registry audit log — not just "who approved" but "what they saw"

    **Audit trail:**
    - Every stage transition logged: timestamp, user, reason, metrics at time of decision
    - Immutable: audit records cannot be modified or deleted (append-only log)
    - Queryable: "show me all models deployed to production in the last 90 days"

    **Data lineage for GDPR:**
    - Right to explanation: any individual prediction traceable to the model version
    - Right to erasure: if a user's data must be deleted, identify which models were trained
      on it and flag for retraining (data poisoning / membership inference risk)

    **Connection to MLOps notebook (Week 4):** this governance layer is the operational layer
    that sits between the training platform and the serving infrastructure. The model registry
    is the handoff point — training produces a versioned, validated, documented artifact; serving
    consumes it. The platform enforces the contract that every artifact in the registry meets
    the governance bar.
    """)
    return


@app.cell
def key_tradeoffs(mo):
    mo.md("""
    ## Key Trade-offs

    | Decision | Option A | Option B | My Recommendation |
    |----------|----------|----------|-------------------|
    | Build vs buy | Custom platform (full control, fits exact needs) | Managed (SageMaker, Vertex AI, AzureML) | Managed for <20 ML engineers. Custom if a dedicated platform team (≥5 engineers) exists. |
    | Experiment tracking | MLflow (open source, self-hosted) | W&B (SaaS, polished UI) | MLflow to avoid vendor lock-in. W&B if collaboration and visualization are the priority. |
    | Compute mix | 100% on-demand GPU (reliable, expensive) | 100% spot/preemptible (70% cheaper, interruptible) | 70% spot + 30% on-demand. Spot for HPO and exploration; on-demand for final production training runs. |
    | Distributed strategy | DDP (simple, full model on each GPU) | FSDP (sharded, memory-efficient) | DDP unless the model doesn't fit on one GPU. FSDP adds complexity; only pay that cost when needed. |
    | HPO | Manual tuning by the data scientist | Automated (Optuna + Hyperband) | Automated. Human intuition doesn't scale and introduces experimenter bias. |
    | Docker images | One fat image with all frameworks installed | Per-framework thin base images + user overlay | Per-framework base images. A 20GB "everything" image is slow to pull and a security liability. |
    | Job scheduling | FIFO queue (simple, fair) | Priority queue with preemption | Priority with preemption. Production retraining jobs must be able to preempt long-running exploration jobs. |
    | Artifact storage | Centralized single bucket | Per-team buckets with separate IAM | Centralized with IAM policies per prefix. Team isolation without operational overhead. |
    """)
    return


@app.cell
def follow_ups(mo):
    mo.md("""
    ## Common Interviewer Follow-ups

    ---

    **"How do you handle GPU cost optimization?"**

    > - Spot/preemptible instances for experimentation (70% savings vs on-demand)
    > - Auto-scaling GPU pool based on queue depth: scale up when queue > N jobs, scale down when idle > 15 min
    > - Job packing: run multiple small jobs on one large GPU instance via NVIDIA MIG (Multi-Instance GPU)
    > - HPO with early stopping: Hyperband kills underperforming configs early, recovering GPU budget
    > - Cost attribution: every job tagged to team + experiment for monthly chargebacks — creates incentives

    ---

    **"What if a training job runs for 48 hours and the node dies?"**

    > - Periodic checkpointing every 30 min (platform SDK provides `run.checkpoint(model, optimizer, step)`)
    > - Checkpoints stored on S3 (survives node death, accessible to any replacement node)
    > - Platform auto-restarts from latest checkpoint on a new node — no user intervention
    > - For spot instances: attach a SIGTERM handler that flushes a checkpoint when AWS signals
    >   interruption (typically 2 min warning). Platform auto-resumes on a new spot instance.
    > - Platform does NOT restart from scratch after N retries — it pages the user

    ---

    **"How do you prevent different team members from stepping on each other?"**

    > - Namespaced experiments per team/project — no cross-team name collisions
    > - Resource quotas per team: max concurrent GPUs, max queue depth, max job duration
    > - Model registry approval workflows: only the model owner can register; promotion requires a second approver
    > - Git-based code versioning: no "I ran this locally" — the platform rejects jobs without a committed git state
    > - Read-only access to other teams' experiments (visibility) without write access (no accidental modification)

    ---

    **"How would you handle a team that wants to use JAX when the platform is PyTorch-focused?"**

    > - Framework-agnostic container-based execution: the platform schedules containers, not Python scripts
    > - User provides a Docker image + entrypoint. Platform handles scheduling, logging, artifact collection.
    > - The SDK wraps standard logging calls — it doesn't care what's inside the container
    > - For logging: the SDK communicates over a local Unix socket or HTTP — framework-agnostic
    > - This is the correct abstraction boundary: the platform owns compute orchestration and tracking;
    >   the user owns the training code and framework

    ---

    **"What about data security and access control?"**

    > - Role-based access to datasets: not all teams see all data (fraud data is PII-sensitive)
    > - Encrypted artifacts at rest (S3 SSE) and in transit (TLS)
    > - Audit logging for all data access: who accessed what dataset, when, from which job
    > - Model registry permissions: only model owner can promote to production; read access is broader
    > - Feature store row-level security: teams can access aggregated features without raw PII
    """)
    return


@app.cell
def talking_points(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    ### 60-Second Pitch

    > "I'd design a three-plane architecture: a **control plane** (API server, scheduler,
    > orchestrator), a **compute plane** (auto-scaling GPU cluster with a 70/30 spot + on-demand
    > mix), and a **data plane** (data lake, feature store, experiment DB backed by Postgres,
    > artifact store on S3, and a model registry).
    >
    > Users interact via a Python SDK and a web UI — they write three lines of logging code
    > and the platform handles everything else. The platform auto-captures experiment lineage
    > at job start (git hash, Docker image, data snapshot ID), supports distributed training
    > via DDP and FSDP without the user configuring NCCL, and manages the full model lifecycle
    > from experiment to production with mandatory validation gates and an immutable audit trail."

    ---

    ### The Senior Signal

    > *"The hardest part isn't the training — it's the reproducibility and governance. Any
    > model in production must be traceable to exact code, data, config, and environment.
    > The platform enforces this by auto-logging lineage at job start. This is what regulators
    > and auditors actually care about, and it's what separates a notebook-based workflow
    > from a production-grade platform."*

    This signals you've thought about the system's obligations beyond the happy path.

    ---

    ### Connecting to My Work

    **Backtesting engine as a training platform:**
    My backtesting engine is structurally a training platform for trading strategies. It
    orchestrates experiments (strategy parameter variants), tracks inputs and results
    (walk-forward metrics, drawdowns, Sharpe ratios), manages data versioning (rolling
    time windows, out-of-sample splits), and produces evaluated artifacts (strategy configs
    ready for paper trading). The architecture patterns — experiment tracking, artifact
    management, evaluation gates before promotion — are identical. The domain is different;
    the engineering is the same.

    **Fraud detection design (Week 3) as a consumer of this platform:**
    When I said "every declined transaction needs a traceable reason code" — this is the
    model registry's lineage that makes it possible. The fraud model is trained on this
    platform, registered with full lineage, and the serving infrastructure references the
    registered version. Auditors can trace any prediction back to training data and config.

    **MLOps notebook (Week 4) as the operational layer above this platform:**
    The training platform produces validated model artifacts. The MLOps layer (deployment,
    monitoring, drift detection, retraining triggers) consumes them. The model registry is
    the handoff interface between the two layers.

    ---

    ### Red Flags to Avoid

    - Jumping straight to "we'd use Kubernetes" without explaining WHY (compute plane
      does need K8s, but motivate it: auto-scaling, pod isolation, resource quotas)
    - Treating experiment tracking as optional — it's the most important component for
      reproducibility and team collaboration
    - Forgetting the data versioning problem: pointing at a path like `s3://data/fraud/`
      is NOT versioning — you need a snapshot hash that identifies the exact state at
      training time
    - Conflating experiment tracking with the model registry — they serve different purposes:
      tracking is for exploration; the registry is for promotion and governance
    - Not mentioning fault tolerance for long-running distributed jobs — spot instances
      are the correct cost choice, but they require checkpoint-based restart support
    """)
    return


if __name__ == "__main__":
    app.run()
