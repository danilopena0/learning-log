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
    # Recommender Systems — Collaborative Filtering, Content-Based, Hybrid, Matrix Factorization vs Deep

    | Field | Value |
    |-------|-------|
    | Date  | 2026-05-03 |
    | Track | ML Theory |
    | Time  | 60 min |
    | Topics | Content-Based · Collaborative Filtering · Matrix Factorization · Two-Tower · Hybrid · Evaluation |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## The Recommendation Problem

    The input is a **user-item interaction matrix**: users as rows, items as columns, values
    representing some signal (rating, click, application, purchase). Most entries are missing —
    the user simply hasn't encountered that item yet.

    **The goal**: predict the missing entries — which items will a user like that they haven't
    seen yet?

    **Scale**: Netflix has 230M users × 15K titles = 3.4 trillion possible interactions. 99.9%+
    are unknown. You can't enumerate them; you need to generalize from sparse signal.

    ---

    ### Two Fundamental Approaches

    1. **Collaborative filtering** — "users who liked similar items will like similar items."
       Uses interaction patterns only. No item features required. The wisdom of crowds.

    2. **Content-based filtering** — "recommend items similar to what the user liked before."
       Uses item features (skills, location, level). Works from day one; no community needed.

    ---

    ### Why This Matters for AI Engineering

    Recommendation **is** retrieval. The **two-tower architecture** from the RAG system design
    notebook is the same pattern used at YouTube, Spotify, and Pinterest:

    - Encode the query (user profile) with one encoder.
    - Encode all documents (items) with another encoder.
    - Score by dot product; retrieve top K with ANN.

    The same stack — embeddings, vector search, reranking — appears in every modern recommender.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Part 1: Content-Based Filtering — Start Here

    **Idea**: build a profile of what the user likes based on *features* of items they've
    interacted with, then find new items with similar features.

    For jobs: if a user applied to jobs with `{Python, ML, San Antonio, remote-friendly}`,
    recommend jobs with similar features.

    **Algorithm**:
    1. Represent each item as a feature vector.
    2. Build a user preference vector — average of liked item vectors, or an explicit profile.
    3. Rank all unseen items by similarity to the user vector.

    This is exactly what Canopy's embedding similarity does: embed job descriptions → compare to
    user profile embedding → rank by cosine similarity. Content-based filtering, end to end.
    """)
    return


@app.cell
def content_based_impl():
    import numpy as _np
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity

    _rng = _np.random.default_rng(42)

    _skill_pools = [
        ["Python", "ML", "Polars", "PyTorch"],
        ["Python", "SQL", "Spark", "dbt"],
        ["Java", "Scala", "Kafka", "Flink"],
        ["Python", "FastAPI", "Postgres", "Docker"],
        ["React", "TypeScript", "GraphQL", "Node"],
    ]
    _locations = ["San Antonio", "Austin", "Remote", "New York", "Chicago"]
    _levels = ["Junior", "Mid", "Senior", "Staff"]
    _industries = ["FinTech", "EdTech", "HealthTech", "AdTech", "HRTech"]

    jobs = []
    for _i in range(50):
        _pool = _skill_pools[_i % len(_skill_pools)]
        _skills = list(_rng.choice(_pool, size=_rng.integers(2, len(_pool) + 1), replace=False))
        jobs.append({
            "id": _i,
            "title": f"Job {_i}",
            "skills": " ".join(_skills),
            "location": _locations[_i % len(_locations)],
            "level": _levels[_i % len(_levels)],
            "industry": _industries[_i % len(_industries)],
            "salary_k": int(_rng.integers(80, 220)),
            "remote": bool(_i % 3 == 0),
        })

    # User profile as text — same space as job representations
    user_text = "Python ML Polars PyTorch San Antonio Senior HRTech"

    _job_texts = [
        f"{j['skills']} {j['location']} {j['level']} {j['industry']}"
        for j in jobs
    ]

    _vectorizer = _TfidfVectorizer()
    _job_vectors = _vectorizer.fit_transform(_job_texts)
    _user_vector = _vectorizer.transform([user_text])

    _similarities = _cosine_similarity(_user_vector, _job_vectors)[0]
    _top5_idx = _np.argsort(_similarities)[::-1][:5]

    print("Content-Based Top 5 Job Recommendations")
    print("=" * 50)
    for _rank, _idx in enumerate(_top5_idx, 1):
        _j = jobs[_idx]
        print(f"{_rank}. Job {_idx:02d} | {_j['skills']} | {_j['location']} | "
              f"{_j['level']} | score={_similarities[_idx]:.3f}")

    job_vectors_dense = _job_vectors.toarray()
    content_similarities = _similarities
    return content_similarities, job_vectors_dense, jobs, user_text


@app.cell
def _(mo):
    mo.md("""
    ### Content-Based: Strengths and Weaknesses

    **Strengths**:
    - No cold-start for items — features are known from day one, even for brand-new job postings.
    - Transparent — you can explain *why* each job was recommended ("matched Python, ML, San Antonio").
    - Works with a single user — no community of interactions required.
    - Handles niche items that no one else has seen yet.

    **Weaknesses**:
    - **Filter bubble** — only recommends things similar to past likes. A Python ML engineer
      never discovers that they might love a data engineering role. Discovery is limited by what
      you've already seen.
    - Requires well-structured item features. For jobs, features are explicit and clean. For music
      or movies, features are harder to define.
    - Can't capture "people who liked X also liked Y" without knowing *why* they're related.

    *Canopy is currently a content-based recommender. It works well for job search because job
    features are explicit and structured. The limitation: every user gets similar recommendations
    regardless of what other users with the same profile actually applied to.*
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Part 2: Collaborative Filtering — The Power of the Crowd

    **Idea**: leverage *other users'* behavior to make predictions. If user A and user B applied
    to similar jobs, they'll likely apply to similar future jobs.

    You don't need to know WHY people like things. Just the patterns of co-occurrence are enough.

    **Two sub-approaches**:
    1. **User-based CF**: find users similar to me → recommend what they liked that I haven't seen.
    2. **Item-based CF**: find items similar to what I liked → recommend those items.
    """)
    return


@app.cell
def user_based_cf(jobs):
    import numpy as _np
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity

    _rng = _np.random.default_rng(7)
    _n_users = 20
    _n_jobs = len(jobs)  # 50

    # 1 = applied, 0 = not applied / unknown (not "disliked")
    interaction_matrix = (_rng.random((_n_users, _n_jobs)) < 0.15).astype(float)
    # Ensure target user has some history
    interaction_matrix[0, [2, 5, 12, 19, 31]] = 1.0

    _target_user = 0

    # Cosine similarity between target user and all others
    _user_sims = _cosine_similarity(
        interaction_matrix[_target_user:_target_user + 1], interaction_matrix
    )[0]

    # Top 5 similar users (excluding self)
    _similar_users = _np.argsort(_user_sims)[::-1][1:6]

    _target_applied = set(_np.where(interaction_matrix[_target_user] == 1)[0])
    _rec_counts: dict[int, int] = {}
    for _u in _similar_users:
        _their_jobs = set(_np.where(interaction_matrix[_u] == 1)[0])
        _new_jobs = _their_jobs - _target_applied
        for _jid in _new_jobs:
            _rec_counts[_jid] = _rec_counts.get(_jid, 0) + 1

    # Sort by how many similar users applied
    _user_cf_recs = sorted(_rec_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    print("User-Based CF — Top 5 Recommendations for User 0")
    print("=" * 55)
    print(f"User 0 applied to jobs: {sorted(_target_applied)}")
    print(f"Most similar users (indices): {list(_similar_users)}")
    print()
    for _rank, (_jid, _count) in enumerate(_user_cf_recs, 1):
        _j = jobs[_jid]
        print(f"{_rank}. Job {_jid:02d} | {_j['skills']} | {_j['location']} | "
              f"voted by {_count} similar users")

    return interaction_matrix


@app.cell
def _(mo):
    mo.md("""
    ### Item-Based Collaborative Filtering

    Instead of finding similar *users*, find similar *items* based on their co-application pattern.

    **Key insight**: if jobs A and B are consistently applied to by the same users, they're similar —
    regardless of whether they share explicit features. Users might not know why; the pattern is enough.

    **Why item-based over user-based?**
    - Items change less frequently than users. A job posting's co-application pattern stabilizes
      over days; a user's preferences can shift by the hour.
    - Item-item similarity matrix can be precomputed offline. O(items²) precompute vs O(users²) at
      query time.
    - Amazon's original collaborative filter was item-based for exactly this reason.
    """)
    return


@app.cell
def item_based_cf(interaction_matrix, jobs):
    import numpy as _np
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity

    # Item-item similarity: transpose so items are rows, compute row-wise similarity
    _item_sim = _cosine_similarity(interaction_matrix.T)  # shape (50, 50)

    _target_user = 0
    _applied_jobs = _np.where(interaction_matrix[_target_user] == 1)[0]

    # Aggregate similarity scores across all jobs the user applied to
    _scores = _np.zeros(len(jobs))
    for _jid in _applied_jobs:
        _scores += _item_sim[_jid]

    # Zero out already-applied jobs
    _scores[_applied_jobs] = -1.0
    _item_cf_recs = _np.argsort(_scores)[::-1][:5]

    print("Item-Based CF — Top 5 Recommendations for User 0")
    print("=" * 55)
    print(f"Based on applied jobs: {list(_applied_jobs)}")
    print()
    for _rank, _idx in enumerate(_item_cf_recs, 1):
        _j = jobs[_idx]
        print(f"{_rank}. Job {_idx:02d} | {_j['skills']} | {_j['location']} | "
              f"agg_score={_scores[_idx]:.3f}")

    return


@app.cell
def _(mo):
    mo.md("""
    ### Collaborative Filtering Trade-offs

    **Strengths**:
    - Discovers unexpected connections — users who applied to ML jobs also applied to data
      engineering roles. No feature engineering required to surface this.
    - Captures implicit patterns: cultural fit, prestige signals, hidden preferences.
    - Scales to any item type — the algorithm is identical for jobs, movies, or products.

    **Weaknesses**:

    | Problem | Description | Fix |
    |---------|-------------|-----|
    | **Cold start (users)** | New user has no interactions → no similarity signal | Content-based fallback, onboarding quiz |
    | **Cold start (items)** | New job posting has no applications → invisible to CF | Use content features until data accumulates |
    | **Sparsity** | 99.9%+ of entries are unknown. Direct cosine similarity is noisy on sparse vectors | Matrix factorization (see Part 3) |
    | **Scalability** | All-pairs user or item similarity is O(n²) | Approximate methods, LSH, ANN retrieval |
    | **Popularity bias** | Popular items get recommended more → rich get richer | Exposure correction, inverse popularity weighting |

    The sparsity problem is why raw CF rarely works in production. Matrix factorization solves
    it by learning dense low-dimensional representations.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Part 3: Matrix Factorization — Making CF Scale

    **The problem**: the interaction matrix R is huge (millions of users × millions of items)
    and 99.9%+ sparse. Direct similarity on sparse vectors is noisy. We need a way to generalize.

    **The idea**: decompose R (m × n) into two smaller matrices:

    $$R \\approx U \\cdot V^T$$

    - **U** — shape (m, k): one row per user. Each row is the user's embedding in a k-dimensional latent space.
    - **V** — shape (n, k): one row per item. Each row is the item's embedding.
    - **k** is small (50–200), much smaller than m or n.

    **Prediction**: user i's score for item j = `dot(U[i], V[j])`.

    The k dimensions are **latent factors** — automatically discovered hidden features like
    "prefers remote work," "likes ML roles," "early-career-friendly." These are the *same* learned
    embeddings from the dimensionality reduction notebook — SVD and PCA are special cases of MF.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The Math

    **Objective**: minimize reconstruction error on *observed* entries only. Zeros mean "not seen,"
    not "disliked" — so you must not penalize them.

    $$\mathcal{L} = \sum_{(i,j) \in \text{observed}} \left(R_{ij} - U_i \cdot V_j\right)^2 + \lambda \left(\|U\|_F^2 + \|V\|_F^2\right)$$

    The regularization term (λ) prevents the embeddings from overfitting to the sparse
    observations — without it, embeddings collapse to memorize the training set.

    **Two optimization approaches**:

    - **ALS (Alternating Least Squares)**: fix U, solve for V in closed form. Fix V, solve for U.
      Alternate. Each step is a linear regression — parallelizable across users/items. Used in
      production (Spark MLlib).
    - **SGD**: standard gradient descent on the loss. Simpler to implement, flexible, the standard
      choice when building from scratch.

    **Typical k**: 50–200 latent factors. More k = more expressive but more parameters and more
    risk of overfitting on sparse data.
    """)
    return


@app.cell
def matrix_factorization_impl(interaction_matrix):
    import numpy as _np

    def _mf_sgd(R, k=10, lr=0.01, reg=0.02, epochs=150):
        m, n = R.shape
        _rng = _np.random.default_rng(42)
        U = _rng.normal(0, 0.1, (m, k))
        V = _rng.normal(0, 0.1, (n, k))
        observed = list(zip(*_np.where(R > 0)))
        losses = []
        for epoch in range(epochs):
            _rng.shuffle(observed)
            total_loss = 0.0
            for i, j in observed:
                error = R[i, j] - U[i] @ V[j]
                Vj = V[j].copy()
                U[i] += lr * (error * Vj - reg * U[i])
                V[j] += lr * (error * U[i] - reg * V[j])
                total_loss += error ** 2
            losses.append(total_loss / len(observed))
        return U, V, losses

    U, V, losses = _mf_sgd(interaction_matrix)
    R_hat = U @ V.T

    # Reconstruction sanity check
    _observed_idx = list(zip(*_np.where(interaction_matrix > 0)))
    _actuals = [interaction_matrix[i, j] for i, j in _observed_idx[:10]]
    _preds = [R_hat[i, j] for i, j in _observed_idx[:10]]
    print("Reconstruction check — observed entries (first 10):")
    print(f"  Actual: {[round(v, 2) for v in _actuals]}")
    print(f"  Pred:   {[round(v, 2) for v in _preds]}")

    # Top 5 recommendations for user 0
    _target = 0
    _applied = set(_np.where(interaction_matrix[_target] == 1)[0])
    _scores = R_hat[_target].copy()
    for _jid in _applied:
        _scores[_jid] = -999.0
    _mf_top5 = _np.argsort(_scores)[::-1][:5]
    print("\nMF SGD — Top 5 Recommendations for User 0:")
    for _rank, _idx in enumerate(_mf_top5, 1):
        print(f"  {_rank}. Job {_idx:02d}  predicted_score={R_hat[_target, _idx]:.3f}")

    return U, V, R_hat, losses


@app.cell
def mf_loss_viz(losses):
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _fig, _ax = _plt.subplots(figsize=(9, 4))
    _ax.plot(losses, color="#3498DB", lw=2)
    _ax.set_xlabel("Epoch", fontsize=12)
    _ax.set_ylabel("Mean Squared Error", fontsize=12)
    _ax.set_title("Matrix Factorization — SGD Loss Convergence", fontsize=13, fontweight="bold")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    return _fig


@app.cell
def embedding_viz(U, V, interaction_matrix):
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.decomposition import PCA as _PCA

    # Stack user and item embeddings for joint PCA
    _all_embs = _np.vstack([U, V])  # (20 users + 50 items, 10)
    _pca = _PCA(n_components=2)
    _coords = _pca.fit_transform(_all_embs)

    _user_coords = _coords[:len(U)]
    _item_coords = _coords[len(U):]

    _target = 0
    _applied_jobs = set(_np.where(interaction_matrix[_target] == 1)[0])

    _fig, _ax = _plt.subplots(figsize=(11, 7))

    # Plot items
    _ax.scatter(
        _item_coords[:, 0], _item_coords[:, 1],
        c="#BDC3C7", s=55, alpha=0.6, zorder=2, label="Jobs (not applied)"
    )
    # Highlight applied jobs
    _applied_arr = list(_applied_jobs)
    _ax.scatter(
        _item_coords[_applied_arr, 0], _item_coords[_applied_arr, 1],
        c="#2ECC71", s=120, zorder=4, label="Jobs User 0 applied to", edgecolors="white", lw=1.5
    )
    # Label applied jobs
    for _jid in _applied_arr:
        _ax.annotate(f"J{_jid}", (_item_coords[_jid, 0], _item_coords[_jid, 1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7, color="#27AE60")

    # Plot users
    _ax.scatter(
        _user_coords[1:, 0], _user_coords[1:, 1],
        c="#85C1E9", s=90, marker="D", alpha=0.7, zorder=3, label="Other users"
    )
    # Highlight target user
    _ax.scatter(
        _user_coords[0, 0], _user_coords[0, 1],
        c="#E74C3C", s=200, marker="*", zorder=5, label="User 0 (target)"
    )
    _ax.annotate("User 0", (_user_coords[0, 0], _user_coords[0, 1]),
                 textcoords="offset points", xytext=(8, 6), fontsize=10, color="#E74C3C",
                 fontweight="bold")

    _var = _pca.explained_variance_ratio_
    _ax.set_xlabel(f"PC1 ({_var[0]*100:.1f}% variance)", fontsize=11)
    _ax.set_ylabel(f"PC2 ({_var[1]*100:.1f}% variance)", fontsize=11)
    _ax.set_title(
        "Learned Embedding Space — Users and Jobs after Matrix Factorization\n"
        "(PCA projection to 2D; users close to jobs they applied to)",
        fontsize=12, fontweight="bold"
    )
    _ax.legend(fontsize=10, loc="upper right")
    _ax.grid(True, alpha=0.25)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    The latent space **is** the recommendation space. Users positioned close to items in this
    2D projection are predicted to have high affinity — the dot product `U[i] · V[j]` is large
    when they point in the same direction.

    This is the same dot-product similarity that drives vector search in Canopy and the attention
    mechanism in transformers. Matrix factorization is essentially learning task-specific embeddings
    for users and items jointly — the same intuition behind learned embeddings from the
    dimensionality reduction notebook.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Part 4: Deep Learning for Recommendations

    ### Why Go Deep?

    Matrix factorization: `score(i, j) = dot(U[i], V[j])` — a weighted sum. Captures linear
    interactions only. User embedding and item embedding interact through a single dot product.

    **Deep learning**: replace the dot product with a neural network. The model can learn
    *nonlinear* interactions — "a senior user who likes Python AND ML together (not just each
    separately) is likely to apply to this job."

    **Neural Collaborative Filtering (NCF)**:
    ```
    user_id → Embedding → ┐
                           ├── Concatenate → MLP → Sigmoid → score
    item_id → Embedding → ┘
    ```

    The MLP can capture cross-feature interactions that a dot product cannot.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### The Two-Tower Architecture (Production Standard)

    ```
    [User Features]               [Item Features]
    (history, demographics)        (title, skills, location)
         ↓                              ↓
    [User Encoder Tower]          [Item Encoder Tower]
    (MLP or transformer)          (MLP or transformer)
         ↓                              ↓
    [User Embedding (128-dim)]   [Item Embedding (128-dim)]
         ↓                              ↓
         └──────── dot product ─────────┘
                      ↓
               [Relevance Score]
    ```

    **Why two separate towers?**

    - **User embedding** can be precomputed once per session and cached. The user tower runs
      once; the result is reused for all candidate scoring.
    - **Item embeddings** can be precomputed offline for all items and indexed in a vector
      database (HNSW) for fast approximate nearest-neighbor retrieval.
    - At serving time: compute user embedding once → ANN search against all item embeddings →
      retrieve top K candidates in milliseconds — regardless of whether there are 1M or 100M items.

    This is the bridge between recommender systems and vector search. **The vector database
    (sqlite-vec, FAISS, Pinecone) IS the serving layer** for the candidate generation stage of a
    deep recommender. The retrieval stack from the vector databases notebook is the production
    implementation of this.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Matrix Factorization vs Deep Learning

    | Dimension | Matrix Factorization | Deep Learning (Two-Tower, NCF) |
    |-----------|---------------------|-------------------------------|
    | Data size needed | Works with less data | Needs lots of interaction data |
    | Feature types | IDs only (collaborative signal) | Can incorporate any features |
    | Interaction modeling | Linear (dot product) | Nonlinear (MLP layers) |
    | Training speed | Fast (ALS, SVD) | Slower (GPU, backprop) |
    | Cold start | Bad — no interactions = no embedding | Better — can use item/user features |
    | Interpretability | Latent factors somewhat interpretable | Black box |
    | Deployment complexity | Simple to deploy | Needs ML infra, model serving |
    | When to pick | < 10M interactions, no rich features | Rich features, lots of data, nonlinear patterns |

    **Practical rule**: start with matrix factorization. When you have rich side features
    (job skills, user history) and enough data, graduate to a two-tower deep model. The two-tower
    architecture also composes naturally with vector search infrastructure you already have.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Part 5: Hybrid Systems — The Production Reality

    No single approach works everywhere. Every production system combines multiple signals.

    **Combination strategies**:

    | Strategy | Mechanism | When to use |
    |----------|-----------|-------------|
    | **Weighted hybrid** | `score = α·content + β·CF + γ·constraints` | When all signals are always available |
    | **Switching** | Content-based for new items, CF for warm items | When signal availability depends on item age |
    | **Cascade** | Content filter → CF rerank → rule filter | When precision matters more than recall |
    | **Feature stacking** | Feed CF predictions as features to a meta-learner | When you want a single model to learn the blend |

    **Cold-start mapping**:
    - New job posting (no applications yet) → content-based features
    - New user (no applications yet) → popularity baseline + onboarding quiz
    - Warm job, warm user → collaborative + hybrid
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Two-Stage Production Architecture

    ```
    [All Items (millions)]
        ↓
    [Candidate Generation]  — fast, recall-focused (~milliseconds)
        ├── Content-based ANN:    embed query → HNSW search → 500 candidates
        ├── Collaborative CF:     user's CF neighbors' items  → 500 candidates
        └── Popularity/trending:  top items globally          → 100 candidates
        ↓
    [Dedup + merge]  →  ~800 unique candidates
        ↓
    [Ranking Model]  — slower, precision-focused (~50ms)
        Features: user features + item features + interaction features + CF scores
        Model: XGBoost or deep ranker (cross-features, nonlinear interactions)
        Output: scored + ranked list
        ↓
    [Top 20]
        ↓
    [Business Rules / Post-Filtering]
        - Diversity: not all from same company
        - Freshness boost: newer postings ranked higher
        - Already-seen filter
        - Hard constraints: location, visa, salary floor
        ↓
    [Top 10 shown to user]
    ```

    This is the **exact same architecture** from the RAG system design and fraud detection
    notebooks:

    - **Candidate generation** = retrieval (fast, high recall)
    - **Ranking model** = reranking (slower, high precision)
    - **Business rules** = guardrails

    Candidate generation → ranking → guardrails. The same three-layer pattern appears in
    search, recommendations, and RAG. It's not a coincidence — it's the fundamental answer
    to the problem of "I have millions of options and need the best 10 in under 100ms."
    """)
    return


@app.cell
def hybrid_scoring(content_similarities, interaction_matrix, R_hat, jobs):
    import numpy as _np

    _target = 0
    _n_jobs = len(jobs)

    # Normalize each signal to [0, 1]
    def _norm(v):
        mn, mx = v.min(), v.max()
        return (v - mn) / (mx - mn + 1e-9)

    _content_score = _norm(content_similarities)
    _cf_score = _norm(R_hat[_target])

    # Freshness proxy: newer job IDs (higher idx) are slightly newer
    _freshness = _norm(_np.arange(_n_jobs).astype(float))

    # Weighted hybrid
    _alpha, _beta, _gamma = 0.5, 0.35, 0.15
    _hybrid_score = _alpha * _content_score + _beta * _cf_score + _gamma * _freshness

    # Exclude already-applied jobs
    _applied = set(_np.where(interaction_matrix[_target] == 1)[0])
    for _jid in _applied:
        _hybrid_score[_jid] = -1.0

    _top5 = _np.argsort(_hybrid_score)[::-1][:5]

    print("Hybrid Scoring — Top 5 Recommendations for User 0")
    print(f"Weights: content={_alpha}, CF={_beta}, freshness={_gamma}")
    print("=" * 65)
    for _rank, _idx in enumerate(_top5, 1):
        _j = jobs[_idx]
        print(f"{_rank}. Job {_idx:02d} | {_j['skills'][:30]:30s} | "
              f"content={_content_score[_idx]:.2f}  CF={_cf_score[_idx]:.2f}  "
              f"hybrid={_hybrid_score[_idx]:.3f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Part 6: Evaluation for Recommender Systems

    ### Offline Metrics

    | Metric | Formula | What it measures |
    |--------|---------|-----------------|
    | **Precision@K** | `|relevant ∩ recommended@K| / K` | Of top K, how many were actually relevant? |
    | **Recall@K** | `|relevant ∩ recommended@K| / |relevant|` | Of all relevant, how many did we surface? |
    | **NDCG@K** | Discounted sum of graded relevance | Precision@K + credit for *rank* of relevant items |
    | **MAP** | Mean average precision across users | Average precision at each relevant position |
    | **Coverage** | `|unique recommended items| / |all items|` | Diversity of catalog exposure |
    | **Diversity** | Intra-list dissimilarity | Are recommended items varied? |

    **NDCG is the standard ranking metric** because it gives more credit for relevant items
    ranked higher. Being relevant at rank 1 is worth more than being relevant at rank 10.

    $$NDCG@K = \\frac{DCG@K}{IDCG@K}, \\quad DCG@K = \\sum_{i=1}^{K} \\frac{\\text{rel}_i}{\\log_2(i+1)}$$

    ### Offline vs Online Evaluation Gap

    Offline metrics can be misleading:
    - You only evaluate against **observed** interactions. A user might love a job they never saw.
    - **Position bias**: users only click on results shown at the top — your label data is biased
      by your current ranker.
    - **Feedback loops**: recommending popular items generates more interactions for popular items,
      making them look even better offline. The rich get richer.

    **Online evaluation (A/B testing)** is the true measure:
    - CTR, application rate, session length, user retention
    - Run for 2+ weeks to capture weekly patterns (users behave differently on Mondays vs Fridays)

    *Offline eval narrows the candidates. Online eval makes the final decision. Same layered
    eval philosophy from the LLM evaluation notebook: automatic metrics guide, humans decide.*
    """)
    return


@app.cell
def eval_metrics_impl(interaction_matrix, R_hat):
    import numpy as _np

    def precision_at_k(recommended, relevant, k):
        rec_k = set(recommended[:k])
        return len(rec_k & relevant) / k

    def recall_at_k(recommended, relevant, k):
        rec_k = set(recommended[:k])
        return len(rec_k & relevant) / max(len(relevant), 1)

    def ndcg_at_k(recommended, relevant, k):
        dcg = sum(
            1.0 / _np.log2(rank + 2)
            for rank, item in enumerate(recommended[:k])
            if item in relevant
        )
        ideal = sum(
            1.0 / _np.log2(rank + 2)
            for rank in range(min(len(relevant), k))
        )
        return dcg / max(ideal, 1e-9)

    _results = []
    for _user in range(len(interaction_matrix)):
        _applied = set(_np.where(interaction_matrix[_user] == 1)[0])
        if len(_applied) < 2:
            continue
        # Leave one out: hold back half the applied jobs as ground truth
        _applied_list = list(_applied)
        _train_applied = set(_applied_list[:len(_applied_list)//2])
        _test_relevant = set(_applied_list[len(_applied_list)//2:])

        _scores = R_hat[_user].copy()
        for _jid in _train_applied:
            _scores[_jid] = -999.0
        _ranked = list(_np.argsort(_scores)[::-1])

        _results.append({
            "p@5": precision_at_k(_ranked, _test_relevant, 5),
            "r@5": recall_at_k(_ranked, _test_relevant, 5),
            "ndcg@5": ndcg_at_k(_ranked, _test_relevant, 5),
        })

    if _results:
        print("Offline Evaluation — Leave-One-Out on Synthetic Data")
        print("=" * 45)
        print(f"  Precision@5 : {_np.mean([r['p@5']   for r in _results]):.3f}")
        print(f"  Recall@5    : {_np.mean([r['r@5']   for r in _results]):.3f}")
        print(f"  NDCG@5      : {_np.mean([r['ndcg@5'] for r in _results]):.3f}")
        print(f"  (n={len(_results)} users evaluated)")
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Connection to My Projects

    **Canopy IS a content-based recommender**: embed job descriptions with sentence-transformers,
    compute cosine similarity to the user's profile embedding, rank by similarity. This is exactly
    Part 1 of this notebook — pure content-based, candidate generation stage only.

    **What I'd add for a production-quality Canopy**:

    1. **Collaborative signal**: track which jobs users with similar profiles applied to. As
       multiple users use Canopy, the interaction matrix fills in. Feed this into item-based CF
       or matrix factorization to surface jobs the content signal would miss.

    2. **Two-stage pipeline**: fast embedding retrieval via sqlite-vec (candidate generation, ~500
       candidates) → XGBoost reranker with richer features — salary fit, posting age, company
       size, skills overlap count, location distance (ranking stage).

    3. **Hybrid scoring**: `content_similarity + freshness_boost + salary_fit + cf_signal` →
       weighted combination tuned by A/B test on application rate.

    **Architecture connections across this learning log**:

    | Notebook | Connection to Recommenders |
    |----------|---------------------------|
    | Dimensionality Reduction | Matrix factorization learns user/item embeddings — same intuition as SVD/PCA |
    | Vector Databases (HNSW) | The serving layer for item embedding retrieval in two-tower architectures |
    | RAG System Design | Candidate generation = retrieval; reranking = ranking model; guardrails = business rules |
    | Fraud Detection | Same candidate → rank → filter three-layer pattern |
    | Evaluation Metrics | NDCG, Precision@K — the same metrics used for recommender offline eval |

    *Everything connects: embeddings → vector search → candidate generation → reranking →
    evaluation. It's the same stack viewed from different angles.*
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Flashcard Summary

    **Content-based vs collaborative filtering?**
    Content-based uses item features (skill matching, location). CF uses interaction patterns
    (users who applied to X also applied to Y). Content handles cold start. CF captures
    unexpected patterns. Production systems use both.

    ---

    **What is matrix factorization?**
    Decompose the user-item matrix R into user embeddings U and item embeddings V such that
    R ≈ U × Vᵀ. Predictions = dot product. Latent factors are automatically discovered hidden
    preferences. Train by SGD or ALS on *observed* entries only (zeros = unknown, not disliked).

    ---

    **Cold start problem?**
    New user: no interaction history → no collaborative signal → fallback to content-based or
    popularity. New item: no interactions → no CF embedding → use content features until data
    accumulates. Fix strategies: onboarding quiz, content-based bootstrap, hybrid switching.

    ---

    **Two-tower architecture?**
    Separate user encoder and item encoder produce fixed-size embeddings. Score = dot product.
    User embedding computed once and cached. Item embeddings pre-indexed in a vector DB (HNSW)
    for sub-millisecond ANN retrieval at scale. This is the serving architecture for deep recs.

    ---

    **Why two-stage (candidate generation + ranking)?**
    Can't run an expensive ranking model (XGBoost, neural) on millions of items. Retrieve
    500–1000 cheap candidates first with fast ANN retrieval, then rank with the expensive model.
    Candidate generation maximizes recall. Ranking maximizes precision on the shortlist.

    ---

    **NDCG vs Precision@K?**
    Precision@K measures how many of the top K were relevant — but treats rank 1 and rank K
    equally. NDCG gives *more credit for relevant items ranked higher*, weighted by 1/log₂(rank+1).
    NDCG better reflects user experience — the top result matters most.

    ---

    **Matrix factorization vs deep learning for recs?**
    MF: simpler, works with less data, fast training, linear interactions. Deep learning: captures
    nonlinear cross-feature interactions, incorporates rich side features (job skills, user history),
    needs more data and GPU infra. Start with MF; move to deep when you have the data and features.

    ---

    **Filter bubble problem?**
    Content-based keeps recommending items similar to past interactions — the user never discovers
    new interests outside their known preferences. Fix: inject exploration (epsilon-greedy random
    items), add diversity constraints to the ranking stage, serendipity scoring.

    ---

    **How does recommendation connect to RAG?**
    Same architecture with different names: candidate generation = retrieval. Ranking model =
    reranker. Business rules = guardrails. Two-tower encoder = bi-encoder. The retrieval stack
    (vector DB + ANN) is the serving layer in both. Building a recommender and building a RAG
    pipeline are the same problem at different angles.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Interview Talking Points

    **"Design a recommendation system"**
    Use the two-stage architecture: candidate generation (content-based ANN + collaborative CF
    + popularity baseline) → ranking model (XGBoost with cross-features: user × item interactions,
    freshness, salary fit) → business rules (diversity, already-seen filter, hard constraints).
    Start with content-based, add collaborative signal as interaction data accumulates.

    ---

    **"How does Canopy relate to recommendation systems?"**
    "Canopy is a content-based recommender. I embed job descriptions and user profiles with
    sentence-transformers and rank by cosine similarity — that's exactly the candidate generation
    stage. For production, I'd layer on a ranking model with richer features and a collaborative
    signal from user interaction patterns once multiple users are active."

    ---

    **"Walk me through matrix factorization from scratch"**
    Initialize random U (users × k) and V (items × k). Loop over observed interactions. For each
    (user i, item j): compute error = actual − dot(U[i], V[j]). SGD update U[i] toward lower error,
    regularize to prevent overfitting. Same for V[j]. After convergence, predict any entry as
    dot(U[i], V[j]). The k latent factors are automatically learned representations of hidden
    user preferences and item characteristics. (See the SGD implementation in this notebook.)

    ---

    **"What connects recommendation, RAG, and search?"**
    "They're the same architecture with different names. Two-tower encoding → ANN retrieval →
    reranking → business rules. Whether I'm recommending jobs, retrieving documents for RAG, or
    ranking search results, the pattern is identical: fast candidate generation for recall,
    expensive reranking for precision, rule-based filtering for business constraints. My vector
    database, embedding, and evaluation notebooks all feed into this — it's the same stack viewed
    from different angles."
    """)
    return


if __name__ == "__main__":
    app.run()
