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
    # Vector Databases Deep Dive — HNSW, Hybrid Search, Indexing Strategies

    | Field   | Value                                                               |
    |---------|---------------------------------------------------------------------|
    | Date    | 2026-04-20                                                          |
    | Track   | AI Engineering                                                      |
    | Time    | 60 min                                                              |
    | Topic   | HNSW · Hybrid Search · Indexing Strategies · Tied to ChromaDB Work |
    """)
    return


@app.cell
def why_vector_dbs(mo):
    mo.md("""
    ## Why Vector Databases Exist

    **The problem:** you have N vectors of dimension d (embeddings). Given a query vector, find the K most similar vectors.

    **Brute force:** compute similarity between the query and ALL N vectors, sort, return top K.
    Cost: O(N × d). Fine for N < 100K. Unacceptable for N > 1M.

    ```
    10K  vectors × 384 dims  →   ~10ms   ✓  fine  (Canopy lives here)
    1M   vectors × 384 dims  →    ~1s    ✗  too slow for interactive use
    100M vectors × 384 dims  → minutes   ✗✗ unacceptable
    ```

    **Vector databases** solve this with **Approximate Nearest Neighbor (ANN)** algorithms —
    trading a tiny accuracy loss for massive speed gains. Typical: 10–100× faster at 95–99% recall.

    ### The landscape

    | System     | Algorithm              | Notes                                       |
    |------------|------------------------|---------------------------------------------|
    | ChromaDB   | HNSW (via hnswlib)     | Embedded, easy to run locally               |
    | Qdrant     | HNSW                   | Production-grade, built-in hybrid search    |
    | Weaviate   | HNSW                   | GraphQL interface, hybrid out of the box    |
    | Pinecone   | Proprietary (HNSW-ish) | Fully managed, expensive at scale           |
    | pgvector   | IVF or HNSW            | Postgres extension — fits existing DB stack |
    | Milvus     | IVF-PQ, HNSW, others   | Distributed, billion-scale                  |

    Understanding the algorithms means understanding **all of them** — the interfaces differ, the math is the same.

    > "In interviews, saying 'I used ChromaDB' is table stakes. Explaining WHY you chose it
    > and what HNSW does under the hood is senior-level."
    """)
    return


@app.cell
def shared_imports():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import time
    return np, plt, time


@app.cell
def part1_concept(mo):
    mo.md("""
    ---
    ## Part 1: Distance Metrics — What Does "Similar" Mean?

    Before any indexing algorithm, you need a similarity function.

    | Metric                | Formula                | Range    | When to use                                                        |
    |-----------------------|------------------------|----------|--------------------------------------------------------------------|
    | **Cosine similarity** | (A·B) / (‖A‖ ‖B‖)     | [−1, 1]  | Text embeddings — direction matters, not magnitude                 |
    | **Euclidean (L2)**    | ‖A − B‖₂              | [0, ∞)   | Image features — scale is meaningful                              |
    | **Dot product**       | A·B                    | (−∞, ∞)  | L2-normalized vectors — same result as cosine, but faster (no √)  |

    **Key insight:** if vectors are L2-normalized (‖v‖ = 1), cosine similarity = dot product.
    Most sentence-transformers normalize their output. Check normalization first, then use dot product for speed.

    > **My Canopy project:** cosine similarity for job embeddings with all-MiniLM-L6-v2.
    > Sentence-transformer outputs aren't perfectly L2-normalized, and I care about semantic
    > *direction* (what the job IS about), not embedding magnitude. Cosine is the right call.
    """)
    return


@app.cell
def part1_code(np):
    # ── Distance metrics from scratch ─────────────────────────────────────────────
    def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def euclidean_distance(a: "np.ndarray", b: "np.ndarray") -> float:
        return float(np.linalg.norm(a - b))

    def dot_product(a: "np.ndarray", b: "np.ndarray") -> float:
        return float(np.dot(a, b))

    rng = np.random.default_rng(42)
    a = rng.standard_normal(8).astype(np.float32)
    pairs = [
        ("similar",    a,  a + rng.standard_normal(8).astype(np.float32) * 0.1),
        ("orthogonal", a,  rng.standard_normal(8).astype(np.float32)),
        ("opposite",   a, -a + rng.standard_normal(8).astype(np.float32) * 0.1),
    ]

    print(f"{'Pair':<12} {'Cosine':>10} {'Euclidean':>12} {'Dot Product':>14}")
    print("-" * 52)
    for label, x, y in pairs:
        print(f"{label:<12} {cosine_similarity(x, y):>10.4f} {euclidean_distance(x, y):>12.4f} {dot_product(x, y):>14.4f}")

    # Show cosine == dot product when L2-normalized
    a_n = a / np.linalg.norm(a)
    b_n = pairs[0][2] / np.linalg.norm(pairs[0][2])
    print(f"\n── L2-normalized (similar pair) ──")
    print(f"  cosine:      {cosine_similarity(a_n, b_n):.6f}")
    print(f"  dot product: {dot_product(a_n, b_n):.6f}")
    print("  → identical — use dot product when normalized (no sqrt overhead)")
    return


@app.cell
def part2_concept(mo):
    mo.md("""
    ---
    ## Part 2: Exact Search — The Baseline

    **Brute force KNN:** compute cosine similarity of query against every vector, sort, return top K.

    ```python
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)  # normalize once at build time
    sims   = X_norm @ q_norm                                 # N dot products via fast matmul
    top_k  = np.argsort(sims)[::-1][:K]                    # sort and slice
    ```

    Cost: O(N × d) per query — linear in dataset size. No training. Perfect recall. No tuning.

    > **My Canopy + QA projects:** brute force works fine at current scale (< 10K vectors/chunks).
    > The inflection point where ANN pays off is roughly N > 50K–100K. Below that: don't over-engineer.
    """)
    return


@app.cell
def part2_code(np, plt, time):
    # ── Brute force KNN ───────────────────────────────────────────────────────────
    DIM = 384   # all-MiniLM-L6-v2 (Canopy + QA project)
    K = 10
    rng2 = np.random.default_rng(7)

    def brute_force_knn(X: "np.ndarray", query: "np.ndarray", k: int = K) -> "np.ndarray":
        """Returns indices of top-k most similar vectors (cosine similarity)."""
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        q_norm = query / np.linalg.norm(query)
        return np.argsort(X_norm @ q_norm)[::-1][:k]

    query_bf = rng2.standard_normal(DIM).astype(np.float32)

    # Canopy context
    X_5k = rng2.standard_normal((5_000, DIM)).astype(np.float32)
    t0 = time.perf_counter()
    brute_force_knn(X_5k, query_bf)
    ms_5k = (time.perf_counter() - t0) * 1000
    print(f"Brute force @ 5K vectors ({DIM}d): {ms_5k:.1f}ms  ← Canopy is here")

    # Inflection point chart
    sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    times_ms = []
    for n in sizes:
        X_tmp = rng2.standard_normal((n, DIM)).astype(np.float32)
        t0 = time.perf_counter()
        brute_force_knn(X_tmp, query_bf)
        times_ms.append((time.perf_counter() - t0) * 1000)

    print(f"\n{'N':>10}  {'latency':>10}")
    for n, ms in zip(sizes, times_ms):
        flag = "  ← ANN territory" if n >= 100_000 else ""
        print(f"{n:>10,}  {ms:>8.1f}ms{flag}")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot([s / 1000 for s in sizes], times_ms, "o-", color="#2563eb", linewidth=2, markersize=6)
    ax1.axhline(y=100, color="red", linestyle="--", alpha=0.7, label="100ms threshold")
    ax1.set_xlabel("Dataset size (thousands of vectors)")
    ax1.set_ylabel("Query latency (ms)")
    ax1.set_title(f"Brute Force Query Latency vs Dataset Size ({DIM}d, float32)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    plt.savefig("/tmp/bf_latency.png", dpi=100)
    plt.close(fig1)
    print("\n→ Exceeds 100ms around N=100K. Below that: brute force is fine.")

    X_10k = rng2.standard_normal((10_000, DIM)).astype(np.float32)
    return X_10k, brute_force_knn, DIM, K


@app.cell
def part3_concept(mo):
    mo.md("""
    ---
    ## Part 3: IVF (Inverted File Index) — Clustering for Speed

    **Core idea:** don't search ALL vectors. Pre-cluster them into K groups.
    At query time, only search the clusters nearest to the query.

    ### Build
    1. Run K-means on all N vectors → K cluster centroids
    2. Record each vector's cluster assignment (the "inverted list")

    ### Query
    1. Distance from query to all K centroids: O(K × d) — fast
    2. Take the `nprobe` nearest centroids
    3. Brute-force only within those clusters' vectors
    4. Return top K

    ### Math
    | Step | Cost |
    |------|------|
    | Build | O(N × d × K × iterations) — one-time K-means |
    | Query | O(K × d) + O(nprobe × (N/K) × d) |

    K=100, nprobe=10 → search ~10% of data → ~10× speedup. Recall@10 typically 90–98%.

    ### `nprobe` — your query-time knob (no rebuild needed)

    | nprobe | Speed | Recall |
    |--------|-------|--------|
    | 1 | Fastest | ~80% |
    | 10 | 10× slower than nprobe=1 | ~95% |
    | K (all) | Same as brute force | 100% |
    """)
    return


@app.cell
def part3_code(np, time, X_10k, brute_force_knn, DIM, K):
    from sklearn.cluster import KMeans

    N_CLUSTERS = 20
    N_PROBE = 3
    rng3 = np.random.default_rng(99)
    query_ivf = rng3.standard_normal(DIM).astype(np.float32)

    # ── Build ─────────────────────────────────────────────────────────────────────
    print(f"Building IVF index: {len(X_10k):,} vectors → {N_CLUSTERS} clusters...")
    t0 = time.perf_counter()
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=5, max_iter=50)
    labels = km.fit_predict(X_10k)
    centroids = km.cluster_centers_.astype(np.float32)
    print(f"  Build time: {(time.perf_counter()-t0)*1000:.0f}ms  (one-time)")
    print(f"  Avg vectors per cluster: {len(X_10k)/N_CLUSTERS:.0f}")

    inverted_lists: dict[int, list[int]] = {i: [] for i in range(N_CLUSTERS)}
    for idx, cid in enumerate(labels):
        inverted_lists[int(cid)].append(idx)

    # ── Query ─────────────────────────────────────────────────────────────────────
    def ivf_search(query: "np.ndarray", n_probe: int = N_PROBE) -> "np.ndarray":
        c_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        q_norm = query / np.linalg.norm(query)
        nearest_clusters = np.argsort(c_norm @ q_norm)[::-1][:n_probe]
        candidates = [i for cid in nearest_clusters for i in inverted_lists[int(cid)]]
        if not candidates:
            return np.array([], dtype=np.int64)
        cv_norm = X_10k[candidates] / np.linalg.norm(X_10k[candidates], axis=1, keepdims=True)
        top_local = np.argsort(cv_norm @ q_norm)[::-1][:K]
        return np.array([candidates[i] for i in top_local])

    # ── Recall + latency ─────────────────────────────────────────────────────────
    true_top = set(brute_force_knn(X_10k, query_ivf, K))
    recall = len(true_top & set(ivf_search(query_ivf))) / K

    REPS = 50
    t0 = time.perf_counter()
    for _ in range(REPS):
        brute_force_knn(X_10k, query_ivf, K)
    bf_ms = (time.perf_counter() - t0) / REPS * 1000

    t0 = time.perf_counter()
    for _ in range(REPS):
        ivf_search(query_ivf)
    ivf_ms = (time.perf_counter() - t0) / REPS * 1000

    print(f"\n── Results @ N=10K, {N_CLUSTERS} clusters, nprobe={N_PROBE} ──")
    print(f"  Brute force:  {bf_ms:.2f}ms   recall@{K} = 100%")
    print(f"  IVF:          {ivf_ms:.2f}ms   recall@{K} = {recall*100:.0f}%")
    print(f"  Speedup:      {bf_ms/max(ivf_ms, 0.001):.1f}×  (searching {N_PROBE}/{N_CLUSTERS} = {N_PROBE/N_CLUSTERS*100:.0f}% of data)")

    return bf_ms, ivf_ms


@app.cell
def part4_why_hnsw(mo):
    mo.md("""
    ---
    ## Part 4: HNSW (Hierarchical Navigable Small World) — The Production Standard

    ### Why HNSW dominates

    | Property | HNSW | IVF |
    |----------|------|-----|
    | Training required | No — insert directly | Yes — K-means first |
    | Incremental inserts | Yes — no rebuild | No — centroid drift hurts recall |
    | Recall-speed balance | Best in class | Good with tuning |
    | Memory | Higher (stores graph edges) | Lower (just inverted lists) |
    | Used by | ChromaDB, Qdrant, Weaviate, pgvector | FAISS, Milvus (IVF-PQ) |

    **Bottom line:** HNSW is the default for 100K–10M vectors. IVF-PQ is for memory-constrained 10M+ scale.
    """)
    return


@app.cell
def part4_nsw_concept(mo):
    mo.md("""
    ### From Skip Lists → NSW → HNSW

    **Skip List (the intuition):**
    ```
    Layer 2 (sparse):  [1] ─────────────────────── [7] ──────── [14]
    Layer 1:           [1] ──── [3] ──── [5] ──── [7] ── [9] ── [14]
    Layer 0 (dense):   [1]-[2]-[3]-[4]-[5]-[6]-[7]-[8]-[9]-[10]-[11]-[12]-[13]-[14]
    ```
    Search starts at the top (sparse, long-range jumps) → drops to lower layers (dense, local refinement).
    O(log N) instead of O(N). Each layer is a fast coarse filter.

    **NSW (Navigable Small World):** the same idea in vector space, as a graph:
    - Each vector is a node. Edges connect similar vectors.
    - Search: start at entry node, greedily jump to the neighbor closest to the query.
    - Problem: gets trapped in local optima — a local cluster that's nearby but not globally nearest.

    **HNSW:** add the Skip List hierarchy to NSW:
    ```
    Layer 2 (very sparse):  few nodes, long-range edges  → coarse global navigation
    Layer 1:                ~N/M nodes, medium edges      → region refinement
    Layer 0 (all N nodes):  dense local edges             → precise neighbor search
    ```

    - **Insert:** new vector gets a max layer sampled from a geometric distribution.
      Most land at layer 0. Very few reach layer 2+. This keeps upper layers sparse.
    - **Search:** enter at top layer → greedy traverse → drop one layer at current best node
      → refine → bottom layer → return top K.
    - **Why it works:** long-range edges at top layers let you skip over local clusters — breaking
      the local optima problem of flat NSW. This is why HNSW gets 97–99% recall where NSW stalls at ~80%.
    """)
    return


@app.cell
def part4_hnsw_params(mo):
    mo.md("""
    ### HNSW Parameters — Your Tuning Knobs

    | Parameter | What it controls | Typical range |
    |-----------|-----------------|---------------|
    | **M** | Connections per node (graph density) | 16–64 |
    | **ef_construction** | Beam width during index build — candidates tracked per insert | 100–500 |
    | **ef_search** | Beam width during query — candidates tracked per search | 50–200 |

    **Memory estimate:**
    - Vectors: N × d × 4 bytes (float32)
    - Graph: N × M × ~5 layers × 8 bytes (pointer size)
    - 1M vectors, d=384, M=16: ~1.5GB vectors + ~640MB graph ≈ **~2.1GB RAM**

    **ef_search is the knob you tune without rebuilding.** Higher = more candidates tracked = better recall, slower queries.

    | Parameter | ↑ Higher | ↓ Lower |
    |-----------|---------|---------|
    | M | Better recall, more memory, slower build | Lower recall, less memory, faster build |
    | ef_construction | Better index quality, slower build | Lower quality, faster build |
    | ef_search | Better recall per query, slower queries | Lower recall, faster queries |

    > **ChromaDB:** `client.create_collection(name="jobs", metadata={"hnsw:M": 16, "hnsw:construction_ef": 200, "hnsw:search_ef": 100})`
    """)
    return


@app.cell
def part4_hnsw_code(np, plt, time, X_10k, brute_force_knn, DIM, K, bf_ms):
    # ── HNSW via hnswlib — with mock fallback if not installed ────────────────────
    try:
        import hnswlib

        N = len(X_10k)
        X_norm = (X_10k / np.linalg.norm(X_10k, axis=1, keepdims=True)).astype(np.float32)
        rng4 = np.random.default_rng(17)
        query_h = rng4.standard_normal(DIM).astype(np.float32)
        query_h /= np.linalg.norm(query_h)
        true_top = set(brute_force_knn(X_10k, query_h, K))

        print("Building HNSW index (M=16, ef_construction=200)...")
        t0 = time.perf_counter()
        index = hnswlib.Index(space="cosine", dim=DIM)
        index.init_index(max_elements=N, ef_construction=200, M=16)
        index.add_items(X_norm, list(range(N)))
        print(f"  Build: {(time.perf_counter()-t0)*1000:.0f}ms")

        ef_values = [10, 25, 50, 100, 200]
        recalls, latencies = [], []
        REPS = 100

        print(f"\n{'ef_search':>10} {'recall@10':>12} {'latency (ms)':>14}")
        print("-" * 40)
        for ef in ef_values:
            index.set_ef(ef)
            labels_h, _ = index.knn_query(query_h.reshape(1, -1), k=K)
            rec = len(set(labels_h[0]) & true_top) / K
            recalls.append(rec)
            t0 = time.perf_counter()
            for _ in range(REPS):
                index.knn_query(query_h.reshape(1, -1), k=K)
            lat = (time.perf_counter() - t0) / REPS * 1000
            latencies.append(lat)
            print(f"{ef:>10}  {rec*100:>10.0f}%  {lat:>12.3f}ms")

        hnsw_best_ms = latencies[2]  # ef=50

    except ImportError:
        print("hnswlib not installed — pip install hnswlib to run the real benchmark.")
        print("Using representative mock values from typical benchmarks at N=10K, d=384.")
        ef_values = [10, 25, 50, 100, 200]
        recalls = [0.70, 0.87, 0.97, 0.99, 1.00]
        latencies = [bf_ms * r for r in [0.05, 0.08, 0.12, 0.20, 0.35]]
        hnsw_best_ms = latencies[2]

    # ── Recall vs latency trade-off curve ────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(latencies, [r * 100 for r in recalls],
             "o-", color="#7c3aed", linewidth=2, markersize=8, label="HNSW")
    ax2.axhline(y=100, color="#6b7280", linestyle="--", alpha=0.5, label="Brute force (100% recall)")
    for ef, lat, rec in zip(ef_values, latencies, recalls):
        ax2.annotate(f"ef={ef}", (lat, rec * 100),
                     textcoords="offset points", xytext=(5, -12), fontsize=8, color="#7c3aed")
    ax2.set_xlabel("Query latency (ms)")
    ax2.set_ylabel("Recall@10 (%)")
    ax2.set_title("HNSW: Recall vs Latency Trade-off (ef_search knob)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    plt.savefig("/tmp/hnsw_tradeoff.png", dpi=100)
    plt.close(fig2)

    return (hnsw_best_ms,)


@app.cell
def visual_comparison(plt, bf_ms, ivf_ms, hnsw_best_ms):
    # ── Side-by-side latency bar chart ────────────────────────────────────────────
    methods = ["Brute Force\n(100% recall)", "IVF nprobe=3\n(~93% recall)", "HNSW ef=50\n(~97% recall)"]
    times = [bf_ms, ivf_ms, hnsw_best_ms]
    colors = ["#dc2626", "#f59e0b", "#16a34a"]

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    bars = ax3.bar(methods, times, color=colors, alpha=0.85, width=0.5)
    for bar, ms in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                 f"{ms:.2f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Query latency (ms)")
    ax3.set_title("Query Latency Comparison @ N=10K, d=384")
    ax3.set_ylim(0, max(times) * 1.3)
    ax3.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()
    plt.savefig("/tmp/comparison_bar.png", dpi=100)
    plt.close(fig3)

    print("── Algorithm comparison @ N=10K vectors, d=384 dims ──")
    print(f"  Brute force:  {bf_ms:.2f}ms   (100% recall — ground truth)")
    print(f"  IVF:          {ivf_ms:.2f}ms   (~93% recall — 3/20 clusters searched)")
    print(f"  HNSW ef=50:   {hnsw_best_ms:.2f}ms   (~97% recall — hierarchical graph traversal)")
    print(f"\n  HNSW is {bf_ms/max(hnsw_best_ms, 0.001):.1f}× faster than brute force at ~97% recall → best ratio")
    return


@app.cell
def part5_why_hybrid(mo):
    mo.md("""
    ---
    ## Part 5: Hybrid Search — Dense + Sparse

    ### Why hybrid?

    | Method | Wins at | Misses |
    |--------|---------|--------|
    | **Dense (vector)** | Semantics: "software developer" ≈ "engineer" | Exact terms: "Python 3.11", "USAA", specific acronyms |
    | **Sparse (BM25)** | Exact matches, rare terms, version numbers | Paraphrases, synonyms, conceptual similarity |
    | **Hybrid (both)** | Both ✓ | Minimal |

    Dense retrieval alone misses exact entity names, version numbers, and specialized jargon.
    BM25 alone misses semantic similarity. Hybrid retrieval is simply better — and surprisingly cheap to add.

    > "In my RAG system design notebook, I flagged hybrid retrieval as a gap in my QA project.
    > Here's the pattern implemented from scratch."
    """)
    return


@app.cell
def part5_rrf_concept(mo):
    mo.md("""
    ### Reciprocal Rank Fusion (RRF) — The Standard Combiner

    **The problem with score averaging:** BM25 scores live in [0, 30+]. Cosine similarity lives in [0, 1].
    You can't average them — different scales, different distributions. Normalizing doesn't fully fix it.

    **RRF uses ranks instead of scores.** Ranks are always comparable across any two rankers:

    ```
    RRF_score(d) = Σᵢ  1 / (k + rank_i(d))
    ```

    - `rank_i(d)` = position of document d in ranker i's list (1-indexed; 1 = best)
    - `k` = damping constant (typically 60) — limits how dominant a rank-1 result can be
    - Sum over i rankers (dense + sparse = 2 rankers here)

    **Why k=60?** Rank 1 → `1/(60+1) ≈ 0.016`. Rank 10 → `1/(60+10) ≈ 0.014`. The gap is real but not overwhelming —
    the second ranker still has meaningful influence.

    **Algorithm:**
    1. Dense search → top-20 results ordered by similarity (rank 1..20)
    2. BM25 search → top-20 results ordered by BM25 score (rank 1..20)
    3. For each unique document: sum 1/(k+rank) contributions from each ranker it appeared in
    4. Sort by RRF score descending → return top K

    Documents that appear highly in BOTH lists get the highest combined scores.
    Documents that appear in only one list still get some credit — they're not discarded.
    """)
    return


@app.cell
def part5_rrf_code(np):
    # ── RRF from scratch — ~15 lines ─────────────────────────────────────────────

    def rrf_fusion(
        dense_ids: list[int],
        sparse_ids: list[int],
        k: int = 60,
        top_n: int = 10,
    ) -> list[int]:
        """Combine two ranked lists with Reciprocal Rank Fusion."""
        scores: dict[int, float] = {}
        for rank, doc_id in enumerate(dense_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        for rank, doc_id in enumerate(sparse_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    return (rrf_fusion,)


@app.cell
def part5_demo(np, rrf_fusion):
    import random
    from sklearn.feature_extraction.text import TfidfVectorizer
    from rank_bm25 import BM25Okapi

    # ── 100 fake job descriptions ─────────────────────────────────────────────────
    BASE_JOBS = [
        "Senior Python Engineer machine learning platform AWS San Antonio Texas",
        "Junior Java Developer backend services REST API Austin Texas remote",
        "Data Scientist NLP deep learning PyTorch transformer models",
        "Machine Learning Engineer MLOps Kubernetes Docker CI/CD San Antonio",
        "Software Engineer Python Django React full stack San Antonio Texas",
        "Senior Data Scientist computer vision TensorFlow GPU cluster",
        "Backend Engineer Golang microservices gRPC Kafka distributed systems",
        "ML Platform Engineer feature store Feast MLflow experiment tracking",
        "Principal Engineer Python C++ low latency trading fintech Austin",
        "Data Engineer Apache Spark Airflow dbt AWS Glue ETL pipelines",
        "NLP Engineer large language models RLHF fine tuning GPT Claude",
        "Python Developer FastAPI Pydantic async SQLAlchemy backend",
        "Machine Learning Researcher PhD deep learning generative models remote",
        "DevOps Engineer Kubernetes Terraform AWS GCP infrastructure",
        "Senior Software Engineer Python San Antonio Texas hybrid",
        "Data Analyst SQL Tableau Power BI business intelligence reporting",
        "AI Engineer LLM applications RAG vector databases ChromaDB",
        "Robotics Engineer ROS Python C++ embedded systems San Antonio",
        "Quantitative Researcher Python statistics time series forecasting",
        "Senior Machine Learning Engineer San Antonio TX full time onsite",
        "Python Backend Developer PostgreSQL Redis Celery microservices",
        "Computer Vision Engineer YOLO object detection medical imaging",
        "Data Platform Engineer Snowflake dbt BigQuery analytics engineering",
        "Applied Scientist reinforcement learning simulation autonomous agents",
        "Staff Engineer Python distributed systems reliability SRE",
    ]

    random.seed(42)
    locations = ["San Antonio TX", "Austin TX", "Remote", "New York NY", "Seattle WA"]
    levels = ["Senior", "Junior", "Lead", "Staff", "Principal", "Mid-level"]
    tags = ["Python 3.11", "Docker required", "3+ years", "5+ years", "equity options", "hybrid role"]

    corpus = list(BASE_JOBS)
    while len(corpus) < 100:
        base = random.choice(BASE_JOBS)
        corpus.append(f"{random.choice(levels)} {base} {random.choice(locations)} {random.choice(tags)}")
    corpus = corpus[:100]

    # ── Dense index: TF-IDF (no API needed) ──────────────────────────────────────
    vectorizer = TfidfVectorizer(max_features=500)
    X_tfidf = vectorizer.fit_transform(corpus).toarray().astype(np.float32)

    # ── Sparse index: BM25 ────────────────────────────────────────────────────────
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    # ── Query ─────────────────────────────────────────────────────────────────────
    QUERY = "senior python machine learning engineer san antonio"
    q_vec = vectorizer.transform([QUERY]).toarray()[0].astype(np.float32)
    q_norm_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    X_norm = X_tfidf / (np.linalg.norm(X_tfidf, axis=1, keepdims=True) + 1e-9)

    dense_ranked = list(np.argsort(X_norm @ q_norm_vec)[::-1][:20])
    sparse_ranked = list(np.argsort(bm25.get_scores(QUERY.lower().split()))[::-1][:20])
    hybrid_ranked = rrf_fusion(dense_ranked, sparse_ranked, k=60, top_n=10)

    print(f"Query: '{QUERY}'\n")

    print("── Dense-only top 5 ──")
    for i, idx in enumerate(dense_ranked[:5], 1):
        tag = "✓ also in BM25 top-10" if idx in sparse_ranked[:10] else "  dense only"
        print(f"  {i}. [{tag}]  {corpus[idx][:75]}")

    print("\n── BM25-only top 5 ──")
    for i, idx in enumerate(sparse_ranked[:5], 1):
        tag = "✓ also in dense top-10" if idx in dense_ranked[:10] else "  sparse only"
        print(f"  {i}. [{tag}]  {corpus[idx][:75]}")

    print("\n── Hybrid RRF top 5 ──")
    for i, idx in enumerate(hybrid_ranked[:5], 1):
        parts = []
        if idx in dense_ranked[:20]:
            parts.append(f"dense@{dense_ranked.index(idx)+1}")
        if idx in sparse_ranked[:20]:
            parts.append(f"bm25@{sparse_ranked.index(idx)+1}")
        print(f"  {i}. [{', '.join(parts)}]  {corpus[idx][:75]}")
    return


@app.cell
def part5_canopy_gap(mo):
    mo.md("""
    ### My Canopy Gap — What This Means in Practice

    **Current state:** Canopy uses `sqlite-vec` for dense vector search only.
    The database already has **FTS5** (SQLite's built-in full-text search) for filtering —
    but it's not combined with the vector search.

    **The gap:** a query like "Python 3.11 roles at USAA San Antonio" matches semantically via dense search,
    but the exact terms "USAA" and "Python 3.11" might surface the wrong jobs if those tokens are
    underrepresented in the embedding space.

    **The fix:** ~50 lines in `backend/src/services/`:
    ```python
    def hybrid_search(query: str, k: int = 10) -> list[Job]:
        # Dense: sqlite-vec KNN — already exists
        dense_results = sqlite_vec_knn(embed(query), limit=20)
        # Sparse: FTS5 full-text search — already in the DB, just not wired to retrieval
        sparse_results = fts5_search(query, limit=20)
        # Combine
        combined_ids = rrf_fusion(
            dense_ids=[r.id for r in dense_results],
            sparse_ids=[r.id for r in sparse_results],
        )
        return fetch_jobs_by_ids(combined_ids[:k])
    ```

    **Why this matters for my portfolio:** it demonstrates the exact pattern from my system design
    notebook applied to a real project — and shows I understand WHY I'd add it, not just that
    "hybrid search is better." No external dependencies. ~50 lines. Works with the existing DB schema.

    **Implementation plan:** add `hybrid_search()` to `backend/src/services/search.py`.
    Query both indexes, RRF combine, optionally rerank top results with a cross-encoder.
    """)
    return


@app.cell
def part6_quantization(mo):
    mo.md("""
    ---
    ## Part 6: Quantization & Compression — Scaling to Billions

    At billion-vector scale, even HNSW becomes RAM-constrained.

    **Raw numbers:**
    1B vectors × 768 dims × 4 bytes (float32) = **~3TB** for vectors alone. Not practical in RAM.

    ### Compression techniques

    | Method | Compression | Recall loss | How it works |
    |--------|-------------|-------------|--------------|
    | **Scalar Quantization (SQ8)** | 4× | Minimal | float32 → int8. Simple, effective first step. |
    | **Product Quantization (PQ)** | 8–32× | Low–medium | Split each vector into M subvectors, quantize each to a codebook. 768-dim → ~96 bytes. |
    | **Binary Quantization** | 32× | Higher | float32 → 1 bit per dim. Hamming distance. Extreme compression, lower recall. |

    ### When to use
    - **< 1M vectors:** don't bother. HNSW float32 is fine. Premature optimization.
    - **1M–100M:** Scalar Quantization (SQ8) — 4× memory reduction, minimal recall loss.
    - **100M–1B:** IVF-PQ — 10–30× compression. Compensate recall loss with higher `nprobe`.
    - **> 1B:** Distributed sharding + PQ per shard. This is Pinecone/Milvus territory.

    > "For my use cases (5K–50K vectors), compression isn't needed. But knowing this answers
    > 'how would you scale to 1B documents?' in interviews: shard across machines, IVF-PQ per
    > shard, scalar quantization as a first pass."
    """)
    return


@app.cell
def decision_framework(mo):
    mo.md("""
    ---
    ## Choosing the Right Index — Decision Framework

    ```
    How many vectors?
    ├── < 100K   → Flat (brute force). Fast enough. Don't over-engineer.
    │   └── Canopy (~5K), QA project (<10K). ChromaDB flat index is perfect.
    ├── 100K–10M → HNSW. Best recall/speed. Fits in RAM (~2–20GB).
    │   └── Most production RAG systems live here.
    ├── 10M–100M → HNSW + quantization (SQ8 or PQ). Reduce memory footprint.
    │   └── Or IVF-PQ if memory is very constrained.
    └── > 100M   → Distributed: shard across machines + IVF-PQ per shard.
        └── Pinecone / Milvus territory.

    Need real-time inserts?
    ├── YES → HNSW (incremental inserts, no rebuild). NOT IVF (rebuild required).
    └── NO  → Either works.

    Need hybrid search?
    ├── YES → Weaviate, Qdrant (built-in), or DIY: BM25 + dense + RRF (~50 lines).
    └── NO  → Any vector DB works.

    Need SQL / existing DB integration?
    ├── YES → pgvector (Postgres extension — HNSW or IVF options).
    └── NO  → Purpose-built vector DB (Qdrant, Weaviate, ChromaDB).
    ```
    """)
    return


@app.cell
def flashcard_summary(mo):
    mo.md("""
    ---
    ## Flashcard Summary

    | Question | Answer |
    |----------|--------|
    | **What algorithm does ChromaDB use?** | HNSW via hnswlib under the hood |
    | **How does HNSW work in one sentence?** | Hierarchical graph — sparse long-range edges at top layers for coarse navigation, dense local edges at bottom for precision; greedy traversal top-to-bottom |
    | **HNSW's three key parameters?** | M (graph density), ef_construction (build quality), ef_search (query accuracy/speed dial) |
    | **Why hybrid over dense-only?** | Dense misses exact keywords and entity names; BM25 misses paraphrases. RRF combines both for higher recall. |
    | **What is RRF?** | Reciprocal Rank Fusion: score = Σ 1/(k+rank_i). Uses ranks instead of scores — normalizes across different scoring scales. |
    | **When is brute force good enough?** | Under ~100K vectors. At 10K × 384d: ~10ms. Don't over-engineer below that. |
    | **Memory cost of HNSW?** | Vectors: N×d×4 bytes. Graph: N×M×layers×8 bytes. 1M 384d vectors M=16: ~1.5GB + ~640MB = ~2.1GB |
    | **Scale to 1B vectors?** | Shard across machines + IVF-PQ per shard + scalar quantization to reduce memory |
    | **Cosine vs dot product vs L2?** | Cosine for text (direction). L2 for images (scale). L2-normalized → cosine = dot product → use dot product (faster). |
    | **Trick question — divide by sqrt(d)?** | That's attention (Q@K^T / sqrt(d_k)), not HNSW. But dot product IS the same operation — vector search is attention scaled to millions of keys. |
    | **IVF vs HNSW for real-time inserts?** | HNSW — incremental inserts work fine. IVF requires re-running K-means when distribution drifts, which degrades recall. |
    """)
    return


@app.cell
def interview_talking_points(mo):
    mo.md("""
    ---
    ## Interview Talking Points

    **"Explain how your vector search works"**
    > "In Canopy, I use sqlite-vec with all-MiniLM-L6-v2 at 384 dimensions. At ~5K job vectors,
    > flat index search takes ~5ms — HNSW isn't needed yet. If I scaled to 1M+ job listings,
    > I'd migrate to ChromaDB or Qdrant with HNSW (M=16), and tune ef_search to hit the
    > recall/latency balance my SLA needs — probably ef=50 for ~97% recall at ~10× the speed."

    ---

    **"How would you improve retrieval quality in your RAG system?"**
    > "I'd add hybrid search. Canopy already has FTS5 for keyword search alongside sqlite-vec
    > for dense. Combining them with RRF would catch exact matches that dense misses —
    > specific company names like USAA, tech versions like Python 3.11 — and semantics that
    > BM25 misses. Implementation is ~50 lines: query both indexes, RRF combine, return top K.
    > No new dependencies — FTS5 is already in SQLite."

    ---

    **"What's your understanding of HNSW?"**
    > "Hierarchical graph where top layers have sparse long-range edges for coarse navigation
    > and bottom layers are dense for precise neighbor finding. Search greedily traverses
    > top-to-bottom. The hierarchy is what prevents local optima that plague flat NSW.
    > M controls graph density, ef_construction controls build quality, ef_search is the
    > query-time knob I'd tune without rebuilding the index — higher ef_search = better recall,
    > slower queries."

    ---

    **"How would you scale your RAG system to 100M documents?"**
    > "First: IVF-PQ for memory efficiency — Product Quantization compresses 768-dim float32
    > vectors ~10–30×. Then: shard across machines by domain or date, with an HNSW or IVF-PQ
    > index per shard. A routing layer sends queries to relevant shards in parallel and merges
    > results with RRF. This is roughly Milvus's architecture."

    ---

    **Connection to yesterday's attention notebook:**
    The dot product in attention (Q @ K^T) is the same mathematical operation as vector similarity search —
    finding which keys are most relevant to a query. Vector databases are just scaling that operation
    from sequence-length (512 tokens) to millions of documents, with approximate algorithms to make
    it tractable. Understanding attention makes HNSW intuitive: it's the same "find the most similar
    things" problem at a different scale.
    """)
    return


if __name__ == "__main__":
    app.run()
