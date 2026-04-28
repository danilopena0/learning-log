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
    # Dimensionality Reduction — PCA Intuition, Embeddings, When/Why to Reduce Dimensions

    | Field | Value |
    |-------|-------|
    | Date  | 2026-04-27 |
    | Track | ML Theory |
    | Time  | 60 min |
    | Topics | Curse of Dimensionality · PCA · t-SNE · UMAP · Learned Embeddings · Feature Selection |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## The Curse of Dimensionality — Why This Matters

    The paradox: more features should mean more information. And at first, they do. But past a
    certain point, adding dimensions makes everything *worse* — models overfit more easily,
    distances lose meaning, and computation explodes. This is the **curse of dimensionality**.

    ### Why does it happen?

    1. **Sparsity**: in high dimensions, data points become equidistant from each other. A 100-dim
       unit cube has volume 1, but almost all that volume sits in the corners — the interior is
       essentially empty. KNN breaks because "nearest neighbor" becomes meaningless when every
       point is roughly the same distance away.

    2. **Overfitting**: more features = more parameters = more capacity to memorize noise. Filling
       a high-dimensional space requires exponentially more data. Doubling features roughly requires
       *squaring* the number of samples to maintain the same data density.

    3. **Computation**: distance calculations scale with d. Training time and memory grow linearly
       or worse with dimensionality.

    4. **Visualization**: you can't plot anything beyond 3D. Understanding your data requires
       reducing to 2–3 dimensions first.

    **Rule of thumb**: if you have N samples, you need d << N features. With 1,000 samples,
    500 features is asking for trouble.
    """)
    return


@app.cell
def equidistance_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy.spatial.distance import pdist as _pdist

    _rng = _np.random.default_rng(42)
    _n = 100
    _dims = [2, 50, 500]
    _colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    _fig, _axes = _plt.subplots(1, 3, figsize=(14, 4))

    for _ax, _d, _color in zip(_axes, _dims, _colors):
        _pts = _rng.uniform(0, 1, (_n, _d))
        _dists = _pdist(_pts, metric="euclidean")
        _ax.hist(_dists, bins=30, color=_color, alpha=0.85, edgecolor="white", linewidth=0.5)
        _cv = _dists.std() / _dists.mean()
        _ax.set_title(f"d = {_d}", fontsize=13, fontweight="bold")
        _ax.set_xlabel("Pairwise Distance", fontsize=11)
        if _d == 2:
            _ax.set_ylabel("Count", fontsize=11)
        _ax.text(
            0.95, 0.95, f"CV = {_cv:.3f}",
            transform=_ax.transAxes, ha="right", va="top", fontsize=11, color=_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=_color, alpha=0.9),
        )
        _ax.grid(True, alpha=0.3)

    _fig.suptitle(
        "Distance Concentration: All Pairwise Distances in d Dimensions",
        fontsize=13, fontweight="bold",
    )
    _fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig


@app.cell
def _(mo):
    mo.md("""
    The **coefficient of variation (CV = std / mean)** measures how spread out the distances are.
    At d = 2, distances vary widely — you can reliably tell near from far. At d = 500, all
    distances cluster around the same value (CV ≈ 0). When all points are equidistant,
    distance-based methods (KNN, clustering, vector search) lose discriminative power.
    This is the curse in action.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 1: PCA — The Core Method

    ### Intuition

    Your data might live in 100 dimensions, but maybe 95% of the variance lies along just 10
    directions. PCA finds those directions.

    **Goal**: find a lower-dimensional subspace that preserves as much *variance* (information)
    as possible.

    **The pancake analogy**: imagine a pancake floating in 3D space. The pancake is fundamentally
    2D — PCA finds the plane of the pancake and projects onto it, discarding the thin dimension.
    The projected data loses almost nothing because the thin direction had almost no variance.

    PCA finds the directions of **maximum variance**, ranked by importance. You keep the top k
    and discard the rest.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The Math — Step by Step

    Given data matrix **X** of shape (n, d) — n samples, d features.

    **Step 1: Center the data** — subtract the mean from each feature.

    $$X_c = X - \bar{X}$$

    *Why*: PCA finds directions of variance. If data isn't centered, the mean direction dominates.
    Centering isolates the spread around zero.

    ---

    **Step 2: Compute the covariance matrix** — shape (d, d).

    $$C = \frac{1}{n} \, X_c^T X_c$$

    C[i, j] = how much feature i and feature j vary together. Diagonal = variance of each feature.
    Off-diagonal = covariance between feature pairs.

    ---

    **Step 3: Eigendecomposition** — $C = V \Lambda V^T$

    - **Eigenvectors V** = the principal component directions (axes of the new coordinate system)
    - **Eigenvalues Λ** = the variance along each direction (importance of each component)
    - Eigenvalues sorted descending: first component has most variance, second has next most, etc.

    ---

    **Step 4: Project** onto the top k eigenvectors.

    $$X_{\text{reduced}} = X_c \cdot V[:, :k] \quad \text{shape: } (n, k)$$

    Each sample now has k features instead of d.

    *Why eigenvectors?* They are the directions where the data is most spread out. Projecting
    onto them preserves the maximum amount of information in a variance sense — provably optimal
    among all linear projections of rank k.
    """)
    return


@app.cell
def pca_scratch():
    import numpy as _np
    from sklearn.decomposition import PCA as _SklearnPCA

    _rng = _np.random.default_rng(42)

    def pca(X, k):
        X_c = X - X.mean(axis=0)
        cov = X_c.T @ X_c / len(X)
        eigenvalues, eigenvectors = _np.linalg.eigh(cov)
        # eigh returns ascending order — flip to descending
        idx = _np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return X_c @ eigenvectors[:, :k], eigenvalues, eigenvectors

    # Correlated 2D data: a tilted ellipse
    _cov_matrix = [[3.0, 2.5], [2.5, 2.5]]
    X_corr = _rng.multivariate_normal([0.0, 0.0], _cov_matrix, 200)

    _, eigenvalues, eigenvectors = pca(X_corr, k=1)

    # Verify against sklearn
    _sk = _SklearnPCA(n_components=2).fit(X_corr)
    _ev_ratio = eigenvalues / eigenvalues.sum()

    print("=== PCA from Scratch vs sklearn ===")
    print(f"Eigenvalues:              {eigenvalues.round(4)}")
    print(f"Explained var (scratch):  {_ev_ratio.round(4)}")
    print(f"Explained var (sklearn):  {_sk.explained_variance_ratio_.round(4)}")
    print(f"Results match:            {_np.allclose(_ev_ratio, _sk.explained_variance_ratio_, atol=1e-3)}")

    return X_corr, eigenvalues, eigenvectors, pca


@app.cell
def pca_projection_viz(X_corr, eigenvalues, eigenvectors):
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _mean = X_corr.mean(axis=0)
    _X_c = X_corr - _mean
    _pc1 = eigenvectors[:, 0]
    _pc2 = eigenvectors[:, 1]

    _fig, (_ax_2d, _ax_1d) = _plt.subplots(
        2, 1, figsize=(9, 10),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Original data
    _ax_2d.scatter(
        X_corr[:, 0], X_corr[:, 1],
        alpha=0.45, s=28, color="#3498DB", label="Original data", zorder=3,
    )

    # PC arrows scaled by sqrt(eigenvalue)
    for _j, (_ev, _col, _label) in enumerate(zip(
        eigenvalues, ["#E74C3C", "#2ECC71"], ["PC1", "PC2"]
    )):
        _scale = _np.sqrt(_ev) * 2.2
        _vec = eigenvectors[:, _j]
        _ax_2d.annotate(
            "", xy=_mean + _scale * _vec, xytext=_mean - _scale * _vec,
            arrowprops=dict(arrowstyle="-|>", color=_col, lw=2.5, mutation_scale=18),
        )
        _pct = eigenvalues[_j] / eigenvalues.sum()
        _offset = _vec * _scale * 1.22
        _ax_2d.text(
            *(_mean + _offset), f"{_label}\n({_pct:.0%} var)",
            color=_col, fontsize=10, fontweight="bold", ha="center", va="center",
        )

    # Projection lines from every 3rd point onto PC1
    _proj_coords = (_X_c @ _pc1)[:, None] * _pc1 + _mean
    for _i in range(0, len(X_corr), 3):
        _ax_2d.plot(
            [X_corr[_i, 0], _proj_coords[_i, 0]],
            [X_corr[_i, 1], _proj_coords[_i, 1]],
            color="#E74C3C", alpha=0.18, lw=0.9, linestyle="--",
        )

    # Projected points lying on the PC1 axis
    _ax_2d.scatter(
        _proj_coords[:, 0], _proj_coords[:, 1],
        s=18, color="#E74C3C", alpha=0.65, zorder=4, label="Projected onto PC1",
    )

    _ax_2d.set_aspect("equal")
    _ax_2d.set_title(
        "PCA: 2D → 1D Projection  (the whiteboard diagram)", fontsize=14, fontweight="bold"
    )
    _ax_2d.legend(fontsize=10, loc="upper left")
    _ax_2d.grid(True, alpha=0.3)
    _ax_2d.set_xlabel("Feature 1", fontsize=11)
    _ax_2d.set_ylabel("Feature 2", fontsize=11)

    # 1D number line below
    _proj_1d = _X_c @ _pc1
    _ax_1d.scatter(
        _proj_1d, _np.zeros_like(_proj_1d),
        alpha=0.5, s=25, color="#E74C3C", zorder=3,
    )
    _ax_1d.axhline(0, color="black", lw=1.5)
    _ax_1d.set_xlim(_proj_1d.min() * 1.25, _proj_1d.max() * 1.25)
    _ax_1d.set_ylim(-0.6, 0.6)
    _ax_1d.set_yticks([])
    _ax_1d.set_xlabel("Position along PC1", fontsize=11)
    _ax_1d.set_title(
        f"1D projection — PC1 retains {eigenvalues[0] / eigenvalues.sum():.0%} of the variance",
        fontsize=12,
    )
    _ax_1d.grid(True, alpha=0.3, axis="x")

    _fig.tight_layout(pad=2.5)
    return _fig


@app.cell
def explained_variance_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_classification as _make_clf
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _SS

    _X, _ = _make_clf(
        n_samples=500, n_features=20, n_informative=5,
        n_redundant=5, n_clusters_per_class=1, random_state=42,
    )
    _X_scaled = _SS().fit_transform(_X)
    _pca = _PCA(n_components=20).fit(_X_scaled)

    _ev = _pca.explained_variance_ratio_
    _cum = _np.cumsum(_ev)
    _comps = _np.arange(1, 21)
    _k90 = int(_np.searchsorted(_cum, 0.90)) + 1

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart — blue = kept, grey = discarded
    _bar_colors = ["#3498DB" if _i < _k90 else "#BDC3C7" for _i in range(20)]
    _ax1.bar(_comps, _ev * 100, color=_bar_colors, alpha=0.85, edgecolor="white")
    _ax1.set_xlabel("Principal Component", fontsize=12)
    _ax1.set_ylabel("Explained Variance (%)", fontsize=12)
    _ax1.set_title("Variance per Component", fontsize=13, fontweight="bold")
    _ax1.set_xticks(_comps)
    _ax1.grid(True, alpha=0.3, axis="y")

    # Cumulative scree plot
    _ax2.plot(_comps, _cum * 100, "o-", color="#E74C3C", lw=2.5, ms=7, zorder=4)
    _ax2.fill_between(_comps, _cum * 100, alpha=0.08, color="#E74C3C")
    _ax2.axhline(90, color="#27AE60", lw=2, linestyle="--", label="90% threshold")
    _ax2.axhline(95, color="#F39C12", lw=2, linestyle="--", label="95% threshold")
    _ax2.axvline(_k90, color="#27AE60", lw=1.5, linestyle=":", alpha=0.8)
    _ax2.text(
        _k90 + 0.25, 52, f"k = {_k90}\n({_cum[_k90 - 1]:.0%} var)",
        color="#27AE60", fontsize=11,
    )
    _ax2.set_xlabel("Number of Components", fontsize=12)
    _ax2.set_ylabel("Cumulative Variance (%)", fontsize=12)
    _ax2.set_title("Cumulative Explained Variance (Scree Plot)", fontsize=13, fontweight="bold")
    _ax2.set_xticks(_comps)
    _ax2.set_ylim(0, 107)
    _ax2.legend(fontsize=10)
    _ax2.grid(True, alpha=0.3)

    _fig.suptitle(
        "20 features, 5 informative — elbow k matches the ground truth",
        fontsize=13, fontweight="bold",
    )
    _fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig


@app.cell
def _(mo):
    mo.md("""
    The **blue bars** are components you'd keep; **grey bars** are discarded noise. The elbow
    shows where additional components add diminishing returns. Here, ~5–7 components capture 90%
    of variance, roughly matching the 5 informative features used to generate the data.

    **The scree plot is how you choose k in practice.** Pick k where cumulative variance reaches
    90–95%, or at the natural elbow.

    ### Scree Plot Interpretation: Three Scenarios

    1. **Sharp elbow at k=3** → data is truly ~3-dimensional. Strong case for reduction — you
       lose almost nothing and gain massive simplicity.

    2. **Gradual decline** → no dominant subspace. PCA will lose important structure at any
       reasonable k. Consider whether you need reduction at all.

    3. **All components roughly equal** → data is truly high-dimensional; information is spread
       uniformly. PCA won't help much.

    If the scree plot is gradual, ask whether you even *need* dimensionality reduction. Tree
    models handle high dimensions natively and gain nothing from PCA.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 2: PCA Limitations & Alternatives

    ### PCA Limitations

    - **Linear only**: PCA finds linear subspaces. If your data lies on a curve or manifold
      (spiral, Swiss roll), PCA can't unroll it — it needs nonlinear methods.

    - **Variance ≠ usefulness**: the direction of maximum variance isn't necessarily the most
      discriminative for your task. A feature with small variance might be the most predictive;
      PCA would discard it.

    - **Sensitive to scaling**: features with larger numerical scales dominate the covariance
      matrix. A feature in thousands will dwarf one in decimals. *Always standardize before PCA.*
      This is a common interview trap.

    - **Orthogonality constraint**: each component must be orthogonal to all previous ones.
      Mathematically elegant, but real-world structure may not respect orthogonality.
    """)
    return


@app.cell
def scaling_trap_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _SS

    _rng = _np.random.default_rng(42)
    _n = 200
    _f1 = _rng.normal(0.5, 0.15, _n)
    _f2 = 800 * _f1 + _rng.normal(0, 40, _n)
    _X_raw = _np.column_stack([_f1, _f2])
    _X_scaled = _SS().fit_transform(_X_raw)

    def _draw(ax, X, title):
        _pca = _PCA(n_components=2).fit(X)
        _mean = X.mean(axis=0)
        ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=20, color="#95A5A6", zorder=2)
        for _j, _col in enumerate(["#E74C3C", "#3498DB"]):
            _scale = _np.sqrt(_pca.explained_variance_[_j]) * 2.0
            _vec = _pca.components_[_j] * _scale
            ax.annotate(
                "", xy=_mean + _vec, xytext=_mean,
                arrowprops=dict(arrowstyle="-|>", color=_col, lw=3, mutation_scale=16),
                zorder=5,
            )
            _pct = _pca.explained_variance_ratio_[_j]
            ax.text(
                *(_mean + _vec * 1.28), f"PC{_j + 1}\n({_pct:.0%})",
                color=_col, fontsize=10, fontweight="bold", ha="center",
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(
            f"Feature 1  ({X[:, 0].min():.2f} – {X[:, 0].max():.2f})", fontsize=10
        )
        ax.set_ylabel(
            f"Feature 2  ({X[:, 1].min():.0f} – {X[:, 1].max():.0f})", fontsize=10
        )
        ax.grid(True, alpha=0.3)

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(13, 5))
    _draw(_ax1, _X_raw, "Without Scaling\nPC1 ≈ Feature 2  (large scale dominates)")
    _draw(_ax2, _X_scaled, "With StandardScaler\nPC1 reflects true correlation structure")
    _fig.suptitle(
        "Scaling Trap: ALWAYS Standardize Before PCA",
        fontsize=14, fontweight="bold", color="#C0392B",
    )
    _fig.tight_layout(rect=[0, 0, 1, 0.90])
    return _fig


@app.cell
def _(mo):
    mo.md("""
    Left: PCA on raw data where Feature 2 is on a 0–1000 scale. PC1 aligns almost entirely with
    the large-scale feature — not because it's more informative, but because it's bigger. Right:
    after standardization, PC1 correctly captures the diagonal correlation between both features.

    **Always standardize before PCA.**

    ---

    ### Nonlinear Alternatives

    | Method | Core idea | Speed | Project new data? | Best use |
    |--------|-----------|-------|-------------------|----------|
    | **t-SNE** | Preserve local neighborhoods via probability matching | O(N²) exact | No | Visualization only |
    | **UMAP** | Manifold approximation; local + global structure | O(N log N) | Yes | Visualization; sometimes preprocessing |
    | **Autoencoders** | Neural encoder (d→k) + decoder (k→d) | Slow — needs training | Yes (forward pass) | Nonlinear reduction + reconstruction |

    **t-SNE** — great for revealing cluster structure in 2D, but axes are meaningless and you
    can't add new points. The `perplexity` parameter (5–50) controls local vs global structure.
    *Visualization only — never as features for an ML model.*

    **UMAP** — similar quality to t-SNE but faster and better at preserving global layout.
    Has `transform()` for new data. Sometimes used as preprocessing, but be cautious — the
    embedding depends heavily on hyperparameters and changes with random seed.

    **Autoencoders** — most powerful, most expensive. Can learn arbitrary manifolds.
    Start with PCA; move to autoencoders only if linear reduction clearly loses important structure.
    """)
    return


@app.cell
def nonlinear_comparison_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_swiss_roll as _msw
    from sklearn.decomposition import PCA as _PCA
    from sklearn.manifold import TSNE as _TSNE
    from sklearn.preprocessing import StandardScaler as _SS
    import warnings as _warnings

    _warnings.filterwarnings("ignore")

    _X_sr, _color = _msw(n_samples=1000, noise=0.15, random_state=42)
    _X_sc = _SS().fit_transform(_X_sr)

    _X_pca_2d = _PCA(n_components=2).fit_transform(_X_sc)
    _X_tsne = _TSNE(
        n_components=2, perplexity=30, random_state=42,
        n_iter=1000, init="pca",
    ).fit_transform(_X_sc)

    _umap_ok = False
    _X_umap = None
    try:
        import umap as _umap
        _X_umap = _umap.UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
        ).fit_transform(_X_sc)
        _umap_ok = True
    except ImportError:
        pass

    _results = [
        (_X_pca_2d, "PCA  (linear — flattens but can't unroll)"),
        (_X_tsne, "t-SNE  (nonlinear — reveals local structure)"),
    ]
    if _umap_ok:
        _results.append((_X_umap, "UMAP  (nonlinear — faster, preserves global structure)"))

    _ncols = len(_results)
    _fig, _axes = _plt.subplots(1, _ncols, figsize=(5 * _ncols, 5))
    if _ncols == 1:
        _axes = [_axes]

    for _ax, (_X_2d, _title) in zip(_axes, _results):
        _sc = _ax.scatter(_X_2d[:, 0], _X_2d[:, 1], c=_color, cmap="viridis", s=12, alpha=0.75)
        _ax.set_title(_title, fontsize=11, fontweight="bold")
        _ax.set_xticks([])
        _ax.set_yticks([])

    _fig.colorbar(_sc, ax=_axes[-1], label="Position along Swiss Roll", shrink=0.8)
    _fig.suptitle(
        "Swiss Roll: PCA vs Nonlinear Methods", fontsize=13, fontweight="bold"
    )
    _fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig


@app.cell
def _(mo):
    mo.md("""
    PCA projects the Swiss roll flat, mixing colors that should be separated — it can't unroll
    the manifold. t-SNE and UMAP reveal the continuous gradient, correctly separating nearby
    positions.

    **But remember**: t-SNE and UMAP are for visualization. Using t-SNE output as features
    for a downstream ML model is usually a mistake — axes have no consistent meaning and the
    embedding changes with every random seed.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 3: Learned Embeddings — The Modern Approach

    PCA is unsupervised and task-agnostic. It doesn't know what you're trying to predict — it
    finds directions of maximum variance regardless of whether those directions are useful.

    **Learned embeddings** are different: a neural network is trained to compress inputs into
    low-dimensional vectors that are *useful for a specific task*. The loss function encodes
    what "similar" means.

    ### Examples

    - **Word2Vec / GloVe**: learn word embeddings where similar words are geometrically close.
      Trained on co-occurrence — words appearing in similar contexts get similar vectors.
    - **Sentence-Transformers**: learn sentence embeddings optimized for semantic similarity.
      Trained on pairs of similar/dissimilar sentences.
    - **Image embeddings (ResNet, CLIP)**: compressed representations optimized for
      classification or image-text similarity.
    - **Collaborative filtering embeddings**: users who like similar items end up close in
      embedding space. The embedding encodes preference structure, not raw features.

    **Key difference from PCA**: embeddings are *trained with a loss function* that captures
    task-relevant structure. PCA just maximizes variance regardless of your goal.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### PCA vs Learned Embeddings

    | | PCA | Learned Embeddings |
    |--|-----|-------------------|
    | Supervision | Unsupervised | Task-specific (supervised or self-supervised) |
    | Structure captured | Linear | Arbitrary (neural network) |
    | Interpretability | Linear combos of features | Opaque (black box) |
    | Compute | Fast (eigendecomposition) | Expensive (neural network training) |
    | New data | Project with saved eigenvectors | Forward pass through encoder |
    | Quality | Preserves variance | Preserves task-relevant similarity |
    | Use case | Tabular data, preprocessing | Text, images, recommendations |

    ---

    ### Connection to My Work

    **Canopy** uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) for job similarity.
    These are learned embeddings — a transformer trained to put semantically similar sentences
    close in vector space. The 384 dimensions are already a compressed, task-optimized
    representation of raw text.

    **Vector databases**: everything in vector search is built on embeddings. The quality of
    your embeddings determines the quality of your retrieval. No amount of HNSW tuning fixes
    bad embeddings.

    **Why not PCA the job embeddings?** The 384-dim output already encodes semantic structure
    the model learned. PCA on top destroys task-specific structure by discarding low-variance
    directions that may be semantically critical. PCA on raw TF-IDF features *would* be
    appropriate — vocabulary size can be 50k+ and PCA to ~300 works well there.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 4: When to Reduce Dimensions — Decision Framework

    ```
    What's your goal?
    ├── Visualization
    │   └── t-SNE or UMAP  (reduce to 2–3D)
    │
    ├── Preprocessing for an ML model
    │   ├── Tabular data, many correlated features
    │   │   └── PCA — standardize first, keep 90–95% variance
    │   ├── Text / image / audio
    │   │   └── Use pre-trained embeddings, NOT PCA on raw features
    │   ├── Tree model (XGBoost, RF, LightGBM)
    │   │   └── Don't reduce — trees handle high dims natively
    │   ├── KNN / SVM / logistic regression
    │   │   └── Reduce — these struggle in high dimensions
    │   └── Deep learning
    │       └── Skip — the network learns its own representations
    │
    ├── Speed / memory optimization
    │   └── PCA or random projection
    │
    └── Denoising
        └── PCA (keep top components, discard noisy tail)

    How many dimensions to keep?
    ├── Visualization → 2 or 3
    ├── Preprocessing → scree plot elbow (90–95% threshold)
    ├── Speed constraint → smallest k with acceptable accuracy at target latency
    └── Unsure → try k = [10, 50, 100, ...], cross-validate downstream model
    ```

    ---

    ### When NOT to Reduce Dimensions

    - **Tree-based models** (XGBoost, random forest): handle high dimensions via feature
      importance. PCA *hurts* because you lose the ability to trace predictions to original
      features — the interpretability advantage of trees evaporates.

    - **Deep learning**: learns its own internal representations. PCA before a neural net is
      almost always pointless.

    - **When you need feature interpretability**: "PC3" is meaningless to stakeholders. Use
      feature *selection* instead — keep original features, drop irrelevant ones.

    - **When features are already independent**: PCA helps most when features are correlated.
      If they're already uncorrelated, PCA just reorders them without reducing anything useful.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Feature Selection vs Dimensionality Reduction

    These are often conflated but are fundamentally different tools.

    **Feature Selection** — keep a subset of *original* features, drop the rest.
    - Interpretable: you can still name the features that survive
    - Methods: correlation analysis, mutual information, LASSO (L1 zeroes out coefficients),
      recursive feature elimination, permutation importance
    - Best when: stakeholders need to know which inputs matter, or features are meaningful on
      their own

    **Dimensionality Reduction** — create *new* features that are combinations of originals.
    - Not interpretable: "PC1" or "dim 47" means nothing by itself
    - Methods: PCA, autoencoders, t-SNE, UMAP
    - Best when: maximum information retention matters and interpretability doesn't

    | | Feature Selection | Dimensionality Reduction |
    |--|-----------------|------------------------|
    | Features kept | Original subset | New combinations |
    | Interpretability | High | Low |
    | Information retained | Partial (dropped features lost) | Maximum for given k |
    | Use case | Stakeholder explainability, tree models | KNN/SVM preprocessing, visualization |

    **Interview tip**: if someone says "reduce dimensionality," clarify — "do you mean select
    a subset of original features, or project into a new lower-dimensional space?" That
    distinction matters for the method choice, the compute cost, and what information you lose.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Flashcard Summary

    **"What does PCA do?"**
    > Finds the directions of maximum variance and projects data onto the top k, discarding
    > the rest. Optimal linear compression for a given k.

    ---

    **"Walk me through the PCA math."**
    > Center the data, compute the covariance matrix (X_c^T X_c / n), eigendecompose it,
    > project onto the top k eigenvectors. Four steps.

    ---

    **"Why standardize before PCA?"**
    > Features with larger scales dominate the covariance matrix, distorting which directions
    > look most important. Standardization puts all features on equal footing.

    ---

    **"How do you choose k?"**
    > Explained variance scree plot. Keep enough components for 90–95% cumulative variance,
    > or at the natural elbow.

    ---

    **"PCA vs t-SNE?"**
    > PCA is linear, fast, preserves global structure, projects new data. t-SNE is nonlinear,
    > slow, preserves local neighborhoods, can't project new data. PCA for preprocessing;
    > t-SNE for visualization only.

    ---

    **"PCA vs learned embeddings?"**
    > PCA is unsupervised, linear, task-agnostic — maximizes variance. Embeddings are trained
    > with a loss function, capture task-relevant structure, can be nonlinear.

    ---

    **"When should you NOT use PCA?"**
    > Tree models (handle high dims natively), deep learning (learns its own reps), when you
    > need feature interpretability (PCA components are opaque).

    ---

    **"What's the curse of dimensionality?"**
    > In high dims: data becomes sparse, pairwise distances converge to the same value, models
    > overfit easily. Need exponentially more data to fill the space.

    ---

    **"Feature selection vs dimensionality reduction?"**
    > Selection keeps original features (interpretable, partial info). Reduction creates new
    > combined features (maximum info retention, opaque).

    ---

    **"Why divide by √d_k in attention? Is there a PCA analogue?"**
    > Trick question — that's attention scaling, not PCA. But PCA does normalize via the
    > covariance matrix, which accounts for the scale of variance in each direction. The
    > intuition (preventing large-magnitude directions from dominating) loosely connects to
    > centering and standardization before PCA.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    ### "When would you use PCA?"

    > "On tabular data with many correlated features, as preprocessing for models that struggle
    > with high dimensionality — KNN, logistic regression, SVM. I'd standardize first, use the
    > explained variance elbow to pick k, and cross-validate the downstream model to verify the
    > reduction actually helps. I wouldn't use it with tree models or neural nets."

    ---

    ### "Explain PCA to a non-technical stakeholder."

    > "Imagine your data has 100 measurements per sample. PCA discovers that most of the useful
    > information can be captured with just 10 summary measurements, each a weighted combination
    > of the originals. We keep those 10 and throw away the other 90 — simplifying without
    > losing much. Think of it as finding the most informative angles to view your data from."

    ---

    ### "How does this connect to your projects?"

    > "In Canopy I use sentence-transformer embeddings — a learned compression from raw text
    > down to 384 dimensions, optimized for semantic similarity. That's dimensionality reduction
    > trained end-to-end. My vector database then indexes these for fast similarity search. The
    > pipeline is: raw text → embedding (reduction) → HNSW index (fast search) → results. The
    > embedding quality is the bottleneck, not the index."

    ---

    ### "PCA vs autoencoders?"

    > "PCA is the linear baseline — fast, interpretable, requires no training. Autoencoders
    > capture nonlinear structure but need architecture choices and training time. I'd start
    > with PCA and only move to autoencoders if the scree plot is flat and the data has obvious
    > nonlinear structure that PCA is clearly losing."
    """)
    return


if __name__ == "__main__":
    app.run()
