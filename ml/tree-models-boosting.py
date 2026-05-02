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
    # Tree Models — Decision Trees, Random Forests, XGBoost. When to Use What. Feature Importance.

    | Field | Value |
    |-------|-------|
    | Date  | 2026-04-09 |
    | Track | ML Theory |
    | Time  | 60 min |
    | Topics | Decision Trees · Random Forests · Gradient Boosting · XGBoost · Feature Importance |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Why Trees Dominate Tabular Data

    In 2025, **gradient boosted trees** (XGBoost, LightGBM, CatBoost) are *still* the best
    default for tabular data — despite a decade of deep learning advances. The reasons are structural:

    - **Handle mixed feature types natively** — trees split on thresholds, so numeric and categorical
      features coexist without normalization or encoding
    - **Capture nonlinear relationships and interactions automatically** — every split is a conditional;
      every branch adds an interaction term. Deep trees build high-order interactions for free.
    - **Robust to feature scaling** — no gradient magnitudes to balance; just threshold comparisons
    - **Fast to train and serve** — XGBoost on 1M rows in seconds; inference is a single tree traversal
    - **Built-in feature importance** — which features actually drive predictions?

    **When deep learning wins instead:** images, text, audio, sequences, very large datasets
    (100M+ rows), tasks that require *learning representations* from raw unstructured inputs.

    **Interview framing:** *"For tabular data, my first model is always XGBoost or LightGBM.
    I'd need a strong reason to reach for a neural net — and 'deep learning is fancier' isn't one."*
    """)
    return


# ─── Part 1: Decision Trees ───────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 1: Decision Trees — The Building Block

    A decision tree **recursively splits the feature space** into rectangular regions, assigning
    a prediction (majority class or mean) to each leaf.

    ### How it decides where to split

    At each node, search over all features and thresholds to find the split that maximizes
    **information gain** (classification) or minimizes **variance** (regression).

    **Gini impurity** — how "mixed" are the classes in this node?
    > Gini = 1 − Σ pᵢ²
    > Pure node → 0.  Max impurity (binary, 50/50) → 0.5.

    **Entropy / Information Gain** — measures disorder, then maximizes the *drop*:
    > Entropy = −Σ pᵢ log₂(pᵢ)

    **Variance reduction (regression)** — minimize MSE within each child node.

    **Gini vs Entropy:** results are nearly identical in practice. Gini is slightly faster
    (no logarithm). Most libraries default to Gini.

    ### When to stop splitting

    `max_depth`, `min_samples_leaf`, `min_impurity_decrease`. Without stopping criteria,
    the tree memorizes every training sample — perfect train accuracy, terrible generalization.
    """)
    return


@app.cell
def _dt_boundary():
    import numpy as _np
    import matplotlib as _matplotlib
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.tree import DecisionTreeClassifier as _DTC

    _X, _y = _make_moons(n_samples=300, noise=0.25, random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.2, random_state=42)

    _tree = _DTC(max_depth=3, random_state=42)
    _tree.fit(_X_tr, _y_tr)

    _x0_min, _x0_max = _X[:, 0].min() - 0.4, _X[:, 0].max() + 0.4
    _x1_min, _x1_max = _X[:, 1].min() - 0.4, _X[:, 1].max() + 0.4
    _xx, _yy = _np.meshgrid(_np.linspace(_x0_min, _x0_max, 300),
                             _np.linspace(_x1_min, _x1_max, 300))
    _Z = _tree.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _cmap_bg = _matplotlib.colors.ListedColormap(["#FADBD8", "#D6EAF8"])
    _cmap_pt = _matplotlib.colors.ListedColormap(["#E74C3C", "#3498DB"])

    _fig, _ax = _plt.subplots(figsize=(8, 5.5))
    _ax.contourf(_xx, _yy, _Z, alpha=0.4, cmap=_cmap_bg)
    _ax.contour(_xx, _yy, _Z, colors="k", linewidths=0.8, alpha=0.5)
    _ax.scatter(_X_tr[:, 0], _X_tr[:, 1], c=_y_tr, cmap=_cmap_pt,
                s=35, edgecolors="white", linewidths=0.5, label="Train")
    _ax.scatter(_X_te[:, 0], _X_te[:, 1], c=_y_te, cmap=_cmap_pt,
                s=55, marker="^", edgecolors="k", linewidths=0.8, label="Test")

    _test_acc = _tree.score(_X_te, _y_te)
    _ax.set_title(
        f"Decision Tree (max_depth=3) — Test Accuracy: {_test_acc:.1%}\n"
        "Each rectangular region = one leaf node.  Boundaries are always axis-aligned.",
        fontsize=11,
    )
    _ax.set_xlabel("Feature 0")
    _ax.set_ylabel("Feature 1")
    _ax.legend(fontsize=9, loc="upper right")
    _ax.grid(True, alpha=0.2)
    _fig.tight_layout()
    return _fig


@app.cell
def _tree_structure():
    import numpy as _np
    import matplotlib.pyplot as _plt

    from sklearn.datasets import make_moons as _make_moons
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.tree import DecisionTreeClassifier as _DTC, plot_tree as _plot_tree

    _X, _y = _make_moons(n_samples=300, noise=0.25, random_state=42)
    _X_tr, _, _y_tr, _ = _tts(_X, _y, test_size=0.2, random_state=42)

    _tree = _DTC(max_depth=3, random_state=42)
    _tree.fit(_X_tr, _y_tr)

    _fig, _ax = _plt.subplots(figsize=(15, 6))
    _plot_tree(
        _tree, ax=_ax,
        feature_names=["x₀", "x₁"],
        class_names=["Class 0", "Class 1"],
        filled=True, rounded=True, fontsize=9,
        impurity=True, proportion=False,
    )
    _ax.set_title(
        "Decision Tree Structure (max_depth=3)\n"
        "Each node: split condition · gini impurity · sample count · class distribution\n"
        "Each split in this tree creates one of the axis-aligned boundaries in the plot above.",
        fontsize=11, fontweight="bold",
    )
    _fig.tight_layout()
    return _fig


@app.cell
def _overfitting_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.tree import DecisionTreeClassifier as _DTC

    _X, _y = _make_moons(n_samples=300, noise=0.25, random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.2, random_state=42)

    _configs = [
        (2,    "max_depth=2\nUnderfitting — too shallow"),
        (5,    "max_depth=5\nGood fit"),
        (None, "max_depth=None\nOverfitting — memorized training data"),
    ]

    _x0_min, _x0_max = _X[:, 0].min() - 0.4, _X[:, 0].max() + 0.4
    _x1_min, _x1_max = _X[:, 1].min() - 0.4, _X[:, 1].max() + 0.4
    _xx, _yy = _np.meshgrid(_np.linspace(_x0_min, _x0_max, 300),
                             _np.linspace(_x1_min, _x1_max, 300))
    _cmap_bg = _matplotlib.colors.ListedColormap(["#FADBD8", "#D6EAF8"])
    _cmap_pt = _matplotlib.colors.ListedColormap(["#E74C3C", "#3498DB"])

    _fig, _axes = _plt.subplots(1, 3, figsize=(15, 5))

    for _ax, (_depth, _label) in zip(_axes, _configs):
        _tree = _DTC(max_depth=_depth, random_state=42)
        _tree.fit(_X_tr, _y_tr)
        _Z = _tree.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
        _tr_acc = _tree.score(_X_tr, _y_tr)
        _te_acc = _tree.score(_X_te, _y_te)
        _ax.contourf(_xx, _yy, _Z, alpha=0.4, cmap=_cmap_bg)
        _ax.contour(_xx, _yy, _Z, colors="k", linewidths=0.6, alpha=0.4)
        _ax.scatter(_X_tr[:, 0], _X_tr[:, 1], c=_y_tr, cmap=_cmap_pt,
                    s=22, edgecolors="white", linewidths=0.4)
        _ax.set_title(f"{_label}\nTrain: {_tr_acc:.1%}  Test: {_te_acc:.1%}", fontsize=10)
        _ax.set_xlabel("Feature 0")
        _ax.set_xlim(_x0_min, _x0_max)
        _ax.set_ylim(_x1_min, _x1_max)
        _ax.grid(True, alpha=0.2)

    _axes[0].set_ylabel("Feature 1")
    _fig.suptitle("Decision Tree Depth — From Underfitting to Overfitting",
                  fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    > **A single deep tree memorizes noise.** The unlimited-depth tree above reaches ~100% train
    > accuracy but performs worse on the test set — it has memorized labels, not learned the
    > pattern. This is the fundamental problem that random forests and boosting solve.

    ### Decision Tree: Strengths and Weaknesses

    **Strengths:**
    - **Fully interpretable** — trace any prediction from root to leaf; every split has a plain-language explanation
    - Handles nonlinearity naturally; no feature engineering required
    - No feature scaling needed — splits are threshold comparisons, not distances
    - Fast inference: O(depth) per prediction

    **Weaknesses:**
    - **High variance** — a small change in training data can produce an entirely different tree
    - Prone to overfitting without tight depth/leaf constraints
    - **Axis-aligned boundaries only** — can't efficiently represent a diagonal decision surface;
      requires many splits to approximate a simple line at 45°
    - Unstable: one mislabeled sample near a boundary can restructure the whole tree

    > *"A single decision tree is rarely used alone in production. It's the building block for ensembles."*
    """)
    return


# ─── Part 2: Random Forests ──────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 2: Random Forests — Variance Reduction Through Bagging

    **The insight:** a single tree is unstable (high variance). What if we build many trees and average them?

    ### Bagging (Bootstrap AGGregating)

    1. Draw B **bootstrap samples** — random samples *with replacement* from training data (each ~63% unique)
    2. Train one fully-grown tree on each bootstrap sample
    3. **Aggregate:** average predictions (regression) or majority vote (classification)

    ### Random Feature Subsampling

    At each split, only consider a random subset of features:
    - Classification: √n\_features
    - Regression: n\_features / 3

    **Why this matters:** without it, every tree would lead with the same strong features → correlated
    trees → averaging correlated trees barely reduces variance.  Forcing different feature subsets
    → decorrelated trees → much better variance reduction.

    ### The Math Intuition

    Variance of the mean of B variables with variance σ² and pairwise correlation ρ:
    > Var(mean) = ρσ² + (1−ρ)σ²/B

    As B → ∞ this approaches **ρσ²** — the correlation floor.
    Feature subsampling lowers ρ.  Lower ρ → lower ensemble variance.
    """)
    return


@app.cell
def _rf_vs_tree():
    import numpy as _np
    import matplotlib as _matplotlib
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.tree import DecisionTreeClassifier as _DTC
    from sklearn.ensemble import RandomForestClassifier as _RFC

    _X, _y = _make_moons(n_samples=300, noise=0.25, random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.2, random_state=42)

    _tree = _DTC(max_depth=None, random_state=42)
    _tree.fit(_X_tr, _y_tr)
    _rf = _RFC(n_estimators=100, random_state=42)
    _rf.fit(_X_tr, _y_tr)

    _x0_min, _x0_max = _X[:, 0].min() - 0.4, _X[:, 0].max() + 0.4
    _x1_min, _x1_max = _X[:, 1].min() - 0.4, _X[:, 1].max() + 0.4
    _xx, _yy = _np.meshgrid(_np.linspace(_x0_min, _x0_max, 300),
                             _np.linspace(_x1_min, _x1_max, 300))
    _cmap_bg = _matplotlib.colors.ListedColormap(["#FADBD8", "#D6EAF8"])
    _cmap_pt = _matplotlib.colors.ListedColormap(["#E74C3C", "#3498DB"])

    _fig, _axes = _plt.subplots(1, 2, figsize=(13, 5.5))
    _models = [
        (_tree, "Single Decision Tree (max_depth=None)"),
        (_rf,   "Random Forest (100 trees)"),
    ]

    for _ax, (_model, _title) in zip(_axes, _models):
        _Z = _model.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
        _te_acc = _model.score(_X_te, _y_te)
        _ax.contourf(_xx, _yy, _Z, alpha=0.4, cmap=_cmap_bg)
        _ax.contour(_xx, _yy, _Z, colors="k", linewidths=0.6, alpha=0.4)
        _ax.scatter(_X_tr[:, 0], _X_tr[:, 1], c=_y_tr, cmap=_cmap_pt,
                    s=25, edgecolors="white", linewidths=0.4)
        _ax.scatter(_X_te[:, 0], _X_te[:, 1], c=_y_te, cmap=_cmap_pt,
                    s=50, marker="^", edgecolors="k", linewidths=0.7)
        _ax.set_title(f"{_title}\nTest Accuracy: {_te_acc:.1%}", fontsize=11)
        _ax.set_xlabel("Feature 0")
        _ax.set_xlim(_x0_min, _x0_max)
        _ax.set_ylim(_x1_min, _x1_max)
        _ax.grid(True, alpha=0.2)

    _axes[0].set_ylabel("Feature 1")
    _fig.suptitle("Single Tree vs Random Forest: Smoother Boundaries, Better Generalization",
                  fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _variance_reduction():
    import numpy as _np
    import matplotlib as _matplotlib
    import matplotlib.pyplot as _plt
    from matplotlib.lines import Line2D as _Line2D
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.tree import DecisionTreeClassifier as _DTC
    from sklearn.ensemble import RandomForestClassifier as _RFC

    _X, _y = _make_moons(n_samples=300, noise=0.25, random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.2, random_state=42)
    _rng = _np.random.default_rng(0)

    _x0_min, _x0_max = _X[:, 0].min() - 0.4, _X[:, 0].max() + 0.4
    _x1_min, _x1_max = _X[:, 1].min() - 0.4, _X[:, 1].max() + 0.4
    _xx, _yy = _np.meshgrid(_np.linspace(_x0_min, _x0_max, 200),
                             _np.linspace(_x1_min, _x1_max, 200))
    _grid = _np.c_[_xx.ravel(), _yy.ravel()]

    _fig, _ax = _plt.subplots(figsize=(9, 6))

    # 10 individual trees on bootstrap samples — each different, showing variance
    for _i in range(10):
        _idx = _rng.choice(len(_X_tr), len(_X_tr), replace=True)
        _t = _DTC(max_depth=None, random_state=int(_i * 7))
        _t.fit(_X_tr[_idx], _y_tr[_idx])
        _Z_i = _t.predict(_grid).reshape(_xx.shape).astype(float)
        _ax.contour(_xx, _yy, _Z_i, levels=[0.5], colors=["#AAB7B8"],
                    linewidths=0.9, alpha=0.55)

    # Random forest — the smooth average of all those noisy boundaries
    _rf = _RFC(n_estimators=200, random_state=42)
    _rf.fit(_X_tr, _y_tr)
    _Z_rf = _rf.predict_proba(_grid)[:, 1].reshape(_xx.shape)
    _ax.contour(_xx, _yy, _Z_rf, levels=[0.5], colors=["#2C3E50"], linewidths=2.5)

    _cmap_pt = _matplotlib.colors.ListedColormap(["#E74C3C", "#3498DB"])
    _ax.scatter(_X_tr[:, 0], _X_tr[:, 1], c=_y_tr, cmap=_cmap_pt,
                s=22, edgecolors="white", linewidths=0.4, alpha=0.75, zorder=5)

    _handles = [
        _Line2D([0], [0], color="#AAB7B8", lw=1.5, alpha=0.7,
                label="10 individual trees (each on a different bootstrap sample)"),
        _Line2D([0], [0], color="#2C3E50", lw=2.5,
                label="Random Forest boundary (smooth average of all 10)"),
    ]
    _ax.legend(handles=_handles, fontsize=9, loc="upper right")
    _ax.set_title(
        "Variance Reduction Through Bagging\n"
        "Gray lines = individual tree boundaries — all different, showing high variance.\n"
        "Bold line = their smooth average.  Averaging decorrelated trees kills the noise.",
        fontsize=11,
    )
    _ax.set_xlabel("Feature 0")
    _ax.set_ylabel("Feature 1")
    _ax.set_xlim(_x0_min, _x0_max)
    _ax.set_ylim(_x1_min, _x1_max)
    _ax.grid(True, alpha=0.2)
    _fig.tight_layout()
    return _fig


@app.cell
def _oob_demo():
    import numpy as _np
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.ensemble import RandomForestClassifier as _RFC

    _X, _y = _make_moons(n_samples=300, noise=0.25, random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.2, random_state=42)

    _rf = _RFC(n_estimators=200, oob_score=True, random_state=42)
    _rf.fit(_X_tr, _y_tr)

    _oob = _rf.oob_score_
    _test_acc = _rf.score(_X_te, _y_te)

    _fig, _ax = _plt.subplots(figsize=(5, 3.5))
    _bars = _ax.bar(["OOB Score\n(free, no holdout)", "Test Accuracy\n(separate holdout)"],
                    [_oob, _test_acc],
                    color=["#3498DB", "#27AE60"], alpha=0.85, edgecolor="white", width=0.5)
    for _bar, _val in zip(_bars, [_oob, _test_acc]):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.005,
                 f"{_val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    _ax.set_ylim(0, 1.05)
    _ax.set_ylabel("Accuracy")
    _ax.set_title(f"OOB ≈ Test Accuracy  (diff = {abs(_oob - _test_acc):.4f})\n"
                  "OOB is a free cross-validation estimate computed during training",
                  fontsize=10)
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ### Out-of-Bag (OOB) Error

    Each bootstrap sample uses ~63% of the data (sampling with replacement means ~37% of
    samples are never drawn for a given tree). Those are "out of bag" — the tree never saw them,
    so we can evaluate predictions on them honestly.

    **Result:** a free validation estimate — no cross-validation or holdout set needed.
    OOB error ≈ leave-one-out cross-validation accuracy, computed during training at no extra cost.
    Enable with `RandomForestClassifier(oob_score=True)` → check `.oob_score_` after fitting.

    ### Random Forest: Strengths and Weaknesses

    **Strengths:**
    - Much lower variance than a single tree — averaging decorrelated trees is a powerful stabilizer
    - **Parallelizable** — each tree is independent; all 500 can train simultaneously
    - **OOB error for free** — honest validation estimate without a separate holdout
    - Robust to overfitting — more trees never hurts (unlike boosting)
    - Feature importance built in

    **Weaknesses:**
    - **Doesn't reduce bias** — if individual trees underfit, the forest still underfits.
      Bagging reduces variance, not bias. You need boosting for bias reduction.
    - Slower inference than a single tree (must traverse all N trees)
    - Less interpretable than a single tree — you lose the "explain every prediction" property

    > *"Random forests reduce variance but don't reduce bias. Boosting reduces bias."*
    """)
    return


# ─── Part 3: Gradient Boosting / XGBoost ─────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 3: Gradient Boosting / XGBoost — Bias Reduction Through Sequential Learning

    **The insight:** instead of building trees in *parallel* (bagging), build them *sequentially*
    where each new tree corrects the mistakes of all previous trees.

    ### Gradient Boosting Algorithm

    1. **Initialize:** start with a simple prediction — the mean of training targets
    2. **Compute residuals:** errors of the current ensemble on training data
    3. **Fit a shallow tree** to the residuals — this tree learns what the ensemble currently gets wrong
    4. **Add the tree** to the ensemble, scaled by learning rate η:
       > F(x) ← F(x) + η · h(x)
    5. **Repeat:** each new tree focuses on the remaining errors

    ### Why shallow trees?

    Deep trees overfit to residuals. Shallow trees (depth 3–6) are **weak learners** — each
    contributes a small, targeted correction. Many weak learners → strong learner.

    ### Learning Rate

    Shrinks each tree's contribution. Lower η → smaller steps → better generalization → more trees needed.
    Typical range: 0.01–0.3. Always pair a lower learning rate with more estimators.

    ### "Boosting is gradient descent in function space"

    At each step, we move in the direction that reduces the loss — exactly like gradient descent,
    but we're moving through the space of *functions*, not parameter vectors.
    For squared-error loss, the residuals ARE the negative gradient.

    **Regular gradient descent** updates parameters:
    > θ ← θ − α · ∇L(θ)

    The gradient tells you: *nudge each parameter in this direction to reduce loss.*

    **Gradient boosting's insight:** instead of parameters, you have *predictions* — one per training
    example. Think of the prediction vector `F = [F(x₁), F(x₂), ..., F(xₙ)]` as the thing being
    optimized. The gradient is now: *how should each prediction change to reduce loss?*

    > ∂L / ∂F(xᵢ)  ←  gradient w.r.t. a prediction, not a weight

    You move in function space by **adding a new tree** that approximates the negative gradient.

    **Why residuals = negative gradient for MSE:**

    With squared-error loss `L = Σ (yᵢ − F(xᵢ))²`:

    > ∂L / ∂F(xᵢ) = −2(yᵢ − F(xᵢ))  =  −2 · residualᵢ

    The negative gradient is proportional to the residual. Fitting each tree to residuals *is*
    gradient descent — you're chasing the steepest downhill direction in prediction space.

    **Why this matters for other losses:** for log-loss or MAE, the negative gradient is *not*
    the residual — it's something else. Gradient boosting still works (fit a tree to the negative
    gradient), but you can no longer call them residuals. Residuals are just the special case MSE
    gives you for free.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### XGBoost: Key Advantages Over Vanilla Gradient Boosting

    XGBoost keeps the core algorithm and adds:

    - **Regularization on leaf weights** — L1 (`reg_alpha`) and L2 (`reg_lambda`) penalties on
      leaf values; the trees themselves become regularized, not just the ensemble
    - **Learned missing value handling** — at each split, XGBoost learns the optimal direction
      for samples with missing values; no imputation needed
    - **Column subsampling** (`colsample_bytree`) — borrows the random forest idea; decorrelates
      trees and acts as additional regularization
    - **Row subsampling** (`subsample`) — use a random fraction of rows per tree; faster + regularizing
    - **Histogram-based splitting** (LightGBM extends this further) — bucketize continuous features
      into bins → faster split finding, smaller memory footprint
    - **System optimizations** — parallelized tree building within each level, cache-aware access,
      out-of-core computation for very large datasets

    > *"XGBoost = gradient boosting + regularization + engineering optimizations"*
    """)
    return


@app.cell
def _boosting_residuals():
    import numpy as _np
    import matplotlib.pyplot as _plt
    from sklearn.tree import DecisionTreeRegressor as _DTR

    _rng = _np.random.default_rng(42)
    _n = 150
    _x = _np.linspace(0, 6, _n)
    _y_true = _np.sin(_x) + 0.5 * _np.sin(2 * _x)
    _y = _y_true + _rng.normal(0, 0.2, _n)
    _X = _x.reshape(-1, 1)

    _x_plot = _np.linspace(0, 6, 400)
    _X_plot = _x_plot.reshape(-1, 1)
    _y_true_plot = _np.sin(_x_plot) + 0.5 * _np.sin(2 * _x_plot)

    _lr = 0.55
    _depth = 2

    # Ensemble state on training grid and plot grid
    _F = _np.full(_n, _y.mean())
    _F_plot = _np.full(400, _y.mean())

    # Snapshot: (title_suffix, F_plot_copy, residuals_on_train)
    _snapshots = [("Round 0: F(x) = mean(y)\nAll signal remains in residuals",
                   _F_plot.copy(), _y - _F)]

    for _rnd in range(1, 4):
        _resid = _y - _F
        _t = _DTR(max_depth=_depth, random_state=_rnd)
        _t.fit(_X, _resid)
        _F      += _lr * _t.predict(_X)
        _F_plot += _lr * _t.predict(_X_plot)
        _snapshots.append((
            f"After Round {_rnd}: +tree fitted to residuals\n"
            + ["Capturing the main shape", "Filling in finer structure",
               "Closely tracking the true function"][_rnd - 1],
            _F_plot.copy(), _y - _F,
        ))

    _fig, _axes = _plt.subplots(2, 2, figsize=(13, 9))

    for _i, (_ax, (_title, _fp, _resid)) in enumerate(zip(_axes.flat, _snapshots)):
        # Data points
        _ax.scatter(_x, _y, color="#BDC3C7", s=14, alpha=0.65, zorder=2, label="Training data")
        # True function
        _ax.plot(_x_plot, _y_true_plot, "k--", lw=1.5, alpha=0.55, label="True f(x)", zorder=3)
        # Ensemble prediction
        _ax.plot(_x_plot, _fp, color="#E74C3C", lw=2.5, label="Ensemble prediction", zorder=4)
        # Residuals as vertical orange lines (sparse for clarity)
        _f_at_x = _np.interp(_x, _x_plot, _fp)
        for _xi, _yi, _fi in zip(_x[::6], _y[::6], _f_at_x[::6]):
            _ax.plot([_xi, _xi], [_fi, _yi], color="#F39C12", lw=1.3, alpha=0.8, zorder=5)

        _rmse = float(_np.sqrt(_np.mean(_resid ** 2)))
        _ax.set_title(f"{_title}\nRMSE = {_rmse:.3f}", fontsize=10)
        _ax.set_xlabel("x", fontsize=9)
        _ax.set_ylabel("y", fontsize=9)
        _ax.set_xlim(0, 6)
        _ax.set_ylim(-2.3, 2.3)
        _ax.grid(True, alpha=0.2)
        if _i == 0:
            _ax.legend(fontsize=8, loc="upper right")

    _fig.suptitle(
        "Gradient Boosting: Each Tree Corrects What the Previous Ones Missed\n"
        "Orange lines = current residuals (what the next tree will be fitted to). "
        "Watch the RMSE shrink each round.",
        fontsize=12, fontweight="bold",
    )
    _fig.tight_layout()
    return _fig


@app.cell
def _boosting_overfitting():
    import numpy as _np
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.metrics import log_loss as _log_loss

    _X, _y = _make_moons(n_samples=500, noise=0.25, random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.25, random_state=42)

    _ns = list(range(1, 251, 4))
    _tr_losses, _te_losses = [], []

    try:
        import xgboost as _xgb
        for _n in _ns:
            _m = _xgb.XGBClassifier(n_estimators=_n, max_depth=4, learning_rate=0.3,
                                     eval_metric="logloss", verbosity=0, random_state=42)
            _m.fit(_X_tr, _y_tr)
            _tr_losses.append(_log_loss(_y_tr, _m.predict_proba(_X_tr)))
            _te_losses.append(_log_loss(_y_te, _m.predict_proba(_X_te)))
        _lib = "XGBoost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier as _GBC
        for _n in _ns:
            _m = _GBC(n_estimators=_n, max_depth=4, learning_rate=0.3, random_state=42)
            _m.fit(_X_tr, _y_tr)
            _tr_losses.append(_log_loss(_y_tr, _m.predict_proba(_X_tr)))
            _te_losses.append(_log_loss(_y_te, _m.predict_proba(_X_te)))
        _lib = "GradientBoosting (sklearn)"

    _best_idx = int(_np.argmin(_te_losses))
    _best_n   = _ns[_best_idx]
    _best_val = _te_losses[_best_idx]

    _fig, _ax = _plt.subplots(figsize=(10, 5.5))
    _ax.plot(_ns, _tr_losses, color="#27AE60", lw=2.5, label="Train Log-Loss")
    _ax.plot(_ns, _te_losses, color="#E74C3C", lw=2.5, label="Validation Log-Loss")
    _ax.axvline(_best_n, color="#F39C12", lw=2, linestyle="--")
    _ax.scatter([_best_n], [_best_val], color="#F39C12", s=130, zorder=6)
    _ax.annotate(f"Early stopping here\n(n_estimators ≈ {_best_n})",
                 xy=(_best_n, _best_val),
                 xytext=(_best_n + 18, _best_val + 0.05),
                 fontsize=10, color="#F39C12",
                 arrowprops=dict(arrowstyle="->", color="#F39C12"))
    _ax.axvspan(_best_n, max(_ns), alpha=0.07, color="#E74C3C")
    _ax.text(max(_ns) * 0.82, max(_te_losses) * 0.88,
             "Overfitting region\n(val loss rises)", fontsize=10, color="#C0392B",
             ha="center", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    _ax.set_xlabel("Number of Trees (n_estimators)", fontsize=12)
    _ax.set_ylabel("Log-Loss", fontsize=12)
    _ax.set_title(
        f"{_lib}: More Trees Can Overfit — Unlike Random Forests\n"
        "Use early_stopping_rounds to automatically find the optimal n_estimators",
        fontsize=11, fontweight="bold",
    )
    _ax.legend(fontsize=11)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ### Hyperparameter Guide — The 5 That Matter Most

    In priority order:

    | # | Parameter | Range | What it controls |
    |---|-----------|-------|-----------------|
    | 1 | `n_estimators` + `early_stopping_rounds` | 100–5000 | How many trees — let early stopping decide |
    | 2 | `learning_rate` | 0.01–0.3 | Step size; lower = better generalization but needs more trees |
    | 3 | `max_depth` | 3–8 | Individual tree complexity; start at 4–6 |
    | 4 | `subsample` | 0.7–0.9 | Row sampling fraction per tree — stochastic regularization |
    | 5 | `colsample_bytree` | 0.7–0.9 | Feature sampling per tree — like random forests |

    **Tuning strategy:**
    1. Start with defaults (`learning_rate=0.1`, `max_depth=6`, `subsample=0.8`)
    2. Add `early_stopping_rounds=50` on a validation set → optimal `n_estimators` for free
    3. Tune `learning_rate` + `n_estimators` together (lower LR → more trees)
    4. Then tune `max_depth`, `subsample`, `colsample_bytree`

    **L1/L2 regularization** (`reg_alpha`, `reg_lambda`) are useful fine-tuning levers but rarely
    the first knobs to turn.
    """)
    return


# ─── Part 4: Feature Importance ──────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 4: Feature Importance — Interpretability

    Three flavors, each with different tradeoffs:

    | Method | How | Speed | Bias | When to use |
    |--------|-----|-------|------|-------------|
    | **Split-based (impurity)** | Total impurity reduction across all splits on feature X | Fast (free from training) | Biased toward high-cardinality features | Quick feature screening |
    | **Permutation importance** | Shuffle feature X; measure accuracy drop | Moderate (N×features evaluations) | Unbiased; works for any model | Reliable global ranking |
    | **SHAP values** | Game-theoretic: each feature's marginal contribution to each prediction | Slow | Most rigorous; consistent and locally accurate | Explaining individual predictions; stakeholder reports |

    **When to use which:**
    - Quick feature screening → **split-based** (built into sklearn/xgboost)
    - Reliable global importance → **permutation importance**
    - Explaining individual predictions → **SHAP**
    - Production stakeholder reports → **SHAP** (most defensible; contributions sum to the prediction)
    """)
    return


@app.cell
def _importance_comparison():
    import numpy as _np
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_classification as _make_clf
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.ensemble import RandomForestClassifier as _RFC
    from sklearn.inspection import permutation_importance as _sk_perm

    _X, _y = _make_clf(n_samples=600, n_features=10, n_informative=4,
                        n_redundant=2, n_repeated=0, random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.25, random_state=42)
    _feat_names = [f"Feature {i}" for i in range(10)]

    _rf = _RFC(n_estimators=200, random_state=42)
    _rf.fit(_X_tr, _y_tr)

    try:
        import xgboost as _xgb
        _gb = _xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                   verbosity=0, random_state=42)
        _gb.fit(_X_tr, _y_tr)
        _gb_label = "XGBoost Split-Based"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier as _GBC
        _gb = _GBC(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        _gb.fit(_X_tr, _y_tr)
        _gb_label = "GradientBoosting Split-Based"

    _rf_split  = _rf.feature_importances_
    _perm      = _sk_perm(_rf, _X_te, _y_te, n_repeats=15, random_state=42).importances_mean
    _gb_split  = _gb.feature_importances_

    # Sort by RF split importance for a consistent visual ordering
    _order = _np.argsort(_rf_split)
    _labels = [_feat_names[i] for i in _order]

    _data = [
        (_rf_split[_order], "RF Split-Based (Gini Impurity)",  "#3498DB"),
        (_perm[_order],     "RF Permutation Importance",       "#E67E22"),
        (_gb_split[_order], _gb_label,                         "#27AE60"),
    ]

    _fig, _axes = _plt.subplots(1, 3, figsize=(15, 5.5))
    for _ax, (_imp, _title, _color) in zip(_axes, _data):
        _ax.barh(_labels, _imp, color=_color, alpha=0.82, edgecolor="white")
        _ax.set_title(_title, fontsize=11, fontweight="bold")
        _ax.set_xlabel("Importance Score", fontsize=9)
        _ax.axvline(0, color="k", lw=0.8)
        _ax.grid(True, axis="x", alpha=0.3)

    _fig.suptitle(
        "Feature Importance Methods — Do They Agree on the Top Features?\n"
        "(Dataset: 10 features, only 4 truly informative — those should rank highest)",
        fontsize=12, fontweight="bold",
    )
    _fig.tight_layout()
    return _fig


@app.cell
def _perm_from_scratch():
    import numpy as _np
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_classification as _make_clf
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.ensemble import RandomForestClassifier as _RFC
    from sklearn.metrics import accuracy_score as _acc
    from sklearn.inspection import permutation_importance as _sk_perm

    _X, _y = _make_clf(n_samples=600, n_features=10, n_informative=4,
                        n_redundant=2, random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.25, random_state=42)
    _feat_names = [f"Feature {i}" for i in range(10)]

    _rf = _RFC(n_estimators=200, random_state=42)
    _rf.fit(_X_tr, _y_tr)

    # ── Permutation importance from scratch ──────────────────────────────────
    # 1. Get baseline score on validation set
    # 2. For each feature: shuffle that column, recompute score, record the drop
    # 3. Larger drop = more important feature (model relied on it more)
    _rng = _np.random.default_rng(42)
    _baseline = _acc(_y_te, _rf.predict(_X_te))
    _n_repeats = 15
    _scratch_imp = _np.zeros(10)

    for _feat in range(10):
        _drops = []
        for _ in range(_n_repeats):
            _X_perm = _X_te.copy()
            _X_perm[:, _feat] = _rng.permutation(_X_perm[:, _feat])
            _drops.append(_baseline - _acc(_y_te, _rf.predict(_X_perm)))
        _scratch_imp[_feat] = float(_np.mean(_drops))

    # sklearn permutation importance — should match
    _sk_imp = _sk_perm(_rf, _X_te, _y_te, n_repeats=_n_repeats, random_state=42).importances_mean

    _order = _np.argsort(_scratch_imp)
    _labels = [_feat_names[i] for i in _order]

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 5))
    for _ax, (_imp, _title) in zip(
        _axes,
        [(_scratch_imp[_order], "From Scratch (manual loop)"),
         (_sk_imp[_order],      "sklearn permutation_importance")],
    ):
        _ax.barh(_labels, _imp,
                 color=["#E74C3C" if v > 0 else "#95A5A6" for v in _imp],
                 edgecolor="white", alpha=0.85)
        _ax.set_title(f"Permutation Importance\n{_title}", fontsize=11)
        _ax.set_xlabel("Mean accuracy drop when feature is shuffled", fontsize=9)
        _ax.axvline(0, color="k", lw=0.8)
        _ax.grid(True, axis="x", alpha=0.3)

    _fig.suptitle(
        "Permutation Importance: Scratch Implementation vs sklearn — Results Should Match\n"
        "Simple enough to implement in an interview if asked.",
        fontsize=11, fontweight="bold",
    )
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ### SHAP Values — The Gold Standard for Interpretability

    **SHAP** (SHapley Additive exPlanations) comes from cooperative game theory.

    **The question:** for a specific prediction, how much did each feature *contribute* to pushing
    the prediction away from the baseline (average prediction across all data)?

    **Key properties:**
    - Contributions **sum exactly** to (prediction − baseline) — they add up
    - **Consistent** — if feature X always has a larger effect in model A than B, SHAP ranks it higher in A
    - **Fair** — considers all possible orderings in which features could have been introduced

    **Two main views:**

    - **Force plot** (one prediction): shows which features pushed toward the positive vs negative
      class, and by how much. Color = direction; length = magnitude.
    - **Summary plot** (all predictions): shows feature importance (y-axis = feature, x-axis = SHAP value)
      AND direction of effect (feature value mapped to color). Tells you both *which* features matter
      and *how* they matter.

    **Usage:** `pip install shap`, then `shap.TreeExplainer(model)` for tree models
    (much faster than model-agnostic SHAP).

    > *"SHAP is the gold standard for model interpretability. If someone asks 'why did the model
    > predict this?' — SHAP is the answer."*
    """)
    return


# ─── When to Use What ────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## When to Use What — The Decision Framework

    | Scenario | Model | Why |
    |----------|-------|-----|
    | First model on tabular data | **XGBoost / LightGBM** | Best default; handles everything out of the box |
    | Need full interpretability | **Shallow decision tree** or logistic regression | Stakeholders can read every split |
    | Very small dataset (<1K rows) | **Random forest** | Less prone to overfitting than boosting |
    | Large dataset, need speed | **LightGBM** | Histogram-based splitting; fastest tree library |
    | Categorical features dominate | **CatBoost** | Native categorical handling; no encoding required |
    | Feature interactions unknown | **Any tree-based** | Trees capture interactions automatically |
    | Text / images / sequences | **Neural networks** | Trees can't learn representations from raw input |
    | Kaggle competition | **Stack XGBoost + LightGBM + CatBoost** | Diversity in ensemble wins competitions |
    | Strict inference latency | **Single shallow tree** or logistic regression | An ensemble of 500 trees adds up; trees aren't free |

    ---

    ### Bagging vs Boosting — Side-by-Side

    | | Random Forest (Bagging) | XGBoost (Boosting) |
    |--|------------------------|-------------------|
    | **Training** | Parallel — trees are independent | Sequential — each tree depends on all previous |
    | **Goal** | Reduce variance | Reduce bias (plus variance via regularization) |
    | **Individual trees** | Deep, fully grown | Shallow weak learners |
    | **Overfitting risk** | Low — more trees never hurts | Higher — too many trees overfits; use early stopping |
    | **Training speed** | Embarrassingly parallelizable | Sequential, though each level is parallelizable |
    | **When to prefer** | Small data, want stability, quick baseline | Large data, want maximum accuracy |
    """)
    return


# ─── Flashcard Summary ───────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Flashcard Summary

    | Question | Answer |
    |----------|--------|
    | How does a decision tree decide where to split? | Finds the feature + threshold that maximizes information gain (Gini or entropy reduction for classification, variance reduction for regression) |
    | Why do single decision trees overfit? | High variance — small data changes produce completely different trees. Deep trees memorize noise. |
    | What does a random forest add over a single tree? | Trains many trees on bootstrap samples with random feature subsets at each split, then averages. Reduces variance through decorrelation. |
    | Why does random forest use feature subsampling? | Without it, all trees lead with the same strong features → correlated trees → averaging them doesn't reduce variance much |
    | How does boosting differ from bagging? | Bagging trains trees independently in parallel (reduces variance). Boosting trains sequentially, each tree correcting the previous one's errors (reduces bias). |
    | What is the learning rate in XGBoost? | Shrinks each tree's contribution. Lower = more trees needed but better generalization. Prevents overfitting by taking smaller steps in function space. |
    | Why use early stopping for XGBoost but not random forests? | More boosting trees can overfit (sequential error correction overshoots). More bagging trees never hurts (averaging can only stabilize, not overfit). |
    | Name 3 types of feature importance | Split-based (fast, biased toward high cardinality), permutation (reliable, model-agnostic), SHAP (best, expensive — local and global) |
    | When would you use a neural net instead of XGBoost? | Non-tabular data (images, text, sequences), very large datasets (100M+ rows), tasks requiring representation learning from raw input |
    | What's XGBoost's key advantage over vanilla gradient boosting? | Regularization on leaf weights, native missing value handling, column/row subsampling, system-level speed optimizations |
    """)
    return


# ─── Interview Talking Points ─────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Interview Talking Points

    ---

    ### "Walk me through how gradient boosting works."

    > "Start with a simple prediction — typically the mean of the targets. Compute the residuals:
    > what the current model gets wrong. Fit a shallow tree to those residuals — it learns to
    > predict the errors, not the original labels. Add that tree to the ensemble, scaled by a
    > learning rate. Repeat: each new tree focuses on what all previous trees combined still miss.
    > Use early stopping on a validation set — too many trees and you overfit to the residuals."

    ---

    ### "Why would you pick XGBoost over a neural network?"

    > "For tabular data, trees almost always win. They handle mixed types without encoding,
    > capture nonlinear interactions automatically, train faster, and give you feature importance
    > for free. XGBoost adds regularization, missing value handling, and serious engineering
    > optimizations on top of that. I'd only reach for a neural net if I needed to learn
    > representations from raw text or images, or if my dataset was massive — 100M+ rows — where
    > deep learning's capacity starts to matter more."

    ---

    ### "How do you explain model predictions to stakeholders?"

    > "I use SHAP values. For a specific prediction, SHAP tells me exactly which features pushed
    > the output up or down and by how much — and those contributions sum to the prediction
    > itself, which makes them easy to explain. For global understanding I use SHAP summary plots,
    > which show which features matter most *and* in which direction. It's the most defensible
    > method because it's grounded in game theory."

    ---

    ### Connection to my work

    In my **fraud detection system design**, I chose XGBoost as the V1 model specifically for
    the interpretability requirement — regulators need to know *why* a transaction was declined.
    SHAP values generate human-readable reason codes directly from model output with no extra
    engineering.

    In my **backtesting engine**, feature importance helps distinguish which technical indicators
    actually drive strategy performance vs which are noise that happened to correlate in the
    training window — a key part of avoiding overfitting to historical data.
    """)
    return


if __name__ == "__main__":
    app.run()
