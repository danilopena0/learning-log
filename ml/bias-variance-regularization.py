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
    # Bias-Variance Tradeoff, Overfitting & Regularization (L1/L2)

    | Field | Value |
    |-------|-------|
    | Date  | 2026-03-31 |
    | Track | ML Theory |
    | Time  | 60 min |
    | Topics | Bias-Variance Tradeoff · Overfitting · Ridge · Lasso · Elastic Net |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Bias-Variance Tradeoff

    **Bias** is the error from a model that is too simple to capture the true underlying
    pattern — it systematically misses the signal (underfitting).  **Variance** is the error
    from a model that is so complex it latches onto the noise in the training data and fails
    to generalize (overfitting).  The tradeoff is a fundamental tension: as you increase model
    complexity to reduce bias, variance grows, and vice versa.  The goal is to find the sweet
    spot where *total* error (bias² + variance + irreducible noise) is minimized.
    """)
    return


@app.cell
def bias_variance_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _rng = _np.random.default_rng(42)

    def _true_fn(x):
        return _np.sin(1.5 * x)

    _n = 25
    _x_train = _rng.uniform(-3, 3, _n)
    _y_train = _true_fn(_x_train) + _rng.normal(0, 0.4, _n)
    _x_plot = _np.linspace(-3.2, 3.2, 300)

    _fig, _ax = _plt.subplots(figsize=(10, 5))

    _ax.scatter(_x_train, _y_train, color="#555555", s=35, zorder=5, label="Training data")
    _ax.plot(_x_plot, _true_fn(_x_plot), "k--", lw=2, label="True function (unknown)")

    _c1 = _np.polyfit(_x_train, _y_train, 1)
    _ax.plot(_x_plot, _np.polyval(_c1, _x_plot), color="#E74C3C", lw=2.5,
             label="Degree 1 — High Bias (underfit)")

    _c4 = _np.polyfit(_x_train, _y_train, 4)
    _ax.plot(_x_plot, _np.polyval(_c4, _x_plot), color="#2ECC71", lw=2.5,
             label="Degree 4 — Good Fit")

    _c15 = _np.polyfit(_x_train, _y_train, 15)
    _y15_clipped = _np.clip(_np.polyval(_c15, _x_plot), -4, 4)
    _ax.plot(_x_plot, _y15_clipped, color="#3498DB", lw=2.5,
             label="Degree 15 — High Variance (overfit)")

    _ax.set_xlim(-3.2, 3.2)
    _ax.set_ylim(-3, 3)
    _ax.set_xlabel("x", fontsize=12)
    _ax.set_ylabel("y", fontsize=12)
    _ax.set_title("Polynomial Fits: Bias vs Variance", fontsize=14, fontweight="bold")
    _ax.legend(loc="upper right", fontsize=10)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    return _fig


@app.cell
def ucurve_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _complexity = _np.linspace(0.5, 10, 300)
    _bias_sq = 4.0 / _complexity**1.2
    _variance = 0.08 * _complexity**1.8
    _irreducible = 0.3
    _total = _bias_sq + _variance + _irreducible
    _sweet_idx = int(_np.argmin(_total))

    _fig, _ax = _plt.subplots(figsize=(9, 5))

    _ax.plot(_complexity, _bias_sq, color="#E74C3C", lw=2.5, label="Bias²")
    _ax.plot(_complexity, _variance, color="#3498DB", lw=2.5, label="Variance")
    _ax.plot(_complexity, _total, color="#2C3E50", lw=3, label="Total Error (U-shape)")
    _ax.axhline(_irreducible, color="#95A5A6", lw=1.5, linestyle=":", label="Irreducible Noise")

    _ax.axvline(_complexity[_sweet_idx], color="#27AE60", lw=2, linestyle="--", alpha=0.8)
    _ax.scatter([_complexity[_sweet_idx]], [_total[_sweet_idx]],
                color="#27AE60", s=120, zorder=6, label="Sweet Spot")
    _ax.annotate("Sweet Spot\n(optimal complexity)",
                 xy=(_complexity[_sweet_idx], _total[_sweet_idx]),
                 xytext=(_complexity[_sweet_idx] + 1.2, _total[_sweet_idx] + 0.4),
                 fontsize=10, color="#27AE60",
                 arrowprops=dict(arrowstyle="->", color="#27AE60"))

    _ax.axvspan(0.5, _complexity[_sweet_idx], alpha=0.05, color="#E74C3C", label="Underfitting region")
    _ax.axvspan(_complexity[_sweet_idx], 10, alpha=0.05, color="#3498DB", label="Overfitting region")

    _ax.text(1.5, 3.5, "High Bias\n(Underfit)", fontsize=10, color="#E74C3C", ha="center", style="italic")
    _ax.text(8.5, 3.5, "High Variance\n(Overfit)", fontsize=10, color="#3498DB", ha="center", style="italic")

    _ax.set_xlabel("Model Complexity", fontsize=12)
    _ax.set_ylabel("Error", fontsize=12)
    _ax.set_title("Bias-Variance Decomposition: The Classic U-Curve", fontsize=14, fontweight="bold")
    _ax.legend(loc="upper center", fontsize=9, ncol=3)
    _ax.set_ylim(0, 5)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## Overfitting Deep Dive

    Overfitting shows up most clearly when you compare training and validation metrics over
    time.  **Signs to watch for:**

    - Training loss keeps dropping while validation loss plateaus or climbs — the model is
      memorizing training examples, not learning the underlying pattern.
    - A large gap between train accuracy (~99%) and val accuracy (~70%) in the same epoch.
    - Very large weight magnitudes — the model has learned spurious correlations that require
      extreme weights to encode.
    - Predictions are brittle: slightly perturbed inputs produce wildly different outputs.

    The practical fix is to monitor val loss and stop training at its minimum (**early stopping**).
    """)
    return


@app.cell
def train_val_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _rng = _np.random.default_rng(7)
    _epochs = _np.arange(1, 81)

    _train_loss = 1.0 * _np.exp(-0.045 * _epochs) + 0.05 + _rng.normal(0, 0.008, len(_epochs))
    _train_loss = _np.maximum(_train_loss, 0.04)

    _val_noise = _rng.normal(0, 0.018, len(_epochs))
    _val_loss = (0.9 * _np.exp(-0.04 * _epochs) + 0.22
                 + 0.003 * _np.maximum(_epochs - 35, 0)**1.05
                 + _val_noise)

    _best_epoch = int(_np.argmin(_val_loss)) + 1
    _best_val = _val_loss[_best_epoch - 1]

    _fig, _ax = _plt.subplots(figsize=(10, 5))

    _ax.plot(_epochs, _train_loss, color="#27AE60", lw=2.5, label="Training Loss")
    _ax.plot(_epochs, _val_loss, color="#E74C3C", lw=2.5, label="Validation Loss")

    _ax.axvline(_best_epoch, color="#F39C12", lw=2, linestyle="--")
    _ax.scatter([_best_epoch], [_best_val], color="#F39C12", s=150, zorder=6)
    _ax.annotate(f"Early Stopping\n(epoch {_best_epoch})",
                 xy=(_best_epoch, _best_val),
                 xytext=(_best_epoch + 6, _best_val + 0.08),
                 fontsize=10, color="#F39C12",
                 arrowprops=dict(arrowstyle="->", color="#F39C12"))

    _ax.axvspan(_best_epoch, 80, alpha=0.08, color="#E74C3C")
    _ax.text(60, 0.62, "Overfitting\nRegion", fontsize=11, color="#E74C3C",
             ha="center", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    _ax.set_xlabel("Epoch", fontsize=12)
    _ax.set_ylabel("Loss", fontsize=12)
    _ax.set_title("Training vs Validation Loss — Detecting Overfitting", fontsize=14, fontweight="bold")
    _ax.legend(fontsize=11)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## Regularization

    ### L2 Regularization (Ridge)

    Ridge adds a penalty proportional to the **sum of squared weights** to the loss:

    $$\mathcal{L}_{ridge} = \mathcal{L}_{base} + \lambda \sum_j w_j^2$$

    **Effect:** Weights are shrunk toward zero but rarely reach exactly zero — every feature
    retains a small influence.  **Geometric intuition:** The feasible weight region is a
    **sphere** (ellipse in 2D).  Loss contours typically meet the sphere at a smooth point
    off the axes, so no coefficient goes to exactly zero.  **When to use:** You believe most
    features carry real signal but want to prevent any single feature from dominating (e.g.
    collinear features, large-weight instability).  λ controls the strength: larger λ → more
    shrinkage → higher bias, lower variance.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### L1 Regularization (Lasso)

    Lasso adds a penalty proportional to the **sum of absolute weight values**:

    $$\mathcal{L}_{lasso} = \mathcal{L}_{base} + \lambda \sum_j |w_j|$$

    **Effect:** Drives some weights to *exactly* zero — this is built-in **feature selection**.
    **Geometric intuition:** The feasible region is a **diamond** (rhombus in 2D).  The corners
    of the diamond sit exactly on the axes (one coordinate = 0), and loss contours are likely
    to first touch the constraint region at one of those corners, zeroing out a coefficient.
    **When to use:** You suspect many features are irrelevant and want a sparse model that is
    easier to interpret.  Especially powerful in high-dimensional settings (p >> n).
    """)
    return


@app.cell
def constraint_region_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 5.5))

    for _ax, _title, _shape in zip(
        _axes,
        ["L2 (Ridge) — Sphere Constraint", "L1 (Lasso) — Diamond Constraint"],
        ["circle", "diamond"],
    ):
        _w1 = _np.linspace(-2.0, 2.0, 400)
        _w2 = _np.linspace(-2.0, 2.0, 400)
        _W1, _W2 = _np.meshgrid(_w1, _w2)
        _loss = 1.2 * (_W1 - 1.2)**2 + 2.5 * (_W2 - 0.9)**2
        _levels = [0.3, 0.7, 1.2, 2.0, 3.2]
        _cs = _ax.contour(_W1, _W2, _loss, levels=_levels, colors="#95A5A6",
                          linewidths=1.2, linestyles="--", alpha=0.8)
        _ax.clabel(_cs, fmt="loss=%.1f", fontsize=7, colors="#7F8C8D")

        _r = 0.95

        if _shape == "circle":
            _theta = _np.linspace(0, 2 * _np.pi, 300)
            _cx, _cy = _r * _np.cos(_theta), _r * _np.sin(_theta)
            _ax.fill(_cx, _cy, alpha=0.15, color="#3498DB")
            _ax.plot(_cx, _cy, color="#3498DB", lw=2.5)
            _intersect = (0.62, 0.74)
            _ax.scatter(*_intersect, color="#E74C3C", s=180, zorder=7, label="Constrained optimum")
            _ax.annotate("Optimum\n(off-axis → no zero)", xy=_intersect,
                         xytext=(-1.5, 1.4), fontsize=9, color="#E74C3C",
                         arrowprops=dict(arrowstyle="->", color="#E74C3C"))
        else:
            _d_pts = _np.array([[_r, 0], [0, _r], [-_r, 0], [0, -_r], [_r, 0]])
            _ax.fill(_d_pts[:, 0], _d_pts[:, 1], alpha=0.15, color="#E67E22")
            _ax.plot(_d_pts[:, 0], _d_pts[:, 1], color="#E67E22", lw=2.5)
            _intersect = (0.95, 0.0)
            _ax.scatter(*_intersect, color="#E74C3C", s=180, zorder=7, label="Constrained optimum")
            _ax.annotate("Optimum hits corner\n→ w₂ = 0 (sparse!)", xy=_intersect,
                         xytext=(-0.3, 1.4), fontsize=9, color="#E74C3C",
                         arrowprops=dict(arrowstyle="->", color="#E74C3C"))
            for _px, _py in [(_r, 0), (0, _r), (-_r, 0), (0, -_r)]:
                _ax.scatter(_px, _py, color="#E67E22", s=60, zorder=5)

        _ax.axhline(0, color="black", lw=0.8, alpha=0.4)
        _ax.axvline(0, color="black", lw=0.8, alpha=0.4)
        _ax.set_xlim(-2, 2)
        _ax.set_ylim(-2, 2)
        _ax.set_xlabel("w₁", fontsize=12)
        _ax.set_ylabel("w₂", fontsize=12)
        _ax.set_title(_title, fontsize=13, fontweight="bold")
        _ax.legend(fontsize=9)
        _ax.set_aspect("equal")
        _ax.grid(True, alpha=0.2)

    _fig.suptitle("Constraint Regions: The Geometric Intuition Behind L1 vs L2",
                  fontsize=14, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def sklearn_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_regression as _make_regression
    from sklearn.linear_model import LinearRegression as _LR, Ridge as _Ridge, Lasso as _Lasso
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    _X, _y = _make_regression(n_samples=200, n_features=20, n_informative=5,
                               noise=15, random_state=42)
    _X_scaled = _StandardScaler().fit_transform(_X)

    _lr = _LR().fit(_X_scaled, _y)
    _ridge = _Ridge(alpha=10.0).fit(_X_scaled, _y)
    _lasso = _Lasso(alpha=5.0, max_iter=10000).fit(_X_scaled, _y)

    _coefs = {
        "Linear\nRegression": _lr.coef_,
        "Ridge\n(α=10)": _ridge.coef_,
        "Lasso\n(α=5)": _lasso.coef_,
    }
    _colors = {"Linear\nRegression": "#95A5A6", "Ridge\n(α=10)": "#3498DB", "Lasso\n(α=5)": "#E67E22"}
    _n_features = 20
    _x_pos = _np.arange(_n_features)
    _width = 0.28
    _offsets = [-_width, 0, _width]

    _fig, _ax = _plt.subplots(figsize=(13, 5))

    for (_label, _coef), _offset in zip(_coefs.items(), _offsets):
        _ax.bar(_x_pos + _offset, _coef, width=_width, label=_label,
                color=_colors[_label], alpha=0.85, edgecolor="white")

    _ax.axhline(0, color="black", lw=1)
    _ax.set_xlabel("Feature Index", fontsize=12)
    _ax.set_ylabel("Coefficient Value", fontsize=12)
    _ax.set_title("L1 vs L2 vs No Regularization: Coefficient Comparison\n"
                  "(only 5 of 20 features are truly informative)",
                  fontsize=13, fontweight="bold")
    _ax.set_xticks(_x_pos)
    _ax.set_xticklabels([f"f{i}" for i in range(_n_features)], fontsize=9)
    _ax.legend(fontsize=11)
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()

    _lr_nz = int(_np.sum(_np.abs(_lr.coef_) > 0.01))
    _ridge_nz = int(_np.sum(_np.abs(_ridge.coef_) > 0.01))
    _lasso_nz = int(_np.sum(_lasso.coef_ != 0))

    _summary = (
        f"**Non-zero coefficients:**  "
        f"LinearRegression = {_lr_nz}/20  ·  "
        f"Ridge = {_ridge_nz}/20  ·  "
        f"**Lasso = {_lasso_nz}/20** ← sparse!"
    )
    return _fig, _summary


@app.cell
def _(mo):
    mo.md("""
    ## Elastic Net (Bonus)

    **Elastic Net** combines L1 and L2 penalties:
    $$\mathcal{L}_{EN} = \mathcal{L}_{base} + \lambda \left[ \text{l1\_ratio} \cdot \sum|w_j| + \frac{1 - \text{l1\_ratio}}{2} \cdot \sum w_j^2 \right]$$

    The `l1_ratio` parameter blends the two: `1.0` = pure Lasso, `0.0` = pure Ridge.  Reach
    for Elastic Net when you want Lasso's sparsity but your features are **correlated** —
    Lasso tends to arbitrarily pick one correlated feature and zero the rest, while Elastic Net
    handles groups of correlated features more gracefully.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Flashcard-Style Summary

    | Question | Answer |
    |----------|--------|
    | What is bias? | Error from a model that is too simple to capture the true pattern — systematic underfitting. |
    | What is variance? | Error from a model so complex it fits noise in the training set and fails to generalize. |
    | What happens as model complexity increases? | Bias decreases, variance increases; total error follows a U-shape with an optimal sweet spot. |
    | How does L1 differ from L2? | L1 adds an absolute-value penalty (can zero weights → sparse); L2 adds a squared penalty (shrinks weights but keeps all non-zero). |
    | Why does L1 produce sparse solutions? | The L1 constraint region is a diamond; loss contours tend to first touch it at a corner where one or more weights are exactly zero. |
    | When would you choose Ridge over Lasso? | When you believe most features are relevant and just want to prevent large weights — especially with correlated features. |
    | What is the bias-variance tradeoff? | Reducing bias (more complex model) increases variance and vice versa; the goal is to minimize total error at the optimal complexity. |
    | How do you detect overfitting? | Training loss keeps decreasing while validation loss plateaus or rises; large train/val performance gap; very large weight magnitudes. |
    | What does the regularization parameter λ control? | The strength of the penalty: larger λ → stronger regularization → more shrinkage → higher bias, lower variance. |
    | What's Elastic Net? | A combination of L1 and L2 regularization controlled by l1_ratio; balances sparsity (L1) with correlated-feature stability (L2). |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    ### "Walk me through the bias-variance tradeoff."

    > "Every model makes a bias-variance tradeoff.  A model with high bias is too simple —
    > it consistently misses the signal and underfits.  A model with high variance is too
    > complex — it memorizes noise and fails on new data.  As you increase complexity, bias
    > drops but variance rises, so total error forms a U-shape.  The optimal model sits at
    > the bottom of that U.  In practice you tune this via model selection (polynomial degree,
    > tree depth, number of layers) and regularization."

    ---

    ### "How do you prevent overfitting?"

    1. **More data** — the single most effective lever when available.
    2. **Regularization** — L1/L2 penalize complexity directly in the loss.
    3. **Dropout** (neural nets) — randomly deactivates neurons during training.
    4. **Early stopping** — halt training when validation loss starts rising.
    5. **Cross-validation** — use k-fold CV so your metric reflects generalization, not memorization.
    6. **Simpler model** — reduce depth, features, or parameters.

    ---

    ### "Explain L1 vs L2 regularization."

    > "Both add a penalty term to the loss function that discourages large weights.  L2 (Ridge)
    > penalizes the *sum of squared weights* — it shrinks all weights toward zero but never
    > exactly zero.  L1 (Lasso) penalizes the *sum of absolute weights* — it can drive weights
    > to exactly zero, giving built-in feature selection.  Geometrically, L2's constraint region
    > is a sphere so the loss contour meets it smoothly; L1's region is a diamond and contours
    > tend to hit the corners, which sit on an axis — zeroing a coefficient."

    ---

    ### Common follow-up: "When would you NOT use regularization?"

    > "When your model is already simple relative to the data — for instance, a linear model
    > on a small dataset with only a handful of features.  Adding regularization in that
    > scenario increases bias without meaningfully reducing variance, and you may end up
    > with a worse model.  Also, if you have a huge amount of data, regularization matters
    > less because variance is naturally controlled by sample size."
    """)
    return


if __name__ == "__main__":
    app.run()
