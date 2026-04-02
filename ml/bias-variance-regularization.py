import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


# ---------------------------------------------------------------------------
# 1. Header
# ---------------------------------------------------------------------------

@app.cell
def header(mo):
    mo.md(
        """
        # Bias-Variance Tradeoff, Overfitting & Regularization (L1/L2)

        | Field | Value |
        |-------|-------|
        | Date  | 2026-03-31 |
        | Track | ML Theory |
        | Time  | 60 min |
        | Topics | Bias-Variance Tradeoff · Overfitting · Ridge · Lasso · Elastic Net |
        """
    )
    return


# ---------------------------------------------------------------------------
# 2. Bias-Variance Tradeoff
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    mo.md(
        """
        ## Bias-Variance Tradeoff

        **Bias** is the error from a model that is too simple to capture the true underlying
        pattern — it systematically misses the signal (underfitting).  **Variance** is the error
        from a model that is so complex it latches onto the noise in the training data and fails
        to generalize (overfitting).  The tradeoff is a fundamental tension: as you increase model
        complexity to reduce bias, variance grows, and vice versa.  The goal is to find the sweet
        spot where *total* error (bias² + variance + irreducible noise) is minimized.
        """
    )
    return


@app.cell
def bias_variance_viz(mo):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rng = np.random.default_rng(42)

    # True underlying function: sin curve
    def true_fn(x):
        return np.sin(1.5 * x)

    # Generate noisy training data
    n = 25
    x_train = rng.uniform(-3, 3, n)
    y_train = true_fn(x_train) + rng.normal(0, 0.4, n)

    x_plot = np.linspace(-3.2, 3.2, 300)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Scatter: training points
    ax.scatter(x_train, y_train, color="#555555", s=35, zorder=5, label="Training data")

    # True function
    ax.plot(x_plot, true_fn(x_plot), "k--", lw=2, label="True function (unknown)")

    # Degree 1 — High Bias (underfit)
    c1 = np.polyfit(x_train, y_train, 1)
    ax.plot(x_plot, np.polyval(c1, x_plot), color="#E74C3C", lw=2.5,
            label="Degree 1 — High Bias (underfit)")

    # Degree 4 — Good fit
    c4 = np.polyfit(x_train, y_train, 4)
    ax.plot(x_plot, np.polyval(c4, x_plot), color="#2ECC71", lw=2.5,
            label="Degree 4 — Good Fit")

    # Degree 15 — High Variance (overfit)
    c15 = np.polyfit(x_train, y_train, 15)
    y15 = np.polyval(c15, x_plot)
    # Clip extreme oscillations for readability
    y15_clipped = np.clip(y15, -4, 4)
    ax.plot(x_plot, y15_clipped, color="#3498DB", lw=2.5,
            label="Degree 15 — High Variance (overfit)")

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Polynomial Fits: Bias vs Variance", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return mo.as_html(fig)


@app.cell
def ucurve_viz(mo):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complexity = np.linspace(0.5, 10, 300)

    # Bias² decreases as complexity grows (high-bias at low complexity)
    bias_sq = 4.0 / complexity**1.2

    # Variance increases as complexity grows
    variance = 0.08 * complexity**1.8

    # Total error = bias² + variance + irreducible noise
    irreducible = 0.3
    total = bias_sq + variance + irreducible

    sweet_idx = int(np.argmin(total))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(complexity, bias_sq, color="#E74C3C", lw=2.5, label="Bias²")
    ax.plot(complexity, variance, color="#3498DB", lw=2.5, label="Variance")
    ax.plot(complexity, total, color="#2C3E50", lw=3, label="Total Error (U-shape)")
    ax.axhline(irreducible, color="#95A5A6", lw=1.5, linestyle=":", label="Irreducible Noise")

    # Mark sweet spot
    ax.axvline(complexity[sweet_idx], color="#27AE60", lw=2, linestyle="--", alpha=0.8)
    ax.scatter([complexity[sweet_idx]], [total[sweet_idx]],
               color="#27AE60", s=120, zorder=6, label=f"Sweet Spot")
    ax.annotate("Sweet Spot\n(optimal complexity)",
                xy=(complexity[sweet_idx], total[sweet_idx]),
                xytext=(complexity[sweet_idx] + 1.2, total[sweet_idx] + 0.4),
                fontsize=10, color="#27AE60",
                arrowprops=dict(arrowstyle="->", color="#27AE60"))

    # Shade regions
    ax.axvspan(0.5, complexity[sweet_idx], alpha=0.05, color="#E74C3C",
               label="Underfitting region")
    ax.axvspan(complexity[sweet_idx], 10, alpha=0.05, color="#3498DB",
               label="Overfitting region")

    ax.text(1.5, 3.5, "High Bias\n(Underfit)", fontsize=10, color="#E74C3C",
            ha="center", style="italic")
    ax.text(8.5, 3.5, "High Variance\n(Overfit)", fontsize=10, color="#3498DB",
            ha="center", style="italic")

    ax.set_xlabel("Model Complexity", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Bias-Variance Decomposition: The Classic U-Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="upper center", fontsize=9, ncol=3)
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return mo.as_html(fig)


# ---------------------------------------------------------------------------
# 3. Overfitting Deep Dive
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    mo.md(
        """
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
        """
    )
    return


@app.cell
def train_val_viz(mo):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(7)
    epochs = np.arange(1, 81)

    # Train loss: smoothly decreasing
    train_loss = 1.0 * np.exp(-0.045 * epochs) + 0.05 + rng.normal(0, 0.008, len(epochs))
    train_loss = np.maximum(train_loss, 0.04)

    # Val loss: decreases then increases (overfitting kicks in ~epoch 35)
    val_noise = rng.normal(0, 0.018, len(epochs))
    val_loss = (0.9 * np.exp(-0.04 * epochs) + 0.22
                + 0.003 * np.maximum(epochs - 35, 0)**1.05
                + val_noise)

    # Early stopping: min val loss
    best_epoch = int(np.argmin(val_loss)) + 1
    best_val = val_loss[best_epoch - 1]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, train_loss, color="#27AE60", lw=2.5, label="Training Loss")
    ax.plot(epochs, val_loss, color="#E74C3C", lw=2.5, label="Validation Loss")

    # Early stopping vertical line
    ax.axvline(best_epoch, color="#F39C12", lw=2, linestyle="--")
    ax.scatter([best_epoch], [best_val], color="#F39C12", s=150, zorder=6)
    ax.annotate(f"Early Stopping\n(epoch {best_epoch})",
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 6, best_val + 0.08),
                fontsize=10, color="#F39C12",
                arrowprops=dict(arrowstyle="->", color="#F39C12"))

    # Overfitting region shading
    ax.axvspan(best_epoch, 80, alpha=0.08, color="#E74C3C")
    ax.text(60, 0.62, "Overfitting\nRegion", fontsize=11, color="#E74C3C",
            ha="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training vs Validation Loss — Detecting Overfitting", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return mo.as_html(fig)


# ---------------------------------------------------------------------------
# 4. Regularization
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    mo.md(
        """
        ## Regularization

        ### L2 Regularization (Ridge)

        Ridge adds a penalty proportional to the **sum of squared weights** to the loss:

        $$\\mathcal{L}_{ridge} = \\mathcal{L}_{base} + \\lambda \\sum_j w_j^2$$

        **Effect:** Weights are shrunk toward zero but rarely reach exactly zero — every feature
        retains a small influence.  **Geometric intuition:** The feasible weight region is a
        **sphere** (ellipse in 2D).  Loss contours typically meet the sphere at a smooth point
        off the axes, so no coefficient goes to exactly zero.  **When to use:** You believe most
        features carry real signal but want to prevent any single feature from dominating (e.g.
        collinear features, large-weight instability).  λ controls the strength: larger λ → more
        shrinkage → higher bias, lower variance.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### L1 Regularization (Lasso)

        Lasso adds a penalty proportional to the **sum of absolute weight values**:

        $$\\mathcal{L}_{lasso} = \\mathcal{L}_{base} + \\lambda \\sum_j |w_j|$$

        **Effect:** Drives some weights to *exactly* zero — this is built-in **feature selection**.
        **Geometric intuition:** The feasible region is a **diamond** (rhombus in 2D).  The corners
        of the diamond sit exactly on the axes (one coordinate = 0), and loss contours are likely
        to first touch the constraint region at one of those corners, zeroing out a coefficient.
        **When to use:** You suspect many features are irrelevant and want a sparse model that is
        easier to interpret.  Especially powerful in high-dimensional settings (p >> n).
        """
    )
    return


@app.cell
def constraint_region_viz(mo):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, title, shape in zip(axes, ["L2 (Ridge) — Sphere Constraint", "L1 (Lasso) — Diamond Constraint"],
                                 ["circle", "diamond"]):
        # Draw loss function contours centered off-origin (true minimum at w1=1.2, w2=0.9)
        w1 = np.linspace(-2.0, 2.0, 400)
        w2 = np.linspace(-2.0, 2.0, 400)
        W1, W2 = np.meshgrid(w1, w2)
        # Loss contours: ellipses centered at unconstrained optimum
        loss = 1.2 * (W1 - 1.2)**2 + 2.5 * (W2 - 0.9)**2
        levels = [0.3, 0.7, 1.2, 2.0, 3.2]
        cs = ax.contour(W1, W2, loss, levels=levels, colors="#95A5A6", linewidths=1.2,
                        linestyles="--", alpha=0.8)
        ax.clabel(cs, fmt="loss=%.1f", fontsize=7, colors="#7F8C8D")

        r = 0.95  # constraint radius / size

        if shape == "circle":
            theta = np.linspace(0, 2 * np.pi, 300)
            cx, cy = r * np.cos(theta), r * np.sin(theta)
            ax.fill(cx, cy, alpha=0.15, color="#3498DB")
            ax.plot(cx, cy, color="#3498DB", lw=2.5)
            # Intersection point — smooth, off-axis
            intersect = (0.62, 0.74)
            ax.scatter(*intersect, color="#E74C3C", s=180, zorder=7, label="Constrained optimum")
            ax.annotate("Optimum\n(off-axis → no zero)", xy=intersect,
                        xytext=(-1.5, 1.4), fontsize=9, color="#E74C3C",
                        arrowprops=dict(arrowstyle="->", color="#E74C3C"))

        else:  # diamond
            d_pts = np.array([[r, 0], [0, r], [-r, 0], [0, -r], [r, 0]])
            ax.fill(d_pts[:, 0], d_pts[:, 1], alpha=0.15, color="#E67E22")
            ax.plot(d_pts[:, 0], d_pts[:, 1], color="#E67E22", lw=2.5)
            # Intersection at a corner → sparse solution
            intersect = (0.95, 0.0)
            ax.scatter(*intersect, color="#E74C3C", s=180, zorder=7, label="Constrained optimum")
            ax.annotate("Optimum hits corner\n→ w₂ = 0 (sparse!)", xy=intersect,
                        xytext=(-0.3, 1.4), fontsize=9, color="#E74C3C",
                        arrowprops=dict(arrowstyle="->", color="#E74C3C"))
            # Label corners
            for cx, cy, lbl in [(r, 0, "(r, 0)"), (0, r, "(0, r)"), (-r, 0, "(-r, 0)"), (0, -r, "(0, -r)")]:
                ax.scatter(cx, cy, color="#E67E22", s=60, zorder=5)

        ax.axhline(0, color="black", lw=0.8, alpha=0.4)
        ax.axvline(0, color="black", lw=0.8, alpha=0.4)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel("w₁", fontsize=12)
        ax.set_ylabel("w₂", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Constraint Regions: The Geometric Intuition Behind L1 vs L2", fontsize=14, fontweight="bold")
    fig.tight_layout()

    return mo.as_html(fig)


@app.cell
def sklearn_demo(mo):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)

    X, y = make_regression(n_samples=200, n_features=20, n_informative=5,
                           noise=15, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit models
    lr = LinearRegression().fit(X_scaled, y)
    ridge = Ridge(alpha=10.0).fit(X_scaled, y)
    lasso = Lasso(alpha=5.0, max_iter=10000).fit(X_scaled, y)

    coefs = {
        "Linear\nRegression": lr.coef_,
        "Ridge\n(α=10)": ridge.coef_,
        "Lasso\n(α=5)": lasso.coef_,
    }

    n_features = 20
    x_pos = np.arange(n_features)
    colors = {"Linear\nRegression": "#95A5A6", "Ridge\n(α=10)": "#3498DB", "Lasso\n(α=5)": "#E67E22"}
    width = 0.28
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(13, 5))

    for (label, coef), offset in zip(coefs.items(), offsets):
        bars = ax.bar(x_pos + offset, coef, width=width, label=label,
                      color=colors[label], alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Feature Index", fontsize=12)
    ax.set_ylabel("Coefficient Value", fontsize=12)
    ax.set_title("L1 vs L2 vs No Regularization: Coefficient Comparison\n"
                 "(only 5 of 20 features are truly informative)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"f{i}" for i in range(n_features)], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    # Summary stats
    lr_nonzero = int(np.sum(np.abs(lr.coef_) > 0.01))
    ridge_nonzero = int(np.sum(np.abs(ridge.coef_) > 0.01))
    lasso_nonzero = int(np.sum(lasso.coef_ != 0))

    summary = (
        f"**Non-zero coefficients:**  "
        f"LinearRegression = {lr_nonzero}/20  ·  "
        f"Ridge = {ridge_nonzero}/20  ·  "
        f"**Lasso = {lasso_nonzero}/20** ← sparse!"
    )

    return mo.vstack([mo.as_html(fig), mo.md(summary)])


# ---------------------------------------------------------------------------
# 5. Elastic Net Bonus
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    mo.md(
        """
        ## Elastic Net (Bonus)

        **Elastic Net** combines L1 and L2 penalties:
        $$\\mathcal{L}_{EN} = \\mathcal{L}_{base} + \\lambda \\left[ \\text{l1\\_ratio} \\cdot \\sum|w_j| + \\frac{1 - \\text{l1\\_ratio}}{2} \\cdot \\sum w_j^2 \\right]$$

        The `l1_ratio` parameter blends the two: `1.0` = pure Lasso, `0.0` = pure Ridge.  Reach
        for Elastic Net when you want Lasso's sparsity but your features are **correlated** —
        Lasso tends to arbitrarily pick one correlated feature and zero the rest, while Elastic Net
        handles groups of correlated features more gracefully.
        """
    )
    return


# ---------------------------------------------------------------------------
# 6. Flashcard Summary
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    mo.md(
        """
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
        """
    )
    return


# ---------------------------------------------------------------------------
# 7. Interview Talking Points
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    mo.md(
        """
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
        """
    )
    return


if __name__ == "__main__":
    app.run()
