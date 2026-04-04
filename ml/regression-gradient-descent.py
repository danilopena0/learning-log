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
    # Linear & Logistic Regression from Scratch
    ## Gradient Descent · Loss Functions · Decision Boundaries

    | Field  | Value |
    |--------|-------|
    | Date   | 2026-04-03 |
    | Track  | ML Theory |
    | Time   | 60 min |
    | Topics | Linear Regression · Logistic Regression · Gradient Descent · MSE · Binary Cross-Entropy · Decision Boundaries |
    """)
    return


# ── Part 1: Linear Regression ────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Part 1: Linear Regression from Scratch

    **What it does:** Finds the line (or hyperplane) that minimizes the sum of squared errors
    between predictions and targets.

    **The model:** $y = Xw + b$
    (equivalently $y = \tilde{X}\tilde{w}$ with bias absorbed into the design matrix,
    $\tilde{X} = [X \mid \mathbf{1}]$, $\tilde{w} = [w;\, b]$)

    **Why "linear"?** Linear in the *parameters* — the output is a weighted sum of features.
    Features can be nonlinear ($x^2$, $\sin x$, $\log x$) and the model is still "linear."
    Interviewers probe this distinction; it is worth having a crisp answer ready.
    """)
    return


@app.cell
def generate_linreg_data():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _rng = _np.random.default_rng(42)
    _n = 100

    # True parameters we will try to recover
    w_true = 2.5
    b_true = 0.8

    # x ~ Uniform(-3, 3), noise ~ N(0, 0.8)
    X_linreg = _rng.uniform(-3.0, 3.0, _n).reshape(-1, 1)
    y_linreg = w_true * X_linreg.ravel() + b_true + _rng.normal(0, 0.8, _n)

    # "Before" scatter — raw data, no model fitted
    _fig, _ax = _plt.subplots(figsize=(8, 5))
    _ax.scatter(
        X_linreg, y_linreg,
        color="#3498DB", alpha=0.65, s=45,
        edgecolors="white", linewidths=0.5,
        label="Observed data  (n=100)",
    )
    _ax.set_xlabel("x", fontsize=12)
    _ax.set_ylabel("y", fontsize=12)
    _ax.set_title(
        "Synthetic Regression Data — Before Fitting",
        fontsize=13, fontweight="bold",
    )
    _ax.legend(fontsize=11)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    return _fig, X_linreg, y_linreg, w_true, b_true


@app.cell
def _(mo):
    mo.md(r"""
    ### Loss Function: Mean Squared Error (MSE)

    $$L = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

    **Why squared?**
    - Penalizes large errors disproportionately — a $2\times$ error costs $4\times$ as much
    - Differentiable everywhere; gradient descent can follow it all the way to the minimum
    - For linear models, MSE is **strictly convex** — one global minimum, no local traps
    - Without squaring, positive and negative residuals cancel; the model could be wildly wrong

    **Gradients** (direction of steepest *ascent* in loss — gradient descent steps opposite):

    $$\frac{\partial L}{\partial w} = \frac{2}{n}\, X^T (Xw + b - y)$$

    $$\frac{\partial L}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (X_i w + b - y_i)$$
    """)
    return


@app.cell
def linreg_gd(X_linreg, y_linreg, w_true, b_true):
    import numpy as _np

    # ── MSE loss: L = (1/n) * ||Xw + b - y||² ────────────────────────────────
    def _mse(X, y, w, b):
        _pred = X.ravel() * w + b
        return float(_np.mean((_pred - y) ** 2))

    # ── MSE gradients ─────────────────────────────────────────────────────────
    def _grad_mse(X, y, w, b):
        _n = len(y)
        _res = X.ravel() * w + b - y         # residuals, shape (n,)
        _dw = (2.0 / _n) * float(_np.dot(X.ravel(), _res))
        _db = (2.0 / _n) * float(_np.sum(_res))
        return _dw, _db

    # ── Gradient descent: init at 0, lr=0.01, 1000 iterations ────────────────
    _lr = 0.01
    _n_iter = 1000
    w_gd = 0.0
    b_gd = 0.0
    loss_history_linreg = []

    for _i in range(_n_iter):
        loss_history_linreg.append(_mse(X_linreg, y_linreg, w_gd, b_gd))
        _dw, _db = _grad_mse(X_linreg, y_linreg, w_gd, b_gd)
        w_gd = w_gd - _lr * _dw
        b_gd = b_gd - _lr * _db

    print(f"Gradient descent ({_n_iter} iter, lr={_lr})")
    print(f"  Learned:  w = {w_gd:.4f}  b = {b_gd:.4f}")
    print(f"  True:     w = {w_true:.4f}  b = {b_true:.4f}")
    print(f"  Abs err:  w = {abs(w_gd - w_true):.4f}  b = {abs(b_gd - b_true):.4f}")

    return w_gd, b_gd, loss_history_linreg


@app.cell
def linreg_training_viz(
    X_linreg, y_linreg, w_gd, b_gd, loss_history_linreg, w_true, b_true
):
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _losses = _np.array(loss_history_linreg)
    _iters = _np.arange(len(_losses))
    _final_loss = float(_losses[-1])

    # First iteration where loss is within 5% of the final (converged) value
    _conv_idx = int(_np.argmax(_losses < 1.05 * _final_loss))
    _conv_idx = max(_conv_idx, 1)

    _x_plot = _np.linspace(-3.2, 3.2, 300)

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: loss curve ──────────────────────────────────────────────────────
    _ax1.plot(_iters, _losses, color="#2C3E50", lw=2, label="MSE loss")
    _ax1.axvline(_conv_idx, color="#27AE60", lw=1.8, linestyle="--", alpha=0.85)
    _ax1.scatter([_conv_idx], [_losses[_conv_idx]], color="#27AE60", s=110, zorder=6)
    _annot_x = min(_conv_idx + 80, len(_losses) - 100)
    _ax1.annotate(
        f"~Converged  (iter {_conv_idx})",
        xy=(_conv_idx, _losses[_conv_idx]),
        xytext=(_annot_x, _losses[_conv_idx] + 0.8),
        fontsize=9, color="#27AE60",
        arrowprops=dict(arrowstyle="->", color="#27AE60"),
    )
    _ax1.set_xlabel("Iteration", fontsize=12)
    _ax1.set_ylabel("MSE Loss", fontsize=12)
    _ax1.set_title("Loss Curve — Gradient Descent", fontsize=13, fontweight="bold")
    _ax1.legend(fontsize=10)
    _ax1.grid(True, alpha=0.3)

    # ── Right: data + regression lines ────────────────────────────────────────
    _ax2.scatter(
        X_linreg, y_linreg,
        color="#3498DB", alpha=0.5, s=40,
        edgecolors="white", linewidths=0.4,
        label="Data", zorder=2,
    )
    _ax2.plot(
        _x_plot, w_gd * _x_plot + b_gd,
        color="#E74C3C", lw=2.5, zorder=3,
        label=f"Learned:  y = {w_gd:.2f}x + {b_gd:.2f}",
    )
    _ax2.plot(
        _x_plot, w_true * _x_plot + b_true,
        color="#27AE60", lw=2, linestyle="--", zorder=3,
        label=f"True:  y = {w_true}x + {b_true}",
    )
    _ax2.set_xlabel("x", fontsize=12)
    _ax2.set_ylabel("y", fontsize=12)
    _ax2.set_title("Learned vs True Regression Line", fontsize=13, fontweight="bold")
    _ax2.legend(fontsize=10)
    _ax2.grid(True, alpha=0.3)

    _fig.suptitle("Gradient Descent — Training Results", fontsize=15, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def linreg_lr_exploration(X_linreg, y_linreg):
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    def _mse(X, y, w, b):
        return float(_np.mean((X.ravel() * w + b - y) ** 2))

    def _grad_mse(X, y, w, b):
        _n = len(y)
        _res = X.ravel() * w + b - y
        return (
            (2.0 / _n) * float(_np.dot(X.ravel(), _res)),
            (2.0 / _n) * float(_np.sum(_res)),
        )

    _lr_configs = [
        (0.001, "#3498DB", "lr=0.001 — too small"),
        (0.01,  "#27AE60", "lr=0.010 — just right"),
        (0.5,   "#E74C3C", "lr=0.500 — too large"),
    ]
    _n_iter = 300
    _cap = 120.0          # cap diverging curve for readability

    _all_hists = []
    for _lr_val, _color, _label in _lr_configs:
        _w, _b = 0.0, 0.0
        _hist = []
        for _ in range(_n_iter):
            _l = _mse(X_linreg, y_linreg, _w, _b)
            _hist.append(min(_l, _cap))
            _dw, _db = _grad_mse(X_linreg, y_linreg, _w, _b)
            if _np.isfinite(_dw) and _np.isfinite(_db):
                _w = _w - _lr_val * _dw
                _b = _b - _lr_val * _db
        _all_hists.append(_hist)

    _hist_small, _hist_good, _hist_large = _all_hists

    _fig, _ax = _plt.subplots(figsize=(10, 5.5))

    for (_lr_val, _color, _label), _hist in zip(_lr_configs, _all_hists):
        _ax.plot(_hist, color=_color, lw=2.5, label=f"  {_label}", alpha=0.9)

    # Annotations: show the interviewer exactly what each regime looks like
    _ax.annotate(
        "Too small:\nnever reaches minimum\nin budget iterations",
        xy=(285, _hist_small[285]),
        xytext=(200, _hist_small[285] + 12),
        fontsize=9, color="#3498DB",
        arrowprops=dict(arrowstyle="->", color="#3498DB"),
    )
    _ax.annotate(
        "Just right:\nsmooth, reliable convergence",
        xy=(70, _hist_good[70]),
        xytext=(80, _hist_good[70] + 14),
        fontsize=9, color="#27AE60",
        arrowprops=dict(arrowstyle="->", color="#27AE60"),
    )
    _ax.annotate(
        "Too large:\noscillates, never converges\n(capped at 120 for scale)",
        xy=(10, _hist_large[10]),
        xytext=(35, _hist_large[10] + 8),
        fontsize=9, color="#E74C3C",
        arrowprops=dict(arrowstyle="->", color="#E74C3C"),
    )

    _ax.set_ylim(0, _cap + 10)
    _ax.set_xlabel("Iteration", fontsize=12)
    _ax.set_ylabel("MSE Loss  (capped at 120)", fontsize=12)
    _ax.set_title(
        "Learning Rate Comparison — Too Small · Just Right · Too Large",
        fontsize=13, fontweight="bold",
    )
    _ax.legend(fontsize=11, loc="upper right")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md(r"""
    ### Closed-Form Solution: The Normal Equation

    For linear regression with MSE loss, there is an exact analytic solution:

    $$\hat{w} = (X^T X)^{-1} X^T y$$

    In code, one line of NumPy suffices:

    ```python
    X_aug = np.column_stack([X, np.ones(n)])        # absorb bias
    w_cf  = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
    ```

    **Why not always use the closed form?**
    - Matrix inversion is $O(d^3)$ in number of features $d$ — catastrophic at scale
      (GPT-level models have billions of parameters; inverting that matrix is infeasible)
    - Requires the full data in memory at once
    - Does not generalize to non-MSE losses or non-linear models
    - **Gradient descent is $O(d)$ per step** and works for any differentiable loss
    """)
    return


@app.cell
def linreg_closed_form(X_linreg, y_linreg, w_gd, b_gd, w_true, b_true):
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    # Normal equation — one line of numpy
    _n = len(y_linreg)
    _X_aug = _np.column_stack([X_linreg, _np.ones(_n)])        # (n, 2)
    _w_cf = _np.linalg.solve(_X_aug.T @ _X_aug, _X_aug.T @ y_linreg)
    _w_closed = float(_w_cf[0])
    _b_closed = float(_w_cf[1])

    # Comparison bar chart: true vs GD vs closed-form
    _methods = ["True", "Gradient\nDescent", "Normal\nEquation"]
    _w_vals = [w_true, w_gd, _w_closed]
    _b_vals = [b_true, b_gd, _b_closed]
    _colors = ["#27AE60", "#E74C3C", "#F39C12"]
    _x_pos = _np.arange(3)
    _width = 0.35

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(11, 5))

    for _ax, _vals, _param in zip((_ax1, _ax2), (_w_vals, _b_vals), ("w  (slope)", "b  (intercept)")):
        _bars = _ax.bar(_x_pos, _vals, width=_width + 0.1, color=_colors, alpha=0.85, edgecolor="white")
        for _bar, _v in zip(_bars, _vals):
            _ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                _bar.get_height() + 0.02,
                f"{_v:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        _ax.set_xticks(_x_pos)
        _ax.set_xticklabels(_methods, fontsize=11)
        _ax.set_ylabel(_param, fontsize=12)
        _ax.set_title(f"Parameter: {_param}", fontsize=12, fontweight="bold")
        _ax.grid(True, axis="y", alpha=0.3)
        _ax.set_ylim(0, max(_vals) * 1.25)

    _fig.suptitle(
        "Closed-Form vs Gradient Descent — Both Match the True Parameters",
        fontsize=13, fontweight="bold",
    )
    _fig.tight_layout()

    print(f"Normal equation:   w = {_w_closed:.4f}  b = {_b_closed:.4f}")
    print(f"Gradient descent:  w = {w_gd:.4f}  b = {b_gd:.4f}")
    print(f"True parameters:   w = {w_true:.4f}  b = {b_true:.4f}")

    return _fig


# ── Part 2: Logistic Regression ──────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Part 2: Logistic Regression from Scratch

    **What it does:** Classification by modeling $P(y=1 \mid x)$ using the sigmoid function.

    **Key insight:** Logistic regression is linear regression passed through a sigmoid to
    squash outputs to $[0, 1]$, giving interpretable probabilities.

    $$P(y=1 \mid x) = \sigma(Xw + b) \qquad \text{where} \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

    **Despite the name, logistic regression is a *classifier*, not a regressor.**
    Interviewers will ask this — be ready to explain why: the output is a probability (0–1),
    and we threshold at 0.5 to produce a binary decision.
    """)
    return


@app.cell
def sigmoid_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _z = _np.linspace(-10, 10, 500)
    _sig = 1 / (1 + _np.exp(-_z))

    _fig, _ax = _plt.subplots(figsize=(9, 4.5))

    _ax.plot(_z, _sig, color="#E74C3C", lw=3, label=r"$\sigma(z) = 1 / (1 + e^{-z})$")
    _ax.axhline(0.5, color="#95A5A6", lw=1.5, linestyle="--", alpha=0.8)
    _ax.axhline(0.0, color="#2C3E50", lw=0.8, alpha=0.4)
    _ax.axhline(1.0, color="#2C3E50", lw=0.8, alpha=0.4)
    _ax.axvline(0.0, color="#2C3E50", lw=0.8, alpha=0.4)

    # Key property annotations
    _ax.scatter([0], [0.5], color="#E74C3C", s=120, zorder=6)
    _ax.annotate(
        r"$\sigma(0) = 0.5$  ← decision threshold",
        xy=(0, 0.5), xytext=(1.5, 0.44),
        fontsize=10, color="#2C3E50",
        arrowprops=dict(arrowstyle="->", color="#2C3E50"),
    )
    _ax.annotate(
        "Flat near ±∞\n(saturates)",
        xy=(8, _sig[_np.searchsorted(_z, 8)]),
        xytext=(5.5, 0.65),
        fontsize=9, color="#95A5A6",
        arrowprops=dict(arrowstyle="->", color="#95A5A6"),
    )
    _ax.annotate(
        "Steep near 0\n(most informative region)",
        xy=(0.5, _sig[_np.searchsorted(_z, 0.5)]),
        xytext=(-7, 0.7),
        fontsize=9, color="#27AE60",
        arrowprops=dict(arrowstyle="->", color="#27AE60"),
    )

    _ax.fill_between(_z, _sig, 0.5, where=_sig > 0.5, alpha=0.08, color="#E74C3C",
                     label="Predict class 1")
    _ax.fill_between(_z, _sig, 0.5, where=_sig < 0.5, alpha=0.08, color="#3498DB",
                     label="Predict class 0")

    _ax.set_xlabel("z  (linear score: Xw + b)", fontsize=12)
    _ax.set_ylabel(r"$\sigma(z)$  — predicted probability", fontsize=12)
    _ax.set_title("The Sigmoid Function — Maps Any Real Number to a Probability",
                  fontsize=13, fontweight="bold")
    _ax.set_ylim(-0.05, 1.1)
    _ax.legend(fontsize=10, loc="upper left")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    return _fig


@app.cell
def generate_cls_data():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _rng = _np.random.default_rng(7)
    _n_per_class = 100

    # Two 2D Gaussian clusters with some overlap
    _X0 = _rng.normal(loc=[-1.5, -1.5], scale=0.9, size=(_n_per_class, 2))
    _X1 = _rng.normal(loc=[1.5, 1.5],  scale=0.9, size=(_n_per_class, 2))
    X_cls = _np.vstack([_X0, _X1])
    y_cls = _np.array([0] * _n_per_class + [1] * _n_per_class, dtype=float)

    # Shuffle for good measure
    _idx = _rng.permutation(len(y_cls))
    X_cls = X_cls[_idx]
    y_cls = y_cls[_idx]

    # "Before" scatter — colored by class
    _fig, _ax = _plt.subplots(figsize=(7, 6))
    _mask0 = y_cls == 0
    _mask1 = y_cls == 1
    _ax.scatter(X_cls[_mask0, 0], X_cls[_mask0, 1], color="#3498DB", s=50, alpha=0.75,
                edgecolors="white", linewidths=0.5, label="Class 0", zorder=3)
    _ax.scatter(X_cls[_mask1, 0], X_cls[_mask1, 1], color="#E74C3C", s=50, alpha=0.75,
                edgecolors="white", linewidths=0.5, label="Class 1", zorder=3)
    _ax.set_xlabel("Feature 1", fontsize=12)
    _ax.set_ylabel("Feature 2", fontsize=12)
    _ax.set_title("2D Binary Classification Data — Before Fitting",
                  fontsize=13, fontweight="bold")
    _ax.legend(fontsize=11)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    return _fig, X_cls, y_cls


@app.cell
def _(mo):
    mo.md(r"""
    ### Loss Function: Binary Cross-Entropy (Log Loss)

    $$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$

    where $p_i = \sigma(X_i w + b)$.

    **Why not MSE for classification?**
    MSE + sigmoid creates a **non-convex** loss surface with local minima — gradient descent
    can get stuck. Binary cross-entropy is convex when composed with the sigmoid, so gradient
    descent always finds the global minimum.

    **Gradients:**

    $$\frac{\partial L}{\partial w} = \frac{1}{n}\, X^T \left(\sigma(Xw + b) - y\right)$$

    $$\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} \left(\sigma(X_i w + b) - y_i\right)$$

    **Note:** The gradient has the *same form* as linear regression — $X^T(\text{predictions} - \text{targets})$ — just with the sigmoid baked in.  The math works out beautifully.
    """)
    return


@app.cell
def logreg_gd(X_cls, y_cls):
    import numpy as _np

    # ── Sigmoid with numerical stability ──────────────────────────────────────
    def _sigmoid(z):
        return 1.0 / (1.0 + _np.exp(-_np.clip(z, -500, 500)))

    # ── Binary cross-entropy loss ─────────────────────────────────────────────
    def _bce(X, y, w, b):
        _eps = 1e-15
        _p = _sigmoid(X @ w + b)
        return float(-_np.mean(y * _np.log(_p + _eps) + (1 - y) * _np.log(1 - _p + _eps)))

    # ── Gradients (same structure as linear regression!) ──────────────────────
    def _grad_bce(X, y, w, b):
        _n = len(y)
        _err = _sigmoid(X @ w + b) - y          # (predictions - targets)
        _dw = (1.0 / _n) * (X.T @ _err)         # same form as linear: X^T(pred - y)
        _db = (1.0 / _n) * float(_np.sum(_err))
        return _dw, _db

    # ── Gradient descent: init at 0, lr=0.1, 2000 iterations ─────────────────
    _lr = 0.1
    _n_iter = 2000
    w_log = _np.zeros(X_cls.shape[1])
    b_log = 0.0
    loss_history_log = []

    for _i in range(_n_iter):
        loss_history_log.append(_bce(X_cls, y_cls, w_log, b_log))
        _dw, _db = _grad_bce(X_cls, y_cls, w_log, b_log)
        w_log = w_log - _lr * _dw
        b_log = b_log - _lr * _db

    _preds = (_sigmoid(X_cls @ w_log + b_log) >= 0.5).astype(float)
    _acc = float(_np.mean(_preds == y_cls))

    print(f"Logistic regression ({_n_iter} iter, lr={_lr})")
    print(f"  Final log loss:      {loss_history_log[-1]:.4f}")
    print(f"  Training accuracy:   {_acc * 100:.1f}%")
    print(f"  Learned weights:     w = {w_log}  b = {b_log:.4f}")

    return w_log, b_log, loss_history_log


@app.cell
def decision_boundary_viz(X_cls, y_cls, w_log, b_log):
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.colors import LinearSegmentedColormap as _LSC

    def _sigmoid(z):
        return 1.0 / (1.0 + _np.exp(-_np.clip(z, -500, 500)))

    # ── Probability heatmap over feature space ─────────────────────────────────
    _pad = 0.7
    _x1_min = X_cls[:, 0].min() - _pad
    _x1_max = X_cls[:, 0].max() + _pad
    _x2_min = X_cls[:, 1].min() - _pad
    _x2_max = X_cls[:, 1].max() + _pad

    _xx1, _xx2 = _np.meshgrid(
        _np.linspace(_x1_min, _x1_max, 350),
        _np.linspace(_x2_min, _x2_max, 350),
    )
    _grid = _np.column_stack([_xx1.ravel(), _xx2.ravel()])
    _probs = _sigmoid(_grid @ w_log + b_log).reshape(_xx1.shape)

    _fig, _ax = _plt.subplots(figsize=(9, 7))

    # Blue → white → red probability colormap
    _cmap = _LSC.from_list("prob", ["#3498DB", "#f5f5f5", "#E74C3C"])
    _cf = _ax.contourf(_xx1, _xx2, _probs, levels=60, cmap=_cmap, alpha=0.65)
    _plt.colorbar(_cf, ax=_ax, label="P(y=1 | x)", shrink=0.85)

    # Decision boundary: where P(y=1) = 0.5, i.e. w·x + b = 0
    _ax.contour(_xx1, _xx2, _probs, levels=[0.5],
                colors=["#2C3E50"], linewidths=2.5)

    # Annotate the boundary line
    _x1_mid_idx = _xx1.shape[1] // 2
    _x1_mid = _xx1[0, _x1_mid_idx]
    if abs(w_log[1]) > 1e-8:
        _x2_mid = -(w_log[0] * _x1_mid + b_log) / w_log[1]
        _ax.annotate(
            "Decision boundary\n$P(y=1) = 0.5$",
            xy=(_x1_mid, _x2_mid),
            xytext=(_x1_mid + 0.6, _x2_mid + 0.7),
            fontsize=10, color="#2C3E50",
            arrowprops=dict(arrowstyle="->", color="#2C3E50"),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
        )

    # Data points
    _mask0 = y_cls == 0
    _mask1 = y_cls == 1
    _ax.scatter(X_cls[_mask0, 0], X_cls[_mask0, 1], color="#3498DB", s=55, alpha=0.9,
                edgecolors="white", linewidths=0.6, label="Class 0", zorder=5)
    _ax.scatter(X_cls[_mask1, 0], X_cls[_mask1, 1], color="#E74C3C", s=55, alpha=0.9,
                edgecolors="white", linewidths=0.6, label="Class 1", zorder=5)

    _ax.set_xlabel("Feature 1", fontsize=12)
    _ax.set_ylabel("Feature 2", fontsize=12)
    _ax.set_title(
        "Logistic Regression — Decision Boundary & Probability Heatmap",
        fontsize=13, fontweight="bold",
    )
    _ax.legend(fontsize=11, loc="upper left")
    _ax.grid(True, alpha=0.2)
    _fig.tight_layout()
    return _fig


@app.cell
def logreg_loss_curve(loss_history_log, loss_history_linreg):
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _log_losses = _np.array(loss_history_log)
    _lin_losses = _np.array(loss_history_linreg)

    # Normalize both to [0, 1] so they're visually comparable
    _log_norm = (_log_losses - _log_losses[-1]) / (_log_losses[0] - _log_losses[-1] + 1e-10)
    _lin_norm = (_lin_losses - _lin_losses[-1]) / (_lin_losses[0] - _lin_losses[-1] + 1e-10)
    _n_shared = min(len(_log_norm), len(_lin_norm))

    _fig, _ax = _plt.subplots(figsize=(10, 5))

    _ax.plot(_log_norm[:_n_shared], color="#E74C3C", lw=2.5,
             label="Log loss  (logistic regression, lr=0.1)")
    _ax.plot(_lin_norm[:_n_shared], color="#3498DB", lw=2.5,
             label="MSE loss  (linear regression, lr=0.01)")

    _ax.set_xlabel("Iteration", fontsize=12)
    _ax.set_ylabel("Normalized loss  (0 = start, 1 = converged)", fontsize=12)
    _ax.set_title(
        "Convergence Comparison — Log Loss vs MSE  (normalized to same scale)",
        fontsize=13, fontweight="bold",
    )
    _ax.legend(fontsize=11)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    return _fig


# ── Part 3: Connecting the Two ───────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## Part 3: Connecting the Two

    | | Linear Regression | Logistic Regression |
    |---|---|---|
    | **Task** | Regression — continuous output | Classification — binary decision |
    | **Model** | $y = Xw + b$ | $P(y=1) = \sigma(Xw + b)$ |
    | **Loss** | MSE | Binary cross-entropy |
    | **Output range** | $(-\infty,\, +\infty)$ | $(0,\, 1)$ |
    | **Decision boundary** | N/A | Where $\sigma(Xw+b) = 0.5$, i.e. $Xw+b = 0$ |
    | **Loss landscape** | Convex (GD or closed-form) | Convex (GD only) |
    | **Gradient form** | $X^T(Xw+b - y)$ | $X^T(\sigma(Xw+b) - y)$ |
    | **Regularization** | Ridge (L2), Lasso (L1) | Same — very common in practice |

    **The unifying pattern:** Both models share the same gradient *structure* — the difference
    is only whether you apply a sigmoid or not.  The sigmoid is what makes logistic regression
    a classifier and changes the loss from MSE to cross-entropy.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### When to Use Which

    **Linear regression:**
    - Predicting a continuous value — price, temperature, revenue, time-to-event
    - When you need an interpretable coefficient for each feature
    - When residuals are approximately Gaussian

    **Logistic regression:**
    - Binary classification with calibrated probabilities — churn, fraud, click, default
    - When you need a fast, interpretable baseline before trying complex models
    - When the decision boundary is roughly linear in feature space
    - When model size and training speed matter

    **Rule of thumb:** Logistic regression is almost always the first model to try for
    classification.  It is fast, interpretable, and probabilistic.
    *"If logistic regression works well enough, you probably don't need anything fancier."*
    """)
    return


# ── Flashcard Summary ────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ## Flashcard Summary

    | Question | Answer |
    |---|---|
    | What loss does linear regression use? | MSE — mean squared error |
    | Why not MSE for logistic regression? | MSE + sigmoid is non-convex; log loss is convex, so GD is guaranteed to find the global minimum |
    | What does the sigmoid do? | Squashes any real number to (0, 1) — maps the linear score to a probability |
    | What is the decision boundary? | The hyperplane where $P(y=1) = 0.5$, i.e. $Xw+b=0$ |
    | Why use gradient descent over the normal equation? | Scales to large datasets; works for any differentiable loss; the normal equation requires $O(d^3)$ matrix inversion |
    | What happens if learning rate is too high? | Loss oscillates or diverges — the update overshoots the minimum each step |
    | What does "linear in the parameters" mean? | The output is a weighted sum of the features; even if features are nonlinear, the model is still linear |
    | How do you regularize logistic regression? | L1 (Lasso) for sparsity, L2 (Ridge) for shrinkage — add penalty to weights in the loss |
    | What is the gradient of logistic regression? | $X^T(\\sigma(Xw+b) - y)$ — same form as linear regression, sigmoid baked in |
    | When pick logistic over a neural net? | Interpretability, small data, fast baseline, calibrated probabilities, regulatory requirements |
    """)
    return


# ── Interview Talking Points ─────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    ### "Walk me through gradient descent."

    > "Start by initializing the model weights — often randomly or at zero.  Then iterate:
    > compute the loss on the current batch, compute the gradient of the loss with respect to
    > each parameter (the direction of steepest ascent), and step the parameters in the
    > *opposite* direction by a small amount controlled by the learning rate.  Repeat until
    > the loss stops improving.  The key insight is that for convex losses like MSE and
    > log loss, this process is guaranteed to find the global minimum."

    ---

    ### "Implement logistic regression from scratch."

    Walk through this notebook cell by cell:
    1. Generate binary classification data
    2. Define sigmoid and binary cross-entropy loss
    3. Compute gradients: `X.T @ (sigmoid(Xw + b) - y) / n`
    4. Gradient descent loop: predict → loss → gradient → update
    5. Visualize the decision boundary as the line where `Xw + b = 0`

    ---

    ### "What is the difference between linear and logistic regression?"

    > "Same optimization algorithm, same gradient structure — the difference is the output
    > transformation and the loss function.  Linear regression predicts a continuous value
    > directly; logistic regression passes the linear score through a sigmoid to get a
    > probability, and swaps MSE for cross-entropy to preserve convexity.  Both use gradient
    > descent; linear regression also has a closed-form normal equation, while logistic does not."

    ---

    ### Connection to my work

    > "In my backtesting engine, I use similar optimization patterns — iteratively updating
    > portfolio weights to minimize a loss function over historical data, which is fundamentally
    > the same gradient-based approach.  The difference is the loss function: instead of MSE or
    > cross-entropy, I'm minimizing something like negative Sharpe ratio or drawdown.  The
    > update rule is the same structure."
    """)
    return


if __name__ == "__main__":
    app.run()
