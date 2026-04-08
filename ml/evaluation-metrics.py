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
    # Evaluation Metrics — Precision/Recall/F1, AUC-ROC, AUC-PR, Calibration

    | Field | Value |
    |-------|-------|
    | Date  | 2026-04-08 |
    | Track | ML Theory |
    | Time  | 60 min |
    | Topics | Precision · Recall · F1 · AUC-ROC · AUC-PR · Calibration · When Accuracy Misleads |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Why Metrics Matter More Than Models

    > **"Choosing the wrong metric is worse than choosing the wrong model.
    > A perfect model optimized for the wrong metric is useless."**

    The question to always ask first: **"What's the cost of being wrong?"** — this determines
    your metric, and the metric determines everything downstream.

    Two types of being wrong — they almost never cost the same:

    | Error Type | What it means | Also called |
    |-----------|--------------|-------------|
    | **False Positive** (FP) | Predicted positive, actually negative | Type I error — "crying wolf" |
    | **False Negative** (FN) | Predicted negative, actually positive | Type II error — "missing the threat" |

    Ask this before picking any metric: *"Which error is more expensive for this problem?"*
    That answer drives every metric choice that follows.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## The Accuracy Trap — Start Here

    **Accuracy** = (correct predictions) / (total predictions)

    **The trap:** with 99% negative class, a model that *always* predicts "not fraud" gets
    **99% accuracy** while catching exactly zero fraudsters.

    > **This is the first thing to say in an interview when asked about metrics:**
    > "Accuracy is misleading for imbalanced problems — which is most real-world problems."

    **Rule of thumb:** if your classes are more than 80/20 skewed, accuracy is nearly useless.
    """)
    return


@app.cell
def accuracy_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _rng = _np.random.default_rng(42)
    _n = 1000
    _y_true = _np.array([1] * 10 + [0] * 990)  # 1% positive class

    # Model A: always predicts negative — 99% accuracy, catches 0 fraud
    _pred_A = _np.zeros(_n, dtype=int)

    # Model B: random classifier
    _pred_B = _rng.integers(0, 2, size=_n)

    # Model C: decent classifier — catches 8/10 fraud, ~97% accuracy
    _pred_C = _np.zeros(_n, dtype=int)
    _pred_C[_np.where(_y_true == 1)[0][:8]] = 1
    _pred_C[_rng.choice(_np.where(_y_true == 0)[0], size=22, replace=False)] = 1

    def _metrics(y, yp):
        _tp = int(_np.sum((yp == 1) & (y == 1)))
        _fp = int(_np.sum((yp == 1) & (y == 0)))
        _fn = int(_np.sum((yp == 0) & (y == 1)))
        _acc = float(_np.mean(y == yp))
        _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        _rec = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        return _acc, _prec, _rec

    _results = {
        "Model A\n(always negative)": _metrics(_y_true, _pred_A),
        "Model B\n(random)": _metrics(_y_true, _pred_B),
        "Model C\n(decent)": _metrics(_y_true, _pred_C),
    }
    _mnames = list(_results.keys())
    _colors = ["#E74C3C", "#F39C12", "#27AE60"]
    _metric_labels = ["Accuracy", "Precision", "Recall"]

    _fig, _axes = _plt.subplots(1, 3, figsize=(14, 5))
    for _idx, (_ax, _mlabel) in enumerate(zip(_axes, _metric_labels)):
        _vals = [_results[m][_idx] for m in _mnames]
        _bars = _ax.bar(_mnames, _vals, color=_colors, alpha=0.85, edgecolor="white")
        for _bar, _val in zip(_bars, _vals):
            _ax.text(_bar.get_x() + _bar.get_width() / 2, _val + 0.015,
                     f"{_val:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        _ax.set_ylim(0, 1.18)
        _ax.set_title(_mlabel, fontsize=13, fontweight="bold")
        _ax.set_ylabel(_mlabel, fontsize=11)
        _ax.grid(True, axis="y", alpha=0.3)
        _ax.tick_params(axis="x", labelsize=9)

    _fig.suptitle("The Accuracy Trap: 3 Models on 1% Fraud Dataset (1000 samples, 10 positives)",
                  fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    > **Model A has the highest accuracy (99%) but is completely useless** — it catches zero fraud.
    > Model C has *lower* accuracy (~97%) but catches 80% of actual fraud cases.
    > Accuracy alone hid the truth. Precision and recall reveal it.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Confusion Matrix — The Foundation

    The 2×2 grid that every classification metric derives from:

    |  | Predicted Positive | Predicted Negative |
    |--|-------------------|-------------------|
    | **Actually Positive** | **TP** — predicted fraud, was fraud ✓ | **FN** — predicted legit, was fraud ✗ ← expensive |
    | **Actually Negative** | **FP** — predicted fraud, was legit ✗ | **TN** — predicted legit, was legit ✓ |

    - **TP** (True Positive): caught it — correct alarm
    - **FP** (False Positive): crying wolf — blocked a good customer
    - **FN** (False Negative): missed it — fraud got through (often the expensive mistake)
    - **TN** (True Negative): normal operation — no news is good news

    > **Every metric below is just a different way of combining these 4 numbers.**
    """)
    return


@app.cell
def confusion_matrix_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_classification as _make_clf
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.model_selection import train_test_split as _tts

    _X, _y = _make_clf(n_samples=2000, n_features=20, n_informative=5,
                       weights=[0.95, 0.05], random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.3, random_state=42, stratify=_y)
    _y_pred = _LR(max_iter=1000, random_state=42).fit(_X_tr, _y_tr).predict(_X_te)

    _tp = int(_np.sum((_y_pred == 1) & (_y_te == 1)))
    _fp = int(_np.sum((_y_pred == 1) & (_y_te == 0)))
    _fn = int(_np.sum((_y_pred == 0) & (_y_te == 1)))
    _tn = int(_np.sum((_y_pred == 0) & (_y_te == 0)))

    # Rows = Actual (Neg, Pos), Cols = Predicted (Neg, Pos)
    _grid      = [[_tn, _fp], [_fn, _tp]]
    _lbl       = [["TN", "FP"], ["FN", "TP"]]
    _desc      = [["Legit → Pred Legit\n(normal operation ✓)",
                   "Legit → Pred Fraud\n(blocked good customer ✗)"],
                  ["Fraud → Pred Legit\n(missed — expensive! ✗)",
                   "Fraud → Pred Fraud\n(caught it ✓)"]]
    _bg        = [["#a9dfbf", "#f1948a"], ["#c0392b", "#27ae60"]]
    _fg        = [["#145a32", "#78281f"], ["white",   "white"  ]]

    _fig, _ax = _plt.subplots(figsize=(9, 6.5))
    for _i in range(2):
        for _j in range(2):
            _ax.add_patch(_plt.Rectangle([_j - 0.5, _i - 0.5], 1, 1,
                          facecolor=_bg[_i][_j], edgecolor="white", linewidth=3))
            _ax.text(_j, _i - 0.22, str(_grid[_i][_j]),
                     ha="center", va="center", fontsize=22, fontweight="bold", color=_fg[_i][_j])
            _ax.text(_j, _i + 0.10, _lbl[_i][_j],
                     ha="center", va="center", fontsize=13, fontweight="bold", color=_fg[_i][_j])
            _ax.text(_j, _i + 0.35, _desc[_i][_j],
                     ha="center", va="center", fontsize=8,  color=_fg[_i][_j], style="italic")

    _ax.set_xlim(-0.5, 1.5)
    _ax.set_ylim(-0.5, 1.5)
    _ax.set_xticks([0, 1])
    _ax.set_yticks([0, 1])
    _ax.set_xticklabels(["Predicted Negative", "Predicted Positive"], fontsize=11)
    _ax.set_yticklabels(["Actually Negative", "Actually Positive"], fontsize=11)
    _ax.set_title("Confusion Matrix — Logistic Regression (5% Positive Class, n=600 test)",
                  fontsize=13, fontweight="bold")
    _ax.tick_params(length=0)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## Precision & Recall — The Core Trade-off

    **Precision** = TP / (TP + FP) — *"How trustworthy are my positive predictions?"*
    - Of everything I flagged as positive, what % was actually positive?
    - High precision = few false alarms
    - **Optimize when false positives are expensive:** spam filter (don't bin real email), medical diagnosis (don't scare healthy patients), content recommendation (irrelevant recs erode trust)

    **Recall** (Sensitivity) = TP / (TP + FN) — *"How many positives am I missing?"*
    - Of all actual positives, what % did I catch?
    - High recall = few missed cases
    - **Optimize when false negatives are expensive:** fraud detection (catch every fraud), cancer screening (don't miss any cancer), legal document review (missing docs = malpractice)

    > **The trade-off:** increasing one typically decreases the other. You can't maximize both.
    > The optimal threshold is determined by the *relative cost* of each error type — a business decision, not a statistical one.
    """)
    return


@app.cell
def lr_data():
    """Shared data for all threshold-related visualizations."""
    import numpy as _np
    from sklearn.datasets import make_classification as _make_clf
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.model_selection import train_test_split as _tts

    _X, _y = _make_clf(n_samples=3000, n_features=20, n_informative=8,
                       weights=[0.90, 0.10], random_state=7)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.3, random_state=7, stratify=_y)
    _lr = _LR(max_iter=1000, random_state=7).fit(_X_tr, _y_tr)
    lr_probs_test = _lr.predict_proba(_X_te)[:, 1]
    y_test_pr = _y_te
    return lr_probs_test, y_test_pr


@app.cell
def _(mo):
    threshold_slider = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01,
                                    label="Classification Threshold")
    return (threshold_slider,)


@app.cell
def _(lr_probs_test, mo, threshold_slider, y_test_pr):
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _thresh = threshold_slider.value
    _yp = (lr_probs_test >= _thresh).astype(int)

    _tp = int(_np.sum((_yp == 1) & (y_test_pr == 1)))
    _fp = int(_np.sum((_yp == 1) & (y_test_pr == 0)))
    _fn = int(_np.sum((_yp == 0) & (y_test_pr == 1)))
    _tn = int(_np.sum((_yp == 0) & (y_test_pr == 0)))
    _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
    _rec  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
    _f1   = 2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else 0.0

    _fig, _axes = _plt.subplots(1, 2, figsize=(13, 5))

    # Left: confusion matrix
    _ax_cm = _axes[0]
    _bg = [["#a9dfbf", "#f1948a"], ["#c0392b", "#27ae60"]]
    _fg = [["#145a32", "#78281f"], ["white", "white"]]
    _cm = [[_tn, _fp], [_fn, _tp]]
    _lbl = [["TN", "FP"], ["FN", "TP"]]
    for _i in range(2):
        for _j in range(2):
            _ax_cm.add_patch(_plt.Rectangle([_j - 0.5, _i - 0.5], 1, 1,
                             facecolor=_bg[_i][_j], edgecolor="white", linewidth=3))
            _ax_cm.text(_j, _i, f"{_lbl[_i][_j]}\n{_cm[_i][_j]}",
                       ha="center", va="center", fontsize=14, fontweight="bold",
                       color=_fg[_i][_j])
    _ax_cm.set_xlim(-0.5, 1.5)
    _ax_cm.set_ylim(-0.5, 1.5)
    _ax_cm.set_xticks([0, 1])
    _ax_cm.set_yticks([0, 1])
    _ax_cm.set_xticklabels(["Pred Negative", "Pred Positive"])
    _ax_cm.set_yticklabels(["Actual Negative", "Actual Positive"])
    _ax_cm.set_title(f"Confusion Matrix @ threshold = {_thresh:.2f}", fontsize=12, fontweight="bold")
    _ax_cm.tick_params(length=0)

    # Right: precision / recall / F1 bars
    _ax_b = _axes[1]
    _mvals   = [_prec, _rec, _f1]
    _mnames  = ["Precision", "Recall", "F1"]
    _mcolors = ["#3498DB", "#E67E22", "#9B59B6"]
    _bars = _ax_b.bar(_mnames, _mvals, color=_mcolors, alpha=0.85, edgecolor="white")
    for _bar, _val in zip(_bars, _mvals):
        _ax_b.text(_bar.get_x() + _bar.get_width() / 2, _val + 0.02,
                   f"{_val:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    _ax_b.set_ylim(0, 1.2)
    _ax_b.set_title(f"Metrics @ threshold = {_thresh:.2f}", fontsize=12, fontweight="bold")
    _ax_b.axhline(0.5, color="#95A5A6", linestyle="--", lw=1.5, alpha=0.5)
    _ax_b.grid(True, axis="y", alpha=0.3)

    _fig.tight_layout()

    mo.vstack([
        mo.md(f"""
        ### Interactive Threshold Visualization

        Move the slider to see precision, recall, and F1 update in real time.

        {threshold_slider}

        **Threshold: {_thresh:.2f}** → Precision: **{_prec:.3f}** | Recall: **{_rec:.3f}** | F1: **{_f1:.3f}**
        """),
        _fig,
        mo.md("""
        > **Low threshold** = high recall, low precision (flag everything, catch all fraud, many false alarms).
        > **High threshold** = high precision, low recall (only flag when very confident, miss more fraud).
        """),
    ])
    return


@app.cell
def pr_threshold_curve(lr_probs_test, y_test_pr):
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _thresholds = _np.linspace(0.01, 0.99, 200)
    _precs, _recs = [], []
    for _t in _thresholds:
        _yp = (lr_probs_test >= _t).astype(int)
        _tp = int(_np.sum((_yp == 1) & (y_test_pr == 1)))
        _fp = int(_np.sum((_yp == 1) & (y_test_pr == 0)))
        _fn = int(_np.sum((_yp == 0) & (y_test_pr == 1)))
        _precs.append(_tp / (_tp + _fp) if (_tp + _fp) > 0 else 1.0)
        _recs.append(_tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0)

    _pa = _np.array(_precs)
    _ra = _np.array(_recs)
    _cross = int(_np.argmin(_np.abs(_pa - _ra)))

    _fig, _ax = _plt.subplots(figsize=(10, 5))
    _ax.plot(_thresholds, _pa, color="#3498DB", lw=2.5, label="Precision")
    _ax.plot(_thresholds, _ra, color="#E67E22", lw=2.5, label="Recall")
    _ax.axvline(0.5, color="#95A5A6", lw=1.5, linestyle="--", alpha=0.7,
               label="Default threshold (0.5)")
    _ax.scatter([_thresholds[_cross]], [_pa[_cross]], color="#9B59B6", s=120, zorder=5,
               label=f"Crossover @ {_thresholds[_cross]:.2f}")
    _ax.fill_between(_thresholds, _pa, _ra, where=(_pa > _ra),
                    alpha=0.07, color="#3498DB")
    _ax.fill_between(_thresholds, _pa, _ra, where=(_ra >= _pa),
                    alpha=0.07, color="#E67E22")
    _ax.set_xlabel("Classification Threshold", fontsize=12)
    _ax.set_ylabel("Score", fontsize=12)
    _ax.set_title("Precision and Recall vs Threshold (10% Positive Class)\nEvery point on this curve is one setting of the interactive slider above",
                  fontsize=12, fontweight="bold")
    _ax.legend(fontsize=10)
    _ax.grid(True, alpha=0.3)
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1.05)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ### When to Optimize for Precision vs Recall

    | Scenario | Optimize for | Why |
    |----------|-------------|-----|
    | Fraud detection | Recall (with precision floor) | Missing fraud costs more than false alarms |
    | Spam filter | Precision | Sending real email to spam is unacceptable |
    | Cancer screening | Recall | Missing cancer is life-threatening |
    | Content recommendation | Precision | Irrelevant recs erode user trust |
    | Legal document review | Recall | Missing relevant docs = malpractice risk |
    | Manufacturing defect | Recall | Shipping defective products is expensive |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## F1 Score — The Compromise

    **F1** = 2 × (precision × recall) / (precision + recall) — the **harmonic mean**, not arithmetic.

    **Why harmonic mean?** It punishes extreme imbalance between precision and recall:

    - precision = 0.99, recall = 0.01 → **F1 = 0.02** (correctly terrible)
    - precision = 0.99, recall = 0.01 → arithmetic mean = **0.50** (misleadingly OK)

    **When to use F1:** when you need one number and both precision and recall matter roughly equally.

    **When NOT to use F1:** when cost asymmetry is clear — optimize precision or recall directly instead.

    **F-beta score:** F_β weights recall β times more than precision.
    - **F2**: emphasizes recall → fraud detection, cancer screening
    - **F0.5**: emphasizes precision → spam filter, recommendation
    """)
    return


@app.cell
def f1_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _p = _np.linspace(0.01, 1.0, 200)
    _r = _np.linspace(0.01, 1.0, 200)
    _P, _R = _np.meshgrid(_p, _r)
    _F1 = 2 * _P * _R / (_P + _R)

    _fig, _axes = _plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: F1 contour surface
    _ax1 = _axes[0]
    _cs = _ax1.contourf(_P, _R, _F1, levels=20, cmap="RdYlGn")
    _plt.colorbar(_cs, ax=_ax1, label="F1 Score")
    _ax1.contour(_P, _R, _F1, levels=[0.2, 0.4, 0.6, 0.8],
                colors="white", linewidths=1.2, alpha=0.6)
    _ax1.set_xlabel("Precision", fontsize=12)
    _ax1.set_ylabel("Recall", fontsize=12)
    _ax1.set_title("F1 Score Surface\nHigh only when BOTH precision AND recall are high",
                  fontsize=12, fontweight="bold")
    _ax1.set_aspect("equal")

    # Right: harmonic vs arithmetic for precision=0.99
    _ax2 = _axes[1]
    _r_range = _np.linspace(0.001, 1.0, 300)
    _p_fixed = 0.99
    _f1_curve = 2 * _p_fixed * _r_range / (_p_fixed + _r_range)
    _am_curve = (_p_fixed + _r_range) / 2
    _r_pt, _p_pt = 0.01, _p_fixed
    _f1_pt = 2 * _p_pt * _r_pt / (_p_pt + _r_pt)
    _am_pt = (_p_pt + _r_pt) / 2

    _ax2.plot(_r_range, _f1_curve, color="#E74C3C", lw=2.5,
             label=f"F1 harmonic mean  (P = {_p_fixed})")
    _ax2.plot(_r_range, _am_curve, color="#3498DB", lw=2.5, linestyle="--",
             label=f"Arithmetic mean  (P = {_p_fixed})")
    _ax2.scatter([_r_pt], [_f1_pt], color="#E74C3C", s=90, zorder=5)
    _ax2.annotate(f"R=0.01:\nF1 = {_f1_pt:.3f}  ← correctly terrible\nAM = {_am_pt:.2f}  ← misleadingly OK",
                 xy=(_r_pt, _f1_pt), xytext=(0.18, 0.22), fontsize=9, color="#E74C3C",
                 arrowprops=dict(arrowstyle="->", color="#E74C3C"))
    _ax2.set_xlabel("Recall", fontsize=12)
    _ax2.set_ylabel("Score", fontsize=12)
    _ax2.set_title("Harmonic vs Arithmetic Mean\n(why harmonic punishes extremes)",
                  fontsize=12, fontweight="bold")
    _ax2.legend(fontsize=9)
    _ax2.grid(True, alpha=0.3)
    _ax2.set_ylim(0, 1.05)

    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## AUC-ROC — Threshold-Independent Evaluation

    **ROC curve:** plots TPR (recall) vs FPR (= FP / (FP + TN)) at every threshold.

    **AUC-ROC:** area under this curve. 1.0 = perfect, 0.5 = random classifier.

    > **Intuition:** "The probability that the model ranks a random positive higher than a random negative."

    **When it's great:** comparing models across all thresholds, balanced class distributions.

    **When it LIES:** imbalanced data.

    Why it lies — **the FPR denominator problem:**
    FPR = FP / (FP + TN). When negatives dominate (e.g., 99% of data), TN is enormous.
    Even with hundreds of false positives, FPR stays near zero — making the ROC curve look
    impressive while the model is actually failing on the minority class.
    """)
    return


@app.cell
def roc_curves_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_classification as _make_clf
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.ensemble import RandomForestClassifier as _RF
    from sklearn.dummy import DummyClassifier as _Dummy
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.metrics import roc_curve as _roc_curve, auc as _auc

    _X, _y = _make_clf(n_samples=3000, n_features=20, n_informative=8,
                       weights=[0.90, 0.10], random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.3, random_state=42, stratify=_y)

    _models = [
        ("Logistic Regression", _LR(max_iter=1000, random_state=42), "#3498DB"),
        ("Random Forest",       _RF(n_estimators=100, random_state=42), "#E67E22"),
        ("Dummy (random)",      _Dummy(strategy="stratified", random_state=42), "#95A5A6"),
    ]

    _fig, _ax = _plt.subplots(figsize=(8, 7))
    _ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random classifier (AUC = 0.50)")

    for _name, _model, _color in _models:
        _model.fit(_X_tr, _y_tr)
        _fpr, _tpr, _ = _roc_curve(_y_te, _model.predict_proba(_X_te)[:, 1])
        _score = _auc(_fpr, _tpr)
        _ax.plot(_fpr, _tpr, color=_color, lw=2.5, label=f"{_name}  (AUC = {_score:.3f})")

    _ax.set_xlabel("False Positive Rate", fontsize=12)
    _ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    _ax.set_title("ROC Curves — 3 Models on 10% Positive Class Data", fontsize=13, fontweight="bold")
    _ax.legend(fontsize=10, loc="lower right")
    _ax.grid(True, alpha=0.3)
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1.02)
    _fig.tight_layout()
    return _fig


@app.cell
def roc_vs_pr_trap():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_classification as _make_clf
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.ensemble import RandomForestClassifier as _RF
    from sklearn.dummy import DummyClassifier as _Dummy
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.metrics import (roc_curve as _roc_curve, auc as _auc,
                                 precision_recall_curve as _prc)

    # 1% positive — the imbalanced trap
    _X, _y = _make_clf(n_samples=8000, n_features=20, n_informative=8,
                       weights=[0.99, 0.01], random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.3, random_state=42, stratify=_y)
    _prev = float(_y_te.mean())

    _models = [
        ("Logistic Regression", _LR(max_iter=1000, random_state=42, class_weight="balanced"), "#3498DB"),
        ("Random Forest",       _RF(n_estimators=100, random_state=42, class_weight="balanced"), "#E67E22"),
        ("Dummy (stratified)",  _Dummy(strategy="stratified", random_state=42), "#95A5A6"),
    ]

    _fig, (_ax_roc, _ax_pr) = _plt.subplots(1, 2, figsize=(14, 6))

    _ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random (AUC = 0.50)")
    _ax_pr.axhline(_prev, color="k", linestyle="--", lw=1.5, alpha=0.5,
                  label=f"Baseline = prevalence ({_prev:.3f})")

    for _name, _model, _color in _models:
        _model.fit(_X_tr, _y_tr)
        _probs = _model.predict_proba(_X_te)[:, 1]

        _fpr, _tpr, _ = _roc_curve(_y_te, _probs)
        _ax_roc.plot(_fpr, _tpr, color=_color, lw=2.5,
                    label=f"{_name}\nAUC-ROC = {_auc(_fpr, _tpr):.3f}")

        _p_c, _r_c, _ = _prc(_y_te, _probs)
        _ax_pr.plot(_r_c, _p_c, color=_color, lw=2.5,
                   label=f"{_name}\nAUC-PR = {_auc(_r_c, _p_c):.3f}")

    _ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    _ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    _ax_roc.set_title("ROC Curve (1% positive)\n← Looks impressive!", fontsize=12, fontweight="bold")
    _ax_roc.legend(fontsize=8, loc="lower right")
    _ax_roc.grid(True, alpha=0.3)
    _ax_roc.set_xlim(0, 1)
    _ax_roc.set_ylim(0, 1.02)

    _ax_pr.set_xlabel("Recall", fontsize=11)
    _ax_pr.set_ylabel("Precision", fontsize=11)
    _ax_pr.set_title("Precision-Recall Curve (1% positive)\n← Reveals the truth", fontsize=12, fontweight="bold")
    _ax_pr.legend(fontsize=8, loc="upper right")
    _ax_pr.grid(True, alpha=0.3)
    _ax_pr.set_xlim(0, 1)
    _ax_pr.set_ylim(0, 1.02)

    _fig.suptitle("The Imbalanced Data Trap: AUC-ROC vs AUC-PR (1% Positive Class)",
                  fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    > **In my fraud detection system design, I specifically chose AUC-PR over AUC-ROC for this exact reason.**
    > With 0.1% fraud rate, AUC-ROC inflates performance because the enormous negative class keeps FPR
    > artificially low — even a mediocre model looks great. AUC-PR forces real performance on the class that matters.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## AUC-PR — The Right Metric for Imbalanced Data

    **Precision-Recall curve:** plots precision vs recall at every threshold.

    **AUC-PR (Average Precision):** area under the PR curve.
    **Baseline = prevalence of positive class** (not 0.5!).

    Why it's better for imbalanced data:
    - **Does not use TN at all** — the huge negative class cannot inflate scores
    - Harder to beat: a useless model that predicts all positive achieves recall = 1.0 but precision = prevalence (e.g., 0.01)
    - Directly measures performance on the positive class, where it actually matters

    > **Rule of thumb:** if positive class < 10% of data, use AUC-PR not AUC-ROC.

    `sklearn.metrics.average_precision_score` computes AP — a close approximation of AUC-PR.
    """)
    return


@app.cell
def pr_curves_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_classification as _make_clf
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.ensemble import RandomForestClassifier as _RF
    from sklearn.dummy import DummyClassifier as _Dummy
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.metrics import precision_recall_curve as _prc, average_precision_score as _ap

    _X, _y = _make_clf(n_samples=5000, n_features=20, n_informative=8,
                       weights=[0.95, 0.05], random_state=99)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.3, random_state=99, stratify=_y)
    _prev = float(_y_te.mean())

    _models = [
        ("Logistic Regression", _LR(max_iter=1000, random_state=42), "#3498DB"),
        ("Random Forest",       _RF(n_estimators=100, random_state=42), "#E67E22"),
        ("Dummy (stratified)",  _Dummy(strategy="stratified", random_state=42), "#95A5A6"),
    ]

    _fig, _ax = _plt.subplots(figsize=(9, 7))
    _ax.axhline(_prev, color="k", linestyle="--", lw=1.5, alpha=0.6,
               label=f"Baseline = positive prevalence ({_prev:.3f})")

    for _name, _model, _color in _models:
        _model.fit(_X_tr, _y_tr)
        _probs = _model.predict_proba(_X_te)[:, 1]
        _p_c, _r_c, _ = _prc(_y_te, _probs)
        _ap_score = _ap(_y_te, _probs)
        _ax.plot(_r_c, _p_c, color=_color, lw=2.5, label=f"{_name}  (AP = {_ap_score:.3f})")
        _ax.fill_between(_r_c, _p_c, _prev,
                        where=(_p_c >= _prev), alpha=0.07, color=_color)

    _ax.set_xlabel("Recall", fontsize=12)
    _ax.set_ylabel("Precision", fontsize=12)
    _ax.set_title("Precision-Recall Curves (5% Positive Class)\nBaseline = positive prevalence, NOT 0.5",
                  fontsize=13, fontweight="bold")
    _ax.legend(fontsize=10, loc="upper right")
    _ax.grid(True, alpha=0.3)
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1.05)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## Calibration — Are Your Probabilities Real?

    A model is **calibrated** if "when it says 70% probability of fraud, it's actually fraud 70% of the time."

    **Why calibration matters:**
    - Using probabilities to set thresholds (approve / review / decline queue)
    - Multiplying probability × cost for expected value calculations
    - Prioritizing a review queue by risk score
    - Reporting confidence scores to end users

    | Model | Calibrated? | Why |
    |-------|------------|-----|
    | Logistic Regression | ✓ Well-calibrated by design | Trained directly on log-loss (cross-entropy) |
    | Random Forest | ✗ Overconfident | Probabilities cluster near 0 and 1 |
    | Naive Bayes | ✗ Notoriously miscalibrated | Feature independence assumption violated in practice |
    | Deep Learning | ✗ Often overconfident | Softmax outputs aren't true probabilities without calibration |

    **When calibration matters most:** when you're using the probability score itself — for
    thresholding, queuing, or expected-value decisions. If you only care about rank order
    (e.g., AUC-ROC), calibration doesn't matter.
    """)
    return


@app.cell
def calibration_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_classification as _make_clf
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.ensemble import RandomForestClassifier as _RF
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.calibration import calibration_curve as _cal_curve

    _X, _y = _make_clf(n_samples=5000, n_features=20, n_informative=8,
                       weights=[0.85, 0.15], random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.3, random_state=42, stratify=_y)

    _lr = _LR(max_iter=1000, random_state=42).fit(_X_tr, _y_tr)
    _rf = _RF(n_estimators=100, random_state=42).fit(_X_tr, _y_tr)

    _fig, _ax = _plt.subplots(figsize=(8, 7))
    _ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect calibration (diagonal)")

    for _model, _name, _color in [
        (_lr, "Logistic Regression",        "#3498DB"),
        (_rf, "Random Forest (uncalibrated)", "#E74C3C"),
    ]:
        _probs = _model.predict_proba(_X_te)[:, 1]
        _frac_pos, _mean_pred = _cal_curve(_y_te, _probs, n_bins=10)
        _ax.plot(_mean_pred, _frac_pos, marker="o", markersize=8, lw=2.5,
                color=_color, label=_name)
        _ax.fill_between(_mean_pred, _frac_pos, _mean_pred, alpha=0.10, color=_color)

    _ax.annotate("RF says ~80% confident\nbut actually right ~60%\n→ overconfident",
                xy=(0.80, 0.62), xytext=(0.48, 0.38), fontsize=9, color="#E74C3C",
                arrowprops=dict(arrowstyle="->", color="#E74C3C"))

    _ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    _ax.set_ylabel("Fraction of Positives (Actual Rate)", fontsize=12)
    _ax.set_title("Calibration Plot (Reliability Diagram)\nDiagonal = perfect calibration",
                  fontsize=13, fontweight="bold")
    _ax.legend(fontsize=10)
    _ax.grid(True, alpha=0.3)
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1.05)
    _fig.tight_layout()
    return _fig


@app.cell
def calibration_fix():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_classification as _make_clf
    from sklearn.ensemble import RandomForestClassifier as _RF
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.calibration import (CalibratedClassifierCV as _CCV,
                                     calibration_curve as _cal_curve)

    _X, _y = _make_clf(n_samples=5000, n_features=20, n_informative=8,
                       weights=[0.85, 0.15], random_state=42)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.3, random_state=42, stratify=_y)

    _rf_raw   = _RF(n_estimators=100, random_state=42).fit(_X_tr, _y_tr)
    _rf_platt = _CCV(_RF(n_estimators=100, random_state=42), method="sigmoid",  cv=5).fit(_X_tr, _y_tr)
    _rf_iso   = _CCV(_RF(n_estimators=100, random_state=42), method="isotonic", cv=5).fit(_X_tr, _y_tr)

    _fig, _ax = _plt.subplots(figsize=(9, 7))
    _ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect calibration")

    for _model, _name, _color in [
        (_rf_raw,   "Random Forest (uncalibrated)",      "#E74C3C"),
        (_rf_platt, "RF + Platt Scaling (sigmoid)",      "#3498DB"),
        (_rf_iso,   "RF + Isotonic Regression",          "#27AE60"),
    ]:
        _probs = _model.predict_proba(_X_te)[:, 1]
        _frac_pos, _mean_pred = _cal_curve(_y_te, _probs, n_bins=10)
        _ax.plot(_mean_pred, _frac_pos, marker="o", markersize=7, lw=2.5,
                color=_color, label=_name)

    _ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    _ax.set_ylabel("Fraction of Positives", fontsize=12)
    _ax.set_title("Fixing Calibration: Platt Scaling vs Isotonic Regression",
                  fontsize=13, fontweight="bold")
    _ax.legend(fontsize=10)
    _ax.grid(True, alpha=0.3)
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1.05)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    > In production, always calibrate tree-based models before using their probabilities
    > for threshold-based decisions. `CalibratedClassifierCV(method='sigmoid')` (Platt scaling)
    > is the go-to; isotonic regression is more flexible but needs more positive samples (≥1000).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Choosing the Right Metric — Decision Framework

    ```
    Is the problem imbalanced (>80/20)?
    ├── YES → Don't use accuracy
    │   ├── Is cost asymmetry clear?
    │   │   ├── FN is much worse → Optimize recall (with precision floor)
    │   │   ├── FP is much worse → Optimize precision (with recall floor)
    │   │   └── Roughly equal   → Use F1
    │   └── Need threshold-independent comparison?
    │       └── Use AUC-PR (NOT AUC-ROC)
    └── NO (balanced) → Accuracy is OK, AUC-ROC is fine

    Do you use predicted probabilities for decisions?
    ├── YES → Check calibration, fix if needed (Platt / isotonic)
    └── NO  → Calibration doesn't matter
    ```

    **Bonus — Precision@K:** For ranking problems with a fixed review budget
    (e.g., "we can review 100 flagged transactions per day"), ask: "of the top K items I
    surfaced, what fraction are relevant?" More actionable than AUC-PR when capacity is constrained.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Flashcard Summary

    | Question | Answer |
    |----------|--------|
    | When does accuracy mislead? | Imbalanced classes. 99% accuracy on 1% fraud = predicting everything as not-fraud. |
    | What's precision? | TP / (TP + FP). Of predicted positives, what fraction are correct? Trustworthiness of flags. |
    | What's recall? | TP / (TP + FN). Of actual positives, what fraction did we catch? Completeness of detection. |
    | Why is F1 a harmonic mean not arithmetic? | Punishes when either P or R is near zero. (0.99 + 0.01)/2 = 0.50 but F1 = 0.02. |
    | When to use AUC-ROC vs AUC-PR? | AUC-PR for imbalanced data (<10% positive). AUC-ROC inflates when negatives dominate (FPR denominator = TN + FP is huge). |
    | What does calibration mean? | Predicted probabilities match actual frequencies. 70% prediction → 70% actual positive rate. |
    | Which models are well-calibrated by default? | Logistic regression. Trees and neural nets typically need calibration. |
    | How do you fix poor calibration? | Platt scaling (`method='sigmoid'`) or isotonic regression via `CalibratedClassifierCV`. |
    | Fraud detection: precision or recall? | Recall (don't miss fraud), but with a precision floor (don't block all customers). |
    | Practical metric for imbalanced ranking? | Precision@K — of the top K flagged, how many are actually positive? |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    ### "How do you choose an evaluation metric?"

    > Start with the cost asymmetry: FN vs FP — which is worse? That determines whether you
    > optimize precision or recall. Then check class balance — if imbalanced (>80/20), avoid
    > accuracy and AUC-ROC, use AUC-PR or precision@K instead. Always check calibration if
    > using predicted probabilities for threshold decisions.

    ---

    ### "Walk me through the precision-recall trade-off."

    > As I lower the classification threshold, I catch more positives (recall goes up) but also
    > flag more negatives (precision goes down). The optimal threshold depends on the business
    > cost of each error type — missing fraud vs. blocking legitimate customers. The PR curve
    > visualizes this full trade-off; the operating point is a business decision, not a statistical one.

    ---

    ### "Tell me about a time you chose metrics carefully."

    > In my fraud detection system design, I chose AUC-PR over AUC-ROC because with 0.1% fraud
    > rate, ROC gives misleadingly high scores. I also recommended calibrating the XGBoost model
    > before using its probabilities for the approve/review/decline threshold decisions, since
    > tree-based models are overconfident without calibration.

    ---

    ### Connection to my other work

    In my backtesting engine, I evaluate trading strategies with risk-adjusted metrics (Sharpe,
    Sortino) rather than raw returns — for the same reason you use precision/recall instead of
    accuracy. The naive metric hides the real performance characteristics. High raw return with
    high drawdown = high accuracy with 0% fraud recall. The right metric always reflects the
    actual cost structure of the problem.
    """)
    return


if __name__ == "__main__":
    app.run()
