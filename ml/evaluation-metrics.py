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

    return mo.vstack([
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
    ## Classification Metrics — Full Comparison (Pros, Cons & Scale)

    | Metric | Formula | Pros ✓ | Cons ✗ | Scalable? |
    |--------|---------|--------|--------|-----------|
    | **Accuracy** | (TP+TN)/N | Simple, universally understood | Misleads on imbalanced data — 99% acc can mean 0% recall | O(n) ✓ |
    | **Precision** | TP/(TP+FP) | Direct: "of all flags, how many are real?" | Threshold-dependent; ignores FN entirely | O(n) ✓ |
    | **Recall / TPR** | TP/(TP+FN) | Direct: "of all positives, how many caught?" | Threshold-dependent; ignores FP entirely | O(n) ✓ |
    | **F1** | 2PR/(P+R) | Single number balancing P and R; harmonic punishes extremes | Assumes P and R equally important — rarely true in practice | O(n) ✓ |
    | **F-beta** | (1+β²)PR/(β²P+R) | Tune β to reflect actual cost asymmetry (F2 = recall-heavy, F0.5 = precision-heavy) | β requires domain knowledge to set | O(n) ✓ |
    | **AUC-ROC** | Area under TPR/FPR curve | Threshold-free; great for model comparison; probabilistic interpretation | Inflated on imbalanced data — huge TN pool artificially deflates FPR | O(n log n) ✓ |
    | **AUC-PR** | Area under Precision/Recall curve | Correct for rare events; does not use TN at all | Baseline = prevalence (varies per dataset); harder to explain | O(n log n) ✓ |
    | **Log-loss** | −mean(y log ŷ + (1−y) log(1−ŷ)) | Captures calibration quality; the actual training objective | Sensitive to overconfident wrong predictions; unbounded above | O(n) ✓ |
    | **Brier Score** | mean((ŷ−y)²) | Measures calibration; bounded [0,1]; decomposable | Dominated by majority class in imbalanced setting | O(n) ✓ |
    | **MCC** | (TP·TN−FP·FN)/√(…) | Symmetric; works for multi-class; best single metric for imbalanced binary | Less intuitive to explain to stakeholders | O(n) ✓ |
    | **Precision@K** | relevant in top-K / K | Actionable when review budget is fixed (e.g., "we can review 100/day") | Rank-insensitive within top K | O(n log n) ✓ |

    **Scalability at production scale:**
    - O(n) metrics (accuracy, precision, F1, log-loss) → trivially parallelizable: just shard and aggregate counts
    - AUC metrics (O(n log n)) → sklearn handles millions of rows; at billions, use random sampling (1M sample gives ~0.001 error)
    - MCC → same cost as F1 once you have TP/FP/FN/TN counts
    - **When to escalate:** AUC-ROC → AUC-PR when positive class < 10%. F1 → MCC for extreme imbalance or multi-class.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Regression Metrics

    > **"Classification is about being right. Regression is about being *how* wrong — and in what direction."**

    Every regression metric encodes a different answer to: *"Which errors do I care about most?"*

    | Metric | Formula | Unit | Outlier Sensitivity | Minimize by predicting |
    |--------|---------|------|--------------------|-----------------------|
    | **MAE** | mean(\|y − ŷ\|) | Same as y | Low (linear) | Conditional median |
    | **MSE** | mean((y − ŷ)²) | y² | High (quadratic) | Conditional mean |
    | **RMSE** | √MSE | Same as y | High (quadratic) | Conditional mean |
    | **MAPE** | mean(\|y−ŷ\|/\|y\|)×100 | % | Low (relative) | Conditional median (approximately) |
    | **SMAPE** | mean(2\|y−ŷ\|/(|y|+|ŷ|))×100 | % | Low (symmetric) | No clean closed-form |
    | **R²** | 1 − SS_res/SS_tot | Dimensionless | High (squared) | Same as MSE |
    | **Huber(δ)** | MSE if \|e\|≤δ, else MAE shifted | Same as y | Tunable via δ | Huber M-estimator |

    **Key intuitions:**
    - **RMSE > MAE** always (Jensen's inequality). A large gap → heavy-tailed errors → outliers present.
    - **R² = 0**: model does no better than predicting the mean every time.
    - **R² < 0**: your model is actively worse than "always predict the mean" — this is possible and a red flag.
    - **MAPE breaks** when y = 0 (division by zero). Default to MAE or SMAPE for sparse targets.
    - **Huber δ** controls the boundary: small δ → behaves like MAE, large δ → behaves like MSE.
    """)
    return


@app.cell
def regression_metrics_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _rng = _np.random.default_rng(42)
    _n = 100
    _X = _np.linspace(0, 10, _n)
    _y_true = 2.5 * _X + 5 + _rng.normal(0, 3, _n)
    _outlier_idx = [10, 30, 55, 78]
    _y_true[_outlier_idx] += _rng.choice([-1, 1], size=len(_outlier_idx)) * 22

    def _ols_predict(x, y):
        _c = _np.polyfit(x, y, 1)
        return _np.polyval(_c, x)

    _y_pred = _ols_predict(_X, _y_true)
    _errors = _y_true - _y_pred

    _mae  = float(_np.mean(_np.abs(_errors)))
    _rmse = float(_np.sqrt(_np.mean(_errors ** 2)))
    _ss_res = float(_np.sum(_errors ** 2))
    _ss_tot = float(_np.sum((_y_true - _y_true.mean()) ** 2))
    _r2   = float(1 - _ss_res / _ss_tot)

    # Huber penalty
    _e_range = _np.linspace(-15, 15, 400)
    _delta = 4.0
    _huber_pen = _np.where(
        _np.abs(_e_range) <= _delta,
        0.5 * _e_range ** 2,
        _delta * (_np.abs(_e_range) - 0.5 * _delta),
    )

    _fig, _axes = _plt.subplots(1, 3, figsize=(16, 5))

    # — Left: scatter with residuals highlighted —
    _ax1 = _axes[0]
    _mask_out = _np.zeros(_n, dtype=bool)
    _mask_out[_outlier_idx] = True
    _ax1.scatter(_X[~_mask_out], _y_true[~_mask_out], color="#3498DB", alpha=0.6, s=28, label="Normal points")
    _ax1.scatter(_X[_mask_out],  _y_true[_mask_out],  color="#E74C3C", s=90,  zorder=5, label="Outliers", marker="*")
    _ax1.plot(_X, _y_pred, color="#E67E22", lw=2.5, label="OLS fit")
    for _i in _outlier_idx:
        _ax1.plot([_X[_i], _X[_i]], [_y_pred[_i], _y_true[_i]], color="#E74C3C", lw=1.5, linestyle="--", alpha=0.7)
    _ax1.set_title(f"Data with Outliers\nMAE={_mae:.1f}  RMSE={_rmse:.1f}  R²={_r2:.3f}",
                  fontsize=11, fontweight="bold")
    _ax1.set_xlabel("X"); _ax1.set_ylabel("y")
    _ax1.legend(fontsize=9); _ax1.grid(True, alpha=0.3)

    # — Center: penalty curves —
    _ax2 = _axes[1]
    _ax2.plot(_e_range, _np.abs(_e_range),          color="#3498DB", lw=2.5, label="MAE  |e|")
    _ax2.plot(_e_range, _e_range ** 2 / 8,          color="#E74C3C", lw=2.5, label="MSE  e²/8 (scaled)")
    _ax2.plot(_e_range, _huber_pen / 4,              color="#27AE60", lw=2.5, linestyle="--", label=f"Huber (δ={_delta}) /4")
    _ax2.annotate("MSE explodes\nfor outliers",
                 xy=(12, 18), xytext=(5, 14), fontsize=8, color="#E74C3C",
                 arrowprops=dict(arrowstyle="->", color="#E74C3C"))
    _ax2.set_xlim(-15, 15); _ax2.set_ylim(0, 20)
    _ax2.set_xlabel("Error  (y − ŷ)", fontsize=11); _ax2.set_ylabel("Penalty", fontsize=11)
    _ax2.set_title("Penalty Curves\nHuber = best of both worlds", fontsize=11, fontweight="bold")
    _ax2.legend(fontsize=9); _ax2.grid(True, alpha=0.3)

    # — Right: residual histogram — mean vs median gap reveals outlier influence —
    _ax3 = _axes[2]
    _ax3.hist(_errors, bins=22, color="#9B59B6", alpha=0.75, edgecolor="white")
    _ax3.axvline(0, color="k", lw=1.5, linestyle="--")
    _ax3.axvline(float(_errors.mean()),   color="#E74C3C", lw=2.2, label=f"Mean  = {_errors.mean():.1f}  ← MSE minimizer")
    _ax3.axvline(float(_np.median(_errors)), color="#3498DB", lw=2.2, label=f"Median = {_np.median(_errors):.1f}  ← MAE minimizer")
    _ax3.set_xlabel("Residual", fontsize=11); _ax3.set_ylabel("Count", fontsize=11)
    _ax3.set_title("Residual Distribution\nGap between mean and median = outlier influence",
                  fontsize=11, fontweight="bold")
    _ax3.legend(fontsize=9); _ax3.grid(True, alpha=0.3)

    _fig.suptitle("Regression Metrics: How Outliers Shift MAE vs MSE vs Huber",
                 fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ### Regression Metrics — Pros, Cons & Scalability

    | Metric | Pros ✓ | Cons ✗ | Scalable? | Interview tip |
    |--------|--------|--------|-----------|---------------|
    | **MAE** | Robust to outliers; interpretable in original units | Not differentiable at 0 (gradient issues); doesn't penalize catastrophic errors | O(n) ✓ | "Use MAE when outliers are real signal, not noise — e.g., demand spikes" |
    | **MSE** | Differentiable everywhere; standard training loss; penalizes large errors heavily | Unit is y² (unintuitive); one outlier can dominate the whole metric | O(n) ✓ | "MSE is my training loss; RMSE is my reported metric" |
    | **RMSE** | Same units as target; penalizes large errors; standard benchmark | Sensitive to outliers; optimizes mean not median | O(n) ✓ | "RMSE >> MAE signals heavy-tailed errors — go check for outliers" |
    | **MAPE** | Scale-free; percentage is intuitive; compare across different datasets | Undefined at y=0; biased toward underestimates; asymmetric | O(n) ✓ | "I avoid MAPE for demand forecasting — zero-sales days blow it up" |
    | **SMAPE** | Symmetric; handles near-zero y better than MAPE | Still breaks at exact zeros; less standardized | O(n) ✓ | "SMAPE is MAPE's more stable sibling" |
    | **R²** | Intuitive % of variance explained; automatic baseline comparison | Increases with features (use Adj. R²); meaningless to compare across datasets | O(n) ✓ | "R²=0.85 means the model explains 85% of variance vs. always predicting the mean" |
    | **Huber** | Robust AND differentiable; best of both worlds | Hyperparameter δ to tune; less standardized | O(n) ✓ | "Huber is my default when I suspect label noise in regression" |

    > **Diagnostic rule:** if RMSE/MAE ratio > 1.5, you have outliers dominating MSE. Investigate before choosing which metric to optimize.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Ranking Metrics

    > **"In search, recommendations, and retrieval — the order matters as much as the set."**

    Ranking metrics answer: *not just "did we find it?" but "did we surface it early enough?"*

    A correct result at position 1 is worth far more than the same result at position 100.

    **Precision@K** = (relevant items in top K) / K
    - "Of the first K results shown, what fraction were relevant?"
    - Pros: simple, maps directly to fixed review budgets ("we review top 100/day")
    - Cons: rank-insensitive within K (pos 1 = pos K if both relevant); ignores everything below K

    **Recall@K** = (relevant items in top K) / (total relevant items)
    - "Of all relevant items, what fraction did we surface in top K?"
    - Pros: completeness-focused; essential for legal discovery, medical retrieval
    - Cons: trivially gamed by inflating K; doesn't penalize returning irrelevant items

    **MRR (Mean Reciprocal Rank)** = mean(1 / rank of first relevant result)
    - "How far down did the user have to scroll to find *something* useful?"
    - Pros: fast to compute; captures first-hit quality
    - Cons: ignores everything after the first relevant result

    **MAP (Mean Average Precision)** = mean of per-query Average Precision
    - AP = mean of P@k at each position k where a relevant item appears
    - Pros: rank-aware; classical IR benchmark; rewards getting relevant items earlier
    - Cons: binary relevance only — can't express "somewhat relevant"

    **NDCG@K (Normalized Discounted Cumulative Gain)**
    - DCG = Σ relevanceᵢ / log₂(rankᵢ + 1) — position discount makes rank 1 worth ~14× rank 10
    - NDCG = DCG / IDCG (ideal DCG = DCG of the perfect ranking), normalizes to [0, 1]
    - Pros: graded relevance (0=bad, 1=ok, 2=great, 3=perfect); gold standard in industry
    - Cons: requires relevance grades (labeling cost); more complex to implement and explain
    """)
    return


@app.cell
def ranking_metrics_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _fig, _axes = _plt.subplots(1, 3, figsize=(16, 6))

    # — Left: position discount curve —
    _ax1 = _axes[0]
    _pos = _np.arange(1, 21)
    _disc = 1.0 / _np.log2(_pos + 1)
    _colors_disc = _plt.cm.RdYlGn(_disc / _disc.max())
    _ax1.bar(_pos, _disc, color=_colors_disc, edgecolor="white", alpha=0.9)
    _ax1.set_xlabel("Rank Position", fontsize=11)
    _ax1.set_ylabel("Discount  1/log₂(pos+1)", fontsize=11)
    _ax1.set_title("NDCG Position Discount\nPos 1 is worth ~14× pos 10",
                  fontsize=11, fontweight="bold")
    _ax1.annotate(f"Rank 1: {_disc[0]:.2f}",
                 xy=(1, _disc[0]), xytext=(4, 0.88),
                 fontsize=9, arrowprops=dict(arrowstyle="->"))
    _ax1.annotate(f"Rank 10: {_disc[9]:.2f}",
                 xy=(10, _disc[9]), xytext=(12, 0.28),
                 fontsize=9, arrowprops=dict(arrowstyle="->"))
    _ax1.grid(True, axis="y", alpha=0.3)

    # — Center: ranked list comparison (three systems) —
    _ax2 = _axes[1]
    _K = 10
    # graded relevance: 0=irrelevant, 1=ok, 2=good, 3=perfect
    _sys_A = [3, 2, 2, 1, 0, 1, 0, 0, 1, 0]
    _sys_B = [0, 0, 1, 2, 3, 1, 0, 2, 0, 0]
    _sys_C = [2, 0, 3, 0, 1, 0, 2, 0, 0, 1]

    def _dcg(rels):
        return sum(r / _np.log2(i + 2) for i, r in enumerate(rels))

    def _ndcg(rels):
        _ideal = sorted(rels, reverse=True)
        _idcg = _dcg(_ideal)
        return _dcg(rels) / _idcg if _idcg > 0 else 0.0

    _color_map = {0: "#ECF0F1", 1: "#F9E79F", 2: "#F39C12", 3: "#27AE60"}
    _sys_defs = [
        ("System A  (NDCG={:.3f})".format(_ndcg(_sys_A)), _sys_A, "#27AE60"),
        ("System B  (NDCG={:.3f})".format(_ndcg(_sys_B)), _sys_B, "#E74C3C"),
        ("System C  (NDCG={:.3f})".format(_ndcg(_sys_C)), _sys_C, "#F39C12"),
    ]
    _y_starts = [0.68, 0.38, 0.08]
    _bh = 0.22
    for _y0, (_label, _rels, _lc) in zip(_y_starts, _sys_defs):
        _ax2.text(-0.3, _y0 + _bh / 2, _label, ha="right", va="center",
                 fontsize=8, fontweight="bold", color=_lc)
        for _p, _r in enumerate(_rels):
            _rect = _plt.Rectangle([_p, _y0], 0.88, _bh,
                                   facecolor=_color_map[_r], edgecolor="white", lw=2)
            _ax2.add_patch(_rect)
            _ax2.text(_p + 0.44, _y0 + _bh / 2, str(_r),
                     ha="center", va="center", fontsize=9, fontweight="bold")
    _ax2.set_xlim(-3.8, 10); _ax2.set_ylim(0, 0.98)
    _ax2.set_xticks(range(_K))
    _ax2.set_xticklabels([f"@{i+1}" for i in range(_K)], fontsize=8)
    _ax2.set_yticks([])
    _ax2.set_title("Three Systems, Same Results — Different Positions\n0=irrelevant  1=ok  2=good  3=perfect",
                  fontsize=11, fontweight="bold")
    _ax2.set_xlabel("Rank Position", fontsize=11)
    _ax2.grid(True, axis="x", alpha=0.2)

    # — Right: metric comparison bar chart —
    _ax3 = _axes[2]

    def _p_at_k(rels, k=5):
        return sum(1 for r in rels[:k] if r > 0) / k

    def _mrr(rels):
        for i, r in enumerate(rels):
            if r > 0:
                return 1.0 / (i + 1)
        return 0.0

    _metric_labels = ["Precision@5", "MRR", "NDCG@10"]
    _metric_fns    = [_p_at_k, _mrr, _ndcg]
    _bar_colors    = ["#27AE60", "#E74C3C", "#F39C12"]
    _sys_labels    = ["Sys A\n(good)", "Sys B\n(poor)", "Sys C\n(mixed)"]

    _x_pos = _np.arange(len(_metric_labels))
    _w = 0.25
    for _idx, (_rels, _bc, _sl) in enumerate(zip([_sys_A, _sys_B, _sys_C], _bar_colors, _sys_labels)):
        _vals = [fn(_rels) for fn in _metric_fns]
        _bars = _ax3.bar(_x_pos + (_idx - 1) * _w, _vals, _w,
                        label=_sl, color=_bc, alpha=0.85, edgecolor="white")
        for _bar, _val in zip(_bars, _vals):
            _ax3.text(_bar.get_x() + _bar.get_width() / 2, _val + 0.01,
                     f"{_val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    _ax3.set_xticks(_x_pos)
    _ax3.set_xticklabels(_metric_labels, fontsize=10)
    _ax3.set_ylim(0, 1.25)
    _ax3.set_ylabel("Score", fontsize=11)
    _ax3.set_title("P@5 vs MRR vs NDCG — Same Data, Different Verdict\nMetric choice can change which system wins",
                  fontsize=11, fontweight="bold")
    _ax3.legend(fontsize=9); _ax3.grid(True, axis="y", alpha=0.3)

    _fig.suptitle("Ranking Metrics: Why Position Matters and How Each Metric Captures It",
                 fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ### Ranking Metrics — Pros, Cons & Scalability

    | Metric | Pros ✓ | Cons ✗ | Scalable? | Use when |
    |--------|--------|--------|-----------|----------|
    | **Precision@K** | Simple; maps to fixed review budgets | Rank-insensitive within K | O(n log n) ✓ | "We review top 100 flagged items per day" |
    | **Recall@K** | Completeness; legal/compliance retrieval | Gamed by increasing K; ignores precision | O(n log n) ✓ | Document retrieval where missing results is costly |
    | **MRR** | Captures first-hit latency; fast to compute | Ignores all results after first relevant one | O(n log n) ✓ | Q&A / navigational search with one right answer |
    | **MAP** | Rank-aware; classic IR benchmark | Binary relevance only — can't express "somewhat relevant" | O(n log n) ✓ | Academic IR benchmarks; binary-label evaluation |
    | **NDCG** | Graded relevance; normalized; industry standard (Netflix, Spotify, Google) | Requires relevance grades (labeling cost); more complex | O(n log n) ✓ | Recommender systems; search with graded labels |
    | **Hit Rate@K** | Simple; "did we include the clicked item?" | Doesn't distinguish rank within top K | O(n log n) ✓ | Collaborative filtering; implicit feedback systems |

    **Production scalability notes:**
    - All ranking metrics require sorting → O(n log n) per query; trivially parallelizable across queries
    - At millions of queries: compute on a stratified sample (10K queries), average — error < 0.001 for NDCG
    - Online A/B testing: use proxy metrics (CTR, dwell time, conversion) — cheaper and real-signal
    - NDCG vs MAP in industry: MAP is common in academic IR papers; NDCG is the default in production recsys
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Clustering & Unsupervised Metrics

    When there are no ground-truth labels, evaluation becomes harder — you can only measure internal consistency or compare against held-out structure.

    **Internal metrics** (no ground truth):

    | Metric | Measures | Range | Better | Complexity | Caveat |
    |--------|---------|-------|--------|-----------|--------|
    | **Silhouette Score** | Cohesion vs. separation for each point | [−1, 1] | Higher | O(n²) — slow! | Assumes convex clusters; fails for non-spherical shapes |
    | **Davies-Bouldin Index** | Avg ratio of within-cluster scatter to between-cluster distance | [0, ∞) | Lower | O(nk) | Biased toward compact, well-separated blobs |
    | **Calinski-Harabasz** | Variance ratio (between-cluster / within-cluster) | [0, ∞) | Higher | O(nk) | Favors large, dense clusters; increases with k |
    | **Inertia (WCSS)** | Within-cluster sum of squares | [0, ∞) | Lower | O(nk) | Always decreases with more clusters — use elbow method |

    **External metrics** (require ground truth labels):

    | Metric | Range | Perfect | Notes |
    |--------|-------|---------|-------|
    | **ARI** (Adjusted Rand Index) | [−1, 1] | 1.0 | Chance-corrected; 0 = random assignment |
    | **NMI** (Normalized Mutual Info) | [0, 1] | 1.0 | Handles different numbers of clusters well |
    | **Homogeneity** | [0, 1] | 1.0 | Each cluster contains only one class |
    | **Completeness** | [0, 1] | 1.0 | All instances of a class in one cluster |
    | **V-measure** | [0, 1] | 1.0 | Harmonic mean of H and C — F1 analogue for clustering |

    **Scalability warning:** Silhouette is O(n²) pairwise distances — infeasible above ~50K samples.
    At scale: run on a random subsample (5K–10K points), or switch to Davies-Bouldin which is O(nk).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## LLM Evaluation — Why It's Fundamentally Different

    > **Classification:** one right answer — compare directly.
    > **LLM generation:** infinite valid outputs — "Paris is the capital of France" and
    > "The capital of France is Paris" are both correct but score differently on n-gram metrics.

    This is the core challenge: surface-form comparison fails for generation tasks.

    ### The evaluation stack (cheapest → most reliable):

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │ Level 4: Human Evaluation      ← gold standard, expensive  │
    │          (human raters, preference A/B tests)               │
    ├─────────────────────────────────────────────────────────────┤
    │ Level 3: LLM-as-Judge          ← strong proxy, scalable    │
    │          (GPT-4 / Claude scoring generations)               │
    ├─────────────────────────────────────────────────────────────┤
    │ Level 2: Semantic Metrics      ← model-based, handles para- │
    │          (BERTScore, BLEURT)     phrase; needs GPU          │
    ├─────────────────────────────────────────────────────────────┤
    │ Level 1: Reference-Based       ← fast, cheap, brittle      │
    │          (BLEU, ROUGE)           use only where surface     │
    │                                  form matters (translation) │
    └─────────────────────────────────────────────────────────────┘
    ```

    ### Task-to-metric mapping:

    | Task | Primary | Secondary | Avoid |
    |------|---------|-----------|-------|
    | Machine translation | BLEU, BLEURT | BERTScore | Single ROUGE |
    | Summarization | ROUGE-L, BERTScore | LLM-judge (coherence, coverage) | BLEU |
    | Question answering | Exact Match, Token-F1 | BERTScore | BLEU |
    | Open-ended generation | LLM-as-judge | Human eval, G-Eval | Any n-gram metric |
    | RAG / grounded QA | RAGAS faithfulness | Context precision/recall | Unconstrained BLEU |
    | Code generation | Pass@K (unit tests) | CodeBLEU | Pure text metrics |
    | Dialog / chatbot | LLM-judge, user retention | BLEU (for templated forms) | Accuracy |
    """)
    return


@app.cell
def llm_ngram_metrics_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from collections import Counter as _Counter
    import math as _math

    def _ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def _bleu(reference, hypothesis, max_n=4):
        _ref = reference.lower().split()
        _hyp = hypothesis.lower().split()
        if not _hyp:
            return 0.0
        _bp = min(1.0, _math.exp(1 - len(_ref) / len(_hyp)))
        _scores = []
        for _n in range(1, max_n + 1):
            _ref_ng = _Counter(_ngrams(_ref, _n))
            _hyp_ng = _Counter(_ngrams(_hyp, _n))
            _clipped = sum(min(c, _ref_ng[g]) for g, c in _hyp_ng.items())
            _total = max(sum(_hyp_ng.values()), 1)
            _scores.append(_clipped / _total)
        if min(_scores) == 0:
            return 0.0
        return round(_bp * _math.exp(sum(_math.log(p) for p in _scores) / max_n), 4)

    def _rouge_n(reference, hypothesis, n=1):
        _ref = reference.lower().split()
        _hyp = hypothesis.lower().split()
        _ref_ng = _Counter(_ngrams(_ref, n))
        _hyp_ng = _Counter(_ngrams(_hyp, n))
        _overlap = sum(min(c, _hyp_ng[g]) for g, c in _ref_ng.items())
        return round(_overlap / max(sum(_ref_ng.values()), 1), 4)

    def _rouge_l(reference, hypothesis):
        _ref = reference.lower().split()
        _hyp = hypothesis.lower().split()
        _m, _n = len(_ref), len(_hyp)
        if _m == 0 or _n == 0:
            return 0.0
        _dp = [[0] * (_n + 1) for _ in range(_m + 1)]
        for _i in range(1, _m + 1):
            for _j in range(1, _n + 1):
                if _ref[_i - 1] == _hyp[_j - 1]:
                    _dp[_i][_j] = _dp[_i - 1][_j - 1] + 1
                else:
                    _dp[_i][_j] = max(_dp[_i - 1][_j], _dp[_i][_j - 1])
        _lcs = _dp[_m][_n]
        _p = _lcs / _n if _n else 0
        _r = _lcs / _m if _m else 0
        if _p + _r == 0:
            return 0.0
        return round(2 * _p * _r / (_p + _r), 4)

    _ref = "The mitochondria is the powerhouse of the cell and produces ATP through cellular respiration"

    _candidates = {
        "Perfect match":             "The mitochondria is the powerhouse of the cell and produces ATP through cellular respiration",
        "Paraphrase\n(semantic ≈)":  "Mitochondria generate ATP via cellular respiration and act as the cell power source",
        "Partial answer":            "The mitochondria produces ATP",
        "Wrong order\n(same words)": "ATP through cellular respiration the powerhouse of the cell is the mitochondria",
        "Completely wrong":          "The nucleus contains genetic material and controls cell division and protein synthesis",
        "Too short":                 "mitochondria ATP",
    }

    _labels     = list(_candidates.keys())
    _bleu_s     = [_bleu(_ref, c) for c in _candidates.values()]
    _rouge1_s   = [_rouge_n(_ref, c, 1) for c in _candidates.values()]
    _rouge2_s   = [_rouge_n(_ref, c, 2) for c in _candidates.values()]
    _rougel_s   = [_rouge_l(_ref, c) for c in _candidates.values()]

    _fig, _axes = _plt.subplots(1, 2, figsize=(16, 6))

    # Left: grouped bars
    _ax1 = _axes[0]
    _x   = _np.arange(len(_labels))
    _w   = 0.21
    _b1 = _ax1.bar(_x - 1.5*_w, _bleu_s,   _w, label="BLEU-4",  color="#3498DB", alpha=0.85, edgecolor="white")
    _b2 = _ax1.bar(_x - 0.5*_w, _rouge1_s, _w, label="ROUGE-1", color="#E67E22", alpha=0.85, edgecolor="white")
    _b3 = _ax1.bar(_x + 0.5*_w, _rouge2_s, _w, label="ROUGE-2", color="#9B59B6", alpha=0.85, edgecolor="white")
    _b4 = _ax1.bar(_x + 1.5*_w, _rougel_s, _w, label="ROUGE-L", color="#27AE60", alpha=0.85, edgecolor="white")
    for _bset in [_b1, _b2, _b3, _b4]:
        for _bar in _bset:
            _h = _bar.get_height()
            if _h > 0.03:
                _ax1.text(_bar.get_x() + _bar.get_width() / 2, _h + 0.005,
                         f"{_h:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    _ax1.set_xticks(_x)
    _ax1.set_xticklabels(_labels, rotation=22, ha="right", fontsize=9)
    _ax1.set_ylim(0, 1.25)
    _ax1.set_ylabel("Score", fontsize=11)
    _ax1.set_title("BLEU & ROUGE on 6 Candidate Outputs\nParaphrase scores nearly as low as 'completely wrong'",
                  fontsize=11, fontweight="bold")
    _ax1.legend(fontsize=9); _ax1.grid(True, axis="y", alpha=0.3)
    _ax1.annotate("Semantically correct\nbut different words →\nscores poorly!",
                 xy=(1, _rouge1_s[1] + 0.02), xytext=(2.5, 0.65),
                 fontsize=8, color="#E74C3C",
                 arrowprops=dict(arrowstyle="->", color="#E74C3C"))

    # Right: heatmap
    _ax2 = _axes[1]
    _matrix = _np.array([_bleu_s, _rouge1_s, _rouge2_s, _rougel_s])
    _im = _ax2.imshow(_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    _plt.colorbar(_im, ax=_ax2, shrink=0.8)
    _ax2.set_xticks(range(len(_labels)))
    _ax2.set_xticklabels(_labels, rotation=28, ha="right", fontsize=8)
    _ax2.set_yticks(range(4))
    _ax2.set_yticklabels(["BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"], fontsize=10)
    for _i in range(4):
        for _j in range(len(_labels)):
            _v = _matrix[_i, _j]
            _ax2.text(_j, _i, f"{_v:.2f}", ha="center", va="center",
                     fontsize=9, fontweight="bold",
                     color="white" if _v < 0.35 else "black")
    _ax2.set_title("Heatmap: n-gram Metrics Miss Semantic Equivalence\n'Paraphrase' should score near 'Perfect match'",
                  fontsize=11, fontweight="bold")

    _fig.suptitle("BLEU & ROUGE: Fast and Cheap — but Surface-Form Metrics Miss Meaning",
                 fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ### BLEU & ROUGE — Mechanics, Pros, Cons & When to Use

    **BLEU** measures n-gram *precision* (how much of the hypothesis appears in the reference):
    - BLEU = BP × exp(Σ wₙ log pₙ) where pₙ = clipped n-gram precision, BP = brevity penalty
    - Clipping: each n-gram in the hypothesis is counted at most as many times as it appears in the reference (prevents padding)
    - BLEU-4 (4-gram) is the standard; single-sentence BLEU is unreliable — always average over the corpus

    **ROUGE** measures n-gram *recall* (how much of the reference is covered):
    - **ROUGE-1:** unigram recall — broad coverage signal
    - **ROUGE-2:** bigram recall — phrase-level signal
    - **ROUGE-L:** F1 of Longest Common Subsequence — word order matters but allows gaps

    | Metric | Precision/Recall | Order-sensitive | Multi-reference | Scalable | Best for |
    |--------|-----------------|-----------------|-----------------|---------|---------|
    | BLEU-4 | Precision | Partially (BP) | Yes (take max) | O(n) ✓ | Machine translation |
    | ROUGE-1 | Recall | No | Yes | O(n) ✓ | Summarization coverage |
    | ROUGE-2 | Recall | No | Yes | O(n) ✓ | Summarization fluency |
    | ROUGE-L | F1 (LCS) | Yes | Yes | O(nm) ✓ | Summarization; sentence-level |
    | BERTScore | F1 (cosine sim) | No | Single | O(n·d) GPU | Any generation task |

    > **The key failure mode:** a semantically perfect paraphrase scores near 0 on BLEU and low on ROUGE.
    > Reserve n-gram metrics for tasks where surface form genuinely matters (translation, template filling).
    > For anything requiring understanding of meaning, use BERTScore or LLM-as-judge.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## BERTScore — Semantic Similarity via Contextual Embeddings

    BERTScore fixes the paraphrase problem by comparing *representations* not *tokens*.

    ### How it works:
    1. Encode every token in reference and hypothesis using a pre-trained BERT/RoBERTa model
    2. For each hypothesis token, find its most similar reference token (greedy max cosine similarity)
    3. **Precision:** mean of max-similarity for each hypothesis token → "how much of what I said is in the reference?"
    4. **Recall:** mean of max-similarity for each reference token → "how much of the reference did I cover?"
    5. **F1:** harmonic mean of BERTScore-P and BERTScore-R

    ```python
    # pip install bert-score
    from bert_score import score
    P, R, F1 = score(candidates, references, lang="en", model_type="roberta-large")
    # Returns tensors of shape (num_samples,)
    ```

    ### Pros, Cons & Scalability:

    | Aspect | BLEU / ROUGE | BERTScore |
    |--------|-------------|-----------|
    | Paraphrase handling | Poor | Good |
    | Human correlation | ~0.5–0.6 | ~0.7–0.8 |
    | Speed | Very fast (CPU, O(n)) | Slower (GPU needed for scale) |
    | Interpretability | High — count n-grams | Lower — lives in embedding space |
    | Domain sensitivity | None | Must pick the right BERT variant |
    | Cost at 1M samples | Negligible | ~1–2 GPU-hours |
    | Multi-language | Via multi-ref tricks | xlm-roberta-large natively |

    **Model selection guide:**
    - General text: `roberta-large` (default)
    - Code: `microsoft/codebert-base`
    - Multilingual: `xlm-roberta-large`
    - Biomedical: `allenai/scibert_scivocab_uncased`

    **Scalability tip:** pre-compute and cache reference embeddings. At eval time, only encode candidates — halves compute for fixed test sets.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## LLM-as-Judge — Scalable Human-Quality Evaluation

    Use a powerful LLM (GPT-4, Claude Sonnet) to evaluate other model outputs.
    Human agreement: **~80–85%** — far better than BLEU/ROUGE (~50–65%).

    ### Two evaluation modes:

    **Pointwise (absolute scoring):**
    ```
    Rate the following answer on each dimension from 1 (poor) to 5 (excellent):
    - Correctness: Does it answer the question accurately?
    - Completeness: Does it cover all relevant aspects?
    - Conciseness: Is it appropriately brief?

    Question: {question}
    Reference answer: {reference}
    Model answer: {answer}

    Respond with JSON: {"correctness": X, "completeness": X, "conciseness": X, "reasoning": "..."}
    ```

    **Pairwise (comparative — higher reliability):**
    ```
    Which answer better addresses the question? Respond with "A", "B", or "tie".
    Reason step-by-step before your verdict.

    Question: {question}
    Answer A: {answer_a}
    Answer B: {answer_b}
    ```

    ### G-Eval: chain-of-thought scoring
    Ask the judge to reason before scoring → improves reliability by ~10% vs. direct scoring.
    Ask for a probability distribution over scores, not just a single score → more signal.

    ### Pros, Cons & Scalability:

    | Aspect | Detail |
    |--------|--------|
    | **Pros** | High human correlation; flexible criteria; handles open-ended tasks |
    | **Positional bias** | In pairwise, favors the first answer shown → always swap A/B and average |
    | **Verbose bias** | Longer answers rated higher even if less accurate → penalize verbosity explicitly |
    | **Self-preference** | GPT-4 rates GPT-4 outputs higher → use a different judge model |
    | **Cost** | ~$0.01–0.05/eval at GPT-4 pricing. 10K evals ≈ $100–500 |
    | **Scale** | Async API calls in parallel; cache judge responses; deduplicate identical outputs |
    | **When to use** | Open-ended generation, instruction following, chatbot quality, creative tasks |
    | **When to avoid** | Exact-match tasks (use EM); structured outputs (use schema validation + EM) |

    > **Production eval cadence:** run BLEU/ROUGE on every PR (fast regression check), LLM-judge
    > on a 5% sample nightly (quality signal), human eval on major releases or product launches.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## RAG Evaluation — The RAGAS Framework

    RAG has two independent components to evaluate: **retrieval quality** and **generation quality**.
    RAGAS provides metrics for both without requiring human labels for most of them.

    ```
    User query
        │
        ▼
    [Retriever] ──→ context chunks  ←── evaluate: Context Precision, Context Recall
        │
        ▼
    [Generator] ──→ final answer    ←── evaluate: Faithfulness, Answer Relevancy
    ```

    ### RAGAS Metrics:

    | Metric | What it measures | Requires | Score |
    |--------|-----------------|---------|-------|
    | **Faithfulness** | Does the answer contain *only* claims supported by the context? | LLM judge (NLI) | [0, 1] higher = better |
    | **Answer Relevancy** | Does the answer actually address the question asked? | Embeddings (cosine sim) | [0, 1] |
    | **Context Precision** | Are the retrieved chunks actually relevant to the question? | LLM judge | [0, 1] |
    | **Context Recall** | Were all facts needed to answer the question retrieved? | Ground truth answers | [0, 1] |
    | **Context Relevance** | What fraction of retrieved context is relevant to the query? | LLM judge | [0, 1] |

    ### Debugging with RAGAS:

    ```
    Low Faithfulness      → Generator hallucinates beyond context
                            Fix: stricter system prompt ("only use provided context"); retrieval-constrained decoding
    Low Context Precision → Retriever fetches irrelevant chunks
                            Fix: better embedding model; cross-encoder reranker; smaller chunk size
    Low Context Recall    → Retriever misses relevant chunks
                            Fix: increase K; larger chunk size; hybrid search (dense + BM25)
    Low Answer Relevancy  → Generator produces generic / off-topic answers
                            Fix: better instruction prompting; add query to generation prompt explicitly
    ```

    ```python
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    # dataset: HuggingFace Dataset with columns:
    #   question, answer, contexts (list of strings), ground_truth
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    print(results)  # {'faithfulness': 0.82, 'answer_relevancy': 0.91, 'context_precision': 0.74}
    ```

    **Cost at scale:** RAGAS uses LLM calls → ~$0.005–0.02/sample. 10K eval set ≈ $50–200.
    Cache LLM responses. Run nightly, not on every commit.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Alignment & Safety Metrics — Beyond Quality

    For production LLMs, quality is necessary but not sufficient.

    | Metric | What it measures | How measured | Tools |
    |--------|-----------------|-------------|-------|
    | **Hallucination Rate** | % of outputs with ungrounded factual claims | LLM NLI judge; FactScore | RAGAS faithfulness, FactScore |
    | **Groundedness** | Does answer stay within provided context? | NLI: premise=context, hypothesis=answer | RAGAS, Azure AI Evaluation |
    | **Toxicity** | % of outputs with harmful/offensive content | Classifier (Perspective API, Detoxify) | `pip install detoxify` |
    | **Refusal Rate** | Does model decline genuinely harmful requests? | Red-team prompts + classifier | Custom; Anthropic eval suite |
    | **Instruction Following** | Does output match format/length/style instructions? | LLM-judge on rubric; regex for structure | IFEval benchmark |
    | **Bias / Fairness** | Demographic disparities in outputs | Counterfactual pairs; group disparity metrics | BOLD, WinoBias |
    | **Calibration (LLM)** | Does expressed confidence match actual accuracy? | ECE on verbalized confidence scores | Custom eval harness |

    ### FactScore — atomic fact verification:
    ```
    Long-form output
        → decompose into atomic facts (LLM): ["Mitochondria produce ATP", "ATP = adenosine triphosphate", ...]
        → verify each fact against a knowledge source (Wikipedia, retrieval index)
        → FactScore = % of atomic facts that are supported
    ```
    Key advantage: granular — tells you *which* facts are wrong, not just "this output is bad."

    ### Production monitoring checklist:
    - **Hallucination rate** — track per query type; alert if rate rises above baseline
    - **Toxicity** — real-time classifier on every output before serving; block + log
    - **Groundedness** — especially critical for RAG; cite sources so users can verify
    - **Latency** — P50/P95/P99 end-to-end; quality means nothing if the system is too slow
    - **User feedback** — thumbs up/down, retry rate, session abandonment — free signal
    - **Drift** — monitor metric distributions over time; input distribution shift degrades all metrics
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Choosing the Right Metric — Decision Framework

    ```
    ┌─── What type of problem? ─────────────────────────────────────────┐
    │                                                                    │
    │  CLASSIFICATION                                                    │
    │  ├─ Imbalanced (>80/20)?                                          │
    │  │   ├─ YES → skip accuracy; use AUC-PR or F-beta                 │
    │  │   │        FN >> FP? → recall / F2                             │
    │  │   │        FP >> FN? → precision / F0.5                        │
    │  │   └─ NO  → accuracy OK; AUC-ROC fine                          │
    │  └─ Using probabilities for decisions? → check calibration        │
    │       (Platt/isotonic for trees; LR is already calibrated)        │
    │                                                                    │
    │  REGRESSION                                                        │
    │  ├─ Outliers in data?         → MAE or Huber                      │
    │  ├─ Large errors catastrophic? → RMSE or MSE                      │
    │  ├─ Need scale-free comparison? → MAPE (if y ≠ 0) / SMAPE        │
    │  └─ Explaining variance to stakeholders? → R²                     │
    │                                                                    │
    │  RANKING / RETRIEVAL                                               │
    │  ├─ Fixed review budget? → Precision@K                            │
    │  ├─ One right answer (Q&A)? → MRR                                 │
    │  ├─ Graded relevance available? → NDCG (industry standard)        │
    │  └─ Binary labels only? → MAP                                     │
    │                                                                    │
    │  CLUSTERING                                                        │
    │  ├─ No ground truth → Silhouette (small n), Davies-Bouldin (large)│
    │  └─ Ground truth available → ARI or NMI                           │
    │                                                                    │
    │  LLM / GENERATION                                                  │
    │  ├─ Translation / templated output → BLEU, ROUGE                  │
    │  ├─ Summarization → ROUGE-L, BERTScore                            │
    │  ├─ Open-ended / creative → LLM-as-judge (pointwise or pairwise)  │
    │  ├─ RAG system → RAGAS (faithfulness + context precision/recall)   │
    │  ├─ Code generation → Pass@K (unit tests)                         │
    │  └─ Safety-critical → Hallucination rate, toxicity, groundedness  │
    └────────────────────────────────────────────────────────────────────┘
    ```

    **Cross-cutting rules:**
    - Never optimize a proxy metric that doesn't match business cost — always trace metric back to FP/FN costs
    - Report multiple metrics: headline metric + sanity-check metric (e.g., NDCG + coverage)
    - At scale: O(n) metrics are free; O(n²) metrics (Silhouette, pairwise BLEU) need sampling strategies
    - For LLMs in production: BLEU/ROUGE for CI regression tests, LLM-judge on 5% sample nightly, human eval for releases
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Flashcard Summary

    | Question | Answer |
    |----------|--------|
    | When does accuracy mislead? | Imbalanced classes. 99% accuracy on 1% fraud dataset = predicting nothing as fraud. |
    | What's precision? | TP/(TP+FP). Of predicted positives, what fraction are correct? Trustworthiness of flags. |
    | What's recall? | TP/(TP+FN). Of actual positives, what fraction did we catch? Completeness of detection. |
    | Why F1 harmonic not arithmetic? | Punishes extremes. (0.99+0.01)/2 = 0.50 but F1 = 0.02. Arithmetic hides failure. |
    | AUC-ROC vs AUC-PR? | AUC-PR for imbalanced (<10% positive). ROC inflates because huge TN pool deflates FPR. |
    | What does calibration mean? | Predicted probability matches actual frequency. 70% prediction → 70% actually positive. |
    | Which models are uncalibrated? | Random Forest, Gradient Boosting, naive Bayes, deep nets. Logistic regression is calibrated by design. |
    | How to fix calibration? | Platt scaling (sigmoid) or isotonic regression via CalibratedClassifierCV. |
    | MAE vs RMSE — which to use? | RMSE > MAE when outliers present. If RMSE/MAE > 1.5, investigate outliers. Use MAE for robustness. |
    | MAPE failure case? | y = 0 → division by zero. Use SMAPE or MAE for sparse/zero targets. |
    | What does R² = 0 mean? | Model no better than predicting the mean. R² < 0 = worse than mean. |
    | Huber loss advantage? | Robust to outliers AND differentiable everywhere. MAE is non-differentiable at 0; MSE explodes. |
    | NDCG vs MAP? | NDCG supports graded relevance; MAP is binary only. NDCG = industry standard (recsys). |
    | MRR best for what? | Navigational queries / Q&A where there's one right answer you need to surface first. |
    | Precision@K vs NDCG? | P@K is rank-insensitive within K; NDCG rewards putting best results first. |
    | Why does BLEU fail on paraphrases? | Measures surface n-gram precision — different words = low score even if meaning is identical. |
    | ROUGE-1 vs ROUGE-L? | ROUGE-1: unigram recall (coverage). ROUGE-L: LCS-based F1 (word order somewhat matters). |
    | When to use BERTScore? | Any generation task where paraphrasing is valid. Needs GPU; ~7–8× more correlated with humans than BLEU. |
    | LLM-as-judge biases? | Positional (favors first in pairwise), verbose (longer = better), self-preference. Swap order; use multiple judges. |
    | RAGAS faithfulness = ? | % of answer claims that are supported by the retrieved context. Low → generator is hallucinating. |
    | RAGAS context precision = ? | % of retrieved chunks that are relevant. Low → retriever fetching junk; fix embedding model or add reranker. |
    | Silhouette scalability limit? | O(n²) pairwise distances — infeasible above ~50K samples. Subsample or use Davies-Bouldin (O(nk)). |
    | ARI vs NMI for clustering? | ARI is chance-corrected (better for comparing); NMI handles different cluster counts better. |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    ### "How do you choose an evaluation metric?"

    > Start with the cost asymmetry: FN vs FP — which error is more expensive? That determines
    > whether to optimize precision or recall. Then check class balance — if imbalanced (>80/20),
    > avoid accuracy and AUC-ROC, use AUC-PR or precision@K. Check calibration if using probabilities
    > for threshold or expected-value decisions. For regression, ask whether outliers are real signal
    > or noise — that determines MAE vs RMSE vs Huber. For LLMs, ask whether surface form matters
    > (use BLEU/ROUGE) or meaning matters (use BERTScore or LLM-as-judge).

    ---

    ### "Walk me through the precision-recall trade-off."

    > As I lower the classification threshold, I catch more positives (recall goes up) but also flag
    > more negatives as positive (precision goes down). The optimal threshold is determined by the
    > *relative cost* of each error type — missing fraud vs. blocking a real customer. The PR curve
    > visualizes this full trade-off; the operating point is a business decision, not a statistical one.
    > The AUC-PR summarizes this curve as a single number — better than AUC-ROC for imbalanced data
    > because it doesn't use TN in its denominator.

    ---

    ### "Tell me about the difference between BLEU and BERTScore."

    > BLEU measures n-gram precision — how many word sequences in my output appear in the reference.
    > It's fast and cheap but completely fails on paraphrases: "Paris is the capital of France" and
    > "The capital of France is Paris" score near zero BLEU against each other. BERTScore instead
    > computes contextual embeddings for every token and takes greedy cosine similarity — so semantically
    > similar tokens match even if they're different words. BERTScore correlates with human judgment
    > at ~0.75 vs BLEU's ~0.55. The tradeoff: BERTScore needs GPU and is ~50× slower, so in practice
    > I use BLEU/ROUGE for CI regression checks and BERTScore for deeper offline evaluation.

    ---

    ### "How would you evaluate a RAG system?"

    > I'd decompose it into retrieval and generation, evaluate each independently with RAGAS.
    > Faithfulness measures whether the generated answer sticks to what the context actually says —
    > low faithfulness means the generator is hallucinating beyond the retrieved documents.
    > Context precision checks if the retriever is fetching relevant chunks — low precision means
    > we're wasting context window on noise. Context recall checks if we retrieved everything needed —
    > low recall means the answer will be incomplete. That diagnostic decomposition tells me exactly
    > where to focus: retriever problems (embedding model, chunk size, reranker) vs. generator problems
    > (prompt constraints, decoding strategy).

    ---

    ### "What's the scalability concern with ranking metrics?"

    > All ranking metrics require sorting — O(n log n) per query — which is fine for a single query
    > but becomes expensive at millions of queries. In practice: compute offline on a stratified sample
    > of 10K queries, which gives NDCG error < 0.001. For online A/B testing I use proxy metrics instead —
    > CTR, dwell time, conversion — because they're real user signal and cheap to collect at scale.
    > NDCG is for offline model comparison; online proxies are for production decisions.

    ---

    ### Connection to my other work

    In my backtesting engine I evaluate trading strategies with risk-adjusted metrics (Sharpe, Sortino,
    max drawdown) rather than raw returns — for the same reason I use precision/recall instead of accuracy.
    High raw return with high drawdown = high accuracy with 0% fraud recall. The naive metric hides the
    actual cost structure. Picking the right metric is always about encoding what "expensive errors" means
    in your specific domain.
    """)
    return


if __name__ == "__main__":
    app.run()

