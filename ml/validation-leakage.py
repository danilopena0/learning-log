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
    # Cross-Validation, Data Leakage, Train/Val/Test Splits

    **Common Mistakes That Fail Interviews**

    | Field  | Value |
    |--------|-------|
    | Date   | 2026-04-15 |
    | Track  | ML Theory |
    | Time   | 60 min |
    | Topics | CV · Leakage · Train/Val/Test · TimeSeriesSplit · GroupKFold |
    """)
    return


@app.cell
def why_this_wins(mo):
    mo.md("""
    ## Why This Topic Wins Interviews

    Most candidates can fit a model. Few can validate it correctly.

    Interviewers test this with traps: *"Here's a dataset and a metric — train a model."*
    If you don't ask about the split strategy, you've already lost. The asymmetry is brutal:

    - **Catching leakage** in an interview signals senior-level thinking.
    - **Missing leakage** signals junior — even if your model architecture is technically better.

    > **"If your validation strategy is wrong, your model metrics are lies."**

    This is high ROI to know cold. Nearly every DS/ML interview gotcha is about leakage or
    improper splits. Fix those and you eliminate a whole class of failure modes.
    """)
    return


@app.cell
def three_way_split(mo):
    mo.md("""
    ## Train / Val / Test — The Three-Way Split

    | Set | Purpose |
    |-----|---------|
    | **Training** | Model learns weights from this data |
    | **Validation** | Model selection, hyperparameter tuning, early stopping. You optimize *against* this set. |
    | **Test** | Held-out, untouched. Final unbiased estimate of generalization. Touch **once** at the end. |

    **Common ratios:**
    - 60/20/20 or 70/15/15 for medium datasets
    - 98/1/1 for huge datasets (1% of 100M is still 1M samples)

    ### Why three sets, not two?

    If you tune hyperparameters using test set feedback — even implicitly — you've leaked the test
    set into model selection. Every decision about your model eats into the test set's independence.

    **Rule:** every decision you make about your model = uses validation. Test set is for the final
    report ONLY.

    **The cardinal sin:** looking at test metrics, then "fixing" the model. Now your test set is
    contaminated. You'll report a metric that's more optimistic than true generalization.
    """)
    return


@app.cell
def cross_validation_concept(mo):
    mo.md("""
    ## Cross-Validation — When You Don't Have Enough Data

    A single train/val split = one estimate of performance. It could be lucky or unlucky depending
    on which 20% ended up in the validation set.

    **K-fold CV:** split data into K equal folds. Train K times — each time one fold is validation,
    the rest are training. Average the K scores.

    | Variant | When to use |
    |---------|-------------|
    | **K-Fold** (K=5 or 10) | Default. Higher K = more compute, less variance in the estimate. |
    | **Stratified K-Fold** | Preserves class proportions in each fold. Essential for imbalanced data. |
    | **LOOCV** (K=N) | Very expensive, high variance. Rarely used in practice. |

    ### CV vs single split

    | Use CV when... | Use single split when... |
    |----------------|--------------------------|
    | Small dataset (<10K rows) | Large dataset (CV is expensive overkill) |
    | Need robust performance estimate | Time-series data (CV breaks temporal order) |
    | Doing hyperparameter search | Production model where you'll keep iterating |
    """)
    return


@app.cell
def cv_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_classification as _make_clf
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.model_selection import StratifiedKFold as _SKF, cross_val_score as _cvs
    from sklearn.preprocessing import StandardScaler as _SS

    _rng = _np.random.default_rng(42)
    _X, _y = _make_clf(n_samples=500, n_features=20, n_informative=6,
                       n_redundant=4, random_state=42)
    _X = _SS().fit_transform(_X)

    _skf = _SKF(n_splits=5, shuffle=True, random_state=42)
    _scores = _cvs(_LR(max_iter=1000, random_state=42), _X, _y,
                   cv=_skf, scoring="roc_auc")

    _fig, _ax = _plt.subplots(figsize=(9, 4))
    _colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]
    _bars = _ax.bar([f"Fold {i+1}" for i in range(5)], _scores,
                    color=_colors, alpha=0.85, edgecolor="white", width=0.55)
    _ax.axhline(_scores.mean(), color="#2C3E50", lw=2.5, linestyle="--",
                label=f"Mean = {_scores.mean():.3f}")
    _ax.axhspan(_scores.mean() - _scores.std(), _scores.mean() + _scores.std(),
                alpha=0.12, color="#2C3E50", label=f"± 1 std = {_scores.std():.3f}")
    for _bar, _score in zip(_bars, _scores):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _score + 0.005,
                 f"{_score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    _ax.set_ylim(0.6, 1.0)
    _ax.set_ylabel("ROC-AUC", fontsize=12)
    _ax.set_title("5-Fold Stratified CV — Fold scores vary. That variance IS the point.",
                  fontsize=12, fontweight="bold")
    _ax.legend(fontsize=10)
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()

    # NOTE: Mean tells you expected performance. Std tells you how stable it is.
    # High std = your model is sensitive to which data it sees.
    return _fig, _scores


@app.cell
def leakage_intro(mo):
    mo.md("""
    ## The 7 Most Common Forms of Data Leakage

    **Data leakage** = your model has access to information at training time that it
    **won't have at inference time**.

    Symptom: model crushes validation/test, fails catastrophically in production.

    > *"If your validation accuracy seems too good to be true, it probably is. Look for leakage."*

    The 7 types below cover ~90% of leakage bugs seen in real production ML systems and
    in DS/ML interviews. Know each one cold.
    """)
    return


@app.cell
def leak1_concept(mo):
    mo.md("""
    ### Leak 1: Target Leakage

    **Definition:** a feature directly contains the target, or a near-perfect proxy for it.

    **Example:** predicting customer churn, but one of your features is `days_since_account_closed`
    — a field only populated for *churned* customers. The model trivially learns: if this field is
    non-null → churn. At inference time on active customers, the field is always null, so the model
    is useless.

    **Detection signals:**
    - AUC suspiciously close to 1.0
    - One feature has dominant importance — everything else is nearly zero
    - Model degrades drastically on holdout or production

    **Fix:** audit every feature. Ask *"would this value be available BEFORE the event we're predicting?"*
    Apply the timeline test: draw a line at prediction time. Any feature that crosses that line is suspect.
    """)
    return


@app.cell
def leak1_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.ensemble import GradientBoostingClassifier as _GBC
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.metrics import roc_auc_score as _auc

    _rng = _np.random.default_rng(0)
    _n = 1000

    # Legitimate features
    _age = _rng.normal(35, 10, _n)
    _tenure = _rng.normal(24, 12, _n).clip(1, None)
    _support_calls = _rng.poisson(2, _n)

    # True churn label
    _churn_prob = 1 / (1 + _np.exp(-(0.03 * _support_calls - 0.01 * _tenure + 0.5)))
    _y = (_rng.random(_n) < _churn_prob).astype(int)

    # LEAKY feature: days_since_closed — only set for churned customers
    _days_since_closed = _np.where(_y == 1, _rng.integers(1, 90, _n), 0)

    import pandas as _pd
    _X_leaky = _pd.DataFrame({
        "age": _age, "tenure": _tenure,
        "support_calls": _support_calls,
        "days_since_closed": _days_since_closed,  # THE LEAK
    })
    _X_clean = _X_leaky.drop(columns=["days_since_closed"])

    _X_tr_l, _X_te_l, _y_tr, _y_te = _tts(_X_leaky, _y, test_size=0.2, random_state=1)
    _X_tr_c, _X_te_c, _, _ = _tts(_X_clean, _y, test_size=0.2, random_state=1)

    _model_l = _GBC(n_estimators=100, random_state=42).fit(_X_tr_l, _y_tr)
    _model_c = _GBC(n_estimators=100, random_state=42).fit(_X_tr_c, _y_tr)

    _auc_leaky = _auc(_y_te, _model_l.predict_proba(_X_te_l)[:, 1])
    _auc_clean = _auc(_y_te, _model_c.predict_proba(_X_te_c)[:, 1])

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4))

    # Feature importances — leaky model
    _imp_l = _model_l.feature_importances_
    _names_l = _X_leaky.columns.tolist()
    _axes[0].barh(_names_l, _imp_l,
                  color=["#E74C3C" if n == "days_since_closed" else "#3498DB" for n in _names_l])
    _axes[0].set_title(f"Leaky Model  (AUC = {_auc_leaky:.3f})\n"
                       f"'days_since_closed' dominates — red flag!", fontsize=11, fontweight="bold")
    _axes[0].set_xlabel("Feature Importance")

    # Feature importances — clean model
    _imp_c = _model_c.feature_importances_
    _names_c = _X_clean.columns.tolist()
    _axes[1].barh(_names_c, _imp_c, color="#2ECC71")
    _axes[1].set_title(f"Clean Model  (AUC = {_auc_clean:.3f})\n"
                       f"Honest performance — realistic!", fontsize=11, fontweight="bold")
    _axes[1].set_xlabel("Feature Importance")

    _fig.suptitle("Target Leakage: AUC drops from {:.3f} → {:.3f} after removing the leaky feature"
                  .format(_auc_leaky, _auc_clean), fontsize=12, fontweight="bold")
    _fig.tight_layout()
    return _fig, _auc_leaky, _auc_clean


@app.cell
def leak2_concept(mo):
    mo.md("""
    ### Leak 2: Train-Test Contamination via Preprocessing

    **Definition:** fitting a transformer (scaler, imputer, encoder, PCA) on the **full dataset**
    before splitting.

    **Why it leaks:** the scaler's mean/std is computed using test data information. The model
    indirectly "sees" the test set through the scaling parameters. Same for imputers (using test
    set means to fill gaps) and encoders (vocabulary includes test-only classes).

    **The fix is simple:** split first, then fit transformers on train only. Transform both
    train and test using the *train-derived* parameters.

    **Even simpler fix:** use `sklearn.Pipeline` — it guarantees this automatically. The scaler's
    `fit` is called only on training data at each CV fold. **This is THE reason Pipelines exist.**
    """)
    return


@app.cell
def leak2_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_classification as _make_clf
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.model_selection import cross_val_score as _cvs, StratifiedKFold as _SKF
    from sklearn.pipeline import Pipeline as _Pipeline
    from sklearn.metrics import accuracy_score as _acc

    _X, _y = _make_clf(n_samples=300, n_features=30, n_informative=5,
                       n_redundant=5, random_state=7)

    _skf = _SKF(n_splits=5, shuffle=True, random_state=42)

    # WRONG: scale full dataset first, then cross-validate
    _X_scaled_full = _SS().fit_transform(_X)
    _scores_wrong = _cvs(_LR(max_iter=1000), _X_scaled_full, _y,
                         cv=_skf, scoring="accuracy")

    # RIGHT: Pipeline handles fit/transform scoping per fold automatically
    _pipe = _Pipeline([("scaler", _SS()), ("clf", _LR(max_iter=1000))])
    _scores_right = _cvs(_pipe, _X, _y, cv=_skf, scoring="accuracy")

    _fig, _ax = _plt.subplots(figsize=(9, 4.5))
    _x = _np.arange(5)
    _w = 0.35
    _ax.bar(_x - _w / 2, _scores_wrong, _w, label=f"WRONG — scale then split  (mean={_scores_wrong.mean():.3f})",
            color="#E74C3C", alpha=0.85)
    _ax.bar(_x + _w / 2, _scores_right, _w, label=f"RIGHT — Pipeline  (mean={_scores_right.mean():.3f})",
            color="#2ECC71", alpha=0.85)
    _ax.set_xticks(_x)
    _ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
    _ax.set_ylabel("Accuracy", fontsize=12)
    _ax.set_ylim(0.5, 1.0)
    _ax.set_title("Preprocessing Leakage: Wrong pipeline inflates accuracy\n"
                  "The gap is the leakage signal — small here, large with small datasets or strong leakers",
                  fontsize=11, fontweight="bold")
    _ax.legend(fontsize=10)
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    return _fig


@app.cell
def leak3_concept(mo):
    mo.md("""
    ### Leak 3: Temporal Leakage

    **Definition:** randomly splitting time-series data, allowing future data into training.

    **Example:** predicting Q4 sales. A random split might put some Q4 rows in train and some in
    test. Your model trains on future Q4 patterns and then "predicts" them — trivially.

    **The insidious version:** any feature derived from future aggregates (rolling means that
    include future rows, lag features computed incorrectly) leaks future into past.

    **Fix:** use `sklearn.model_selection.TimeSeriesSplit`. Train on past, validate on future.
    Always. No exceptions for time-ordered data.

    **Connection to backtesting:** walk-forward validation IS TimeSeriesSplit applied to financial
    data. Random splits in backtests produce spectacular-looking Sharpe ratios that collapse to
    zero in live trading. Same principle — train on past, test on future, never the reverse.
    """)
    return


@app.cell
def leak3_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.ensemble import GradientBoostingRegressor as _GBR
    from sklearn.model_selection import train_test_split as _tts, TimeSeriesSplit as _TSS
    from sklearn.metrics import mean_absolute_error as _mae
    from sklearn.preprocessing import StandardScaler as _SS

    _rng = _np.random.default_rng(0)
    _n = 800
    _t = _np.arange(_n)

    # Time series with trend + seasonality + noise
    _y_ts = 0.05 * _t + 10 * _np.sin(2 * _np.pi * _t / 52) + _rng.normal(0, 2, _n)

    # Features: lagged values (computed correctly — using only past data)
    _X_ts = _np.column_stack([_y_ts[i: _n - (5 - i)] for i in range(5)])
    _y_aligned = _y_ts[5:]
    _t_aligned = _t[5:]

    # WRONG: random split
    _X_tr_r, _X_te_r, _y_tr_r, _y_te_r = _tts(_X_ts, _y_aligned, test_size=0.2, random_state=42)
    _model_r = _GBR(n_estimators=100, random_state=42).fit(_X_tr_r, _y_tr_r)
    _mae_random = _mae(_y_te_r, _model_r.predict(_X_te_r))

    # RIGHT: TimeSeriesSplit — last fold gives most realistic estimate
    _tscv = _TSS(n_splits=5)
    _maes_ts = []
    for _tr_idx, _te_idx in _tscv.split(_X_ts):
        _m = _GBR(n_estimators=100, random_state=42)
        _m.fit(_X_ts[_tr_idx], _y_aligned[_tr_idx])
        _maes_ts.append(_mae(_y_aligned[_te_idx], _m.predict(_X_ts[_te_idx])))
    _mae_ts = float(_np.mean(_maes_ts))

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: TimeSeriesSplit visualization
    _tscv_vis = _TSS(n_splits=5)
    _colors_fold = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#3498DB"]
    for _fold_i, (_tr_i, _te_i) in enumerate(_tscv_vis.split(_X_ts)):
        _axes[0].barh(_fold_i, len(_tr_i), left=_tr_i[0], height=0.5,
                      color="#95A5A6", alpha=0.5)
        _axes[0].barh(_fold_i, len(_te_i), left=_te_i[0], height=0.5,
                      color=_colors_fold[_fold_i], alpha=0.85,
                      label=f"Fold {_fold_i+1} test  (MAE={_maes_ts[_fold_i]:.2f})")
    _axes[0].set_xlabel("Time index", fontsize=11)
    _axes[0].set_yticks(range(5))
    _axes[0].set_yticklabels([f"Fold {i+1}" for i in range(5)])
    _axes[0].set_title("TimeSeriesSplit: train grows, test is always future",
                       fontsize=11, fontweight="bold")
    _axes[0].legend(fontsize=8, loc="lower right")

    # Right: MAE comparison
    _axes[1].bar(["Random Split\n(WRONG)", "TimeSeriesSplit\n(RIGHT)"],
                 [_mae_random, _mae_ts],
                 color=["#E74C3C", "#2ECC71"], alpha=0.85, width=0.45)
    _axes[1].set_ylabel("MAE", fontsize=12)
    _axes[1].set_title(f"Temporal Leakage: random split MAE={_mae_random:.2f}\n"
                       f"True MAE (walk-forward)={_mae_ts:.2f}  ← the real number",
                       fontsize=11, fontweight="bold")
    for _i, _v in enumerate([_mae_random, _mae_ts]):
        _axes[1].text(_i, _v + 0.05, f"{_v:.2f}", ha="center", fontsize=12, fontweight="bold")
    _axes[1].grid(True, axis="y", alpha=0.3)

    _fig.suptitle("Leak 3: Temporal Leakage — random split makes time series look deceptively easy",
                  fontsize=12, fontweight="bold")
    _fig.tight_layout()
    return _fig, _mae_random, _mae_ts


@app.cell
def leak4_concept(mo):
    mo.md("""
    ### Leak 4: Group Leakage

    **Definition:** the same entity appears in both train and test (different rows, same underlying
    group).

    **Example:** predicting whether a patient will get a disease. Patient A has 5 visits in the
    dataset — 3 in train, 2 in test. The model memorizes Patient A's idiosyncratic patterns
    (their baseline vitals, their writing style in notes) and "predicts" them in test.

    The key question: **"What is the unit of generalization?"** If you want to predict for a
    *new unseen patient*, all rows from Patient A must be entirely in train OR test — not both.

    **Detection:** ask about every natural grouping: patient, user, customer, product, region,
    document, game session. If you'd predict for unseen entities of that type, you have groups.

    **Fix:** `sklearn.model_selection.GroupKFold` or `GroupShuffleSplit`, passing `groups=patient_id`.
    """)
    return


@app.cell
def leak4_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.model_selection import (cross_val_score as _cvs,
                                         StratifiedKFold as _SKF,
                                         GroupKFold as _GKF)
    from sklearn.preprocessing import StandardScaler as _SS

    _rng = _np.random.default_rng(1)
    _n_patients = 80
    _visits_per = 5
    _n = _n_patients * _visits_per

    # Patient-level signal + visit-level noise + patient-specific bias
    _patient_ids = _np.repeat(_np.arange(_n_patients), _visits_per)
    _patient_label = (_rng.random(_n_patients) > 0.5).astype(int)
    _y_grp = _patient_label[_patient_ids]

    _patient_bias = _rng.normal(0, 2, _n_patients)[_patient_ids]  # memorizable patient signature
    _X_grp = _np.column_stack([
        _y_grp * 0.5 + _rng.normal(0, 1.5, _n),  # weak signal
        _patient_bias + _rng.normal(0, 0.3, _n),   # strong patient-specific signature (memorizable)
        _rng.normal(0, 1, _n),
    ])
    _X_grp = _SS().fit_transform(_X_grp)

    # Random KFold — rows from same patient can land in train+test
    _skf = _SKF(n_splits=5, shuffle=True, random_state=42)
    _scores_random = _cvs(_LR(max_iter=1000), _X_grp, _y_grp, cv=_skf, scoring="roc_auc")

    # GroupKFold — entire patient must be in one split only
    _gkf = _GKF(n_splits=5)
    _scores_group = _cvs(_LR(max_iter=1000), _X_grp, _y_grp,
                         cv=_gkf, groups=_patient_ids, scoring="roc_auc")

    _fig, _ax = _plt.subplots(figsize=(9, 4.5))
    _x = _np.arange(5)
    _w = 0.35
    _ax.bar(_x - _w / 2, _scores_random, _w,
            label=f"StratifiedKFold — rows split randomly  (mean={_scores_random.mean():.3f})",
            color="#E74C3C", alpha=0.85)
    _ax.bar(_x + _w / 2, _scores_group, _w,
            label=f"GroupKFold — patients kept intact  (mean={_scores_group.mean():.3f})",
            color="#2ECC71", alpha=0.85)
    _ax.set_xticks(_x)
    _ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
    _ax.set_ylabel("ROC-AUC", fontsize=12)
    _ax.set_ylim(0.4, 1.05)
    _ax.set_title("Group Leakage: same patient in train+test inflates AUC\n"
                  "GroupKFold reveals true generalization to unseen patients",
                  fontsize=11, fontweight="bold")
    _ax.legend(fontsize=10)
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    return _fig


@app.cell
def leak5_concept(mo):
    mo.md("""
    ### Leak 5: Look-Ahead in Features

    **Definition:** a feature uses aggregated statistics that include the target row's own time period.

    **Example:** `user_avg_purchase_amount_last_30_days` — if the window is computed including
    the target transaction itself, the feature leaks the target's value into the feature vector.
    The model sees a version of the label dressed up as a feature.

    This is subtle and extremely common in production ML, especially at companies with large
    feature stores or complex ETL pipelines.

    **Real-world example:** predicting whether a user clicks an ad. A feature is
    `session_ctr_today` computed using all clicks in the session — but the target event IS one
    of those clicks. The model knows the answer before it makes the prediction.

    **Fix:** compute features **as-of** the target timestamp. Exclude the target row from any
    aggregation window. In feature stores, this is called a **point-in-time join** — each row
    retrieves feature values from a snapshot taken *before* its event timestamp.

    ```python
    # WRONG
    df["avg_spend_30d"] = df.groupby("user_id")["amount"].transform(
        lambda x: x.rolling(30).mean()  # includes current row
    )

    # RIGHT — use shift(1) or explicit point-in-time exclusion
    df["avg_spend_30d"] = df.groupby("user_id")["amount"].transform(
        lambda x: x.shift(1).rolling(30).mean()  # excludes current row
    )
    ```
    """)
    return


@app.cell
def leak6_concept(mo):
    mo.md("""
    ### Leak 6: Target Encoding Leakage

    **Definition:** target encoding categorical features (replace each category with its mean
    target value) using the **full dataset** — including the row being encoded.

    **Why it leaks:** each row's own target value influences its encoding. The model effectively
    sees a noisy version of the label as a feature. On small or high-cardinality categories this
    is especially damaging — the encoding becomes nearly the label itself.

    **Code pattern that causes this:**
    ```python
    # WRONG: computes mean using full dataset, including the row's own label
    df["cat_encoded"] = df.groupby("category")["target"].transform("mean")
    X_train = df[features]
    ```

    **Fixes:**
    1. **Within-fold target encoding:** compute target encoding using only training fold data.
       Never let the encoding see validation fold labels.
    2. **Leave-one-out target encoding:** when computing the encoding for row i, exclude row i
       from the mean calculation. Reduces but doesn't eliminate leakage in CV.
    3. **sklearn's `TargetEncoder`** (sklearn ≥ 1.3) handles this correctly when used inside a
       `Pipeline` with CV — it fits encoding only on the training portion.

    ```python
    # RIGHT: use Pipeline so TargetEncoder only sees training fold labels at each CV split
    from sklearn.preprocessing import TargetEncoder
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("enc", TargetEncoder(target_type="binary")),
        ("clf", LogisticRegression()),
    ])
    cross_val_score(pipe, X, y, cv=StratifiedKFold(5))
    ```
    """)
    return


@app.cell
def leak7_concept(mo):
    mo.md("""
    ### Leak 7: Eval Set Drift / Repeated Evaluation

    **Definition:** looking at test metrics repeatedly during development effectively turns your
    test set into a validation set. Each peek leaks information into your decisions.

    **How it happens:** you train model v1, check test AUC → 0.78. Train v2 with more features,
    check test AUC → 0.81. Try v3 with different hyperparams, test AUC → 0.79. Keep v2. You've
    now done model selection on the test set.

    **Why it's insidious:** you never explicitly used the test set in training. But every time
    you looked at its metric and made a decision based on it, you were effectively selecting
    for models that happen to fit the test set. The reported metric is biased upward.

    > *"If you've made 10 model versions and looked at test performance for each, you've selected
    > a model that fits the test set. The reported metric is not an unbiased estimate of
    > generalization."*

    **Fixes:**
    1. **Discipline:** test set = ONE evaluation at the very end. Use validation for all iteration.
    2. **Hold out a second test set** from the start if you need early sanity checks.
    3. **Track decisions:** log every time you look at the test set. If the count > 1, your metric
       is suspect.
    4. **Statistical correction:** Bonferroni or train-test protocol to adjust for multiple
       comparisons — but discipline is simpler and more reliable.
    """)
    return


@app.cell
def special_cases(mo):
    mo.md("""
    ## Special Cases — Validation Strategies for Tricky Data

    ### Time-Series Data

    Never random split. Always temporal split.

    | Strategy | Description | Use when |
    |----------|-------------|----------|
    | **TimeSeriesSplit** | Walk-forward: train on [t0, t1], test [t1, t2], expand each fold | Standard time-series CV |
    | **Expanding window** | Training window grows each fold — most realistic for production | Non-stationary data with long-range dependencies |
    | **Sliding window** | Fixed-size training window slides forward | Non-stationary where old data is irrelevant |
    | **Gap parameter** | Leave a gap between train end and test start | Simulate prediction lag (e.g., predict T+7 using data through T) |

    ```python
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5, gap=7)  # 7-step gap between train and test
    ```

    ---

    ### Imbalanced Classes

    - Always use `StratifiedKFold` — preserves class proportions in each fold
    - Without stratification: rare class might be entirely absent from a fold, making the
      fold's metric meaningless
    - For multi-label: `sklearn` doesn't support this natively — use `scikit-multilearn`'s
      iterative stratification

    ---

    ### Grouped Data

    Detect groups by asking: *"What entity am I ultimately predicting for?"*
    If the answer is a type of entity (patient, user, region) rather than a row, you have groups.

    | Situation | Splitter |
    |-----------|----------|
    | Groups only | `GroupKFold` or `GroupShuffleSplit` |
    | Groups + imbalanced | `StratifiedGroupKFold` |
    | One group per fold | `LeaveOneGroupOut` |

    ---

    ### Nested CV — When Hyperparameter Search Matters

    **The problem:** tuning hyperparameters on K-fold CV inflates the reported score.
    You're optimizing against validation folds — they become contaminated by your tuning choices.

    **Solution:** nested CV
    - **Outer loop** (5 folds): estimates true performance on held-out data
    - **Inner loop** (3 folds): tunes hyperparameters using only the outer training portion

    **Expensive but correct.** Use when reporting performance for a paper or a high-stakes decision.

    **Production shortcut:** skip nested CV. Use a single train/val/test split with grid search
    on the validation set. Less rigorous, but faster and usually sufficient.

    ```python
    from sklearn.model_selection import cross_val_score, GridSearchCV

    inner_cv = StratifiedKFold(n_splits=3)
    outer_cv = StratifiedKFold(n_splits=5)

    gs = GridSearchCV(estimator=model, param_grid=params, cv=inner_cv)
    nested_scores = cross_val_score(gs, X, y, cv=outer_cv)  # outer loop
    ```
    """)
    return


@app.cell
def decision_framework(mo):
    mo.md("""
    ## Decision Framework — Which Validation Strategy?

    ```
    Is this time-series data?
    ├── YES → TimeSeriesSplit (with gap= if prediction lag exists)
    └── NO → Continue

    Are there natural groups (patients, users, geographic regions, etc.)?
    ├── YES → GroupKFold or StratifiedGroupKFold
    └── NO → Continue

    Is the target imbalanced (minority class < 10-15%)?
    ├── YES → StratifiedKFold
    └── NO → KFold (regular)

    Are you tuning hyperparameters AND need an unbiased performance report?
    ├── YES → Nested CV  (or train/val/test with single test eval at the very end)
    └── NO → Standard K-fold is fine

    Is your dataset large (>100K rows)?
    ├── YES → Single train/val/test split (CV is expensive overkill)
    └── NO → CV recommended for stability
    ```

    **Checklist before training any model:**

    1. What is the unit of generalization? (row? patient? time period?)
    2. Is there temporal structure?
    3. Is there group structure?
    4. Is the target imbalanced?
    5. Have I split BEFORE fitting any transformers?
    6. Would any feature be unavailable at inference time?
    7. Have I touched the test set zero times so far?
    """)
    return


@app.cell
def flashcards(mo):
    mo.md("""
    ## Flashcard Summary

    | Question | Answer |
    |----------|--------|
    | Why three sets, not two? | If you tune on test, test = validation — you have no unbiased estimate of generalization. |
    | When does single split fail? | Small datasets: the random split could be lucky or unlucky. Use CV for stability. |
    | What's stratified K-fold? | Preserves class proportions in each fold. Essential for imbalanced data. |
    | Time series + random split = ? | Temporal leakage. Future data leaks into training. Use `TimeSeriesSplit`. |
    | Same patient in train and test = ? | Group leakage. The model memorizes patient idiosyncrasies. Use `GroupKFold`. |
    | Scale before split or after? | **After.** Fit scaler on train only, transform both train and test with those params. |
    | What's target leakage? | A feature contains the target or a near-perfect proxy. Symptoms: AUC ≈ 1.0, dominant feature importance. |
    | Why use sklearn Pipelines? | They prevent preprocessing leakage by fitting transformers only on the training portion at each CV fold. |
    | Look-ahead in features? | Feature uses information from the target's own time period. Fix: compute features as-of, use point-in-time joins. |
    | Why is repeated test set eval bad? | Each look leaks info into your decisions. You end up selecting a model that fits the test set. |
    | What's nested CV for? | Unbiased performance estimate when you're also tuning hyperparameters. Outer loop = eval, inner loop = tuning. |
    | Leave-one-out CV — when to use? | Almost never. Very expensive, high variance estimate. K=5 or 10 is almost always better. |
    """)
    return


@app.cell
def interview_talking_points(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    ### "Walk me through your validation strategy."

    > "First I ask about the data: is it time-series? Are there natural groups — patients, users,
    > regions? Is the target imbalanced? Based on that I pick the appropriate splitter:
    > `TimeSeriesSplit` for temporal data, `GroupKFold` for grouped, `StratifiedKFold` for imbalanced.
    > Then I use a three-way split: training for fitting, validation for model selection and
    > hyperparameter tuning, test set touched exactly once for the final unbiased metric. Every
    > transformer — scalers, encoders, imputers — is fit only on the training portion, never on
    > the full dataset."

    ---

    ### "Tell me about a time you found data leakage."

    Pick one of the 7 types. Tell a specific story — even a synthetic one. Vague = untrustworthy.

    > "I was building a churn model and noticed the AUC was 0.98. I looked at feature importances
    > and one feature — `days_since_account_closed` — had 85% importance. I realized that field
    > was only populated for churned customers, so it was a direct proxy for the label. After
    > removing it, AUC dropped to 0.72, which was the honest number. The model was actually useful
    > for the remaining signal — we just needed to be honest about what it could do."

    ---

    ### "How do you know your model will work in production?"

    > "Proper validation that matches production conditions. If we're predicting future events,
    > I use a temporal split — train on past, validate on a forward-looking window. If we're
    > predicting for unseen entities (new customers, new products), I use group-aware splits.
    > I also make sure no preprocessing transformer sees test data during training. After all
    > that, I'd want shadow mode deployment — run the model in parallel with the current system,
    > compare predictions to actuals, before rolling out fully."

    ---

    ### "Why might val accuracy be way higher than test accuracy?"

    Three causes:
    1. **Repeated tuning against val:** the validation set got contaminated — you optimized toward it.
    2. **Distribution shift:** val and test came from different time periods or populations.
    3. **Data leakage present in val but not in test:** a leaky feature exists in your val window
       but the test window doesn't have it (e.g., a temporal leak).

    ---

    ### Connection to backtesting / my work

    > "My backtesting engine enforces walk-forward validation for trading strategies. Random
    > shuffling of financial data produces spectacular-looking Sharpe ratios that collapse to zero
    > in live trading — the model has seen the future. Walk-forward = TimeSeriesSplit applied to
    > financial data: train on everything up to date T, test on T+1 through T+N, roll forward.
    > This is exactly why walk-forward backtests are considered honest and random-split backtests
    > are not. Same principle, different domain."
    """)
    return


if __name__ == "__main__":
    app.run()
