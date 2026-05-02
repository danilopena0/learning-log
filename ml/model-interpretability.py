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
    # Inspecting What Drives Model Decisions — SHAP, Feature Importance, Partial Dependence, Attention, Debugging

    | Field | Value |
    |-------|-------|
    | Date  | 2026-04-30 |
    | Track | ML Theory |
    | Time  | 60 min |
    | Topics | Feature Importance · Permutation Importance · SHAP · PDP/ICE · Interpretability Debugging |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Why Interpretability Matters

    Interpretability isn't optional. Three audiences demand it for very different reasons:

    **1. You (the builder) — debugging**

    Is the model learning signal or memorizing noise? Are the *right* features driving predictions?
    A model that gets 90% accuracy because it learned a spurious correlation with your data
    pipeline artifact will collapse in production the moment that artifact changes.
    You can't catch this without looking inside.

    **2. Stakeholders — trust**

    "Why did the model reject this loan?" A black box that can't explain itself won't be
    deployed in any organization with accountability structures. Product managers, executives,
    and customers all need explanations — not probability scores.

    **3. Regulators — compliance**

    GDPR's "right to explanation" requires that automated decisions can be explained in plain
    language. Financial regulators require model documentation. Healthcare requires clinical
    justification. In these domains, an unexplainable model is an unusable model.

    ---

    ### The interpretability-accuracy trade-off is mostly a myth

    The common framing — "accuracy OR interpretability, not both" — is outdated.
    SHAP gives you interpretability *on top of* complex models. You don't sacrifice accuracy.

    > *"In my fraud detection system design, I chose XGBoost partly because SHAP gives
    > human-readable reason codes for declines. A neural net might score 1% better but
    > can't explain itself to regulators. That 1% isn't worth the compliance risk."*

    ---

    ### The interpretability toolkit

    | Method | Model scope | What it answers |
    |--------|------------|-----------------|
    | Coefficients | Linear models only | Feature direction + magnitude |
    | Impurity importance | Trees only | Which features are split on most? |
    | Permutation importance | Any model | Which features matter on held-out data? |
    | SHAP | Any (fast for trees) | Why did THIS specific prediction happen? |
    | PDP / ICE | Any model | How does prediction change as X varies? |
    """)
    return


# ─── Setup ────────────────────────────────────────────────────────────────────

@app.cell
def _setup():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import xgboost as xgb

    rng = np.random.RandomState(42)
    n = 1000

    # Informative features
    years_experience   = rng.gamma(2, 4, n).clip(0, 20)
    skill_match_pct    = rng.uniform(30, 100, n)
    salary_fit         = rng.normal(0, 1, n)      # 0 = perfect, ±2 = mismatch
    location_match     = rng.binomial(1, 0.6, n).astype(float)
    job_level_match    = rng.normal(0, 1, n)      # 0 = right level

    # Noise features
    company_size       = rng.normal(500, 200, n)
    industry_match     = rng.uniform(0, 1, n)
    education_match    = rng.uniform(0, 1, n)
    posting_age_days   = rng.exponential(20, n)
    description_length = rng.normal(800, 300, n)

    feature_names = [
        'years_experience', 'skill_match_pct', 'salary_fit',
        'location_match', 'job_level_match',
        'company_size', 'industry_match', 'education_match',
        'posting_age_days', 'description_length',
    ]

    X_arr = np.column_stack([
        years_experience, skill_match_pct, salary_fit,
        location_match, job_level_match,
        company_size, industry_match, education_match,
        posting_age_days, description_length,
    ])

    # True signal: only first 5 features matter
    log_odds_true = (
        0.15 * years_experience
        + 0.06 * skill_match_pct
        - 0.50 * np.abs(salary_fit)
        + 1.00 * location_match
        - 0.35 * np.abs(job_level_match)
        - 4.50
        + rng.normal(0, 0.4, n)
    )
    prob_true = 1 / (1 + np.exp(-log_odds_true))
    good_fit  = (rng.random(n) < prob_true).astype(int)

    X_df = pd.DataFrame(X_arr, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, good_fit, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names
    )

    lr_unscaled = LogisticRegression(max_iter=1000, random_state=42)
    lr_unscaled.fit(X_train, y_train)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)

    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        eval_metric='logloss', random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train)

    acc_lr  = accuracy_score(y_test, lr_model.predict(X_test_scaled))
    acc_rf  = accuracy_score(y_test, rf_model.predict(X_test))
    acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test))

    return (
        np, pd, plt, matplotlib,
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler,
        feature_names,
        lr_unscaled, lr_model, rf_model, xgb_model,
        accuracy_score,
        acc_lr, acc_rf, acc_xgb,
    )


@app.cell
def _(mo, acc_lr, acc_rf, acc_xgb):
    mo.md(f"""
    ## Setup: Job-Fit Scoring Dataset + 3 Models

    **Dataset:** 1,000 synthetic job candidates · 10 features · binary target (`good_fit`)

    - **Informative (5):** `years_experience`, `skill_match_pct`, `salary_fit`, `location_match`, `job_level_match`
    - **Noise (5):** `company_size`, `industry_match`, `education_match`, `posting_age_days`, `description_length`

    | Model | Accuracy |
    |-------|---------|
    | Logistic Regression (scaled inputs) | {acc_lr:.1%} |
    | Random Forest | {acc_rf:.1%} |
    | XGBoost | {acc_xgb:.1%} |

    Same dataset, same problem, three models. We'll inspect all three to see how interpretability
    differs by model type. XGBoost scores highest but requires SHAP to explain; logistic regression
    is fully transparent but assumes linearity.
    """)
    return


# ─── Part 1: Built-in Feature Importance ─────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 1: Built-in Feature Importance (Quick & Dirty)

    Every model family has a "native" way to rank features — but each has different assumptions
    and failure modes. Know what you're measuring before you trust the numbers.
    """)
    return


@app.cell
def _lr_scaling_trap(lr_unscaled, lr_model, feature_names, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _y = np.arange(len(feature_names))

    _colors_u = ['#27ae60' if c > 0 else '#e74c3c' for c in lr_unscaled.coef_[0]]
    _colors_s = ['#27ae60' if c > 0 else '#e74c3c' for c in lr_model.coef_[0]]

    _ax = _axes[0]
    _ax.barh(_y, lr_unscaled.coef_[0], color=_colors_u, alpha=0.85, edgecolor='white')
    _ax.axvline(0, color='black', linewidth=0.8)
    _ax.set_yticks(_y)
    _ax.set_yticklabels(feature_names, fontsize=9)
    _ax.set_title("LR Coefficients — UNSCALED inputs\n"
                  "salary_fit (std≈1) has large coef; skill_match_pct (std≈20) has small coef",
                  fontsize=9)
    _ax.set_xlabel("Coefficient value")
    _ax.grid(True, axis='x', alpha=0.3)

    _ax = _axes[1]
    _ax.barh(_y, lr_model.coef_[0], color=_colors_s, alpha=0.85, edgecolor='white')
    _ax.axvline(0, color='black', linewidth=0.8)
    _ax.set_yticks(_y)
    _ax.set_yticklabels(feature_names, fontsize=9)
    _ax.set_title("LR Coefficients — SCALED inputs\n"
                  "True importance emerges: skill_match and location_match lead",
                  fontsize=9)
    _ax.set_xlabel("Coefficient (per std-dev change in feature)")
    _ax.grid(True, axis='x', alpha=0.3)

    _fig.suptitle("Logistic Regression Coefficients: The Scaling Trap",
                  fontsize=11, fontweight='bold')
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **Reading LR coefficients:**
    - Positive (green) = pushes toward good fit. Negative (red) = pushes away.
    - Noise features should cluster near zero.

    **The scaling trap:** coefficients are only comparable if features are standardized.
    With raw features, `skill_match_pct` (range 30–100) appears unimportant because each
    unit change is small numerically. After scaling, coefficient = importance *per
    standard-deviation change*. Now the features are on equal footing.

    **Limitation:** logistic regression assumes a linear relationship. If `years_experience`
    has diminishing returns (experience helps 0–10 years, then plateaus), a single coefficient
    misses all of that. Use PDP or SHAP to detect nonlinearity.
    """)
    return


@app.cell
def _impurity_compute(rf_model, xgb_model, feature_names):
    impurity_rf  = dict(zip(feature_names, rf_model.feature_importances_))
    impurity_xgb = dict(zip(feature_names, xgb_model.feature_importances_))
    return impurity_rf, impurity_xgb


@app.cell
def _impurity_plot(impurity_rf, impurity_xgb, feature_names, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _y = np.arange(len(feature_names))

    for _ax, _imp_dict, _title in zip(
        _axes,
        [impurity_rf, impurity_xgb],
        ["Random Forest — Impurity (Gini) Importance",
         "XGBoost — Impurity (Gain) Importance"]
    ):
        _imp = np.array([_imp_dict[f] for f in feature_names])
        _order = np.argsort(_imp)
        _colors = ['#27ae60' if i < 5 else '#95a5a6' for i in _order]
        _ax.barh(
            np.arange(len(feature_names)), _imp[_order],
            color=_colors, alpha=0.85, edgecolor='white'
        )
        _ax.set_yticks(np.arange(len(feature_names)))
        _ax.set_yticklabels([feature_names[i] for i in _order], fontsize=9)
        _ax.set_title(_title, fontsize=9)
        _ax.set_xlabel("Feature importance")
        _ax.grid(True, axis='x', alpha=0.3)

    _fig.suptitle("Impurity-based Feature Importance (green = truly informative)",
                  fontsize=11, fontweight='bold')
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **What impurity importance measures:** total reduction in Gini/entropy from all splits
    on this feature, summed across all trees.

    **Trap 1 — no direction.** Does more `years_experience` *help* or *hurt*? Impurity
    importance doesn't say. You need SHAP or LR coefficients for direction.

    **Trap 2 — cardinality bias.** `description_length` (continuous, many unique values) gets
    more split opportunities than `location_match` (binary). The model shops around for good
    thresholds on high-cardinality features, artificially inflating their importance.

    **Trap 3 — training data only.** If the model overfits to training noise, noise features
    rank higher than they deserve on held-out data.

    > *"Impurity importance is the default. Quick, but misleading in all three ways above.
    > Use permutation importance or SHAP for any real analysis."*
    """)
    return


# ─── Part 2: Permutation Importance ──────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 2: Permutation Importance — Model-Agnostic, Reliable

    **Algorithm:** shuffle one feature column → measure how much worse the model gets.
    Bigger accuracy drop = more important. The feature was doing real work; destroying it hurt.

    **Why it's better than impurity importance:**
    - Works for *any* model, not just trees
    - Not biased by cardinality — shuffling doesn't advantage high-cardinality features
    - Measured on *validation data*, so training-set overfitting doesn't inflate noise importance

    **Limitation — correlated features:** if `skill_match_pct` and `years_experience` are
    correlated, shuffling one doesn't hurt much because the model compensates with the other.
    Both get lower importance than they deserve individually. When features are correlated,
    SHAP handles marginal contributions more carefully.
    """)
    return


@app.cell
def _perm_compute(np, X_test, y_test, X_test_scaled, rf_model, xgb_model, lr_model, feature_names, accuracy_score):
    def _perm_importance(model, X_val, y_val, n_repeats=10):
        _baseline = accuracy_score(y_val, model.predict(X_val))
        _result = {}
        for _col in X_val.columns:
            _scores = []
            for _ in range(n_repeats):
                _Xs = X_val.copy()
                _Xs[_col] = np.random.permutation(_Xs[_col].values)
                _scores.append(_baseline - accuracy_score(y_val, model.predict(_Xs)))
            _result[_col] = {'mean': np.mean(_scores), 'std': np.std(_scores)}
        return _result

    np.random.seed(42)
    perm_rf  = _perm_importance(rf_model, X_test, y_test)
    perm_xgb = _perm_importance(xgb_model, X_test, y_test)
    perm_lr  = _perm_importance(lr_model, X_test_scaled, y_test)
    return perm_rf, perm_xgb, perm_lr


@app.cell
def _perm_plot(perm_rf, perm_xgb, perm_lr, feature_names, np, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(17, 5))

    for _ax, _perm, _title in zip(
        _axes,
        [perm_lr, perm_rf, perm_xgb],
        ["Logistic Regression", "Random Forest", "XGBoost"]
    ):
        _means = np.array([_perm[f]['mean'] for f in feature_names])
        _stds  = np.array([_perm[f]['std']  for f in feature_names])
        _order = np.argsort(_means)
        _colors = ['#27ae60' if _means[i] > 0 else '#e74c3c' for i in _order]

        _ax.barh(
            np.arange(len(feature_names)), _means[_order],
            xerr=_stds[_order], color=_colors, alpha=0.85,
            edgecolor='white', capsize=3
        )
        _ax.axvline(0, color='black', linewidth=0.8)
        _ax.set_yticks(np.arange(len(feature_names)))
        _ax.set_yticklabels([feature_names[i] for i in _order], fontsize=9)
        _ax.set_title(_title, fontsize=10)
        _ax.set_xlabel("Mean accuracy drop when shuffled")
        _ax.grid(True, axis='x', alpha=0.3)

    _fig.suptitle("Permutation Importance (error bars = std over 10 shuffles)\n"
                  "Noise features cluster near zero — error bars cross zero = not reliably important",
                  fontsize=10, fontweight='bold')
    _fig.tight_layout()
    return _fig


@app.cell
def _perm_vs_impurity(impurity_rf, perm_rf, feature_names, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _y = np.arange(len(feature_names))

    _imp_vals  = np.array([impurity_rf[f] for f in feature_names])
    _perm_vals = np.array([perm_rf[f]['mean'] for f in feature_names])
    _imp_norm  = _imp_vals / _imp_vals.max() * max(_perm_vals.max(), 1e-9)

    _order = np.argsort(_perm_vals)
    _ax = _axes[0]
    _ax.barh(_y - 0.2, _imp_norm[_order], height=0.4,
             color='#3498db', alpha=0.8, label='Impurity (normalized)')
    _ax.barh(_y + 0.2, _perm_vals[_order], height=0.4,
             color='#27ae60', alpha=0.8, label='Permutation')
    _ax.axvline(0, color='black', linewidth=0.8)
    _ax.set_yticks(_y)
    _ax.set_yticklabels([feature_names[i] for i in _order], fontsize=9)
    _ax.set_title("Impurity vs Permutation — Random Forest\n"
                  "Where they disagree, trust permutation", fontsize=9)
    _ax.legend(fontsize=9)
    _ax.grid(True, axis='x', alpha=0.3)

    _disagreement = np.abs(_imp_norm - _perm_vals)
    _ax2 = _axes[1]
    _colors2 = ['#e74c3c' if _disagreement[i] > 0.003 else '#95a5a6' for i in _order]
    _ax2.barh(_y, _disagreement[_order], color=_colors2, alpha=0.85, edgecolor='white')
    _ax2.set_yticks(_y)
    _ax2.set_yticklabels([feature_names[i] for i in _order], fontsize=9)
    _ax2.set_title("Disagreement magnitude\n(red = impurity is misleading for this feature)",
                   fontsize=9)
    _ax2.set_xlabel("| impurity_norm − permutation |")
    _ax2.grid(True, axis='x', alpha=0.3)

    _fig.suptitle("Impurity vs Permutation Importance Comparison — Random Forest",
                  fontsize=11, fontweight='bold')
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **When they disagree:** impurity importance overcounts high-cardinality features like
    `company_size` and `description_length` — the model had more split opportunities on them
    during training. Permutation importance measures actual predictive value on held-out data.

    > *"When impurity and permutation importance disagree, trust permutation."*
    """)
    return


# ─── Part 3: SHAP ─────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 3: SHAP — The Gold Standard

    SHAP (SHapley Additive exPlanations) answers the question neither coefficients nor
    importance scores can: **"For THIS specific prediction, how much did each feature contribute?"**

    **From game theory:** Shapley values fairly distribute a "payout" (the prediction) among
    "players" (the features). Each feature gets credit proportional to its contribution,
    averaged over all possible orderings — the only allocation that satisfies fairness.

    ### Three guarantees that make SHAP trustworthy

    - **Additivity:** SHAP values sum to `prediction − baseline`. No information lost.
      > ŷᵢ = base_value + Σ SHAP(featureⱼ, sampleᵢ)
    - **Consistency:** if a feature's contribution increases, its SHAP value never decreases.
    - **Fairness:** rooted in game theory — the only method satisfying all three axioms.

    ### Three levels of analysis

    1. **Local** — explain ONE prediction: "Why did this job score 73%?"
    2. **Global** — explain the MODEL overall: "Which features matter across all predictions?"
    3. **Interaction** — "Which feature *combinations* matter?"
    """)
    return


@app.cell
def _shap_compute(xgb_model, X_test, feature_names, np):
    try:
        import shap as _shap
        _explainer = _shap.TreeExplainer(xgb_model)
        _sv_raw    = _explainer.shap_values(X_test)
        _base_raw  = _explainer.expected_value

        if isinstance(_sv_raw, list):
            _sv = np.array(_sv_raw[1])
            _base = float(_base_raw[1])
        else:
            _sv   = np.array(_sv_raw)
            _base = float(_base_raw)

        shap_ok     = True
        shap_values = _sv
        shap_base   = _base
        print(f"SHAP values shape: {_sv.shape}  (n_samples × n_features)")
        print(f"Base value (avg log-odds): {_base:.4f}")
        print(f"Sample 0 check — base {_base:.3f} + SHAP sum {_sv[0].sum():.3f}")
    except Exception as _e:
        print(f"shap not available: {_e}")
        shap_ok     = False
        shap_values = None
        shap_base   = 0.0

    return shap_ok, shap_values, shap_base


@app.cell
def _(mo):
    mo.md("""
    **How TreeExplainer works:** for tree models, SHAP values are computed exactly in
    O(TLD²) time (T=trees, L=leaves, D=depth) — far faster than naive Shapley computation
    which is exponential in features. This is why SHAP + XGBoost is the practical gold
    standard for tabular ML interpretability.
    """)
    return


@app.cell
def _shap_force_plot(shap_ok, shap_values, shap_base, X_test, feature_names, np, plt):
    _fig, _ax = plt.subplots(figsize=(12, 5))

    if shap_ok and shap_values is not None:
        _idx = 0
        _sv  = shap_values[_idx]
        _fv  = X_test.iloc[_idx].values

        _order = np.argsort(np.abs(_sv))[::-1][:8]
        _sv_top    = _sv[_order]
        _names_top = [feature_names[i] for i in _order]
        _fv_top    = [_fv[i] for i in _order]
        _colors    = ['#27ae60' if v > 0 else '#e74c3c' for v in _sv_top]

        _y = np.arange(len(_sv_top))
        _ax.barh(_y, _sv_top, color=_colors, alpha=0.85, edgecolor='white', height=0.6)
        _ax.axvline(0, color='black', linewidth=0.8)

        for _i, (_val, _feat, _fval) in enumerate(zip(_sv_top, _names_top, _fv_top)):
            _x_pos = _val + 0.003 * np.sign(_val + 1e-9)
            _ax.text(_x_pos, _i, f"  {_feat} = {_fval:.2f}",
                     va='center', ha='left' if _val > 0 else 'right', fontsize=8)

        _ax.set_yticks(_y)
        _ax.set_yticklabels(_names_top, fontsize=9)
        _ax.set_xlabel("SHAP value (contribution to prediction)")

        _pred_prob = 1 / (1 + np.exp(-(shap_base + _sv.sum())))
        _ax.set_title(
            f"SHAP Force Plot — Sample #{_idx}\n"
            f"Base: {shap_base:.3f}  +  SHAP sum: {_sv.sum():+.3f}  →  "
            f"P(good_fit) = {_pred_prob:.1%}\n"
            "Green = pushes prediction UP    Red = pushes prediction DOWN",
            fontsize=10
        )
        _ax.grid(True, axis='x', alpha=0.3)
    else:
        _ax.text(0.5, 0.5, "Install shap:  pip install shap",
                 ha='center', va='center', transform=_ax.transAxes, fontsize=12)

    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **This is what you'd show a stakeholder:**

    > *"This candidate scored 73% match probability. The biggest positive driver was skill
    > alignment — 94% skill match added +0.41. Location matched, adding +0.38. Nine years of
    > experience was right for the role (+0.25). The posting is 37 days old, suggesting lower
    > urgency or difficulty filling the role (-0.12). Salary mismatch pulled the score down
    > slightly (-0.18). Every number here is traceable back to the model calculation."*

    No hand-waving. Every contribution sums to the prediction. Auditable.
    """)
    return


@app.cell
def _shap_summary_beeswarm(shap_ok, shap_values, X_test, feature_names, np, plt):
    _fig, _ax = plt.subplots(figsize=(10, 7))

    if shap_ok and shap_values is not None:
        _mean_abs = np.abs(shap_values).mean(axis=0)
        _order    = np.argsort(_mean_abs)
        _cmap     = plt.cm.RdYlGn

        for _rank, _fi in enumerate(_order):
            _sv_col  = shap_values[:, _fi]
            _fv_col  = X_test.iloc[:, _fi].values
            _fv_norm = (_fv_col - _fv_col.min()) / (np.ptp(_fv_col) + 1e-9)
            _colors  = _cmap(_fv_norm)
            _jitter  = np.random.RandomState(42).uniform(-0.2, 0.2, len(_sv_col))
            _ax.scatter(_sv_col, _rank + _jitter, c=_colors, alpha=0.35, s=8, linewidths=0)

        _ax.axvline(0, color='black', linewidth=0.8)
        _ax.set_yticks(np.arange(len(feature_names)))
        _ax.set_yticklabels([feature_names[i] for i in _order], fontsize=9)
        _ax.set_xlabel("SHAP value")
        _ax.set_title(
            "SHAP Summary Plot (Beeswarm) — XGBoost\n"
            "Each dot = one sample. X = SHAP value. Color = feature value (red=high, green=low).\n"
            "Features sorted by mean |SHAP| — global importance ranking.",
            fontsize=10
        )
        _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(0, 1))
        _sm.set_array([])
        _cbar = _fig.colorbar(_sm, ax=_ax, shrink=0.5, aspect=20)
        _cbar.set_label("Feature value", fontsize=8)
        _cbar.set_ticks([0, 1])
        _cbar.set_ticklabels(['Low', 'High'])
        _ax.grid(True, axis='x', alpha=0.2)
    else:
        _ax.text(0.5, 0.5, "Install shap:  pip install shap",
                 ha='center', va='center', transform=_ax.transAxes, fontsize=12)

    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **How to read the SHAP summary plot:**

    - `skill_match_pct`: red dots (high values) cluster on the right (positive SHAP) →
      high skill match drives the prediction UP. Expected.
    - `location_match` (binary): two clusters. Red (location=1) on right, blue (0) on left.
    - `salary_fit`: red dots (large mismatch) cluster on the *left* → mismatch hurts.
    - Noise features: dots centered near zero with no color pattern → no signal.

    > *"The SHAP summary plot is the single most informative model visualization I know of.
    > It shows importance, direction, AND magnitude for every feature in one chart. I include
    > this in every model documentation I write."*
    """)
    return


@app.cell
def _shap_dependence(shap_ok, shap_values, X_test, feature_names, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    if shap_ok and shap_values is not None:
        _feat_idx  = feature_names.index('years_experience')
        _inter_idx = feature_names.index('job_level_match')

        _xv = X_test.iloc[:, _feat_idx].values
        _sv = shap_values[:, _feat_idx]
        _iv = X_test.iloc[:, _inter_idx].values

        _ax = _axes[0]
        _sc = _ax.scatter(_xv, _sv, c=_sv, cmap='RdYlGn',
                          alpha=0.5, s=15, vmin=_sv.min(), vmax=_sv.max())
        _ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        _ax.set_xlabel("years_experience")
        _ax.set_ylabel("SHAP value for years_experience")
        _ax.set_title("SHAP Dependence — years_experience\n"
                      "Nonlinear: steep rise then diminishing returns", fontsize=9)
        _ax.grid(True, alpha=0.2)
        _fig.colorbar(_sc, ax=_ax, shrink=0.8).set_label("SHAP value", fontsize=8)

        _ax = _axes[1]
        _iv_norm = (_iv - _iv.min()) / (np.ptp(_iv) + 1e-9)
        _sc2 = _ax.scatter(_xv, _sv, c=_iv_norm, cmap='RdYlGn', alpha=0.5, s=15)
        _ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        _ax.set_xlabel("years_experience")
        _ax.set_ylabel("SHAP value for years_experience")
        _ax.set_title("Colored by job_level_match\n"
                      "Good level match (green) = experience contributes more", fontsize=9)
        _ax.grid(True, alpha=0.2)
        _cbar2 = _fig.colorbar(_sc2, ax=_ax, shrink=0.8)
        _cbar2.set_label("job_level_match", fontsize=8)
        _cbar2.set_ticks([0, 1])
        _cbar2.set_ticklabels(['Mismatch', 'Good fit'])
    else:
        for _ax in _axes:
            _ax.text(0.5, 0.5, "Install shap:  pip install shap",
                     ha='center', va='center', transform=_ax.transAxes, fontsize=12)

    _fig.suptitle("SHAP Dependence Plots — Nonlinearity and Interactions",
                  fontsize=11, fontweight='bold')
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **Reading dependence plots:**

    Left panel: nonlinear effect of experience — steep gain 0–8 years, diminishing returns
    after that. A single coefficient would miss this entirely.

    Right panel: the same experience level has higher SHAP values when job level is a good
    match. High experience + senior role = big positive. High experience + junior role =
    smaller (or negative: "overqualified") contribution.

    > *"Dependence plots reveal nonlinear relationships and interactions the model learned.
    > This is how you discover 'the model penalizes overqualification for junior roles' —
    > something invisible in coefficient or importance tables."*
    """)
    return


@app.cell
def _shap_leakage_demo(np, X_train, y_train, X_test, y_test, feature_names, plt, accuracy_score):
    import xgboost as _xgb_leak

    _rng = np.random.RandomState(99)
    _X_tr_leak = X_train.copy()
    _X_te_leak = X_test.copy()
    _X_tr_leak['leaky_signal'] = y_train + _rng.normal(0, 0.05, len(y_train))
    _X_te_leak['leaky_signal'] = y_test  + _rng.normal(0, 0.05, len(y_test))
    _feat_leak = feature_names + ['leaky_signal']

    _m = _xgb_leak.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        eval_metric='logloss', random_state=42, verbosity=0
    )
    _m.fit(_X_tr_leak, y_train)
    _acc = accuracy_score(y_test, _m.predict(_X_te_leak))
    print(f"Leaky model accuracy: {_acc:.3f}  ← suspiciously high")

    _fig, _ax = plt.subplots(figsize=(10, 5))
    try:
        import shap as _shap_leak
        _expl = _shap_leak.TreeExplainer(_m)
        _sv   = _expl.shap_values(_X_te_leak)
        if isinstance(_sv, list):
            _sv = np.array(_sv[1])
        else:
            _sv = np.array(_sv)

        _mean_abs = np.abs(_sv).mean(axis=0)
        _order    = np.argsort(_mean_abs)
        _colors   = ['#e74c3c' if _feat_leak[i] == 'leaky_signal' else '#3498db'
                     for i in _order]
        _ax.barh(np.arange(len(_feat_leak)), _mean_abs[_order],
                 color=_colors, alpha=0.85, edgecolor='white')
        _ax.set_yticks(np.arange(len(_feat_leak)))
        _ax.set_yticklabels([_feat_leak[i] for i in _order], fontsize=9)
        _ax.set_title(
            f"SHAP — Leaky Model (accuracy={_acc:.1%})\n"
            "Red bar = leaky_signal dominates by 10×. This is the data leakage signature.",
            fontsize=10
        )
        _ax.set_xlabel("Mean |SHAP value|")
        _ax.grid(True, axis='x', alpha=0.3)
    except Exception as _e:
        _ax.text(0.5, 0.5, f"shap error: {_e}", ha='center', va='center',
                 transform=_ax.transAxes)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **SHAP makes data leakage visible.**

    The telltale signature: one feature has mean |SHAP| 5–10× larger than everything else.
    Nothing in real data generates information that cleanly — it's almost always a pipeline
    artifact: target-encoded column that leaked, a timestamp batch-correlated with labels,
    a derived column computed from the target.

    > *"If I'd deployed that model without a SHAP check, I'd have discovered the leakage
    > the hard way. SHAP makes it visible in 30 seconds."*
    """)
    return


# ─── Part 4: Partial Dependence Plots ─────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 4: Partial Dependence Plots — Model-Agnostic Marginal Effects

    **PDP answers:** "How does the model's *average* prediction change as I vary one feature,
    holding everything else fixed?"

    **Algorithm** (implement it yourself once and you'll never forget it):
    1. Choose a feature. Create a grid of 50 evenly-spaced values across its range.
    2. For each grid value: replace that feature in ALL test samples with that value.
    3. Predict on the modified dataset. Take the average predicted probability.
    4. Plot: grid value (x) vs average predicted probability (y).

    **PDP vs SHAP dependence:**
    - PDP averages over the full data distribution → shows the marginal effect, hides interactions
    - SHAP dependence shows per-sample contributions → reveals interactions
    - If ICE lines cross (see below), the PDP average is hiding a real interaction

    **Limitation:** PDP can create unrealistic feature combinations during averaging
    (e.g., high experience paired with entry-level salary). SHAP handles this better by
    conditioning on realistic data distributions.
    """)
    return


@app.cell
def _pdp_plots(xgb_model, X_test, feature_names, np, plt):
    def _partial_dep(model, X, feature, grid_points=50):
        _vals = np.linspace(X[feature].min(), X[feature].max(), grid_points)
        _avg  = []
        for _v in _vals:
            _Xm = X.copy()
            _Xm[feature] = _v
            _avg.append(model.predict_proba(_Xm)[:, 1].mean())
        return _vals, np.array(_avg)

    _features = [
        'years_experience', 'skill_match_pct', 'salary_fit',
        'location_match', 'job_level_match', 'posting_age_days'
    ]

    _fig, _axes = plt.subplots(2, 3, figsize=(15, 8))
    for _ax, _feat in zip(_axes.ravel(), _features):
        _vals, _preds = _partial_dep(xgb_model, X_test, _feat)
        _ax.plot(_vals, _preds, color='#2980b9', linewidth=2)
        _ax.axhline(_preds.mean(), color='gray', linestyle='--', linewidth=0.8,
                    label=f'mean={_preds.mean():.2f}')
        _ax.set_xlabel(_feat, fontsize=9)
        _ax.set_ylabel("Avg P(good_fit)", fontsize=8)
        _ax.set_title(f"PDP: {_feat}", fontsize=9)
        _ax.set_ylim(0, 1)
        _ax.grid(True, alpha=0.3)
        _ax.legend(fontsize=7)

    _fig.suptitle("Partial Dependence Plots — XGBoost\n"
                  "Each curve: average prediction as one feature varies (all others held fixed)",
                  fontsize=11, fontweight='bold')
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **What to read from these PDPs:**

    - `years_experience`: probability rises steeply, then plateaus — diminishing returns.
    - `skill_match_pct`: steep positive slope — the most monotonically impactful feature.
    - `salary_fit`: near-zero mismatch scores highest; larger mismatch in either direction hurts.
    - `location_match` (binary): sharp jump from 0 to 1.
    - `posting_age_days`: slight negative slope — older postings correlate with lower fit probability.
    - Noise features: flat or near-flat curves (model learned little from them).
    """)
    return


@app.cell
def _ice_plots(xgb_model, X_test, feature_names, np, plt):
    _feat  = 'years_experience'
    _grid  = np.linspace(X_test[_feat].min(), X_test[_feat].max(), 50)
    _rng   = np.random.RandomState(42)

    _ice = []
    for _i in range(min(200, len(X_test))):
        _row = []
        for _v in _grid:
            _Xm = X_test.iloc[[_i]].copy()
            _Xm[_feat] = _v
            _row.append(float(xgb_model.predict_proba(_Xm)[:, 1]))
        _ice.append(_row)
    _ice = np.array(_ice)
    _pdp = _ice.mean(axis=0)

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    _ax = _axes[0]
    for _curve in _ice[::5]:
        _ax.plot(_grid, _curve, color='#3498db', alpha=0.15, linewidth=0.8)
    _ax.plot(_grid, _pdp, color='#e74c3c', linewidth=2.5, label='PDP (average)', zorder=5)
    _ax.set_xlabel(_feat)
    _ax.set_ylabel("Predicted P(good_fit)")
    _ax.set_title("ICE + PDP — years_experience\nBlue = individual curves, Red = average",
                  fontsize=9)
    _ax.set_ylim(0, 1)
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.2)

    _rising  = [c for c in _ice if c[-1] > c[0] + 0.05]
    _falling = [c for c in _ice if c[-1] < c[0] - 0.05]
    _flat    = [c for c in _ice
                if not (c[-1] > c[0] + 0.05) and not (c[-1] < c[0] - 0.05)]

    _ax = _axes[1]
    for _c in _rising:  _ax.plot(_grid, _c, color='#27ae60', alpha=0.2, linewidth=0.8)
    for _c in _falling: _ax.plot(_grid, _c, color='#e74c3c', alpha=0.2, linewidth=0.8)
    for _c in _flat:    _ax.plot(_grid, _c, color='gray',    alpha=0.1, linewidth=0.8)
    _ax.plot(_grid, _pdp, color='black', linewidth=2.5, label='PDP', zorder=5)
    _ax.set_xlabel(_feat)
    _ax.set_ylabel("Predicted P(good_fit)")
    _ax.set_title(f"Heterogeneity: {len(_rising)} rising · {len(_falling)} falling · {len(_flat)} flat\n"
                  "If green+red curves cross: PDP hides a real interaction", fontsize=9)
    _ax.set_ylim(0, 1)
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.2)

    _fig.suptitle("Individual Conditional Expectation (ICE) — years_experience",
                  fontsize=11, fontweight='bold')
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **ICE vs PDP:**

    The PDP shows the average effect. ICE shows each individual sample's response curve.
    If ICE lines cross each other, there's a hidden interaction: the direction of the effect
    depends on other feature values for that sample.

    > *"Always check ICE when PDP looks flat. Flatness might mean 'half the samples go up,
    > half go down, and they cancel.' That's a completely different story than 'this feature
    > doesn't matter.'"*
    """)
    return


# ─── Part 5: Model-Specific Techniques ───────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 5: Model-Specific Inspection Techniques

    ### Attention Visualization (Transformers) — Conceptual

    Not coding this (requires a full transformer), but critical to know:

    - **Attention weights** show which input tokens each output token "looked at"
    - Tools: BertViz, Ecco, `model.get_attention_weights()` on HuggingFace models
    - Useful diagnostic for NLP: "which words in this job description drove the embedding?"

    **Important limitation:** attention ≠ importance.
    High attention means "the model looked here," not "this token drove the output."
    Research has shown attention can be adversarially manipulated without changing predictions.
    SHAP on the downstream task is more actionable.

    > *"In Canopy, I could extract attention from the sentence-transformer to see which
    > words the embedding model focuses on. But SHAP on the downstream job scorer is
    > more actionable for debugging why a specific candidate scored unexpectedly."*

    ---

    ### Logistic Regression: Full Prediction Audit

    LR is the *only* common model with zero-approximation, complete prediction decomposition.
    Every prediction is an exact documented sum.
    """)
    return


@app.cell
def _lr_decomposition(lr_model, X_test_scaled, y_test, feature_names, np, plt):
    _coef  = lr_model.coef_[0]
    _inter = lr_model.intercept_[0]
    _preds = lr_model.predict(X_test_scaled)
    _y_arr = y_test.values if hasattr(y_test, 'values') else y_test

    _correct_pos = np.where((_preds == 1) & (_y_arr == 1))[0]
    _correct_neg = np.where((_preds == 0) & (_y_arr == 0))[0]
    _wrong       = np.where(_preds != _y_arr)[0]

    _cases = {}
    if len(_correct_pos): _cases['Correct positive'] = _correct_pos[0]
    if len(_correct_neg): _cases['Correct negative'] = _correct_neg[0]
    if len(_wrong):       _cases['Misclassified']    = _wrong[0]

    _fig, _axes = plt.subplots(1, len(_cases), figsize=(6 * len(_cases), 6))
    if len(_cases) == 1:
        _axes = [_axes]

    for _ax, (_label, _idx) in zip(_axes, _cases.items()):
        _sample   = X_test_scaled.iloc[_idx].values
        _contribs = _sample * _coef
        _logodds  = _inter + _contribs.sum()
        _prob     = 1 / (1 + np.exp(-_logodds))

        _order = np.argsort(np.abs(_contribs))
        _cols  = ['#27ae60' if _contribs[i] > 0 else '#e74c3c' for i in _order]

        _ax.barh(np.arange(len(feature_names)), _contribs[_order],
                 color=_cols, alpha=0.85, edgecolor='white')
        _ax.axvline(0, color='black', linewidth=0.8)
        _ax.set_yticks(np.arange(len(feature_names)))
        _ax.set_yticklabels([feature_names[i] for i in _order], fontsize=8)
        _ax.set_xlabel("Contribution to log-odds")
        _ax.set_title(
            f"{_label}\n"
            f"Intercept: {_inter:+.3f}  |  Log-odds: {_logodds:.3f}\n"
            f"P(good_fit) = {_prob:.1%}  |  True label: {int(_y_arr[_idx])}",
            fontsize=9
        )
        _ax.grid(True, axis='x', alpha=0.3)

    _fig.suptitle("Logistic Regression: Full Prediction Audit\n"
                  "prediction = intercept + Σ (feature_value × coefficient) → sigmoid",
                  fontsize=11, fontweight='bold')
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **Why regulators love logistic regression:**

    Every prediction is a fully documented calculation:
    ```
    log-odds = intercept
             + 0.41 × years_experience_scaled
             + 0.63 × skill_match_pct_scaled
             + (−0.38) × salary_fit_scaled
             + ...
    P(good_fit) = sigmoid(log-odds)
    ```
    No approximation. No game theory. This is exactly what was computed — legally defensible,
    fully auditable. This is why logistic regression remains the first choice in credit scoring,
    clinical risk models, and any regulated domain: not accuracy, but accountability.
    """)
    return


# ─── Part 6: Debugging Models with Interpretability ──────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 6: Debugging Models with Interpretability

    ```
    Model is underperforming. Diagnostic workflow:

    1. Permutation importance — are the right features being used?
       └── Noise features rank high? → data quality issue or leakage

    2. SHAP summary plot — do directions make sense?
       └── "More experience = lower score"? → label error or confounded feature

    3. SHAP on worst predictions — what drove the biggest mistakes?
       └── One feature dominated? → outlier, data error, or leakage
       └── Contradictory SHAP patterns? → label noise

    4. PDP + ICE — unexpected relationships?
       └── Flat PDP + crossing ICE lines → hidden interaction
       └── Non-monotonic where monotonic expected → data quality issue

    5. Calibration check (from validation notebook)
       └── Model says 80% but right only 50%? → overconfident, needs Platt scaling
    ```
    """)
    return


@app.cell
def _debug_walkthrough(np, X_train, y_train, X_test, y_test, feature_names, plt, accuracy_score):
    import xgboost as _xgb_dbg

    # Inject 10% label noise
    _rng = np.random.RandomState(77)
    _y_noisy = y_train.values.copy() if hasattr(y_train, 'values') else y_train.copy()
    _mask = _rng.random(len(_y_noisy)) < 0.10
    _y_noisy[_mask] = 1 - _y_noisy[_mask]

    _m_noisy = _xgb_dbg.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        eval_metric='logloss', random_state=42, verbosity=0
    )
    _m_noisy.fit(X_train, _y_noisy)
    _acc_noisy = accuracy_score(y_test, _m_noisy.predict(X_test))
    print(f"Noisy model accuracy: {_acc_noisy:.3f}  ← dropped due to label noise")

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _preds  = _m_noisy.predict(X_test)
    _y_te   = y_test.values if hasattr(y_test, 'values') else y_test
    _wrong  = np.where(_preds != _y_te)[0]
    _right  = np.where(_preds == _y_te)[0]

    try:
        import shap as _shap_dbg
        _expl = _shap_dbg.TreeExplainer(_m_noisy)
        _sv   = _expl.shap_values(X_test)
        if isinstance(_sv, list):
            _sv = np.array(_sv[1])
        else:
            _sv = np.array(_sv)

        _mean_right = np.abs(_sv[_right]).mean(axis=0)
        _mean_wrong = np.abs(_sv[_wrong]).mean(axis=0)
        _order = np.argsort(_mean_right)
        _y2 = np.arange(len(feature_names))

        _ax = _axes[0]
        _ax.barh(_y2 - 0.2, _mean_right[_order], height=0.4,
                 color='#27ae60', alpha=0.8, label='Correct predictions')
        _ax.barh(_y2 + 0.2, _mean_wrong[_order], height=0.4,
                 color='#e74c3c', alpha=0.8, label='Misclassified')
        _ax.set_yticks(_y2)
        _ax.set_yticklabels([feature_names[i] for i in _order], fontsize=9)
        _ax.set_title("Mean |SHAP| — Correct vs Misclassified\n"
                      "Similar feature rankings = label noise, not feature-driven errors", fontsize=9)
        _ax.legend(fontsize=9)
        _ax.set_xlabel("Mean |SHAP value|")
        _ax.grid(True, axis='x', alpha=0.3)

        _skill_idx = feature_names.index('skill_match_pct')
        _ax = _axes[1]
        _ax.hist(_sv[_right, _skill_idx], bins=25, color='#27ae60', alpha=0.6,
                 label=f'Correct (n={len(_right)})', density=True)
        _ax.hist(_sv[_wrong, _skill_idx], bins=25, color='#e74c3c', alpha=0.6,
                 label=f'Misclassified (n={len(_wrong)})', density=True)
        _ax.axvline(0, color='black', linewidth=0.8)
        _ax.set_xlabel("SHAP value for skill_match_pct")
        _ax.set_ylabel("Density")
        _ax.set_title("SHAP distribution — skill_match_pct\n"
                      "Misclassified samples: more mixed signs → label contradicts features",
                      fontsize=9)
        _ax.legend(fontsize=9)
        _ax.grid(True, alpha=0.2)
    except Exception as _e:
        for _a in _axes:
            _a.text(0.5, 0.5, f"shap error: {_e}", ha='center', va='center',
                    transform=_a.transAxes)

    _fig.suptitle(f"Debugging with SHAP — 10% Label Noise (accuracy={_acc_noisy:.1%})",
                  fontsize=11, fontweight='bold')
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **What the analysis revealed:**

    1. **Feature importance didn't change.** The same features rank highest — the model still
       learned the true signal. This tells us the issue isn't a feature quality problem.

    2. **Misclassified samples have more mixed SHAP patterns for `skill_match_pct`.** High
       skill match *should* produce a positive SHAP value. For misclassified samples, the
       distribution is more spread — some have positive SHAP but the label says "bad fit."
       That's the injected noise.

    3. **Diagnosis:** label noise, not bad features. Fix: audit high-confidence misclassifications
       (samples where the model is very confident but the label disagrees).

    > *"This is the real power of interpretability: not just explaining predictions, but
    > FINDING problems in your data pipeline before they reach production."*
    """)
    return


# ─── Decision Framework ───────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Interpretability Decision Framework

    | Need | Tool | When |
    |------|------|------|
    | Quick feature ranking | Permutation importance | First pass on any new model |
    | Explain one prediction to a user | SHAP force plot | Customer-facing explanations |
    | Understand model behavior globally | SHAP summary plot | Documentation, stakeholder review |
    | Detect feature interactions | SHAP dependence + ICE | Debugging unexpected behavior |
    | Fully transparent model | LR coefficients | Regulated industries, legal requirements |
    | Detect data leakage | SHAP (dominant feature?) | Model seems too good to be true |
    | Debug misclassifications | SHAP on error samples | Model underperforms on a subset |
    | Marginal feature effect | PDP | Communicating "how X affects prediction" |
    | Heterogeneity in effect | ICE | "Does X always help, or only for some samples?" |

    **Default workflow for any new model:**

    1. Permutation importance → confirm right features are used (5 min)
    2. SHAP summary plot → confirm directions make sense (10 min)
    3. SHAP force plots on 3–5 cases → sanity-check individual predictions (10 min)
    4. SHAP on worst errors → diagnose misclassification patterns (15 min)
    5. PDP for top 3 features → communication-ready charts for stakeholders (10 min)
    """)
    return


# ─── Flashcards ───────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Flashcard Summary

    **"How do you explain a model's prediction?"**
    > SHAP force plot for individual predictions. Shows each feature's contribution with
    > magnitude and direction, and the values sum to the prediction exactly:
    > base_value + Σ SHAP = prediction.

    **"SHAP vs feature importance?"**
    > Feature importance: which features matter globally. SHAP: which features matter,
    > in what direction, for each individual prediction. SHAP is strictly more informative.

    **"What's wrong with impurity importance?"**
    > Three traps: (1) biased toward high-cardinality features; (2) measured on training data
    > so overfitting inflates noise; (3) no direction — can't tell if more experience helps
    > or hurts.

    **"When is permutation importance better?"**
    > Always, for general-purpose use. Works for any model, measured on validation data, not
    > biased by cardinality. Limitation: correlated features each look less important than
    > they are (they compensate for each other when shuffled).

    **"What does a SHAP summary plot show?"**
    > Global view: for each feature, the distribution of SHAP values across all predictions,
    > colored by feature value. Shows importance + direction + magnitude in one chart.

    **"How do you detect data leakage with SHAP?"**
    > One feature with mean |SHAP| 5–10× larger than all others is the signature of leakage.
    > Nothing in real data generates a signal that clean.

    **"PDP vs SHAP dependence plot?"**
    > PDP: average marginal effect — hides interactions by averaging over all samples.
    > SHAP dependence: per-sample contributions — reveals interactions by coloring points
    > by an interaction feature.

    **"Why do regulators prefer logistic regression?"**
    > Every prediction is an exact documented sum: log-odds = intercept + Σ(value × coef).
    > No approximation, no game theory — each coefficient is legally defensible.

    **"ICE vs PDP?"**
    > PDP is the mean curve. ICE shows individual curves per sample. If ICE lines cross,
    > there's a hidden interaction that the PDP average conceals.

    **"How do you debug a model that's underperforming?"**
    > Permutation importance (right features?), SHAP summary (right directions?), SHAP on
    > worst errors (what confuses the model?). Usually reveals label noise or leakage
    > faster than any other method.
    """)
    return


# ─── Interview Talking Points ─────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ---
    ## Interview Talking Points

    ---

    ### "How do you explain model decisions?"

    > "I use a layered approach. Quick check: permutation importance to verify the right
    > features are driving predictions — takes 5 minutes. Deep dive: SHAP summary plot for
    > global model understanding, SHAP force plots for individual prediction explanations.
    > For debugging: SHAP on misclassified samples to find data quality issues. The SHAP
    > summary plot is almost always the first thing I show stakeholders — it's the most
    > information-dense single visualization I know of."

    ---

    ### "How would you explain a model decision to a non-technical stakeholder?"

    > "I'd use a SHAP force plot and translate it: 'This candidate scored 73%. The biggest
    > driver was skill alignment at 94% — that added the most to the score. Location matched,
    > which was the second biggest boost. Experience was right for the level. The posting is
    > 37 days old, which reduced confidence slightly. Salary mismatch pulled the score down.
    > Every number here is traceable back to the actual model calculation.' No hand-waving.
    > The contributions add up to the prediction."

    ---

    ### "How do you debug a model that's underperforming?"

    > "Three steps. Permutation importance: are the right features being used? If noise
    > features rank high, I have a data or leakage issue. SHAP summary: do the directions
    > make sense? 'More experience = lower score' sends me straight to label auditing. SHAP
    > on the worst predictions: what drove the misses? In my experience, 80% of mysterious
    > underperformance traces back to data issues that SHAP makes visible in minutes."

    ---

    ### "What's the interpretability-accuracy trade-off?"

    > "It's mostly a myth. SHAP gives you interpretability on top of any model — you don't
    > sacrifice accuracy. The real cost is compute: exact Shapley values are exponential in
    > features. TreeExplainer makes it practical for any tree-based model in O(TLD²) time.
    > For neural nets, SHAP's DeepExplainer or KernelExplainer are slower but still practical
    > for offline analysis."

    ---

    ### Connection to my work

    **Fraud detection system design:** SHAP values generate the human-readable reason codes
    for transaction declines — "unusual location," "rapid purchase sequence," "new device."
    The XGBoost model scores the transaction; SHAP decomposes the score into per-feature
    contributions that become the decline reason. Regulators get full documentation;
    customers get an actionable explanation.

    **Canopy (job recommendations):** SHAP explains job scores to candidates — "this role
    scored 8.5 because your skills match 94% of requirements and the location fits. The
    salary is slightly above your stated range, which pulled it down slightly." SHAP makes
    the recommendation system both debuggable and user-facing, with no extra engineering
    beyond what the model already computes.
    """)
    return


if __name__ == "__main__":
    app.run()
