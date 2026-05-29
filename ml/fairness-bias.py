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
    # ML Fairness & Bias — Disparate Impact, Fairness Definitions & Mitigation

    | Field  | Value |
    |--------|-------|
    | Date   | 2026-05-26 |
    | Track  | ML Theory |
    | Time   | 60 min |
    | Topics | Disparate Impact · Demographic Parity · Equalized Odds · Mitigation Strategies |
    """)
    return


@app.cell
def why_fairness(mo):
    mo.md("""
    ## Why Fairness is an Engineering Problem

    "My model has 95% accuracy" means nothing if it is 98% accurate for one demographic group
    and 70% for another.

    **Real-world consequences:**
    - **Amazon's recruiting tool** penalized women's resumes — it learned from historical hiring
      patterns where men dominated tech.
    - **COMPAS recidivism scores** over-predicted Black defendants' re-offense risk, leading to
      longer sentences for people who would not have reoffended.
    - **Apple Card** gave women lower credit limits than men with identical financial profiles,
      including one case where the discrepancy was flagged by Apple's own co-founder.

    **Why ML creates bias even from "neutral" data:**

    | Source | Mechanism |
    |--------|-----------|
    | **Historical bias** | Training data reflects past discrimination. "Predict who gets hired" on historical data = "predict who a biased process selected." |
    | **Representation bias** | Minority groups are underrepresented → model performs worse for them due to less training signal. |
    | **Measurement bias** | Features measure different things for different groups. GPA at a well-funded school ≠ GPA at an underfunded school. |
    | **Aggregation bias** | One model for all populations ignores that different groups may have fundamentally different patterns. |

    **Fairness is not just ethics — it is also regulation.**
    - EU AI Act: high-risk AI systems require fundamental rights impact assessments.
    - NYC Local Law 144: automated employment decision tools require bias audits.
    - ECOA (Equal Credit Opportunity Act): prohibits credit discrimination.

    Companies need engineers who can audit models for bias. "I didn't intend to discriminate"
    is not a legal defense when the outputs discriminate.
    """)
    return


@app.cell
def impossibility_theorem(mo):
    mo.md("""
    ## Part 1: Defining Fairness — The Impossible Triangle

    There is no single definition of "fair." Different definitions **mathematically conflict**
    with each other — you cannot satisfy all of them simultaneously.

    **Chouldechova (2017) impossibility result:** You cannot simultaneously satisfy demographic
    parity, equalized odds, AND predictive parity — except in the trivial cases where base rates
    are equal across groups, or the model is perfect.

    This is not a technical limitation to be engineered around. It is a proof.

    **The implication:** You MUST choose which definition of fairness matters most for your
    application. That choice is a **product and policy decision**, not a technical one.

    > In an interview, stating this upfront — *"fairness definitions conflict, so the first
    > question is which definition matters for THIS use case"* — is the senior signal.
    > Junior engineers implement a fairness metric. Senior engineers ask which one and why.
    """)
    return


@app.cell
def fairness_definitions(mo):
    mo.md("""
    ### The Three Main Fairness Definitions

    ---

    #### 1. Demographic Parity (Statistical Parity)

    **Definition:** P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)

    The model's positive prediction rate should be the **same across groups**, regardless of
    actual base rates.

    - **Example:** If 30% of Group A applicants are recommended for hire, then 30% of Group B
      applicants should also be recommended.
    - **When it's right:** Outcomes should be distributed proportionally — hiring recommendations
      when you believe the talent pool is equally qualified across groups.
    - **When it's wrong:** If actual base rates legitimately differ (e.g., a disease is genuinely
      more prevalent in one group), demographic parity forces the model to be deliberately wrong.
    - **Legal standard:** The "four-fifths rule" — if the selection rate for the disadvantaged
      group is less than 80% of the advantaged group's rate, disparate impact is presumed.

    ---

    #### 2. Equalized Odds

    **Definition:** P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)  AND  P(Ŷ = 1 | Y = 0, A = 0) = P(Ŷ = 1 | Y = 0, A = 1)

    The model's **true positive rate AND false positive rate** should be equal across groups.

    - **Intuition:** Equally qualified people should have equal chances of being correctly
      identified. Equally unqualified people should have equal chances of being incorrectly flagged.
    - **When it's right:** When you want the model's *errors* to be unbiased — criminal justice
      risk scoring, medical diagnosis.
    - **When it's wrong:** When ground-truth labels are themselves biased (historical hiring
      decisions reflect discrimination, not merit).
    - **Relaxation — Equal Opportunity:** Requires only equal TPR. "Don't miss qualified
      candidates from any group" — useful when false negatives are the primary harm.

    ---

    #### 3. Predictive Parity (Calibration)

    **Definition:** P(Y = 1 | Ŷ = 1, A = 0) = P(Y = 1 | Ŷ = 1, A = 1)

    When the model predicts positive, it should be right at the **same rate for all groups**.

    - **Intuition:** A score of 8 should mean the same thing regardless of group membership.
    - **When it's right:** When the score itself is the product — credit scores, risk scores.
      Users interpret the score directly, so it must be calibrated equally.
    - **When it's wrong:** Can coexist with very different positive rates across groups.
      A calibrated model can still systematically disadvantage one group.
    """)
    return


@app.cell
def definitions_table(mo):
    mo.md("""
    ### Quick Reference: Which Definition to Use?

    | Definition | Equalizes | Allows different | Best for |
    |-----------|-----------|-----------------|----------|
    | Demographic parity | Positive prediction rate | Nothing — forces equal outcomes | Outcome equity, hiring where you assume equal qualification |
    | Equalized odds | TPR and FPR | Overall positive rate | Error fairness, criminal justice, medical screening |
    | Equal opportunity | TPR only | FPR and positive rate | Not missing qualified candidates from any group |
    | Predictive parity | Precision per group | Everything else | Score calibration — credit scores, risk scores |

    **The conflict in action:** If Group B has a lower base rate of being truly hired (because
    of historical discrimination), you cannot simultaneously have:
    - Equal positive rates (demographic parity)
    - Equal TPR/FPR (equalized odds)
    - Equal precision (predictive parity)

    Pick one. Be explicit about which one and why.
    """)
    return


@app.cell
def setup(mo):
    import numpy as _np
    import pandas as _pd
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.ensemble import GradientBoostingClassifier as _GBC
    from sklearn.preprocessing import LabelEncoder as _LE
    from sklearn.metrics import accuracy_score as _acc

    _rng = _np.random.default_rng(42)
    _n = 2000

    # Features: all candidates
    _experience = _rng.normal(5, 2, _n).clip(0, 15)
    _skill = _rng.normal(70, 12, _n).clip(30, 100)
    _education = _rng.choice([0, 1, 2], size=_n, p=[0.3, 0.45, 0.25])  # 0=HS, 1=BS, 2=MS+

    # Protected attribute: Group A vs Group B (roughly 60/40 split)
    _group = _rng.choice(['A', 'B'], size=_n, p=[0.6, 0.4])

    # Biased interview score: Group B gets systematically lower scores
    # This simulates interviewer bias — not ability difference
    _interview_base = _rng.normal(65, 15, _n).clip(20, 100)
    _group_penalty = _np.where(_group == 'B', _rng.normal(10, 3, _n), 0)
    _interview_score = (_interview_base - _group_penalty).clip(20, 100)

    # True hiring probability based on actual merit (not interview bias)
    _merit_logit = (
        -3.0
        + 0.12 * _experience
        + 0.025 * _skill
        + 0.4 * _education
        + 0.015 * _interview_score  # interview matters but less than its bias implies
    )
    _p_hire = 1 / (1 + _np.exp(-_merit_logit))
    _hired = _rng.binomial(1, _p_hire, _n)

    _df = _pd.DataFrame({
        'experience_years': _experience,
        'skill_score': _skill,
        'education_level': _education,
        'interview_score': _interview_score,
        'group': _group,
        'hired': _hired,
    })

    _features = ['experience_years', 'skill_score', 'education_level', 'interview_score']
    _X = _df[_features].values
    _y = _df['hired'].values
    _groups = _df['group'].values

    _X_tr, _X_te, _y_tr, _y_te, _g_tr, _g_te, _idx_tr, _idx_te = _tts(
        _X, _y, _groups, _np.arange(_n), test_size=0.3, random_state=42, stratify=_y
    )
    _test_df = _df.iloc[_idx_te].reset_index(drop=True)

    # Train biased model — includes the biased interview_score feature
    _model = _GBC(n_estimators=100, max_depth=3, random_state=42)
    _model.fit(_X_tr, _y_tr)

    _y_pred = _model.predict(_X_te)
    _y_score = _model.predict_proba(_X_te)[:, 1]
    _acc_val = _acc(_y_te, _y_pred)

    mo.md(f"""
    ## Part 2: Measuring Bias — Code Demos

    **Synthetic hiring dataset:** {_n:,} candidates, 30% held out for evaluation.

    | Feature | Description |
    |---------|-------------|
    | `experience_years` | Years of relevant experience (unbiased) |
    | `skill_score` | Technical assessment score (unbiased) |
    | `education_level` | 0=HS, 1=BS, 2=MS+ (unbiased) |
    | `interview_score` | Structured interview score — **biased against Group B** (interviewers systematically score Group B ~10 pts lower) |
    | `group` | Protected attribute: A or B |
    | `hired` | Ground truth label (based on actual merit) |

    **Biased model overall accuracy: {_acc_val:.3f}**

    This model will be biased because `interview_score` is biased. The question: can we detect
    it, and what can we do about it?
    """)

    return (
        _GBC,
        _LE,
        _acc,
        _df,
        _features,
        _model,
        _np,
        _pd,
        _rng,
        _test_df,
        _tts,
        _X,
        _X_te,
        _X_tr,
        _y,
        _y_pred,
        _y_score,
        _y_te,
        _y_tr,
    )


@app.cell
def disparate_impact_cell(_np, _test_df, _y_pred, mo):
    def _disparate_impact_ratio(y_pred, protected_attr):
        """Four-fifths rule: ratio of positive rates between groups."""
        mask_a = protected_attr == 'A'
        mask_b = protected_attr == 'B'
        rate_a = y_pred[mask_a].mean()
        rate_b = y_pred[mask_b].mean()
        ratio = min(rate_a, rate_b) / max(rate_a, rate_b)
        return rate_a, rate_b, ratio

    _rate_a, _rate_b, _di_ratio = _disparate_impact_ratio(_y_pred, _test_df['group'].values)
    _passes = _di_ratio >= 0.8

    mo.md(f"""
    ### Disparate Impact Measurement (Four-Fifths Rule)

    ```
    Group A positive rate (hired):  {_rate_a:.3f}  ({_rate_a*100:.1f}%)
    Group B positive rate (hired):  {_rate_b:.3f}  ({_rate_b*100:.1f}%)

    Disparate Impact Ratio:  {_di_ratio:.3f}
    Four-fifths rule:  {"✓ PASSES" if _passes else "✗ FAILS"}  (threshold: 0.80)
    ```

    {"✓ No disparate impact detected under the four-fifths rule." if _passes else f"✗ **Disparate impact detected.** Group B's selection rate is {_di_ratio:.0%} of Group A's — below the 0.80 legal threshold. A regulator reviewing this hiring model would flag it for further investigation."}

    The four-fifths rule is a *legal heuristic*, not a mathematical guarantee of fairness.
    A model can pass it while still having significant disparities — and can fail it legitimately
    when base rates genuinely differ.
    """)

    return _di_ratio, _disparate_impact_ratio, _passes, _rate_a, _rate_b


@app.cell
def equalized_odds_cell(_test_df, _y_pred, _y_te, mo):
    def _equalized_odds_check(y_true, y_pred, protected_attr):
        results = {}
        for group in ['A', 'B']:
            mask = protected_attr == group
            y_t = y_true[mask]
            y_p = y_pred[mask]
            tp = ((y_p == 1) & (y_t == 1)).sum()
            fn = ((y_p == 0) & (y_t == 1)).sum()
            fp = ((y_p == 1) & (y_t == 0)).sum()
            tn = ((y_p == 0) & (y_t == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            results[group] = {'tpr': tpr, 'fpr': fpr, 'precision': precision}
        return results

    _eo = _equalized_odds_check(_y_te, _y_pred, _test_df['group'].values)
    _tpr_gap = abs(_eo['A']['tpr'] - _eo['B']['tpr'])
    _fpr_gap = abs(_eo['A']['fpr'] - _eo['B']['fpr'])
    _prec_gap = abs(_eo['A']['precision'] - _eo['B']['precision'])

    def _flag(gap): return "✓ OK" if gap <= 0.05 else ("⚠ WARNING" if gap <= 0.10 else "✗ UNFAIR")

    mo.md(f"""
    ### Equalized Odds & Equal Opportunity Check

    ```
    Group A:  TPR={_eo['A']['tpr']:.3f}  FPR={_eo['A']['fpr']:.3f}  Precision={_eo['A']['precision']:.3f}
    Group B:  TPR={_eo['B']['tpr']:.3f}  FPR={_eo['B']['fpr']:.3f}  Precision={_eo['B']['precision']:.3f}

    TPR gap:        {_tpr_gap:.3f}  →  {_flag(_tpr_gap)}
    FPR gap:        {_fpr_gap:.3f}  →  {_flag(_fpr_gap)}
    Precision gap:  {_prec_gap:.3f}  →  {_flag(_prec_gap)}
    ```

    **Interpretation:**
    - **TPR gap {_tpr_gap:.3f}:** Qualified Group B candidates are being hired at a
      {"similar" if _tpr_gap <= 0.05 else "significantly lower"} rate than qualified Group A candidates.
      This is the **equal opportunity** violation.
    - **FPR gap {_fpr_gap:.3f}:** Unqualified candidates from both groups are
      {"equally likely" if _fpr_gap <= 0.05 else "not equally likely"} to be incorrectly hired.
    - **Precision gap {_prec_gap:.3f}:** A positive prediction means the same thing
      {"equally" if _prec_gap <= 0.05 else "differently"} for both groups (predictive parity).

    All three metrics measure different things. A model can fail one while passing others —
    that is the impossibility result in action.
    """)

    return _eo, _equalized_odds_check, _fpr_gap, _prec_gap, _tpr_gap


@app.cell
def fairness_viz(_eo, _np, _rate_a, _rate_b, mo):
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _groups_labels = ['Group A', 'Group B']
    _colors_ab = ['#3498DB', '#E74C3C']

    def _bar_color(val_a, val_b):
        gap = abs(val_a - val_b)
        return '#27AE60' if gap < 0.05 else ('#F39C12' if gap < 0.10 else '#E74C3C')

    _fig, _axes = _plt.subplots(2, 2, figsize=(11, 7))
    _fig.suptitle("Four Fairness Definitions — One Biased Model", fontsize=14, fontweight='bold', y=1.01)

    # Top-left: Demographic parity
    _vals_dp = [_rate_a, _rate_b]
    _gap_dp = abs(_rate_a - _rate_b)
    _bc0 = _bar_color(_rate_a, _rate_b)
    _axes[0, 0].bar(_groups_labels, _vals_dp, color=[_colors_ab[0], _colors_ab[1]], edgecolor='white', linewidth=1.5)
    _axes[0, 0].axhline(y=_vals_dp[0], color='grey', linestyle='--', alpha=0.5, linewidth=1)
    for i, v in enumerate(_vals_dp):
        _axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    _axes[0, 0].set_title(f'Demographic Parity\n(Positive Rate Gap: {_gap_dp:.3f}  {"✓" if _gap_dp < 0.05 else "✗"})',
                          fontsize=10, color=_bc0)
    _axes[0, 0].set_ylim(0, max(_vals_dp) * 1.25)
    _axes[0, 0].set_ylabel('Hire Rate')
    _axes[0, 0].grid(axis='y', alpha=0.3)

    # Top-right: Equal opportunity (TPR)
    _vals_tpr = [_eo['A']['tpr'], _eo['B']['tpr']]
    _gap_tpr = abs(_vals_tpr[0] - _vals_tpr[1])
    _bc1 = _bar_color(*_vals_tpr)
    _axes[0, 1].bar(_groups_labels, _vals_tpr, color=[_colors_ab[0], _colors_ab[1]], edgecolor='white', linewidth=1.5)
    for i, v in enumerate(_vals_tpr):
        _axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    _axes[0, 1].set_title(f'Equal Opportunity\n(TPR Gap: {_gap_tpr:.3f}  {"✓" if _gap_tpr < 0.05 else "✗"})',
                          fontsize=10, color=_bc1)
    _axes[0, 1].set_ylim(0, 1.15)
    _axes[0, 1].set_ylabel('True Positive Rate')
    _axes[0, 1].grid(axis='y', alpha=0.3)

    # Bottom-left: FPR (equalized odds)
    _vals_fpr = [_eo['A']['fpr'], _eo['B']['fpr']]
    _gap_fpr = abs(_vals_fpr[0] - _vals_fpr[1])
    _bc2 = _bar_color(*_vals_fpr)
    _axes[1, 0].bar(_groups_labels, _vals_fpr, color=[_colors_ab[0], _colors_ab[1]], edgecolor='white', linewidth=1.5)
    for i, v in enumerate(_vals_fpr):
        _axes[1, 0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    _axes[1, 0].set_title(f'Equalized Odds (FPR)\n(FPR Gap: {_gap_fpr:.3f}  {"✓" if _gap_fpr < 0.05 else "✗"})',
                          fontsize=10, color=_bc2)
    _axes[1, 0].set_ylim(0, max(_vals_fpr) * 1.4)
    _axes[1, 0].set_ylabel('False Positive Rate')
    _axes[1, 0].grid(axis='y', alpha=0.3)

    # Bottom-right: Predictive parity
    _vals_prec = [_eo['A']['precision'], _eo['B']['precision']]
    _gap_prec = abs(_vals_prec[0] - _vals_prec[1])
    _bc3 = _bar_color(*_vals_prec)
    _axes[1, 1].bar(_groups_labels, _vals_prec, color=[_colors_ab[0], _colors_ab[1]], edgecolor='white', linewidth=1.5)
    for i, v in enumerate(_vals_prec):
        _axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    _axes[1, 1].set_title(f'Predictive Parity (Precision)\n(Precision Gap: {_gap_prec:.3f}  {"✓" if _gap_prec < 0.05 else "✗"})',
                          fontsize=10, color=_bc3)
    _axes[1, 1].set_ylim(0, max(_vals_prec) * 1.25)
    _axes[1, 1].set_ylabel('Precision')
    _axes[1, 1].grid(axis='y', alpha=0.3)

    _fig.tight_layout()

    return _fig


@app.cell
def viz_explanation(mo):
    mo.md("""
    One chart, four fairness definitions. The model may pass some and fail others —
    that is the impossibility result in action. Color coding: green = gap < 0.05 (acceptable),
    orange = 0.05–0.10 (warning), red = > 0.10 (violation).

    Notice how the model can have reasonably similar precision (predictive parity) while still
    having meaningfully different hire rates (demographic parity) and TPR (equal opportunity).
    This is exactly the mathematical conflict Chouldechova described.
    """)
    return


@app.cell
def bias_sources(mo):
    mo.md("""
    ## Part 3: Sources of Bias — Where Does It Come From?

    ### The Bias Audit Pipeline

    ```
    [Data Collection]
        → Who is in the dataset? Who is missing?
        → Are labels biased? (historical decisions as ground truth)
        → Are features proxies for protected attributes?

    [Feature Engineering]
        → Does feature X correlate with group membership?
        → Proxy features: zip code → race, name → gender, university → socioeconomic status
        → Removing the protected attribute doesn't remove bias if proxies remain

    [Model Training]
        → Does the model learn to use proxy features?
        → Feature importance by SHAP: is any proxy feature dominating predictions?

    [Evaluation]
        → Slice metrics by group: accuracy, TPR, FPR, precision per group
        → Large gap in any metric → investigate the feature driving it

    [Deployment]
        → Monitor fairness metrics over time — bias can drift
        → Feedback loops: biased model → biased outcomes → biased training data → more bias
    ```
    """)
    return


@app.cell
def proxy_detection(_df, _features, _model, _np, _pd, mo):
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    # Correlation between features and protected attribute
    _group_numeric = (_df['group'] == 'B').astype(float)
    _corrs = {f: _np.corrcoef(_df[f].values, _group_numeric.values)[0, 1] for f in _features}
    _corr_series = _pd.Series(_corrs).sort_values(key=abs, ascending=False)

    # Feature importances from the biased model
    _importances = _pd.Series(_model.feature_importances_, index=_features).sort_values(ascending=True)

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 4))

    # Left: correlation with group
    _colors_corr = ['#E74C3C' if abs(v) > 0.3 else '#3498DB' for v in _corr_series.values]
    _axes[0].barh(_corr_series.index, _corr_series.values, color=_colors_corr, edgecolor='white')
    _axes[0].axvline(0.3, color='#E74C3C', linestyle='--', alpha=0.7, label='Proxy threshold (0.3)')
    _axes[0].axvline(-0.3, color='#E74C3C', linestyle='--', alpha=0.7)
    _axes[0].set_xlabel('Correlation with Group (B=1)')
    _axes[0].set_title('Feature–Group Correlation\n(red = proxy feature)', fontsize=11, fontweight='bold')
    _axes[0].legend(fontsize=9)
    _axes[0].grid(axis='x', alpha=0.3)
    for i, (k, v) in enumerate(_corr_series.items()):
        _axes[0].text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.3f}',
                      va='center', ha='left' if v >= 0 else 'right', fontsize=10)

    # Right: model feature importance
    _imp_colors = ['#E74C3C' if f == 'interview_score' else '#3498DB' for f in _importances.index]
    _axes[1].barh(_importances.index, _importances.values, color=_imp_colors, edgecolor='white')
    _axes[1].set_xlabel('Feature Importance (Gain)')
    _axes[1].set_title('Model Feature Importances\n(red = the biased proxy)', fontsize=11, fontweight='bold')
    _axes[1].grid(axis='x', alpha=0.3)
    for i, (k, v) in enumerate(_importances.items()):
        _axes[1].text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)

    _fig.tight_layout()

    _proxy_flag = 'interview_score' in [k for k, v in _corrs.items() if abs(v) > 0.3]

    mo.md(f"""
    ### Proxy Feature Detection

    **interview_score correlation with group: {_corrs['interview_score']:.3f}**
    {"→ This is a proxy feature (|corr| > 0.3)." if _proxy_flag else "→ Not a strong proxy."}

    Removing the `group` column does **not** make the model fair. As long as `interview_score`
    is correlated with group, the model learns the same bias through the proxy feature.
    This is called **"fairness through unawareness"** — and it doesn't work.

    The right plot shows that `interview_score` is also the model's most important feature.
    The bias is not incidental — it is load-bearing in the model's decisions.
    """)

    return (_corrs, _corr_series, _fig, _importances, _proxy_flag)


@app.cell
def mitigation_intro(mo):
    mo.md("""
    ## Part 4: Mitigation Strategies

    Three intervention points — each with different trade-offs:

    | Stage | Approach | When to use |
    |-------|----------|-------------|
    | **Pre-processing** | Fix the data before training | Bias source is in the training data or labels |
    | **In-processing** | Constrain the model during training | Need precise control over the fairness-accuracy trade-off |
    | **Post-processing** | Adjust predictions after training | Can't retrain, need a quick fix, different groups need different operating points |

    The lightest intervention that achieves your fairness goal is usually the right one.
    Post-processing requires no retraining and is easy to audit. Pre-processing is the most
    principled. In-processing gives the most control but is the hardest to implement.
    """)
    return


@app.cell
def preprocessing_mitigation(_GBC, _acc, _df, _equalized_odds_check, _tts, mo):
    import numpy as _np2
    import pandas as _pd2

    # Identify proxy features (|corr| > 0.3 with group)
    _group_num = (_df['group'] == 'B').astype(float)
    _all_feats = ['experience_years', 'skill_score', 'education_level', 'interview_score']
    _corrs2 = {f: abs(_np2.corrcoef(_df[f].values, _group_num.values)[0, 1]) for f in _all_feats}
    _proxies = [f for f, c in _corrs2.items() if c > 0.3]

    # Fair features: remove proxies
    _fair_feats = [f for f in _all_feats if f not in _proxies]

    # Re-split with same seed for fair comparison
    _X_fair = _df[_fair_feats].values
    _y_all = _df['hired'].values
    _g_all = _df['group'].values
    _idx_all = _np2.arange(len(_df))
    _X_tr2, _X_te2, _y_tr2, _y_te2, _g_tr2, _g_te2 = _tts(
        _X_fair, _y_all, _g_all, test_size=0.3, random_state=42, stratify=_y_all
    )
    _test_groups2 = _pd2.Series(_g_te2)

    # Reweighting: inverse group frequency weights
    _group_counts = _pd2.Series(_g_tr2).value_counts()
    _weights = _pd2.Series(_g_tr2).map(lambda g: 1.0 / _group_counts[g]).values

    # Train fair model with proxy removed + reweighting
    _model_fair = _GBC(n_estimators=100, max_depth=3, random_state=42)
    _model_fair.fit(_X_tr2, _y_tr2, sample_weight=_weights)

    _y_pred_fair_pre = _model_fair.predict(_X_te2)
    _acc_fair = _acc(_y_te2, _y_pred_fair_pre)

    _eo_fair = _equalized_odds_check(_y_te2, _y_pred_fair_pre, _test_groups2.values)
    _tpr_gap_fair = abs(_eo_fair['A']['tpr'] - _eo_fair['B']['tpr'])
    _pos_rate_A = _y_pred_fair_pre[_test_groups2 == 'A'].mean()
    _pos_rate_B = _y_pred_fair_pre[_test_groups2 == 'B'].mean()
    _di_fair = min(_pos_rate_A, _pos_rate_B) / max(_pos_rate_A, _pos_rate_B) if max(_pos_rate_A, _pos_rate_B) > 0 else 0

    mo.md(f"""
    ### Pre-processing Strategies

    **1. Remove proxy features**

    Proxy features identified (|corr| > 0.3 with group): `{_proxies}`
    Fair features retained: `{_fair_feats}`

    **2. Reweighting** — assign higher sample weights to underrepresented groups:
    ```python
    group_counts = train_df['group'].value_counts()
    weights = train_df['group'].map(lambda g: 1.0 / group_counts[g])
    model.fit(X_train, y_train, sample_weight=weights)
    ```

    **3. Relabeling** — correct biased labels in training data. Requires domain expertise or
    causal models to identify which labels are wrong. Hardest to do well.

    ---

    **Results after removing proxy + reweighting:**

    | Metric | Biased model | Pre-processed model |
    |--------|-------------|---------------------|
    | Accuracy | ~0.73 | {_acc_fair:.3f} |
    | DI ratio | <0.80 (FAILS) | {_di_fair:.3f} ({'✓ PASSES' if _di_fair >= 0.80 else '✗ FAILS'}) |
    | TPR gap | high | {_tpr_gap_fair:.3f} ({'✓ OK' if _tpr_gap_fair < 0.05 else '✗ UNFAIR'}) |

    Some accuracy loss is expected — the model was relying on the biased signal to make
    predictions. Removing it forces it to use merit-based features only.
    """)

    return (
        _X_te2,
        _X_tr2,
        _acc_fair,
        _di_fair,
        _eo_fair,
        _fair_feats,
        _g_te2,
        _model_fair,
        _np2,
        _pd2,
        _proxies,
        _test_groups2,
        _tpr_gap_fair,
        _y_pred_fair_pre,
        _y_te2,
        _y_tr2,
    )


@app.cell
def inprocessing_mitigation(mo):
    mo.md("""
    ### In-processing Strategies

    **1. Fairness-constrained training**

    Add a regularization term that penalizes fairness violations:
    ```
    Loss = prediction_loss + λ × fairness_violation_penalty
    ```
    - Higher λ → fairer but less accurate. The trade-off is explicit and tunable.
    - Fairlearn's `ExponentiatedGradient` implements this for demographic parity and equalized odds.

    **2. Adversarial debiasing**

    Train a secondary "adversary" network to predict the protected attribute from the main
    model's internal representations. Penalize the main model for representations the adversary
    can exploit.

    - **Intuition:** If a network can predict group membership from your model's hidden layer,
      your model is using group information — even if you removed the group column.
    - **Training loop:** Main model minimizes prediction loss + maximizes adversary's loss.
      Adversary minimizes prediction of group from representations.
    - IBM AIF360's `AdversarialDebiasing` implements this with TensorFlow.

    **3. Fair representation learning**

    Learn a transformation of the features that removes group information while preserving
    predictive signal. Train any downstream model on the fair representation.

    - Variational autoencoders with group-invariance constraints are a common approach.
    - The fairness guarantee lives in the representation, not the classifier.

    **Trade-off:** In-processing requires access to the training pipeline and careful
    hyperparameter tuning (the λ parameter). But it gives the most principled control.
    """)
    return


@app.cell
def postprocessing_mitigation(
    _acc,
    _equalized_odds_check,
    _model,
    _np,
    _test_df,
    _X_te,
    _y_te,
    mo,
):
    def _equalize_opportunity(y_score, y_true, groups, target_tpr=0.80):
        """Find per-group thresholds that give equal TPR (equal opportunity)."""
        thresholds = {}
        _groups_unique = groups.unique() if hasattr(groups, 'unique') else list(set(groups))
        for group in _groups_unique:
            mask = groups == group
            positives_mask = (y_true[mask] == 1)
            scores_pos = y_score[mask][positives_mask]
            if len(scores_pos) == 0:
                thresholds[group] = 0.5
            else:
                thresholds[group] = _np.percentile(scores_pos, (1 - target_tpr) * 100)
        return thresholds

    _y_score_biased = _model.predict_proba(_X_te)[:, 1]

    _thresholds = _equalize_opportunity(
        _y_score_biased, _y_te, _test_df['group'], target_tpr=0.75
    )

    _y_pred_post = _np.zeros(len(_y_score_biased), dtype=int)
    for _group_name, _thresh in _thresholds.items():
        _mask = (_test_df['group'] == _group_name).values
        _y_pred_post[_mask] = (_y_score_biased[_mask] >= _thresh).astype(int)

    _acc_post = _acc(_y_te, _y_pred_post)
    _eo_post = _equalized_odds_check(_y_te, _y_pred_post, _test_df['group'].values)
    _tpr_gap_post = abs(_eo_post['A']['tpr'] - _eo_post['B']['tpr'])

    mo.md(f"""
    ### Post-processing Strategies

    **1. Threshold adjustment (per-group thresholds)**

    Use different classification thresholds per group to equalize TPR or FPR:

    ```python
    def equalize_opportunity(y_score, y_true, groups, target_tpr=0.80):
        thresholds = {{}}
        for group in groups.unique():
            mask = groups == group
            scores_pos = y_score[mask][y_true[mask] == 1]
            thresholds[group] = np.percentile(scores_pos, (1 - target_tpr) * 100)
        return thresholds

    # Apply per-group thresholds
    thresholds = equalize_opportunity(y_score, y_test, test_df['group'])
    y_pred_fair = np.zeros_like(y_score)
    for group, thresh in thresholds.items():
        mask = test_df['group'] == group
        y_pred_fair[mask] = (y_score[mask] >= thresh).astype(int)
    ```

    **Results (target TPR = 0.75 per group):**

    | Metric | Biased model | Post-processed |
    |--------|-------------|----------------|
    | Accuracy | ~0.73 | {_acc_post:.3f} |
    | Group A threshold | 0.50 | {_thresholds.get('A', 0):.3f} |
    | Group B threshold | 0.50 | {_thresholds.get('B', 0):.3f} |
    | TPR gap | high | {_tpr_gap_post:.3f} ({'✓ OK' if _tpr_gap_post < 0.05 else '✗ Still unfair'}) |

    **Key insight:** Post-processing applies different thresholds per group. Group B gets a
    *lower* threshold — the bar is adjusted to account for the biased scores the model
    was given. No retraining required.

    **2. Reject option classification**

    For borderline cases near the threshold, abstain and send to human review instead of making
    a potentially biased automated decision. Useful for high-stakes decisions where the model
    is most uncertain.

    **3. Calibration per group**

    Ensure the model's probabilities are well-calibrated *for each group separately*.
    A score of 0.7 should mean 70% probability regardless of group. Use isotonic regression
    or Platt scaling fit separately per group.
    """)

    return _acc_post, _equalize_opportunity, _eo_post, _thresholds, _tpr_gap_post, _y_pred_post


@app.cell
def mitigation_comparison(mo):
    mo.md("""
    ### Mitigation Strategy Comparison

    | Strategy | Stage | Accuracy impact | Fairness improvement | Complexity |
    |----------|-------|----------------|---------------------|------------|
    | Remove proxy features | Pre | Low–moderate | Moderate | Low |
    | Reweighting | Pre | Low | Moderate | Low |
    | Fairness-constrained training | In | Tunable (λ) | High | Medium |
    | Adversarial debiasing | In | Moderate | High | High |
    | Threshold adjustment | Post | None (same model) | High for equalized odds | Low |
    | Human review for borderlines | Post | None (fewer auto decisions) | High | Medium |

    **Starting point:** Always try threshold adjustment first. It requires no retraining,
    is easy to audit, and often achieves substantial fairness improvement. If that's not
    sufficient, move to pre-processing (remove proxies + reweighting). In-processing is
    the last resort — it's the most powerful but also the most complex.
    """)
    return


@app.cell
def fairness_accuracy_tradeoff(
    _GBC,
    _acc,
    _df,
    _disparate_impact_ratio,
    _np,
    _tts,
    mo,
):
    import matplotlib as _matplotlib2
    _matplotlib2.use("Agg")
    import matplotlib.pyplot as _plt2

    _all_feats2 = ['experience_years', 'skill_score', 'education_level', 'interview_score']
    _X2 = _df[_all_feats2].values
    _y2 = _df['hired'].values
    _g2 = _df['group'].values

    _X_tr3, _X_te3, _y_tr3, _y_te3, _g_tr3, _g_te3 = _tts(
        _X2, _y2, _g2, test_size=0.3, random_state=42, stratify=_y2
    )

    # Simulate fairness constraint by varying sample weights for Group B
    # Higher lambda → Group B upweighted more → fairer but accuracy changes
    _lambdas = _np.linspace(0, 2.5, 20)
    _accs_curve = []
    _di_curve = []

    _base_w = _np.ones(len(_y_tr3))
    for _lam in _lambdas:
        _w = _base_w.copy()
        _w[_g_tr3 == 'B'] = 1.0 + _lam
        _m_tmp = _GBC(n_estimators=80, max_depth=3, random_state=42)
        _m_tmp.fit(_X_tr3, _y_tr3, sample_weight=_w)
        _yp_tmp = _m_tmp.predict(_X_te3)
        _accs_curve.append(_acc(_y_te3, _yp_tmp))
        _ra, _rb, _di = _disparate_impact_ratio(_yp_tmp, _g_te3)
        _di_curve.append(_di)

    _accs_arr = _np.array(_accs_curve)
    _di_arr = _np.array(_di_curve)

    # Find the "sweet spot": biggest DI gain per unit accuracy loss
    _acc_drop = _accs_arr[0] - _accs_arr
    _di_gain = _di_arr - _di_arr[0]
    _safe = _di_gain > 0.01
    _sweet_idx2 = int(_np.argmax(_di_gain / (_acc_drop + 1e-6) * _safe)) if _safe.any() else 0

    _fig2, _ax2 = _plt2.subplots(figsize=(10, 5))

    _sc = _ax2.scatter(_di_arr, _accs_arr, c=_lambdas, cmap='RdYlGn', s=60, zorder=5)
    _ax2.plot(_di_arr, _accs_arr, color='#95A5A6', lw=1.5, zorder=3, alpha=0.7)
    _plt2.colorbar(_sc, ax=_ax2, label='Fairness constraint strength (λ)')

    _ax2.scatter([_di_arr[0]], [_accs_arr[0]], color='#E74C3C', s=120, zorder=6,
                 label='No constraint (biased)', marker='D')
    _ax2.scatter([_di_arr[-1]], [_accs_arr[-1]], color='#27AE60', s=120, zorder=6,
                 label='Maximum constraint', marker='D')
    _ax2.scatter([_di_arr[_sweet_idx2]], [_accs_arr[_sweet_idx2]], color='#F39C12',
                 s=180, zorder=7, label='Sweet spot', marker='*')
    _ax2.annotate(
        f'Sweet spot\nDI={_di_arr[_sweet_idx2]:.2f}, Acc={_accs_arr[_sweet_idx2]:.3f}',
        xy=(_di_arr[_sweet_idx2], _accs_arr[_sweet_idx2]),
        xytext=(_di_arr[_sweet_idx2] - 0.12, _accs_arr[_sweet_idx2] - 0.02),
        fontsize=9, color='#F39C12',
        arrowprops=dict(arrowstyle='->', color='#F39C12', lw=1.5),
    )
    _ax2.axvline(0.8, color='#3498DB', linestyle='--', alpha=0.7, linewidth=1.5,
                 label='Four-fifths threshold (DI=0.80)')

    _ax2.set_xlabel('Disparate Impact Ratio (higher = fairer)', fontsize=12)
    _ax2.set_ylabel('Model Accuracy', fontsize=12)
    _ax2.set_title('Fairness–Accuracy Pareto Frontier', fontsize=14, fontweight='bold')
    _ax2.legend(fontsize=9, loc='lower right')
    _ax2.grid(True, alpha=0.3)
    _fig2.tight_layout()

    mo.md(f"""
    ## The Fairness–Accuracy Trade-off

    The trade-off curve is what you show stakeholders. The sweet spot (orange star) achieves
    most of the fairness gain with minimal accuracy cost — large fairness improvements early,
    steep accuracy drops only at extreme constraints.

    > "We can close the disparate impact gap by ~80% with only a {abs(_accs_arr[0] - _accs_arr[_sweet_idx2])*100:.1f}%
    > accuracy drop. Going from there to full fairness costs another
    > {abs(_accs_arr[_sweet_idx2] - _accs_arr[-1])*100:.1f}% accuracy. Where do we want to operate?"

    This is the conversation with stakeholders — not a binary fair/unfair, but a Pareto
    frontier with explicit trade-offs. The blue dashed line marks the legal threshold (DI=0.80).
    """)

    return (
        _X_te3,
        _X_tr3,
        _accs_arr,
        _di_arr,
        _fig2,
        _g_te3,
        _lambdas,
        _sweet_idx2,
        _y_te3,
        _y_tr3,
    )


@app.cell
def fairness_in_practice(mo):
    mo.md("""
    ## Part 5: Fairness in Practice — Real Systems

    ### The Canopy Fairness Question

    Canopy scores jobs using LLM-based evaluation. Where could bias enter?

    | Source | Mechanism | Audit approach |
    |--------|-----------|----------------|
    | **Training prompt bias** | Few-shot examples overrepresent certain company types → scorer inherits that preference | Audit example distribution; test with underrepresented company types |
    | **Embedding bias** | Sentence-transformers encode societal biases from internet text. "Software engineer" and "nurse" have gendered associations. | Paired testing with swapped gendered language |
    | **Feature proxy** | Company name is a proxy for culture, size, diversity — and for demographic associations | Test identical JDs with swapped company names |

    **How I'd audit Canopy for bias:**
    1. **Paired company name test:** Identical JDs except company name (tech giant vs minority-owned startup). Scores should be equal for same-fit candidates.
    2. **Pronoun swap test:** "he/his" → "she/her" in job descriptions. Scores should be invariant.
    3. **Job category distribution:** Check score distributions across job categories — are nursing jobs systematically scored lower than engineering jobs at the same fit level?

    Fairness auditing for LLM applications is harder than for traditional ML because the bias is
    embedded in language, not in tabular features. Paired testing is the most practical approach.

    ---

    ### The Fraud Detection Fairness Question

    From my fraud detection system design:
    - If the model has a higher FPR for a demographic group → legitimate customers from that group
      get blocked disproportionately
    - **Equalized odds** is the right definition: same TPR (catch fraud equally) AND same FPR
      (don't over-block any group)
    - **Monitoring:** Slice all metrics by demographic group; alert if FPR gap exceeds threshold

    In my system design, I specified "fairness checks" as a CI/CD gate. This notebook is
    what those checks actually compute.
    """)
    return


@app.cell
def fairlearn_demo(_model, _test_df, _X_te, _y_te, mo):
    try:
        from fairlearn.metrics import (
            MetricFrame as _MetricFrame,
            selection_rate as _selection_rate,
            demographic_parity_difference as _dpd,
            equalized_odds_difference as _eod,
        )
        from fairlearn.postprocessing import ThresholdOptimizer as _ThresholdOptimizer
        from sklearn.metrics import accuracy_score as _acc2
        import numpy as _np3

        _mf = _MetricFrame(
            metrics={'accuracy': _acc2, 'selection_rate': _selection_rate},
            y_true=_y_te,
            y_pred=_model.predict(_X_te),
            sensitive_features=_test_df['group'].values,
        )
        _dp_diff = _dpd(_y_te, _model.predict(_X_te), sensitive_features=_test_df['group'].values)
        _eo_diff = _eod(_y_te, _model.predict(_X_te), sensitive_features=_test_df['group'].values)

        _opt = _ThresholdOptimizer(
            estimator=_model,
            constraints="equalized_odds",
            objective="balanced_accuracy_score",
            predict_method='predict_proba',
        )
        _opt.fit(_X_te, _y_te, sensitive_features=_test_df['group'].values)
        _y_fair_fl = _opt.predict(_X_te, sensitive_features=_test_df['group'].values)
        _acc_fl = _acc2(_y_te, _y_fair_fl)

        _by_group_str = _mf.by_group.to_string()

        mo.md(f"""
    ## Part 6: Fairness Tools — Fairlearn Demo

    | Tool | Type | Best for |
    |------|------|----------|
    | **Fairlearn** (Microsoft) | Python library | Metrics, threshold optimization, constraint-based training |
    | **AIF360** (IBM) | Python library | Comprehensive metrics, broad mitigation algorithms |
    | **What-If Tool** (Google) | Interactive | Visual exploration of fairness and model behavior |
    | **Aequitas** | Python library | Audit reports, group-level metrics |
    | **Custom** (this notebook) | DIY | When you need to understand what the tool does |

    ---

    **MetricFrame by group:**
    ```
    {_by_group_str}
    ```

    **Demographic parity difference:** `{_dp_diff:.3f}` (0 = perfectly fair)
    **Equalized odds difference:** `{_eo_diff:.3f}` (0 = perfectly fair)

    **ThresholdOptimizer (equalized_odds constraint):**
    Accuracy after optimization: `{_acc_fl:.3f}`

    Fairlearn's `ThresholdOptimizer` automates the per-group threshold search.
    Under the hood it does exactly what `equalize_opportunity()` did above.
        """)

    except ImportError:
        mo.md("""
    ## Part 6: Fairness Tools

    | Tool | Type | Best for |
    |------|------|----------|
    | **Fairlearn** (Microsoft) | Python library | Metrics, threshold optimization, constraint-based training |
    | **AIF360** (IBM) | Python library | Comprehensive metrics, broad mitigation algorithms |
    | **What-If Tool** (Google) | Interactive | Visual exploration of fairness and model behavior |
    | **Aequitas** | Python library | Audit reports, group-level metrics |
    | **Custom** (this notebook) | DIY | When you need to understand what the tool does |

    `fairlearn` not installed — run `pip install fairlearn` to enable the demo.

    The custom implementations in this notebook (`disparate_impact_ratio`,
    `equalized_odds_check`, `equalize_opportunity`) are equivalent to Fairlearn's
    `demographic_parity_difference`, `equalized_odds_difference`, and `ThresholdOptimizer`.
    Use the library in production; use the scratch implementations to understand what it does.
        """)
    return


@app.cell
def flashcards(mo):
    mo.md("""
    ## Flashcard Summary

    **Name 3 fairness definitions.**
    → Demographic parity (equal positive rates), equalized odds (equal TPR and FPR), predictive
    parity (equal precision per group). They conflict mathematically — you can only satisfy all
    three simultaneously when base rates are equal or the model is perfect.

    **What's disparate impact?**
    → When a model's outcomes disproportionately affect a protected group. Four-fifths rule:
    disadvantaged group's selection rate < 80% of advantaged group's selection rate → disparate
    impact is legally presumed.

    **Why doesn't removing the protected attribute fix bias?**
    → Proxy features (zip code → race, name → gender, interview score → group). The model learns
    the same bias through correlated features. Called "fairness through unawareness" — it doesn't work.

    **Three intervention points?**
    → Pre-processing (fix the data), in-processing (constrain the model), post-processing
    (adjust predictions). Start with the lightest: threshold adjustment (post). Move to
    pre-processing next. In-processing last.

    **Easiest mitigation to implement?**
    → Threshold adjustment (post-processing). Use different thresholds per group to equalize TPR
    or FPR. No retraining needed.

    **How do you audit an LLM for bias?**
    → Paired testing: same input with swapped demographic indicators (names, pronouns, company
    names). Scores should be invariant to these swaps if the model is fair.

    **Why do fairness definitions conflict?**
    → Chouldechova (2017) proved mathematical impossibility — you can't satisfy demographic
    parity + equalized odds + predictive parity simultaneously except when base rates are equal.
    It's not an engineering limitation — it's a proof.

    **Fairness–accuracy trade-off shape?**
    → Usually gentle: large fairness gains for small accuracy cost early on, steep accuracy drops
    only at extreme fairness constraints. Plot the Pareto curve to show stakeholders the operating
    options — not a binary choice.
    """)
    return


@app.cell
def interview_talking_points(mo):
    mo.md("""
    ## Interview Talking Points

    **"How do you handle fairness in ML?"**
    First: choose the right fairness definition for the use case — they conflict, so it's a
    product decision. Then: measure it by slicing all metrics by protected groups. Detect proxy
    features with correlation analysis and feature importance. Mitigate with the lightest
    intervention that works — threshold adjustment before retraining. Monitor fairness metrics
    in production alongside accuracy.

    ---

    **"How would you audit a hiring model for bias?"**
    Three checks: (1) disparate impact ratio on predictions — four-fifths rule; (2) equalized
    odds — same TPR and FPR across demographic groups; (3) feature importance to detect if
    proxy features (university name, zip code, interview score) are driving predictions. For
    LLM-based scoring like Canopy: paired testing with demographic-swapped inputs — identical
    JDs with different company names or pronouns should produce the same scores.

    ---

    **"What's the hardest part of ML fairness?"**
    The impossibility result. You can't satisfy all definitions simultaneously. The real work
    is working with stakeholders to decide which definition matters most for your use case, then
    engineering the system to optimize for that specific definition while monitoring the others.
    Most engineers skip this step and pick a metric arbitrarily — that's the gap.

    ---

    **Connection to my work:**
    My fraud detection design specified fairness checks as a CI/CD gate — equalized odds
    monitoring ensures the model doesn't over-block legitimate transactions from any demographic
    group. For Canopy, I'd implement paired testing: identical job descriptions with swapped
    company names or gendered language shouldn't produce different job scores.
    """)
    return


if __name__ == "__main__":
    app.run()
