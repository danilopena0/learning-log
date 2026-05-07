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
    # Statistical Tests in Data Science & Machine Learning

    | Field | Value |
    |-------|-------|
    | Date  | 2026-05-06 |
    | Track | ML Theory |
    | Time  | 90 min |
    | Topics | Hypothesis Testing · t-tests · ANOVA · Chi-Square · Power Analysis · Effect Size · Non-Parametric Tests · Scalability |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Why Statistical Tests Matter in ML

    Statistical tests are the backbone of **A/B experiments, feature selection, model comparisons,
    and data validation**.  In ML workflows you use them to:

    - Decide if a new model is *significantly* better than a baseline — or just luckier on a holdout
    - Validate assumptions before applying algorithms (normality for linear regression, variance
      homogeneity for ANOVA)
    - Detect distribution shift between training and production data (KS test, chi-square)
    - Compare group behavior in user segmentation and fraud analysis

    The core vocabulary:

    | Term | Definition |
    |------|-----------|
    | **H₀ (null hypothesis)** | Default assumption — no effect, no difference |
    | **H₁ (alternative hypothesis)** | The claim you want to support |
    | **p-value** | Probability of observing results at least this extreme *if H₀ is true* |
    | **α (significance level)** | Threshold for rejecting H₀ (commonly 0.05) |
    | **β** | Probability of failing to detect a real effect (Type II error rate) |
    | **Power (1 − β)** | Probability of correctly detecting a real effect |
    | **Effect size** | Magnitude of the difference, independent of sample size |

    **Type I error (false positive):** reject H₀ when it is actually true — controlled by α.
    **Type II error (false negative):** fail to reject H₀ when H₁ is true — controlled by β.
    """)
    return


@app.cell
def error_types_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy import stats as _stats

    _fig, _ax = _plt.subplots(figsize=(11, 5))

    _x = _np.linspace(-5, 9, 600)
    _null = _stats.norm.pdf(_x, loc=0, scale=1.5)
    _alt  = _stats.norm.pdf(_x, loc=3.5, scale=1.5)
    _alpha = 0.05
    _crit = _stats.norm.ppf(1 - _alpha, loc=0, scale=1.5)

    _ax.plot(_x, _null, color="#3498DB", lw=2.5, label="H₀ distribution (no effect)")
    _ax.plot(_x, _alt,  color="#E74C3C", lw=2.5, label="H₁ distribution (real effect)")

    _x_t1 = _np.linspace(_crit, 9, 300)
    _ax.fill_between(_x_t1, _stats.norm.pdf(_x_t1, 0, 1.5),
                     alpha=0.35, color="#3498DB", label=f"Type I error (α={_alpha})")

    _x_t2 = _np.linspace(-5, _crit, 300)
    _ax.fill_between(_x_t2, _stats.norm.pdf(_x_t2, 3.5, 1.5),
                     alpha=0.35, color="#E74C3C", label="Type II error (β)")

    _x_pow = _np.linspace(_crit, 9, 300)
    _ax.fill_between(_x_pow, _stats.norm.pdf(_x_pow, 3.5, 1.5),
                     alpha=0.25, color="#2ECC71", label="Power (1−β)")

    _ax.axvline(_crit, color="#F39C12", lw=2, linestyle="--")
    _ax.text(_crit + 0.15, 0.27, f"Critical\nvalue\n({_crit:.2f})", fontsize=9,
             color="#F39C12", va="top")

    _ax.set_xlabel("Test Statistic", fontsize=12)
    _ax.set_ylabel("Density", fontsize=12)
    _ax.set_title("Type I & II Errors — Where They Live in the Distributions",
                  fontsize=13, fontweight="bold")
    _ax.legend(fontsize=9, loc="upper right")
    _ax.set_ylim(0, 0.32)
    _ax.grid(True, alpha=0.25)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## Power Analysis & Sample Size

    **Power** (1 − β) is the probability your test will detect a real effect if one exists.  The
    four quantities are locked together — fix any three and the fourth is determined:

    | Variable | Typical value | What it controls |
    |----------|--------------|-----------------|
    | **α** | 0.05 | False positive rate |
    | **Power** | 0.80 | True positive rate |
    | **Effect size (d)** | domain-dependent | How big the difference is |
    | **n** | *what you solve for* | Sample size per group |

    **Cohen's d** for two means:
    $$d = \\frac{\\mu_1 - \\mu_2}{\\sigma_{pooled}}$$

    | d value | Interpretation |
    |---------|---------------|
    | 0.2 | Small effect |
    | 0.5 | Medium effect |
    | 0.8 | Large effect |

    **Key insight for ML:** Small effects require *much* larger samples.  Doubling n raises power
    moderately; doubling effect size raises power dramatically.  Always run a power analysis
    *before* your experiment — not after you already have the data.
    """)
    return


@app.cell
def power_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy import stats as _stats

    def _compute_power(n, d, alpha=0.05):
        se = _np.sqrt(2 / n)
        crit = _stats.norm.ppf(1 - alpha / 2)
        ncp = d / se
        power = 1 - _stats.norm.cdf(crit - ncp) + _stats.norm.cdf(-crit - ncp)
        return power

    _n_range = _np.arange(5, 401, 1)
    _effect_sizes = {"Small (d=0.2)": 0.2, "Medium (d=0.5)": 0.5, "Large (d=0.8)": 0.8}
    _colors = {"Small (d=0.2)": "#E74C3C", "Medium (d=0.5)": "#F39C12", "Large (d=0.8)": "#2ECC71"}

    _fig, _ax = _plt.subplots(figsize=(10, 5.5))

    for _label, _d in _effect_sizes.items():
        _powers = [_compute_power(n, _d) for n in _n_range]
        _ax.plot(_n_range, _powers, lw=2.5, color=_colors[_label], label=_label)
        _n80 = next((n for n, p in zip(_n_range, _powers) if p >= 0.80), None)
        if _n80:
            _ax.axvline(_n80, color=_colors[_label], lw=1, linestyle=":")
            _ax.text(_n80 + 3, 0.12 + list(_effect_sizes.values()).index(_d) * 0.07,
                     f"n={_n80}", fontsize=9, color=_colors[_label])

    _ax.axhline(0.80, color="#7F8C8D", lw=1.5, linestyle="--", label="80% power threshold")
    _ax.set_xlabel("Sample Size per Group (n)", fontsize=12)
    _ax.set_ylabel("Statistical Power (1 − β)", fontsize=12)
    _ax.set_title("Power Curves: How n and Effect Size Drive Detection Ability",
                  fontsize=13, fontweight="bold")
    _ax.legend(fontsize=10)
    _ax.set_ylim(0, 1.05)
    _ax.grid(True, alpha=0.25)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## Parametric Tests

    Parametric tests assume your data follows a known distribution (usually normal).  They are
    generally **more powerful** when assumptions hold, and more sensitive to violations when they
    don't.

    ---

    ### t-tests (comparing means)

    | Variant | When to use |
    |---------|------------|
    | **One-sample t-test** | Compare sample mean to a known constant (e.g., is avg latency > 200ms?) |
    | **Independent two-sample t-test** | Compare means of two independent groups (A/B test) |
    | **Paired t-test** | Compare two measurements from the *same* units (before/after experiment) |

    **Assumptions:**
    1. Continuous data
    2. Approximately normal (or n > 30 by CLT)
    3. Homogeneity of variance (Welch's t-test relaxes this)
    4. Independent observations (except paired)

    **Pros:** Widely understood, high power when assumptions hold, easy to interpret.
    **Cons:** Sensitive to heavy-tailed distributions; breaks down for ordinal or skewed data with
    small n; assumes equality of variance (use Welch's variant to be safe).
    """)
    return


@app.cell
def ttest_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy import stats as _stats

    _rng = _np.random.default_rng(42)
    _n = 60
    _control    = _rng.normal(loc=0.0, scale=1.0, size=_n)
    _treatment  = _rng.normal(loc=0.4, scale=1.1, size=_n)

    _t_stat, _p_val = _stats.ttest_ind(_control, _treatment, equal_var=False)
    _d = (_treatment.mean() - _control.mean()) / _np.sqrt(
        ((_control.std(ddof=1)**2 + _treatment.std(ddof=1)**2) / 2))

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 5))

    _ax = _axes[0]
    _bins = _np.linspace(-3.5, 3.5, 25)
    _ax.hist(_control, bins=_bins, alpha=0.6, color="#3498DB", label=f"Control  μ={_control.mean():.2f}")
    _ax.hist(_treatment, bins=_bins, alpha=0.6, color="#E74C3C", label=f"Treatment  μ={_treatment.mean():.2f}")
    _ax.axvline(_control.mean(), color="#3498DB", lw=2, linestyle="--")
    _ax.axvline(_treatment.mean(), color="#E74C3C", lw=2, linestyle="--")
    _ax.set_title("Sample Distributions", fontsize=12, fontweight="bold")
    _ax.set_xlabel("Value"); _ax.set_ylabel("Count")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.25)
    _ax.text(0.05, 0.97,
             f"Welch's t = {_t_stat:.3f}\np = {_p_val:.4f}\nCohen's d = {_d:.3f}",
             transform=_ax.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    _ax2 = _axes[1]
    _n_range = _np.arange(10, 301, 5)
    _p_vals = []
    for _nn in _n_range:
        _c = _rng.normal(0.0, 1.0, _nn)
        _tr = _rng.normal(0.4, 1.1, _nn)
        _, _pv = _stats.ttest_ind(_c, _tr, equal_var=False)
        _p_vals.append(_pv)

    _ax2.plot(_n_range, _p_vals, color="#8E44AD", lw=2, alpha=0.8)
    _ax2.axhline(0.05, color="#E74C3C", lw=1.5, linestyle="--", label="α = 0.05")
    _ax2.set_xlabel("Sample Size per Group (n)", fontsize=11)
    _ax2.set_ylabel("p-value", fontsize=11)
    _ax2.set_title("p-value vs Sample Size (same d≈0.4 effect)", fontsize=12, fontweight="bold")
    _ax2.legend(fontsize=9)
    _ax2.grid(True, alpha=0.25)
    _fig.suptitle("Welch's Two-Sample t-test Demo", fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ### ANOVA (Analysis of Variance)

    ANOVA generalizes the two-sample t-test to **k groups** (k ≥ 2).  It tests whether *any*
    group mean differs from the others — it does not tell you *which* groups differ (use
    post-hoc tests: Tukey HSD, Bonferroni).

    **F-statistic:**
    $$F = \\frac{\\text{Between-group variance}}{\\text{Within-group variance}} = \\frac{MS_{between}}{MS_{within}}$$

    A large F means the group means are spread out relative to within-group noise.

    **One-way ANOVA assumptions:**
    - Normality within each group
    - Homogeneity of variance across groups (Levene's test)
    - Independent observations

    **Pros:** Single test controls familywise error rate across all groups.  Much better than
    running all pairwise t-tests (which inflates Type I error).
    **Cons:** Requires post-hoc testing to find *which* groups differ; sensitive to assumption
    violations with unequal group sizes; non-robust to outliers.
    """)
    return


@app.cell
def anova_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy import stats as _stats

    _rng = _np.random.default_rng(7)
    _groups = {
        "Model A": _rng.normal(0.820, 0.03, 40),
        "Model B": _rng.normal(0.835, 0.03, 40),
        "Model C": _rng.normal(0.810, 0.03, 40),
        "Model D": _rng.normal(0.850, 0.03, 40),
    }

    _f_stat, _p_val = _stats.f_oneway(*_groups.values())
    _grand_mean = _np.mean([v.mean() for v in _groups.values()])

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 5))

    _ax = _axes[0]
    _colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12"]
    for _i, (_label, _vals) in enumerate(_groups.items()):
        _ax.boxplot(_vals, positions=[_i], widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor=_colors[_i], alpha=0.5),
                    medianprops=dict(color="black", lw=2))
        _ax.scatter([_i] * len(_vals), _vals, alpha=0.4, s=15, color=_colors[_i])
    _ax.axhline(_grand_mean, color="black", lw=1.5, linestyle="--", label="Grand mean")
    _ax.set_xticks(range(4))
    _ax.set_xticklabels(_groups.keys(), fontsize=10)
    _ax.set_ylabel("Accuracy", fontsize=11)
    _ax.set_title("Model Accuracy Distributions", fontsize=12, fontweight="bold")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.25, axis="y")
    _ax.text(0.05, 0.05,
             f"F({len(_groups)-1},{sum(len(v) for v in _groups.values())-len(_groups)}) = {_f_stat:.2f}\np = {_p_val:.4f}",
             transform=_ax.transAxes, va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    _ax2 = _axes[1]
    _means = [v.mean() for v in _groups.values()]
    _sems  = [_stats.sem(v) for v in _groups.values()]
    _bars = _ax2.bar(_groups.keys(), _means, yerr=_sems, capsize=5,
                     color=_colors, alpha=0.7, edgecolor="white")
    _ax2.axhline(_grand_mean, color="black", lw=1.5, linestyle="--", label="Grand mean")
    _ax2.set_ylabel("Mean Accuracy ± SEM", fontsize=11)
    _ax2.set_title("Group Means with Error Bars", fontsize=12, fontweight="bold")
    _ax2.set_ylim(0.79, 0.875)
    _ax2.legend(fontsize=9)
    _ax2.grid(True, alpha=0.25, axis="y")
    _fig.suptitle("One-Way ANOVA: Comparing k=4 Model Accuracies", fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## Non-Parametric Tests

    Non-parametric tests make **no distributional assumptions** — they work on ranks rather than
    raw values.  The price is slightly lower power than their parametric equivalents *when the
    parametric assumptions actually hold*.  Use them when:

    - Data is ordinal (star ratings, Likert scales)
    - Distribution is heavily skewed or has extreme outliers
    - Sample size is small (n < 30) and normality cannot be verified
    - You want a robust fallback for production monitoring

    | Non-parametric test | Parametric equivalent | Tests |
    |--------------------|----------------------|-------|
    | Mann-Whitney U | Two-sample t-test | Whether one group tends to have higher values |
    | Wilcoxon signed-rank | Paired t-test | Paired differences from zero |
    | Kruskal-Wallis | One-way ANOVA | Whether any group has a different location |
    | Spearman correlation | Pearson correlation | Monotonic (not just linear) association |
    | Kolmogorov-Smirnov | — | Whether two samples come from the same distribution |

    ---

    ### Mann-Whitney U

    Ranks all observations from both groups together, then checks if one group's ranks are
    systematically higher.  Equivalent to asking: *"If I pick one observation from each group at
    random, what's the probability it comes from group A?"*  This probability (AUC interpretation)
    makes it directly interpretable as an effect size estimate.

    **Pros:** Robust to outliers; works on ordinal data; interprets naturally as a probability.
    **Cons:** Lower power than t-test under normality; doesn't compare means — tests stochastic
    dominance, which can be non-intuitive.
    """)
    return


@app.cell
def nonparam_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy import stats as _stats

    _rng = _np.random.default_rng(99)
    _skewed_a = _rng.exponential(scale=1.0, size=50)
    _skewed_b = _rng.exponential(scale=1.4, size=50)

    _t_stat, _t_p = _stats.ttest_ind(_skewed_a, _skewed_b, equal_var=False)
    _u_stat, _u_p = _stats.mannwhitneyu(_skewed_a, _skewed_b, alternative="two-sided")

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 5))

    _ax = _axes[0]
    _bins = _np.linspace(0, 7, 25)
    _ax.hist(_skewed_a, bins=_bins, alpha=0.6, color="#3498DB",
             label=f"Group A (exp, λ=1.0)  μ={_skewed_a.mean():.2f}")
    _ax.hist(_skewed_b, bins=_bins, alpha=0.6, color="#E74C3C",
             label=f"Group B (exp, λ=1.4)  μ={_skewed_b.mean():.2f}")
    _ax.set_title("Skewed (non-normal) Data", fontsize=12, fontweight="bold")
    _ax.set_xlabel("Value"); _ax.set_ylabel("Count")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.25)
    _ax.text(0.55, 0.95,
             f"t-test p = {_t_p:.4f}\nMann-Whitney p = {_u_p:.4f}",
             transform=_ax.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    _ax2 = _axes[1]
    _n_range = _np.arange(10, 201, 5)
    _t_powers, _u_powers = [], []
    for _nn in _n_range:
        _t_ps, _u_ps = [], []
        for _ in range(200):
            _a = _rng.exponential(1.0, _nn)
            _b = _rng.exponential(1.4, _nn)
            _, _tp = _stats.ttest_ind(_a, _b, equal_var=False)
            _, _up = _stats.mannwhitneyu(_a, _b, alternative="two-sided")
            _t_ps.append(_tp < 0.05)
            _u_ps.append(_up < 0.05)
        _t_powers.append(_np.mean(_t_ps))
        _u_powers.append(_np.mean(_u_ps))

    _ax2.plot(_n_range, _t_powers, color="#3498DB", lw=2, label="t-test power")
    _ax2.plot(_n_range, _u_powers, color="#E74C3C", lw=2, label="Mann-Whitney power")
    _ax2.axhline(0.80, color="#7F8C8D", lw=1.5, linestyle="--", alpha=0.7, label="80% threshold")
    _ax2.set_xlabel("Sample Size per Group", fontsize=11)
    _ax2.set_ylabel("Empirical Power", fontsize=11)
    _ax2.set_title("Power Comparison on Non-Normal Data\n(200 simulations per n)", fontsize=11, fontweight="bold")
    _ax2.legend(fontsize=9)
    _ax2.grid(True, alpha=0.25)
    _fig.suptitle("When Distributions are Skewed, Mann-Whitney Often Outperforms t-test",
                  fontsize=12, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## Correlation Tests

    Correlation quantifies the **strength and direction** of the relationship between two variables.

    ### Pearson Correlation

    $$r = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum(x_i-\\bar{x})^2 \\sum(y_i-\\bar{y})^2}}$$

    - Measures **linear** association; r ∈ [−1, 1]
    - Assumes both variables are continuous and normally distributed
    - Sensitive to outliers — one extreme point can swing r significantly

    ### Spearman Rank Correlation

    Applies Pearson on the *ranks* of the data instead of raw values.

    - Measures **monotonic** (not just linear) association
    - Robust to outliers and works on ordinal data
    - Use when the relationship is curved but still monotone (e.g., diminishing returns)

    **Rule of thumb:** start with Spearman; upgrade to Pearson only when linearity is confirmed
    and there are no outliers.

    | | Pearson | Spearman |
    |--|---------|---------|
    | Data type | Continuous | Continuous or ordinal |
    | Relationship | Linear | Monotonic |
    | Outlier sensitivity | High | Low |
    | Normality required | Yes | No |
    """)
    return


@app.cell
def correlation_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy import stats as _stats

    _rng = _np.random.default_rng(13)
    _n = 80

    _fig, _axes = _plt.subplots(1, 3, figsize=(15, 4.5))

    _scenarios = [
        ("Linear (no outliers)", _rng.normal(0,1,_n), None, False),
        ("Monotone (nonlinear)", _np.sort(_rng.uniform(0,5,_n)), None, True),
        ("Linear + outliers", _rng.normal(0,1,_n), True, False),
    ]

    for _i, (_title, _x, _add_outliers, _nonlinear) in enumerate(_scenarios):
        _ax = _axes[_i]
        if _nonlinear:
            _y = _np.log(_x + 1) + _rng.normal(0, 0.2, _n)
        else:
            _y = 0.7 * _x + _rng.normal(0, 0.5, _n)
        if _add_outliers:
            _x = _np.append(_x, [3.0, -3.0])
            _y = _np.append(_y, [-2.5, 2.5])

        _r_p, _p_p = _stats.pearsonr(_x, _y)
        _r_s, _p_s = _stats.spearmanr(_x, _y)

        _ax.scatter(_x, _y, alpha=0.5, s=25, color="#3498DB",
                    edgecolors="white", linewidths=0.3)
        _m, _b = _np.polyfit(_x, _y, 1)
        _xl = _np.linspace(_x.min(), _x.max(), 100)
        _ax.plot(_xl, _m*_xl+_b, color="#E74C3C", lw=2)
        _ax.set_title(_title, fontsize=11, fontweight="bold")
        _ax.set_xlabel("x"); _ax.set_ylabel("y")
        _ax.text(0.05, 0.95,
                 f"Pearson r = {_r_p:.3f} (p={_p_p:.3f})\nSpearman ρ = {_r_s:.3f} (p={_p_s:.3f})",
                 transform=_ax.transAxes, va="top", fontsize=8.5,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        _ax.grid(True, alpha=0.2)

    _fig.suptitle("Pearson vs Spearman: When They Agree and When They Diverge",
                  fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## Chi-Square Test (Categorical Data)

    The chi-square (χ²) test checks whether observed category frequencies deviate from expected
    frequencies.  It is the go-to test for **categorical variables** in ML pipelines.

    **Chi-square statistic:**
    $$\\chi^2 = \\sum \\frac{(O_i - E_i)^2}{E_i}$$

    ### Two main variants

    **1. Goodness-of-fit:** Does the observed distribution match a theoretical one?
    - E.g., is the class distribution in my training data uniform?

    **2. Test of independence (contingency table):** Are two categorical variables independent?
    - E.g., does click-through rate differ across device types?
    - This drives chi-square feature selection in sklearn (`SelectKBest(chi2, k=...)`)

    **Assumptions:**
    - Expected frequency in each cell ≥ 5 (use Fisher's exact test for smaller cells)
    - Observations are independent

    **Effect size — Cramér's V:**
    $$V = \\sqrt{\\frac{\\chi^2 / n}{\\min(r-1, c-1)}}$$

    V ∈ [0, 1]; 0.1 = small, 0.3 = medium, 0.5+ = large (varies by table size).

    **Pros:** No distributional assumptions on the variables; widely applicable.
    **Cons:** Cannot detect direction of association; sensitive to sample size like all tests.
    """)
    return


@app.cell
def chisquare_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy import stats as _stats

    _rng = _np.random.default_rng(21)
    _n = 500

    _mobile_click  = _rng.binomial(1, 0.12, _n)
    _desktop_click = _rng.binomial(1, 0.18, _n)
    _tablet_click  = _rng.binomial(1, 0.14, _n)

    _table = _np.array([
        [_mobile_click.sum(),  _n - _mobile_click.sum()],
        [_desktop_click.sum(), _n - _desktop_click.sum()],
        [_tablet_click.sum(),  _n - _tablet_click.sum()],
    ])

    _chi2, _p, _dof, _expected = _stats.chi2_contingency(_table)
    _cramers_v = _np.sqrt(_chi2 / (_table.sum() * (min(_table.shape) - 1)))

    _devices = ["Mobile", "Desktop", "Tablet"]
    _ctrs = [_mobile_click.mean(), _desktop_click.mean(), _tablet_click.mean()]
    _ci_half = [1.96 * _np.sqrt(c*(1-c)/_n) for c in _ctrs]

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 5))

    _ax = _axes[0]
    _bars = _ax.bar(_devices, _ctrs, color=["#3498DB", "#E74C3C", "#2ECC71"],
                    alpha=0.75, edgecolor="white", width=0.5, yerr=_ci_half, capsize=5)
    for _bar, _ctr in zip(_bars, _ctrs):
        _ax.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 0.005,
                 f"{_ctr:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    _ax.set_ylabel("Click-Through Rate", fontsize=11)
    _ax.set_title("CTR by Device Type", fontsize=12, fontweight="bold")
    _ax.set_ylim(0, 0.28)
    _ax.grid(True, alpha=0.25, axis="y")
    _ax.text(0.05, 0.97,
             f"χ²({_dof}) = {_chi2:.2f}\np = {_p:.4f}\nCramér's V = {_cramers_v:.3f}",
             transform=_ax.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    _ax2 = _axes[1]
    _obs = _table.flatten()
    _exp = _expected.flatten()
    _x_pos = _np.arange(len(_obs))
    _ax2.bar(_x_pos - 0.2, _obs, width=0.4, color="#3498DB", alpha=0.7, label="Observed")
    _ax2.bar(_x_pos + 0.2, _exp, width=0.4, color="#E74C3C", alpha=0.7, label="Expected (if independent)")
    _ax2.set_xticks(_x_pos)
    _ax2.set_xticklabels(["Mob-Click", "Mob-No", "Desk-Click", "Desk-No", "Tab-Click", "Tab-No"],
                          fontsize=8, rotation=20)
    _ax2.set_ylabel("Count", fontsize=11)
    _ax2.set_title("Observed vs Expected Frequencies", fontsize=12, fontweight="bold")
    _ax2.legend(fontsize=9)
    _ax2.grid(True, alpha=0.25, axis="y")
    _fig.suptitle("Chi-Square Test of Independence: CTR by Device", fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## The Scalability Problem: Large n Makes Everything Significant

    This is the **most important practical issue** with statistical tests in production ML.

    As n grows, any test's power approaches 1.0.  With millions of rows — which is routine in
    production — even effects that are completely meaningless in practice will produce
    p < 0.0001.  p-values become useless for decision-making at scale.

    **The fix: always report effect size alongside p-value.**

    A statistically significant result with d = 0.02 means your model improved by 0.3ms latency
    on average — probably not worth shipping.  A result with d = 0.5 is worth attention.

    | Situation | What to use |
    |-----------|------------|
    | n < 1,000 | p-value is primary; include effect size |
    | n ∈ 1K–100K | Both p-value and effect size; set a minimum effect threshold |
    | n > 100K | Effect size + confidence interval; treat p-value as a sanity check only |
    | n > 1M | Practical significance only — define a minimum detectable effect up front |

    **Practical significance test:**
    Before running the experiment, decide: *"What is the smallest effect that would be worth
    acting on?"*  This is your MDE (minimum detectable effect).  If the observed effect is below
    your MDE, do not ship the change regardless of p-value.
    """)
    return


@app.cell
def scalability_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy import stats as _stats

    _rng = _np.random.default_rng(55)
    _sample_sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    _tiny_d = 0.05

    _p_vals, _d_vals = [], []
    for _n in _sample_sizes:
        _a = _rng.normal(0.0, 1.0, _n)
        _b = _rng.normal(_tiny_d, 1.0, _n)
        _, _pv = _stats.ttest_ind(_a, _b)
        _d_obs = (_b.mean() - _a.mean()) / _np.sqrt(
            (_a.std(ddof=1)**2 + _b.std(ddof=1)**2) / 2)
        _p_vals.append(_pv)
        _d_vals.append(_d_obs)

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 5))

    _ax = _axes[0]
    _ax.semilogx(_sample_sizes, _p_vals, "o-", color="#E74C3C", lw=2.5, markersize=7,
                 label="p-value")
    _ax.axhline(0.05, color="#F39C12", lw=1.5, linestyle="--", label="α = 0.05")
    _ax.axhline(0.001, color="#8E44AD", lw=1.5, linestyle=":", label="α = 0.001")
    _ax.set_xlabel("Sample Size (log scale)", fontsize=11)
    _ax.set_ylabel("p-value", fontsize=11)
    _ax.set_title(f"p-value Collapse as n Grows\n(true effect d={_tiny_d} — practically tiny)",
                  fontsize=11, fontweight="bold")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.25)

    _ax2 = _axes[1]
    _n_fine = _np.logspace(2, 5.5, 200).astype(int)
    _d_effects = [0.05, 0.2, 0.5, 0.8]
    _colors2 = ["#E74C3C", "#F39C12", "#2ECC71", "#3498DB"]
    for _d, _c in zip(_d_effects, _colors2):
        _ps = []
        for _nn in _n_fine:
            _se = _np.sqrt(2 / _nn)
            _z = _d / _se
            _p = 2 * (1 - _stats.norm.cdf(abs(_z)))
            _ps.append(_p)
        _ax2.loglog(_n_fine, _ps, color=_c, lw=2, label=f"d = {_d}")
    _ax2.axhline(0.05, color="black", lw=1.5, linestyle="--", alpha=0.5, label="α = 0.05")
    _ax2.set_xlabel("Sample Size (log scale)", fontsize=11)
    _ax2.set_ylabel("p-value (log scale)", fontsize=11)
    _ax2.set_title("All Effect Sizes Hit Significance\nGiven Enough Data",
                   fontsize=11, fontweight="bold")
    _ax2.legend(fontsize=9)
    _ax2.grid(True, alpha=0.25)

    _fig.suptitle("The Scalability Problem: Why p-values Alone Fail at Production Scale",
                  fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## Multiple Comparisons & Familywise Error Rate

    Running k independent tests each at α = 0.05 does not give you an overall 5% false
    positive rate — it compounds:

    $$P(\\text{at least one false positive}) = 1 - (1-\\alpha)^k$$

    With 20 features tested independently: 1 − 0.95²⁰ ≈ **64%** chance of at least one
    spurious result.

    ### Corrections

    **Bonferroni correction:** use α' = α / k per test.
    - Simple and conservative; controls familywise error rate (FWER).
    - Becomes very strict for large k (genomics with 20K genes → α' = 2.5e-6).

    **Benjamini-Hochberg (BH / FDR):** controls the *false discovery rate* rather than FWER.
    - Allows more discoveries by accepting that some fraction of positives may be false.
    - Used in genomics, feature selection, and large-scale A/B experiments.
    - Less conservative than Bonferroni; preferred when you can tolerate a small fraction of
      false positives.

    | Method | Controls | Strictness | Typical use |
    |--------|----------|------------|------------|
    | No correction | Nothing | Least | Single test |
    | Bonferroni | FWER | Very strict | Few tests, no FP tolerated |
    | Holm-Bonferroni | FWER | Moderately strict | Sequential step-down |
    | Benjamini-Hochberg | FDR | Moderate | Many tests, some FP OK |
    """)
    return


@app.cell
def multiple_comp_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _k_range = _np.arange(1, 101)
    _fwer = 1 - (1 - 0.05)**_k_range
    _bonf = _np.minimum(0.05 * _k_range, 1.0)

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 4.5))

    _ax = _axes[0]
    _ax.plot(_k_range, _fwer, color="#E74C3C", lw=2.5, label="FWER (uncorrected)")
    _ax.axhline(0.05, color="#3498DB", lw=1.5, linestyle="--", label="Target α = 0.05")
    _ax.axvline(20, color="#F39C12", lw=1.5, linestyle=":", label="k=20 features")
    _ax.text(20.5, 0.3, f"FWER={1-(0.95**20):.2f}\nat k=20", fontsize=9, color="#F39C12")
    _ax.set_xlabel("Number of Tests (k)", fontsize=11)
    _ax.set_ylabel("P(at least one false positive)", fontsize=11)
    _ax.set_title("Familywise Error Rate Inflation", fontsize=12, fontweight="bold")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.25)

    _ax2 = _axes[1]
    _rng = _np.random.default_rng(42)
    _p_uncorrected = _np.sort(_rng.uniform(0, 1, 50))
    _alpha = 0.05
    _bh_thresh = (_np.arange(1, 51) / 50) * _alpha
    _bonf_thresh = _np.full(50, _alpha / 50)

    _ax2.plot(_np.arange(1, 51), _p_uncorrected, "o", color="#3498DB", markersize=5, alpha=0.7,
              label="Observed p-values (sorted)")
    _ax2.plot(_np.arange(1, 51), _bh_thresh, color="#E74C3C", lw=2,
              label=f"BH threshold (FDR={_alpha})")
    _ax2.axhline(_bonf_thresh[0], color="#2ECC71", lw=2, linestyle="--",
                 label=f"Bonferroni threshold ({_bonf_thresh[0]:.4f})")
    _ax2.set_xlabel("Rank of p-value", fontsize=11)
    _ax2.set_ylabel("p-value", fontsize=11)
    _ax2.set_title("BH Procedure: Step-Up Test\n(reject where p ≤ BH threshold)",
                   fontsize=11, fontweight="bold")
    _ax2.legend(fontsize=8)
    _ax2.grid(True, alpha=0.25)
    _fig.suptitle("Multiple Comparisons: Error Inflation and How to Control It",
                  fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ## Test Selection Guide

    ```
    Start here
    │
    ├─ Data type: CATEGORICAL?
    │   ├─ One variable (goodness-of-fit)    → Chi-square goodness-of-fit
    │   └─ Two variables (independence)      → Chi-square / Fisher's exact (n < 5 per cell)
    │
    └─ Data type: CONTINUOUS / ORDINAL?
        │
        ├─ Check normality (Shapiro-Wilk, QQ-plot) and sample size
        │
        ├─ NORMALLY DISTRIBUTED (or n > 30 per group)?
        │   ├─ One group vs constant          → One-sample t-test
        │   ├─ Two independent groups         → Welch's t-test (default) / Student's t
        │   ├─ Two paired groups              → Paired t-test
        │   ├─ 3+ groups, one factor          → One-way ANOVA + Tukey HSD
        │   ├─ 3+ groups, 2+ factors          → Two-way / n-way ANOVA
        │   └─ Linear relationship between    → Pearson correlation
        │      two variables?
        │
        └─ NOT NORMAL (or ordinal / heavy-tailed)?
            ├─ Two independent groups         → Mann-Whitney U
            ├─ Two paired groups              → Wilcoxon signed-rank
            ├─ 3+ groups                      → Kruskal-Wallis + Dunn post-hoc
            ├─ Monotone relationship?         → Spearman correlation
            └─ Two full distributions?        → Kolmogorov-Smirnov (KS) test
    ```

    **Distribution shift monitoring (MLOps):**

    | Scenario | Test |
    |---------|------|
    | Numerical feature distribution | KS test or Population Stability Index (PSI) |
    | Categorical feature distribution | Chi-square or Jensen-Shannon divergence |
    | Model output distribution | KS test on prediction scores |
    | Concept drift (labels) | Chi-square on confusion matrix counts |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Pros, Cons & When to Use

    | Test | Best for | Pros | Cons / Watch out |
    |------|----------|------|-----------------|
    | **One-sample t-test** | Compare sample mean to benchmark | Simple; well-understood | Requires normality; useless without effect size |
    | **Welch's t-test** | Two-group comparison | Robust to unequal variance; high power under normality | Sensitive to non-normality with small n |
    | **Paired t-test** | Before/after, same subjects | Removes between-subject noise; more powerful | Only valid when pairing is meaningful |
    | **One-way ANOVA** | k ≥ 3 groups, one factor | Controls FWER vs. multiple t-tests | Assumes normality + homoscedasticity; needs post-hoc |
    | **Mann-Whitney U** | Two groups, non-normal | Robust to outliers; works on ordinal | Doesn't test means; slight power loss under normality |
    | **Kruskal-Wallis** | k ≥ 3 groups, non-normal | No distributional assumptions | Less powerful; needs post-hoc (Dunn's) |
    | **Pearson r** | Linear correlation | Intuitive magnitude; widely reported | Heavily influenced by outliers; assumes linearity |
    | **Spearman ρ** | Monotone correlation | Robust; works on ordinal | Misses curvature that isn't monotone |
    | **Chi-square** | Categorical independence | No distributional assumption | Needs expected freq ≥ 5; can't show direction |
    | **Fisher's exact** | Small categorical tables | Exact p-value regardless of n | Computationally expensive for large tables |
    | **KS test** | Full distribution comparison | Sensitive to any distributional difference | Hard to interpret for large n (everything differs) |
    | **Shapiro-Wilk** | Testing normality | Most powerful normality test for n < 2000 | Very sensitive at large n — fails on trivially non-normal data |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Flashcard-Style Summary

    | Question | Answer |
    |----------|--------|
    | What does a p-value represent? | The probability of observing a result at least this extreme *if the null hypothesis is true* — not the probability that H₀ is true. |
    | What is statistical power? | 1 − β: the probability that the test detects a real effect when one exists. Aim for 0.80 as a minimum. |
    | What are the four factors that determine sample size? | Effect size (d), significance level (α), desired power (1−β), and the test type. Fix three → solve for the fourth. |
    | Why is effect size essential alongside p-value? | A large n will make even trivially small differences statistically significant. Effect size tells you whether the difference *matters* in practice. |
    | When should you use a non-parametric test? | When data is ordinal, heavily skewed, has extreme outliers, or n is small and normality cannot be verified. |
    | What does ANOVA actually test? | Whether the between-group variance is significantly larger than within-group variance. A significant result means *at least one* group differs — not which ones. |
    | How does the Bonferroni correction work? | Divides α by the number of tests (α' = α/k) to control the probability of any false positive across all tests. |
    | When is BH correction preferred over Bonferroni? | When running many tests (feature selection, genomics) and you can tolerate a controlled fraction of false discoveries instead of zero. |
    | What is Cohen's d? | (μ₁ − μ₂) / σ_pooled. d ≈ 0.2 small, 0.5 medium, 0.8 large. |
    | What is the scalability problem with statistical tests? | At large n (millions of rows), all tests reach significance — even for effects too small to matter. Rely on effect sizes and minimum detectable effects instead. |
    | When should you use Fisher's exact test vs chi-square? | Fisher's exact when any expected cell frequency < 5; chi-square otherwise. |
    | What is Cramér's V? | Effect size for chi-square: √(χ²/n·min(r−1,c−1)). Ranges 0–1; 0.1 small, 0.3 medium, 0.5+ large. |
    | What test detects distribution shift in production? | KS test for continuous features, chi-square for categorical. Also use PSI (Population Stability Index) as a monitoring heuristic. |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    ### "How do you choose between a t-test and a Mann-Whitney U test?"

    > "First I check the assumptions: is the data continuous, is the sample large enough to invoke
    > the CLT (n > 30 per group), and are there extreme outliers?  If normality holds, Welch's
    > t-test gives higher power.  If the data is ordinal, heavily skewed, or small-n non-normal,
    > I use Mann-Whitney — it tests stochastic dominance rather than mean difference, which is
    > often the right question anyway.  In practice I default to Welch's (not Student's) because
    > it handles unequal variances without assuming equal n."

    ---

    ### "You run an A/B test on a product with 10 million users and get p < 0.0001. Should you ship?"

    > "Not based on p-value alone.  With 10M users, statistical power is effectively 1 for any
    > non-zero effect — the test will flag differences far too small to matter.  I'd look at the
    > effect size: Cohen's d or the absolute lift in the metric.  Before the experiment I'd have
    > defined a minimum detectable effect — the smallest lift worth shipping for.  If the observed
    > effect is below that threshold, I don't ship regardless of the p-value.  I'd also report
    > confidence intervals, not just the point estimate."

    ---

    ### "How do you handle multiple comparisons in feature selection?"

    > "Running k independent chi-square or F-tests at α = 0.05 gives up to a 64% chance of
    > at least one false positive across 20 features.  I use one of two strategies:  Bonferroni
    > (α / k) if I want strict FWER control and there are few tests; Benjamini-Hochberg if I'm
    > scanning hundreds of features — it controls the false discovery rate rather than FWER, so
    > it's less conservative and captures more real signals.  In sklearn this is built into
    > `SelectFpr` (FWER) and `SelectFdr` (FDR)."

    ---

    ### "Walk me through designing a sample size calculation for an A/B test."

    > "I need four inputs: (1) the metric's baseline and its standard deviation; (2) the minimum
    > detectable effect — the smallest lift the business cares about; (3) the significance level
    > (α, usually 0.05); (4) the desired power (1−β, typically 0.80).  From these I compute
    > Cohen's d = MDE / σ, then use a standard power formula (or `statsmodels.stats.power`) to
    > get n per group.  If the required n is too large for the experiment duration, I can raise
    > the MDE threshold — but that's a business decision, not a statistical one."

    ---

    ### "What is the KS test and when do you use it in ML?"

    > "The Kolmogorov-Smirnov test compares two empirical CDFs.  The test statistic is the maximum
    > absolute difference between them.  In ML I use it for distribution shift monitoring: I compare
    > the training feature distribution against the production distribution.  If the KS statistic
    > exceeds a threshold, it signals that the model may be operating out-of-distribution and
    > needs retraining.  The caveat at large n is the same as with all tests — I set a practical
    > threshold on the KS statistic (e.g., > 0.1) rather than relying on p-value alone."
    """)
    return


if __name__ == "__main__":
    app.run()
