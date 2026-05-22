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
    # Sampling, A/B Testing & Causal Inference

    | Field | Value |
    |-------|-------|
    | Date  | 2026-05-21 |
    | Track | ML Theory |
    | Time  | 60 min |
    | Topics | Sampling · CLT · Hypothesis Testing · A/B Test Design · Causal Inference |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Why ML Engineers Need This

    You trained a new model. Offline metrics say it's better. Do you ship it?

    **NO.** Offline metrics don't reliably predict online impact. You need an A/B test.

    The questions this notebook answers:
    - How many users do I need in my experiment?
    - How long should I run it?
    - How do I know the result is real and not noise?
    - When can I trust an observational analysis vs when do I need a proper experiment?

    > *"Every model launch at a serious company goes through an A/B test. If you can design one correctly,
    you're immediately useful on the team."*
    """)
    return


# ─── PART 1: SAMPLING ────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Part 1: Sampling — The Foundation

    ### Why sampling matters

    You can't test your model on ALL users. You test on a **sample** and infer about the population.
    If the sample is biased, your conclusions are wrong — no amount of statistical sophistication
    fixes bad sampling.

    | Concept | Definition |
    |---------|-----------|
    | **Population** | All users/requests your model will serve |
    | **Sample** | The subset you actually measure |
    | **Sampling bias** | When your sample systematically differs from the population |
    | **Random sampling** | Every member of the population has equal probability of being selected |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Common sampling biases in ML

    - **Survivorship bias** — only analyzing users who stayed (ignoring churned users who left
      *because of* the model)
    - **Selection bias in A/B tests** — treatment group has different user composition than control
      (broken randomization)
    - **Position bias in recommendations** — users interact more with top-ranked items regardless of
      quality. High-ranked items appear better.
    - **Feedback loops** — model recommends popular items → they get more engagement → they appear even
      more popular. Popularity reinforces itself.
    - **Temporal bias** — running the test during a holiday week when behavior is abnormal

    > *In my Canopy gap analysis, I flagged that job scoring has no eval against actual outcomes
    (did the user apply? get an interview?). Without that, I'm measuring model confidence, not
    real-world impact — a form of selection bias.*
    """)
    return


@app.cell
def clt_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _rng = _np.random.default_rng(42)

    # Skewed population: exponential (like session lengths / purchase amounts)
    _population = _rng.exponential(scale=2.0, size=100_000)
    _pop_mean = _population.mean()
    _pop_std = _population.std()

    _n_trials = 1000
    _sample_sizes = [5, 30, 100]
    _colors = ["#E74C3C", "#F39C12", "#2ECC71"]

    _fig, _axes = _plt.subplots(1, 4, figsize=(14, 4))

    # Population distribution
    _axes[0].hist(_population, bins=80, color="#95A5A6", edgecolor="none", density=True)
    _axes[0].set_title("Population\n(Exponential — skewed)", fontsize=11, fontweight="bold")
    _axes[0].set_xlabel("Value")
    _axes[0].set_ylabel("Density")
    _axes[0].axvline(_pop_mean, color="black", lw=2, linestyle="--", label=f"Mean={_pop_mean:.2f}")
    _axes[0].legend(fontsize=9)

    # Sample mean distributions for each n
    for _ax, _n, _color in zip(_axes[1:], _sample_sizes, _colors):
        _sample_means = [_rng.choice(_population, size=_n).mean() for _ in range(_n_trials)]
        _sample_means = _np.array(_sample_means)
        _theoretical_std = _pop_std / _np.sqrt(_n)

        _ax.hist(_sample_means, bins=50, color=_color, edgecolor="none", density=True, alpha=0.85)
        _ax.axvline(_pop_mean, color="black", lw=2, linestyle="--")

        # Overlay theoretical normal
        _x = _np.linspace(_sample_means.min(), _sample_means.max(), 200)
        _pdf = (1 / (_theoretical_std * _np.sqrt(2 * _np.pi))) * \
               _np.exp(-0.5 * ((_x - _pop_mean) / _theoretical_std) ** 2)
        _ax.plot(_x, _pdf, "k-", lw=2)

        _ax.set_title(f"Sample Means (n={_n})\nstd = {_sample_means.std():.3f} ≈ σ/√n={_theoretical_std:.3f}",
                      fontsize=10, fontweight="bold")
        _ax.set_xlabel("Sample Mean")

    _fig.suptitle(
        "Central Limit Theorem: Averages become Normal regardless of population shape",
        fontsize=13, fontweight="bold", y=1.02
    )
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **Key insight:** Even though individual session lengths are exponentially distributed (highly skewed),
    the *average* session length computed from samples of n=30+ is well-approximated by a normal
    distribution. Standard error = σ/√n — larger samples → tighter, more reliable estimates.

    This is why hypothesis tests work: we don't need individual data points to be normal,
    we need *averages* to be normal — and CLT guarantees that.
    """)
    return


# ─── PART 2: HYPOTHESIS TESTING ──────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Part 2: Hypothesis Testing — The Mechanics

    ### The framework

    | Term | Definition |
    |------|-----------|
    | **Null hypothesis (H₀)** | New model has NO effect — any difference is random noise |
    | **Alternative hypothesis (H₁)** | New model IS different (better or worse) |
    | **Test statistic** | A number summarizing how different the two groups are (z-score, t-statistic) |
    | **p-value** | Probability of seeing a result this extreme IF H₀ is true. Low p = unlikely to be noise. |
    | **Significance level (α)** | Threshold for rejecting H₀. Typically α = 0.05. |

    **Decision rule:** if p < α → reject H₀ → "the effect is statistically significant"

    > *p < 0.05 does NOT mean "the model is 95% likely to be better." It means "if there were no
    difference, we'd see this result less than 5% of the time." The distinction matters.*
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Two types of errors

    | | H₀ is actually **true** (no effect) | H₀ is actually **false** (real effect) |
    |--|---|---|
    | **Reject H₀** | **Type I Error (α)**: false positive. Think there's an effect but there isn't. | **Correct**: true positive. |
    | **Fail to reject H₀** | **Correct**: true negative. | **Type II Error (β)**: false negative. Real effect exists but we missed it. |

    - **Power = 1 − β**: probability of detecting a real effect. Typically target **80%**.
    - The trade-off: lower α → fewer false positives but more false negatives (less power).
      You can't minimize both without more data.

    > *In ML terms: precision (low α) vs recall (high power). Same trade-off from the evaluation
    metrics notebook.*
    """)
    return


@app.cell
def hypothesis_simulation():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy import stats as _stats

    _rng = _np.random.default_rng(0)
    _n_simulations = 10_000
    _n_per_group = 1000
    _alpha = 0.05

    # Scenario A: H₀ is true — no real effect
    _pvals_null = []
    for _ in range(_n_simulations):
        _a = _rng.normal(0, 1, _n_per_group)
        _b = _rng.normal(0, 1, _n_per_group)
        _, _p = _stats.ttest_ind(_a, _b)
        _pvals_null.append(_p)
    _pvals_null = _np.array(_pvals_null)
    _fp_rate = (_pvals_null < _alpha).mean()

    # Scenario B: H₀ is false — real 5% effect
    _pvals_real = []
    for _ in range(_n_simulations):
        _a = _rng.normal(0, 1, _n_per_group)
        _b = _rng.normal(0.05, 1, _n_per_group)
        _, _p = _stats.ttest_ind(_a, _b)
        _pvals_real.append(_p)
    _pvals_real = _np.array(_pvals_real)
    _power_1000 = (_pvals_real < _alpha).mean()

    # Power vs sample size
    _sample_sizes = _np.array([50, 100, 200, 500, 1000, 2000, 5000])
    _power_curve = []
    for _n in _sample_sizes:
        _pw = []
        for _ in range(2000):
            _a = _rng.normal(0, 1, _n)
            _b = _rng.normal(0.05, 1, _n)
            _, _p = _stats.ttest_ind(_a, _b)
            _pw.append(_p < _alpha)
        _power_curve.append(_np.mean(_pw))
    _power_curve = _np.array(_power_curve)

    _fig, _axes = _plt.subplots(1, 3, figsize=(15, 4))

    # p-value histogram under H₀ (should be uniform)
    _axes[0].hist(_pvals_null, bins=40, color="#95A5A6", edgecolor="none", density=True)
    _axes[0].axvline(_alpha, color="#E74C3C", lw=2, linestyle="--", label=f"α = {_alpha}")
    _axes[0].axvspan(0, _alpha, alpha=0.15, color="#E74C3C")
    _axes[0].text(0.025, 12, f"False positive\nrate = {_fp_rate:.1%}", fontsize=9,
                  ha="center", color="#E74C3C", fontweight="bold")
    _axes[0].set_title("p-values when H₀ is TRUE\n(uniform — all values equally likely)", fontsize=10, fontweight="bold")
    _axes[0].set_xlabel("p-value")
    _axes[0].set_ylabel("Density")
    _axes[0].legend(fontsize=9)

    # p-value histogram when real effect exists
    _axes[1].hist(_pvals_real, bins=40, color="#3498DB", edgecolor="none", density=True)
    _axes[1].axvline(_alpha, color="#E74C3C", lw=2, linestyle="--", label=f"α = {_alpha}")
    _axes[1].axvspan(0, _alpha, alpha=0.15, color="#2ECC71")
    _axes[1].text(0.025, 20, f"Power\n= {_power_1000:.1%}", fontsize=9,
                  ha="center", color="#27AE60", fontweight="bold")
    _axes[1].set_title(f"p-values when real effect exists\n(n={_n_per_group} per group, 5% effect)",
                       fontsize=10, fontweight="bold")
    _axes[1].set_xlabel("p-value")
    _axes[1].legend(fontsize=9)

    # Power vs sample size
    _axes[2].plot(_sample_sizes, _power_curve, "o-", color="#2C3E50", lw=2.5, ms=7)
    _axes[2].axhline(0.80, color="#E74C3C", lw=2, linestyle="--", label="80% power target")
    _axes[2].fill_between(_sample_sizes, _power_curve, 0.80,
                          where=(_power_curve < 0.80), alpha=0.15, color="#E74C3C",
                          label="Underpowered")
    _axes[2].fill_between(_sample_sizes, _power_curve, 0.80,
                          where=(_power_curve >= 0.80), alpha=0.15, color="#2ECC71",
                          label="Adequate power")
    _axes[2].set_xscale("log")
    _axes[2].set_title("Power vs Sample Size\n(detecting a 5% effect, α=0.05)", fontsize=10, fontweight="bold")
    _axes[2].set_xlabel("Sample size per group (log scale)")
    _axes[2].set_ylabel("Power (1 − β)")
    _axes[2].set_ylim(0, 1.05)
    _axes[2].legend(fontsize=9)
    _axes[2].grid(True, alpha=0.3)

    _fig.tight_layout()
    return _fig, _power_1000


@app.cell
def _(mo, power_1000):
    mo.md(f"""
    **Simulation results** (10,000 experiments each):
    - Under H₀ (no effect): ~5% of tests return p < 0.05 — exactly the false positive rate α.
      If you run 20 A/B tests and there's truly no effect in any of them, you'll still get
      ~1 "significant" result by chance. This is the **multiple testing problem.**
    - With a real 5% effect, n=1,000 per group: power = **{power_1000:.1%}**.
    - The power curve shows why sample size planning matters: small n → low power → you'll
      miss real improvements.
    """)
    return


# ─── PART 3: A/B TEST DESIGN ─────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Part 3: A/B Test Design — The Practical Guide
    """)
    return


@app.cell
def sample_size_calc():
    import numpy as _np
    from scipy import stats as _stats

    def required_sample_size(
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """Minimum samples per group for a two-sample proportion test."""
        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect
        p_bar = (p1 + p2) / 2
        z_alpha = _stats.norm.ppf(1 - alpha / 2)
        z_beta = _stats.norm.ppf(power)
        n = ((z_alpha * _np.sqrt(2 * p_bar * (1 - p_bar)) +
              z_beta * _np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2
        return int(_np.ceil(n))

    # Canopy example: 10% application rate, detect 1% absolute improvement
    _n = required_sample_size(0.10, 0.01)
    print(f"Baseline: 10% application rate")
    print(f"MDE: +1% absolute (10% → 11%)")
    print(f"Required: {_n:,} users per group")
    print(f"Total users: {2*_n:,}")
    print(f"At 1,000 users/day → {2*_n/1000:.0f} days")
    print()

    # Sensitivity table: MDE vs required n
    _baselines = [0.05, 0.10, 0.20]
    _mdes = [0.005, 0.01, 0.02, 0.05]
    print("Sample size per group (α=0.05, power=80%):")
    print(f"{'MDE →':>12}", end="")
    for _mde in _mdes:
        print(f"  +{_mde*100:.1f}%  ", end="")
    print()
    for _base in _baselines:
        print(f"base={_base*100:.0f}%   ", end="")
        for _mde in _mdes:
            _n_cell = required_sample_size(_base, _mde)
            print(f" {_n_cell:>6,}  ", end="")
        print()

    return required_sample_size,


@app.cell
def sample_size_viz(required_sample_size):
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _mdes = _np.linspace(0.002, 0.10, 200)
    _configs = [
        (0.05, "#E74C3C", "Baseline 5%"),
        (0.10, "#3498DB", "Baseline 10%"),
        (0.20, "#2ECC71", "Baseline 20%"),
    ]

    _fig, _axes = _plt.subplots(1, 2, figsize=(13, 5))

    for _base, _color, _label in _configs:
        _ns = [required_sample_size(_base, _mde) for _mde in _mdes]
        _axes[0].plot(_mdes * 100, _ns, color=_color, lw=2.5, label=_label)

    _axes[0].axhline(10_000, color="#95A5A6", lw=1.5, linestyle=":", label="10K threshold")
    _axes[0].set_xlabel("Minimum Detectable Effect (% absolute)", fontsize=11)
    _axes[0].set_ylabel("Required n per group", fontsize=11)
    _axes[0].set_title("Sample Size vs MDE\n(Smaller effect = exponentially more data needed)",
                        fontsize=11, fontweight="bold")
    _axes[0].legend(fontsize=10)
    _axes[0].set_ylim(0, 100_000)
    _axes[0].grid(True, alpha=0.3)

    # Power vs sample size for different baselines
    _ns_range = _np.logspace(2, 5, 100).astype(int)
    from scipy import stats as _stats

    def _approx_power(baseline, mde, n, alpha=0.05):
        p1, p2 = baseline, baseline + mde
        p_bar = (p1 + p2) / 2
        se_null = _np.sqrt(2 * p_bar * (1 - p_bar) / n)
        se_alt = _np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
        z_alpha = _stats.norm.ppf(1 - alpha / 2)
        z = (mde - z_alpha * se_null) / se_alt
        return _stats.norm.cdf(z)

    _powers = [_approx_power(0.10, 0.01, _n) for _n in _ns_range]
    _axes[1].plot(_ns_range, _powers, color="#2C3E50", lw=2.5)
    _axes[1].axhline(0.80, color="#E74C3C", lw=2, linestyle="--", label="80% power target")
    _axes[1].fill_between(_ns_range, _powers, 0.80,
                          where=(_np.array(_powers) >= 0.80), alpha=0.12, color="#2ECC71")
    _n_80 = required_sample_size(0.10, 0.01, power=0.80)
    _axes[1].axvline(_n_80, color="#27AE60", lw=2, linestyle="--",
                     label=f"n={_n_80:,} → 80% power")
    _axes[1].set_xscale("log")
    _axes[1].set_xlabel("Sample size per group (log scale)", fontsize=11)
    _axes[1].set_ylabel("Power (1 − β)", fontsize=11)
    _axes[1].set_title("Power Curve: 10% baseline, 1% MDE\n(The key planning visualization)",
                        fontsize=11, fontweight="bold")
    _axes[1].legend(fontsize=10)
    _axes[1].set_ylim(0, 1.05)
    _axes[1].grid(True, alpha=0.3)

    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    > *This is the FIRST thing to compute before any experiment. If you need 100K users and
    you have 1K/day, your test takes 100 days. Better to know upfront than halfway through.*

    ### Duration planning

    - **Minimum duration**: enough days to reach required sample size
    - **Also**: at least **1 full week** (capture day-of-week effects), ideally **2 weeks**
    - **Avoid**: holidays, major product launches, any period where behavior is abnormal
    - **Traffic allocation:**
      - 50/50 split: maximum power, fastest results
      - 90/10 split: less risk (only 10% see new model), but 5× slower to reach significance
      - **Recommendation**: start at 95/5 for safety (canary), expand to 50/50 for the actual test

    > *This connects directly to my MLOps notebook: canary deploy (95/5 for safety) → A/B test
    (50/50 for measurement) → full rollout.*
    """)
    return


@app.cell
def peeking_simulation():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy import stats as _stats

    _rng = _np.random.default_rng(7)
    _n_days = 21
    _users_per_day = 200
    _true_effect = 0.05  # real 5% lift

    _rng2 = _np.random.default_rng(99)  # second run with no effect

    def _simulate_peeking(rng, effect, n_days, users_per_day):
        _a_all, _b_all = [], []
        _pvals = []
        for _day in range(1, n_days + 1):
            _a_all.extend(rng.normal(0, 1, users_per_day).tolist())
            _b_all.extend(rng.normal(effect, 1, users_per_day).tolist())
            if _day < 3:
                _pvals.append(None)
                continue
            _, _p = _stats.ttest_ind(_a_all, _b_all)
            _pvals.append(_p)
        return _pvals

    _pvals_real = _simulate_peeking(_rng, _true_effect, _n_days, _users_per_day)
    _pvals_null = _simulate_peeking(_rng2, 0.0, _n_days, _users_per_day)

    _fig, _axes = _plt.subplots(1, 2, figsize=(13, 5))
    _days = list(range(1, _n_days + 1))

    for _ax, _pvals, _title, _effect_label in [
        (_axes[0], _pvals_real, "Real 5% effect exists", "5% effect"),
        (_axes[1], _pvals_null, "No real effect (H₀ true)", "No effect"),
    ]:
        _valid_days = [d for d, p in zip(_days, _pvals) if p is not None]
        _valid_pvals = [p for p in _pvals if p is not None]

        _ax.plot(_valid_days, _valid_pvals, "o-", color="#2C3E50", lw=2, ms=5, label="p-value")
        _ax.axhline(0.05, color="#E74C3C", lw=2, linestyle="--", label="α = 0.05")
        _ax.axvline(14, color="#27AE60", lw=2, linestyle=":", label="Pre-planned end (day 14)")

        # Shade the "danger zone" where peeking might trigger early stop
        _early_sig = [d for d, p in zip(_valid_days, _valid_pvals) if p < 0.05 and d < 14]
        if _early_sig:
            _ax.axvspan(min(_early_sig), max(_early_sig), alpha=0.12, color="#E74C3C",
                        label="Peeking danger zone")

        _ax.set_xlabel("Day of experiment", fontsize=11)
        _ax.set_ylabel("p-value", fontsize=11)
        _ax.set_title(f"The Peeking Problem\n{_title}", fontsize=11, fontweight="bold")
        _ax.set_ylim(-0.02, 1.02)
        _ax.legend(fontsize=9)
        _ax.grid(True, alpha=0.3)

    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **What the peeking plots show:** p-values fluctuate wildly in the first few days even when
    a real effect exists. Stopping the moment you see p < 0.05 inflates the actual false positive
    rate far above your chosen α = 0.05.

    **Fixes:**
    - **Pre-commit to duration.** Don't look. Simplest and most common.
    - **Sequential testing (O'Brien-Fleming)**: pre-planned interim analyses with adjusted thresholds.
    - **Always-valid p-values**: methods that remain valid regardless of when you look (`confseq` library).

    > *Peeking is the #1 way A/B tests go wrong in practice. Pre-commit to your test duration
    and DO NOT stop early because the numbers look good.*
    """)
    return


# ─── PART 4: COMMON PITFALLS ─────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Part 4: Common A/B Testing Pitfalls for ML

    ### Pitfall 1: Novelty / Primacy Effects

    - **Novelty**: users engage more with something *new* just because it's different — fades over time
    - **Primacy**: users engage more with the *old* because it's familiar — also fades
    - Both effects mean short tests over/underestimate the true effect
    - **Fix**: run for at least 2 weeks. Analyze first-week vs second-week metrics separately.
      If effect fades, it's novelty.

    ### Pitfall 2: Interference / Network Effects

    - In social platforms: user A (treatment) recommends an item to user B (control) → B's behavior
      is influenced by A's treatment. Violates the independence assumption of A/B testing.
    - **Fix**: cluster randomization — randomize entire friend groups or geographic regions, not individuals.

    > *Most job search platforms don't have strong network effects, so standard user-level
    randomization works for Canopy.*
    """)
    return


@app.cell
def simpsons_paradox_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    # Simpson's Paradox: Model B looks better overall but worse in every segment
    # High-converting segment (power users): 80% conversion rate
    # Low-converting segment (new users): 20% conversion rate
    # Model B gets disproportionately assigned to power users → appears better overall

    _segments = ["Power users\n(high converters)", "New users\n(low converters)"]

    # Model A: balanced 50/50 across segments
    _a_power_n, _a_power_rate = 100, 0.80
    _a_new_n, _a_new_rate = 100, 0.20
    _a_overall = (_a_power_n * _a_power_rate + _a_new_n * _a_new_rate) / (_a_power_n + _a_new_n)

    # Model B: 80% power users (imbalanced randomization)
    _b_power_n, _b_power_rate = 160, 0.78   # slightly WORSE in this segment
    _b_new_n, _b_new_rate = 40, 0.18         # slightly WORSE in this segment too
    _b_overall = (_b_power_n * _b_power_rate + _b_new_n * _b_new_rate) / (_b_power_n + _b_new_n)

    _fig, _axes = _plt.subplots(1, 2, figsize=(13, 5))

    # Segment-level comparison
    _x = _np.arange(len(_segments))
    _width = 0.35
    _bars_a = _axes[0].bar(_x - _width/2, [_a_power_rate, _a_new_rate], _width,
                           color="#3498DB", alpha=0.85, label="Model A")
    _bars_b = _axes[0].bar(_x + _width/2, [_b_power_rate, _b_new_rate], _width,
                           color="#E74C3C", alpha=0.85, label="Model B")
    _axes[0].set_xticks(_x)
    _axes[0].set_xticklabels(_segments, fontsize=10)
    _axes[0].set_ylabel("Conversion Rate", fontsize=11)
    _axes[0].set_title("Within Each Segment:\nModel A is BETTER", fontsize=11, fontweight="bold")
    _axes[0].legend(fontsize=10)
    _axes[0].set_ylim(0, 1.0)
    for _bar in _bars_a:
        _axes[0].text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 0.01,
                      f"{_bar.get_height():.0%}", ha="center", fontsize=10, color="#2C3E50")
    for _bar in _bars_b:
        _axes[0].text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 0.01,
                      f"{_bar.get_height():.0%}", ha="center", fontsize=10, color="#2C3E50")
    _axes[0].grid(True, alpha=0.3, axis="y")

    # Overall comparison
    _axes[1].bar(["Model A", "Model B"], [_a_overall, _b_overall],
                 color=["#3498DB", "#E74C3C"], alpha=0.85, width=0.5)
    _axes[1].set_ylabel("Overall Conversion Rate", fontsize=11)
    _axes[1].set_title("Overall Aggregate:\nModel B appears BETTER (!)", fontsize=11, fontweight="bold")
    _axes[1].set_ylim(0, 0.80)
    _axes[1].text(0, _a_overall + 0.01, f"{_a_overall:.1%}", ha="center", fontsize=13,
                  fontweight="bold", color="#3498DB")
    _axes[1].text(1, _b_overall + 0.01, f"{_b_overall:.1%}", ha="center", fontsize=13,
                  fontweight="bold", color="#E74C3C")
    _axes[1].grid(True, alpha=0.3, axis="y")

    _note = ("Model B gets assigned to 80% power users vs 50% for A.\n"
             "More high-converters inflates its overall rate — not a real improvement.")
    _axes[1].text(0.5, 0.05, _note, transform=_axes[1].transAxes, ha="center",
                  fontsize=9, style="italic", color="#7F8C8D",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    _fig.suptitle("Simpson's Paradox: Imbalanced Groups Reverse the True Effect",
                  fontsize=13, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    **Simpson's Paradox is why you must verify your randomization balanced the groups.** Always check
    that treatment and control have similar distributions of key covariates (user type, tenure, region)
    before analyzing results.

    ### Pitfall 4: Metric Sensitivity

    Choosing the wrong primary metric:
    - **Too noisy** → need enormous sample size → test never reaches significance
    - **Too narrow** → misses the real impact

    **Metric hierarchy for a job recommendation system:**

    | Type | Example | Role |
    |------|---------|------|
    | **Primary** | Application rate | Decides ship/no-ship — direct business value |
    | **Secondary** | CTR, session length, return rate | Supporting evidence — more sensitive, earlier signal |
    | **Guardrail** | Page load time, error rate, user complaints | Must NOT degrade — can BLOCK a launch even if primary is positive |

    > *Primary metric decides ship/no-ship. Guardrail metrics can block a launch even if
    primary is positive.*
    """)
    return


# ─── PART 5: CAUSAL INFERENCE ────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Part 5: Causal Inference — When You Can't Run an Experiment

    ### When A/B tests aren't possible

    You can't always randomize:
    - **Ethical**: can't randomly deny users a safety feature
    - **Practical**: can't A/B test a complete platform redesign (too disruptive)
    - **Historical**: you want to measure the effect of something that already happened

    **Causal inference**: statistical methods to estimate causal effects from observational data.
    Weaker than A/B tests but sometimes the best option.
    """)
    return


@app.cell
def confounding_viz():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy import stats as _stats

    _rng = _np.random.default_rng(42)
    _n = 300

    # Temperature (confounder) causes both ice cream sales AND drowning
    _temperature = _rng.normal(20, 8, _n)  # degrees C
    _ice_cream = 0.8 * _temperature + _rng.normal(0, 3, _n)
    _drowning = 0.6 * _temperature + _rng.normal(0, 4, _n)

    # Residuals after controlling for temperature
    _ice_resid = _ice_cream - (0.8 * _temperature)
    _drown_resid = _drowning - (0.6 * _temperature)

    _r_naive, _ = _stats.pearsonr(_ice_cream, _drowning)
    _r_controlled, _ = _stats.pearsonr(_ice_resid, _drown_resid)

    _fig, _axes = _plt.subplots(1, 3, figsize=(14, 4))

    # Naive correlation
    _axes[0].scatter(_ice_cream, _drowning, alpha=0.4, color="#E74C3C", s=25)
    _m, _b, *_ = _stats.linregress(_ice_cream, _drowning)
    _x_line = _np.linspace(_ice_cream.min(), _ice_cream.max(), 100)
    _axes[0].plot(_x_line, _m * _x_line + _b, "k-", lw=2)
    _axes[0].set_title(f"Naive: Ice Cream vs Drowning\nr = {_r_naive:.2f} (strong!)",
                        fontsize=10, fontweight="bold")
    _axes[0].set_xlabel("Ice Cream Sales")
    _axes[0].set_ylabel("Drowning Rate")

    # Confounder
    _sc = _axes[1].scatter(_temperature, _ice_cream, c=_drowning, cmap="RdYlGn_r",
                           alpha=0.6, s=30)
    _plt.colorbar(_sc, ax=_axes[1], label="Drowning rate")
    _axes[1].set_title("The Confounder: Temperature\ncauses BOTH ice cream AND drowning",
                        fontsize=10, fontweight="bold")
    _axes[1].set_xlabel("Temperature (°C)")
    _axes[1].set_ylabel("Ice Cream Sales")

    # After controlling
    _axes[2].scatter(_ice_resid, _drown_resid, alpha=0.4, color="#2ECC71", s=25)
    _m2, _b2, *_ = _stats.linregress(_ice_resid, _drown_resid)
    _x2 = _np.linspace(_ice_resid.min(), _ice_resid.max(), 100)
    _axes[2].plot(_x2, _m2 * _x2 + _b2, "k-", lw=2)
    _axes[2].set_title(f"After controlling for temperature:\nr = {_r_controlled:.2f} (gone!)",
                        fontsize=10, fontweight="bold")
    _axes[2].set_xlabel("Ice Cream (residual)")
    _axes[2].set_ylabel("Drowning (residual)")

    _fig.suptitle("Confounding: Temperature Causes Both Ice Cream Sales AND Drowning",
                  fontsize=12, fontweight="bold")
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    > *The confounder question to always ask: "What third variable could cause BOTH the treatment
    and the outcome?" If you can't answer "nothing," you can't claim causation.*

    ### Key causal inference methods

    - **Difference-in-Differences (DiD)**: compare the *change* over time in a treatment group vs a
      control group. Assumes parallel trends before treatment.
      - *Example*: new scoring model launched in Austin but not San Antonio. Compare application rates
        before/after in both cities. The **difference in the changes** = causal effect.

    - **Regression Discontinuity (RD)**: when treatment is assigned by a threshold (e.g., score > 7
      gets highlighted). Compare outcomes just above vs just below the threshold.
      - *Example*: jobs scoring 7.0 vs 6.9 are nearly identical but one is highlighted. The outcome
        difference estimates the causal effect of highlighting.

    - **Propensity Score Matching (PSM)**: for each treated unit, find an untreated unit with similar
      characteristics. Compare matched pairs.
      - *Example*: users who adopted the new feature vs those who didn't. Match on demographics and
        history, then compare outcomes.

    - **Instrumental Variables (IV)**: find a variable that affects treatment but not outcome directly.
      Use it to isolate the causal effect.

    ### When to use what

    | Method | When to use | Key assumption | Evidence strength |
    |--------|------------|----------------|-------------------|
    | A/B test (RCT) | You can randomize | Randomization works | Gold standard |
    | Diff-in-Diff | Natural experiment, before/after data | Parallel trends | Strong |
    | Regression Discontinuity | Treatment assigned by threshold | Continuity at threshold | Strong |
    | Propensity Score Matching | Observational data, rich covariates | No unmeasured confounders | Moderate |
    | Simple regression | Quick exploratory analysis | No confounding (rarely true) | Weak |
    """)
    return


# ─── PART 6: PRACTICAL CHECKLIST ─────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Part 6: Practical Experiment Design for ML Systems

    ### The ML experiment checklist

    ```
    Before the experiment:
    □ Define primary metric (one, not five)
    □ Define secondary metrics (2-3, supporting evidence)
    □ Define guardrail metrics (must not degrade)
    □ Compute required sample size (power analysis)
    □ Determine test duration (minimum 1 week, ideally 2)
    □ Set randomization unit (user-level for most ML, cluster for network effects)
    □ Pre-register: write down hypothesis, metrics, duration, decision criteria BEFORE starting
    □ Verify randomization: check that treatment/control groups are balanced on key covariates

    During the experiment:
    □ Don't peek (or use sequential testing with adjusted thresholds)
    □ Monitor guardrail metrics continuously
    □ If guardrails violated → stop, rollback

    After the experiment:
    □ Check sample sizes match plan (did users drop out differentially?)
    □ Compute primary metric + confidence interval
    □ Check secondary metrics for consistent signal
    □ Confirm guardrails passed
    □ Segment analysis: is the effect consistent across user segments? (watch for Simpson's paradox)
    □ Write a decision document: hypothesis, setup, results, decision, learnings
    ```

    ### Connecting to my MLOps pipeline

    From my MLOps notebook: **shadow → canary → A/B → rollout**

    - **Shadow**: no randomization needed, just log predictions alongside production
    - **Canary (5%)**: not a proper A/B test — just checking for crashes, errors, latency
    - **A/B (50/50)**: the REAL experiment. Power analysis, pre-registration, proper duration,
      statistical analysis.
    - **Rollout (100%)**: only after A/B shows positive or neutral results on primary + guardrails

    > *The A/B test is the statistical gate between "this model looks good offline" and "this model
    actually improves the product."*
    """)
    return


# ─── FLASHCARDS ───────────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Flashcard Summary

    **What's a p-value?**
    Probability of seeing results this extreme IF there's no real effect. NOT the probability
    that the effect is real.

    **Type I vs Type II error?**
    Type I: false positive (see an effect that isn't there), controlled by α.
    Type II: false negative (miss a real effect), controlled by power = 1 − β.

    **How do you calculate sample size for an A/B test?**
    Power analysis. Need: baseline rate, minimum detectable effect, α, and desired power (1 − β).
    Smaller MDE → exponentially more samples.

    **What's the peeking problem?**
    Checking results before the test is complete inflates the false positive rate. Pre-commit to
    duration or use sequential testing.

    **When can't you run an A/B test?**
    Ethical constraints, impractical to randomize, or analyzing historical events. Use causal
    inference methods instead (DiD, RD, PSM).

    **Difference-in-differences in one sentence?**
    Compare the change over time in treated vs untreated groups — the difference in changes
    estimates the causal effect.

    **What's Simpson's paradox?**
    An effect that appears in aggregate reverses in every subgroup. Caused by imbalanced group
    composition — always verify randomization balanced key covariates.

    **Primary vs secondary vs guardrail metrics?**
    Primary decides ship/no-ship. Secondary provides supporting evidence. Guardrails can block
    a launch even if the primary is positive.

    **Correlation ≠ causation because?**
    Confounders: a third variable causing both the apparent cause and the effect. Only randomization
    (A/B test) or careful causal inference methods can establish causation.

    **What's statistical power?**
    Probability of detecting a real effect (1 − β). Target 80%. Low power means you'll miss
    real improvements.
    """)
    return


# ─── INTERVIEW TALKING POINTS ─────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Interview Talking Points

    **"How do you decide whether to launch a new model?"**
    A/B test. Compute required sample size from a power analysis based on baseline metric, MDE,
    and traffic volume. Run for at least 2 weeks with 50/50 randomization. Primary metric must be
    statistically significant (p < 0.05) with a practical effect size worth shipping. Guardrail
    metrics (latency, error rate) must not degrade. Write a decision doc and get stakeholder sign-off.

    **"Design an A/B test for a new recommendation model."**
    Primary metric: application rate (direct business value). Secondary: CTR, session length.
    Guardrails: page load time, error rate. Sample size: power analysis for detecting 1% absolute
    improvement at 80% power. Duration: 2 weeks minimum. Randomize at user level. Pre-register
    everything. No peeking.

    **"When would you use causal inference instead of A/B testing?"**
    When randomization isn't possible. If a new scoring model was already launched in one market,
    I'd use difference-in-differences comparing that market to a similar control market. If treatment
    was assigned by a threshold (score > 7), regression discontinuity. These are weaker than A/B
    tests but sometimes the only option.

    **Connection to my work:**
    My backtesting engine IS an experiment framework — it tests strategy variants against historical
    data with proper temporal train/test splits. Walk-forward validation prevents look-ahead bias,
    which is the time-series equivalent of peeking in an A/B test. Same statistical principles,
    different domain.
    """)
    return


if __name__ == "__main__":
    app.run()
