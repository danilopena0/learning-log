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
    # Feature Engineering — Encoding, Scaling, Missing Data, Feature Stores

    | Field  | Value |
    |--------|-------|
    | Date   | 2026-04-30 |
    | Track  | ML Theory |
    | Time   | 60 min |
    | Topics | Missing Data · Encoding · Scaling · Feature Creation · Feature Stores |
    """)
    return


@app.cell
def why_fe_wins(mo):
    mo.md("""
    ## Why Feature Engineering Wins Competitions and Jobs

    **"Applied ML is 80% feature engineering, 10% model selection, 10% tuning"** — every
    practitioner says this, and it's true. Good features make simple models work. Bad features
    make complex models fail.

    What interviewers test: can you take raw messy data and turn it into something a model can
    learn from? Do you know the traps — leakage, scaling pitfalls, encoding gotchas?

    **Feature engineering IS the difference between a Kaggle notebook and production ML.**

    Models are commoditized. The data you feed them is the differentiator.

    > "In my fraud detection system design, I spent more time on the feature engineering section
    > than the model section — because that's where the signal lives."
    """)
    return


# ---------------------------------------------------------------------------
# Synthetic dataset — shared across all Part 1–4 cells
# ---------------------------------------------------------------------------

@app.cell
def synthetic_data():
    import numpy as _np
    import pandas as _pd

    _rng = _np.random.default_rng(42)
    _n = 1000

    # Numeric features
    _salary_raw = _np.exp(_rng.normal(11.2, 0.5, _n)).clip(40_000, 350_000)
    _years_exp  = _rng.exponential(4, _n).clip(0, 25).round(1)
    _skills_mat = _rng.integers(0, 11, _n)
    _skills_req = _rng.integers(3, 13, _n)
    _app_count  = _rng.negative_binomial(3, 0.3, _n)

    # Categorical features
    _cities = _rng.choice(
        ['Austin', 'San Antonio', 'Dallas', 'Houston', 'Remote', 'New York', 'Seattle', 'Chicago'],
        _n, p=[0.18, 0.14, 0.16, 0.12, 0.20, 0.10, 0.06, 0.04],
    )
    _job_type = _rng.choice(['Full-time', 'Contract', 'Part-time'], _n, p=[0.70, 0.22, 0.08])
    _level_map = ['Junior', 'Mid', 'Senior', 'Lead']
    _level_idx = _np.clip(_rng.poisson(1.2, _n), 0, 3)
    _level     = _np.array(_level_map)[_level_idx]
    _companies = _np.array([f"Company_{i:02d}" for i in range(50)])
    _company   = _companies[_rng.integers(0, 50, _n)]

    # Timestamps
    _base = _pd.Timestamp('2026-01-01')
    _posted_at = [
        _base + _pd.Timedelta(days=int(d), hours=int(h))
        for d, h in zip(_rng.integers(0, 120, _n), _rng.integers(0, 24, _n))
    ]

    # Binary target: good match?
    _score = (
        0.4 * (_salary_raw / 150_000).clip(0, 1)
        + 0.4 * (_skills_mat / 10)
        + 0.2 * (_level_idx / 3)
        + _rng.normal(0, 0.1, _n)
    )
    _is_good_match = (_score > 0.55).astype(int)

    df = _pd.DataFrame({
        'salary':           _salary_raw,
        'years_experience': _years_exp,
        'skills_matched':   _skills_mat,
        'skills_required':  _skills_req,
        'application_count': _app_count,
        'city':             _cities,
        'job_type':         _job_type,
        'level':            _level,
        'company':          _company,
        'posted_at':        _posted_at,
        'is_good_match':    _is_good_match,
    })

    # --- Introduce realistic missing values ---
    # MNAR: high earners skip the salary field (~15% overall, correlated with high salary)
    _high_earner   = df['salary'] > df['salary'].quantile(0.70)
    _missing_salary = (
        (_rng.random(_n) < 0.08)
        | (_high_earner & (_rng.random(_n) < 0.25))
    )
    df.loc[_missing_salary, 'salary'] = _np.nan

    # MCAR: 3% of experience missing at random
    df.loc[_rng.random(_n) < 0.03, 'years_experience'] = _np.nan

    # MAR: contract jobs less likely to list skills_required
    _contract_mask = (df['job_type'] == 'Contract') & (_rng.random(_n) < 0.20)
    df.loc[_contract_mask, 'skills_required'] = _np.nan

    return (df,)


@app.cell
def train_test_prep(df):
    from sklearn.model_selection import train_test_split as _tts

    _feature_cols = [
        'salary', 'years_experience', 'skills_matched', 'skills_required',
        'application_count', 'city', 'job_type', 'level', 'company',
    ]
    _X = df[_feature_cols]
    _y = df['is_good_match']

    numeric_features     = ['salary', 'years_experience', 'skills_matched',
                             'skills_required', 'application_count']
    categorical_features = ['city', 'job_type', 'level', 'company']

    X_train, X_test, y_train, y_test = _tts(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )
    return (X_train, X_test, y_train, y_test, numeric_features, categorical_features)


# ---------------------------------------------------------------------------
# Part 1: Handling Missing Data
# ---------------------------------------------------------------------------

@app.cell
def part1_header(mo):
    mo.md("## Part 1: Handling Missing Data")
    return


@app.cell
def missing_data_why(mo):
    mo.md("""
    ### Why Missing Data Matters

    Most real-world datasets have missing values. Ignoring them = crashing or biased models.

    **Types of missingness** — the mechanism determines the strategy:

    | Type | What it means | Real example | Difficulty |
    |------|--------------|--------------|------------|
    | **MCAR** — Missing Completely At Random | unrelated to any data | random data entry errors | Easy |
    | **MAR** — Missing At Random | depends on *observed* variables | contract jobs skip `skills_required` | Manageable |
    | **MNAR** — Missing Not At Random | depends on the *missing value itself* | high earners hide salary | Hard |

    The strategy depends on both the mechanism AND the downstream model. Tree models
    (XGBoost, LightGBM) handle `NaN` natively. Linear models, KNN, neural nets require
    imputation before training.
    """)
    return


@app.cell
def missing_overview(mo, df):
    _missing_count = df.isnull().sum()
    _missing_pct   = (df.isnull().mean() * 100).round(1)
    _mech = {
        'salary':           'MNAR — high earners hide salary',
        'years_experience': 'MCAR — random 3%',
        'skills_required':  'MAR — contract jobs skip it',
    }

    _rows = "\n".join(
        f"| `{col}` | {int(_missing_count[col])} | {_missing_pct[col]}% | {_mech[col]} |"
        for col in _mech
    )

    mo.md(f"""
    ### Dataset: {len(df):,} synthetic job postings

    Three columns have missing values — each with a different mechanism:

    | Feature | Missing count | Missing % | Mechanism |
    |---------|--------------|-----------|-----------|
    {_rows}

    Salary is the interesting case: missingness is *correlated with the value itself* (MNAR).
    Dropping those rows would bias the model toward lower-salary data.
    """)
    return


@app.cell
def missing_viz(df):
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _fig, _axes = _plt.subplots(1, 2, figsize=(13, 4))

    # Left: horizontal bar chart of missing %
    _target_cols  = ['salary', 'years_experience', 'skills_required']
    _pcts         = [df[c].isnull().mean() * 100 for c in _target_cols]
    _bar_colors   = ['#E74C3C' if p > 10 else '#F39C12' if p > 5 else '#3498DB' for p in _pcts]
    _mechanisms   = ['MNAR', 'MCAR', 'MAR']

    _bars = _axes[0].barh(_target_cols, _pcts, color=_bar_colors, height=0.5, edgecolor='white')
    for _i, (_p, _m) in enumerate(zip(_pcts, _mechanisms)):
        _axes[0].text(_p + 0.3, _i, f'{_p:.1f}%  ({_m})', va='center', fontsize=10, fontweight='bold')
    _axes[0].axvline(5,  color='#27AE60', linestyle='--', lw=1.5, alpha=0.8, label='5% threshold')
    _axes[0].axvline(50, color='#C0392B', linestyle='--', lw=1.5, alpha=0.8, label='50% → drop feature?')
    _axes[0].set_xlabel('% Missing', fontsize=11)
    _axes[0].set_title('Missing Data by Feature', fontsize=13, fontweight='bold')
    _axes[0].set_xlim(0, 22)
    _axes[0].legend(fontsize=9)
    _axes[0].grid(axis='x', alpha=0.3)
    _axes[0].spines['top'].set_visible(False)
    _axes[0].spines['right'].set_visible(False)

    # Right: missingness pattern grid (200 rows sample)
    _sample     = df[_target_cols].head(200)
    _is_missing = _sample.isnull().astype(int).values
    _im = _axes[1].imshow(
        _is_missing.T, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1, interpolation='none'
    )
    _axes[1].set_yticks(range(3))
    _axes[1].set_yticklabels(_target_cols, fontsize=10)
    _axes[1].set_xlabel('Row index (first 200 rows)', fontsize=11)
    _axes[1].set_title('Missingness Pattern\ngreen = present  |  red = missing', fontsize=13, fontweight='bold')
    _cb = _plt.colorbar(_im, ax=_axes[1], fraction=0.046, pad=0.04)
    _cb.set_ticks([0, 1])
    _cb.set_ticklabels(['Present', 'Missing'])

    _fig.tight_layout()
    return _fig


@app.cell
def missing_strategies_demo(mo, X_train, X_test):
    import numpy as _np
    import pandas as _pd
    from sklearn.impute import SimpleImputer as _SI, KNNImputer as _KNNI

    try:
        from sklearn.experimental import enable_iterative_imputer as _  # noqa: pre-1.4 compat
    except ImportError:
        pass
    from sklearn.impute import IterativeImputer as _IterImp

    _num_cols  = ['salary', 'years_experience', 'skills_matched', 'skills_required', 'application_count']
    _Xn_train  = X_train[_num_cols].copy()
    _n_before  = len(_Xn_train)

    # Strategy 1: drop rows
    _n_after_drop = len(_Xn_train.dropna())

    # Strategy 2: median imputation
    _med_imp = _SI(strategy='median')
    _X_med   = _pd.DataFrame(_med_imp.fit_transform(_Xn_train), columns=_num_cols)

    # Strategy 3: missingness indicator + median
    _X_ind = _Xn_train.copy()
    for _col in _num_cols:
        if _Xn_train[_col].isnull().any():
            _X_ind[f'{_col}_missing'] = _Xn_train[_col].isnull().astype(int)
    _X_ind[_num_cols] = _SI(strategy='median').fit_transform(_Xn_train)

    # Strategy 4: KNN
    _X_knn = _pd.DataFrame(
        _KNNI(n_neighbors=5).fit_transform(_Xn_train), columns=_num_cols
    )

    # Strategy 5: iterative (model-based)
    _X_iter = _pd.DataFrame(
        _IterImp(max_iter=5, random_state=42).fit_transform(_Xn_train), columns=_num_cols
    )

    _sal_orig   = _Xn_train['salary'].dropna().mean()
    _sal_med    = _X_med['salary'].mean()
    _sal_knn    = _X_knn['salary'].mean()
    _sal_iter   = _X_iter['salary'].mean()
    _indicator_cols = [c for c in _X_ind.columns if c.endswith('_missing')]

    mo.md(f"""
    ### Imputation Strategies — Live Demo

    Training rows: **{_n_before}** | Salary missing: **{_Xn_train['salary'].isnull().sum()}
    ({_Xn_train['salary'].isnull().mean()*100:.1f}%)**

    | Strategy | Rows kept | NaN left | Mean salary after | Notes |
    |----------|-----------|----------|-------------------|-------|
    | Drop rows | {_n_after_drop} ({_n_after_drop/_n_before*100:.0f}%) | 0 | ${_sal_orig:,.0f} | Biased — drops high earners (MNAR) |
    | Median impute | {_n_before} (100%) | 0 | ${_sal_med:,.0f} | Fast baseline, no row loss |
    | Median + indicator | {_n_before} (100%) | 0 | ${_sal_med:,.0f} | Adds {len(_indicator_cols)} binary signal columns |
    | KNN impute | {_n_before} (100%) | 0 | ${_sal_knn:,.0f} | Uses correlated features, slower |
    | Iterative impute | {_n_before} (100%) | 0 | ${_sal_iter:,.0f} | Best quality, slowest |

    **The golden rule in code:**
    ```python
    # WRONG: fitting on full data before split → leakage
    imputer = SimpleImputer(strategy='median')
    imputer.fit(df[numeric_cols])           # sees test statistics!

    # RIGHT: fit on train only, transform both
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train[numeric_cols])      # train stats only
    X_train_imp = imputer.transform(X_train[numeric_cols])
    X_test_imp  = imputer.transform(X_test[numeric_cols])   # same params, no leakage

    # Missingness indicator: always add BEFORE imputing
    df['salary_missing'] = df['salary'].isna().astype(int)
    df['salary']         = df['salary'].fillna(df['salary'].median())
    # "Missing salary + high experience" → model can learn this pattern
    ```
    """)
    return


@app.cell
def missing_decision(mo):
    mo.md("""
    ### Missing Data Decision Framework

    ```
    What % is missing?
    ├── >50%  → Consider dropping the FEATURE entirely (not rows)
    ├── 5–50% → Impute (median for numeric, mode for categorical) + missingness indicator
    └── <5%   → Drop rows if MCAR, impute if MAR or MNAR

    Is missingness itself informative?
    ├── YES       → Always add a binary indicator column before imputing
    └── NO/Unsure → Add indicator anyway — it costs nothing and lets the model decide

    What model will you use?
    ├── Tree-based (XGBoost, LightGBM) → can handle NaN natively — pass df as-is
    └── Linear / KNN / Neural net      → must impute first

    CRITICAL: fit imputer on TRAIN only, transform train + test with the same fitted params.
    Fitting on full data before the split = leakage (same rule as your validation notebook).
    ```
    """)
    return


# ---------------------------------------------------------------------------
# Part 2: Encoding Categorical Variables
# ---------------------------------------------------------------------------

@app.cell
def part2_header(mo):
    mo.md("## Part 2: Encoding Categorical Variables")
    return


@app.cell
def encoding_why(mo):
    mo.md("""
    ### Why Encoding Matters

    Models need numbers. `"San Antonio"` and `"Austin"` are strings. The wrong encoding
    can destroy information or create false relationships.

    The key question: **does the category have a natural order?** If yes → ordinal. If no →
    the *cardinality* decides which technique to use. Every technique has a failure mode —
    knowing them is what separates an ML engineer from a notebook jockey.
    """)
    return


@app.cell
def encoding_demo(mo, X_train, X_test, y_train):
    import numpy as _np
    import pandas as _pd
    from sklearn.preprocessing import OneHotEncoder as _OHE, OrdinalEncoder as _OE, TargetEncoder as _TE

    # One-hot encoding: city (8 unique values)
    _ohe = _OHE(handle_unknown='ignore', sparse_output=False)
    _city_ohe_train = _ohe.fit_transform(X_train[['city']])
    _city_ohe_test  = _ohe.transform(X_test[['city']])
    _n_ohe_cols     = _city_ohe_train.shape[1]

    # Ordinal encoding: level has a meaningful order
    _oe = _OE(
        categories=[['Junior', 'Mid', 'Senior', 'Lead']],
        handle_unknown='use_encoded_value', unknown_value=-1,
    )
    _level_ord = _oe.fit_transform(X_train[['level']])

    # Target encoding: company (50 unique values — high cardinality)
    _te = _TE(smooth='auto')
    _co_te_train = _te.fit_transform(X_train[['company']], y_train)
    _co_te_test  = _te.transform(X_test[['company']])
    _sample_pairs = list(zip(
        X_train['company'].head(4).values,
        _co_te_train[:4].flatten().round(3)
    ))

    # Frequency encoding: leakage-safe alternative
    _freq          = X_train['company'].value_counts(normalize=True)
    _co_freq_train = X_train['company'].map(_freq).values
    _co_freq_test  = X_test['company'].map(_freq).fillna(0).values

    mo.md(f"""
    ### Encoding Strategies — Live Demo

    ---

    **One-hot encoding** (city, {X_train['city'].nunique()} unique → {_n_ohe_cols} binary columns):
    ```python
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    enc.fit(X_train[['city']])
    X_train_ohe = enc.transform(X_train[['city']])  # (n_train, {_n_ohe_cols})
    X_test_ohe  = enc.transform(X_test[['city']])   # unknown cities → all zeros (no crash)
    # drop_first=True in get_dummies() avoids multicollinearity — but use sklearn in production.
    # pd.get_dummies() does NOT handle unseen categories at inference → column mismatch crash.
    ```

    ---

    **Ordinal encoding** (level: Junior=0, Mid=1, Senior=2, Lead=3):
    ```python
    enc = OrdinalEncoder(categories=[['Junior', 'Mid', 'Senior', 'Lead']])
    df['level_encoded'] = enc.fit_transform(df[['level']])
    # TRAP: only for ORDERED categories.
    # Ordinal-encoding 'city' implies Austin < Dallas — meaningless and harms the model.
    ```

    ---

    **Target encoding** (company, {X_train['company'].nunique()} unique → 1 numeric column):
    ```python
    enc = TargetEncoder(smooth='auto')   # sklearn 1.3+
    enc.fit(X_train[['company']], y_train)
    X_train['company_enc'] = enc.transform(X_train[['company']])
    X_test['company_enc']  = enc.transform(X_test[['company']])
    # smooth='auto': rare companies shrink toward the global match rate — reduces noise.
    # TRAP: encode on full dataset → company "XYZ" embeds the target of the row being predicted.
    # FIX: always fit TargetEncoder inside CV folds, never on full data.
    ```

    Sample: company → estimated match probability:
    {" | ".join(f"`{co}` → {v}" for co, v in _sample_pairs)}

    ---

    **Frequency encoding** (leakage-safe, no target needed):
    ```python
    freq = X_train['company'].value_counts(normalize=True)
    X_train['company_freq'] = X_train['company'].map(freq)
    X_test['company_freq']  = X_test['company'].map(freq).fillna(0)  # unseen → 0
    # Useful when company popularity IS a signal (big companies post more jobs).
    # Zero leakage: frequency is computed from training features only, not the target.
    ```

    ---

    **Embeddings** (for text-like categories at very high cardinality):
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['company_emb'] = df['company_name'].apply(model.encode)
    # Captures semantic similarity: "Google" ≈ "Alphabet" ≈ "DeepMind"
    # This is exactly what Canopy does for job description matching.
    ```
    """)
    return


@app.cell
def encoding_decision(mo):
    mo.md("""
    ### Encoding Decision Framework

    ```
    How many unique values?
    ├── 2          → Binary (0/1)
    ├── 3–20       → One-hot encoding
    ├── 20–1,000   → Target encoding (within CV folds) or frequency encoding
    ├── 1,000+     → Embeddings or hashing trick
    └── Ordered    → Ordinal encoding (education level, job seniority, star rating)

    What model?
    ├── Tree-based (XGBoost, RF) → label/ordinal encoding works fine — trees split at any value
    ├── Linear models            → one-hot required (label encoding implies false ordinal order)
    └── Neural networks          → embeddings or one-hot

    Production rule: never use pd.get_dummies() at inference time.
    Use sklearn's OneHotEncoder(handle_unknown='ignore') so new categories don't crash the pipeline.
    ```
    """)
    return


# ---------------------------------------------------------------------------
# Part 3: Feature Scaling
# ---------------------------------------------------------------------------

@app.cell
def part3_header(mo):
    mo.md("## Part 3: Feature Scaling")
    return


@app.cell
def scaling_why(mo):
    mo.md("""
    ### Why Scaling Matters — and When It Doesn't

    Distance-based and gradient-based models are dominated by whichever feature has the
    largest scale. **Salary** (40K–200K) vs **years_experience** (0–25) — without scaling,
    salary overwhelms the gradient and experience is invisible.

    | Model | Needs scaling? | Reason |
    |-------|---------------|--------|
    | Linear / Logistic Regression | **Yes** | gradient magnitude is scale-dependent |
    | SVM | **Yes** | kernel distances are scale-sensitive |
    | KNN | **Yes** | Euclidean distance dominated by large-scale features |
    | Neural networks | **Yes** | vanishing/exploding gradients; optimizer convergence |
    | PCA | **Yes** | variance is scale-dependent — large features dominate components |
    | XGBoost / LightGBM / Random Forest | **No** | split thresholds are scale-invariant |

    **"Do I need to scale for XGBoost?"** → No. Trees are immune to monotonic transformations.
    This is a standard interview question.
    """)
    return


@app.cell
def scaling_demo(mo, X_train, X_test, numeric_features):
    import numpy as _np
    import pandas as _pd
    from sklearn.preprocessing import StandardScaler as _SS, MinMaxScaler as _MMS, RobustScaler as _RS
    from sklearn.impute import SimpleImputer as _SI

    # Impute first so scalers don't fail on NaN
    _imp    = _SI(strategy='median')
    _Xn_tr  = _pd.DataFrame(_imp.fit_transform(X_train[numeric_features]), columns=numeric_features)
    _Xn_te  = _pd.DataFrame(_imp.transform(X_test[numeric_features]),  columns=numeric_features)
    _sal    = _Xn_tr['salary']

    _ss  = _SS();  _X_ss  = _pd.DataFrame(_ss.fit_transform(_Xn_tr),  columns=numeric_features)
    _mms = _MMS(); _X_mms = _pd.DataFrame(_mms.fit_transform(_Xn_tr), columns=numeric_features)
    _rs  = _RS();  _X_rs  = _pd.DataFrame(_rs.fit_transform(_Xn_tr),  columns=numeric_features)
    _sal_log = _np.log1p(_sal)

    def _fmt(series):
        return f"mean={series.mean():.2f}, std={series.std():.2f}, range=[{series.min():.2f}, {series.max():.2f}]"

    mo.md(f"""
    ### Scaling Methods — Live Demo on Salary Feature

    Raw salary: mean=${_sal.mean():,.0f}, std=${_sal.std():,.0f},
    range=[${_sal.min():,.0f}, ${_sal.max():,.0f}]

    ---

    **StandardScaler** — `(x - mean) / std`
    ```python
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
    X_test_scaled  = scaler.transform(X_test)        # same mean/std, no leakage
    # Result: mean ≈ 0, std ≈ 1
    # When: default choice, data is roughly Gaussian
    # TRAP: outliers shift the mean and inflate the std → distorts scaling for everyone else
    ```
    After: {_fmt(_X_ss['salary'])}

    ---

    **MinMaxScaler** — `(x - min) / (max - min)`
    ```python
    scaler = MinMaxScaler()
    # Result: all values in [0, 1]
    # When: bounded features, neural nets that expect [0, 1], image pixel values
    # TRAP: one outlier (salary=$1M) compresses everything else into a tiny range near 0
    ```
    After: {_fmt(_X_mms['salary'])}

    ---

    **RobustScaler** — `(x - median) / IQR`
    ```python
    scaler = RobustScaler()
    # Uses median and interquartile range — both robust to extreme values
    # When: salary, price, income, any monetary amount — real-world data has outliers
    # Outliers still exist after scaling, but they don't distort the bulk of the data
    ```
    After: {_fmt(_X_rs['salary'])}

    ---

    **Log transform** — `log1p(x)` — not a scaler but often the right first move
    ```python
    df['salary_log'] = np.log1p(df['salary'])   # log1p handles zeros safely
    # Compresses right-skewed distributions, makes them more Gaussian-like
    # When: monetary amounts, counts, anything with a long right tail
    # Pair with StandardScaler afterward for full normalization
    ```
    After: {_fmt(_sal_log)}
    """)
    return


@app.cell
def scaling_viz(X_train, numeric_features):
    import numpy as _np
    import pandas as _pd
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from sklearn.preprocessing import StandardScaler as _SS, MinMaxScaler as _MMS, RobustScaler as _RS
    from sklearn.impute import SimpleImputer as _SI

    # Prep: impute so scalers don't crash on NaN
    _imp   = _SI(strategy='median')
    _Xn    = _pd.DataFrame(_imp.fit_transform(X_train[numeric_features]), columns=numeric_features)
    _sal   = _Xn['salary'].values

    _transforms = [
        ('Raw Salary\n(right-skewed, outliers)',
         _sal, '#95A5A6', 'Skewed distribution\noutliers visible on right'),
        ('StandardScaler\n(z-score normalization)',
         _SS().fit_transform(_sal.reshape(-1, 1)).ravel(), '#E74C3C',
         'Mean distorted by outliers\nstd inflated'),
        ('RobustScaler\n(median / IQR)',
         _RS().fit_transform(_sal.reshape(-1, 1)).ravel(), '#27AE60',
         'Robust to outliers\nbulk well-centered'),
        ('Log Transform\n(log1p — fixes skew)',
         _np.log1p(_sal), '#3498DB',
         'Near-Gaussian shape\npair with StandardScaler'),
    ]

    _fig, _axes = _plt.subplots(1, 4, figsize=(15, 4))

    for _ax, (_title, _data, _color, _note) in zip(_axes, _transforms):
        _ax.hist(_data, bins=45, color=_color, alpha=0.80, edgecolor='white', linewidth=0.4)
        _mu = _np.mean(_data)
        _ax.axvline(_mu, color='black', lw=1.8, linestyle='--')
        _ax.text(
            0.97, 0.95, f'mean={_mu:.1f}',
            transform=_ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        )
        _ax.set_title(_title, fontsize=10, fontweight='bold', pad=6)
        _ax.set_xlabel(_note, fontsize=8, color='#555555')
        _ax.set_ylabel('Count' if _ax is _axes[0] else '', fontsize=9)
        _ax.grid(axis='y', alpha=0.3)
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)

    _fig.suptitle(
        'Scaling Comparison — Salary Feature (800 training rows)',
        fontsize=13, fontweight='bold', y=1.02,
    )
    _fig.tight_layout()
    return _fig


@app.cell
def scaling_traps(mo):
    mo.md("""
    ### Scaling Traps

    | Trap | What goes wrong | Fix |
    |------|----------------|-----|
    | Fit scaler on full dataset before split | Leakage: test statistics bleed into scaler params | Always fit on train only |
    | Scale at train time, forget at inference | Raw features hit the model → silent garbage predictions | Use sklearn Pipeline |
    | Scale binary / one-hot features | Unnecessary, hurts interpretability | Only scale continuous numeric features |
    | Scale the target, forget `inverse_transform` | Predictions are in the wrong units | `scaler.inverse_transform(preds)` before evaluation |

    The fix for almost all of these: **sklearn Pipeline**. It makes the right behavior automatic.
    """)
    return


@app.cell
def pipeline_demo(mo, X_train, X_test, y_train, y_test, numeric_features, categorical_features):
    import numpy as _np
    from sklearn.pipeline import Pipeline as _Pipeline
    from sklearn.compose import ColumnTransformer as _CT
    from sklearn.preprocessing import StandardScaler as _SS, OneHotEncoder as _OHE
    from sklearn.impute import SimpleImputer as _SI
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.metrics import roc_auc_score as _auc, accuracy_score as _acc

    preprocessor = _CT(
        transformers=[
            ('num', _Pipeline([
                ('impute', _SI(strategy='median')),
                ('scale',  _SS()),
            ]), numeric_features),
            ('cat', _Pipeline([
                ('impute', _SI(strategy='most_frequent')),
                ('encode', _OHE(handle_unknown='ignore', sparse_output=False)),
            ]), categorical_features),
        ],
        remainder='drop',
    )

    pipeline = _Pipeline([
        ('preprocess', preprocessor),
        ('model',      _LR(max_iter=1000, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    _y_pred = pipeline.predict(X_test)
    _y_prob = pipeline.predict_proba(X_test)[:, 1]
    _acc_score = _acc(y_test, _y_pred)
    _auc_score = _auc(y_test, _y_prob)

    mo.md(f"""
    ### sklearn Pipeline — The Pattern to Copy Into Every Project

    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale',  StandardScaler()),
        ]), numeric_features),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical_features),
    ])

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model',      LogisticRegression()),
    ])

    pipeline.fit(X_train, y_train)   # imputer/scaler fit on X_train only — no leakage
    pipeline.predict(X_test)         # same transforms applied automatically at test time
    pipeline.predict(live_data)      # same transforms applied at inference — no skew possible
    ```

    **Results on held-out test set (20% of {len(X_train) + len(X_test):,} job postings):**

    | Metric | Score |
    |--------|-------|
    | Accuracy | **{_acc_score:.3f}** |
    | ROC-AUC  | **{_auc_score:.3f}** |

    > "sklearn Pipeline is the single most important tool for preventing preprocessing leakage.
    > Fit preprocessors on train only within each CV fold — automatically. Save one object for
    > production deployment. If you're not using Pipelines, you're probably leaking somewhere."

    One object to `fit()`. One object to `predict()`. One object to `pickle` and deploy.
    """)
    return (pipeline,)


# ---------------------------------------------------------------------------
# Part 4: Feature Creation Patterns
# ---------------------------------------------------------------------------

@app.cell
def part4_header(mo):
    mo.md("## Part 4: Feature Creation Patterns")
    return


@app.cell
def temporal_features(mo, df):
    import numpy as _np
    import pandas as _pd

    _df   = df.copy()
    _now  = _pd.Timestamp('2026-04-30')
    _dt   = _pd.to_datetime(_df['posted_at'])

    _df['hour']               = _dt.dt.hour
    _df['day_of_week']        = _dt.dt.dayofweek       # 0=Mon, 6=Sun
    _df['is_weekend']         = _df['day_of_week'].isin([5, 6]).astype(int)
    _df['month']              = _dt.dt.month
    _df['days_since_posted']  = (_now - _dt).dt.days

    # Cyclical encoding — hour 23 and hour 0 are neighbours, not extremes
    _df['hour_sin'] = _np.sin(2 * _np.pi * _df['hour'] / 24)
    _df['hour_cos'] = _np.cos(2 * _np.pi * _df['hour'] / 24)
    _df['dow_sin']  = _np.sin(2 * _np.pi * _df['day_of_week'] / 7)
    _df['dow_cos']  = _np.cos(2 * _np.pi * _df['day_of_week'] / 7)

    # Distances at hour 23 vs hour 0: raw gap = 23, cyclical distance ≈ 0.52 rad ≈ correct
    _raw_dist = abs(23 - 0)
    _cyc_dist = _np.sqrt(
        (_np.sin(2*_np.pi*23/24) - _np.sin(2*_np.pi*0/24))**2 +
        (_np.cos(2*_np.pi*23/24) - _np.cos(2*_np.pi*0/24))**2
    )

    mo.md(f"""
    ### Temporal Features

    **Decompose timestamps into learnable signals:**
    ```python
    df['hour']              = df['posted_at'].dt.hour
    df['day_of_week']       = df['posted_at'].dt.dayofweek   # 0=Mon, 6=Sun
    df['is_weekend']        = df['day_of_week'].isin([5, 6]).astype(int)
    df['month']             = df['posted_at'].dt.month
    df['days_since_posted'] = (pd.Timestamp.now() - df['posted_at']).dt.days
    ```

    **Cyclical encoding** — preserves circular distance (encode in sin/cos pairs):
    ```python
    # Without cyclical encoding: hour 23 and hour 0 are 23 units apart — wrong.
    # With cyclical encoding: they're adjacent.
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    # Always use PAIRS — sin alone is ambiguous: sin(30°) == sin(150°)
    ```

    Demonstration on this dataset:

    | | Hours 23 → 0 |
    |-|--------------|
    | Raw distance | {_raw_dist} units (treats them as far apart) |
    | Cyclical (sin/cos) Euclidean distance | {_cyc_dist:.3f} (treats them as adjacent) ✓ |

    Apply the same pattern to `day_of_week` (/7), `month` (/12), minute (/60).
    """)
    return


@app.cell
def aggregation_features(mo, df):
    import numpy as _np
    import pandas as _pd

    _rng = _np.random.default_rng(99)
    _df  = df.copy()
    _now = _pd.Timestamp('2026-04-30')

    # Simulate multiple applications per user
    _df['user_id']   = _rng.integers(0, 200, len(_df))
    _df['score']     = (_df['skills_matched'] / 10 + _rng.normal(0, 0.1, len(_df))).clip(0, 1)
    _df['applied_at'] = _pd.to_datetime(_df['posted_at'])

    _user_feats = _df.groupby('user_id').agg(
        total_applications       = ('score', 'count'),
        avg_score                = ('score', 'mean'),
        score_std                = ('score', 'std'),
        unique_companies         = ('company', 'nunique'),
        days_active              = ('applied_at', lambda x: (x.max() - x.min()).days),
        last_application_days_ago= ('applied_at', lambda x: (_now - x.max()).days),
    ).reset_index()

    _sample_rows = _user_feats.head(4)
    _rows_md = "\n".join(
        f"| {int(r.user_id)} | {int(r.total_applications)} | {r.avg_score:.2f} | "
        f"{r.score_std:.2f} | {int(r.unique_companies)} | {int(r.days_active)} | "
        f"{int(r.last_application_days_ago)} |"
        for _, r in _sample_rows.iterrows()
    )

    mo.md(f"""
    ### Aggregation Features — Behavioral Patterns

    Row-level features miss behavioral signals. Aggregate per entity (user, company, session):

    ```python
    user_features = df.groupby('user_id').agg(
        total_applications        = ('application_id', 'count'),
        avg_score                 = ('score', 'mean'),
        score_std                 = ('score', 'std'),            # consistency metric
        unique_companies          = ('company', 'nunique'),
        days_active               = ('applied_at', lambda x: (x.max() - x.min()).days),
        last_application_days_ago = ('applied_at', lambda x: (now - x.max()).days),
    ).reset_index()
    ```

    Sample user feature table ({len(_user_feats):,} users from {len(_df):,} rows):

    | user_id | applications | avg_score | score_std | companies | days_active | last_app_days_ago |
    |---------|-------------|-----------|-----------|-----------|-------------|-------------------|
    {_rows_md}

    **TRAP: compute aggregates as-of the prediction timestamp, not globally.**
    `avg_score` must not include scores from after the event being predicted — this is the
    aggregation version of temporal leakage from the validation notebook.

    In fraud detection: velocity features (transactions-per-hour, distinct-merchants-per-week,
    amount z-score vs user history) follow exactly this pattern — and they dominate feature
    importance. Moving from raw transaction features to behavioral aggregates is where the
    signal lives.
    """)
    return


@app.cell
def interaction_and_text_features(mo):
    mo.md("""
    ### Interaction Features and Text Features

    **Interaction features** — combinations that capture relationships linear models miss:
    ```python
    df['salary_per_year_exp']  = df['salary'] / (df['years_experience'] + 1)
    df['skills_match_ratio']   = df['skills_matched'] / (df['skills_required'] + 1)
    df['is_remote_and_senior'] = (
        (df['city'] == 'Remote') & (df['level'].isin(['Senior', 'Lead']))
    ).astype(int)
    df['over_qualified']       = (df['years_experience'] > 8) & (df['level'] == 'Junior')
    ```

    Tree models discover interactions automatically (they split on combinations). Linear models
    need them explicit. If your linear model underperforms XGBoost, add interaction features first.

    ---

    **Text features** — complement embeddings with structural signals:
    ```python
    df['description_length']   = df['description'].str.len()
    df['word_count']           = df['description'].str.split().str.len()
    df['has_salary_mentioned'] = df['description'].str.contains(r'\\$[\\d,]+', regex=True).astype(int)
    df['num_requirements']     = df['description'].str.count(
                                     r'(?:require|must have|minimum)', flags=re.IGNORECASE)
    df['exclamation_count']    = df['description'].str.count('!')  # spammy JDs
    df['all_caps_ratio']       = df['description'].apply(
                                     lambda x: sum(c.isupper() for c in x) / max(len(x), 1))
    ```

    Embeddings capture semantic meaning. Hand-crafted text features capture style, format, and
    structure — signals that embeddings smooth over. Use both.
    """)
    return


# ---------------------------------------------------------------------------
# Part 5: Feature Stores
# ---------------------------------------------------------------------------

@app.cell
def part5_header(mo):
    mo.md("## Part 5: Feature Stores — Production Feature Engineering")
    return


@app.cell
def feature_store_why(mo):
    mo.md("""
    ### Why Feature Stores Exist

    **The problem:** features computed one way in a training notebook, slightly differently in
    production → training-serving skew → silent model degradation. No errors, no crashes — just
    quietly wrong predictions.

    **Feature store** = a central system that computes, stores, and serves features consistently
    for both training and real-time inference.

    **What it guarantees:**
    - Same feature computation logic for training AND serving (one code path)
    - Point-in-time correctness (no future leakage in training features)
    - Low-latency serving for real-time inference (< 10ms P99 from an online store)
    - Feature discovery and reuse across teams — no duplicate computation
    """)
    return


@app.cell
def feature_store_arch(mo):
    mo.md("""
    ### Feature Store Architecture

    ```
    [Raw Data Sources]
    Databases · Event streams · Third-party APIs
               |
               ▼
    [Feature Computation Layer]   ← SAME transformation code for both paths
          /              \\
         ▼                ▼
    [Offline Store]    [Online Store]
    Parquet / Delta    Redis / DynamoDB / Bigtable
    Historical data    Latest feature values per entity
    Point-in-time      Key-value lookup
    joins              < 10ms P99
         |                |
         ▼                ▼
    Training jobs    Real-time inference
    Batch scoring    (model + features fetched at request time)
    ```

    **Training path — point-in-time join:**
    ```python
    # For each training label at time T, get features available BEFORE T
    training_df = feature_store.get_historical_features(
        entity_rows   = training_labels,          # [(user_id, event_timestamp), ...]
        feature_refs  = ["user_stats:avg_score",
                         "user_stats:days_active"],
    )
    # The store handles: no future leakage, correct feature versions, time-travel queries
    ```

    **Serving path — online lookup:**
    ```python
    # At inference time: fetch current precomputed features in < 10ms
    features = feature_store.get_online_features(
        entity_rows  = [{"user_id": "u123"}],
        feature_refs = ["user_stats:avg_score",
                        "user_stats:days_active"],
    )
    prediction = model.predict(features)
    ```

    Training and serving use the same feature definitions → skew is impossible by construction.
    """)
    return


@app.cell
def feature_store_tools(mo):
    mo.md("""
    ### Feature Store Tools

    | Tool | Type | Best for |
    |------|------|----------|
    | **Feast** | Open source | Small-medium teams, self-hosted, Kubernetes |
    | **Tecton** | Managed SaaS | Enterprise, real-time feature pipelines, strict SLAs |
    | **Databricks Feature Store** | Managed | Databricks shops, Unity Catalog integration |
    | **Vertex AI Feature Store** | Managed | GCP shops, tight BigQuery / Vertex AI integration |
    | **Hopsworks** | Open source / managed | EU-compliant, on-prem, HopsML stack |
    | **Custom: Redis + Spark/Polars** | DIY | Full control, specific latency requirements |

    For most teams starting out: **Feast** (open source, battle-tested, Kubernetes-native).
    For enterprise with real-time requirements: **Tecton**. Custom Redis + Polars is common in
    fintech where latency SLAs are single-digit milliseconds.
    """)
    return


@app.cell
def my_projects_mapping(mo):
    mo.md("""
    ### My Projects — Feature Store Patterns in the Wild

    **Canopy (job search assistant):**
    - No formal feature store (portfolio project), but the same pattern applies.
    - Job features (embeddings, metadata, scraped fields) are computed during scraping and
      stored in SQLite — that's a rudimentary offline store.
    - Productionizing: Feast for offline features (historical match scores by user/job pair),
      Redis for real-time serving of pre-computed job ranking scores.

    **Backtesting engine (Polars + rolling indicators):**
    - Rolling SMA, RSI, and Bollinger Bands are features computed with Polars in a streaming
      pass. The same code that produces historical features for backtesting would compute
      live features for a trading system.
    - This is structurally identical to a feature store: same code path for offline (backtest)
      and online (live trading), point-in-time correctness enforced by the walk-forward split.

    > **Interview framing:** "My backtesting engine's indicator computation pipeline is
    > structurally identical to a feature store — same code path for historical analysis and
    > live trading, point-in-time correctness enforced by the walk-forward framework. I
    > understand the problem feature stores solve because I've implemented the solution manually."
    """)
    return


# ---------------------------------------------------------------------------
# Anti-Patterns, Flashcards, Interview
# ---------------------------------------------------------------------------

@app.cell
def anti_patterns(mo):
    mo.md("""
    ## Feature Engineering Anti-Patterns

    | Anti-Pattern | Why it's bad | Fix |
    |-------------|-------------|-----|
    | Fit scaler/imputer on full data | Leakage: test statistics contaminate train params | `fit` on train only |
    | Target encoding without CV folds | Massive leakage — target embedded in encoding | Encode inside each CV fold |
    | Features from the future | Temporal leakage, wildly optimistic metrics | Point-in-time computation |
    | One-hot encoding 10K categories | Curse of dimensionality, 10K sparse columns | Target encoding or embeddings |
    | Dropping all rows with missing data | Biased dataset when MNAR | Impute + missingness indicator |
    | Scaling binary / one-hot features | Unnecessary, hurts interpretability | Only scale continuous features |
    | Different preprocessing in training vs serving | Training-serving skew, silent degradation | sklearn Pipeline or feature store |
    | `pd.get_dummies()` at inference time | New categories → column count mismatch → crash | `OneHotEncoder(handle_unknown='ignore')` |
    | Engineered features without domain knowledge | Noise, overfitting to spurious correlations | EDA first, talk to domain experts |
    | Aggregating without point-in-time correctness | Future aggregates in training = leakage | Compute as-of each event timestamp |
    """)
    return


@app.cell
def flashcards(mo):
    mo.md("""
    ## Flashcard Summary

    **"How do you handle missing data?"**
    > Depends on % missing, mechanism (MCAR/MAR/MNAR), and model. Default: median impute +
    > missingness indicator column. XGBoost handles `NaN` natively — no imputation needed for
    > tree models.

    **"MCAR vs MAR vs MNAR?"**
    > MCAR: random, no pattern. MAR: depends on observed features (contract jobs skip
    > `skills_required`). MNAR: depends on the missing value itself (high earners hide salary).
    > MNAR is hardest — dropping rows biases the dataset.

    **"When to one-hot vs target encode?"**
    > One-hot for <20 unique values. Target encode for high cardinality (with CV folds to
    > prevent leakage). Frequency encode as a leakage-safe alternative to target encoding.

    **"Do you need to scale for XGBoost?"**
    > No. Trees are scale-invariant — splits work at any scale. Scale for linear models, KNN,
    > SVM, neural nets, and PCA.

    **"StandardScaler vs RobustScaler?"**
    > StandardScaler for roughly Gaussian data. RobustScaler when outliers are present — uses
    > median and IQR instead of mean and std, so extreme values don't distort the scaling.

    **"What's a feature store?"**
    > Centralized system that computes, stores, and serves features consistently for both
    > training and real-time inference. Prevents training-serving skew. Provides point-in-time
    > correctness for training data.

    **"Why sklearn Pipeline?"**
    > Chains preprocessing + model into one object. Fits preprocessors on train only within
    > each CV fold (no leakage possible). One object to save and deploy to production.

    **"Cyclical encoding?"**
    > Sin/cos pair for periodic features (hour, day of week, month). Without it, hour 23 and
    > hour 0 are maximally distant — which is wrong, they're adjacent. Always encode in pairs.

    **"Training-serving skew?"**
    > Features computed differently in training vs production. Causes silent model degradation —
    > no error, just wrong predictions. Fix: same code path for both (sklearn Pipeline or
    > feature store).

    **"Linear models vs trees on raw categorical encoding?"**
    > Label encoding (integers) implies a false ordinal relationship for linear models — the
    > model treats "Dallas=2" as greater than "Austin=1". One-hot for linear models. Trees don't
    > care about the implied order; they learn the right splits regardless.
    """)
    return


@app.cell
def interview_tips(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    **"Walk me through your feature engineering process."**

    > "I start with EDA to understand distributions, cardinality, and missing patterns. Then
    > in sequence: handle missing data (median impute + missingness indicator, or pass NaN
    > directly to tree models), encode categoricals (one-hot for <20 values, target encoding
    > within CV folds for high cardinality), scale numerics (StandardScaler or RobustScaler
    > depending on outliers), and create domain-specific features (temporal decomposition,
    > behavioral aggregations, interaction terms). Everything goes in a sklearn Pipeline — one
    > object for training, one for production, no leakage possible."

    ---

    **"How do you prevent feature leakage?"**

    > "Three rules: fit preprocessors on train only — Pipeline enforces this automatically
    > within each CV fold. Compute features as-of the prediction timestamp — no future
    > aggregates, no rolling windows that look forward. Target encode within CV folds —
    > never on full data. I have a whole notebook on leakage that covers every failure mode:
    > data leakage, temporal leakage, and group leakage."

    ---

    **"Feature stores — have you used one?"**

    > "Not a managed product directly, but my backtesting engine implements the same core
    > pattern: a rolling indicator computation pipeline (Polars) that produces identical
    > features for historical backtesting and would serve a live trading system. Same code
    > path, point-in-time correctness enforced by the walk-forward framework. I understand
    > the problem feature stores solve because I've implemented the solution manually."

    ---

    **"Most impactful feature engineering you've done?"**

    > "In my fraud detection system design, the user behavioral velocity features —
    > transactions-per-hour, distinct-merchants-per-week, amount z-score vs user history —
    > were more impactful than any model change. Moving from raw transaction features to
    > behavioral aggregates is where the real signal lives. That's also the practical answer
    > to 'why does feature engineering matter more than model selection': a better model on
    > raw features loses to a simple model on well-engineered features every time."

    ---

    *Applied ML is 80% feature engineering. Master this, and the model is almost secondary.*
    """)
    return


if __name__ == "__main__":
    app.run()
