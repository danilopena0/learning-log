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
    # Transformer Intuition — Attention, Embeddings, Positional Encoding (Math + Why)

    | Field | Value |
    |-------|-------|
    | Date  | 2026-04-16 |
    | Track | ML Theory |
    | Time  | 60 min |
    | Topics | Self-Attention · Q/K/V · Scaled Dot-Product · Multi-Head · Positional Encoding · Transformer Block |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Why Transformers Replaced RNNs

    ### The RNN Problem

    RNNs process tokens one at a time, left to right. Three fatal flaws:

    1. **Sequential processing → can't parallelize.** Training on a 1,000-token sequence requires
       1,000 serial steps. GPUs are built for massive parallelism — RNNs can't exploit them efficiently.
    2. **Vanishing gradients over long sequences.** Gradients flow backward through each timestep.
       After 100+ steps, the gradient from the last token barely reaches the first — long-range
       dependencies can't be learned.
    3. **Hidden state bottleneck.** Everything the model has read is compressed into a single
       fixed-size vector. "Squeeze an entire paragraph into 512 numbers." Information gets lost.

    ### The Transformer Insight

    Replace recurrence with **attention**. Every token can directly attend to every other token —
    no intermediary chain, no bottleneck, no sequential dependency.

    Three innovations that made this work:

    | Innovation | What it solves |
    |-----------|---------------|
    | **Self-attention** | Direct token-to-token connections. No more hidden-state bottleneck. |
    | **Positional encoding** | We lost sequence order by going parallel — this adds it back. |
    | **Multi-head attention** | One attention head = one perspective. Multiple heads = multiple simultaneous perspectives. |

    > **Vaswani et al. (2017) — "Attention Is All You Need"** — one of the most-cited ML papers ever.
    > The title is literally true: the architecture drops convolutions and recurrence entirely.
    > Name-drop this paper in interviews.

    **Why this matters:** Transformers underpin GPT, BERT, Claude, Gemini — every modern LLM.
    Understanding attention *is* understanding modern AI.
    """)
    return


@app.cell
def _(mo):
    mo.md("## Part 1: Embeddings — Tokens to Vectors")
    return


@app.cell
def _(mo):
    mo.md("""
    ### The Problem: Neural Networks Need Numbers

    Neural networks speak math. Words are not math. How do we bridge this?

    **Naive approach: one-hot encoding.**
    For a vocabulary of 50,000 words, create a 50,000-dimensional vector with a single 1 at the
    word's index and 0 everywhere else.

    Why this is bad:
    - **Huge dimensionality:** 50K floats per token, almost all zeros — wasteful.
    - **No notion of similarity:** the cosine distance between "cat" and "kitten" equals the cosine
      distance between "cat" and "spaceship". All one-hot vectors are orthogonal — the geometry is meaningless.

    **Better approach: learned dense embeddings.**
    Map each token ID to a learned vector of dimension `d_model` (typically 512, 768, or 4096).

    - **Geometry encodes meaning:** similar words end up close in vector space through training.
      Classic example: `king − man + woman ≈ queen` — arithmetic on meaning.
    - **Trained jointly** with the rest of the network — embeddings learn what's useful for the task.
    - **The embedding matrix:** shape `(vocab_size, d_model)`. Lookup is just matrix row indexing.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Embedding Math

    | Quantity | Shape | Meaning |
    |---------|-------|---------|
    | Embedding matrix `E` | `(vocab_size, d_model)` | All learned token vectors |
    | Single token lookup | `E[token_id]` → `(d_model,)` | One word's representation |
    | Sequence of `n` tokens | `E[input_ids]` → `(n, d_model)` | Full sequence as a matrix |

    ```
    E[token_id]  →  vector of shape (d_model,)

    For input = [token_0, token_1, ..., token_{n-1}]:
    embeddings = E[input_ids]  →  shape (n, d_model)
    ```

    **Parameter count:** `vocab_size × d_model`
    - GPT-2 (small): 50,257 × 768 = **38.6M parameters** in embeddings alone
    - LLaMA-3 8B: 128,256 × 4,096 = **525M parameters** in embeddings alone

    Vocabulary size is a real engineering trade-off: larger vocab = better tokenization efficiency
    (fewer tokens per sentence) but more parameters to store and initialize.
    """)
    return


@app.cell
def embeddings_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _rng = _np.random.default_rng(42)

    # Tiny vocabulary: 7 tokens
    _vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran"]
    _vocab_size = len(_vocab)
    _d_model = 4  # tiny d_model for visualization

    # Random embedding matrix — shape: (vocab_size, d_model)
    # In a real model, these weights are learned during training
    _E = _rng.standard_normal((_vocab_size, _d_model))

    # Lookup for sentence: "the cat sat" → token IDs [0, 1, 2]
    _sentence = ["the", "cat", "sat"]
    _token_ids = [_vocab.index(w) for w in _sentence]
    _embeddings = _E[_token_ids]  # shape: (3, 4)

    print("Embedding matrix E — shape:", _E.shape)
    print()
    print("Sentence: 'the cat sat'")
    print("Token IDs:", _token_ids)
    print("E[token_ids] — shape:", _embeddings.shape)
    print()
    for _word, _tid in zip(_sentence, _token_ids):
        print(f"  E[{_tid}] = '{_word}':  {_np.round(_E[_tid], 3)}")

    # PCA to 2D for visualization — implemented in pure numpy
    def _pca_2d(X):
        X_c = X - X.mean(axis=0)
        cov = (X_c.T @ X_c) / max(len(X) - 1, 1)
        eigenvalues, eigenvectors = _np.linalg.eigh(cov)
        idx = _np.argsort(eigenvalues)[::-1]
        return (X_c @ eigenvectors[:, idx[:2]]).real

    _coords = _pca_2d(_E)

    _fig, _ax = _plt.subplots(figsize=(8, 6))
    _colors = ["#3498DB", "#E74C3C", "#27AE60", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22"]

    for _i, (_word, _coord) in enumerate(zip(_vocab, _coords)):
        _ax.scatter(*_coord, color=_colors[_i], s=200, zorder=5, edgecolors="white", linewidth=1.5)
        _ax.annotate(_word, _coord, fontsize=13, fontweight="bold",
                     xytext=(7, 5), textcoords="offset points", color=_colors[_i])

    _ax.set_title("Word Embeddings in 2D (PCA of random 4-dim vectors)\n"
                  "Random init → random geometry. Meaning only emerges through training.",
                  fontsize=12, fontweight="bold")
    _ax.set_xlabel("PC 1", fontsize=11)
    _ax.set_ylabel("PC 2", fontsize=11)
    _ax.grid(True, alpha=0.3)
    _ax.axhline(0, color="gray", lw=0.5)
    _ax.axvline(0, color="gray", lw=0.5)
    _fig.tight_layout()
    return (_fig,)


@app.cell
def _(mo):
    mo.md("## Part 2: Attention — The Core Mechanism")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Why Attention Exists

    When processing a token, how do we let it *look at* other tokens?

    - **In RNNs:** hidden state passes left-to-right. Distant tokens get "forgotten" as
      gradients and information decay over distance.
    - **In CNNs:** fixed receptive field. Long-range dependencies require stacking many layers,
      and even then the path length grows.

    **Attention's answer:** for each token, compute a *weighted average* of ALL other tokens,
    where the weights depend on relevance. Every token has a direct channel to every other token.

    > **Intuition:** in "The cat sat on the mat because **it** was tired" — when processing "it",
    > attention lets "it" look at "cat" (high weight) more than "mat" (low weight). The model
    > learns which connections matter, regardless of distance.

    This eliminates the hidden-state bottleneck entirely: information flows directly, not through
    a chain of intermediaries.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### The Q, K, V Abstraction

    Each token plays three simultaneous roles:

    | Role | Symbol | Intuition | Shape |
    |------|--------|-----------|-------|
    | **Query** | Q | "What am I looking for?" | `(n, d_k)` |
    | **Key** | K | "What do I contain / what can I be found by?" | `(n, d_k)` |
    | **Value** | V | "What information do I actually provide?" | `(n, d_v)` |

    **Database analogy:** query = your search term, keys = the index, values = the records.
    Attention is a "soft lookup" — instead of one exact match, you get a weighted blend of
    all records, weighted by query-key similarity.

    **Why separate Q, K, V instead of just using embeddings directly?**

    Learned projections let the model separate three distinct representations:
    - What you're *asking* (Q) can differ from what you *are* (K) — same word, different role
    - What you're *identified by* (K) can differ from what you *give* (V) — a pronoun might be
      identified syntactically but provide its antecedent's semantics
    - Projections = the model learns to specialize "asking" vs. "matching" vs. "content" representations

    Each is a learned linear projection of the input X:
    ```
    Q = X @ W_Q    # W_Q shape: (d_model, d_k)
    K = X @ W_K    # W_K shape: (d_model, d_k)
    V = X @ W_V    # W_V shape: (d_model, d_v)
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### The Attention Formula

    **Scaled dot-product attention:**

    ```
    Attention(Q, K, V) = softmax( Q @ Kᵀ / √d_k ) @ V
    ```

    Breaking it down term by term:

    | Term | Shape | What it does | Why |
    |------|-------|-------------|-----|
    | `Q @ Kᵀ` | `(n, n)` | Dot product of every query with every key | Measures pairwise relevance |
    | `/ √d_k` | `(n, n)` | Scale down by √(key dimension) | Prevent softmax saturation (see below) |
    | `softmax(...)` | `(n, n)` | Normalize each row to sum to 1 | Convert raw scores → probability distribution |
    | `@ V` | `(n, d_v)` | Weighted sum of value vectors | Each token gets a weighted blend of all token values |

    **Output:** `(n, d_v)` — for each of the n tokens, a new representation that is a
    content-weighted blend of all other tokens' values.

    The n×n attention matrix is the heart of the computation.
    Entry `[i, j]` = "how much should token i attend to token j?"
    This is where the **O(n²) cost** lives.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Why Divide by √d_k — The Variance Argument

    > This is one of the most common transformer interview questions. Memorize this derivation.

    **Setup:** assume Q and K entries are i.i.d. with mean 0, variance 1 (typical after init or layer norm).

    **The dot product** `q · k = Σᵢ qᵢ kᵢ` is a sum of `d_k` independent terms.

    By variance of a sum of independent variables:
    ```
    Var(q · k) = Var(q₁k₁) + Var(q₂k₂) + ... + Var(q_{d_k} k_{d_k})
               = d_k × Var(qᵢkᵢ)
               = d_k × 1   (since Var(qᵢkᵢ) = E[qᵢ²]E[kᵢ²] - 0 = 1)
    ```

    So **std(q · k) = √d_k**. For d_k = 64, scores have magnitude ~8.

    **Why this breaks softmax:**

    | Scale | Example scores | softmax output | Gradient health |
    |-------|---------------|----------------|-----------------|
    | d_k = 1 | `[0.1, -0.2, 0.8, -0.1]` | `[0.24, 0.19, 0.43, 0.23]` | Well-distributed ✓ |
    | d_k = 64 (unscaled) | `[0.8, -1.6, 6.4, -0.8]` | `[0.001, 0.000, 0.998, 0.001]` | Near-zero gradients ✗ |
    | d_k = 64 (scaled by √64=8) | `[0.1, -0.2, 0.8, -0.1]` | `[0.24, 0.19, 0.43, 0.23]` | Well-distributed ✓ |

    When softmax is nearly one-hot, gradients for all non-maximum entries approach zero — the
    model can't learn from most of its attention positions. Dividing by √d_k brings variance
    back to 1, keeping the softmax in a trainable temperature regime.

    > **30-second interview answer:** "Dot products of d_k-dimensional vectors have variance d_k.
    > Large values saturate softmax into near-zero gradients. Dividing by √d_k restores variance
    > to 1 and keeps training stable."
    """)
    return


@app.cell
def attention_impl():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _rng = _np.random.default_rng(0)

    # ── Dimensions ──────────────────────────────────────────────────────────────
    _n       = 3   # sequence: "the", "cat", "sat"
    _d_model = 4   # embedding dimension (tiny for demonstration)
    _d_k     = 4   # query/key dimension
    _d_v     = 4   # value dimension

    # Input: 3 token embeddings, each of dim d_model   shape: (n, d_model)
    _X = _rng.standard_normal((_n, _d_model))

    # Learned projection matrices — random here, trained in real models
    _W_Q = _rng.standard_normal((_d_model, _d_k))
    _W_K = _rng.standard_normal((_d_model, _d_k))
    _W_V = _rng.standard_normal((_d_model, _d_v))

    # ── Numerically stable softmax ───────────────────────────────────────────────
    def _softmax(x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)   # subtract max for stability
        ex = _np.exp(x)
        return ex / ex.sum(axis=axis, keepdims=True)

    # ── Scaled dot-product attention — step by step ──────────────────────────────
    print("=" * 58)
    print("Scaled Dot-Product Attention — Shape Trace")
    print("=" * 58)
    print(f"\nInput X ('the cat sat'):  shape {_X.shape}  → (n={_n}, d_model={_d_model})")
    print(f"W_Q, W_K:  shape ({_d_model}, {_d_k})")
    print(f"W_V:       shape ({_d_model}, {_d_v})\n")

    _Q = _X @ _W_Q                          # (n, d_k)
    _K = _X @ _W_K                          # (n, d_k)
    _V = _X @ _W_V                          # (n, d_v)
    print(f"Step 1 — Projections:")
    print(f"  Q = X @ W_Q     shape: {_Q.shape}")
    print(f"  K = X @ W_K     shape: {_K.shape}")
    print(f"  V = X @ W_V     shape: {_V.shape}")

    _scores = _Q @ _K.T                     # (n, n)
    print(f"\nStep 2 — Raw similarity scores:")
    print(f"  scores = Q @ Kᵀ  shape: {_scores.shape}  ← n×n grid of dot products")

    _scaled = _scores / _np.sqrt(_d_k)      # (n, n)
    print(f"\nStep 3 — Scaling:")
    print(f"  scaled = scores / √{_d_k}     shape: {_scaled.shape}")
    print(f"  scores std before: {_scores.std():.3f}  |  after: {_scaled.std():.3f}")

    _weights = _softmax(_scaled, axis=-1)   # (n, n), rows sum to 1
    print(f"\nStep 4 — Softmax (row-wise):")
    print(f"  weights = softmax(scaled)  shape: {_weights.shape}")
    print(f"  Row sums (must be 1.0): {_np.round(_weights.sum(axis=-1), 4)}")

    _output = _weights @ _V                 # (n, d_v)
    print(f"\nStep 5 — Weighted sum of values:")
    print(f"  output = weights @ V  shape: {_output.shape}  ← same shape as input embeddings")

    _tokens = ["the", "cat", "sat"]
    print(f"\nAttention weight matrix  (weights[i, j] = how much token i attends to token j):")
    print(f"{'':8}", end="")
    for _t in _tokens:
        print(f"{'→' + _t:>9}", end="")
    print()
    for _i, _ti in enumerate(_tokens):
        print(f"{_ti:8}", end="")
        for _j in range(_n):
            _mark = " ◀" if _j == _i else ""
            print(f"{_weights[_i, _j]:>8.3f}{_mark if _mark else '  '}", end="")
        print()
    print("         (◀ = self-attention weight)")

    # ── Attention heatmap ────────────────────────────────────────────────────────
    _fig, _ax = _plt.subplots(figsize=(6.5, 5.5))
    _im = _ax.imshow(_weights, cmap="Blues", vmin=0, vmax=1)
    _plt.colorbar(_im, ax=_ax, label="Attention weight")
    _ax.set_xticks(range(_n))
    _ax.set_yticks(range(_n))
    _ax.set_xticklabels([f"→ {t}\n(Key / Value)" if i == 1 else t for i, t in enumerate(_tokens)], fontsize=12)
    _ax.set_yticklabels([f"{t}\n(Query)" if i == 1 else t for i, t in enumerate(_tokens)], fontsize=12)
    _ax.set_xlabel("Token being attended to (Key)", fontsize=12)
    _ax.set_ylabel("Attending token (Query)", fontsize=12)
    _ax.set_title("Attention Weights\nweights[i, j] = how much token i attends to token j",
                  fontsize=12, fontweight="bold")
    for _i in range(_n):
        for _j in range(_n):
            _ax.text(_j, _i, f"{_weights[_i, _j]:.3f}",
                     ha="center", va="center", fontsize=13,
                     color="white" if _weights[_i, _j] > 0.6 else "black",
                     fontweight="bold")
    _fig.tight_layout()
    return (_fig,)


@app.cell
def _(mo):
    mo.md("## Part 3: Multi-Head Attention")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Why Multi-Head Attention?

    One attention head = one "perspective" on token relationships.

    But tokens have *multiple types* of relationships simultaneously:

    | Relationship type | Example |
    |------------------|---------|
    | **Syntactic** | subject-verb agreement: "the dogs bark" |
    | **Semantic** | synonymy/antonymy: "happy" ↔ "joyful" |
    | **Referential** | pronoun resolution: "it" → "the cat" |
    | **Positional** | nearby tokens tend to be semantically related |

    One head can only optimize one set of W_Q, W_K, W_V — one way of asking and matching.

    **Multi-head solution:** run `h` independent attention mechanisms in parallel, each with
    their own W_Q, W_K, W_V. Each head can specialize in different relationship types.
    Outputs are **concatenated** then projected back to d_model:

    ```
    MultiHead(X) = Concat(head_1, head_2, ..., head_h) @ W_O
    ```

    Empirically, different heads *do* specialize — interpretability research shows some heads
    track syntax, others track coreference, others track proximity. This specialization
    is learned, not designed.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Multi-Head Math

    For `h` heads, each head operates in reduced dimension `d_k = d_model / h`:

    ```
    For each head i ∈ {1, ..., h}:
        head_i = Attention(X @ W_Q_i,  X @ W_K_i,  X @ W_V_i)
                          (n, d_k)      (n, d_k)     (n, d_v)

    MultiHead(X) = Concat(head_1, ..., head_h) @ W_O
                   (n, d_model) × (d_model, d_model) → (n, d_model)
    ```

    | Tensor | Shape | Notes |
    |--------|-------|-------|
    | Each `W_Q_i`, `W_K_i`, `W_V_i` | `(d_model, d_model/h)` | Per-head projections |
    | Each `head_i` | `(n, d_model/h)` | One head's output |
    | `Concat(heads)` | `(n, d_model)` | Reassembled across head dim |
    | `W_O` | `(d_model, d_model)` | Final projection |
    | **Output** | `(n, d_model)` | Same shape as input — clean! |

    **Total parameters per attention layer: `4 × d_model²`**
    - Q, K, V projections: `3 × [h × (d_model × d_model/h)] = 3 × d_model²`
    - Output projection W_O: `d_model²`
    - **Total = 4 × d_model² — independent of the number of heads h!**

    GPT-3 uses h=96 heads, d_model=12,288, d_k=128 per head.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Why Split d_model Across Heads (The Budget Argument)

    You *could* give each head the full d_model dimensions:
    - Each of h heads uses `W_Q` of shape `(d_model, d_model)`
    - Total params: `h × 3 × d_model²` — **grows linearly with h**

    Instead, split: each head gets `d_model / h` dimensions:
    - Total params: `3 × d_model²` — **constant regardless of h**

    **The intuition:** multi-head attention is about dividing a fixed parameter budget across
    multiple specialized perspectives, not expanding the budget. You get h different relationship
    detectors for the same cost as one.

    The W_O projection at the end allows all heads to recombine their specialized representations
    into a single unified d_model-dimensional token representation.

    > **Interview answer:** "Multi-head doesn't increase parameter count because each head uses
    > d_model/h dimensions. Total stays at 4 × d_model² (3 for Q,K,V projections + 1 for W_O),
    > independent of h."
    """)
    return


@app.cell
def multihead_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _rng = _np.random.default_rng(7)

    _n       = 3    # "the", "cat", "sat"
    _d_model = 8    # slightly larger so d_k = d_model/h = 4 per head
    _h       = 2    # two heads
    _d_k     = _d_model // _h   # 4 per head

    _X = _rng.standard_normal((_n, _d_model))
    _tokens = ["the", "cat", "sat"]

    def _softmax(x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)
        ex = _np.exp(x)
        return ex / ex.sum(axis=axis, keepdims=True)

    def _single_head_attention(X, W_Q, W_K, W_V, d_k):
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        weights = _softmax((Q @ K.T) / _np.sqrt(d_k), axis=-1)
        return weights @ V, weights

    # Run h=2 heads with DIFFERENT projection matrices → different attention patterns
    _head_outputs = []
    _head_weights = []
    for _head_idx in range(_h):
        _W_Q = _rng.standard_normal((_d_model, _d_k))
        _W_K = _rng.standard_normal((_d_model, _d_k))
        _W_V = _rng.standard_normal((_d_model, _d_k))
        _out, _w = _single_head_attention(_X, _W_Q, _W_K, _W_V, _d_k)
        _head_outputs.append(_out)
        _head_weights.append(_w)

    # Concatenate and project
    _W_O = _rng.standard_normal((_d_model, _d_model))
    _concat = _np.concatenate(_head_outputs, axis=-1)  # (n, d_model)
    _mha_output = _concat @ _W_O                        # (n, d_model)

    print("Multi-Head Attention — shape trace")
    print(f"  Input X:              {_X.shape}")
    print(f"  Each head output:     ({_n}, {_d_k})")
    print(f"  Concat(head_1, h_2):  {_concat.shape}")
    print(f"  MHA output = @W_O:    {_mha_output.shape}  ← same as input")
    print()
    for _i, _w in enumerate(_head_weights):
        print(f"  Head {_i+1} attention weights (row sums): {_np.round(_w.sum(axis=-1), 3)}")

    # Visualize: side-by-side heatmaps for both heads
    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 5))
    for _i, (_ax, _hw) in enumerate(zip(_axes, _head_weights)):
        _im = _ax.imshow(_hw, cmap="Blues", vmin=0, vmax=1)
        _plt.colorbar(_im, ax=_ax, label="Attention weight")
        _ax.set_xticks(range(_n))
        _ax.set_yticks(range(_n))
        _ax.set_xticklabels(_tokens, fontsize=13)
        _ax.set_yticklabels(_tokens, fontsize=13)
        _ax.set_xlabel("Key / Value token", fontsize=11)
        _ax.set_ylabel("Query token", fontsize=11)
        _ax.set_title(f"Head {_i + 1}\n(different W_Q, W_K, W_V → different pattern)",
                      fontsize=12, fontweight="bold")
        for _r in range(_n):
            for _c in range(_n):
                _ax.text(_c, _r, f"{_hw[_r, _c]:.2f}",
                         ha="center", va="center", fontsize=13,
                         color="white" if _hw[_r, _c] > 0.6 else "black",
                         fontweight="bold")

    _fig.suptitle("Multi-Head Attention: Same Input, Different Projections → Different Patterns\n"
                  "Each head learns to focus on different token relationships",
                  fontsize=12, fontweight="bold")
    _fig.tight_layout()
    return (_fig,)


@app.cell
def _(mo):
    mo.md("## Part 4: Positional Encoding")
    return


@app.cell
def _(mo):
    mo.md("""
    ### The Problem: Attention Is Permutation-Invariant

    Here's the subtle issue with attention: it computes weighted sums based on *content*, not position.
    There is no notion of "before" or "after" in the formula.

    **Concrete proof:** "the cat sat" and "sat cat the" produce identical attention outputs
    (just with rows permuted). The model literally cannot tell the difference.

    But word order matters critically:
    - "Dog bites man" ≠ "Man bites dog"
    - "Not happy" ≠ "Happy not"

    **Solution:** inject position information into the embeddings *before* attention sees them,
    so each token's vector carries a unique fingerprint of its position.

    ```
    X_input = Embedding(token_ids) + PositionalEncoding(positions)
              (n, d_model)          (n, d_model)
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Sinusoidal Positional Encoding (the original, 2017)

    For position `pos` (0-indexed) and embedding dimension index `i`:

    ```
    PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
    ```

    Even dimensions use sine, odd dimensions use cosine. Each dimension pair `(2i, 2i+1)` gets
    a different frequency determined by `10000^(2i/d_model)`.

    **Why sinusoidal? Four reasons:**

    1. **Unique fingerprint per position:** the combination of different frequencies across dimensions
       means no two positions have the same encoding vector. Like a continuous binary counter.

    2. **Relative positions are linearly recoverable:** by the angle addition formula,
       `PE(pos + k)` is a *linear function* of `PE(pos)`. The model can learn to compute
       relative offsets — "this token is 3 positions after that one."

    3. **Extrapolates beyond training length:** the formula is deterministic and continuous —
       works at any position, including positions never seen during training.

    4. **Zero learned parameters:** purely computed at startup, no training required.

    **Then:** add PE to embeddings before the first attention layer.
    Shape of PE: `(max_seq_len, d_model)`. Slice `PE[:n]` for sequences of length n.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Why ADD Positional Encoding Instead of CONCATENATE?

    **Concatenation** would produce vectors of size `d_model + d_model = 2 × d_model`, forcing
    all downstream weight matrices to double in size. Parameter count doubles.

    **Addition** preserves the `d_model` dimension. The entire rest of the architecture is unchanged.

    **Why does adding work?**

    In high-dimensional space, position signals and semantic signals tend to occupy *different
    directions*. The embedding dimensions that encode "catness" are largely orthogonal to the
    dimensions encoding "position 3". The model can learn to read them independently — they
    coexist without clobbering each other.

    This is somewhat hand-wavy (the true justification is empirical), but it's interview-acceptable.
    The honest answer: addition works just as well as concatenation at the same parameter budget,
    and it's simpler.

    > "Addition preserves dimension, saves parameters, and works empirically — position and
    > semantic signals appear to occupy different subspaces in high-dimensional space."
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Modern Positional Encoding Alternatives

    | Method | How it works | Used in | Key advantage |
    |--------|-------------|---------|---------------|
    | **Sinusoidal** (2017) | Fixed sin/cos patterns added to embeddings | Original Transformer | No learned params, extrapolates |
    | **Learned PE** | Trainable `(max_seq_len, d_model)` weight matrix | BERT, GPT-2/3 | Flexible, adapts to training data distribution |
    | **RoPE** (2021) | Rotate Q and K vectors by angle proportional to position | LLaMA, Mistral, GPT-NeoX | Relative positions only, generalizes to longer contexts |
    | **ALiBi** (2021) | Add a linear position-dependent bias to attention scores | BLOOM, some efficient models | No embedding overhead, very long context |

    **RoPE in more detail (for serious interviews):**

    Instead of adding position to embeddings, RoPE encodes position by *rotating* the Q and K
    vectors before their dot product:
    ```
    Q_rotated = RoPE(Q, pos_q)
    K_rotated = RoPE(K, pos_k)
    score = Q_rotated · K_rotated
    ```
    The dot product naturally depends only on the *relative* position `(pos_q - pos_k)`, not
    absolute positions. This property emerges from the rotation group: rotating both vectors by
    their respective angles, the relative rotation is all that survives in the inner product.

    **What to say in interviews:** know sinusoidal (the original — understand the formula and
    four reasons) and name RoPE as the modern standard with its key property: rotation-based
    relative position encoding, better context length generalization.
    """)
    return


@app.cell
def positional_demo():
    import numpy as _np
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    def _sinusoidal_pe(max_seq_len, d_model):
        PE = _np.zeros((max_seq_len, d_model))
        pos = _np.arange(max_seq_len)[:, _np.newaxis]      # (seq, 1)
        i   = _np.arange(0, d_model, 2)[_np.newaxis, :]   # (1, d_model/2)
        div = _np.power(10000.0, i / d_model)               # (1, d_model/2)
        PE[:, 0::2] = _np.sin(pos / div)                   # even dims → sin
        PE[:, 1::2] = _np.cos(pos / div)                   # odd dims  → cos
        return PE

    _seq_len = 50
    _d_model = 64
    _PE = _sinusoidal_pe(_seq_len, _d_model)

    print(f"Positional encoding shape: {_PE.shape}  → (seq_len, d_model)")
    print(f"PE value range: [{_PE.min():.3f}, {_PE.max():.3f}]  (sin/cos → always in [-1, 1])")
    print(f"\nPE for position 0  (dims 0-7): {_np.round(_PE[0,  :8], 3)}")
    print(f"PE for position 1  (dims 0-7): {_np.round(_PE[1,  :8], 3)}")
    print(f"PE for position 10 (dims 0-7): {_np.round(_PE[10, :8], 3)}")
    print(f"\nTwo positions are equal? pos 0 == pos 1: {_np.allclose(_PE[0], _PE[1])}")

    _fig, _axes = _plt.subplots(1, 2, figsize=(15, 5))

    # Left — Full PE heatmap: x=position, y=dimension
    _ax1 = _axes[0]
    _im1 = _ax1.imshow(_PE.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    _plt.colorbar(_im1, ax=_ax1, label="Encoding value (sin/cos)")
    _ax1.set_xlabel("Position (token index)", fontsize=12)
    _ax1.set_ylabel("Embedding dimension", fontsize=12)
    _ax1.set_title("Sinusoidal Positional Encoding\n(50 positions × 64 dimensions)",
                   fontsize=12, fontweight="bold")
    _ax1.annotate("Low dims: high frequency\n(complete cycles in ~6 positions)",
                  xy=(12, 3), xytext=(18, 12), fontsize=9, color="black",
                  arrowprops=dict(arrowstyle="->", color="black"))
    _ax1.annotate("High dims: low frequency\n(barely oscillate across 50 positions)",
                  xy=(25, 60), xytext=(5, 52), fontsize=9, color="black",
                  arrowprops=dict(arrowstyle="->", color="black"))

    # Right — First 6 dimensions across positions (shows the frequencies)
    _ax2 = _axes[1]
    _dim_colors = ["#E74C3C", "#C0392B", "#3498DB", "#2980B9", "#27AE60", "#1E8449"]
    _dim_labels = [
        "dim 0 (sin, high freq)", "dim 1 (cos, high freq)",
        "dim 2 (sin)",            "dim 3 (cos)",
        "dim 4 (sin, lower freq)", "dim 5 (cos, lower freq)",
    ]
    for _dim_idx, (_color, _label) in enumerate(zip(_dim_colors, _dim_labels)):
        _ax2.plot(_PE[:, _dim_idx], color=_color, lw=2, label=_label)
    _ax2.set_xlabel("Position (token index)", fontsize=12)
    _ax2.set_ylabel("Encoding value", fontsize=12)
    _ax2.set_title("First 6 Dimensions Across Positions\n"
                   "Each column in the full PE is a unique position fingerprint",
                   fontsize=12, fontweight="bold")
    _ax2.legend(fontsize=9, loc="upper right")
    _ax2.grid(True, alpha=0.3)
    _ax2.axhline(0, color="gray", lw=0.5)
    _ax2.set_ylim(-1.2, 1.4)

    _fig.tight_layout()
    return (_fig,)


@app.cell
def _(mo):
    mo.md("## Part 5: Putting It Together — A Full Transformer Block")
    return


@app.cell
def _(mo):
    mo.md("""
    ### The Transformer Block

    A single transformer layer ("block") consists of four sub-components in sequence:

    ```python
    def transformer_block(x):
        # 1. Multi-head self-attention
        attn_out = multi_head_attention(x)

        # 2. Residual connection + Layer Normalization
        x = layer_norm(x + attn_out)

        # 3. Feed-Forward Network (position-wise MLP)
        ffn_out = ffn(x)            # Linear → ReLU/GELU → Linear
                                    # hidden_dim = 4 × d_model typically

        # 4. Residual connection + Layer Normalization
        x = layer_norm(x + ffn_out)

        return x   # shape: (n, d_model) — same as input
    ```

    **Why residual connections `(x + attn_out)`?**

    Without residuals, deep networks (12–96+ layers) suffer vanishing gradients — the gradient
    signal from the loss barely reaches the earliest layers. Residuals create a "gradient highway":
    `∂L/∂x = ∂L/∂(x + f(x)) × (1 + ∂f/∂x)` — the extra `+1` term ensures gradients flow
    back through the addition node unchanged, bypassing attention and FFN entirely. This is what
    makes training very deep transformers stable.

    **Why layer norm instead of batch norm?**

    - **Batch norm** computes mean/variance *across the batch* — requires a stable batch size and
      makes single-sample inference inconsistent with training.
    - **Layer norm** computes mean/variance *across features (d_model)*, per token. Batch-independent,
      identical behavior at train and inference, handles variable-length sequences naturally.

    **Why FFN after attention?**

    - **Attention** *mixes* information across tokens — each token sees every other token.
    - **FFN** processes each token *independently* with nonlinearity (no mixing).
    - Together: **mix → transform → mix → transform**. Attention handles "what to combine",
      FFN handles "how to transform" each token's representation.
    - The FFN hidden dimension is typically 4× d_model — this is where most of the model's
      "factual knowledge" is stored (mechanistic interpretability calls FFNs "memory matrices").

    **Parameter count per block:**

    | Component | Parameters |
    |-----------|------------|
    | Multi-head attention (Q, K, V, O) | `4 × d_model²` |
    | FFN (two linear layers, 4× hidden) | `2 × d_model × 4d_model = 8 × d_model²` |
    | Two layer norms | `2 × 2 × d_model ≈ 0` |
    | **Total per block** | **≈ 12 × d_model²** |

    GPT-3 check: `d_model=12288, 96 blocks → 96 × 12 × 12288² ≈ 175B` ✓
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Why Stack Many Blocks?

    A single transformer block = one round of "every token attends to every other token + FFN."
    One pass of information mixing and transformation.

    **Multiple blocks = compositional reasoning:**

    | Layer range (typical) | What tends to emerge |
    |----------------------|---------------------|
    | Early layers (1–4) | Local syntax, POS tags, short-range co-occurrence |
    | Middle layers | Semantic relationships, entity tracking |
    | Late layers | Task-specific representations, long-range discourse structure |

    Each layer builds on the representations output by the previous layer. Attention at layer 5
    sees vectors that already encode what layer 4 "noticed" — which encodes what layer 3 noticed,
    and so on. This is compositional reasoning through stacked non-linearities.

    **Scale reference:**

    | Model | Layers | d_model | Params |
    |-------|--------|---------|--------|
    | BERT-base | 12 | 768 | 110M |
    | GPT-2 large | 36 | 1280 | 774M |
    | GPT-3 | 96 | 12288 | 175B |
    | GPT-4 (estimated) | ~120 | ~25000 | ~1.8T |

    Depth enables compositional depth of reasoning. Width (d_model) enables richer per-token
    representations. Both matter, and modern scaling laws try to balance them optimally.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Encoder vs. Decoder vs. Decoder-Only

    The original 2017 transformer was an encoder-decoder (for machine translation).
    Modern LLMs have converged on decoder-only.

    | Architecture | Attention type | Key models | Best for |
    |-------------|---------------|-----------|---------|
    | **Encoder-only** | Bidirectional (all tokens attend to all) | BERT, RoBERTa | Understanding: classification, NER, semantic search |
    | **Decoder-only** | Causal / autoregressive (only past tokens) | GPT, Claude, LLaMA, Mistral | Generation: completions, chat, code |
    | **Encoder-decoder** | Encoder: bidirectional + Decoder: causal + cross-attention | T5, BART | Translation, summarization, structured generation |

    **Causal masking (used in decoder-only):**

    In the attention score matrix, set `scores[i, j] = −∞` for all `j > i` (future positions).
    After softmax, `−∞ → 0`, so token i pays zero attention to any future token. This enforces
    the autoregressive constraint: the model can only see the past.

    ```
    Causal mask (n=4, 0=attend, -∞=block):
    [[  0,  -∞,  -∞,  -∞],   ← token 0 only sees itself
     [  0,   0,  -∞,  -∞],   ← token 1 sees tokens 0,1
     [  0,   0,   0,  -∞],   ← token 2 sees tokens 0,1,2
     [  0,   0,   0,   0]]   ← token 3 sees all tokens
    ```

    **Why decoder-only dominates modern LLMs:**

    - Simpler architecture (one module, not two — no cross-attention layers to balance)
    - Can do understanding AND generation at scale
    - Naturally fits the "predict next token" pre-training objective (causal LM)
    - Scales more cleanly with a single set of architectural hyperparameters
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Flashcard Summary

    | Question | Answer |
    |---------|--------|
    | **Attention formula?** | `softmax(QKᵀ / √d_k) V` |
    | **Why divide by √d_k?** | Dot products of d_k-dim vectors have variance d_k. Large values saturate softmax → vanishing gradients. Dividing by √d_k restores variance to 1, keeps training stable. |
    | **Q, K, V intuitively?** | Q = "what I'm looking for", K = "what each token contains", V = "what information each token provides" |
    | **Why multi-head attention?** | Each head learns a different relationship type (syntactic, semantic, etc.) with the same total parameter budget: 4 × d_model², independent of h. |
    | **Why is PE needed?** | Attention is permutation-invariant — "the cat sat" and "sat cat the" produce identical outputs. PE injects position as a unique fingerprint into each token's embedding. |
    | **Why ADD PE, not concatenate?** | Preserves d_model dimension, saves parameters, empirically works equally well — position and semantic signals occupy different subspaces in high-d space. |
    | **BERT vs GPT architecturally?** | BERT = encoder-only, bidirectional attention (all tokens see all tokens). GPT = decoder-only, causal masking (each token only sees past tokens). |
    | **Why residual connections?** | Gradient highway for deep networks. The `+1` in `1 + ∂f/∂x` lets gradients flow back unchanged, enabling stable training of 96+ layer models. |
    | **Why layer norm over batch norm?** | Layer norm is per-token and batch-independent. Identical at train and inference, handles variable-length sequences, more stable for transformers. |
    | **Attention computational complexity?** | O(n²) compute and O(n²) memory in sequence length n. The fundamental bottleneck for long-context models. FlashAttention, sparse attention, and sliding-window methods address this. |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    **"Walk me through self-attention."**

    Each input token is projected into three vectors via learned weight matrices: Query, Key, and Value.
    The Query of each token is dot-producted with the Keys of all tokens to produce raw similarity
    scores — an n×n matrix. These scores are scaled by 1/√d_k for numerical stability, then passed
    through row-wise softmax to get a probability distribution. Finally, we take a weighted sum of
    the Value vectors, weighted by the attention probabilities. Result: each token gets a new
    representation that is a content-weighted blend of all other tokens. Input `(n, d_model)` →
    Output `(n, d_model)` — same shape, richer representations.

    ---

    **"Why do transformers outperform RNNs?"**

    Three fundamental advantages: (1) **Parallelizable training** — no sequential dependency,
    all tokens processed simultaneously, GPUs fully utilized; (2) **Direct long-range connections** —
    token i attends directly to token j, O(1) path length vs O(n) in RNNs, no vanishing-gradient
    chain; (3) **No information bottleneck** — no fixed-size hidden state, every token directly
    accesses every other token's representation.

    ---

    **"Walk me through positional encoding."**

    Attention is permutation-invariant — purely content-based, no position awareness. Fix:
    add sinusoidal patterns to embeddings before attention, where each dimension gets a different
    frequency: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`. This gives each position a unique
    fingerprint, and the relative-position property (`PE(pos+k)` is a linear function of `PE(pos)`)
    lets the model learn relative offsets. Modern standard: **RoPE** — rotates Q and K by
    position-dependent angles so the dot product depends only on relative position, generalizing
    better to longer contexts. Used in LLaMA, Mistral, and most production LLMs.

    ---

    **"What's the computational bottleneck in transformers?"**

    The n×n attention score matrix — O(n²) compute and O(n²) memory. For n=4,096 tokens with
    32 heads and 32 layers, attention scores alone require ~17B operations per forward pass. This
    is why long-context efficiency is a hot research area. Key techniques: **FlashAttention**
    (reorders computation to avoid materializing the full matrix in HBM — same result, 3–10×
    faster), **sliding window attention** (each token attends to only a local neighborhood),
    and **linear attention** approximations (kernel tricks that reduce to O(n)).

    ---

    **"Connection to my work."**

    In my Canopy project, I use sentence-transformers (a fine-tuned BERT variant) to embed job
    descriptions and resumes into dense vectors for similarity-based matching. The embedding step
    *is* transformer attention — BERT's pooled output is a weighted combination of all token
    representations across 12 layers of self-attention. The cosine similarity I compute at
    retrieval time is structurally identical to attention scoring: query dot key, measuring
    relevance. Understanding attention helps me reason about why certain job descriptions cluster
    well in embedding space and where retrieval breaks down — often at ambiguous pronoun
    references or rare domain terminology, exactly where attention heads struggle.
    """)
    return


if __name__ == "__main__":
    app.run()
