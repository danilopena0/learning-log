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
    # Fine-Tuning LLMs — LoRA, QLoRA, RLHF/DPO Intuition

    | Field   | Value                                                                      |
    |---------|----------------------------------------------------------------------------|
    | Date    | 2026-05-07                                                                 |
    | Track   | AI Engineering                                                             |
    | Time    | 60 min                                                                     |
    | Topic   | LoRA · QLoRA · RLHF · DPO · When to Fine-Tune vs Prompt vs RAG            |
    """)
    return


@app.cell
def adaptation_spectrum(mo):
    mo.md("""
    ## The Adaptation Spectrum

    Five ways to customize an LLM, from cheapest to most expensive:

    | # | Approach | Training Cost | When to Use |
    |---|----------|--------------|-------------|
    | 1 | **Prompt engineering** | $0 | First thing to try. Most problems solved here. |
    | 2 | **Few-shot / in-context learning** | $0 | Format or style needs examples in the prompt. |
    | 3 | **RAG** | Embedding pipeline only | Model needs external or fresh knowledge. |
    | 4 | **Fine-tuning (LoRA/QLoRA)** | Hours, one GPU, $10–100 | Style, format, or cost-optimizing a specific task. |
    | 5 | **Full fine-tuning** | Days, multiple GPUs, $1K–100K+ | Pre-training domain shift, very large datasets. Rarely needed. |

    > **The interview question is never "how do you fine-tune?" It's "WHEN do you fine-tune vs the
    > cheaper alternatives?" The decision framework matters more than the technique.**
    """)
    return


@app.cell
def part1_decision_framework(mo):
    mo.md("""
    ---
    ## Part 1: The Decision Framework — When to Fine-Tune

    ### Decision Tree

    ```
    What are you trying to achieve?
    │
    ├── Model needs KNOWLEDGE it doesn't have
    │   └── RAG. Not fine-tuning.
    │       Fine-tuning bakes knowledge into weights (stale fast).
    │       RAG retrieves fresh knowledge at query time.
    │       Example: "What jobs are posted this week?" → RAG
    │
    ├── Model needs to follow a specific OUTPUT FORMAT
    │   ├── Can structured outputs / JSON mode solve it? → Yes → Don't fine-tune
    │   └── Format is complex and prompt-based approaches fail >5% → Consider fine-tuning
    │       Example: generating very specific XML schemas consistently
    │
    ├── Model needs a specific STYLE or TONE
    │   ├── Can few-shot examples capture it? → Yes → Don't fine-tune
    │   └── Style is nuanced and examples aren't enough → Fine-tune on style examples
    │       Example: writing cover letters in MY voice (not generic AI voice)
    │
    ├── Model needs DOMAIN EXPERTISE
    │   ├── Is the domain knowledge available as documents? → RAG
    │   └── Is it about reasoning patterns, not facts? → Fine-tune
    │       Example: medical coding requires specific reasoning patterns, not just facts
    │
    ├── Model needs to be CHEAPER / FASTER at a specific task
    │   └── Fine-tune a small model to match a big model's quality on YOUR task
    │       Example: GPT-4o-mini fine-tuned on 10K examples matches GPT-4 quality
    │       This is the strongest ROI case for fine-tuning at scale
    │
    └── Model needs to STOP doing something (safety, refusals, verbosity)
        └── Alignment fine-tuning (RLHF / DPO)
            Example: model refuses valid queries, or is too verbose
    ```
    """)
    return


@app.cell
def decision_economics(mo):
    mo.md("""
    ### The Economics

    | Approach | Upfront cost | Per-query cost | Latency | When it wins |
    |----------|-------------|----------------|---------|-------------|
    | Prompt eng | $0 | Base model cost | Base | First thing to try, always |
    | Few-shot | $0 | Higher (more tokens) | Higher (longer prompt) | When format/style needs examples |
    | RAG | Embedding pipeline | Base + retrieval | +50–200ms | When model needs external knowledge |
    | LoRA fine-tune | $10–100 (one GPU, hours) | Smaller model = cheaper | Faster (smaller model) | High volume, specific task, cost optimization |
    | Full fine-tune | $1K–100K+ | Same as LoRA | Same | Pre-training domain shift, very large datasets |
    """)
    return


@app.cell
def my_projects_mapping(mo):
    mo.md("""
    ### My Projects — Fine-Tuning Decision Mapping

    - **Canopy job scoring**: prompt engineering + structured outputs. Fine-tuning **NOT needed** because the
      task is well-specified, few-shot examples cover the scoring rubric, and the knowledge (job descriptions)
      changes daily — that's RAG territory.

    - **Canopy cover letters**: this **IS** a potential fine-tuning candidate. I want MY writing style, not
      generic AI. But first: test a constitutional AI approach (generate → critique → revise). Only fine-tune
      if that fails.

    - **Briefing Agent summarization**: RAG handles the knowledge (papers, posts). Prompt engineering handles
      the format. Fine-tuning a smaller model would only make sense at much higher volume (10K+ summaries/day)
      for cost optimization.

    **Interview framing:**
    > "For my projects, I've deliberately avoided fine-tuning because prompt engineering and RAG solve the
    > problem at lower cost and complexity. I'd fine-tune if I needed to match GPT-4 quality at GPT-4o-mini
    > cost for a high-volume task."
    """)
    return


@app.cell
def part2_full_fine_tuning(mo):
    mo.md("""
    ---
    ## Part 2: Full Fine-Tuning — The Baseline

    Take a pre-trained model, continue training on YOUR task-specific dataset. Update **ALL** parameters
    (billions of weights). Standard supervised learning: input-output pairs, cross-entropy loss,
    backpropagation.

    ### The Math (simplified)

    - Pre-trained weights: **θ₀** (learned from internet-scale data)
    - Fine-tuned weights: **θ = θ₀ + Δθ** (Δθ is the change from fine-tuning)
    - Loss: **L = Σ CrossEntropy(model(xᵢ; θ), yᵢ)** for your task-specific (x, y) pairs
    - Gradient descent updates **ALL** of θ: **θ ← θ − lr × ∇L**

    For a **7B parameter model**:
    - Model weights: 14 GB (fp16)
    - Gradients: 14 GB
    - Optimizer state (Adam): 28 GB
    - **Total: ~56 GB** → requires 2–4 A100 GPUs just for a small LLM

    ### Why Full Fine-Tuning Is Usually Overkill

    - **Expensive**: multiple GPUs for hours or days
    - **Catastrophic forgetting**: model gets better at your task but worse at everything else
    - **Data hungry**: need thousands of high-quality examples to justify updating billions of weights
    - **Overfitting risk**: your dataset is tiny relative to the model — easy to memorize

    > "Full fine-tuning is like rebuilding the engine to change the oil. LoRA is changing the oil."
    """)
    return


@app.cell
def part3_lora_intuition(mo):
    mo.md("""
    ---
    ## Part 3: LoRA — The Efficiency Revolution

    ### The Key Insight (Hu et al., 2021)

    When you fine-tune, the weight **changes** (Δθ) are **low-rank**. The update matrix has much less
    information than its full size suggests.

    A 4096×4096 weight matrix has **16M parameters**. But the fine-tuning update to that matrix can be
    well-approximated by the product of two much smaller matrices:

    ```
    ΔW = A × B
    where A ∈ ℝ^(4096×r) and B ∈ ℝ^(r×4096), r = 4, 8, or 16
    ```

    Parameters to train: **4096×8 + 8×4096 = 65K** instead of 16M. That's **250× fewer parameters.**

    The frozen + adapter architecture: freeze ALL original weights. Add small trainable LoRA matrices
    alongside. At inference, merge them back — **no latency overhead**.
    """)
    return


@app.cell
def lora_math(mo):
    mo.md("""
    ### The Math — LoRA Forward Pass

    | Symbol | Meaning |
    |--------|---------|
    | **W₀ ∈ ℝ^(d×d)** | Original weight matrix — **frozen**, not updated |
    | **A ∈ ℝ^(d×r)** | LoRA down-projection — **trainable** (random Gaussian init) |
    | **B ∈ ℝ^(r×d)** | LoRA up-projection — **trainable** (zeros init → ΔW starts at 0) |
    | **r** | Rank — controls expressiveness vs. parameter count |

    **Forward pass:**
    ```
    h = (W₀ + ΔW) · x  =  W₀·x + A·B·x
    ```

    **At init**: B = 0, so ΔW = A·B = 0. Training starts from the exact pre-trained behavior.

    **Rank r** controls the trade-off:
    - r = 4: minimal parameters, good for style/format adaptation
    - r = 8–16: balanced, covers most fine-tuning scenarios
    - r = 32–64: more expressive, closer to full fine-tuning quality, more memory

    ### The Shapes — Visual

    ```
    Full fine-tuning:
    W₀ [4096 × 4096]   ← update all 16M params

    LoRA:
    W₀ [4096 × 4096]   (frozen)
    +
    A  [4096 ×   8]  ×  B [8 × 4096]   (trainable)
    └─────────────── 65K params total ───────────────┘

    Savings: 16M → 65K = 99.6% fewer trainable parameters
    ```
    """)
    return


@app.cell
def lora_where_applied(mo):
    mo.md("""
    ### Where LoRA Is Applied

    In a transformer, LoRA is applied to the **attention weight matrices**: W_Q, W_K, W_V, W_O.
    Sometimes also to FFN layers (W_up, W_down). Not every layer needs LoRA — attention layers alone
    is often enough.

    **Key configuration parameters (PEFT library):**

    ```python
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=8,                          # rank — 4 to 64
        lora_alpha=16,                # scaling factor, typically r or 2r
        lora_dropout=0.05,            # regularization on LoRA layers
        target_modules=["q_proj",     # which weight matrices to add LoRA to
                        "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, config)
    model.print_trainable_parameters()
    # trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%
    ```

    **After training — merge for zero inference overhead:**
    ```python
    merged_model = model.merge_and_unload()
    # W_final = W₀ + A·B  (no LoRA structure at inference time)
    ```
    """)
    return


@app.cell
def lora_why_it_works(mo):
    mo.md("""
    ### Why LoRA Works So Well

    Pre-trained models already know "how to language." Fine-tuning is mostly **steering**, not rebuilding.

    The steering signal **IS** low-rank: you're adjusting the model's behavior in a few specific directions,
    not rewriting everything. A 7B model has billions of parameters, but the behavioral shift from
    "generic assistant" to "medical coder" lives in a low-dimensional subspace.

    **Analogy:** a pre-trained LLM is a skilled general-purpose writer. Fine-tuning with LoRA is like
    giving them a style guide. You don't retrain their writing ability — you adjust their tendencies.
    The adjustment is low-dimensional (rank r).

    ### Practical Advantages

    | Benefit | Detail |
    |---------|--------|
    | **Memory** | 7B full fine-tune ≈ 56 GB → LoRA (r=8) ≈ 16 GB. Fits on one RTX 4090. |
    | **Speed** | 2–5× faster (fewer gradients to compute) |
    | **No catastrophic forgetting** | Original weights are frozen — general capabilities preserved |
    | **Multiple tasks** | Train separate LoRA adapters per task, swap at inference. One base model, many specializations. |
    | **Zero inference overhead** | Merge LoRA weights back into base model after training |
    """)
    return


@app.cell
def part4_qlora(mo):
    mo.md("""
    ---
    ## Part 4: QLoRA — Fine-Tuning on Consumer Hardware

    **QLoRA (Dettmers et al., 2023):** quantize the base model to 4-bit, then train LoRA adapters on top.

    The breakthrough: a **65B parameter model** can be fine-tuned on a **single 48GB GPU** (A40 or A6000).

    ### How It Works

    ```
    1. Load base model in 4-bit (NF4 quantization)
           ↓
    2. Keep quantized weights frozen
           ↓
    3. Add LoRA adapters in fp16/bf16 (full precision for the trainable params)
           ↓
    4. Backprop through quantized model, update ONLY LoRA params
    ```

    **Key innovations:**
    - **NF4 (4-bit NormalFloat)**: a data type optimized for the bell-curve distribution of neural network
      weights. Better than naive 4-bit integer quantization.
    - **Double quantization**: quantize the quantization constants too. Saves ~0.37 bits per parameter.
    - **Paged optimizers**: use CPU RAM as overflow for GPU memory via unified memory. Prevents OOM spikes
      during gradient accumulation.

    **QLoRA setup (bitsandbytes + PEFT):**
    ```python
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",        # NF4 data type
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # double quantization
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-8B",
        quantization_config=bnb_config,
        device_map="auto",
    )
    # Then add LoRA on top as before
    ```
    """)
    return


@app.cell
def qlora_memory_comparison(mo):
    mo.md("""
    ### Memory Comparison

    | Method | 7B Model | 13B Model | 70B Model |
    |--------|----------|-----------|-----------|
    | Full fine-tune (fp16) | ~56 GB | ~104 GB | ~560 GB |
    | LoRA (fp16 base) | ~16 GB | ~28 GB | ~140 GB |
    | **QLoRA (4-bit base)** | **~6 GB** | **~10 GB** | **~48 GB** |
    | Consumer GPU (RTX 4090) | ✅ 24 GB | ✅ 24 GB | ❌ |

    QLoRA makes 7B models trainable on a **laptop GPU** and 70B models trainable on a
    **single datacenter GPU**. This democratized fine-tuning.

    ### Quality: Does Quantization Hurt?

    Dettmers et al. showed QLoRA matches full 16-bit fine-tuning on benchmarks.

    The 4-bit quantization of the **base model** barely affects quality because you're not **training**
    those weights — they're frozen context. The LoRA adapters train in full precision (fp16/bf16) —
    the part that **matters** is still precise.

    > "QLoRA's insight: precision matters for the parameters you're CHANGING, not the ones you're READING."
    """)
    return


@app.cell
def part5_alignment_why(mo):
    mo.md("""
    ---
    ## Part 5: Alignment — RLHF and DPO

    ### Why Alignment Exists

    Pre-training teaches the model to predict the next token on internet text.
    But **predicting internet text ≠ being helpful, honest, and harmless**.

    A raw pre-trained model might:
    - Be toxic or offensive
    - Refuse valid requests
    - Hallucinate confidently
    - Be excessively verbose
    - Follow harmful instructions

    **Alignment fine-tuning**: adjust the model's behavior to match human preferences for what a
    "good response" looks like. Two major approaches: **RLHF** (the original) and **DPO** (the simpler
    replacement that's taking over).
    """)
    return


@app.cell
def rlhf_concept(mo):
    mo.md("""
    ### RLHF (Reinforcement Learning from Human Feedback)

    **The three-stage pipeline:**

    ```
    Stage 1: Supervised Fine-Tuning (SFT)
    ─────────────────────────────────────
    Dataset: high-quality (prompt, ideal response) pairs
    Goal: teach the model the FORMAT of helpful responses
    Output: SFT model (π_SFT)

         ↓

    Stage 2: Reward Model Training
    ───────────────────────────────
    Dataset: human preference pairs — (prompt, response A, response B, which is better)
    Goal: train R(prompt, response) → scalar score predicting human preference
    Output: reward model R

         ↓

    Stage 3: RL Optimization (PPO)
    ───────────────────────────────
    The SFT model generates responses → reward model scores them →
    policy gradient updates the LLM to produce higher-scoring responses
    Output: aligned model (π_RL)
    ```

    This is how GPT-3 → ChatGPT happened. Claude's Constitutional AI extends this pipeline.

    ### RLHF Math Intuition

    **PPO objective:**
    ```
    L = -𝔼[R(x, y)] + β × KL(π_RL ‖ π_SFT)
         └── maximize reward      └── stay close to SFT model
    ```

    - First term: produce responses that score high on the reward model
    - Second term: KL divergence penalty — don't drift too far from SFT baseline
    - **β** controls the trade-off: β too low → reward hacking; β too high → no improvement

    **Why RLHF is hard:**
    - RL training is unstable (PPO is notoriously finicky to tune)
    - Reward model can be gamed — adversarial responses that score high but are actually bad
    - Requires 3 models in memory simultaneously (policy + reward model + reference model)
    - Complex multi-stage engineering: SFT → RM → PPO, each can break independently
    """)
    return


@app.cell
def dpo_concept(mo):
    mo.md("""
    ### DPO (Direct Preference Optimization)

    **The insight (Rafailov et al., 2023):** you don't NEED the reward model or RL. You can directly
    optimize the LLM from preference pairs.

    **Key math insight:** the optimal policy under the RLHF objective has a closed-form relationship
    with the reward model. This means you can reparameterize the loss to **bypass the reward model
    entirely** — the reward model is implicitly defined by the LLM itself.

    **DPO loss** — given a prompt x, a preferred response y_w (winner), and a dispreferred response y_l (loser):

    ```
    L_DPO = -log σ(β × [log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)])
    ```

    **In plain English:**
    - Increase the probability of the preferred response, *relative to the reference model*
    - Decrease the probability of the dispreferred response, *relative to the reference model*
    - Using a simple binary cross-entropy-like loss — no RL, no reward model

    **DPO training data format:**
    ```python
    # Each example is a preference triple
    dataset = [
        {
            "prompt": "Write a cover letter for a data scientist role.",
            "chosen": "Dear Hiring Manager, I am excited to apply...",   # preferred
            "rejected": "Hello, I want this job because I am good at...", # dispreferred
        },
        ...
    ]

    # Standard SFT loop with modified loss — that's it
    from trl import DPOTrainer
    trainer = DPOTrainer(model=model, ref_model=ref_model, beta=0.1, ...)
    ```

    ### Why DPO Is Winning

    - No reward model needed — fewer moving parts, less data collection overhead
    - No RL — stable supervised training, no PPO instability
    - Simpler to implement — standard training loop + modified loss
    - Same or better quality as RLHF on most benchmarks
    - Memory: only 2 models (policy + reference), not 3
    """)
    return


@app.cell
def rlhf_vs_dpo_comparison(mo):
    mo.md("""
    ### RLHF vs DPO — Head to Head

    | Dimension | RLHF | DPO |
    |-----------|------|-----|
    | Training pipeline | 3 stages (SFT → RM → PPO) | 2 stages (SFT → DPO) |
    | Models in memory | 3 (policy + reward + reference) | 2 (policy + reference) |
    | Stability | Unstable (RL training) | Stable (supervised loss) |
    | Complexity | Very high | Moderate |
    | Data needed | Preference pairs | Same preference pairs |
    | Quality | Gold standard (when it works) | Comparable, sometimes better |
    | Who uses it | OpenAI (GPT-4), Anthropic (Claude) | Meta (Llama 2/3), most open-source |
    | Trend | Being replaced by DPO variants | Rapidly becoming the default |

    ### Variants Worth Knowing

    | Method | One-sentence description |
    |--------|-------------------------|
    | **IPO** (Identity Preference Optimization) | Fixes a theoretical issue in DPO's loss. More robust at high β. |
    | **KTO** (Kahneman-Tversky Optimization) | Only needs thumbs up/down per response, not paired comparisons. Easier data collection. |
    | **ORPO** (Odds Ratio Preference Optimization) | Combines SFT and preference optimization into one step. Even simpler. |

    > "Know the names and one-sentence descriptions. Interviewers don't expect you to derive the losses —
    > they want to know you're tracking the field."
    """)
    return


@app.cell
def part6_data(mo):
    mo.md("""
    ---
    ## Part 6: The Fine-Tuning Data Question

    ### Data Quality > Data Quantity

    | Use case | Typical volume |
    |----------|---------------|
    | LoRA fine-tuning (task/style) | 1,000–10,000 examples |
    | Alignment (DPO) | 10,000–50,000 preference pairs |
    | Full fine-tune (domain shift) | 100K–1M+ examples |

    **Quality markers:**
    - **Diverse**: covers the full distribution of real queries you'll see in production
    - **Correct**: bad labels = model learns wrong behavior (garbage in, garbage out)
    - **Consistent**: same type of query → same type of response format
    - **Representative**: matches production distribution, not cherry-picked easy cases

    > "It's better to have 1,000 perfect examples than 100,000 noisy ones. Data curation is the actual
    > work of fine-tuning."

    ### Data Creation Strategies

    | Strategy | Quality | Cost | Notes |
    |----------|---------|------|-------|
    | **Manual curation** | Highest | Expensive | Experts write ideal responses |
    | **Distillation (GPT-4 → small model)** | High | Moderate | Check licenses — OpenAI terms restrict use for competing models |
    | **Production logging** | Most representative | Slow to accumulate | Log real queries, have humans rate them |
    | **Synthetic augmentation** | Good | Low | Use large model to generate variations of seed examples |

    **Connection to Canopy eval work:**
    > "If I had 50 manually scored job descriptions (from my eval gap analysis), I could use those as DPO
    > preference pairs — high-scoring outputs as 'chosen,' low-scoring as 'rejected.' That's both an eval
    > set AND a fine-tuning dataset. One artifact, two uses."
    """)
    return


@app.cell
def practical_checklist(mo):
    mo.md("""
    ---
    ## Practical Fine-Tuning Checklist

    ```
    BEFORE fine-tuning — verify all of these:
    ─────────────────────────────────────────
    □ Prompt engineering thoroughly tested and doesn't solve the problem
    □ RAG doesn't solve the problem (if knowledge is the issue)
    □ 1,000+ high-quality training examples collected
    □ Clear eval suite defined to measure improvement
    □ Baseline established WITHOUT fine-tuning (so you can measure the delta)
    □ Cost understood: compute + data curation + ongoing maintenance
    □ Plan for model updates (when base model gets a new version, you may need to re-fine-tune)

    DURING fine-tuning:
    ──────────────────
    □ Start with LoRA (r=8–16), not full fine-tuning
    □ Use QLoRA if GPU memory is constrained
    □ Train for 1–3 epochs (more = overfitting on small datasets)
    □ Evaluate after EVERY epoch on held-out data
    □ Compare against the base model + prompt engineering baseline
    □ Check for catastrophic forgetting on general tasks

    AFTER fine-tuning:
    ─────────────────
    □ Run full eval suite (not just the task you fine-tuned for)
    □ Merge LoRA weights for inference efficiency (merge_and_unload())
    □ A/B test against the non-fine-tuned version in production
    □ Set up monitoring for quality degradation over time
    ```
    """)
    return


@app.cell
def flashcards(mo):
    mo.md("""
    ---
    ## Flashcard Summary

    **When to fine-tune vs prompt vs RAG?**
    → Need knowledge → RAG. Need format/style that few-shot can't capture → fine-tune. Need to make a
    small model match a big one → fine-tune. Everything else → prompt engineering first.

    ---

    **What is LoRA?**
    → Freeze base model weights. Add small trainable low-rank matrices (A×B) alongside the attention layers.
    99%+ fewer trainable params. No catastrophic forgetting. Merge back for zero inference overhead.

    ---

    **Why is LoRA low-rank?**
    → Fine-tuning updates are low-dimensional. You're steering the model, not rebuilding it. The steering
    signal compresses into a few directions (rank r).

    ---

    **What is QLoRA?**
    → Quantize base model to 4-bit (NF4), train LoRA adapters in fp16 on top. Fits 70B model fine-tuning
    on a single GPU. Matches full 16-bit quality because frozen weights just provide context.

    ---

    **RLHF in one sentence?**
    → Train a reward model on human preferences, then use RL (PPO) to make the LLM produce
    higher-reward responses.

    ---

    **DPO in one sentence?**
    → Skip the reward model and RL. Directly optimize the LLM from preference pairs using a supervised
    loss derived by reparameterizing the RLHF objective.

    ---

    **Why is DPO replacing RLHF?**
    → No reward model needed, no RL instability, simpler to implement, comparable quality. Only downside:
    less flexible for complex multi-dimensional reward signals.

    ---

    **How much data for LoRA fine-tuning?**
    → 1,000–10,000 high-quality examples. Quality > quantity.

    ---

    **What's catastrophic forgetting?**
    → Fine-tuning on task A makes the model worse at everything else. LoRA avoids this by keeping base
    weights frozen — only the small adapters change.
    """)
    return


@app.cell
def interview_talking_points(mo):
    mo.md("""
    ---
    ## Interview Talking Points

    **"When would you fine-tune an LLM?"**
    > "Only after exhausting prompt engineering, few-shot, and RAG. Three valid reasons to fine-tune:
    > matching a specific style that examples can't capture, making a small model match a big model's
    > quality for cost optimization at scale, or alignment tuning. I'd use LoRA with r=8–16, evaluate
    > against a held-out set after each epoch, and A/B test against the non-fine-tuned baseline in
    > production."

    ---

    **"Explain LoRA to me."**
    > "Pre-trained weights are frozen. I add small trainable matrices alongside the attention layers.
    > These matrices are low-rank — rank 8 means each is a 4096×8 and 8×4096 matrix, totaling 65K
    > parameters instead of 16M. The intuition: fine-tuning is steering, and the steering signal is
    > low-dimensional. At inference, the LoRA weights merge back into the base model — zero latency
    > overhead."

    ---

    **"RLHF vs DPO?"**
    > "RLHF trains a separate reward model, then uses PPO to optimize the LLM. Three-stage, three models
    > in memory, RL instability. DPO bypasses the reward model entirely — it directly optimizes from
    > preference pairs with a supervised loss derived by reparameterizing the RLHF objective. Simpler,
    > more stable, comparable quality. DPO and its variants are becoming the default for open-source
    > alignment."

    ---

    **Connection to my work:**
    > "In Canopy, I've deliberately chosen NOT to fine-tune — prompt engineering with structured outputs
    > handles job scoring well, and the knowledge (job descriptions) changes daily, which is RAG territory.
    > If I were running Canopy at scale (100K+ scores/day), I'd fine-tune GPT-4o-mini on my labeled eval
    > set to match GPT-4 quality at 10× lower cost. That's the right economic trigger for fine-tuning."
    """)
    return


if __name__ == "__main__":
    app.run()
