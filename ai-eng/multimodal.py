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
    # Multi-Modal AI — Vision-Language Models, Image Embeddings, CLIP

    | Field | Value |
    |-------|-------|
    | Date  | 2026-05-24 |
    | Track | AI Engineering |
    | Time  | 60 min |
    | Topic | Vision-Language Models · Image Embeddings · CLIP Architecture · Contrastive Learning |
    """)
    return


@app.cell
def what_multimodal_means(mo):
    mo.md("""
    ## What Multi-Modal Means

    **Single-modal**: a model processes ONE type of data.
    - BERT → text
    - ResNet → images
    - Whisper → audio

    **Multi-modal**: a model processes MULTIPLE types and understands relationships BETWEEN them.
    "A photo of a dog playing fetch" — the model connects the image content to the text description.

    ### Why it matters now
    GPT-4o, Claude, Gemini all process text + images natively. Multi-modal is becoming the **DEFAULT**,
    not the exception. AI eng interviews increasingly assume you can reason about these systems.

    ### The core challenge
    Text and images live in completely different representation spaces:
    - Words are sequences of **discrete tokens** — a finite vocabulary, categorical
    - Images are grids of **continuous pixel values** — 224×224×3 = 150,528 numbers

    How do you get them to "talk to each other"?

    > "Understanding multi-modal architecture helps me reason about ANY cross-domain embedding problem —
    > including the text + metadata embeddings in Canopy."
    """)
    return


@app.cell
def part1_header(mo):
    mo.md("---\n## Part 1: Image Representations — From Pixels to Vectors")
    return


@app.cell
def how_cnns_see(mo):
    mo.md("""
    ### How CNNs See Images

    **Raw image**: H × W × 3 tensor (height × width × RGB channels). A 224×224 image = **150,528 numbers**.

    **Conv layers** learn hierarchical features:
    ```
    Early layers  →  edges, gradients
    Middle layers →  textures, shapes, corners
    Deep layers   →  objects, scenes, faces
    ```

    **The "representation"**: the output of the second-to-last layer (before the classification head).
    For ResNet-50: a fixed 2048-dim vector that encodes the image's CONTENT.

    This vector IS an image embedding — same concept as word embeddings, but for visual content.
    The classification head is discarded; the embedding is what we keep.

    | Layer type | What it learns | Output shape (ResNet-50 example) |
    |------------|---------------|----------------------------------|
    | Conv1 | Low-level edges | 112×112×64 |
    | Conv2-4 | Textures, shapes | 56×56×256 |
    | Conv5 | Object parts | 7×7×2048 |
    | Global avg pool | **Image embedding** | 2048 |
    | FC + softmax | Classification (discarded) | 1000 |

    **Key insight**: the embedding layer is a compression of visual content into a dense vector —
    exactly like word2vec compresses semantic meaning into a dense vector.
    """)
    return


@app.cell
def vit_cell(mo):
    mo.md("""
    ### Vision Transformer (ViT)

    **Key insight** (Dosovitskiy et al., 2020): treat an image as a **sequence of patches**,
    just like a sentence is a sequence of word tokens.

    **Process**:
    ```
    1. Divide image into fixed-size patches (16×16 pixels)
       A 224×224 image → 196 patches

    2. Flatten each patch into a vector
       16 × 16 × 3 = 768 dims per patch

    3. Add positional embeddings
       (same concept as transformer positional encoding — patches need location info)

    4. Prepend a [CLS] token (learnable)

    5. Feed the sequence of 197 vectors through a standard transformer encoder

    6. Use the [CLS] token output as the image embedding
    ```

    **Why this works**: patches are to images what tokens are to text. The transformer's self-attention
    lets patches attend to each other — a patch of a dog's ear can "look at" the patch with the dog's
    body to understand context.

    > "ViT applies the EXACT same transformer architecture from my transformer notebook. Same attention,
    > same positional encoding, same multi-head mechanism. The only difference is the input: image patches
    > instead of word tokens."

    **The unification**: the transformer architecture is modality-agnostic. It operates on sequences of
    vectors. Anything you can represent as a sequence of vectors can be fed through a transformer.
    """)
    return


@app.cell
def cnn_vs_vit(mo):
    mo.md("""
    ### CNN vs ViT

    | Dimension | CNN (ResNet) | ViT |
    |-----------|-------------|-----|
    | Inductive bias | Translation invariance (convolution), locality | Minimal — learned from data |
    | Data efficiency | Better with small data (bias helps) | Needs more data (fewer assumptions) |
    | Scale | Saturates at large scale | Keeps improving with more data + compute |
    | Global context | Limited by receptive field (requires stacking layers) | Full global attention from layer 1 |
    | Current status | Still strong for edge/mobile | Dominant for large-scale vision |
    | Use for embeddings | ResNet features are solid baseline | ViT features are state-of-the-art |

    **When to pick which**:
    - Small dataset (<100K images) with limited compute → CNN (ResNet-50 with ImageNet pretraining)
    - Large dataset, best quality needed → ViT (ViT-L or ViT-H)
    - Production edge deployment (phone, IoT) → CNN (EfficientNet)
    - Multi-modal system (CLIP, image+text) → ViT (what CLIP uses)
    """)
    return


@app.cell
def part2_header(mo):
    mo.md("---\n## Part 2: CLIP — The Bridge Between Vision and Language")
    return


@app.cell
def why_clip_matters(mo):
    mo.md("""
    ### Why CLIP Matters

    **Before CLIP (2021)**: separate models for vision and language. To connect them, you'd train
    a model specifically for each task: image captioning model, visual QA model, image-text retrieval model.
    Each requires labeled task-specific data.

    **CLIP's insight**: train ONE model that puts images and text in the **SAME embedding space**.
    Then ANY vision-language task becomes an embedding similarity problem. No task-specific training.

    **Training data**: 400 million (image, text) pairs scraped from the internet.
    - No manual labeling — the paired data IS the supervision.
    - Alt text, captions, surrounding text on web pages.
    - Scale compensates for noise.

    > "CLIP is the visual equivalent of sentence-transformers. Just as sentence-transformers let me
    > compute text similarity with cosine distance, CLIP lets me compute image-text similarity.
    > Same pattern, different modalities."
    """)
    return


@app.cell
def clip_architecture(mo):
    mo.md("""
    ### CLIP Architecture

    ```
    [Image]                             [Text]
    (any photo)             "a photo of a golden retriever"
         |                                  |
    [Image Encoder]                 [Text Encoder]
    (ViT-B/32 or ResNet)           (GPT-2-style transformer)
         |                                  |
    [Image Embedding]               [Text Embedding]
       (512-dim)                       (512-dim)
         |                                  |
         └──────── cosine similarity ───────┘
                          |
                    [Match Score]
                    (0.0 to 1.0)
    ```

    **This is the two-tower architecture.** Same pattern as:
    - RAG: query encoder + document encoder → similarity → retrieve
    - Recommender systems: user encoder + item encoder → similarity → recommend
    - Canopy: job description encoder + profile encoder → similarity → rank

    > "Every time I see 'two separate encoders → shared embedding space → similarity,' it's the same
    > architecture. CLIP, sentence-transformers, two-tower recommendations, bi-encoder retrieval.
    > Understanding one means understanding all of them."

    **The shared space**: after training, `cosine(encode_image(dog_photo), encode_text("a photo of a dog"))`
    will be high (~0.9), and `cosine(encode_image(dog_photo), encode_text("a photo of a car"))` will be
    low (~0.1). The encoders have been trained to agree on meaning across modalities.
    """)
    return


@app.cell
def contrastive_learning(mo):
    mo.md("""
    ### Contrastive Learning — How CLIP Trains

    **Training data**: N (image, text) pairs in a batch.

    **Create an N × N similarity matrix** — all image-text combinations:

    ```
    Text:   "a dog"  "a cat"  "a car"  "a tree"
    Image:
    dog     [ 0.92    0.12     0.05     0.08  ]  ← row should peak at column 0
    cat     [ 0.11    0.89     0.07     0.06  ]  ← row should peak at column 1
    car     [ 0.06    0.08     0.91     0.09  ]  ← row should peak at column 2
    tree    [ 0.07    0.05     0.08     0.93  ]  ← row should peak at column 3
    ```

    - **Diagonal** = correct pairs — image_i should match text_i
    - **Off-diagonal** = incorrect pairs — image_i should NOT match text_j

    **Loss**: maximize diagonal similarity, minimize off-diagonal similarity.
    This is **InfoNCE loss** (contrastive loss):
    - Pull matching pairs TOGETHER in embedding space
    - Push non-matching pairs APART in embedding space

    **No labels needed**: the batch structure IS the label. If an image and text are paired,
    they should be similar. Everything else in the batch is a negative example.

    > "This is the same contrastive learning used in sentence-transformers (SimCSE) and
    > self-supervised learning generally. The batch provides negative examples automatically —
    > no manual annotation of 'this pair doesn't match.'"
    """)
    return


@app.cell
def contrastive_loss_math(mo):
    mo.md("""
    ### The Contrastive Loss Math

    For a batch of N pairs: images {I₁, ..., Iₙ} and texts {T₁, ..., Tₙ}

    **Step 1**: Compute similarity matrix with temperature scaling
    ```
    S_ij = cosine(encode_image(Iᵢ), encode_text(Tⱼ)) / τ
    ```
    - τ (temperature): controls how "peaked" the distribution is. **Learned parameter**.
    - Low τ → very confident predictions (sharp peaks)
    - High τ → smoother predictions (more uncertainty)

    **Step 2**: Image-to-text loss (for each image, find its correct text)
    ```
    L_i2t = (1/N) Σᵢ  -log( exp(S_ii) / Σⱼ exp(S_ij) )
            ↑ average   ↑ correct pair    ↑ all texts for this image
    ```
    = softmax cross-entropy along rows of the similarity matrix

    **Step 3**: Text-to-image loss (symmetric direction)
    ```
    L_t2i = (1/N) Σⱼ  -log( exp(S_jj) / Σᵢ exp(S_ij) )
    ```
    = softmax cross-entropy along columns of the similarity matrix

    **Total loss**:
    ```
    L = (L_i2t + L_t2i) / 2
    ```

    **Intuition**: this is just N-way classification. For each image, "which of the N texts is the match?"
    And for each text, "which of the N images is the match?" Cross-entropy on both directions.

    **Why large batches matter**: more items in the batch = more negative examples per anchor.
    CLIP used batch size 32,768 — each image gets 32,767 negative text examples per step.
    Hard negatives (similar but wrong) are more informative; large batches guarantee some.
    """)
    return


@app.cell
def similarity_matrix_viz(mo):
    mo.md("### Contrastive Training: Before vs After")
    return


@app.cell
def similarity_matrix_code():
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    labels = ["dog photo", "cat photo", "car photo", "tree photo"]
    texts  = ['"a dog"', '"a cat"', '"a car"', '"a tree"']

    # Before training: random similarities, no structure
    np.random.seed(42)
    before = np.random.uniform(0.3, 0.7, (4, 4))

    # After training: diagonal peaks, off-diagonal suppressed
    after = np.array([
        [0.92, 0.11, 0.05, 0.08],
        [0.10, 0.90, 0.07, 0.06],
        [0.06, 0.08, 0.91, 0.09],
        [0.07, 0.05, 0.08, 0.93],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Contrastive Training: Similarity Matrix Before vs After",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ax, matrix, title in zip(axes, [before, after], ["Before training", "After training"]):
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(texts, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Text candidates", fontsize=10)
        ax.set_ylabel("Image queries", fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)

        for i in range(4):
            for j in range(4):
                color = "white" if matrix[i, j] < 0.45 or matrix[i, j] > 0.75 else "black"
                border = " ◆" if i == j else ""
                ax.text(j, i, f"{matrix[i, j]:.2f}{border}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold" if i == j else "normal")

    plt.colorbar(im, ax=axes[1], label="Cosine similarity", shrink=0.85)
    plt.tight_layout()

    return fig, math, matplotlib, mpatches, np, plt, after, before, labels, texts


@app.cell
def show_viz(fig, mo):
    mo.md("""
    **Reading the heatmap**:
    - Green diagonal = correct pairs (high similarity). Contrastive training pushes these UP.
    - Red off-diagonal = wrong pairs (low similarity). Contrastive training pushes these DOWN.
    - Before training: random structure — model has no notion of image-text alignment.
    - After training: clear diagonal pattern — matching pairs cluster together in embedding space.

    The loss function IS the instruction to produce this diagonal pattern.
    """)
    return mo.image(fig)


@app.cell
def part3_header(mo):
    mo.md("---\n## Part 3: Zero-Shot Classification with CLIP")
    return


@app.cell
def zero_shot_concept(mo):
    mo.md("""
    ### Zero-Shot Classification

    **Traditional image classification**: train on labeled images → softmax head over fixed categories.
    Can ONLY classify into categories seen during training.

    **CLIP zero-shot**: encode the image → encode text descriptions of candidate categories →
    return the most similar text.

    ```
    Image: [photo of a golden retriever]

    Candidate texts:
    "a photo of a cat"         →  cosine = 0.18
    "a photo of a dog"         →  cosine = 0.92  ← winner
    "a photo of a car"         →  cosine = 0.05
    "a photo of a building"    →  cosine = 0.08

    Predicted class: "dog"
    ```

    **No training on these specific categories.** Just write the category names in natural language.
    Add a new category? Write its text description. No data collection, no training.

    > "This is classification BY RETRIEVAL. Instead of a softmax layer, you're doing nearest-neighbor
    > in embedding space against text descriptions. Same pattern as Canopy scoring jobs by similarity
    > to a candidate profile."

    **Why it generalizes**: CLIP has seen 400M image-text pairs. It has built a rich shared embedding
    space that covers most of the visual world. The text description acts as the class prototype.
    """)
    return


@app.cell
def prompt_engineering_clip(mo):
    mo.md("""
    ### Prompt Engineering for CLIP

    The text descriptions matter — framing and context affect the embedding.

    | Prompt | Why it works (or doesn't) |
    |--------|--------------------------|
    | `"dog"` | Too short, doesn't match training caption distribution |
    | `"a photo of a dog"` | Matches typical alt-text / caption style. Works well. |
    | `"a photo of a dog in the wild"` | More specific — narrows to wildlife context |
    | `"a satellite image of {category}"` | Domain-specific framing for remote sensing |
    | `"a medical scan showing {condition}"` | Domain framing for medical images |

    **Why framing matters**: CLIP was trained on (image, caption) pairs where captions look like
    "a photo of..." — matching that distribution aligns with the training distribution.

    **Ensemble approach** (more robust):
    1. Generate multiple prompts per class: `"a photo of a {c}"`, `"a {c}"`, `"a picture of a {c}"`
    2. Encode each prompt → average the embeddings → use the average as the class prototype
    3. Averaging smooths over prompt-specific quirks

    > "Same prompt engineering principles from my prompt engineering notebook — context and framing
    > matter for CLIP just like they matter for LLMs. The model was trained on language; what language
    > you use determines what space you land in."
    """)
    return


@app.cell
def clip_code_demo(mo):
    mo.md("""
    ### CLIP Zero-Shot Classification — Code Pattern

    ```python
    # pip install transformers pillow torch
    from transformers import CLIPModel, CLIPProcessor
    import torch
    import torch.nn.functional as F

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # ── Encode candidate texts (class prototypes) ─────────────────────────────
    categories = ["a photo of a dog", "a photo of a cat", "a photo of a car"]
    text_inputs = processor(text=categories, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)  # shape: (3, 512)
        text_embeddings = F.normalize(text_embeddings, dim=-1)    # unit vectors

    # ── Encode query image ────────────────────────────────────────────────────
    # image = Image.open("dog.jpg")
    image_inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        image_embedding = model.get_image_features(**image_inputs)  # shape: (1, 512)
        image_embedding = F.normalize(image_embedding, dim=-1)

    # ── Classify by similarity ────────────────────────────────────────────────
    similarities = (image_embedding @ text_embeddings.T).squeeze(0)  # shape: (3,)
    # [0.05, 0.92, 0.08] → index 1 wins

    probs = similarities.softmax(dim=0)  # optional: normalize to probabilities
    predicted = categories[similarities.argmax()]
    # → "a photo of a dog"
    ```

    **Shape annotations** (the key insight is in the shapes):
    - `text_embeddings`: `(num_classes, 512)` — one prototype vector per class
    - `image_embedding`: `(1, 512)` — one vector per image
    - `image_embedding @ text_embeddings.T`: `(1, 3)` — one score per class
    - Classification = finding the highest score = argmax over that last dimension

    The code is trivial. The insight is deep: **we replaced a task-specific classification head
    with a general similarity computation in shared embedding space.**
    """)
    return


@app.cell
def part4_header(mo):
    mo.md("---\n## Part 4: Beyond CLIP — Modern Vision-Language Models")
    return


@app.cell
def vlm_landscape(mo):
    mo.md("""
    ### The Vision-Language Model Landscape

    | Model | Year | Architecture | Capabilities |
    |-------|------|-------------|-------------|
    | **CLIP** (OpenAI) | 2021 | Two-tower contrastive | Image-text similarity, zero-shot classification |
    | **BLIP-2** (Salesforce) | 2023 | Frozen ViT + Q-Former + frozen LLM | Captioning, visual QA |
    | **LLaVA** | 2023 | CLIP ViT + projection + LLM | Visual instruction following |
    | **GPT-4o** (OpenAI) | 2024 | Native multi-modal | Interleaved text+image reasoning |
    | **Claude 3+** (Anthropic) | 2024 | Native multi-modal | Text+image understanding |
    | **Gemini** (Google) | 2024 | Native multi-modal | Text+image+audio+video |

    **The trend**: from "two separate models connected by an adapter" → "one model that natively
    processes everything."

    Each step:
    1. CLIP: two towers, shared embedding space, no generation
    2. BLIP-2: add a lightweight bridge (Q-Former) to connect a frozen vision encoder to a frozen LLM
    3. LLaVA: train a projection layer to map image patches into LLM token space
    4. GPT-4o / Claude / Gemini: fully unified — no visible seam between modalities
    """)
    return


@app.cell
def how_modern_vllm_work(mo):
    mo.md("""
    ### How Modern Multi-Modal LLMs Work (LLaVA-style)

    **Architecture** (representative of the field):

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │  Step 1: Vision Encoder (frozen CLIP ViT)                       │
    │                                                                 │
    │  [Image 224×224×3]  →  [196 patch embeddings, 768-dim each]     │
    └─────────────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────────────────────────────────────┐
    │  Step 2: Projection Layer (trained)                             │
    │                                                                 │
    │  [196 × 768]  →  [196 × 4096]  (matches LLM embedding dim)     │
    └─────────────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────────────────────────────────────┐
    │  Step 3: LLM processes interleaved tokens                       │
    │                                                                 │
    │  [text tokens] [visual tokens × 196] [text tokens]             │
    │  "Describe"    [img_patch_0 ... img_patch_195]  "in detail:"    │
    │                                                                 │
    │  Attention: text tokens attend to visual tokens & vice versa   │
    └─────────────────────────────────────────────────────────────────┘
    ```

    > "The image patches become 'visual tokens' that the LLM processes alongside text tokens.
    > The LLM's attention mechanism handles the cross-modal interaction. The LLM doesn't know
    > they came from an image — they're just more tokens with positional embeddings."

    **What's trained vs frozen** (LLaVA v1.5):
    - Vision encoder (CLIP ViT-L): **frozen** — preserves CLIP's rich visual representations
    - Projection layer (2-layer MLP): **trained** — learns to map visual space → LLM token space
    - LLM (Vicuna/Mistral): **fine-tuned** — learns to use visual tokens for instruction following

    This is efficient: you don't re-train the entire vision model or the entire LLM.
    The projection layer is the bridge, and it's tiny relative to the full model.
    """)
    return


@app.cell
def part5_header(mo):
    mo.md("---\n## Part 5: Practical Applications")
    return


@app.cell
def image_search(mo):
    mo.md("""
    ### Image Search / Retrieval

    **Pattern**: index images by CLIP embeddings → query with text or image → ANN search.

    ```
    Offline (indexing):
    [Image 1] → CLIP image encoder → embedding → HNSW index
    [Image 2] → CLIP image encoder → embedding → HNSW index
    ...

    Online (query):
    "sunset over a mountain lake" → CLIP text encoder → query embedding
                                 → ANN search in HNSW index
                                 → return top-K matching images
    ```

    **Cross-modal query options** (this is the power of the shared embedding space):
    - Text → images: "sunset over a mountain lake" → find matching photos
    - Image → images: upload a photo, find visually similar photos
    - Image → text: find captions/descriptions that match an image

    > "This is RAG for images. Same architecture: encode query → vector search → return results.
    > The only difference is the encoder. From my vector databases notebook: HNSW handles the
    > ANN search, and the embedding model determines the retrieval quality."

    **Production considerations**:
    - Embed images at upload time, store in vector DB (Pinecone, Weaviate, pgvector)
    - For very large collections (>100M images), use quantization (PQ) to reduce index size
    - CLIP ViT-L/14 gives better quality than ViT-B/32 at 2× the embedding compute cost
    """)
    return


@app.cell
def multimodal_rag(mo):
    mo.md("""
    ### Multi-Modal RAG

    From my advanced RAG notebook: documents contain images, tables, charts — text-only RAG
    misses all visual information.

    **CLIP-based multi-modal RAG pipeline**:

    ```
    Ingestion:
    Document → extract text chunks → embed with text encoder → store in vector DB
            → extract images     → embed with CLIP         → store in same vector DB
            → (optional) caption each image with BLIP-2    → store as text chunk too

    Retrieval:
    Query → embed (text encoder or CLIP text encoder)
          → search across both text chunk embeddings AND image embeddings
          → return top-K text chunks + top-M images

    Generation:
    [query] + [retrieved text chunks] + [retrieved images] → GPT-4o / Claude → answer
    ```

    > "For Canopy: job postings rarely have meaningful images — text-only RAG is sufficient.
    > For the Briefing Agent processing arXiv papers: indexing figure embeddings alongside text
    > would capture architecture diagrams and results charts that text-only RAG misses entirely.
    > The ViT-encoded 'attention is all you need' architecture diagram has a unique embedding —
    > a text query about 'transformer architecture' could retrieve it."

    **When to use multi-modal RAG**:
    - Research papers, technical docs (lots of figures and diagrams)
    - Product catalogs (images convey style, color, shape that text descriptions miss)
    - Medical records (x-rays, scans alongside clinical notes)
    - NOT worth it for: pure text corpora, code docs, most enterprise documents
    """)
    return


@app.cell
def content_moderation(mo):
    mo.md("""
    ### Content Moderation

    **Pattern**: use CLIP zero-shot classification to flag unsafe content without training a custom model.

    ```python
    # No labeled data needed — just write the category descriptions
    safe_categories = [
        "a safe, family-friendly image",
        "a normal photograph of people or objects",
    ]
    unsafe_categories = [
        "a violent or graphic image",
        "an explicit or adult image",
        "an image of illegal activity",
    ]

    all_categories = safe_categories + unsafe_categories
    # Embed all categories with CLIP text encoder
    # Score each uploaded image against all categories
    # Flag if similarity to any unsafe category exceeds threshold
    ```

    **Advantages**:
    - No labeled training data needed
    - Add new categories by writing text descriptions (no retraining)
    - Works across domains (photos, illustrations, screenshots)

    **Limitations**:
    - Less accurate than a fine-tuned classifier on your specific definition of "unsafe"
    - CLIP may be inconsistent at the decision boundary
    - Use as a first-pass filter, not the final decision — route borderline cases to human review

    **Real production pattern**: CLIP first-pass → fine-tuned classifier for high-recall categories →
    human review queue for borderline cases.
    """)
    return


@app.cell
def part6_header(mo):
    mo.md("---\n## Part 6: Training Your Own Multi-Modal Embeddings")
    return


@app.cell
def when_to_finetune(mo):
    mo.md("""
    ### When to Fine-Tune CLIP

    **Pre-trained CLIP works well for**: general image-text matching where categories are
    common in web data (pets, cars, food, landmarks, people, everyday objects).

    **Fine-tuning helps for domain-specific tasks**:

    | Domain | Why CLIP falls short | Fine-tune signal |
    |--------|---------------------|-----------------|
    | Medical imaging | X-rays, histology slides are rare in web alt-text | (image, radiology report) pairs |
    | Satellite imagery | Aerial photos of farmland ≠ typical captions | (image, land-use label) pairs |
    | E-commerce products | Brand-specific product naming, style nuances | (product image, product name/description) |
    | Industrial inspection | Defects are subtle, domain-specific terminology | (image, defect description) pairs |

    **Fine-tuning approach**:
    - **LoRA** (from my fine-tuning notebook): freeze most of CLIP, train small adapter layers.
      Efficient — 1-5% of parameters trained, preserves pre-trained representations.
    - **Full fine-tuning**: better quality if you have enough data, but risks catastrophic forgetting.
    - **Linear probe**: freeze CLIP, train only a classification head. Fastest, weakest.

    **Data requirement**: 10K–100K domain-specific (image, text) pairs is often enough.
    The pre-trained CLIP provides strong initialization — you're adapting, not training from scratch.

    **Data collection strategy**: if you have image + metadata (product title, caption, report),
    you already have (image, text) pairs. No manual labeling needed — same self-supervised approach
    CLIP used.
    """)
    return


@app.cell
def open_source_models(mo):
    mo.md("""
    ### Open-Source Multi-Modal Models

    | Model | Params | Capabilities | Best for |
    |-------|--------|-------------|----------|
    | CLIP ViT-B/32 | 150M | Image-text similarity | Embedding, zero-shot classification (fast) |
    | CLIP ViT-L/14 | 430M | Better accuracy | Same, higher quality, ~2× cost |
    | SigLIP (Google) | 400M | Improved CLIP training objective | Better calibration, scaling |
    | BLIP-2 | 3–12B | Captioning, visual QA | Image understanding tasks |
    | LLaVA 1.5 | 7–13B | Visual instruction following | Chat with images, analysis |
    | Florence-2 | 230M–770M | Many vision tasks (det, seg, cap) | Versatile, efficient |
    | Phi-3-Vision | 4.2B | Efficient visual reasoning | Edge / cost-constrained VLM |

    **My default choices**:
    - Embeddings for retrieval → CLIP ViT-L/14 (quality) or ViT-B/32 (speed)
    - Image captioning → BLIP-2 (open source) or GPT-4o (via API, more capable)
    - Visual QA / instruction following → LLaVA 1.5 7B (self-hosted) or Claude/GPT-4o (API)
    - Domain fine-tuning → start from CLIP ViT-L/14 + LoRA

    **Hosting**: all of the above run on a single A100 (80GB) for inference.
    CLIP ViT-B/32 runs on CPU for batch embedding jobs (slow but free).
    """)
    return


@app.cell
def part7_header(mo):
    mo.md("---\n## Part 7: The Unified Embedding Perspective")
    return


@app.cell
def everything_is_embeddings(mo):
    mo.md("""
    ### Everything is Embeddings

    The grand unification across everything studied in this log:

    ```
    Modality    Encoder                     Embedding     Index          Task
    ─────────   ─────────────────────────   ───────────   ────────────   ─────────────────────
    Text        BERT / sentence-transformers  512-dim     HNSW           Semantic search (RAG)
    Images      CLIP ViT / ResNet            512-dim     HNSW           Image retrieval
    Audio       Whisper encoder               512-dim     HNSW           Audio search
    User prefs  User tower (rec system)       64-dim      ANN            Recommendations
    Job desc    Bi-encoder (Canopy)          768-dim     HNSW           Job matching
    Code        CodeBERT / StarEncoder       512-dim     HNSW           Code search
    Video       CLIP (frame-level)           512-dim     HNSW           Video retrieval
    ```

    **EVERY retrieval / recommendation / matching system follows this pattern:**
    1. Encode the query entity → vector
    2. Encode candidate entities → vectors
    3. Find nearest neighbors in embedding space

    The ONLY differences are: encoder architecture, embedding dimension, training data, index type.

    **Multi-modal = putting MULTIPLE modalities in the SAME embedding space** so that
    cross-modal retrieval works: a text query retrieves images, an image query retrieves captions.

    > "When I see a new AI retrieval system, the first question I ask is: what's the encoder?
    > Everything else follows from that."
    """)
    return


@app.cell
def connection_map(mo):
    mo.md("""
    ### Connection Map — Across This Learning Log

    | Notebook | Embedding pattern used |
    |----------|------------------------|
    | Transformers | Q, K, V are linear projections into embedding subspaces |
    | Dimensionality reduction | PCA / autoencoders project into lower-dim embedding spaces |
    | Vector databases | HNSW indexes and searches embedding spaces efficiently |
    | RAG | Embed queries and documents, retrieve by cosine similarity |
    | Recommender systems | Two-tower = user embedding + item embedding in shared space |
    | Fine-tuning (LoRA) | Adapt embedding spaces to new domains without full retraining |
    | **CLIP (this notebook)** | Image embedding + text embedding in shared space |
    | Canopy (product) | Job description embedding + profile embedding in shared space |

    **The meta-pattern of modern AI**:
    > Encode everything into dense vectors. Compute similarity. Retrieve or rank.
    > Every system I've studied is a variant of this.

    **What changes across systems**:
    - The encoder architecture (CNN, ViT, transformer, two-tower, autoencoder)
    - The training objective (contrastive, masked LM, reconstruction, supervised)
    - The index structure (HNSW, IVF, PQ, flat)
    - The similarity metric (cosine, dot product, L2 — usually equivalent after normalization)

    **What stays the same**:
    - Encode → Index → Query → Retrieve
    - Shared embedding space for cross-entity comparison
    - Scale (more data + compute) improves embedding quality
    """)
    return


@app.cell
def unified_diagram_code():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#fafafa")

    # Central embedding space circle
    center_circle = plt.Circle((6, 2.5), 1.2, color="#4A90D9", alpha=0.15, zorder=1)
    ax.add_patch(center_circle)
    center_circle2 = plt.Circle((6, 2.5), 1.2, color="#4A90D9", fill=False, linewidth=2, zorder=2)
    ax.add_patch(center_circle2)
    ax.text(6, 2.5, "Shared\nEmbedding\nSpace", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#2c5f8a", zorder=3)

    # Input modalities (left side)
    modalities = [
        (0.8, 4.2, "Text", "#E8F5E9", "#2E7D32"),
        (0.8, 3.1, "Images", "#E3F2FD", "#1565C0"),
        (0.8, 2.0, "Audio", "#FFF3E0", "#E65100"),
        (0.8, 0.9, "Code", "#F3E5F5", "#6A1B9A"),
    ]

    # Output tasks (right side)
    tasks = [
        (11.2, 4.2, "RAG retrieval", "#E8F5E9", "#2E7D32"),
        (11.2, 3.1, "Image search", "#E3F2FD", "#1565C0"),
        (11.2, 2.0, "Recommendations", "#FFF3E0", "#E65100"),
        (11.2, 0.9, "Code search", "#F3E5F5", "#6A1B9A"),
    ]

    # Draw modality boxes
    for x, y, label, bg, fg in modalities:
        rect = mpatches.FancyBboxPatch((x - 0.7, y - 0.3), 1.4, 0.6,
                                       boxstyle="round,pad=0.05", color=bg,
                                       ec=fg, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color=fg, zorder=3)
        # Arrow from modality to center
        ax.annotate("", xy=(4.82, 2.5), xytext=(x + 0.72, y),
                    arrowprops=dict(arrowstyle="->", color=fg, lw=1.4,
                                    connectionstyle="arc3,rad=0.0"), zorder=2)

    # Draw task boxes
    for x, y, label, bg, fg in tasks:
        rect = mpatches.FancyBboxPatch((x - 0.75, y - 0.3), 1.5, 0.6,
                                       boxstyle="round,pad=0.05", color=bg,
                                       ec=fg, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=fg, zorder=3)
        # Arrow from center to task
        ax.annotate("", xy=(x - 0.77, y), xytext=(7.18, 2.5),
                    arrowprops=dict(arrowstyle="->", color=fg, lw=1.4,
                                    connectionstyle="arc3,rad=0.0"), zorder=2)

    # Encoder labels on left arrows
    encoders = ["Text encoder", "CLIP ViT", "Whisper", "CodeBERT"]
    enc_y =    [4.2, 3.1, 2.0, 0.9]
    for enc, y in zip(encoders, enc_y):
        ax.text(3.1, y + 0.22, enc, ha="center", va="bottom", fontsize=7.5,
                color="#555", style="italic")

    ax.set_title("The Unified Embedding Pattern — One Architecture, All Modalities",
                 fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    return fig, matplotlib, mpatches, np, plt


@app.cell
def show_unified(fig, mo):
    mo.md("""
    Every retrieval and recommendation system encodes its entities into a shared embedding space,
    then retrieves by nearest neighbor. CLIP extends this to cross-modal retrieval.
    """)
    return mo.image(fig)


@app.cell
def flashcards(mo):
    mo.md("""
    ---
    ## Flashcard Summary

    **What is CLIP?**
    → Two-tower model that encodes images and text into the same 512-dim embedding space.
    Trained with contrastive learning on 400M (image, text) pairs from the internet. No manual labels.

    **How does contrastive learning work?**
    → For a batch of N image-text pairs, compute an N×N similarity matrix. Pull diagonal pairs
    (correct matches) together, push off-diagonal pairs (wrong matches) apart. InfoNCE loss =
    softmax cross-entropy on both rows (image→text) and columns (text→image).

    **What is zero-shot classification?**
    → Classify images by cosine similarity to text descriptions of candidate categories.
    No task-specific training needed — just write the category names. Works because CLIP's shared
    embedding space maps semantically related images and texts to nearby vectors.

    **How does a Vision Transformer (ViT) work?**
    → Split image into 16×16 patches, flatten each patch into a vector, add positional embeddings,
    prepend a [CLS] token, feed through a standard transformer encoder. The [CLS] token output is
    the image embedding. Patches attend to each other via multi-head self-attention.

    **How do multi-modal LLMs process images?**
    → Encode image patches with a vision encoder (CLIP ViT), project into the LLM's token
    embedding space via a projection layer, then process interleaved visual + text tokens
    through the LLM's standard attention mechanism.

    **CLIP vs sentence-transformers?**
    → Same architecture pattern. Both are bi-encoders with a shared embedding space trained
    contrastively. Sentence-transformers: text→text similarity. CLIP: image↔text similarity.
    CLIP is the cross-modal generalization of the same idea.

    **When would you fine-tune CLIP?**
    → Domain-specific tasks where pre-trained CLIP lacks exposure: medical imaging (x-rays),
    satellite imagery (land use), industrial inspection (defect detection). Use LoRA on 10K–100K
    domain (image, text) pairs. Pre-trained CLIP is the initialization, not a blank slate.

    **What connects CLIP to recommender systems?**
    → Both are two-tower architectures. User encoder + item encoder → shared embedding space →
    dot product similarity → rank. CLIP: image encoder + text encoder → shared embedding space →
    cosine similarity → match. The computation is identical; the encoders and training data differ.

    **CNN vs ViT for embeddings?**
    → CNNs have translation invariance built in (inductive bias) — better with small data.
    ViTs have minimal inductive bias — need more data but scale better. ViT features are
    state-of-the-art at large scale. CLIP uses ViT. ResNet still fine for constrained settings.
    """)
    return


@app.cell
def interview_talking_points(mo):
    mo.md("""
    ---
    ## Interview Talking Points

    **"Explain CLIP"**
    > "Two encoders — one for images (ViT), one for text (transformer) — trained to produce
    > embeddings in the same 512-dim space. Trained with contrastive learning on 400M image-text
    > pairs from the web: matching pairs are pulled together in embedding space, non-matching pairs
    > pushed apart. No task-specific labels needed. Enables zero-shot classification, cross-modal
    > image search, and image-text retrieval without fine-tuning."

    **"How does multi-modal AI connect to your work?"**
    > "The two-tower architecture in CLIP is the same pattern I use everywhere. Canopy's job
    > matching uses a bi-encoder: job description encoder and candidate profile encoder, ranked
    > by cosine similarity. My RAG pipeline uses a query encoder and document encoder. My
    > recommender systems notebook shows user and item encoders. CLIP extends this pattern to
    > images — same architecture, different modalities. Understanding the shared pattern means I
    > can apply it to any new domain."

    **"How would you add image understanding to a RAG system?"**
    > "Embed images with CLIP alongside text chunks in the same vector store. At query time,
    > retrieve both text chunks and images by similarity. Pass everything to a multi-modal LLM
    > like GPT-4o or Claude for generation. For the Briefing Agent processing arXiv papers,
    > this would capture architecture diagrams and result charts that text-only RAG misses —
    > a query about 'transformer attention patterns' could retrieve Figure 3 from the attention
    > paper directly."

    **"What's the connection between attention and CLIP's similarity computation?"**
    > "CLIP's similarity is `image_embedding · text_embedding` — a dot product between two
    > vectors. Attention computes `Q · Kᵀ` — dot products between query and key vectors.
    > They're the same operation. Attention asks 'how relevant is this key to this query?'
    > CLIP asks 'how relevant is this image to this text?' Same math, different modalities.
    > The dot product is the universal relevance score in modern AI."

    **"CNN vs ViT — when would you use each?"**
    > "ViT for large-scale, high-quality embeddings — it scales better and is what CLIP uses.
    > CNN (ResNet, EfficientNet) for constrained settings: small datasets where the translation
    > invariance bias helps, or edge deployment where ViT's compute is too expensive. In practice,
    > for new embedding systems I'd start with a pre-trained CLIP ViT — it's already on most
    > cloud ML platforms and the pre-training is exceptional."

    **"How would you build an image moderation system without labeled data?"**
    > "CLIP zero-shot classification. Encode candidate unsafe categories as text: 'a violent
    > image', 'an explicit image'. Score every uploaded image against these categories with
    > cosine similarity. Flag anything above a threshold. Add new categories by writing text
    > descriptions — no retraining. The limitation: accuracy is lower than a fine-tuned classifier
    > on your specific definition. Use CLIP as a first-pass filter, route borderline cases to
    > human review."
    """)
    return


if __name__ == "__main__":
    app.run()
