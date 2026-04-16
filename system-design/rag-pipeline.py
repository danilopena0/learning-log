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
    # Design a RAG System — Chunking, Embeddings, Retrieval, Reranking, Evaluation

    | Field  | Value |
    |--------|-------|
    | Date   | 2026-04-14 |
    | Track  | System Design |
    | Time   | 60 min |
    | Topics | RAG Architecture · Chunking · Embeddings · Hybrid Retrieval · Reranking · RAGAS Evaluation |
    """)
    return


@app.cell
def stage1_problem_framing(mo):
    mo.md("""
    ## Stage 1: Problem Framing

    > **Scenario:** "Design a RAG system for a parenting Q&A assistant. Users ask questions
    > about child development, nutrition, sleep, and behavior. The system retrieves from a
    > corpus of pediatrician-vetted articles and generates grounded answers."

    ---

    ### Clarifying Questions + Assumptions

    **Corpus size?**
    → ~10K articles, ~50M tokens. Updated weekly with new pediatric content.

    **Query volume?**
    → 100 queries/min peak, 50K/day. Latency budget **<3s end-to-end** (retrieval + generation).

    **Hallucination tolerance?**
    → **ZERO for medical advice.** Every claim must cite a source. If no good context is
    retrieved → say *"I don't have a reliable answer"* rather than guess. This is a hard
    product constraint, not a nice-to-have.

    **Languages?**
    → English only initially.

    **Personalization?**
    → Child age range from the user profile influences retrieval (a question about "sleep
    regression" means something different for a 4-month-old vs a 2-year-old).

    ---

    ### Success Metrics

    **Retrieval quality:**
    - **Recall@10** — does the right document make the top 10 retrieved chunks?
    - **MRR (mean reciprocal rank)** — how highly does the *most* relevant chunk rank?

    **Generation quality (RAGAS):**
    - **Faithfulness** — is every claim in the answer supported by the retrieved context?
    - **Answer relevance** — does the answer actually address the question asked?
    - **Context precision** — of retrieved chunks, how many were actually relevant?
    - **Context recall** — did retrieval surface all the information needed to answer?

    **Business metrics:**
    - % of questions answered vs deflected ("I don't know")
    - User thumbs-up/thumbs-down rate on answers
    - Follow-up question rate (proxy for answer completeness)

    ---

    ### The Eval Trap

    > *"Naive eval = 'looks good to me' on 5 examples.
    > Production eval = RAGAS on 200+ test cases run in CI/CD."*

    The difference between a prototype and a production RAG system is automated,
    reproducible evaluation. Without it, every change to chunking, retrieval params,
    or the prompt is a guess.
    """)
    return


@app.cell
def stage2_chunking(mo):
    mo.md("""
    ## Stage 2: Document Ingestion & Chunking

    Chunking is the most underrated decision in RAG system design. Bad chunking
    compounds at every downstream stage — retrieval, reranking, and generation all
    suffer from poorly bounded chunks.

    ---

    ### Chunking Strategies and Trade-offs

    | Strategy | How it works | Pros | Cons |
    |----------|-------------|------|------|
    | **Fixed-size** (e.g., 512 tokens) | Split by token count | Simple, predictable | Splits mid-sentence, breaks concepts |
    | **Sentence-based** | Split on sentence boundaries | Natural language boundaries | Variable size, breaks logical sections |
    | **Recursive character splitting** | Try paragraph → sentence → word boundaries | Good baseline, respects structure | Doesn't understand semantics |
    | **Semantic chunking** | Use embeddings to detect topic shifts | Best quality | More expensive, needs tuning |
    | **Document-structure aware** | Respect H1/H2/H3, lists, tables | Best for structured docs (HTML, Markdown) | Requires parsing |

    **LangChain default:** `RecursiveCharacterTextSplitter` — a solid baseline for most corpora.

    ---

    ### Chunk Size Trade-off

    - **Too small (100 tokens):** loses context. Retrieval matches keywords without meaning.
      The chunk might be the right article but say nothing useful in isolation.
    - **Too large (2000+ tokens):** dilutes relevance. A high similarity score reflects
      only a small fraction of the chunk's content. Wastes context window and slows embedding.
    - **Sweet spot: 256–512 tokens** with 10–20% overlap between adjacent chunks.

    **Why overlap matters:** prevents losing context at chunk boundaries. If a key sentence
    straddles two chunks, at least one chunk will contain it fully.

    ---

    ### Decision for My Parenting Corpus

    - Articles are well-structured with headers (Sleep, Feeding, Milestones, When to Call Doctor)
    - Use **document-structure aware chunking** respecting H2/H3 section headers
    - Target **400 tokens** with **50-token overlap**
    - Each chunk gets metadata attached at ingestion time:

    | Metadata field | Value | Used for |
    |----------------|-------|----------|
    | `article_id` | UUID | Citation rendering |
    | `section_header` | e.g., "Sleep Regression at 4 months" | Context in prompt |
    | `child_age_range` | e.g., "3-6 months" | Pre-filtering at retrieval |
    | `last_updated` | ISO date | Recency filtering |
    | `author_credentials` | e.g., "MD, FAAP" | Trust signal in response |
    """)
    return


@app.cell
def chunking_code(mo):
    mo.md("""
    ### Chunking Code Example

    Illustrative — runs without API keys.
    """)
    return


@app.cell
def _():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    sample_article = """
    ## Sleep Regression at 4 Months

    Around 4 months, many babies experience a significant sleep change often called the
    4-month sleep regression. This is actually a permanent change in how your baby's sleep
    cycles mature — they are now cycling through light and deep sleep like adults do.

    Previously, babies fell into deep sleep quickly. Now they fully wake between cycles
    (every 45–90 minutes) and may struggle to resettle without help.

    ### Signs of the 4-Month Sleep Regression

    - Suddenly waking more often at night after previously sleeping longer stretches
    - Shorter naps (45 minutes or less, ending at the end of one sleep cycle)
    - Increased fussiness and difficulty settling

    ### What Helps

    Introducing independent sleep skills — the ability to fall asleep without being fed
    or rocked — is the most reliable long-term solution. This doesn't require any specific
    method; consistency and timing matter more than technique.
    """

    article_metadata = {
        "article_id": "art_00142",
        "section_header": "Sleep Regression at 4 Months",
        "child_age_range": "3-6 months",
        "last_updated": "2025-11-15",
        "author_credentials": "MD, FAAP",
    }

    raw_chunks = splitter.split_text(sample_article)

    # Attach metadata to every chunk
    chunks_with_metadata = [
        {"text": chunk, "metadata": {**article_metadata, "chunk_index": i}}
        for i, chunk in enumerate(raw_chunks)
    ]

    for c in chunks_with_metadata:
        print(f"--- Chunk {c['metadata']['chunk_index']} ({len(c['text'].split())} words) ---")
        print(c["text"].strip())
        print(f"Metadata: {c['metadata']}\n")

    return article_metadata, chunks_with_metadata, raw_chunks, sample_article, splitter


@app.cell
def stage3_embeddings(mo):
    mo.md("""
    ## Stage 3: Embeddings

    Embeddings map text into a high-dimensional vector space where semantically similar
    text lands near each other. The quality of your embeddings directly bounds the
    quality of your retrieval — no reranker can recover from embeddings that don't
    understand your domain.

    ---

    ### Embedding Model Selection

    | Model | Quality | Cost | Trade-offs |
    |-------|---------|------|------------|
    | **OpenAI text-embedding-3-small** | High | ~$0.02/1M tokens | Easy API, vendor lock-in |
    | **OpenAI text-embedding-3-large** | Higher | ~$0.13/1M tokens | 3072-dim vectors, slower |
    | **Cohere embed-v3** | High | Per token, comparable | Strong multilingual support |
    | **BGE-large-en** (open source) | High | Free (self-hosted) | Top MTEB leaderboard, 512 token limit |
    | **E5-large / GTE-large** | High | Free (self-hosted) | Strong on asymmetric retrieval tasks |
    | **Domain fine-tuned** | Highest | Engineering cost | 5–15% recall gain, requires labeled pairs |

    **Decision framework:**
    Start with `text-embedding-3-small` (API) or `BGE-large` (self-hosted).
    Only invest in fine-tuning if recall is the bottleneck *after* reranking is in place.

    ---

    ### Vector Database Selection

    | DB | Type | Best for | My project uses |
    |----|------|----------|-----------------|
    | **ChromaDB** | Embedded / self-hosted | Small corpora, local dev, simple integration | ✅ |
    | **Pinecone** | Managed cloud | Production at scale, managed ops | |
    | **Weaviate** | Self-hosted / cloud | Hybrid search built-in, GraphQL API | |
    | **Qdrant** | Self-hosted / cloud | Fast, production-ready, open source | |
    | **pgvector** | Postgres extension | Already using Postgres, want simplicity | |

    **Why I chose ChromaDB for my QA project:**
    Small corpus (~10K articles), embedded mode means no extra service to deploy,
    easy local development, trivial FastAPI integration. If the corpus grew to 1M+ docs
    I'd migrate to Qdrant or Pinecone.

    ---

    ### Indexing Strategy

    | Index type | Algorithm | Best for | Notes |
    |------------|-----------|----------|-------|
    | **Flat (exact)** | Brute-force cosine similarity | <100K vectors | Always accurate, no tuning |
    | **HNSW** | Hierarchical Navigable Small Worlds | 100K–100M vectors | Sub-linear search, tunable via `efSearch` |
    | **IVF** | Inverted File Index | 1M+ vectors | Lower memory than HNSW, requires training step |

    For 10K articles → **flat is fine**. For 1M+ → **HNSW** is the production standard.
    ChromaDB uses HNSW under the hood (via hnswlib).
    """)
    return


@app.cell
def stage4_retrieval(mo):
    mo.md("""
    ## Stage 4: Retrieval

    Retrieval is where most RAG systems fail. Getting this right matters more than
    any model upgrade.

    ---

    ### Retrieval Strategies

    **Dense retrieval** (vector similarity):
    - Encodes query and documents into the same embedding space, finds nearest neighbors
    - Captures semantic meaning and paraphrases
    - Misses exact keyword matches (drug names, specific dosages, ages)

    **Sparse retrieval** (BM25 / TF-IDF):
    - Classic keyword/lexical matching
    - Handles exact terms and rare entities well
    - Misses synonyms, paraphrases, concept-level queries

    **Hybrid retrieval** (the production standard):
    - Run both dense and sparse in parallel, merge results with **Reciprocal Rank Fusion (RRF)**
    - "Pure dense misses queries with specific entities. Pure sparse misses paraphrased questions.
      Hybrid fixes both."

    ---

    ### Query Understanding Layer

    Before retrieval, transform the raw query to improve recall:

    | Technique | What it does | Latency cost | When to use |
    |-----------|-------------|--------------|-------------|
    | **Query rewriting** | LLM expands query ("baby won't sleep" → "infant sleep problems, night waking, sleep regression") | +100–300ms | Ambiguous / short queries |
    | **HyDE** | LLM writes a hypothetical answer, embed *that*, search with it | +500ms | When queries are very short; often outperforms raw query embedding |
    | **Multi-query** | Generate 3–5 query variations, retrieve for each, merge | +300ms | High-stakes queries where recall matters most |

    Each technique adds latency and LLM cost. Use sparingly for high-value queries;
    use HyDE only if baseline recall is measurably below target.

    ---

    ### Metadata Filtering

    - **Pre-filter** by hard constraints before vector search: `child_age_range`, recency (`last_updated > 2023`)
    - **Post-filter** for nice-to-haves: preferred author credentials, content type
    - Critical insight: pre-filtering shrinks the search space → *both faster AND more relevant*

    ---

    ### For My QA System

    - Hybrid: ChromaDB dense search + BM25 (`rank_bm25` library), merged via RRF
    - Pre-filter by `child_age_range` from user profile before ANN search
    - Return **top-k = 20 candidates** to feed into reranking
    """)
    return


@app.cell
def rrf_code(mo):
    mo.md("""
    ### Hybrid Retrieval with RRF — Code Example

    Runs without API keys. Shows the fusion logic clearly.
    """)
    return


@app.cell
def _():
    def reciprocal_rank_fusion(
        dense_results: list[str],
        sparse_results: list[str],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion: merge two ranked lists into a single ranking.

        Score for each doc = sum of 1 / (k + rank) across both lists.
        k=60 is the standard default — dampens the influence of very high ranks.
        """
        scores: dict[str, float] = {}

        for rank, doc_id in enumerate(dense_results, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        for rank, doc_id in enumerate(sparse_results, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


    # Simulate dense (semantic) and sparse (BM25) retrieval results
    # These represent chunk IDs returned by each retrieval method
    dense_hits = ["chunk_142_s2", "chunk_089_s1", "chunk_201_s4", "chunk_055_s3", "chunk_310_s1"]
    sparse_hits = ["chunk_055_s3", "chunk_142_s2", "chunk_412_s2", "chunk_089_s1", "chunk_198_s1"]

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits)

    print("Hybrid retrieval results (RRF fusion):")
    print(f"{'Rank':<6} {'Chunk ID':<20} {'RRF Score':<12} {'In dense?':<12} {'In sparse?'}")
    print("-" * 65)
    for rank, (chunk_id, score) in enumerate(fused, start=1):
        in_dense = "✓" if chunk_id in dense_hits else "-"
        in_sparse = "✓" if chunk_id in sparse_hits else "-"
        print(f"{rank:<6} {chunk_id:<20} {score:<12.4f} {in_dense:<12} {in_sparse}")

    return dense_hits, fused, reciprocal_rank_fusion, sparse_hits


@app.cell
def stage5_reranking(mo):
    mo.md("""
    ## Stage 5: Reranking

    Retrieval optimizes for **speed** (millions of docs in milliseconds).
    Reranking optimizes for **accuracy** on the small candidate set.

    ---

    ### Why the Two-Stage Architecture Works

    **Bi-encoder** (used in initial retrieval):
    - Encodes the query and each document *separately* into fixed-size vectors
    - Similarity = dot product or cosine — computed once per doc, cached
    - Fast: one pass per document, comparable across the whole index
    - Less accurate: query and document never "see" each other during encoding

    **Cross-encoder** (used in reranking):
    - Feeds query + document *together* as a single input
    - The attention mechanism can model fine-grained query–document interaction
    - Slow: must run for every (query, candidate) pair — not scalable to full index
    - Accurate: highest-quality relevance scores

    **Pattern:**
    ```
    Bi-encoder retrieval:  query → top 20 from millions  (~50ms)
    Cross-encoder rerank:  query × top 20 → top 5        (~200ms)
    ```

    ---

    ### Reranker Options

    | Reranker | Type | Quality | Latency | Notes |
    |----------|------|---------|---------|-------|
    | **Cohere Rerank** | API | High | +100–200ms | Easy, paid, strong quality |
    | **BGE-reranker-large** | Open source | High | +150–250ms | Self-hosted, comparable to Cohere |
    | **cross-encoder/ms-marco-MiniLM** | Open source | Medium | +50–100ms | Lightweight, fast, lower quality |
    | **LLM-as-reranker** | LLM prompt | Highest | +500ms+ | Most flexible, slowest, most expensive |

    ---

    ### For My QA System

    **BGE-reranker-large**: rerank top 20 → top 5 chunks.

    Adds ~200ms to latency (within the 3s budget) but measurably boosted faithfulness scores.
    The quality gain was clearest for ambiguous queries where the top semantic hit wasn't
    actually the most relevant chunk — the cross-encoder caught this where cosine similarity missed it.
    """)
    return


@app.cell
def stage6_generation(mo):
    mo.md("""
    ## Stage 6: Generation

    If retrieval is done well, generation is mostly prompt engineering and output validation.
    The LLM's job is to synthesize retrieved context into a coherent answer — not to recall facts.

    ---

    ### Prompt Engineering for Grounded Generation

    **System prompt (critical section):**
    ```
    You are a pediatric health assistant. Answer ONLY using the provided context.
    If the context does not contain enough information to answer reliably,
    respond with: "I don't have reliable information on this — please consult your pediatrician."
    Do not make up information. Cite sources by article_id for every factual claim.
    ```

    **Context formatting in the prompt:**
    Include source metadata so the LLM can construct citations:
    ```
    [Article 1247 | Section: Sleep Regression at 4 Months | Updated: 2025-11-15]
    Around 4 months, babies experience a permanent change in sleep cycle architecture...

    [Article 0892 | Section: Establishing Sleep Routines | Updated: 2025-09-30]
    A consistent bedtime routine signals to the infant that sleep is approaching...
    ```

    **Generation parameters:**
    - Temperature: **0.0–0.3** for factual grounded answers (low randomness)
    - Structured output mode with a Pydantic schema:

    ```python
    class ArticleRef(BaseModel):
        article_id: str
        section: str

    class QAResponse(BaseModel):
        answer: str
        citations: list[ArticleRef]
        confidence: Literal["high", "medium", "low", "deflected"]
    ```

    ---

    ### Citation Validation — Hard Gate

    After generation, validate before returning to the user:
    1. Parse all `article_id` values from the structured response
    2. Check each cited ID against the set of IDs that were actually in the retrieved context
    3. If any citation references a doc **not** in context → **reject and retry**, or flag for
       human review

    This catches hallucinated sources — the LLM occasionally fabricates plausible-looking
    article IDs. Citation validation is the last safety net before the user sees the answer.

    ---

    ### For My QA System

    - **GPT-4o-mini**: cost-effective at 50K queries/day volume, quality is sufficient when
      retrieval is good (the model isn't the bottleneck — retrieval is)
    - **Structured output mode**: forces valid JSON with Pydantic schema, eliminates parsing errors
    - **Hard citation validation**: catches ~2–3% of responses with hallucinated references;
      those are deflected to "I don't have reliable information"
    """)
    return


@app.cell
def stage7_ragas(mo):
    mo.md("""
    ## Stage 7: Evaluation with RAGAS

    RAGAS (Retrieval-Augmented Generation Assessment) is the standard framework for
    automated RAG evaluation. It provides LLM-judge-based metrics that correlate well
    with human ratings.

    ---

    ### RAGAS Metrics

    | Metric | What it measures | How it's computed | What a low score means |
    |--------|-----------------|-------------------|------------------------|
    | **Faithfulness** | Is every answer claim supported by context? | Extract claims from answer → check each against context | LLM is hallucinating or reasoning beyond retrieved info |
    | **Answer relevance** | Does the answer address the actual question? | Generate questions from the answer → compare to original | Answer is tangential or off-topic |
    | **Context precision** | Of retrieved chunks, how many are relevant? | LLM judges relevance of each chunk | Retrieval is returning noise alongside signal |
    | **Context recall** | Did retrieval surface all the needed information? | Compare answer against ground-truth, check if context covers it | Retrieval is missing key documents |

    ---

    ### Eval Pipeline Design

    **Test set:** 200+ Q&A pairs with:
    - The original user question
    - Ground-truth answer (written by domain expert)
    - Ground-truth source citations

    Build it once, maintain it over time. Sample production queries regularly to prevent
    the test set from drifting away from actual user behavior.

    **When to run:**
    - After every change to chunking strategy, chunk size, overlap
    - After every change to retrieval parameters (top-k, RRF weights)
    - After every prompt change
    - After every model swap (embedding model, generation model, reranker)
    - **In CI/CD on every PR — block deploy if any metric regresses**

    ---

    ### Common Eval Traps

    **Test set drift:**
    The test set was built from early user queries. 6 months later, users are asking
    different questions. Fix: log production queries, sample 10/week for human labeling,
    rotate into the test set.

    **LLM-judge bias:**
    The LLM used as the RAGAS judge has its own preferences and biases. Use the same
    judge model consistently so scores are comparable over time. Validate against human
    ratings quarterly.

    **Faithfulness ≠ correctness:**
    An answer can be fully faithful to retrieved context that is *wrong*.
    "Garbage in, garbage out." Faithfulness only tells you the LLM didn't go off-script —
    it doesn't tell you whether the retrieved context was accurate. This is why corpus
    curation and author credential metadata matter.

    **Metric gaming:**
    Optimizing a single metric can hurt others. Retrieval that returns fewer, more precise
    chunks improves context precision but may hurt context recall. Track all four metrics
    together, and align with the product constraint (for medical advice, faithfulness is
    non-negotiable; precision matters more than recall).
    """)
    return


@app.cell
def full_architecture(mo):
    mo.md("""
    ## Full System Architecture

    ```
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                              ONLINE PATH (<3s)                               │
    │                                                                              │
    │  [User Query]                                                                │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [Query Understanding Layer]                                                 │
    │      • Query rewriting (expand ambiguous terms)                              │
    │      • Extract child_age_range from user profile                             │
    │      • (Optional) HyDE for short/ambiguous queries                           │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [Pre-filter]                                                                │
    │      • Filter ChromaDB search space by child_age_range                       │
    │      • Optionally: recency filter (last_updated > threshold)                 │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [Hybrid Retrieval]                                                          │
    │      • Dense: ChromaDB ANN (cosine similarity)     ──┐                      │
    │      • Sparse: BM25 (rank_bm25)                    ──┤→ RRF fusion → top 20 │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [Cross-encoder Rerank]  (BGE-reranker-large)                                │
    │      • Score all 20 (query, chunk) pairs jointly                             │
    │      • Return top 5 chunks with highest relevance scores                     │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [Prompt Construction]                                                       │
    │      • Format top 5 chunks with article_id + section metadata                │
    │      • Prepend grounded-generation system prompt                             │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [LLM Generation]  (GPT-4o-mini, structured output, temp=0.1)               │
    │      • Output: {answer, citations: list[ArticleRef], confidence}             │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [Citation Validation]                                                       │
    │      • Every cited article_id must be in the retrieved top-5                 │
    │      • Hallucinated citation → deflect or retry                              │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [Response with cited sources to user]                                       │
    └──────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                           OFFLINE INGESTION PATH                            │
    │                                                                              │
    │  [Pediatric Articles (weekly batch)]                                         │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [Document Parser]  → extract sections, metadata (age range, author, date)  │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [Chunker]  (document-structure aware, 400 tokens / 50 overlap)              │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [Embedding Model]  → embed each chunk                                       │
    │      │                                                                       │
    │      ├──────────────────────────────────────────────┐                       │
    │      ▼                                              ▼                        │
    │  [ChromaDB — vector index]              [BM25 index — rank_bm25]            │
    └──────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                          EVAL PIPELINE (CI/CD)                              │
    │                                                                              │
    │  [PR merged]                                                                 │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [RAGAS on 200+ test set]                                                    │
    │      • Faithfulness  • Answer relevance                                      │
    │      • Context precision  • Context recall                                   │
    │      │                                                                       │
    │      ▼                                                                       │
    │  [Regression check]  →  any metric drops >5%  →  block deploy               │
    │                      →  all metrics stable    →  ship                        │
    └──────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                              MONITORING                                     │
    │                                                                              │
    │  [Production query logs]                                                     │
    │      │                                                                       │
    │      ├──►  [Sampled human review (10/week)]  →  test set updates            │
    │      ├──►  [Thumbs-down rate dashboard]       →  alert if >15%              │
    │      ├──►  [Deflection rate]                  →  alert if >30%              │
    │      └──►  [Citation validation failure rate] →  alert if >5%               │
    └──────────────────────────────────────────────────────────────────────────────┘
    ```
    """)
    return


@app.cell
def my_project_mapping(mo):
    mo.md("""
    ## My QA Project Mapping

    **What I built:**
    ChromaDB + LangChain + FastAPI deployed on Render. Pediatric content corpus,
    dense-only vector retrieval, GPT-4o-mini generation with structured citations,
    basic citation validation.

    ---

    **What I'd add for production:**

    | Gap | Why it matters | Effort |
    |-----|----------------|--------|
    | Hybrid retrieval (BM25 + dense RRF) | Specific medical terms ("pacifier", exact ages) return better results with BM25 | Medium |
    | Cross-encoder reranking (BGE-reranker) | Top semantic hit ≠ most relevant chunk for ambiguous queries | Medium |
    | RAGAS in CI/CD | Right now every change is a guess; need reproducible regression detection | High |
    | Human eval sampling | LLM-judge metrics need periodic grounding in human ratings | Ongoing |
    | Query rewriting | Users write short, ambiguous queries; expansion helps recall | Low |

    ---

    ### Lessons Learned

    **Chunk size moved the needle more than model choice.**
    Switching from 800 → 400 token chunks improved faithfulness scores more than
    upgrading to a larger embedding model. Smaller chunks are more semantically
    homogeneous — easier for both retrieval and the LLM to reason about precisely.

    **Citation validation caught real hallucinations.**
    The LLM occasionally fabricated plausible-looking article IDs when context was thin.
    Hard validation before the response reaches the user is essential — not optional —
    for a zero-hallucination-tolerance medical product.

    **Dense-only retrieval fails on specifics.**
    Queries like "safe ibuprofen dose 18-month-old" require exact keyword matching.
    Dense retrieval returns semantically similar content (pain management articles) but
    can miss the exact dosage table. BM25 finds it directly.

    ---

    ### Interview Framing

    > *"My QA project taught me that RAG is mostly a retrieval problem disguised as a
    > generation problem. If you retrieve the right context, even a small model generates
    > good answers. If you retrieve garbage, the biggest LLM in the world won't save you."*
    """)
    return


@app.cell
def follow_ups(mo):
    mo.md("""
    ## Common Interviewer Follow-Ups

    ---

    **"How do you handle queries with no good context?"**

    > Faithfulness check in the prompt + explicit deflection path: "I don't have reliable
    > information on this — please consult your pediatrician." Track the deflection rate as
    > a metric. High deflection (>30%) = retrieval coverage gap → add more documents.
    > Deflecting is always better than hallucinating for medical advice.

    ---

    **"How do you keep the corpus fresh?"**

    > Daily ingestion job: scrape/receive new articles, parse, chunk, embed, upsert into
    > ChromaDB (upsert by article_id preserves existing chunks while updating stale ones).
    > Atomic index swap: build the new index version in parallel, swap when complete to
    > avoid serving degraded results during a rebuild. Track stale doc warnings via
    > `last_updated` field in metadata.

    ---

    **"What if a user asks something where multiple sources contradict each other?"**

    > Return both perspectives with citations, explicitly flagging the disagreement:
    > "Sources differ on this — [Article A] recommends X while [Article B] suggests Y.
    > Consult your pediatrician for guidance specific to your child." Or flag for human
    > review if the conflict is high-stakes. Never silently pick one source.

    ---

    **"How do you scale to 100M+ documents?"**

    > - HNSW index (already provided by ChromaDB via hnswlib) with tuned `efSearch`
    > - Horizontal sharding by topic cluster (each shard handles a domain subset)
    > - Distributed reranking (the cross-encoder is the latency bottleneck at scale)
    > - Query result caching: common questions return the same top-5 chunks →
    >   cache at the retrieval layer with a short TTL (corpus changes weekly)

    ---

    **"Cost optimization?"**

    > - **Cache embeddings**: embed once at ingestion, query many times → no per-query embedding cost
    > - **Cache common Q&A pairs**: LRU cache of (query hash → validated response) with 24h TTL
    > - **Route by confidence**: if retrieval returns very high similarity scores, skip reranking
    > - **Smaller reranker for low-stakes queries**: use MiniLM for queries below a confidence threshold,
    >   BGE-large only for ambiguous / high-stakes queries
    > - **Model tiering**: GPT-4o-mini for standard queries, skip to GPT-4o only if structured output
    >   parsing fails after two retries

    ---

    **"How do you know your system is hallucinating?"**

    > Layered detection:
    > 1. **Citation validation** (synchronous, hard gate): every cited article_id must appear
    >    in the retrieved context — catches fabricated sources before the user sees them
    > 2. **Faithfulness via RAGAS** (async, on test set): LLM judge checks every answer
    >    claim against the context it was generated from
    > 3. **Thumbs-down rate** (production signal): user feedback is the ground truth
    > 4. **Sampled human review** of low-confidence answers (confidence = "low" in structured output)
    """)
    return


@app.cell
def talking_points(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    ### 60-Second Elevator Pitch

    > *"I'd build a hybrid retrieval RAG with cross-encoder reranking. The pipeline is:
    > query understanding → metadata pre-filter → hybrid retrieval (dense + BM25, merged
    > with RRF) → cross-encoder rerank top 20 to top 5 → grounded generation with
    > structured citations → hard citation validation → RAGAS eval in CI/CD.*
    >
    > *The biggest leverage points in this stack are chunking strategy and reranking —
    > not model choice. A well-chunked corpus with hybrid retrieval + reranking will
    > outperform a poorly chunked corpus with the best embedding model available."*

    ---

    ### The Thing That Impresses Interviewers

    > *"I treat RAG as a retrieval problem first. Most failures I've seen come from bad
    > retrieval, not bad generation. I'd invest in eval infrastructure — RAGAS in CI/CD
    > with a maintained test set — before touching the generation model. You can't improve
    > what you can't measure."*

    This signals production maturity. Anyone can assemble a RAG prototype in an afternoon.
    The discipline of eval infrastructure is what separates prototype from production.

    ---

    ### Connection to My Work

    > *"My parenting QA project on Render uses this stack but simplified: ChromaDB for
    > vector storage, LangChain for orchestration, dense-only retrieval, GPT-4o-mini
    > with citation validation. The main production gap is hybrid retrieval and RAGAS in CI/CD.*
    >
    > *The biggest lesson from building it: chunk size and metadata filtering moved
    > faithfulness scores more than any model swap. Smaller, metadata-rich chunks are
    > worth the ingestion engineering investment."*

    ---

    ### Red Flags to Avoid

    - Jumping to "just use GPT-4" without discussing retrieval quality
    - Forgetting that faithfulness ≠ correctness (faithful to bad context is still wrong)
    - Not mentioning eval infrastructure — retrieval quality without measurement is a guess
    - Treating chunk size as a trivial hyperparameter rather than a core architectural decision
    - No deflection path for low-confidence / no-context scenarios — especially for medical
    """)
    return


if __name__ == "__main__":
    app.run()
