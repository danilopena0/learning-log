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
    # RAG Advanced Patterns

    | Field | Value |
    |-------|-------|
    | Date | 2026-05-05 |
    | Track | AI Engineering |
    | Time | 60 min |
    | Topic | Query Rewriting · HyDE · Reranking · Multi-Modal RAG · Evaluation Frameworks |

    **Sequel to**: Week 3 RAG System Design notebook (architecture). This notebook goes deep
    on the techniques that take RAG from "works in a demo" to "works in production." Every
    pattern maps back to **Canopy** (job search retrieval) and the **Briefing Agent** (research
    summarization).
    """)
    return


@app.cell
def where_basic_rag_fails(mo):
    mo.md("""
    ## Where Basic RAG Fails

    **Basic RAG**: embed query → cosine similarity search → stuff top K into prompt → generate.

    | # | Failure mode | Example |
    |---|-------------|---------|
    | 1 | **Query mismatch** | User asks "what jobs let me work from home?" — JDs say "remote-eligible." Embedding match fails. |
    | 2 | **Ranking quality** | Top K by cosine includes irrelevant chunks that share vocabulary. LLM hallucinates from noisy context. |
    | 3 | **Multi-hop reasoning** | "Compare USAA vs H-E-B benefits" requires retrieving from BOTH and synthesizing. Single pass misses one. |
    | 4 | **Stale context** | Retrieved docs are outdated. LLM generates confidently wrong answers from old data. |
    | 5 | **Wrong granularity** | Query needs a specific number from a table but retrieval returns a 500-token paragraph around it. |

    > "Basic RAG gets you to 70% quality. These advanced patterns get you to 90%+.
    > The gap is where production systems live."
    """)
    return


@app.cell
def shared_imports():
    import asyncio
    import json
    import time
    from dataclasses import dataclass, field
    from typing import Any, Optional

    return Any, Optional, asyncio, dataclass, field, json, time


@app.cell
def part1_query_rewriting_why(mo):
    mo.md("""
    ---
    ## Part 1: Query Rewriting — Fix the Query Before Searching

    **The fundamental problem**: user queries are short, ambiguous, and use different vocabulary
    than the documents.

    ```
    "python ml jobs san antonio"
         ↓  the user MEANS
    "machine learning engineer or data scientist requiring Python,
     located in or near San Antonio, TX, onsite or hybrid"
    ```

    The embedding of the short query is a **noisy approximation** of the user's actual intent.
    Query rewriting uses an LLM to expand or clarify the query **before** retrieval — not after.

    ### Two patterns

    | Pattern | When to use |
    |---------|-------------|
    | **Query expansion** | Ambiguous queries, short keyword queries, synonym-heavy domains |
    | **Step-back prompting** | Specific questions that need broader context first |
    """)
    return


@app.cell
def query_expansion_code(json, asyncio):
    QUERY_EXPANSION_PROMPT = """
You are a search query optimizer. Given a user's search query, generate 3 alternative
phrasings that capture the same intent but use different vocabulary.

Original query: {query}

Return a JSON list of 3 alternative queries. Each should:
- Use synonyms and related terms
- Be more specific where the original is vague
- Cover different aspects of the same intent

Example:
Original: "python ml jobs san antonio"
Alternatives: [
    "machine learning engineer Python San Antonio TX",
    "data scientist role Python programming San Antonio hybrid onsite",
    "AI engineer ML Python developer jobs near San Antonio Texas"
]

Now generate alternatives for the query above.
"""

    def mock_llm_expand(query: str) -> list[str]:
        """Mock expansion — replace with real LLM call in production."""
        expansions = {
            "python ml jobs san antonio": [
                "machine learning engineer Python San Antonio TX",
                "data scientist Python programming San Antonio hybrid onsite",
                "AI engineer ML Python developer jobs near San Antonio Texas",
            ],
            "research summarization agent": [
                "automated research paper summarization pipeline",
                "LLM-powered literature review tool",
                "AI assistant for synthesizing academic papers",
            ],
        }
        return expansions.get(query, [f"{query} expanded", f"{query} alternative", f"{query} variant"])

    def reciprocal_rank_fusion(result_lists: list[list[str]], k: int = 60) -> list[str]:
        """Merge ranked result lists using RRF scoring."""
        scores: dict[str, float] = {}
        for results in result_lists:
            for rank, doc_id in enumerate(results):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    class MockVectorStore:
        def query(self, q: str, k: int = 10) -> list[str]:
            # Returns mock doc IDs — each query returns slightly different overlapping sets
            base = hash(q) % 100
            return [f"doc_{(base + i) % 50}" for i in range(k)]

    async def expanded_retrieval(query: str, vector_store: MockVectorStore, k: int = 10) -> list[str]:
        """Retrieve using original + expanded queries, merge with RRF."""
        expansions = mock_llm_expand(query)
        all_queries = [query] + expansions

        result_lists = [vector_store.query(q, k=k) for q in all_queries]
        merged = reciprocal_rank_fusion(result_lists)
        return merged[:k]

    # Demo
    store = MockVectorStore()
    demo_query = "python ml jobs san antonio"
    demo_results = asyncio.run(expanded_retrieval(demo_query, store, k=5))
    print(f"Query: '{demo_query}'")
    print(f"Top 5 merged results (RRF): {demo_results}")

    return (
        QUERY_EXPANSION_PROMPT,
        MockVectorStore,
        expanded_retrieval,
        mock_llm_expand,
        reciprocal_rank_fusion,
    )


@app.cell
def step_back_prompting_code(json):
    STEP_BACK_PROMPT = """
Given this specific question, generate a broader "step-back" question that would
help retrieve more comprehensive context.

Specific question: {query}

The step-back question should be more general — asking about the broader concept
or category that this specific question falls under.

Example:
Specific: "Does USAA's ML team use PyTorch or TensorFlow?"
Step-back: "What is USAA's machine learning technology stack and infrastructure?"

Return only the step-back question.
"""

    STEP_BACK_RESPONSES = {
        "Does USAA's ML team use PyTorch or TensorFlow?":
            "What is USAA's machine learning technology stack and infrastructure?",
        "What's the latest on RLHF alternatives?":
            "What are the current approaches to aligning large language models?",
        "What is the salary range for senior ML engineers at H-E-B?":
            "What are the compensation structures and benefits at H-E-B's technology division?",
    }

    def mock_step_back(query: str) -> str:
        return STEP_BACK_RESPONSES.get(
            query,
            f"What is the broader context behind: {query}"
        )

    def step_back_retrieval(query: str, vector_store, k: int = 5) -> list[str]:
        """Retrieve from both specific query and step-back query, merge."""
        step_back_query = mock_step_back(query)
        specific_results = vector_store.query(query, k=k)
        broad_results = vector_store.query(step_back_query, k=k)
        from itertools import chain
        seen, merged = set(), []
        for doc_id in chain(specific_results, broad_results):
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc_id)
        return merged[:k]

    # Demo
    q = "Does USAA's ML team use PyTorch or TensorFlow?"
    print(f"Original:  {q}")
    print(f"Step-back: {mock_step_back(q)}")

    return STEP_BACK_PROMPT, mock_step_back, step_back_retrieval


@app.cell
def query_rewriting_projects(mo):
    mo.md("""
    ### My Projects Mapping

    **Canopy** — query expansion would help massively. "python ml jobs" gets expanded to capture
    "machine learning," "data science," "AI engineer" variants that appear in JDs with different
    wording. This directly addresses the vocabulary mismatch that kills naive embedding search
    on short queries.

    **Briefing Agent** — step-back prompting for research queries. "What's the latest on RLHF
    alternatives?" → step-back: "What are the current approaches to aligning large language
    models?" Captures DPO, RLAIF, constitutional AI papers the specific query would miss.
    """)
    return


@app.cell
def part2_hyde_concept(mo):
    mo.md("""
    ---
    ## Part 2: HyDE — Hypothetical Document Embeddings

    **The insight**: instead of embedding the SHORT QUERY (noisy), ask the LLM to write a
    HYPOTHETICAL ANSWER, then embed THAT.

    **Why this works**: the hypothetical answer is in the same "language" as the documents.
    Its embedding is geometrically closer to relevant documents than the raw query embedding.

    ```
    User query: "What San Antonio companies are hiring ML engineers?"
         ↓  LLM generates hypothetical answer
    "Several San Antonio companies are actively hiring ML engineers. USAA has roles focused
     on fraud detection and risk modeling. Rackspace is building AI/ML teams for cloud
     optimization. H-E-B's digital division has data science positions..."
         ↓  embed the hypothetical (NOT the query)
    [0.12, -0.34, 0.87, ...]  ← embedding in document space
         ↓  ANN search
    [relevant JDs and company pages]
    ```

    The hypothetical answer doesn't need to be **correct** — it just needs to be in the right
    semantic neighborhood.
    """)
    return


@app.cell
def hyde_implementation(asyncio):
    HYDE_PROMPT = """
Write a detailed paragraph that would be a good answer to this question.
The answer doesn't need to be factually accurate — just plausible and detailed
enough to capture the right concepts and terminology.

Question: {query}

Write a 100-150 word hypothetical answer:
"""

    def mock_llm_hypothetical(query: str) -> str:
        responses = {
            "What San Antonio companies are hiring ML engineers?": (
                "Several San Antonio companies are actively hiring ML engineers. USAA has roles "
                "focused on fraud detection and risk modeling using Python and PyTorch. Rackspace "
                "is building AI/ML teams for cloud optimization and intelligent infrastructure. "
                "H-E-B's digital division has data science positions working on demand forecasting "
                "and recommendation systems. CPS Energy is hiring for predictive maintenance. "
                "The San Antonio ML market is growing, with most roles preferring hybrid or "
                "onsite candidates with 3-5 years of production ML experience."
            ),
            "How does transformer attention work?": (
                "Transformer attention computes a weighted sum of value vectors, where weights "
                "are derived from the compatibility between query and key vectors. Each token "
                "attends to all other tokens via scaled dot-product attention: softmax(QK^T / sqrt(d_k))V. "
                "Multi-head attention runs this in parallel across h heads, each learning different "
                "relationship patterns. Self-attention allows each position to attend to all others, "
                "enabling long-range dependencies without the sequential bottleneck of RNNs."
            ),
        }
        return responses.get(query, f"A plausible detailed answer to: {query} would involve relevant "
                                     "terminology, specific examples, and domain-appropriate vocabulary "
                                     "that appears in related documents.")

    def mock_embed(text: str) -> list[float]:
        """Mock embedding — returns a fingerprint-style vector for demo."""
        h = hash(text)
        return [(h >> i & 1) * 0.1 - 0.05 for i in range(16)]

    class MockVectorStoreWithVectorQuery:
        def query_by_vector(self, vector: list[float], k: int = 10) -> list[str]:
            base = int(sum(abs(v) for v in vector) * 100) % 100
            return [f"doc_{(base + i) % 50}" for i in range(k)]

        def query(self, q: str, k: int = 10) -> list[str]:
            base = hash(q) % 100
            return [f"doc_{(base + i) % 50}" for i in range(k)]

    async def hyde_retrieval(query: str, vector_store: MockVectorStoreWithVectorQuery, k: int = 10) -> list[str]:
        """Retrieve using HyDE + direct query, merged with RRF."""
        hypothetical = mock_llm_hypothetical(query)
        hyde_embedding = mock_embed(hypothetical)
        direct_embedding = mock_embed(query)

        hyde_results = vector_store.query_by_vector(hyde_embedding, k=k)
        direct_results = vector_store.query_by_vector(direct_embedding, k=k)

        # Merge: HyDE + direct query via RRF (if HyDE is wrong, direct still contributes)
        seen, merged = set(), []
        for doc_id in hyde_results + direct_results:
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc_id)
        return merged[:k]

    # Demo
    q = "What San Antonio companies are hiring ML engineers?"
    hyp = mock_llm_hypothetical(q)
    store2 = MockVectorStoreWithVectorQuery()
    results = asyncio.run(hyde_retrieval(q, store2, k=5))
    print(f"Query: '{q}'")
    print(f"\nHypothetical (first 120 chars): {hyp[:120]}...")
    print(f"\nTop 5 HyDE+direct results: {results}")

    return HYDE_PROMPT, hyde_retrieval, mock_embed, mock_llm_hypothetical


@app.cell
def hyde_tradeoffs(mo):
    mo.md("""
    ### When HyDE Helps vs Hurts

    | Scenario | Effect |
    |----------|--------|
    | Factual questions with domain vocabulary | **Helps** — hypothetical uses correct terms |
    | Complex queries where vocabulary mismatch is the main problem | **Helps** |
    | LLM hallucinates wrong terminology (wrong domain) | **Hurts** — finds documents matching wrong answer |
    | Simple keyword queries that already work | **Wasteful** — adds LLM call with no gain |

    **Cost**: one LLM call per query. Worth it for complex queries, wasteful for simple ones.

    **Mitigation**: always combine HyDE with direct query retrieval via RRF (implemented above).
    If HyDE goes wrong, the direct query still contributes.

    > "I'd use HyDE selectively in Canopy — only for ambiguous queries where the first
    > retrieval pass returns low-confidence results."
    """)
    return


@app.cell
def part3_reranking_why(mo):
    mo.md("""
    ---
    ## Part 3: Reranking — The Quality Multiplier

    **Initial retrieval (bi-encoder)** optimizes for SPEED:
    - Encode query and documents separately
    - Fast ANN search (milliseconds for millions of docs)
    - But: separate encoding can't compare query and document jointly — misses nuance

    **Cross-encoder reranking** optimizes for PRECISION:
    - Take top 20-50 candidates from retrieval
    - Feed each `(query, document)` pair jointly through a model
    - The model sees both together → can reason about their relationship
    - Typical improvement: **10-20% recall@5** over retrieval alone

    This is the **single highest-ROI addition** to any RAG system.

    ```
    Retrieval:  [doc_3, doc_17, doc_42, doc_1, doc_9, ...]   ← fast, noisy (top 20)
         ↓  cross-encoder scores each pair jointly
    Reranked:   [doc_17, doc_1, doc_42, ...]                  ← slow, precise (top 5)
    ```
    """)
    return


@app.cell
def reranking_implementation():
    # sentence-transformers cross-encoder (runs locally, free)
    # pip install sentence-transformers
    # from sentence_transformers import CrossEncoder
    # reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def mock_cross_encoder_scores(query: str, documents: list[str]) -> list[float]:
        """Mock cross-encoder — in production: reranker.predict([(query, doc) for doc in documents])"""
        import hashlib
        scores = []
        for doc in documents:
            combined = query + doc
            h = int(hashlib.md5(combined.encode()).hexdigest(), 16)
            scores.append((h % 1000) / 1000.0)
        return scores

    def rerank(query: str, documents: list[str], top_k: int = 5) -> list[tuple[int, str, float]]:
        """Rerank documents by cross-encoder relevance score."""
        scores = mock_cross_encoder_scores(query, documents)
        ranked = sorted(
            [(i, documents[i], scores[i]) for i in range(len(documents))],
            key=lambda x: x[2],
            reverse=True,
        )
        return ranked[:top_k]

    # Demo: retrieval returns 20, reranker picks best 5
    import time
    query = "senior ML engineer San Antonio Python"
    candidate_docs = [f"Job description for role {i} at company {chr(65 + i % 8)}" for i in range(20)]

    t0 = time.time()
    reranked = rerank(query, candidate_docs, top_k=5)
    rerank_ms = (time.time() - t0) * 1000

    print(f"Query: '{query}'")
    print(f"Candidates: 20 docs → Reranked top 5 in {rerank_ms:.1f}ms")
    print()
    for rank, (idx, doc, score) in enumerate(reranked, 1):
        print(f"  {rank}. [score={score:.3f}] {doc}")

    return rerank, mock_cross_encoder_scores


@app.cell
def reranker_options(mo):
    mo.md("""
    ### Reranker Options

    | Reranker | Quality | Speed | Cost |
    |----------|---------|-------|------|
    | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Good | ~5ms/pair | Free (local) |
    | `BAAI/bge-reranker-v2-m3` | Great | ~15ms/pair | Free (local) |
    | Cohere Rerank | Excellent | ~20ms/pair | $1/1K queries |
    | ColBERT (late interaction) | Great | Fast at scale | Free (local), complex setup |
    | LLM-as-reranker (GPT-4/Claude) | Best | ~500ms | Expensive |

    **Pick**: BGE-reranker for free + high quality. Cohere for managed simplicity.
    LLM-as-reranker only for very high-stakes queries.

    ### My Projects Mapping

    **Canopy**: currently no reranking. Adding `cross-encoder/ms-marco-MiniLM-L-6-v2` on
    top-20 sqlite-vec results would be ~50 lines of code and probably the **single
    highest-impact improvement** I could make. This is gap #5 from my Canopy gap analysis.

    **Briefing Agent**: rerank research paper results by relevance to the day's focus topic.
    Currently takes top K by embedding similarity — sometimes surfaces tangentially related
    papers that share vocabulary but aren't on-topic.
    """)
    return


@app.cell
def part4_advanced_architectures(mo):
    mo.md("""
    ---
    ## Part 4: Advanced Retrieval Architectures

    ### Contextual Retrieval (Anthropic's method)

    **Problem**: when you chunk a document, each chunk loses context of the whole.
    "Treatment of fever in children under 2" makes sense inside a pediatric article —
    as a standalone chunk it's ambiguous.

    **Solution**: prepend a short context blurb to each chunk **before embedding**.

    ```
    Original chunk:
    "Treatment should begin with acetaminophen at 10-15mg/kg..."

    Contextualized chunk (what gets embedded):
    "[From: AAP Guidelines on Pediatric Fever Management, Section 3: Treatment Protocols]
     Treatment should begin with acetaminophen at 10-15mg/kg..."
    ```

    **Implementation**: at index time, for each chunk ask an LLM:
    *"Given the full document, write a 1-2 sentence context for this chunk."*
    Prepend to chunk text before embedding.

    **Cost**: one LLM call per chunk at **index time** (not query time). Affordable for static
    corpora. Run it once when you build the index, not on every query.
    """)
    return


@app.cell
def contextual_retrieval_code():
    CONTEXTUAL_CHUNK_PROMPT = """
<document>
{full_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document
for the purposes of improving search retrieval of the chunk. Answer only with the succinct
context and nothing else.
"""

    def mock_contextualize(document: str, chunk: str) -> str:
        return f"[From: {document[:40]}...] {chunk}"

    def build_contextual_index(documents: list[dict]) -> list[dict]:
        """At index time: contextualize every chunk, then embed."""
        chunks = []
        for doc in documents:
            for chunk in doc["chunks"]:
                context = mock_contextualize(doc["title"], chunk)
                contextualized = f"{context}\n\n{chunk}"
                chunks.append({
                    "doc_id": doc["id"],
                    "text": chunk,
                    "contextualized_text": contextualized,
                    # embed contextualized_text, not text
                })
        return chunks

    # Demo
    sample_docs = [
        {
            "id": "usaa-jd-001",
            "title": "USAA Senior ML Engineer Job Description",
            "chunks": [
                "Required: 5+ years Python, experience with PyTorch or TensorFlow.",
                "Benefits include 15% 401k match, 4 weeks PTO, remote-eligible.",
                "The ML Platform team builds fraud detection models serving 13M members.",
            ],
        }
    ]

    contextual_chunks = build_contextual_index(sample_docs)
    print("Contextual chunks built at index time:\n")
    for c in contextual_chunks:
        print(f"  Original: {c['text'][:60]}")
        print(f"  Embedded: {c['contextualized_text'][:90]}")
        print()

    return CONTEXTUAL_CHUNK_PROMPT, build_contextual_index, contextual_chunks


@app.cell
def parent_child_retrieval(mo):
    mo.md("""
    ### Parent-Child Retrieval

    **Problem**: small chunks embed well (focused, precise) but lack surrounding context.
    Large chunks have context but embed poorly (diluted signal across many topics).

    **Solution**: index SMALL chunks (256 tokens) for precise matching, return the PARENT
    chunk (1024 tokens) for rich context.

    ```
    Index:    [small chunk 1] → [small chunk 2] → [small chunk 3]
                    ↑                   ↑                   ↑
              all point to same parent chunk (1024 tokens)

    Query time:
    1. Retrieve by small chunk similarity  (precise match)
    2. Look up parent chunk pointer        (from metadata)
    3. Return parent chunk to LLM          (rich context)
    ```

    The LLM gets the broader context it needs to generate a good answer, while retrieval
    operates on precise, focused embeddings.

    ### Agentic RAG

    Instead of a fixed retrieve-then-generate pipeline, let an agent **decide** what to retrieve.

    The agent can:
    - **Decide what** to search for (reformulate based on what it already knows)
    - **Decide when** to search (maybe it already knows the answer)
    - **Decide if** results are good enough (if not, reformulate and retry)
    - **Search multiple sources** (vector store, BM25, web search, SQL)

    This is the ReAct pattern from the agent-architectures notebook applied to RAG.

    > "My Briefing Agent's research node is agentic RAG — it decides which source
    > (arXiv, Reddit, Twitter) to query based on topic, and can retry with a refined
    > query if initial results are thin."
    """)
    return


@app.cell
def part5_multimodal_rag(mo):
    mo.md("""
    ---
    ## Part 5: Multi-Modal RAG

    **Standard RAG**: text in, text out.
    **Multi-modal RAG**: handle images, tables, charts, PDFs with visual elements.

    **Why this matters**: many real documents have critical information in figures,
    tables, and diagrams that text-only RAG completely misses.

    ### Three approaches

    | Approach | How | Best for |
    |----------|-----|----------|
    | **Extract-and-embed** | OCR tables, describe images with vision model, embed descriptions | PDFs, structured docs |
    | **Native multi-modal embeddings** | CLIP embeds images + text into same vector space | Image-heavy corpora |
    | **Vision LLM at generation** | Pass retrieved images/charts directly to GPT-4o/Claude | High-value complex documents |
    """)
    return


@app.cell
def multimodal_implementation(mo):
    mo.md("""
    ### Practical Implementation

    **For PDFs with tables** (common in job descriptions, reports):

    ```python
    import pdfplumber
    import camelot

    def extract_tables_as_markdown(pdf_path: str) -> list[str]:
        tables = camelot.read_pdf(pdf_path, pages="all")
        markdown_tables = []
        for table in tables:
            df = table.df
            markdown_tables.append(df.to_markdown(index=False))
        return markdown_tables

    # Then: embed markdown tables alongside regular text chunks
    # When retrieved, LLM can reason over the table structure
    ```

    **For images and charts**:

    ```python
    async def describe_image(image_path: str, vision_llm) -> str:
        # Generate text description of the image
        description = await vision_llm.generate(
            f"Describe this chart/image in detail, including all numerical values and trends.",
            image=image_path,
        )
        return description

    def index_document_with_images(pdf_path: str, vector_store):
        text_chunks = extract_text_chunks(pdf_path)
        image_descriptions = [
            {"text": describe_image(img), "image_path": img, "type": "figure"}
            for img in extract_images(pdf_path)
        ]
        # Store image path as metadata — pass original to multi-modal LLM at generation
        all_chunks = text_chunks + image_descriptions
        vector_store.add(all_chunks)
    ```

    ### When Multi-Modal Matters

    | Document type | What's lost by text-only RAG |
    |--------------|------------------------------|
    | Job descriptions | Salary tables, benefit comparison charts, org chart images |
    | Research papers | Figures, result tables, architecture diagrams — often the most important parts |
    | Business documents | Financial charts, process diagrams, dashboards |

    > "For Canopy, most JDs are text-heavy — multi-modal isn't critical yet. For the
    > Briefing Agent processing arXiv papers, extracting figure descriptions would
    > capture key results that abstracts miss."
    """)
    return


@app.cell
def part6_evaluation_problem(mo):
    mo.md("""
    ---
    ## Part 6: RAG Evaluation Frameworks

    ### The Evaluation Problem

    RAG has **two independent components** to evaluate:

    | Component | Measures | How it fails |
    |-----------|----------|--------------|
    | **Retrieval quality** | Did I find the right documents? | Low precision: noisy context. Low recall: missed relevant docs. |
    | **Generation quality** | Given those docs, did I generate a good answer? | Hallucination, ignored context, wrong format. |

    ```
    Bad retrieval + Good generation  →  confident wrong answer (hallucination from noise)
    Good retrieval + Bad generation  →  wasted potential (had the info, didn't use it)
    ```

    You must evaluate **both independently** to know where to improve.
    """)
    return


@app.cell
def ragas_metrics(mo):
    mo.md("""
    ### RAGAS Metrics

    | Metric | Question it answers | How it's computed |
    |--------|--------------------|--------------------|
    | **Faithfulness** | Is the answer grounded in the context? | Claims in answer ÷ claims supported by context |
    | **Answer relevance** | Does the answer address the question? | Embed reverse-generated questions, compare to original |
    | **Context precision** | Are retrieved chunks actually useful? | Relevant chunks ÷ total chunks retrieved |
    | **Context recall** | Did we retrieve everything needed? | Requires ground-truth source list |
    """)
    return


@app.cell
def ragas_prompts():
    FAITHFULNESS_PROMPT = """
Given the context and the answer, extract every factual claim from the answer.
For each claim, determine if it is supported by the context.

Context: {context}
Answer: {answer}

Return a JSON object:
{{
    "claims": [
        {{"claim": "...", "supported": true/false, "evidence": "quote from context or null"}}
    ],
    "faithfulness_score": <number of supported claims / total claims>
}}
"""

    RELEVANCE_PROMPT = """
Given the question and answer, generate 3 questions that the answer would be
a good response to. Then compute how similar these generated questions are to
the original question.

Original question: {question}
Answer: {answer}

If the generated questions are similar to the original, the answer is relevant.
Score 0-1 where 1 = perfectly relevant.
"""

    CONTEXT_PRECISION_PROMPT = """
Given the question and each retrieved context chunk, rate whether the chunk
contains information useful for answering the question.

Question: {question}
Chunk: {chunk}

Rate: "relevant" or "irrelevant" with a one-sentence justification.
"""

    return CONTEXT_PRECISION_PROMPT, FAITHFULNESS_PROMPT, RELEVANCE_PROMPT


@app.cell
def eval_harness(dataclass, field, time, Any):
    import numpy as np

    @dataclass
    class RAGEvalCase:
        question: str
        ground_truth_answer: str = ""
        expected_source_ids: list = field(default_factory=list)

    @dataclass
    class RAGEvalResult:
        case: RAGEvalCase
        retrieved_docs: list
        generated_answer: str
        faithfulness: float
        relevance: float
        context_precision: float
        context_recall: float
        latency_ms: float

    # Mock scorers — replace with real LLM-as-judge calls
    def score_faithfulness(context_docs: list[str], answer: str) -> float:
        import hashlib
        h = int(hashlib.md5((str(context_docs) + answer).encode()).hexdigest(), 16)
        return round(0.5 + (h % 500) / 1000.0, 3)

    def score_relevance(question: str, answer: str) -> float:
        import hashlib
        h = int(hashlib.md5((question + answer).encode()).hexdigest(), 16)
        return round(0.5 + (h % 500) / 1000.0, 3)

    def score_context_precision(question: str, retrieved_docs: list[str]) -> float:
        import hashlib
        h = int(hashlib.md5((question + str(retrieved_docs)).encode()).hexdigest(), 16)
        return round(0.4 + (h % 600) / 1000.0, 3)

    def compute_context_recall(expected_ids: list[str], retrieved_ids: list[str]) -> float:
        if not expected_ids:
            return 1.0
        hits = len(set(expected_ids) & set(retrieved_ids))
        return round(hits / len(expected_ids), 3)

    class RAGEvalPipeline:
        """Reusable eval harness — plug in any RAG pipeline."""

        def __init__(self, pipeline_fn):
            self.pipeline_fn = pipeline_fn

        def evaluate(self, eval_cases: list[RAGEvalCase]) -> dict:
            results = []
            for case in eval_cases:
                t0 = time.time()
                retrieved_docs, answer = self.pipeline_fn(case.question)
                latency = (time.time() - t0) * 1000

                result = RAGEvalResult(
                    case=case,
                    retrieved_docs=retrieved_docs,
                    generated_answer=answer,
                    faithfulness=score_faithfulness(retrieved_docs, answer),
                    relevance=score_relevance(case.question, answer),
                    context_precision=score_context_precision(case.question, retrieved_docs),
                    context_recall=compute_context_recall(case.expected_source_ids, retrieved_docs),
                    latency_ms=latency,
                )
                results.append(result)

            summary = {
                "n": len(results),
                "avg_faithfulness": round(float(np.mean([r.faithfulness for r in results])), 3),
                "avg_relevance": round(float(np.mean([r.relevance for r in results])), 3),
                "avg_context_precision": round(float(np.mean([r.context_precision for r in results])), 3),
                "avg_context_recall": round(float(np.mean([r.context_recall for r in results])), 3),
                "avg_latency_ms": round(float(np.mean([r.latency_ms for r in results])), 1),
                "results": results,
            }
            return summary

    # Demo: mock pipeline + 3 eval cases
    def mock_rag_pipeline(question: str) -> tuple[list[str], str]:
        docs = [f"doc_{hash(question) % 20 + i}" for i in range(5)]
        answer = f"Based on the retrieved context, {question.lower().replace('?', '')}."
        return docs, answer

    eval_cases = [
        RAGEvalCase(
            question="What Python ML frameworks does USAA use?",
            expected_source_ids=["doc_3", "doc_7"],
        ),
        RAGEvalCase(
            question="What is the salary range for senior ML engineers in San Antonio?",
            expected_source_ids=["doc_12", "doc_15"],
        ),
        RAGEvalCase(
            question="Which San Antonio tech companies offer remote work?",
            expected_source_ids=["doc_1", "doc_8", "doc_19"],
        ),
    ]

    pipeline = RAGEvalPipeline(mock_rag_pipeline)
    summary = pipeline.evaluate(eval_cases)

    print("RAG Eval Results")
    print("=" * 40)
    print(f"Cases evaluated:      {summary['n']}")
    print(f"Avg faithfulness:     {summary['avg_faithfulness']}")
    print(f"Avg relevance:        {summary['avg_relevance']}")
    print(f"Avg context precision:{summary['avg_context_precision']}")
    print(f"Avg context recall:   {summary['avg_context_recall']}")
    print(f"Avg latency:          {summary['avg_latency_ms']} ms")

    return (
        RAGEvalCase,
        RAGEvalPipeline,
        RAGEvalResult,
        compute_context_recall,
        pipeline,
        score_context_precision,
        score_faithfulness,
        score_relevance,
        summary,
    )


@app.cell
def eval_frameworks_comparison(mo):
    mo.md("""
    ### Evaluation Framework Options

    | Framework | Strengths | Best for |
    |-----------|----------|----------|
    | **RAGAS** | Standard metrics, LLM-as-judge, well documented | General RAG evaluation |
    | **TruLens** | Production monitoring, feedback functions | Live system monitoring |
    | **DeepEval** | CI/CD integration, assertion-based | Testing in pipelines |
    | **Promptfoo** | Prompt A/B testing, provider comparison | Comparing prompt versions |
    | **Custom (above)** | Full control, domain-specific metrics | When standard metrics don't fit |

    **Run in CI**: every change to retrieval config or system prompt should trigger the eval suite.
    A 5-point drop in faithfulness is a regression, same as a failing unit test.
    """)
    return


@app.cell
def full_architecture(mo):
    mo.md("""
    ---
    ## The Full Advanced RAG Architecture

    ```
    [User Query]
        → [Query Router]  (simple query? skip rewriting. Complex? continue.)
        → [Query Rewriting — parallel]
            ├── Query expansion     (3 LLM-generated variants)
            ├── Step-back query     (1 broader LLM-generated query)
            └── HyDE                (1 LLM-generated hypothetical document)
        → [Multi-Source Retrieval — parallel]
            ├── Dense search        (all rewritten + original queries)
            ├── Sparse search       (BM25 on original query)
            └── Metadata pre-filter (location, date, job type, etc.)
        → [RRF Fusion]              (merge all result lists into one ranked list)
        → [Cross-Encoder Reranking] (top 50 → top 5, precise joint scoring)
        → [Context Compression]     (optional: summarize long chunks if > context window)
        → [Generation]              (LLM with structured output + inline citations)
        → [Faithfulness Check]      (verify every claim is supported by retrieved context)
        → [Response]

    ─────────────────────────────────────────────────────────────────────

    [Evaluation Pipeline — runs in CI on every retrieval or prompt change]
        → RAGAS metrics on 100+ test cases
        → A/B testing on query rewriting variants
        → Latency regression check (p95 < 2s)
    ```

    **Implementation priority for Canopy** (ordered by ROI):
    1. Cross-encoder reranking — ~50 lines, 10-20% recall improvement
    2. Hybrid search (BM25 + dense + RRF) — from vector-databases notebook
    3. Query expansion — handles vocabulary mismatch on short queries
    4. RAGAS eval suite — can't improve what you don't measure
    5. HyDE — for ambiguous queries only, combined with direct retrieval
    """)
    return


@app.cell
def flashcard_summary(mo):
    mo.md("""
    ---
    ## Flashcard Summary

    | Question | Answer |
    |----------|--------|
    | **What's query rewriting?** | Use an LLM to expand or reformulate the query before retrieval. Captures synonyms and implicit intent the raw query misses. |
    | **What's HyDE?** | Generate a hypothetical answer with an LLM, embed THAT instead of the query. Works because hypothetical is in document-space. Risk: wrong hypothetical → wrong retrieval. |
    | **Why rerank?** | Retrieval optimizes speed (bi-encoder, separate encoding). Reranking optimizes precision (cross-encoder, joint scoring). Typical 10-20% recall@5 improvement. |
    | **What's contextual retrieval?** | Prepend document-level context to each chunk before embedding. Prevents chunks from losing meaning outside their parent document. Done at index time. |
    | **What's parent-child retrieval?** | Index small chunks (256 tokens) for precise matching, return parent chunks (1024 tokens) for rich context. Best of both granularities. |
    | **How do you evaluate RAG?** | Separately evaluate retrieval (precision, recall) and generation (faithfulness, relevance). RAGAS framework. Run in CI on every change. |
    | **Faithfulness vs relevance?** | Faithfulness = answer is grounded in context (no hallucination). Relevance = answer addresses the question asked. Both can fail independently. |
    | **Multi-modal RAG?** | Extract tables/images as text descriptions, embed alongside text chunks. Pass originals to multi-modal LLM at generation time. |
    | **RRF — what and why?** | Reciprocal Rank Fusion. Merges multiple ranked lists: score = Σ 1/(60+rank). Robust to scale differences across retrieval methods. |
    | **Step-back prompting?** | Generate a broader version of the specific query, retrieve for both, merge. Captures context the specific query would miss. |
    """)
    return


@app.cell
def interview_talking_points(mo):
    mo.md("""
    ---
    ## Interview Talking Points

    **"How would you improve a basic RAG system?"**
    > "Three highest-ROI additions in order: cross-encoder reranking (10-20% recall improvement
    > for ~100ms latency cost), hybrid retrieval with BM25 + dense + RRF (catches keyword matches
    > dense search misses), and query rewriting for ambiguous queries. Reranking first — simplest
    > to add, biggest impact. I'd also add a RAGAS eval suite before touching anything, so I have
    > a baseline to measure against."

    ---

    **"When would you use HyDE?"**
    > "For complex queries where vocabulary mismatch is the main failure mode — factual questions
    > about a specific domain where the LLM knows the terminology. I'd always combine HyDE with
    > direct query retrieval via RRF — so if the hypothetical is wrong, direct retrieval still
    > contributes. Never HyDE alone. And I'd A/B test it against the baseline before shipping."

    ---

    **"How do you evaluate your RAG system?"**
    > "RAGAS metrics: faithfulness (is the answer grounded?), context precision (is retrieval
    > focused?), context recall (did we find what we needed?), answer relevance (does it address
    > the question?). 100+ eval cases curated from real usage. Every retrieval config or prompt
    > change triggers the suite in CI. A 5-point drop in faithfulness is a regression, same
    > as a failing unit test."

    ---

    **Connection to my work**
    My Week 3 RAG system design notebook specified the architecture. My vector-databases notebook
    covered hybrid retrieval with RRF. This notebook fills in the techniques: query rewriting,
    HyDE, reranking, and eval. Canopy currently uses basic dense retrieval — adding reranking
    and a RAGAS eval suite would move it from demo quality to production quality. The eval
    harness above (`RAGEvalPipeline`) is directly portable into Canopy.
    """)
    return


if __name__ == "__main__":
    app.run()
