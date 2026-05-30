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
    # AI Eng Mock Interview — Practice Explaining My Agent Architecture, RAG Pipeline, Tool Calling Patterns

    | Field   | Value                                                                                  |
    |---------|----------------------------------------------------------------------------------------|
    | Date    | 2026-05-27                                                                             |
    | Track   | AI Engineering (Interview Prep)                                                        |
    | Time    | 60 min                                                                                 |
    | Topic   | Project walkthroughs · Agent architecture · RAG pipeline · Tool calling · Monitoring  |
    """)
    return


@app.cell
def how_to_use(mo):
    mo.md("""
    ## How to Use This Notebook

    This is a **mock interview**, not a lecture.

    For each question:
    1. **Read the question** (bold, top of each section)
    2. **Close the answer** (click the ▶ triangle to collapse it — or scroll past)
    3. **Answer OUT LOUD** for 60–90 seconds. Time yourself.
    4. **Open the scripted answer** and compare
    5. **Note what you missed or said poorly** — write it down

    > **Practice tip:** Record yourself answering. Listen back.
    > The gap between what you *think* you said and what you *actually* said
    > is where the improvement lives.

    **Focus areas for this session:** project walkthroughs, architecture explanations,
    technical depth on agents / RAG / tool calling.

    **Reference projects:** Canopy · Daily Briefing Agent · Backtesting Engine · multi-agent-lab
    """)
    return


@app.cell
def q1_prompt(mo):
    mo.md("""
    ---

    ## Question 1 — ~90 seconds

    ### **"Tell me about an AI system you've built."**

    *Read this, then collapse the answer below and respond out loud.*
    """)
    return


@app.cell
def q1_answer(mo):
    mo.md("""
    <details>
    <summary><strong>▶ Scripted Answer (click to reveal)</strong></summary>

    > "I built a Daily AI Research Briefing Agent — a LangGraph-based pipeline that runs on
    > GitHub Actions every morning. It ingests from arXiv, Reddit, and Twitter, filters by my
    > research interests using embedding similarity in ChromaDB, generates structured summaries
    > with free-tier LLMs, and delivers a curated briefing to Slack.
    >
    > The architecture has four layers. **Ingestion:** three independent scrapers with error
    > boundaries — if Reddit is down, arXiv and Twitter still run. **Filtering:** hybrid approach
    > using keyword pre-filter then embedding similarity against topic vectors in ChromaDB, with
    > deduplication against previously seen content. **Reasoning:** a LangGraph workflow with
    > conditional edges — if no relevant content passes the filter, it skips the expensive
    > summarization step. **Delivery:** structured Slack message with citations, plus a JSONL log
    > of every run for monitoring.
    >
    > The whole thing runs at zero cost — free-tier APIs, embedded ChromaDB, GitHub Actions for
    > orchestration. That constraint forced better engineering decisions than an unlimited budget
    > would have."

    </details>

    <details>
    <summary>What makes this answer good</summary>

    - Starts with **WHAT it does** (one sentence)
    - Gives the architecture in **layers** — organized, not rambling
    - Mentions specific technologies naturally (LangGraph, ChromaDB — not name-dropping)
    - Highlights a **design decision with reasoning** (error boundaries, conditional edges, zero-cost constraint)
    - Ends with a **senior insight**: constraints improve engineering
    - Total: ~90 seconds, dense, no filler

    </details>
    """)
    return


@app.cell
def followup_1a_prompt(mo):
    mo.md("""
    ---

    ### Follow-up 1a — ~60 seconds

    ### **"What was the hardest technical challenge?"**
    """)
    return


@app.cell
def followup_1a_answer(mo):
    mo.md("""
    <details>
    <summary><strong>▶ Scripted Answer (click to reveal)</strong></summary>

    > "ChromaDB persistence between GitHub Actions runs. Actions are ephemeral — every run starts
    > with a fresh filesystem. I needed the vector store to persist so I could deduplicate against
    > previously seen content.
    >
    > The solution was artifact caching — save the ChromaDB directory as a GitHub Actions artifact
    > at the end of each run, restore it at the start of the next. But the sequencing was tricky:
    > if the run fails after ingesting but before saving, you lose new embeddings. If you save
    > before delivery confirms, you might deduplicate content that was never actually delivered.
    >
    > I solved it by making the save step happen **after** successful delivery, with a fallback
    > that re-ingests on cache miss. It took about a day to debug because the failure modes only
    > appeared in the Actions environment, not locally."

    </details>

    <details>
    <summary>What makes this answer good</summary>

    - **Specific technical problem** — not vague "it was hard to debug"
    - Shows understanding of the **failure modes** (sequencing, partial failure)
    - Describes the solution AND why the naive approach didn't work
    - **Honest about debugging time** — shows it was a real challenge, not a toy

    </details>
    """)
    return


@app.cell
def followup_1b_prompt(mo):
    mo.md("""
    ---

    ### Follow-up 1b — ~60 seconds

    ### **"What would you improve if you had more time?"**
    """)
    return


@app.cell
def followup_1b_answer(mo):
    mo.md("""
    <details>
    <summary><strong>▶ Scripted Answer (click to reveal)</strong></summary>

    > "Three things, in priority order. First, **hybrid retrieval** — right now it's dense-only.
    > Adding BM25 via a simple keyword index plus reciprocal rank fusion would catch exact-match
    > queries that embedding similarity misses, like specific paper titles or author names.
    >
    > Second, a proper **eval framework**. I have regression tests in CI, but no LLM-as-judge
    > scoring on summary quality. I'd build a 100-case eval set and run RAGAS-style faithfulness
    > and relevance checks on every prompt change.
    >
    > Third, **cross-encoder reranking** on the filtered results before summarization. The
    > embedding similarity filter has false positives — tangentially related content that wastes
    > summarization tokens. A lightweight reranker would tighten the selection."

    </details>

    <details>
    <summary>What makes this answer good</summary>

    - **Prioritized** — not a random list
    - Each improvement is **specific and named** (hybrid retrieval with RRF, RAGAS eval, cross-encoder reranking)
    - Shows awareness of the **current system's weaknesses**
    - Each maps to a pattern from study notebooks — demonstrates depth

    </details>
    """)
    return


@app.cell
def q2_prompt(mo):
    mo.md("""
    ---

    ## Question 2 — ~90 seconds

    ### **"Walk me through your RAG pipeline."**

    *Read this, then collapse the answer below and respond out loud.*
    """)
    return


@app.cell
def q2_answer(mo):
    mo.md("""
    <details>
    <summary><strong>▶ Scripted Answer (click to reveal)</strong></summary>

    > "I'll use my parenting QA project as the example — it's a RAG system deployed on Render
    > with FastAPI.
    >
    > The pipeline has five stages. **First, document ingestion:** I chunk pediatric articles
    > using structure-aware splitting — respecting header boundaries so each chunk is a coherent
    > section, typically 400 tokens with 50-token overlap. Each chunk gets metadata: article ID,
    > section header, child age range, author credentials.
    >
    > **Second, embedding:** I use all-MiniLM-L6-v2 running locally, 384 dimensions. Chunks and
    > their metadata are stored in ChromaDB.
    >
    > **Third, retrieval:** when a user asks a question, I embed the query and search ChromaDB for
    > the top 10 most similar chunks. I pre-filter by child age range from the user's profile.
    >
    > **Fourth, generation:** the top chunks go into a prompt with GPT-4o-mini. The system prompt
    > instructs: answer ONLY from provided context, cite sources by article ID, say 'I don't have
    > reliable information' if context is insufficient. I use structured output mode to get a JSON
    > response with answer, citations, and confidence score.
    >
    > **Fifth, validation:** I verify every cited article ID actually appears in the retrieved
    > context. If a citation references something not in context — that's a hallucination — the
    > response is rejected and regenerated.
    >
    > The key insight: RAG is mostly a retrieval problem. If you retrieve the right context, even
    > small models generate good answers. If you retrieve garbage, the biggest model won't save you."

    </details>

    <details>
    <summary>What makes this answer good</summary>

    - **Structured** — five numbered stages, interviewer can follow along
    - **Specific numbers** — 400 tokens, 50 overlap, 384 dim, top 10
    - Mentions specific models and tools naturally
    - Includes the **validation step** — most candidates skip this
    - Ends with a **reusable insight** that shows experience

    </details>
    """)
    return


@app.cell
def followup_2a_prompt(mo):
    mo.md("""
    ---

    ### Follow-up 2a — ~45 seconds

    ### **"How do you handle queries where the context isn't sufficient?"**
    """)
    return


@app.cell
def followup_2a_answer(mo):
    mo.md("""
    <details>
    <summary><strong>▶ Scripted Answer (click to reveal)</strong></summary>

    > "The system prompt explicitly instructs the model to say 'I don't have reliable information
    > on this topic' when context is insufficient. The structured output includes a confidence
    > score — if it's below 0.5, the frontend shows a disclaimer and suggests the user consult
    > their pediatrician.
    >
    > I track the deflection rate as a metric. If it's too high, it means my corpus has gaps I
    > need to fill. If it's too low, the model might be hallucinating instead of deflecting —
    > that's when I check the faithfulness scores.
    >
    > The important thing is: a confident wrong answer about children's health is worse than no
    > answer. I'd rather over-deflect than under-deflect."

    </details>
    """)
    return


@app.cell
def followup_2b_prompt(mo):
    mo.md("""
    ---

    ### Follow-up 2b — ~60 seconds

    ### **"How would you evaluate retrieval quality separately from generation quality?"**
    """)
    return


@app.cell
def followup_2b_answer(mo):
    mo.md("""
    <details>
    <summary><strong>▶ Scripted Answer (click to reveal)</strong></summary>

    > "I evaluate them independently because the failure modes are different.
    >
    > **For retrieval:** context precision — of the chunks retrieved, what percentage is actually
    > relevant to the question? And context recall — of the information needed to answer correctly,
    > how much did retrieval surface? I compute these using a labeled eval set of 50 questions
    > with annotated source articles.
    >
    > **For generation:** faithfulness — does the answer only make claims supported by the
    > retrieved context? And answer relevance — does it actually address the question asked?
    > Both scored by an LLM-as-judge with a rubric.
    >
    > When quality drops, I look at these separately. If context precision is low, the problem is
    > retrieval — I need better embeddings, reranking, or query rewriting. If faithfulness is low,
    > the problem is generation — I need better prompting, a stronger model, or tighter output
    > validation. Fixing the wrong one wastes effort."

    </details>
    """)
    return


@app.cell
def q3_prompt(mo):
    mo.md("""
    ---

    ## Question 3 — ~75 seconds

    ### **"Explain how tool calling works in your agent system."**

    *Read this, then collapse the answer below and respond out loud.*
    """)
    return


@app.cell
def q3_answer(mo):
    mo.md("""
    <details>
    <summary><strong>▶ Scripted Answer (click to reveal)</strong></summary>

    > "In my briefing agent, the LLM doesn't execute tools directly — it decides WHICH tool to
    > call and with WHAT arguments, then my code executes and feeds the result back.
    >
    > The tool definitions are JSON schemas describing each tool's name, description, and parameter
    > types. The LLM sees these schemas in its context. When it determines a tool call is needed,
    > it returns a structured response with the tool name and arguments. My dispatch loop matches
    > the tool name to a Python function, executes it with the provided arguments, captures the
    > result, and appends it to the conversation as an observation.
    >
    > In Canopy, the pattern is similar but sequential: scrape tool → score tool → notify tool.
    > Each tool has a Pydantic contract defining its input and output schemas. The key design
    > decision: **tools are independently testable**. I can test the scorer with a fake job
    > description and a mocked LLM, without the scraper or notifier. If something breaks in
    > production, the structured logs tell me exactly which tool failed and with what inputs.
    >
    > The guardrails: max 5 tool call iterations to prevent infinite loops, timeout per tool call
    > with fallback, and output validation through Pydantic — if the LLM returns malformed tool
    > arguments, the validation catches it before execution."

    </details>
    """)
    return


@app.cell
def followup_3a_prompt(mo):
    mo.md("""
    ---

    ### Follow-up 3a — ~45 seconds

    ### **"How do you handle tool call failures?"**
    """)
    return


@app.cell
def followup_3a_answer(mo):
    mo.md("""
    <details>
    <summary><strong>▶ Scripted Answer (click to reveal)</strong></summary>

    > "Three layers. **First, per-tool retry with exponential backoff** — if the arXiv API times
    > out, retry at 5 seconds, then 15, then 45. Three attempts total.
    >
    > **Second, graceful degradation.** Each tool is wrapped in an error boundary. If the arXiv
    > scraper fails after retries, the pipeline continues with Reddit and Twitter data. Partial
    > results are better than no results. The Slack delivery includes a note: 'arXiv unavailable
    > today.'
    >
    > **Third, structured error logging.** Every tool failure logs: tool name, error type, input
    > arguments, timestamp. I can replay any failed tool call locally from the logs. This is the
    > same observability pattern from my LLM serving notebook — log everything, debug later."

    </details>
    """)
    return


@app.cell
def followup_3b_prompt(mo):
    mo.md("""
    ---

    ### Follow-up 3b — ~45 seconds

    ### **"How do you decide between giving the LLM more tools vs keeping it simple?"**
    """)
    return


@app.cell
def followup_3b_answer(mo):
    mo.md("""
    <details>
    <summary><strong>▶ Scripted Answer (click to reveal)</strong></summary>

    > "I default to fewer tools with well-defined interfaces. Every tool I add increases the
    > LLM's decision space — more ways to choose wrong. I've found that 3–5 well-designed tools
    > outperform 10 mediocre ones.
    >
    > The decision framework: does this tool represent a **genuinely different capability**? If
    > two tools overlap, merge them. Is the LLM's tool selection accurate? I test this with an
    > eval set — 20 queries where I know which tool should be called. If selection accuracy drops
    > below 90%, I have too many tools or confusing descriptions.
    >
    > In my multi-agent-lab, I experimented with giving one agent access to 8 tools vs splitting
    > into two specialized agents with 4 tools each. The two-agent setup had better tool selection
    > accuracy because each agent's decision space was simpler."

    </details>
    """)
    return


@app.cell
def q4_prompt(mo):
    mo.md("""
    ---

    ## Question 4 — ~75 seconds

    ### **"How do you monitor an LLM system in production?"**

    *Read this, then collapse the answer below and respond out loud.*
    """)
    return


@app.cell
def q4_answer(mo):
    mo.md("""
    <details>
    <summary><strong>▶ Scripted Answer (click to reveal)</strong></summary>

    > "I monitor at three levels. **Operational metrics** are real-time: latency, error rate,
    > token usage, and cost per request. These go to a dashboard with alerts. If p99 latency
    > exceeds 5 seconds or error rate exceeds 2%, I get paged.
    >
    > **Quality metrics** are sampled: I run an LLM-as-judge on 10% of daily traffic, scoring
    > faithfulness, relevance, and format compliance. This catches slow degradation that
    > operational metrics miss — like the model becoming more verbose over time, or a provider
    > silently updating their model.
    >
    > **Drift metrics** are daily: I track input query distribution (are users asking about new
    > topics my system wasn't designed for?) and output score distribution (are scores clustering
    > differently than the baseline?). KL divergence above a threshold triggers an investigation.
    >
    > For my briefing agent specifically, I monitor through structured JSONL logs: items ingested,
    > items filtered, items summarized, total tokens, cost, and run duration. If items_ingested
    > drops to zero for two consecutive days, something is wrong with the scrapers. If the filter
    > passes everything, the relevance threshold needs tightening."

    </details>
    """)
    return


@app.cell
def q5_prompt(mo):
    mo.md("""
    ---

    ## Question 5 — ~90 seconds

    ### **"Compare your projects. What patterns do they share?"**

    *Read this, then collapse the answer below and respond out loud.*
    """)
    return


@app.cell
def q5_answer(mo):
    mo.md("""
    <details>
    <summary><strong>▶ Scripted Answer (click to reveal)</strong></summary>

    > "The unifying pattern across all my projects is: **encode → retrieve → reason → act → evaluate.**
    >
    > **Canopy:** encode job descriptions and my profile as embeddings, retrieve similar jobs via
    > vector search, reason about fit with an LLM scorer, act by delivering scored results and
    > generating cover letters, evaluate with a labeled test set.
    >
    > **Briefing Agent:** encode research content and topic preferences, retrieve relevant items
    > via embedding similarity, reason about importance with LLM summarization, act by delivering
    > to Slack, evaluate with regression tests in CI.
    >
    > **Backtesting Engine:** encode market data as feature vectors (technical indicators),
    > retrieve relevant historical windows via walk-forward splits, reason about strategy
    > performance with the backtesting logic, act by producing strategy artifacts, evaluate with
    > out-of-sample metrics.
    >
    > The patterns that transfer: error boundaries around independent components, structured
    > logging for observability, Pydantic contracts between pipeline stages, evaluation
    > infrastructure that runs automatically, and the two-tower encode-then-compare architecture
    > that appears everywhere from vector search to recommendation to RAG.
    >
    > The meta-lesson: once you see that every AI system is encode → retrieve → reason → act →
    > evaluate, you can build in any domain. The encoders and reasoners change, the architecture
    > doesn't."

    </details>

    <details>
    <summary>What makes this answer exceptional</summary>

    - Shows **pattern recognition across domains** — the #1 signal of a senior engineer
    - Specific to real projects, not abstract
    - Names the shared patterns explicitly (error boundaries, Pydantic contracts, two-tower)
    - Ends with a synthesizing insight that's **memorable and quotable**

    </details>
    """)
    return


@app.cell
def self_critique_rubric(mo):
    mo.md("""
    ---

    ## Self-Critique Rubric

    After each practice answer, score yourself 1–5:

    | Dimension | 1 (Poor) | 3 (OK) | 5 (Great) |
    |-----------|----------|--------|-----------|
    | **Structure** | Rambling, no clear flow | Some organization | Numbered stages, clear beginning/middle/end |
    | **Specificity** | Vague ("we used ML") | Some details | Exact tools, numbers, design decisions |
    | **Trade-offs** | Only describes what was done | Mentions one alternative | Explains WHY this choice over alternatives |
    | **Honesty** | Overblows impact | Acknowledges some gaps | Openly discusses failures and what you'd improve |
    | **Timing** | >2 min or <30 sec | ~90 seconds but uneven | 60–90 seconds, well-paced |
    | **Connection** | Isolated answer | Links to one other topic | Connects to broader patterns across projects |

    **Target: 4+ on every dimension.**
    If any dimension scores 2 or below, rehearse that specific question until it improves before moving on.

    ---

    ### Score sheet — fill in after each run

    | Question | Structure | Specificity | Trade-offs | Honesty | Timing | Connection | Notes |
    |----------|-----------|-------------|------------|---------|--------|------------|-------|
    | Q1 — AI system I built | | | | | | | |
    | Follow-up 1a — hardest challenge | | | | | | | |
    | Follow-up 1b — what I'd improve | | | | | | | |
    | Q2 — RAG pipeline | | | | | | | |
    | Follow-up 2a — insufficient context | | | | | | | |
    | Follow-up 2b — eval retrieval vs gen | | | | | | | |
    | Q3 — tool calling | | | | | | | |
    | Follow-up 3a — tool failures | | | | | | | |
    | Follow-up 3b — fewer vs more tools | | | | | | | |
    | Q4 — production monitoring | | | | | | | |
    | Q5 — patterns across projects | | | | | | | |
    """)
    return


@app.cell
def rapid_fire_round(mo):
    mo.md("""
    ---

    ## Rapid-Fire Round

    Answer each in **under 30 seconds** — one or two sentences max.
    No collapsing needed here: the answers are intentionally short. Read the question, look away, answer, check.

    ---

    **"What's the difference between an agent and a pipeline?"**
    > A pipeline has fixed steps. An agent decides its own steps based on intermediate results.
    > Most production systems are pipelines with agentic components at specific decision points.

    ---

    **"Why LangGraph over LangChain chains?"**
    > My workflows have branching and conditional loops. Chains are linear. LangGraph's graph
    > structure makes that explicit and inspectable.

    ---

    **"How do you prevent hallucination in RAG?"**
    > Citation validation — every source the model cites must exist in the retrieved context.
    > If it doesn't, reject and regenerate.

    ---

    **"What's your tech stack?"**
    > Python, LangGraph for orchestration, ChromaDB and sqlite-vec for embeddings, FastAPI and
    > Litestar for APIs, Polars and DuckDB for data processing, React for frontends, GitHub
    > Actions for CI/CD.

    ---

    **"Biggest lesson from building AI systems?"**
    > The LLM is 20% of the system. The other 80% is guardrails, eval, structured parsing,
    > error handling, and monitoring. That's what makes it production-grade.

    ---

    **"Why should we hire you?"**
    > I've built and deployed agentic AI systems end-to-end — not just prompts, but the full
    > stack: retrieval, tool orchestration, structured outputs, eval pipelines, and production
    > monitoring. I think in architectural patterns that transfer across domains, and I ship.
    """)
    return


if __name__ == "__main__":
    app.run()
