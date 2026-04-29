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
    # Design an AI Agent System for Daily Research — Using MY Actual Architecture

    | Field  | Value |
    |--------|-------|
    | Date   | 2026-04-28 |
    | Track  | System Design |
    | Time   | 60 min |
    | Topics | LangGraph · RAG · ChromaDB · GitHub Actions · Structured Outputs · Evals in CI/CD |
    """)
    return


@app.cell
def interview_framing(mo):
    mo.md("""
    ## Interview Framing

    When an interviewer asks **"design a daily research briefing system"**, **"describe an agent you've
    built"**, or **"walk me through an AI system you've shipped"** — THIS is my answer.

    ---

    ### What makes this a strong interview artifact

    - **It's real.** I built it, deployed it, and use it every day. I can answer every follow-up
      from lived experience, not from theory.
    - **It uses production patterns.** RAG retrieval, tool calling, structured outputs, citation
      guardrails, evals running in CI/CD — the full AI engineering stack in one system.
    - **It runs at zero cost.** Free-tier APIs + GitHub Actions = $0/month. That constraint forced
      better engineering decisions than just throwing GPT-4 at everything.
    - **I can speak to failures.** The hardest bug, what I'd change, and why I made each design
      decision — that's the interview signal.

    > *"I'm not describing a hypothetical. I'm walking you through a system I use every day."*

    ---

    ### How to time this in a 45-minute interview

    | Section | Time |
    |---------|------|
    | Problem framing | 5 min |
    | Architecture overview | 5 min |
    | Component deep-dives (2-3 of 4) | 20 min |
    | Trade-offs + what I'd improve | 5 min |
    | Follow-up Q&A | 10 min |
    """)
    return


@app.cell
def stage1_problem(mo):
    mo.md("""
    ## Stage 1: Problem Framing

    ### The problem

    I need to stay current on AI/ML research, industry news, and market trends for my job search
    and portfolio. Reading arXiv, Reddit, and Twitter manually takes 1–2 hours/day — time I could
    spend studying or building. I want a daily briefing delivered to Slack by 8am: curated,
    summarized, and relevant to MY interests.

    ---

    ### Requirements

    **Functional**
    - Ingest from 3+ sources: arXiv, Reddit, Twitter/X
    - Filter by relevance to my topics: AI/ML, data engineering, LLMs, quantitative finance
    - Summarize key findings per item, cite real sources
    - Deliver via Slack daily by 8am CT

    **Non-functional**
    - Runs daily without manual intervention
    - Costs $0 (free-tier everything — hard requirement)
    - Reliable: doesn't silently fail; alerts me when something breaks
    - Extensible: easy to add new sources or topics

    **Quality constraints**
    - Summaries must cite real sources (no hallucinated papers)
    - Stay within topic scope (no noise)
    - Concise: < 500 words per briefing

    ---

    ### Success metrics

    | Metric | Target | Why |
    |--------|--------|-----|
    | Relevance | >80% of briefing items useful | Measured by my own weekly review |
    | Faithfulness | 100% of citations are real | Automated in CI via arxiv_id validation |
    | Reliability | >95% of days delivered on time | Tracked in structured JSONL run log |
    | Cost | $0/month | Tracked via run metadata |
    """)
    return


@app.cell
def stage2_architecture(mo):
    mo.md("""
    ## Stage 2: Architecture Overview

    ```
    ┌─────────────────────────────────────────────────────┐
    │              GITHUB ACTIONS (cron: 6am CT)          │
    │                                                     │
    │  [Trigger] → [Run Pipeline] → [Log Results]        │
    └──────────────────────┬──────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────┐
    │              INGESTION LAYER                         │
    │                                                     │
    │  [arXiv API] → [Reddit API] → [Twitter/X Crawler]  │
    │       ↓              ↓              ↓               │
    │  [Raw Content Pool] (today's new content)           │
    └──────────────────────┬──────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────┐
    │              FILTERING + RAG LAYER                   │
    │                                                     │
    │  [Topic Filter] (is this relevant to my interests?) │
    │       ↓                                             │
    │  [ChromaDB] ← embed + store new content             │
    │       ↓                                             │
    │  [Retrieval] → dedup against previously seen content│
    └──────────────────────┬──────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────┐
    │              REASONING LAYER (LangGraph)             │
    │                                                     │
    │  [Summarize Node] → structured summary per item     │
    │  [Curate Node] → rank by relevance, select top N   │
    │  [Format Node] → compose Slack message              │
    └──────────────────────┬──────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────┐
    │              DELIVERY                                │
    │                                                     │
    │  [Slack Webhook] → daily briefing in my channel     │
    │  [JSONL Log] → structured log of every run          │
    └─────────────────────────────────────────────────────┘
    ```

    ### Data flow summary

    1. GitHub Actions cron fires at 6am CT
    2. Ingestion layer fetches last 24h of content from all 3 sources independently
    3. Topic filter drops irrelevant content (keyword pre-filter → embedding similarity)
    4. ChromaDB stores new content and deduplicates against prior days
    5. LangGraph pipeline: summarize → curate → format
    6. Slack webhook delivers the briefing; JSONL log records every run
    """)
    return


@app.cell
def stage3_ingestion(mo):
    mo.md("""
    ## Stage 3: Component Deep Dive — Ingestion

    ### Source-specific design

    **arXiv**
    - API: arXiv REST API, no auth required
    - Query: `cs.AI`, `cs.LG`, `cs.CL`, `stat.ML` categories, last 24 hours
    - Extract: `title`, `abstract`, `authors`, `arxiv_id`, `published_date`
    - Rate limit: gentle (no aggressive pagination), ~100 papers/day in these categories

    **Reddit**
    - API: PRAW (Python Reddit API Wrapper), free OAuth app credentials
    - Subreddits: `r/MachineLearning`, `r/LocalLLaMA`, `r/datascience`
    - Filter: top posts in last 24h, score > 50 (signal threshold)
    - Extract: `title`, `selftext`, `url`, `score`, `num_comments`

    **Twitter/X**
    - Approach: custom crawler against a curated list of AI researchers and VCs
    - Extract: `tweet_text`, `author`, `engagement_metrics`, `linked_urls`
    - Note: Twitter API access is brittle. This is the most fragile source — acceptable because
      each source is independent.

    ---

    ### Design decisions

    | Decision | What I chose | Why | Alternative considered |
    |----------|-------------|-----|------------------------|
    | Source isolation | Separate scraper per source | Different auth, rate limits, data formats | Unified scraper (too brittle across APIs) |
    | Date filter first | Filter by last 24h before any LLM calls | Reduces volume before expensive ops | Process everything (wasteful) |
    | Store raw before filtering | Write raw content to disk/log first | Debugging, reprocessing, provenance | Filter inline (lose raw data permanently) |

    ---

    ### Failure handling

    - **Source independence**: if Reddit is down, arXiv and Twitter still run. No shared failure mode.
    - **Retry policy**: exponential backoff — 3 attempts at 5s / 15s / 45s delays
    - **Total failure path**: if all sources fail → send "ingestion failed" Slack alert instead
      of an empty or misleading briefing
    - **Every failure logged**: source name, error type, timestamp, attempt count → queryable
      from JSONL log

    > *"This is the defensive design decision I'm most proud of. The system never silently succeeds
    > with empty content — it either delivers real results or tells me it failed."*
    """)
    return


@app.cell
def stage4_rag(mo):
    mo.md("""
    ## Stage 4: Component Deep Dive — RAG + Filtering

    ### Topic relevance filtering

    Two-stage filter:

    1. **Keyword pre-filter** (fast, free): content must contain at least one term from my topic
       list (`["transformer", "LLM", "fine-tuning", "RAG", "embedding", "quantitative", ...]`).
       Drops obvious misses in microseconds, no compute cost.

    2. **Embedding similarity** (semantic, cheap): compare content embedding against topic
       embeddings stored in ChromaDB. Threshold: cosine similarity > 0.7. Catches relevant
       content that doesn't hit keywords (e.g., a paper titled "Scaling Sparse Mixtures" without
       the word "LLM").

    Why two stages: the keyword filter is O(n) string matching with no model calls.
    It cuts volume by ~60% before the embedding step. The embedding filter handles the
    semantic gap the keyword filter misses.

    ---

    ### ChromaDB usage

    - **Collection**: `research_content` with metadata: `{source, date, topic_scores}`
    - **Embedding model**: `all-MiniLM-L6-v2` (384 dim, runs locally, no API cost)
    - **Deduplication**: before adding new content, check cosine similarity against existing
      entries. If > 0.95, skip — likely a near-duplicate (same paper posted in two subreddits,
      same news item from two sources)
    - **Persistence**: ChromaDB persists to disk; survives between GitHub Actions runs via
      artifact caching. This was the hardest engineering problem — see follow-ups section.

    ---

    ### Why ChromaDB for this project

    | Factor | ChromaDB embedded | At-scale alternative |
    |--------|------------------|----------------------|
    | Corpus size | ~50–100 items/day, ~2K/month | 1M+ items |
    | Deployment | Embedded, no separate service | Qdrant / Weaviate as managed service |
    | Cost | Free, open source | ~$50–200/month for managed |
    | Index type | Flat (fine at this scale) | HNSW with tuned `ef_construction` |
    | Hybrid search | Not needed yet | BM25 + dense, would add at scale |

    > *"If I scaled to 1M+ items, I'd move to Qdrant or Weaviate. Hybrid search (BM25 + dense)
    > would also become necessary — keyword recall matters when semantic search misses exact
    > technical terms like model names or paper titles."*
    """)
    return


@app.cell
def stage5_langgraph(mo):
    mo.md("""
    ## Stage 5: Component Deep Dive — LangGraph Reasoning

    ### Graph structure

    ```
    [start]
        → [ingest_node]        # fetch from all 3 sources in parallel
        → [filter_node]        # keyword pre-filter + embedding similarity
            ├── no content → [empty_briefing_node] → [deliver_node]
            └── has content → [summarize_node]
        → [summarize_node]     # LLM summarizes each item, returns structured JSON
        → [curate_node]        # rank by relevance_score, select top 5–8 items
        → [format_node]        # compose Slack message with citations + run stats
        → [deliver_node]       # POST to Slack webhook
        → [log_node]           # write structured JSONL entry
    ```

    ### State definition

    ```python
    class BriefingState(TypedDict):
        raw_content: list[ContentItem]
        filtered_content: list[ContentItem]
        summaries: list[Summary]
        curated: list[Summary]
        slack_message: str
        metadata: RunMetadata  # cost_usd, latency_seconds, source_counts, errors
    ```

    Each `ContentItem` carries: `{source, title, body, url, arxiv_id, published_date, raw_score}`
    Each `Summary` adds: `{key_findings: list[str], relevance_score: float}`

    ---

    ### Why LangGraph over a plain script

    | Feature | Benefit for this project |
    |---------|--------------------------|
    | Conditional edges | Skip summarize_node if no relevant content — saves LLM cost |
    | Typed state | Inspect state at any node during debugging without print-statement archaeology |
    | Explicit graph | Adding a new node (e.g., "translate" for non-English papers) = one function + one edge |
    | Visualization | Export graph as diagram for documentation / interview demos |

    > *"For a 6-node linear pipeline, LangGraph is arguably overkill. A script would work.
    > But the conditional edge alone saved LLM calls on days with no relevant content, and
    > I got practice with the framework — which is a portfolio signal in itself."*

    ---

    ### LLM usage

    - **Model**: Google Gemini free tier (`gemini-pro`) for summarization
    - **Backup**: Groq free tier (`llama-3-8b-8192`) — automatic fallback if Gemini rate-limits
    - **Why free tier**: $0 operational cost is a hard requirement. Free-tier rate limits
      (~60 RPM for Gemini) are fine for daily batch processing of 10–20 items
    - **Structured output**: each summary returned as JSON

      ```json
      {
        "title": "...",
        "key_findings": ["...", "...", "..."],
        "relevance_score": 0.87,
        "source_url": "https://...",
        "arxiv_id": "2604.12345"
      }
      ```

    - **Hallucination guardrail**: reject any summary where `arxiv_id` doesn't match an
      `arxiv_id` from the ingestion step. If a paper ID doesn't exist in the raw content pool,
      the LLM invented it — retry with a stricter prompt. No hallucinated citations reach Slack.
    """)
    return


@app.cell
def stage6_delivery(mo):
    mo.md("""
    ## Stage 6: Delivery + Observability

    ### Slack message format

    ```
    🔬 Daily AI Briefing — Apr 28, 2026

    📄 [Paper] Scaling Laws for Neural Architecture Search
    Key findings:
      • Search cost scales sublinearly with model size above 1B params
      • Proxy tasks transfer well for vision but not for language
      • Random search beats DARTS at scale (surprising)
    Relevance: ⭐⭐⭐⭐ | Source: arXiv (2604.12345)

    💬 [Reddit] New benchmarks show Claude 4 excels at multi-step reasoning
    Key findings:
      • MMLU-Pro: 91.2 vs GPT-4o 88.7
      • Reasoning traces show less backtracking
      • Tool use accuracy up 15% vs prior Claude versions
    Relevance: ⭐⭐⭐ | Source: r/MachineLearning

    ───────────────────────────────────────
    📊 Run stats: 47 items ingested · 12 passed filter · 6 in briefing
    💰 Cost: $0.00 | ⏱ Duration: 45s | 🕕 Delivered: 06:00 CT
    ```

    ---

    ### Structured run log (JSONL)

    Every run writes one line to `logs/runs.jsonl`:

    ```json
    {
      "timestamp": "2026-04-28T11:00:03Z",
      "sources_queried": ["arxiv", "reddit", "twitter"],
      "items_ingested": 47,
      "items_filtered": 12,
      "items_summarized": 12,
      "items_curated": 6,
      "total_tokens": 8420,
      "cost_usd": 0.0,
      "latency_seconds": 45,
      "errors": []
    }
    ```

    Enables: trend analysis (`items_filtered` dropping = API issue), debugging
    (correlate errors with run timestamps), reliability demos in interviews.

    ---

    ### GitHub Actions as orchestrator

    ```yaml
    # .github/workflows/briefing.yml
    on:
      schedule:
        - cron: '0 11 * * 1-5'   # 6am CT, weekdays only

    jobs:
      briefing:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/cache@v4
            with:
              path: .chroma/       # persist ChromaDB between runs
              key: chroma-${{ runner.os }}
          - run: pip install -r requirements.txt
          - run: python pipeline.py
            env:
              SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
              GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          - uses: actions/upload-artifact@v4
            with:
              name: run-logs
              path: logs/
    ```

    > *"This is a $0/month production system. The infrastructure choice IS the design decision.
    > Using GitHub Actions for cron + secret management + artifact storage removes an entire
    > infrastructure layer that would cost $50+/month on managed services."*
    """)
    return


@app.cell
def stage7_eval(mo):
    mo.md("""
    ## Stage 7: Evaluation + Monitoring

    ### Eval suite (runs in CI on every prompt change)

    - **20 test cases**: known on-topic content → expect in output; known off-topic → expect filtered
    - **Faithfulness check**: every `arxiv_id` in summaries must match an `arxiv_id` from the
      ingestion step. Catches hallucinated citations automatically.
    - **Format check**: output matches expected Slack message schema (title, key_findings list,
      relevance_score in [0,1], source_url present)
    - **Token budget**: total summarization output < 2000 tokens. Prevents verbose summaries from
      blowing up cost on the day I switch to a paid model.

    ### CI gate

    Prompt changes that break any eval case block the PR. Ensures I never ship a prompt
    regression that causes hallucinated citations or silent topic drift.

    ---

    ### Production monitoring

    | Signal | How I watch it | What it means |
    |--------|---------------|---------------|
    | Briefing quality | Weekly read-through | Are items useful? What was missed? |
    | `items_ingested` drop | Log analysis | Likely API issue on one source |
    | `items_filtered` drop | Log analysis | Source returning less relevant content |
    | `cost_usd` > 0 | Log alert | Something is wrong — should always be $0 |
    | No briefing by 8am | Slack absence | GitHub Actions job failed |

    ---

    ### What I'd add for production scale

    - **LLM-as-judge**: automated scoring on faithfulness, relevance, conciseness — removes
      the need for weekly manual review
    - **A/B testing**: alternate prompt versions on odd/even days, compare quality scores
    - **User feedback loop**: reaction emoji on Slack messages logged back to the run log
      ("thumbs up/down" as implicit quality signal)
    - **Drift detection**: track topic distribution over time. If "quantitative finance" items
      drop to zero for 2 weeks, something changed upstream.
    """)
    return


@app.cell
def tradeoffs(mo):
    mo.md("""
    ## Key Trade-offs I Made

    | Decision | What I chose | What I traded off | Why |
    |----------|-------------|-------------------|-----|
    | LLM provider | Gemini free tier | Output quality vs GPT-4/Claude | $0 cost is a hard requirement |
    | Orchestration | LangGraph | Simplicity (plain script would work) | Portfolio signal + conditional edges |
    | Vector DB | ChromaDB embedded | Horizontal scale + hybrid search | Corpus is tiny; embedded = zero infra |
    | Scheduling | GitHub Actions cron | Flexibility (vs Prefect/Airflow) | Free, zero infra, built-in secret management |
    | Delivery | Slack webhook | Rich UI (vs web dashboard) | Slack is where I already live |
    | Eval | CI regression tests (20 cases) | Deep quality scoring (LLM-as-judge) | Good enough for v1; extend later |
    | Twitter source | Custom crawler | Official API reliability | Free tier API access is too restrictive |

    ---

    ### The constraint that shaped everything

    The $0 budget forced every decision above into better engineering:
    - Free-tier APIs → batch processing to stay within rate limits → thoughtful pipeline design
    - GitHub Actions → no separate orchestration infra → simpler ops
    - Embedded ChromaDB → no DB to manage → faster iteration

    > *"A system with a $0 budget forces better engineering than one where you can just
    > throw money at infrastructure problems. That's the interview story."*
    """)
    return


@app.cell
def improvements(mo):
    mo.md("""
    ## What I'd Improve With More Time

    ### Near-term (would add in v2)

    - **Hybrid search** (BM25 + dense retrieval): pure embedding search misses exact-match
      technical terms — model names, paper titles, specific benchmark names. BM25 handles these.
      Maps to patterns in my vector databases notebook.

    - **Cross-encoder reranking**: after retrieval, rerank with a cross-encoder before curating
      top N. Better relevance ordering than cosine similarity alone. Maps to my RAG notebook.

    - **Semantic caching**: if today's top paper appeared in yesterday's briefing, skip it even
      if it's still trending. Avoids repeating the same content day over day.

    ### Medium-term (would add in v3)

    - **More sources**: HackerNews, AI company engineering blogs (DeepMind, Anthropic, OpenAI),
      specific newsletters. The source-isolation architecture makes this additive — one new scraper.

    - **Web dashboard**: briefing history, topic distribution over time, quality trend charts.
      Currently only in Slack + JSONL — no visual history.

    - **LLM-as-judge**: automated scoring pipeline replacing manual weekly review.
      Maps to my LLM eval notebook.

    ### Long-term (production-scale)

    - **Per-user personalization**: per-user topic embeddings in namespaced ChromaDB collections.
      Shared ingestion layer, personalized filtering and curation per user.

    - **Fine-tuned topic classifier**: replace keyword + embedding threshold with a fine-tuned
      classifier on my own "useful/not useful" labels from the feedback loop.

    > *"Each of these maps to a pattern I've studied: hybrid search (vector DBs notebook),
    > reranking (RAG notebook), eval (LLM eval notebook), caching (LLM serving patterns notebook).
    > The briefing agent is the system; the notebooks are the depth behind the decisions."*
    """)
    return


@app.cell
def followups(mo):
    mo.md("""
    ## Common Interviewer Follow-ups

    ---

    **"How would you scale this to 1000 users with different interests?"**

    Shared ingestion layer runs once (fetch all content for today). Personalized filtering:
    each user has their own topic embeddings in a namespaced ChromaDB collection.
    Summarization and curation run per user in parallel. Delivery: per-user Slack DM or email.
    At 1000 users, the bottleneck is LLM calls — move to a paid tier or use batched inference.

    ---

    **"What if a source goes down permanently?"**

    Each source is independent — one going down doesn't affect others (graceful degradation).
    Alert on 3 consecutive failures from the same source. Add a replacement source — the
    pluggable scraper architecture makes this additive. Twitter/X is already the most fragile;
    I treat it as optional.

    ---

    **"How do you ensure summaries are accurate?"**

    Three-layer defense: (1) citation validation — every `arxiv_id` in a summary must match
    an ID from the ingestion step, validated at runtime before it reaches Slack; (2) faithfulness
    eval in CI — test cases with known content verify the LLM doesn't fabricate; (3) retry
    on failure — if a summary fails validation, retry with a stricter prompt. No hallucinated
    paper makes it to Slack.

    ---

    **"Why not just use ChatGPT with browsing?"**

    ChatGPT browsing is manual (I have to trigger it), unstructured (no consistent format),
    expensive ($20/month), and doesn't produce the structured logs I use to track quality over
    time. I need automation (daily at 6am, no manual input), specific sources (not generic web),
    structured output (for analysis and CI evals), and $0 cost.

    ---

    **"What was the hardest bug?"**

    ChromaDB persistence between GitHub Actions runs. GH Actions are ephemeral — each run starts
    with a fresh environment. My first version rebuilt the vector store from scratch every day,
    which broke deduplication (same papers kept appearing in briefings). The fix was using
    `actions/cache` to persist the `.chroma/` directory between runs, keyed by OS. Took a full
    day to debug because the CI logs showed the pipeline "succeeding" with no apparent error —
    the deduplication was just silently not working. Taught me to always log dedup hit rate as
    a metric, not just assume it's working.

    ---

    **"How do you handle rate limits?"**

    Each source has different limits — arXiv is generous, Reddit allows ~60 req/min with OAuth,
    Gemini free tier is ~60 RPM. I stay well within limits because I'm doing one daily batch
    run, not continuous polling. If I hit a rate limit during a run, exponential backoff handles
    it. For Gemini, I added a Groq fallback so a rate limit doesn't kill the entire pipeline.
    """)
    return


@app.cell
def talking_points(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    ### 60-second pitch

    > "I built a daily AI research briefing agent that runs on GitHub Actions for free. It ingests
    > from arXiv, Reddit, and Twitter, filters by my topic interests using two-stage filtering —
    > keyword pre-filter plus embedding similarity in ChromaDB — generates structured summaries
    > with free-tier LLMs, and delivers to Slack every morning by 8am. The pipeline is orchestrated
    > with LangGraph, with regression evals running in CI on every prompt change.
    > It costs $0/month to operate and I've been using it daily for several months."

    ---

    ### The thing that impresses

    > "Every design decision was constrained by the $0 budget. That constraint forced better
    > engineering — free-tier APIs meant I had to batch carefully; embedded ChromaDB meant no
    > infra to manage; GitHub Actions meant I had secret management, cron scheduling, and artifact
    > storage for free. The system is more interesting than one that just throws GPT-4 at
    > everything, because every decision has a reason."

    ---

    ### Why this demonstrates AI engineering skills

    This system touches the full AI engineering stack in one artifact:

    | Pattern | Where it appears |
    |---------|-----------------|
    | RAG retrieval | ChromaDB + two-stage topic filter |
    | Tool calling | LangGraph nodes as callable tools |
    | Structured outputs | JSON summaries with schema validation |
    | Guardrails | arxiv_id hallucination check |
    | Evals in CI/CD | 20-case regression suite on prompt changes |
    | LangGraph orchestration | Conditional edges, typed state |
    | Cost optimization | Free-tier LLMs, batch processing, pre-filters |
    | Production monitoring | Structured JSONL logs, daily quality review |

    > *"It's a small system. But it exercises every pattern you'd find in a production AI
    > engineering role — and I can speak to every component from experience, not from memory."*

    ---

    ### Connection to my other notebooks

    - **Vector databases notebook** → ChromaDB selection, HNSW vs flat index, scaling path
    - **RAG notebook** → two-stage filtering, hybrid search improvement, reranking
    - **LLM serving patterns notebook** → free-tier selection, rate limit handling, caching
    - **Agent architectures notebook** → LangGraph graph design, state management, tool calling
    - **LLM eval notebook** → CI eval suite design, faithfulness checks, LLM-as-judge roadmap
    """)
    return


if __name__ == "__main__":
    app.run()
