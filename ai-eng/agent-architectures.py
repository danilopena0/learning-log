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
    # Agent Architectures

    | Field   | Value                                                                      |
    |---------|----------------------------------------------------------------------------|
    | Date    | 2026-04-12                                                                 |
    | Track   | AI Engineering                                                             |
    | Time    | 60 min                                                                     |
    | Topic   | ReAct · Tool Use Patterns · Multi-Agent · LangGraph                        |
    """)
    return


@app.cell
def what_is_an_agent(mo):
    mo.md("""
    ## What is an Agent?

    **Working definition:** an LLM-powered system that decides its own sequence of actions
    to accomplish a goal, rather than following a fixed pipeline.

    ### The spectrum of LLM control

    ```
    Hardcoded pipeline  ──────────────────────────────►  Fully agentic
         (no LLM)          LLM-in-the-loop               (LLM decides all)
                           (most production)
    ```

    The key question is: **how much control does the LLM have over the control flow?**

    ### Anthropic's framing
    - **Workflows** — predefined paths where the LLM fills in reasoning steps within a fixed structure
    - **Agents** — LLM-directed paths where the LLM dynamically decides the sequence of actions

    Most production systems are **workflows with agentic components** — not fully autonomous agents.
    Full autonomy trades debuggability for flexibility, which is usually a bad deal in production.

    ### My interview framing
    > "I build agentic workflows, not fully autonomous agents. The LLM handles reasoning and
    > tool selection, but the graph structure provides guardrails and predictability.
    > Fully autonomous agents are hard to debug and expensive — the graph gives me an audit trail."

    ---

    | Term | Meaning in practice |
    |------|---------------------|
    | **Node** | A function or LLM call — one step in the graph |
    | **Edge** | Transition between nodes — deterministic or conditional |
    | **State** | Typed dict passed between nodes — the "working memory" |
    | **Tool** | A Python function the LLM can decide to call |
    | **Trace** | The full log of inputs/outputs across all nodes |
    """)
    return


@app.cell
def shared_imports():
    import asyncio
    import json
    import re
    import time
    from dataclasses import dataclass, field
    from typing import Any, Callable, Optional

    return Callable, asyncio, dataclass, json, re


@app.cell
def react_concept(mo):
    mo.md("""
    ---
    ## Pattern 1: ReAct (Reason + Act)

    ### The loop
    ```
    Thought  →  Action  →  Observation  →  Thought  →  ...  →  Final Answer
    ```

    | Step | What happens |
    |------|-------------|
    | **Thought** | LLM reasons about what to do next (written as text) |
    | **Action** | LLM calls a tool with specific arguments |
    | **Observation** | Tool result is appended to context |
    | Repeat | LLM incorporates the observation and reasons again |
    | **Final Answer** | LLM signals it's done — no more tool calls |

    ### Why it works
    Externalizing reasoning as text makes the LLM "think out loud." This gives you a readable
    trace of *why* each tool was called, and makes errors much easier to diagnose.

    ### When ReAct wins
    - Multi-step problems where each step depends on the previous result (research, debugging, math with lookups)
    - Tasks where the LLM needs to decide the path dynamically, not follow a fixed script
    - Situations where an auditable reasoning chain matters (for debugging or stakeholder trust)

    ### When ReAct breaks
    - Long loops where the LLM gets confused by its accumulated context
    - Tasks with many independent sub-tasks (better handled by parallel tool calls or multi-agent)
    - High-latency tools — each Thought→Action→Observation cycle is sequential, so it stacks up
    """)
    return


@app.cell
def react_implementation(Callable, re):
    # ── Minimal ReAct loop (~40 lines) ───────────────────────────────────────────
    # Mock LLM returns structured Thought/Action strings for a test query.
    # Parser extracts action name + input, dispatch table runs the tool,
    # observation is appended, and the loop continues until "Final Answer".

    MAX_ITERATIONS = 5  # hard safety cap — no runaway loops

    # ── Tool implementations ──────────────────────────────────────────────────────
    def get_capital(country: str) -> str:
        capitals = {"france": "Paris", "germany": "Berlin", "japan": "Tokyo"}
        return capitals.get(country.lower(), f"Unknown capital for {country}")

    def get_weather(city: str) -> str:
        weather = {"Paris": "15°C, partly cloudy", "Berlin": "8°C, rainy", "Tokyo": "22°C, sunny"}
        return weather.get(city, f"No weather data for {city}")

    TOOLS: dict[str, Callable] = {
        "get_capital": get_capital,
        "get_weather": get_weather,
    }

    # ── Mock LLM — returns hardcoded ReAct traces ─────────────────────────────────
    MOCK_RESPONSES = [
        "Thought: I need to find the capital of France first.\nAction: get_capital\nAction Input: France",
        "Thought: Now I have the capital. I'll look up the weather there.\nAction: get_weather\nAction Input: Paris",
        "Thought: I have all the info I need.\nFinal Answer: The capital of France is Paris. The current weather there is 15°C, partly cloudy.",
    ]

    def mock_llm(prompt: str, step: int) -> str:
        return MOCK_RESPONSES[min(step, len(MOCK_RESPONSES) - 1)]

    # ── Parser: extract Action or Final Answer from LLM response ─────────────────
    def parse_response(response: str) -> tuple[str, str | None, str | None]:
        """Returns (thought, action_name, action_input) — action fields are None on Final Answer."""
        thought = re.search(r"Thought:(.*?)(?:\n|$)", response)
        action = re.search(r"^Action:\s*(.+)$", response, re.MULTILINE)
        action_input = re.search(r"^Action Input:\s*(.+)$", response, re.MULTILINE)
        final = re.search(r"^Final Answer:\s*(.+)$", response, re.MULTILINE | re.DOTALL)

        thought_text = thought.group(1).strip() if thought else ""
        if final:
            return thought_text, None, final.group(1).strip()
        return thought_text, action.group(1).strip() if action else None, action_input.group(1).strip() if action_input else None

    # ── Main ReAct loop ───────────────────────────────────────────────────────────
    def run_react(query: str) -> str:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        context = f"Question: {query}\n"

        for step in range(MAX_ITERATIONS):
            response = mock_llm(context, step)
            thought, action_name, action_value = parse_response(response)

            print(f"\n[Step {step + 1}]")
            print(f"  Thought: {thought}")

            if action_name is None:
                # action_value holds the Final Answer in this branch
                print(f"  Final Answer: {action_value}")
                return action_value or ""

            print(f"  Action: {action_name}({action_value})")
            observation = TOOLS[action_name](action_value or "")
            print(f"  Observation: {observation}")

            context += f"\nThought: {thought}\nAction: {action_name}\nAction Input: {action_value}\nObservation: {observation}"

        return "Max iterations reached — no final answer."

    result = run_react("What's the weather in the capital of France?")
    return


@app.cell
def react_projects(mo):
    mo.md("""
    ### My Projects — ReAct in Practice

    **Daily Briefing Agent**
    Uses ReAct-style reasoning within LangGraph nodes. The "research" node reasons about which
    source to query (arXiv vs Reddit vs Twitter), executes the tool call, observes the result,
    and decides if more queries are needed — all within a single node's execution.

    **Key lesson:** I added a hard `max_loops=3` per topic. Without it, the agent was running
    10+ iterations and hallucinating tool calls — it would "call" tools it'd already used,
    getting confused by the growing context. Bounded loops fixed it immediately.

    ---

    **Interview framing:**
    > "My briefing agent uses a bounded ReAct pattern — max 3 reasoning loops per topic.
    > Without bounds, the loops were running 10+ iterations and hallucinating tool calls.
    > The bound is a hard engineering constraint, not a prompt instruction — the loop exits
    > in code, not because the LLM decides to stop."
    """)
    return


@app.cell
def tool_use_concept(mo):
    mo.md("""
    ---
    ## Pattern 2: Tool Use Patterns

    Not all tool use is the same. The pattern you choose affects latency, cost, and debuggability.

    | Pattern | Description | When to use |
    |---------|-------------|-------------|
    | **Single tool call** | LLM calls one tool, returns result | Simple lookups, one-shot tasks |
    | **Parallel tool calls** | LLM calls multiple tools simultaneously | Independent sub-tasks — faster, no ordering dependency |
    | **Sequential tool calls** | Each tool uses the previous tool's output | When results depend on each other (ReAct is one form) |
    | **Structured outputs** | Tool returns Pydantic-validated data | Any time downstream code needs typed objects |
    | **Tool routing** | Lightweight LLM classifies request, routes to specialized tool | Cost optimization — not every query needs the full toolset |

    ### The routing insight
    A small/cheap LLM classifier in front of your tools can dramatically reduce cost.
    The classifier asks: "which tool should handle this?" — then the right tool is invoked
    without exposing your powerful (expensive) model to irrelevant context.

    ```
    User query → [Classifier LLM] → route to: search / calculator / calendar / ...
                    (cheap)                          (specialized tools)
    ```
    """)
    return


@app.cell
def tool_use_code(asyncio, json):
    # ── Four tool use patterns with mocked LLM calls ─────────────────────────────

    # ── Mock infrastructure ───────────────────────────────────────────────────────
    def mock_search(query: str) -> str:
        return f"[search results for '{query}': 3 relevant documents found]"

    def mock_retrieve(doc_id: str) -> str:
        return f"[document {doc_id}: 'LangGraph enables stateful, multi-actor applications...']"

    def mock_summarize(text: str) -> str:
        return f"[summary: '{text[:40]}...' condensed to 2 sentences]"

    # ── Pattern 1: Single tool call ───────────────────────────────────────────────
    def single_tool_call(query: str) -> str:
        """LLM decides to call exactly one tool."""
        print("\n[Single Tool Call]")
        # Mock: LLM returns a tool call decision
        tool_decision = {"tool": "search", "input": query}
        print(f"  LLM decision: {json.dumps(tool_decision)}")
        result = mock_search(tool_decision["input"])
        print(f"  Result: {result}")
        return result

    # ── Pattern 2: Parallel tool calls ────────────────────────────────────────────
    async def parallel_tool_calls(query: str) -> list[str]:
        """LLM calls multiple independent tools simultaneously."""
        print("\n[Parallel Tool Calls]")
        # Mock: LLM returns multiple tool calls at once
        tool_decisions = [
            {"tool": "search", "input": query},
            {"tool": "retrieve", "input": "doc_42"},
        ]
        print(f"  LLM decisions: {[d['tool'] for d in tool_decisions]} (dispatched in parallel)")

        # Simulate async dispatch — both tools run concurrently
        async def run_tool(decision: dict) -> str:
            await asyncio.sleep(0)  # yield to event loop (real I/O would go here)
            if decision["tool"] == "search":
                return mock_search(decision["input"])
            return mock_retrieve(decision["input"])

        results = await asyncio.gather(*[run_tool(d) for d in tool_decisions])
        for r in results:
            print(f"  Result: {r}")
        return list(results)

    # ── Pattern 3: Sequential tool calls ─────────────────────────────────────────
    def sequential_tool_calls(query: str) -> str:
        """Each tool call depends on the previous result."""
        print("\n[Sequential Tool Calls]")
        step1 = mock_search(query)
        print(f"  Step 1 (search): {step1}")

        # Mock: LLM extracts a doc ID from the search result to retrieve
        doc_id = "doc_001"
        step2 = mock_retrieve(doc_id)
        print(f"  Step 2 (retrieve): {step2}")

        step3 = mock_summarize(step2)
        print(f"  Step 3 (summarize): {step3}")
        return step3

    # ── Pattern 4: Tool routing ───────────────────────────────────────────────────
    def tool_routing(query: str) -> str:
        """A cheap classifier LLM routes to the right specialized tool."""
        print("\n[Tool Routing]")

        # Mock classifier: a small LLM that only decides which tool to use
        routing_table = {
            "search": lambda q: mock_search(q),
            "retrieve": lambda q: mock_retrieve("doc_42"),
            "summarize": lambda q: mock_summarize(q),
        }

        # Classifier picks a route based on keywords (mocked)
        route = "search" if "find" in query.lower() or "what" in query.lower() else "summarize"
        print(f"  Classifier → route: '{route}'")

        result = routing_table[route](query)
        print(f"  Result: {result}")
        return result

    # ── Run all four patterns ─────────────────────────────────────────────────────
    print("=" * 60)
    single_tool_call("What is LangGraph?")
    sequential_tool_calls("LangGraph architecture")
    tool_routing("What is a StateGraph?")

    # Parallel requires async — run via asyncio
    parallel_result = asyncio.run(
        parallel_tool_calls("LangGraph overview")
    )
    return


@app.cell
def tool_use_projects(mo):
    mo.md("""
    ### My Projects — Tool Use Patterns in Practice

    **Canopy (Job Search Agent)**
    Uses sequential tool calls — scrape job postings → score with LLM → structure with
    Pydantic → route to Slack if `fit_score > threshold`. Each step is a separately
    testable function with a typed input/output contract.

    **Multi-agent-lab**
    Experiments with parallel tool calls for research queries — running arXiv + Reddit +
    Twitter fetches simultaneously instead of sequentially. Latency dropped significantly
    (3 sequential API calls → 1 parallel batch).

    ---

    **Interview framing:**
    > "In Canopy, I made each tool call independently testable. The scraper, scorer, and
    > notifier are separate tools with Pydantic contracts between them. This makes the system
    > debuggable — if something breaks, I know exactly which tool. I added tool routing later
    > when I realized not every job posting needed full LLM scoring — a keyword classifier
    > handles the obvious rejects cheaply first."
    """)
    return


@app.cell
def multi_agent_concept(mo):
    mo.md("""
    ---
    ## Pattern 3: Multi-Agent Patterns

    ### When one agent isn't enough
    - Task is too broad for one agent's context window
    - Different sub-tasks need genuinely different expertise / prompts / tools
    - Sub-tasks can run in parallel (different agents doing different things simultaneously)
    - You want specialization: a researcher agent, a writer agent, a critic agent

    ### Four main patterns

    | Pattern | Structure | Debuggability | Flexibility |
    |---------|-----------|---------------|-------------|
    | **Orchestrator-worker** | Supervisor → specialists → aggregator | High — one source of truth | Medium |
    | **Sequential pipeline** | Agent A → Agent B → Agent C | High — fixed order | Low |
    | **Hierarchical** | Supervisor → mid-level supervisors → workers | Medium | Medium |
    | **Swarm / Peer-to-peer** | Any agent can hand off to any other | Low — distributed reasoning | High |

    The trade-off is always **debuggability vs flexibility**.
    Peer-to-peer is the most flexible and the hardest to debug.
    Sequential pipeline is the least flexible and the easiest to debug.

    **For production: default to orchestrator-worker.**
    """)
    return


@app.cell
def multi_agent_when_not_to(mo):
    mo.md("""
    ### When NOT to Go Multi-Agent

    Don't reach for multi-agent just because a task feels complex.
    **A single agent with good tools often handles what looks like a multi-agent problem.**

    #### Multi-agent failure modes

    | Failure mode | Why it hurts |
    |--------------|--------------|
    | **Context loss at handoffs** | Each agent has its own context window — what Agent A knew, Agent B doesn't |
    | **Coordination overhead** | Agents talking to agents multiplies LLM calls and cost |
    | **Debugging nightmare** | "Whose reasoning was wrong?" becomes genuinely hard to answer |
    | **Higher cost** | Every agent boundary is another LLM call |

    #### Use multi-agent when
    - Sub-tasks are genuinely parallel (different agents can work simultaneously)
    - Sub-tasks need specialized context per role (a researcher needs different tools than a writer)
    - A single agent's context window would overflow with the full task
    - You need fault isolation — one agent failing shouldn't crash the whole pipeline

    ---

    **Interview gold:**
    > "I default to single-agent workflows. I only reach for multi-agent when I can articulate
    > specifically WHY one agent won't work. Usually the answer is: the sub-tasks are genuinely
    > parallel, or they need different tools/prompts. If I can't articulate the 'why', I don't
    > add the complexity."
    """)
    return


@app.cell
def orchestrator_worker_code(asyncio, dataclass):
    # ── Orchestrator-worker pattern (~50 lines) ───────────────────────────────────
    # Orchestrator decomposes a research query into sub-queries,
    # dispatches each to a worker in parallel, aggregates results.

    @dataclass
    class WorkerResult:
        worker_id: str
        sub_query: str
        findings: str

    # ── Mock LLM: orchestrator decomposes the task ────────────────────────────────
    def orchestrator_decompose(query: str) -> list[str]:
        """Mock: LLM breaks the main query into sub-queries."""
        return [
            f"{query} — recent papers (arXiv)",
            f"{query} — practitioner discussion (Reddit/HN)",
            f"{query} — implementation examples (GitHub)",
        ]

    # ── Mock LLM: orchestrator aggregates worker results ─────────────────────────
    def orchestrator_aggregate(results: list[WorkerResult]) -> str:
        findings = "\n".join(f"  [{r.worker_id}] {r.findings}" for r in results)
        return f"Synthesized answer from {len(results)} workers:\n{findings}"

    # ── Worker: handles one sub-query ────────────────────────────────────────────
    async def worker(worker_id: str, sub_query: str) -> WorkerResult:
        """Each worker is an independent agent with its own context."""
        await asyncio.sleep(0)  # simulate async I/O (real: API call)
        findings = f"Found 3 relevant results for '{sub_query[:35]}...'"
        return WorkerResult(worker_id=worker_id, sub_query=sub_query, findings=findings)

    # ── Orchestrator: the top-level coordinator ───────────────────────────────────
    async def run_orchestrator_worker(query: str) -> str:
        print(f"\n{'='*60}")
        print(f"Orchestrator received: {query}")

        # Step 1: Decompose
        sub_queries = orchestrator_decompose(query)
        print(f"\nOrchestrator decomposed into {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries):
            print(f"  [{i+1}] {sq}")

        # Step 2: Dispatch workers in parallel
        print("\nDispatching workers in parallel...")
        worker_tasks = [
            worker(f"worker_{i+1}", sq)
            for i, sq in enumerate(sub_queries)
        ]
        results = await asyncio.gather(*worker_tasks)

        # Step 3: Aggregate
        print("\nAll workers complete. Aggregating...")
        final = orchestrator_aggregate(list(results))
        print(f"\n{final}")
        return final

    # Run the orchestrator-worker demo
    final_answer = asyncio.run(
        run_orchestrator_worker("LangGraph multi-agent patterns")
    )
    return


@app.cell
def multi_agent_projects(mo):
    mo.md("""
    ### My Projects — Multi-Agent in Practice

    **Multi-agent-lab**
    My explicit playground for orchestrator-worker patterns. Built a research coordinator
    that delegates to specialist agents (arXiv agent, Reddit agent, Twitter agent), each
    with its own context and prompt. The orchestrator's reasoning is the single source of
    truth — when something goes wrong, I look at the orchestrator trace first.

    **Daily Briefing Agent**
    Uses a **sequential pipeline pattern** within LangGraph: ingest node → filter node →
    summarize node → deliver node. Not multi-agent in the "multiple LLMs reasoning
    simultaneously" sense, but multi-node with LLM calls at specific points. Each node is
    independently testable.

    ---

    **Interview framing:**
    > "My multi-agent-lab was built specifically to learn these patterns. Key lesson:
    > orchestrator-worker was dramatically better than peer-to-peer for debuggability.
    > When a peer-to-peer agent misroutes a task, you have no clear trace of who decided what.
    > With orchestrator-worker, the orchestrator's reasoning is the single source of truth —
    > I can read its output and see exactly why it delegated to which worker."
    """)
    return


@app.cell
def langgraph_concept(mo):
    mo.md("""
    ---
    ## Pattern 4: LangGraph

    LangGraph = graph-based agent framework where **nodes** are functions/LLM calls and
    **edges** define transitions between them.

    ### Key primitives

    | Primitive | What it is |
    |-----------|-----------|
    | **StateGraph** | Typed state (dict) that flows through the graph and is mutated at each node |
    | **Node** | A Python function that takes state as input and returns a state update |
    | **Edge** | A directed connection between two nodes — deterministic transition |
    | **Conditional edge** | A function that inspects state and returns the next node name — branching |
    | **Compile** | Turns the graph definition into an executable runnable |

    ### Why graphs over chains
    - **Explicit control flow** — the graph IS the architecture diagram
    - **Visualizable** — export the graph as an image for stakeholders
    - **Supports loops** — a node can route back to an earlier node (retry, refinement)
    - **Inspectable state** — every node's input and output is a typed dict you can log
    - **Human-in-the-loop** — add approval nodes without changing the surrounding logic

    ### LangGraph vs simpler frameworks
    Use LangGraph when you need:
    - **Branching logic** — "if the LLM flagged uncertainty, route to a verification node"
    - **Loops with termination** — retry failed steps with backoff, refinement cycles
    - **Human approval steps** — pause the graph, wait for a human, then continue
    - **Visualization** — show the workflow to a non-technical stakeholder

    Use a simpler approach (direct LLM calls, basic Python state machine) when:
    - The workflow is genuinely linear with no branching
    - You need total control over every aspect and LangGraph's abstractions get in the way
    """)
    return


@app.cell
def langgraph_example(Callable):
    # ── Minimal LangGraph-style state machine (no langgraph install required) ─────
    # Mocks the core primitives: StateGraph, nodes, conditional edges.
    # Pattern underlying both the Briefing Agent and Canopy.

    from typing import TypedDict, Literal

    # ── Typed state — the "working memory" of the graph ──────────────────────────
    class AgentState(TypedDict):
        query: str
        action_needed: bool
        tool_result: str
        final_answer: str
        step: int

    # ── Nodes — functions that take state and return a partial state update ────────
    def reasoning_node(state: AgentState) -> AgentState:
        """LLM reasons about the query and decides whether a tool call is needed."""
        print(f"\n[reasoning_node] query='{state['query']}'")
        # Mock: decide a tool is needed if query contains a question word
        needs_tool = any(w in state["query"].lower() for w in ["what", "how", "find", "get"])
        update = {"action_needed": needs_tool, "step": state["step"] + 1}
        print(f"  → action_needed={needs_tool}")
        return {**state, **update}

    def tool_node(state: AgentState) -> AgentState:
        """Executes the tool the LLM decided to call."""
        print(f"[tool_node] executing tool for query='{state['query']}'")
        result = f"Tool result: found 3 documents relevant to '{state['query']}'"
        print(f"  → {result}")
        return {**state, "tool_result": result, "step": state["step"] + 1}

    def finalize_node(state: AgentState) -> AgentState:
        """Produces the final answer, optionally incorporating tool results."""
        print(f"[finalize_node]")
        if state["tool_result"]:
            answer = f"Based on tool output — {state['tool_result']}"
        else:
            answer = f"Direct answer to: '{state['query']}'"
        print(f"  → final_answer='{answer}'")
        return {**state, "final_answer": answer, "step": state["step"] + 1}

    # ── Conditional edge — inspects state to decide next node ────────────────────
    def route_after_reasoning(state: AgentState) -> Literal["tool_node", "finalize_node"]:
        """The graph's branching logic — NOT an LLM call, just a Python function."""
        return "tool_node" if state["action_needed"] else "finalize_node"

    # ── Minimal graph runner (mocking LangGraph's compile + invoke) ──────────────
    class MockStateGraph:
        """Lightweight mock of LangGraph's StateGraph for demo purposes."""
        def __init__(self):
            self.nodes: dict[str, Callable] = {}
            self.edges: dict[str, str | Callable] = {}

        def add_node(self, name: str, fn: Callable):
            self.nodes[name] = fn

        def add_conditional_edges(self, from_node: str, condition_fn: Callable):
            self.edges[from_node] = condition_fn  # fn returns node name

        def add_edge(self, from_node: str, to_node: str):
            self.edges[from_node] = to_node  # static string

        def run(self, initial_state: AgentState, entry: str = "reasoning_node") -> AgentState:
            state = initial_state
            current = entry
            visited = 0
            while current and visited < 10:
                node_fn = self.nodes[current]
                state = node_fn(state)
                next_node = self.edges.get(current)
                if callable(next_node):
                    current = next_node(state)  # conditional edge
                else:
                    current = next_node          # static edge or None (terminal)
                visited += 1
            return state

    # ── Wire up the graph ─────────────────────────────────────────────────────────
    graph = MockStateGraph()
    graph.add_node("reasoning_node", reasoning_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("finalize_node", finalize_node)

    graph.add_conditional_edges("reasoning_node", route_after_reasoning)
    graph.add_edge("tool_node", "finalize_node")
    graph.add_edge("finalize_node", None)  # terminal node

    # ── Run with two different queries to show branching ─────────────────────────
    print("=" * 60)
    print("Run 1 — query that needs a tool call:")
    state1: AgentState = {"query": "What are LangGraph best practices?", "action_needed": False,
                          "tool_result": "", "final_answer": "", "step": 0}
    result1 = graph.run(state1)
    print(f"\nFinal: {result1['final_answer']}")

    print("\n" + "=" * 60)
    print("Run 2 — query that goes directly to finalize:")
    state2: AgentState = {"query": "Hello", "action_needed": False,
                          "tool_result": "", "final_answer": "", "step": 0}
    result2 = graph.run(state2)
    print(f"\nFinal: {result2['final_answer']}")
    return


@app.cell
def langgraph_projects(mo):
    mo.md("""
    ### My Projects — LangGraph in Practice

    **Daily Briefing Agent**
    LangGraph with nodes for ingestion (arXiv/Reddit/Twitter), filtering, RAG retrieval
    from ChromaDB, LLM-powered summarization, and Slack delivery. Conditional edges handle:
    "did we find anything new since yesterday?" and "is this content in-scope?"
    The graph structure means I can visualize the entire pipeline and show it to stakeholders.

    **Canopy (Job Search Agent)**
    LangGraph agent where the state is the job search pipeline. Nodes handle scraping,
    enrichment, scoring, and notification. The graph makes it straightforward to add a
    human-in-the-loop approval step before posting to Slack — just add an approval node
    and a conditional edge. With a linear chain, this would require restructuring the whole flow.

    ---

    **Interview framing:**
    > "I chose LangGraph over LangChain's simpler Chain abstraction because my workflows
    > have branching (different behavior for different job categories) and loops (retry
    > failed scrapes with backoff). A linear chain can't express that cleanly. LangGraph
    > lets me encode the control flow in the graph itself — the architecture IS the code."
    """)
    return


@app.cell
def debugging_observability(mo):
    mo.md("""
    ---
    ## Debugging and Observability for Agents

    Agents are **harder to debug than regular code** — non-determinism, long traces,
    subtle prompt issues, and failures that look like success until you read the trace carefully.

    ### Essential practices

    | Practice | Why it matters |
    |----------|---------------|
    | **Log every LLM call** (input, output, latency, cost) | Know what was sent, what came back, how much it cost |
    | **Log every tool call and its result** | Isolate whether the LLM or the tool caused a bad outcome |
    | **Version prompts alongside code** | Prompts as `.txt` files in the repo, not inline strings |
    | **Replay capability** | Given a saved trace, replay with mocked LLM responses — reproduce locally |
    | **Structured trace storage** | JSONL per run — searchable, diffable, feedable to evals |

    ### Tools
    - **LangSmith** — native LangGraph integration, trace visualization, prompt playground
    - **Langfuse** — open-source alternative, good for self-hosted setups
    - **Custom JSONL traces** — simple, always works, no external dependency

    ### My approach
    My briefing agent logs every node's input/output to a JSONL file. When something
    weird happens in production, I can replay the trace locally with mocked LLM calls
    and inspect exactly what the agent saw at each step.

    ```python
    # Minimal structured logging pattern
    import json
    from pathlib import Path
    from datetime import datetime

    def log_node(node_name: str, state_in: dict, state_out: dict):
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "node": node_name,
            "state_in": state_in,
            "state_out": state_out,
        }
        with open("agent_trace.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    ```

    **The rule:** if you can't replay it locally, you can't debug it in production.
    """)
    return


@app.cell
def interview_questions(mo):
    mo.md("""
    ---
    ## Common Interviewer Questions

    **"When would you use an agent vs a fixed pipeline?"**
    Fixed pipeline when the workflow is known and static — the path doesn't change based
    on intermediate results. Agent when the path depends on intermediate results or user
    input. Most production systems are somewhere in between: a workflow with agentic nodes
    at the decision points.

    ---

    **"How do you handle agent failures?"**
    Bounded loops (max iterations in code, not prompts), fallback responses when tools fail,
    retry with exponential backoff for transient errors, observability at every step so I
    know which node failed, and graceful degradation — if one tool fails, continue with others
    rather than crashing the whole pipeline.

    ---

    **"Have you built multi-agent systems? What did you learn?"**
    Yes — multi-agent-lab. Key lesson: start single-agent, go multi-agent only when you have
    a specific reason. Orchestrator-worker is dramatically better than peer-to-peer for
    production — the orchestrator's reasoning is the single source of truth when debugging.

    ---

    **"How do you evaluate agents?"**
    End-to-end task success rate, per-step evals (did tool selection match expected?), trace
    review for qualitative assessment on edge cases, and regression evals on known-hard cases
    whenever I change a prompt or tool definition.

    ---

    **"LangGraph vs building from scratch — when do you pick each?"**
    For prototyping and well-understood patterns, LangGraph is excellent — the graph structure
    gives me visualization and branching for free. For truly custom flows or production systems
    where I need total control, I sometimes build lightweight state machines in pure Python.
    The framework should match the complexity of the problem, not the other way around.
    """)
    return


@app.cell
def flashcard_summary(mo):
    mo.md("""
    ---
    ## Flashcard Summary

    | Question | Answer |
    |----------|--------|
    | **What is ReAct?** | Thought→Action→Observation loop; externalizes reasoning as text for an auditable trace |
    | **Single agent vs multi-agent?** | Default to single. Go multi when sub-tasks are genuinely parallel or need specialized context. |
    | **Orchestrator-worker vs peer-to-peer?** | Orchestrator-worker is more debuggable (one source of truth); peer-to-peer is more flexible. Pick orchestrator for production. |
    | **What's LangGraph's core abstraction?** | Typed state passed between nodes via edges, with conditional branching encoded as Python functions |
    | **Why bound agent loops?** | Without `max_iterations`, agents can loop indefinitely, hallucinate tool calls, or burn through tokens |
    | **Biggest debugging challenge with agents?** | Non-determinism + long traces. Solution: structured logging of every LLM/tool call, replay capability. |
    | **When is multi-agent overkill?** | When a single agent with good tools would work. Don't add complexity without a specific reason. |
    | **How do you version agent prompts?** | Prompts as `.txt` files in the repo, under version control, with regression evals on prompt changes |
    | **Workflow vs agent (Anthropic's framing)?** | Workflow = predefined paths; Agent = LLM-directed paths. Most production = workflow with agentic nodes. |
    | **Tool routing — why bother?** | A cheap classifier LLM in front saves expensive model calls; only route queries to the tools that can handle them. |
    """)
    return


@app.cell
def interview_talking_points(mo):
    mo.md("""
    ---
    ## Interview Talking Points

    **"Walk me through an agent you've built"**
    Daily Briefing Agent. LangGraph workflow with nodes for ingestion from arXiv/Reddit/Twitter,
    RAG retrieval from ChromaDB, LLM-powered summarization with tool calls, and conditional
    delivery to Slack. Took 3 iterations to get the loop bounds right — first version had no
    max-iteration cap and occasionally looped forever.

    ---

    **"What surprised you when building agents?"**
    > "Bounded loops and structured outputs mattered WAY more than picking the right framework.
    > My first version had no max-iteration cap and the agent would occasionally loop forever.
    > Adding a hard limit in code — not a prompt instruction — fixed it immediately. The lesson:
    > reliability comes from engineering constraints, not hoping the LLM decides to stop."

    ---

    **"Multi-agent vs single-agent — when do you reach for each?"**
    > "I built multi-agent-lab specifically to explore these patterns. My takeaway: most
    > production use cases work fine as single-agent workflows with good tool design.
    > Multi-agent shines when sub-tasks are genuinely parallel or need different context per role.
    > If I can't articulate a specific reason why one agent won't work, I don't add the complexity."

    ---

    **Connection to my other work**
    My backtesting engine uses a similar pattern to orchestrator-worker: a strategy coordinator
    dispatches to different strategy modules (mean reversion, momentum, pairs trading), each
    running independently, with results aggregated. Same pattern — different domain. The
    orchestrator-worker pattern is more general than agents; it's a good answer to
    "how do you decompose complex tasks?" regardless of whether LLMs are involved.
    """)
    return


if __name__ == "__main__":
    app.run()
