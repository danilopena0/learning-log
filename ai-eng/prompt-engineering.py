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
    # Prompt Engineering Patterns

    | Field | Value |
    |-------|-------|
    | Date  | 2026-05-01 |
    | Track | AI Engineering |
    | Time  | 60 min |
    | Topic | Few-Shot · Chain-of-Thought · Self-Consistency · Constitutional AI |
    """)
    return


@app.cell
def prompts_are_software(mo):
    mo.md("""
    ## Prompt Engineering is Software Engineering

    Prompts are code. They should be versioned, tested, reviewed, and evaluated — not
    tweaked ad hoc in a playground. A prompt that "feels right" is not a prompt that works.
    A prompt that passes an eval suite is.

    The evolution of the field: prompt engineering started as "just ask nicely." Now it's a
    set of named patterns with known properties, failure modes, and tradeoffs — like design
    patterns in software. Knowing the pattern names matters in interviews: it signals you think
    systematically, not intuitively.

    **Why this matters for AI eng roles:**
    > "I don't just write prompts, I engineer them — version-controlled, regression-tested in CI,
    > evaluated against a rubric. When a prompt change degrades performance, I know it immediately
    > because the eval suite catches it."

    ### The hierarchy of prompt investment

    | Level | Technique | Cost | Impact |
    |-------|-----------|------|--------|
    | 1 | Get the task definition right | Free | Highest |
    | 2 | Structure the output format (schemas, structured outputs) | Low | High |
    | 3 | Add examples (few-shot) | Low | High |
    | 4 | Add reasoning scaffolding (chain-of-thought) | Low | Medium-High |
    | 5 | Add self-verification (self-consistency, constitutional) | Medium | Medium |
    | 6 | Fine-tune | Expensive | Situation-dependent |

    **Move down this list only when the level above isn't good enough.**
    Most problems are solved by levels 1–3. CoT and self-consistency are tools for specific
    failure modes, not defaults.
    """)
    return


@app.cell
def shared_imports():
    import json
    import asyncio
    import numpy as np
    from dataclasses import dataclass
    from typing import Any
    return asyncio, dataclass, json, np


@app.cell
def zero_shot_concept(mo):
    mo.md("""
    ---
    ## Part 1: Zero-Shot — The Baseline

    Zero-shot: describe the task. No examples, no scaffolding. Just role + task + context + format.

    **When it works:** simple tasks, powerful models (GPT-4, Claude Sonnet/Opus), well-defined outputs.

    **When it fails:**
    - Ambiguous tasks where "good output" is hard to specify in prose
    - Specific formatting requirements the model hasn't internalized
    - Domain-specific terminology or scoring rubrics
    - Tasks where the model needs to calibrate against examples (e.g. "what does a 7 look like?")

    **The baseline rule:** always try zero-shot first. If it works, you're done. Don't over-engineer.
    """)
    return


@app.cell
def zero_shot_code(json):
    # ── Zero-shot prompt structure for Canopy job scoring ────────────────────────
    # This is the EXACT prompt text you'd send to an API. Not pseudocode.
    # Braces that are format() slots stay single; literal braces in JSON schema use {{}}.

    ZERO_SHOT_SCORER = """You are a job fit evaluator.

Rate how well this job matches the candidate's profile on a scale of 1-10.

Job Description:
{job_description}

Candidate Profile:
{candidate_profile}

Return a JSON object with:
- score: integer 1-10
- reasoning: one paragraph explaining the score
- matched_skills: list of skills that overlap
- gaps: list of requirements the candidate doesn't meet

Return ONLY the JSON object. No additional text."""

    # ── Mock LLM response (what the API would return) ─────────────────────────────
    MOCK_ZERO_SHOT_RESPONSE = """{
  "score": 7,
  "reasoning": "Strong Python and ML background matches core requirements. Eight years of experience exceeds the minimum. Location is compatible with remote work. Gaps are the specific Spark/Airflow tooling, though the candidate's Polars/DuckDB background is analogous.",
  "matched_skills": ["Python", "machine learning", "data processing", "senior level"],
  "gaps": ["Apache Spark", "Airflow", "pipeline orchestration"]
}"""

    JOB_DESCRIPTION = "Senior Data Engineer, Spark/Airflow/Python, 5+ years, remote-friendly"
    CANDIDATE_PROFILE = "8 years Python, ML engineering projects, Polars/DuckDB, San Antonio TX"

    # Render the prompt (what you'd actually send)
    rendered_prompt = ZERO_SHOT_SCORER.format(
        job_description=JOB_DESCRIPTION,
        candidate_profile=CANDIDATE_PROFILE,
    )

    # Parse the mock response
    result = json.loads(MOCK_ZERO_SHOT_RESPONSE)

    print("── Rendered prompt (sent to API) ─────────────────────────────────────────")
    print(rendered_prompt)
    print("\n── Mock LLM response ──────────────────────────────────────────────────────")
    print(json.dumps(result, indent=2))
    print(f"\nParsed score: {result['score']}/10")
    print(f"Gaps found: {result['gaps']}")

    return ZERO_SHOT_SCORER, JOB_DESCRIPTION, CANDIDATE_PROFILE


@app.cell
def zero_shot_note(mo):
    mo.md("""
    This might be enough for Canopy v1. Test it with your eval suite before adding complexity.
    The failure mode to watch for: scores cluster around 6–7 with no discrimination. Zero-shot
    models tend to hedge — they rarely score below 5 or above 8 without examples to calibrate against.
    """)
    return


@app.cell
def prompt_structure_heuristics(mo):
    mo.md("""
    ### Prompt Structure Heuristics

    | Component | Purpose | Example |
    |-----------|---------|---------|
    | **Role** | Constrains persona and expertise level | `"You are a job fit evaluator."` |
    | **Task** | Clear, specific instruction. One task per prompt. | `"Rate how well this job matches..."` |
    | **Constraints** | What NOT to do | `"Do not make up skills not mentioned in the JD."` |
    | **Context** | The data the model reasons over | job description + candidate profile |
    | **Format** | Exactly how output should be structured | JSON schema with specific field names |

    **Order matters:** role → task → constraints → context → format.

    Most important instructions go at the beginning and end — primacy and recency effects are real.
    Models attend more to the start and end of a prompt than the middle. Put your format spec
    at the end, not buried in the middle.
    """)
    return


@app.cell
def few_shot_concept(mo):
    mo.md("""
    ---
    ## Part 2: Few-Shot — Teaching by Example

    Few-shot: provide 2–5 input/output examples before the actual query. The examples communicate
    format, tone, reasoning depth, and edge case handling far more precisely than prose instructions.

    **Why it works:** "show, don't tell." Describing what a 9/10 score means is harder than showing
    an example that earned a 9. The model infers the implicit rubric from your examples.

    **How many examples:**
    - 2–3 for simple tasks (classification, extraction)
    - 5–8 for complex ones (multi-criteria reasoning)
    - More isn't always better — you're spending context window and cost

    **Selection is the skill:** choose examples that cover different cases — strong match, weak match,
    edge case with dealbreaker. Don't just show easy, clean examples.
    """)
    return


@app.cell
def few_shot_code(json):
    # ── Few-shot scorer — covers the full 1-10 range with 3 examples ─────────────
    # Key design: one 9 (great match), one 2 (skill mismatch), one 4 (partial + dealbreaker).
    # This teaches the model the FULL scoring range, not just what a great match looks like.

    FEW_SHOT_SCORER = """You are a job fit evaluator. Rate jobs 1-10 based on candidate fit.

Example 1:
Job: "Senior Python Engineer, ML focus, 5+ years, San Antonio"
Candidate: "8 years Python, ML projects, based in San Antonio"
Output: {{"score": 9, "reasoning": "Strong skill and location match. Experience exceeds requirement.", "matched_skills": ["Python", "ML", "San Antonio"], "gaps": []}}

Example 2:
Job: "Frontend React Developer, 3+ years, remote"
Candidate: "8 years Python, ML projects, based in San Antonio"
Output: {{"score": 2, "reasoning": "Fundamental skill mismatch. Role requires React/frontend expertise, candidate is Python/ML focused.", "matched_skills": ["remote-friendly"], "gaps": ["React", "frontend", "CSS/HTML"]}}

Example 3:
Job: "ML Engineer, NLP focus, PhD required, New York onsite"
Candidate: "8 years Python, ML projects, based in San Antonio"
Output: {{"score": 4, "reasoning": "ML skills align but PhD requirement and NYC onsite are blockers.", "matched_skills": ["Python", "ML"], "gaps": ["PhD", "NLP specialization", "NYC location"]}}

Now evaluate:
Job: "{job_description}"
Candidate: "{candidate_profile}"
Output:"""

    # Mock response for a new job — model should now calibrate correctly
    MOCK_FEW_SHOT_RESPONSE = """{
  "score": 7,
  "reasoning": "Strong Python/data background matches the core role. Eight years exceeds the 5-year requirement. Polars/DuckDB shows data processing depth. Gaps are Spark and Airflow specifically — but pattern is learnable given related experience. Remote-friendly removes location as a concern.",
  "matched_skills": ["Python", "data processing", "senior experience", "remote"],
  "gaps": ["Apache Spark", "Airflow"]
}"""

    JOB = "Senior Data Engineer, Spark/Airflow/Python, 5+ years, remote-friendly"
    CANDIDATE = "8 years Python, ML engineering, Polars/DuckDB, San Antonio TX"

    rendered = FEW_SHOT_SCORER.format(job_description=JOB, candidate_profile=CANDIDATE)
    result = json.loads(MOCK_FEW_SHOT_RESPONSE)

    print("── Few-shot prompt (sent to API) ──────────────────────────────────────────")
    print(rendered)
    print("\n── Mock response ──────────────────────────────────────────────────────────")
    print(json.dumps(result, indent=2))

    return FEW_SHOT_SCORER,


@app.cell
def few_shot_antipatterns(mo):
    mo.md("""
    ### Few-Shot Anti-Patterns

    | Anti-pattern | Effect | Fix |
    |--------------|--------|-----|
    | **All positive examples** | Model never learns what a bad match looks like → scores cluster 7+ | Include at least one low-score example |
    | **Too similar examples** | All examples are "Python engineer" variants → can't generalize | Cover different role types and skill mismatches |
    | **Example-output format mismatch** | Examples show format A, instructions ask for format B | Your examples ARE the format spec — keep them consistent |
    | **Stale examples** | 6-month-old examples with outdated skills/companies | Treat examples as test fixtures — update when evals degrade |

    **Mental model:** few-shot examples are test fixtures. If your unit tests were 6 months out of
    date, you'd update them. Same standard applies here. When your eval suite shows degradation,
    check whether the examples still represent the task correctly.
    """)
    return


@app.cell
def cot_concept(mo):
    mo.md("""
    ---
    ## Part 3: Chain-of-Thought (CoT) — Making the Model Think

    Chain-of-thought: ask the model to show its reasoning step-by-step before giving the final answer.

    **Why it works:** forces decomposition of complex problems into intermediate steps. Each step acts
    as "working memory" that informs the next. The model can't skip straight to a number — it has to
    work through the problem.

    **The original finding** (Wei et al., 2022): adding "Let's think step by step" to GSM8K math
    problems improved accuracy from 17% to 78% on PaLM. A single phrase, massive accuracy gain.

    ### Two flavors

    | Flavor | How | When |
    |--------|-----|------|
    | **Zero-shot CoT** | Add "Think step by step" or a numbered scaffold to the prompt | Quick win for moderate complexity |
    | **Few-shot CoT** | Provide examples that include the full reasoning chain, not just the answer | When you need to teach HOW to reason, not just what to output |

    **When to use:** multi-step reasoning, math, complex comparisons, scoring with multiple criteria,
    anything where the answer depends on weighing several factors simultaneously.
    """)
    return


@app.cell
def zero_shot_cot_code(json):
    # ── Zero-shot CoT — just add a numbered reasoning scaffold ────────────────────
    # The numbered steps aren't decorative. They create a chain the model must fill in.
    # If the score is wrong, you can read the step-by-step to find WHERE the reasoning broke.

    ZERO_SHOT_COT = """You are a job fit evaluator.

Think through this step by step:
1. First, identify the key requirements from the job description (required skills, experience level, location constraints)
2. Match each requirement against the candidate's profile — note explicit matches and gaps
3. Flag any dealbreakers (hard location mismatch, missing required credential, major skill gap)
4. Weigh the matches vs gaps — consider whether gaps are learnable or fundamental
5. Assign a score 1-10

Job Description:
{job_description}

Candidate Profile:
{candidate_profile}

Think through each step, then provide your final assessment as JSON:
{{"score": <int>, "reasoning": "<paragraph>", "step_by_step": ["<step1 finding>", "<step2 finding>", "<step3 finding>", "<step4 finding>"], "matched_skills": ["<skill>"], "gaps": ["<gap>"]}}"""

    MOCK_COT_RESPONSE = """{
  "score": 7,
  "reasoning": "Strong data engineering background with modern tooling. Experience level exceeds requirement. Spark/Airflow gap is real but learnable given depth in analogous tools. No hard dealbreakers — remote removes location constraint.",
  "step_by_step": [
    "Key requirements: Senior (5+ yrs), Python, Spark, Airflow, remote-friendly",
    "Matches: Python (8 yrs, exceeds), data processing depth (Polars/DuckDB ≈ Spark), remote OK. Gaps: Spark and Airflow specifically.",
    "No hard dealbreakers: no PhD requirement, no onsite mandate, no domain-specific credential required.",
    "Gaps are tooling-level, not skill-level — candidate clearly understands distributed data processing. 1-2 months to learn Spark/Airflow is realistic."
  ],
  "matched_skills": ["Python", "data processing", "senior level", "remote"],
  "gaps": ["Apache Spark", "Airflow"]
}"""

    JOB = "Senior Data Engineer, Spark/Airflow/Python, 5+ years, remote-friendly"
    CANDIDATE = "8 years Python, ML engineering, Polars/DuckDB, San Antonio TX"

    rendered = ZERO_SHOT_COT.format(job_description=JOB, candidate_profile=CANDIDATE)
    result = json.loads(MOCK_COT_RESPONSE)

    print("── Zero-shot CoT prompt ───────────────────────────────────────────────────")
    print(rendered)
    print("\n── Mock response with step-by-step ────────────────────────────────────────")
    print(json.dumps(result, indent=2))
    print("\n── Step-by-step reasoning (inspectable) ───────────────────────────────────")
    for i, step in enumerate(result["step_by_step"], 1):
        print(f"  Step {i}: {step}")

    return ZERO_SHOT_COT,


@app.cell
def few_shot_cot_code(json):
    # ── Few-shot CoT — example includes the full reasoning chain ──────────────────
    # The example teaches HOW to reason (what to check, in what order, how to weigh),
    # not just what the output format looks like.

    FEW_SHOT_COT = """Example:
Job: "Senior Data Engineer, Spark/Airflow, 5+ years, Austin hybrid"
Candidate: "8 years Python, ML projects, Polars/DuckDB, San Antonio"

Step-by-step thinking:
1. Key requirements: Senior level (5+ years), Data Engineering focus, Spark, Airflow, Austin hybrid
2. Skill match: Python ✓, data processing ✓ (Polars/DuckDB are modern alternatives to Spark). Missing Airflow directly but has pipeline experience.
3. Experience: 8 years exceeds the 5+ requirement ✓
4. Location: San Antonio → Austin hybrid is commutable (~1 hr drive), acceptable ✓
5. Dealbreakers: No hard dealbreakers. Spark gap is learnable given strong data processing background.
6. Weighing: Strong experience + related skills + commutable location. Spark/Airflow gap is moderate, not fatal.

Output: {{"score": 7, "reasoning": "Strong data background with modern tooling that translates well. Spark/Airflow gap is notable but learnable. Austin hybrid is commutable from SA.", "matched_skills": ["Python", "data processing", "senior experience"], "gaps": ["Spark", "Airflow"]}}

---

Now evaluate:
Job: "{job_description}"
Candidate: "{candidate_profile}"

Step-by-step thinking:"""

    # Mock: model follows the same reasoning structure from the example
    MOCK_FEW_SHOT_COT_RESPONSE = """1. Key requirements: ML Engineer, Python, TensorFlow, 3+ years, remote
2. Skill match: Python ✓ (8 years), ML engineering ✓, TensorFlow — candidate uses PyTorch, different framework but same paradigm.
3. Experience: 8 years far exceeds 3+ requirement ✓
4. Location: remote — no constraint ✓
5. Dealbreakers: TensorFlow vs PyTorch is the main gap. Not a dealbreaker for a strong ML engineer.
6. Weighing: Strong ML background, experience exceeds requirement, location works. Framework gap is a minor ramp-up, not a fundamental skill gap.

Output: {"score": 8, "reasoning": "Strong ML engineering background well above the experience bar. PyTorch vs TensorFlow is a minor gap for an experienced ML engineer — framework transfer is a week of work, not a career shift. Remote works perfectly.", "matched_skills": ["Python", "ML engineering", "senior level", "remote"], "gaps": ["TensorFlow (has PyTorch)"]}"""

    JOB = "ML Engineer, Python/TensorFlow, 3+ years, remote"
    CANDIDATE = "8 years Python, ML engineering with PyTorch, San Antonio TX"

    rendered = FEW_SHOT_COT.format(job_description=JOB, candidate_profile=CANDIDATE)
    print("── Few-shot CoT prompt ────────────────────────────────────────────────────")
    print(rendered)
    print("\n── Mock model response (follows example's reasoning pattern) ───────────────")
    print(MOCK_FEW_SHOT_COT_RESPONSE)

    return


@app.cell
def cot_when_it_hurts(mo):
    mo.md("""
    ### When CoT Hurts

    | Scenario | Why CoT is wasteful |
    |----------|---------------------|
    | **Simple classification** | "Is this email spam?" doesn't need step-by-step reasoning |
    | **Very short outputs** | If you need yes/no, CoT wastes output tokens producing reasoning you don't use |
    | **Plausible-but-wrong chains** | The model produces a reasoning chain that looks logical but reaches the wrong conclusion. Don't blindly trust the chain. |
    | **Cost-sensitive bulk tasks** | CoT generates 3–5× more output tokens → 3–5× more expensive |

    **Temperature note:** use temperature=0 for CoT (deterministic reasoning). Reserve temperature > 0
    for self-consistency (where you WANT variation). A CoT prompt at temperature=0.7 will produce
    different reasoning chains on each call — that's only useful if you're aggregating them.
    """)
    return


@app.cell
def self_consistency_concept(mo):
    mo.md("""
    ---
    ## Part 4: Self-Consistency — Ensemble for LLMs

    **The problem:** LLMs are non-deterministic at temperature > 0. Same prompt → different answers.
    One sample might be wrong, but the consensus of many is usually right.

    **Self-consistency:** run the SAME prompt N times (temperature > 0), take the majority vote
    or median. This is the random forest of prompting — one tree might be wrong, the ensemble is
    more reliable.

    **Implementation:** 3–5 runs, each with CoT, majority vote on the final answer (ignore the
    reasoning chains — they'll differ. You want the answer distribution, not the reasoning distribution).

    **When to use:**
    - High-stakes decisions where you can afford 3–5× latency and cost
    - The ambiguous zone where you're not confident in a single-sample answer
    - In Canopy: jobs scoring 5–7 (the uncertain range). Clear matches (9–10) and mismatches (1–3) don't need it.
    """)
    return


@app.cell
def self_consistency_code(np, asyncio, json):
    # ── Self-consistency scorer ───────────────────────────────────────────────────
    # Runs the same CoT prompt N times, aggregates via median and confidence interval.
    # Mocked — shows the exact structure you'd use with a real async LLM client.

    # Mock: simulate N LLM calls returning different scores (temperature > 0 = variation)
    MOCK_SCORE_DISTRIBUTIONS = {
        "high_confidence": [8, 8, 7, 8, 8],    # low variance → confident 8
        "low_confidence":  [3, 7, 5, 8, 4],    # high variance → flag for human review
        "borderline":      [6, 7, 6, 7, 6],    # slight variance → solid 6
    }

    async def mock_llm_score(job_description: str, profile: str, temperature: float) -> dict:
        """Mock single LLM call — real version would call the API with temperature param."""
        await asyncio.sleep(0)
        # Return a plausible score based on keyword overlap (mock logic)
        score = 7 if "Python" in job_description or "ML" in job_description else 5
        return {"score": score, "reasoning": f"Mock reasoning at temp={temperature}"}

    async def self_consistent_score(
        job_description: str,
        profile: str,
        n_samples: int = 5,
        mock_scores: list[int] | None = None,
    ) -> dict:
        """Run scorer N times, return median score + confidence metric."""
        scores = []
        reasonings = []

        for i in range(n_samples):
            if mock_scores:
                scores.append(mock_scores[i])
                reasonings.append(f"Reasoning sample {i+1}")
            else:
                result = await mock_llm_score(job_description, profile, temperature=0.7)
                scores.append(result["score"])
                reasonings.append(result["reasoning"])

        median_score = float(np.median(scores))
        std = float(np.std(scores))
        confidence = round(1.0 - (std / 10.0), 3)  # low std → high confidence

        return {
            "score": round(median_score),
            "confidence": confidence,
            "score_range": (min(scores), max(scores)),
            "std_dev": round(std, 2),
            "n_samples": n_samples,
            "all_scores": scores,
            "needs_human_review": confidence < 0.85,
        }

    JOB = "Senior Data Engineer, Spark/Airflow/Python, 5+ years, remote-friendly"
    CANDIDATE = "8 years Python, ML engineering, Polars/DuckDB, San Antonio TX"

    print("── Self-consistency: three scenarios ──────────────────────────────────────\n")

    for scenario_name, mock_scores in MOCK_SCORE_DISTRIBUTIONS.items():
        result = asyncio.run(
            self_consistent_score(JOB, CANDIDATE, n_samples=5, mock_scores=mock_scores)
        )
        review_flag = "⚠ FLAG FOR HUMAN REVIEW" if result["needs_human_review"] else "✓ auto-queue"
        print(f"[{scenario_name}]")
        print(f"  Scores:     {result['all_scores']}")
        print(f"  Median:     {result['score']}/10")
        print(f"  Confidence: {result['confidence']} (std={result['std_dev']})")
        print(f"  Decision:   {review_flag}")
        print()

    return


@app.cell
def self_consistency_note(mo):
    mo.md("""
    The confidence score is the real value here. When self-consistency produces scores of
    `[3, 7, 5, 8, 4]`, that means the model is genuinely uncertain — not that it's broken.
    This job should go to human review, not be auto-scored.

    **Canopy application:** run self-consistency only on the 5–7 range. Clear matches (9–10) and
    mismatches (1–3) get a single CoT call at temperature=0. Self-consistency is a targeted tool
    for the uncertain middle, not a default for all scoring.

    **Cost math:** 5 samples = 5× API calls. At $3/M output tokens (Claude Sonnet), a CoT response
    of ~300 tokens costs $0.0009. Self-consistency costs $0.0045. Cheap enough for the ambiguous zone.
    """)
    return


@app.cell
def constitutional_concept(mo):
    mo.md("""
    ---
    ## Part 5: Constitutional AI — Self-Correcting Outputs

    **Constitutional AI** (Anthropic, 2022): the model critiques its own output against a set of
    principles, then revises. Generate → critique → revise. Sometimes multiple rounds.

    "Constitutional" because the principles act like a constitution the model must follow. The
    model doesn't just generate — it generates AND checks AND revises until it passes its own
    critique.

    **Broader than safety:** the original Anthropic paper used this for harmlessness. But the
    pattern generalizes to any quality dimension — factuality, tone, length, format, bias,
    citation accuracy.

    **The loop:**
    ```
    generate(task) → critique(output, principles) → revise(output, critique) → [repeat if needed] → done
    ```
    """)
    return


@app.cell
def constitutional_code(json):
    # ── Constitutional AI: generate → critique → revise for cover letter generation ─
    # Canopy v1 generates cover letters with no self-check.
    # Constitutional loop adds a quality gate baked into the generation process.

    # ── Step 1: Generate ──────────────────────────────────────────────────────────
    GENERATE_COVER_LETTER = """Write a cover letter for this job application.

Job Description:
{job_description}

Resume Highlights:
{resume_highlights}

Keep it under 250 words. Be specific and concrete — no generic phrases."""

    # ── Step 2: Critique against principles ──────────────────────────────────────
    CRITIQUE_COVER_LETTER = """Review this cover letter against these principles:

1. Does it mention a specific project from the resume that's directly relevant to this job?
2. Is it under 250 words?
3. Does it avoid generic phrases like "I'm a hard worker", "passionate about technology", or "team player"?
4. Does it reference something specific about the company or role (not just the job title)?
5. Is the tone confident but not arrogant?

Cover letter to review:
{draft}

For each principle (1-5), state PASS or FAIL with a one-sentence explanation.
Format: {{"1": {{"result": "PASS/FAIL", "reason": "..."}}, "2": ..., "3": ..., "4": ..., "5": ...}}"""

    # ── Step 3: Revise ────────────────────────────────────────────────────────────
    REVISE_COVER_LETTER = """Revise this cover letter to fix the issues identified in the critique.

Original draft:
{draft}

Critique:
{critique}

Fix every FAIL while keeping everything that passed. Return ONLY the revised cover letter text."""

    # ── Mock LLM responses for each step ─────────────────────────────────────────
    MOCK_DRAFT = """Dear Hiring Manager,

I am writing to express my strong interest in the Senior Data Engineer position. I am a hard worker with a passion for technology and would be a great team player at your company.

I have 8 years of Python experience and have worked on various data projects. I am confident I can contribute to your data infrastructure needs.

I look forward to discussing this opportunity.

Sincerely,
Danilo"""

    MOCK_CRITIQUE = """{
  "1": {"result": "FAIL", "reason": "No specific project from the resume is mentioned — just vague 'data projects'."},
  "2": {"result": "PASS", "reason": "Draft is ~90 words, well under 250."},
  "3": {"result": "FAIL", "reason": "Contains 'hard worker', 'passion for technology', and 'team player' — all generic."},
  "4": {"result": "FAIL", "reason": "No mention of anything specific about the company or why this particular role."},
  "5": {"result": "PASS", "reason": "Tone is professional and not arrogant."}
}"""

    MOCK_REVISION = """Dear Hiring Manager,

Your investment in modern data infrastructure — particularly the shift to streaming pipelines — is what drew me to this role.

My most relevant work: at [Company], I built a Polars-based data processing pipeline that reduced batch job runtime from 4 hours to 22 minutes. That project required the same skills your JD emphasizes: understanding data at scale, knowing when Spark is the right tool vs. when a lighter-weight solution performs better.

Eight years of Python engineering, with the last three focused on data systems. I've used Polars and DuckDB extensively — not because Spark isn't available, but because I pick tools based on the problem, not habit. I'd ramp on Airflow quickly; I already understand the orchestration concepts from managing similar pipelines.

I'd welcome a conversation about the specific data challenges you're solving.

Danilo"""

    # ── The full generate-critique-revise loop ────────────────────────────────────
    def constitutional_loop(
        job_description: str,
        resume_highlights: str,
        max_revisions: int = 2,
    ) -> str:
        """
        Generate → critique → revise loop. Returns final cover letter.
        In production: each mock_llm() call is a real API call.
        """
        def mock_llm(prompt: str, step: str) -> str:
            # Mock: return pre-baked response based on step
            if step == "generate":
                return MOCK_DRAFT
            elif step == "critique":
                return MOCK_CRITIQUE
            elif step == "revise":
                return MOCK_REVISION
            return ""

        # Generate initial draft
        draft_prompt = GENERATE_COVER_LETTER.format(
            job_description=job_description,
            resume_highlights=resume_highlights,
        )
        draft = mock_llm(draft_prompt, "generate")
        print("── Step 1: Initial draft ──────────────────────────────────────────────────")
        print(draft)

        for revision_round in range(max_revisions):
            # Critique
            critique_prompt = CRITIQUE_COVER_LETTER.format(draft=draft)
            critique_raw = mock_llm(critique_prompt, "critique")
            critique = json.loads(critique_raw)

            print(f"\n── Step 2: Critique (round {revision_round + 1}) ────────────────────────────────────")
            passes = [k for k, v in critique.items() if v["result"] == "PASS"]
            fails  = [k for k, v in critique.items() if v["result"] == "FAIL"]
            print(f"  PASS: principles {passes}")
            print(f"  FAIL: principles {fails}")
            for k, v in critique.items():
                if v["result"] == "FAIL":
                    print(f"    [{k}] {v['reason']}")

            if not fails:
                print("\n  All principles pass — no revision needed.")
                break

            # Revise
            revise_prompt = REVISE_COVER_LETTER.format(
                draft=draft,
                critique=critique_raw,
            )
            draft = mock_llm(revise_prompt, "revise")
            print(f"\n── Step 3: Revised draft (round {revision_round + 1}) ──────────────────────────────")
            print(draft)

        return draft

    JOB = "Senior Data Engineer, Spark/Airflow/Python, 5+ years, remote-friendly"
    RESUME = "8 yrs Python; built Polars pipeline reducing batch runtime 4h→22min; ML engineering; Polars/DuckDB expertise"

    final_letter = constitutional_loop(JOB, RESUME)
    print(f"\n── Final output ───────────────────────────────────────────────────────────")
    print(f"Length: {len(final_letter.split())} words")

    return GENERATE_COVER_LETTER, CRITIQUE_COVER_LETTER, REVISE_COVER_LETTER,


@app.cell
def constitutional_projects(mo):
    mo.md("""
    ### My Projects — Constitutional AI in Practice

    **Canopy cover letter generation:** Canopy v1 generates a draft but has no self-check.
    The constitutional loop adds a quality gate: "mentions a real project? under 250 words?
    company-specific? no generic fluff?" Each principle is testable and version-controlled.

    **Briefing Agent:** constitutional check on summaries: "is every citation a real paper?
    is this within topic scope? is it under 200 words?" Currently done as post-hoc guardrails.
    Constitutional approach bakes the check INTO the generation loop — no separate validation pass needed.

    **Interview framing:**
    > "I'd apply constitutional AI to Canopy's cover letter generator: generate a draft, critique
    > against 5 quality principles, revise if any fail. The principles are the spec — they're
    > testable, documentable, and version-controlled. When a letter fails principle 3 consistently,
    > I update the generation prompt. The critique loop makes failure modes visible."
    """)
    return


@app.cell
def prompt_chaining_concept(mo):
    mo.md("""
    ---
    ## Part 6: Advanced Patterns

    ### Prompt Chaining

    Break a complex task into sequential prompts where each step's output feeds the next.

    **Why chaining over one big prompt:**
    - Each step is simpler → more reliable output
    - Each step is independently testable
    - You can cache intermediate results (if step 1 output is stable, don't regenerate it)
    - You can use different models per step — cheap model for extraction, expensive for reasoning

    **For Canopy:**
    1. Prompt 1: "Extract the top 5 requirements from this JD" → structured list (cheap model)
    2. Prompt 2: "Match each requirement against this resume" → match table (cheap model)
    3. Prompt 3: "Given these matches, score the fit 1-10 with reasoning" → final score (expensive model)
    """)
    return


@app.cell
def prompt_chaining_code(json, asyncio):
    # ── Chained scorer: three steps with model routing ────────────────────────────
    # Step 1 and 2 use a cheap model (GPT-4o-mini, $0.15/M tokens).
    # Only step 3 uses the expensive model (Claude Sonnet, $3/M tokens).

    EXTRACT_REQUIREMENTS = """Extract the top 5 requirements from this job description.

Job Description:
{job_description}

Return a JSON list of requirement objects:
[{{"requirement": "...", "importance": "required/preferred", "category": "skill/experience/location/credential"}}]"""

    MATCH_REQUIREMENTS = """Match each job requirement against the candidate's profile.

Requirements:
{requirements}

Candidate Profile:
{candidate_profile}

Return a JSON list — one object per requirement:
[{{"requirement": "...", "match": "full/partial/none", "evidence": "quote from profile or 'not mentioned'"}}]"""

    SCORE_FROM_MATCHES = """You are a senior technical recruiter. Score this candidate's fit based on the requirement analysis.

Job Requirements and Matches:
{matches}

Original Job Description:
{job_description}

Score 1-10 considering:
- Weight "required" requirements more than "preferred"
- Partial matches on learnable skills count more than partial matches on core skills
- Flag any dealbreakers explicitly

Return JSON: {{"score": <int>, "reasoning": "<paragraph>", "dealbreakers": [], "strengths": []}}"""

    # ── Mock responses for each step ─────────────────────────────────────────────
    MOCK_REQUIREMENTS = """[
  {"requirement": "Python (senior level)", "importance": "required", "category": "skill"},
  {"requirement": "Apache Spark", "importance": "required", "category": "skill"},
  {"requirement": "Airflow or similar orchestration", "importance": "required", "category": "skill"},
  {"requirement": "5+ years data engineering", "importance": "required", "category": "experience"},
  {"requirement": "Remote-friendly location", "importance": "preferred", "category": "location"}
]"""

    MOCK_MATCHES = """[
  {"requirement": "Python (senior level)", "match": "full", "evidence": "8 years Python engineering"},
  {"requirement": "Apache Spark", "match": "partial", "evidence": "uses Polars/DuckDB — distributed data processing but not Spark specifically"},
  {"requirement": "Airflow or similar orchestration", "match": "partial", "evidence": "pipeline experience but Airflow not mentioned directly"},
  {"requirement": "5+ years data engineering", "match": "full", "evidence": "8 years exceeds requirement"},
  {"requirement": "Remote-friendly location", "match": "full", "evidence": "San Antonio — no constraint for remote role"}
]"""

    MOCK_SCORE = """{
  "score": 7,
  "reasoning": "Strong Python/data engineering foundation exceeds experience bar. Spark and Airflow gaps are real but the candidate's Polars/DuckDB depth shows they understand distributed data processing — the tooling transfer is a ramp, not a fundamental gap. All required criteria have at least partial matches.",
  "dealbreakers": [],
  "strengths": ["8yr Python exceeds bar", "distributed data processing depth", "no location constraint"]
}"""

    async def chained_scorer(job_description: str, candidate_profile: str) -> dict:
        """Three-step chained scorer with model routing."""
        print("── Step 1: Extract requirements (cheap model) ─────────────────────────────")
        req_prompt = EXTRACT_REQUIREMENTS.format(job_description=job_description)
        # mock_llm_mini(req_prompt)  — real: call GPT-4o-mini
        requirements = json.loads(MOCK_REQUIREMENTS)
        print(f"  Extracted {len(requirements)} requirements")
        for r in requirements:
            print(f"    [{r['importance'].upper()}] {r['requirement']} ({r['category']})")

        print("\n── Step 2: Match requirements (cheap model) ────────────────────────────────")
        match_prompt = MATCH_REQUIREMENTS.format(
            requirements=json.dumps(requirements),
            candidate_profile=candidate_profile,
        )
        # mock_llm_mini(match_prompt)  — real: call GPT-4o-mini
        matches = json.loads(MOCK_MATCHES)
        for m in matches:
            icon = "✓" if m["match"] == "full" else ("~" if m["match"] == "partial" else "✗")
            print(f"    {icon} {m['requirement']}: {m['match']}")

        print("\n── Step 3: Score from matches (expensive model) ────────────────────────────")
        score_prompt = SCORE_FROM_MATCHES.format(
            matches=json.dumps(matches),
            job_description=job_description,
        )
        # mock_llm_sonnet(score_prompt)  — real: call Claude Sonnet
        score_result = json.loads(MOCK_SCORE)
        print(f"  Score: {score_result['score']}/10")
        print(f"  Reasoning: {score_result['reasoning']}")
        print(f"  Dealbreakers: {score_result['dealbreakers'] or 'none'}")

        return score_result

    JOB = "Senior Data Engineer, Spark/Airflow/Python, 5+ years, remote-friendly"
    CANDIDATE = "8 years Python, ML engineering, Polars/DuckDB, San Antonio TX"

    final = asyncio.run(chained_scorer(JOB, CANDIDATE))
    print(f"\n── Cost note ──────────────────────────────────────────────────────────────")
    print("  Steps 1-2: GPT-4o-mini at $0.15/M tokens")
    print("  Step 3:    Claude Sonnet at $3/M tokens")
    print("  Model routing at the prompt level — expensive model only for complex reasoning")

    return


@app.cell
def prompt_template_code(dataclass, json):
    # ── Production-grade prompt template class ────────────────────────────────────
    # Prompts as files, not inline strings. Versioned in git.
    # Validated at render time, output validated against schema.

    from pathlib import Path

    @dataclass
    class PromptTemplate:
        """Version-controlled prompt with render-time validation."""
        name: str
        version: str
        template: str
        max_output_tokens: int
        required_output_fields: list

        def render(self, **kwargs) -> str:
            """Render template with variables. Raises if any slots are unfilled."""
            result = self.template.format(**kwargs)
            # Detect unfilled slots — any remaining { } pair that looks like a placeholder
            import re
            unfilled = re.findall(r'\{[a-z_]+\}', result)
            if unfilled:
                raise ValueError(f"Unfilled template slots: {unfilled}")
            return result

        def validate_output(self, output: dict) -> list:
            """Returns list of missing required fields (empty = valid)."""
            return [f for f in self.required_output_fields if f not in output]

    # ── Example: versioned scorer prompt ─────────────────────────────────────────
    scorer_v2 = PromptTemplate(
        name="job_scorer",
        version="2.1.0",
        template="""You are a job fit evaluator.

Think step by step, then score this job 1-10.

Job Description:
{job_description}

Candidate Profile:
{candidate_profile}

Return JSON: {{"score": <int>, "reasoning": "<str>", "matched_skills": [], "gaps": []}}""",
        max_output_tokens=512,
        required_output_fields=["score", "reasoning", "matched_skills", "gaps"],
    )

    # ── Demonstrate render-time validation ────────────────────────────────────────
    print("── Prompt template v{} ──────────────────────────────────────────────────".format(scorer_v2.version))
    print(f"  Name: {scorer_v2.name}")
    print(f"  Required output fields: {scorer_v2.required_output_fields}")

    rendered = scorer_v2.render(
        job_description="Senior Data Engineer, Spark/Airflow/Python, 5+ years, remote",
        candidate_profile="8 years Python, ML engineering, Polars/DuckDB, San Antonio",
    )
    print(f"\n── Rendered ({len(rendered)} chars) ───────────────────────────────────────────")
    print(rendered)

    # ── Demonstrate missing-slot detection ────────────────────────────────────────
    print("\n── Missing slot detection ─────────────────────────────────────────────────")
    try:
        scorer_v2.render(job_description="Some job")  # missing candidate_profile
    except ValueError as e:
        print(f"  Caught: {e}")

    # ── Demonstrate output validation ─────────────────────────────────────────────
    good_output = {"score": 7, "reasoning": "...", "matched_skills": ["Python"], "gaps": ["Spark"]}
    bad_output  = {"score": 7, "reasoning": "..."}  # missing fields

    print("\n── Output validation ──────────────────────────────────────────────────────")
    print(f"  Good output missing fields: {scorer_v2.validate_output(good_output) or 'none ✓'}")
    print(f"  Bad output missing fields:  {scorer_v2.validate_output(bad_output)}")

    return


@app.cell
def prompt_template_note(mo):
    mo.md("""
    Prompts as files, not inline strings. Versioned in git alongside the code that uses them.
    Templates validated at render time. Output validated against a required-fields schema.
    This is how you prevent the "someone changed the prompt and everything broke" problem —
    the failure is a `ValueError` at render time, not a silent wrong output at runtime.
    """)
    return


@app.cell
def pattern_selection(mo):
    mo.md("""
    ---
    ## Pattern Selection Framework

    ```
    How complex is the task?
    ├── Simple (classify, extract)        → Zero-shot. Done.
    ├── Moderate (score, compare)         → Few-shot with 3 diverse examples.
    ├── Complex (multi-criteria reasoning) → Few-shot CoT.
    └── Very complex (multi-step analysis) → Prompt chaining.

    How important is correctness?
    ├── Low stakes    → single call, temperature 0
    ├── Medium stakes → CoT for inspectable reasoning
    ├── High stakes   → self-consistency (3-5 samples)
    └── Safety/quality-critical → constitutional AI (generate→critique→revise)

    What's your budget?
    ├── Tight   → zero-shot with small model (GPT-4o-mini, Haiku)
    ├── Moderate → few-shot with mid-tier model (Sonnet, GPT-4o)
    └── Flexible → chained prompts, expensive model for reasoning step only,
                   self-consistency for the ambiguous zone
    ```

    **Default decision:** zero-shot CoT at temperature=0. Add complexity only when evals justify it.
    """)
    return


@app.cell
def flashcard_summary(mo):
    mo.md("""
    ---
    ## Flashcard Summary

    | Question | Answer |
    |----------|--------|
    | **What's few-shot prompting?** | Provide 2–5 input/output examples before the query. Teaches format, tone, and edge cases by demonstration — "show don't tell." |
    | **When does CoT help?** | Multi-step reasoning, math, complex comparisons. Adds step-by-step scaffold. Hurts for simple tasks (wasted tokens and cost). |
    | **What's self-consistency?** | Run same prompt N times at temperature > 0, take majority/median. Like random forest for prompts. High score variance = low confidence = flag for human review. |
    | **What's constitutional AI?** | Generate → critique against principles → revise. Self-correcting loop. Use for quality enforcement, not just safety. |
    | **Few-shot anti-pattern?** | All positive examples, too similar examples, stale examples. Need diverse cases covering the full output range. |
    | **Chaining vs one big prompt?** | Chaining: each step simpler (more reliable), independently testable, allows model routing. Use for complex multi-step tasks. |
    | **How do you version prompts?** | Text files in git, PromptTemplate class with render-time validation, output schema validation, regression evals in CI. |
    | **Temperature for CoT?** | Temperature=0 for deterministic CoT. Temperature > 0 only for self-consistency (you need the variation to aggregate). |
    | **When does CoT hurt?** | Simple tasks, very short outputs, when cost matters for bulk classification. Also: plausible-but-wrong chains exist — inspect, don't blindly trust. |
    | **Self-consistency — when to skip?** | Clear matches (9–10) and mismatches (1–3) don't need it. Reserve for the uncertain middle range. |
    """)
    return


@app.cell
def interview_talking_points(mo):
    mo.md("""
    ---
    ## Interview Talking Points

    **"How do you approach prompt engineering?"**
    > "I start with zero-shot and only add complexity when eval scores justify it. Few-shot for
    > format and edge case calibration. CoT for multi-criteria reasoning. Self-consistency for
    > high-stakes ambiguous decisions. Constitutional for quality enforcement. Every prompt is
    > version-controlled and regression-tested — when a prompt change degrades evals, the PR is blocked."

    ---

    **"Tell me about a prompt you iterated on."**
    > "Canopy's job scorer went through 3 versions: v1 was zero-shot (scores clustered 6–7, no
    > discrimination between good and mediocre matches). v2 added few-shot examples covering the
    > full 1–10 range — a 9, a 2, and a 4 with a dealbreaker. Much better spread. v3 added CoT
    > for the reasoning step — caught edge cases like 'great skills but wrong location.' Each version
    > was evaluated against 50 labeled jobs with Spearman correlation as the primary metric."

    ---

    **"How do you know a prompt is working?"**
    > "Eval suite. 50+ test cases with expected outputs. Run on every prompt change. I measure
    > score correlation (Spearman vs human labels), format compliance rate, and reasoning quality
    > (LLM-as-judge on a 1-5 rubric). If any metric degrades past threshold, the PR is blocked."

    ---

    **"Self-consistency seems expensive. When do you use it?"**
    > "Only for the ambiguous zone. In Canopy, clear matches (9–10) and mismatches (1–3) get a
    > single CoT call at temperature=0. Jobs scoring 5–7 get 3–5 samples. The confidence score —
    > basically 1 minus the score variance — determines whether it goes to auto-queue or human
    > review. The cost is worth it precisely because I'm not running it on everything."

    ---

    **Connection to my work**
    > "Every pattern in this notebook maps to something I've built or would build. Constitutional AI
    > is the pattern behind citation validation in my briefing agent — generate a summary, critique
    > whether every citation is a real paper, revise if not. Prompt chaining is how I'd restructure
    > Canopy's scorer to route cheap extraction to GPT-4o-mini and only send the reasoning step to
    > Claude Sonnet. These aren't abstract patterns — they're engineering decisions with known cost
    > and quality tradeoffs."
    """)
    return


if __name__ == "__main__":
    app.run()
