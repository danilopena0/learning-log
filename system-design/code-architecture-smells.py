import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def header(mo):
    mo.md("""
    # Code & Architecture Smells — Anti-Patterns, Clean Code Heuristics, Edge Cases

    | Field  | Value |
    |--------|-------|
    | Date   | 2026-04-28 |
    | Track  | System Design / Software Engineering |
    | Time   | 60 min |
    | Topics | Function Smells · Class Smells · Architecture Smells · Python-Specific Traps |
    """)
    return


@app.cell
def why_smells_matter(mo):
    mo.md("""
    ## Why Smells Matter for Interviews

    **Code smells** = surface symptoms that something deeper is wrong. The code *works* but is
    fragile, hard to test, and hard to extend. **Architecture smells** = structural problems that
    compound over time — fine at demo scale, catastrophic at production scale.

    ### Interview reality

    Interviewers scan your take-home or portfolio code for smells in 5 minutes. One God class or
    hardcoded API key kills the impression faster than a clever algorithm saves it.

    - **Take-home assignments** get rejected for code smells even when they work correctly.
    - **System design follow-ups** often include "how would you structure this code?" — knowing the
      vocabulary (SRP, tight coupling, DI) is the difference between junior and senior answers.
    - **Showing awareness of anti-patterns signals production experience.** Anyone can write code
      that runs. Production engineers write code the team can maintain.

    > *"Clean code isn't about perfection. It's about making the NEXT engineer's job easier —
    > and that engineer might be you in 3 months."*

    ---

    ### How to use this notebook

    Each smell has:
    - **BAD** — the pattern to recognize and avoid
    - **GOOD** — the refactored version
    - **Heuristic** — one sentence to remember it
    - **Edge case / connection** — when the rule bends, or how it maps to my actual work
    """)
    return


@app.cell
def part1_header(mo):
    mo.md("""
    ## Part 1: Function-Level Smells
    """)
    return


@app.cell
def smell1_god_function(mo):
    mo.md("""
    ### Smell 1: Functions That Do Too Much

    The **God function** — one function that fetches, cleans, transforms, trains, evaluates,
    and saves. It works on the first run and is a nightmare to debug, test, or reuse after.

    ---

    **BAD — one function doing five jobs, 80+ lines with no seams:**

    ```python
    def run_pipeline(url, output_path):
        response = requests.get(url)
        data = json.loads(response.text)
        df = pd.DataFrame(data)
        df = df.dropna()
        df['amount'] = df['amount'].clip(0, 10000)
        df['category'] = df['category'].map(CATEGORY_MAP)
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = XGBClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, preds)}")
        joblib.dump(model, output_path)
    ```

    **GOOD — Single Responsibility, each function testable in isolation:**

    ```python
    def fetch_data(url: str) -> dict:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def clean_data(raw: dict) -> pd.DataFrame:
        df = pd.DataFrame(raw)
        df = df.dropna()
        df['amount'] = df['amount'].clip(0, 10000)
        df['category'] = df['category'].map(CATEGORY_MAP)
        return df

    def train_model(df: pd.DataFrame) -> tuple[XGBClassifier, float]:
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = XGBClassifier()
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        return model, accuracy

    def run_pipeline(url: str, output_path: str):
        raw = fetch_data(url)
        df = clean_data(raw)
        model, accuracy = train_model(df)
        logger.info(f"Accuracy: {accuracy}")
        joblib.dump(model, output_path)
    ```

    **Heuristic:** "If you can't describe what a function does without using *and*, it does too much."

    **Edge case:** Don't over-decompose. A function that wraps one line adds indirection without
    clarity. 3–20 lines is the sweet spot. The goal is testability and replaceability, not line count.
    """)
    return


@app.cell
def smell2_boolean_params(mo):
    mo.md("""
    ### Smell 2: Boolean Parameters That Change Behavior

    A boolean flag that causes a function to do fundamentally different things is a code smell
    for a function that wants to be two functions. Callers can't understand what the flag does
    without reading the implementation.

    ---

    **BAD — boolean flag soup, behavior invisible at the call site:**

    ```python
    def process_jobs(jobs: list[dict], include_metadata: bool = False,
                     score: bool = True, notify: bool = False) -> list:
        results = []
        for job in jobs:
            if score:
                job['score'] = compute_score(job)
            if include_metadata:
                job['metadata'] = fetch_metadata(job['id'])
            results.append(job)
        if notify:
            send_notification(results)
        return results
    ```

    **GOOD — separate functions for separate behaviors, composable by the caller:**

    ```python
    def score_jobs(jobs: list[Job]) -> list[ScoredJob]:
        return [ScoredJob(**job.dict(), score=compute_score(job)) for job in jobs]

    def enrich_with_metadata(jobs: list[Job]) -> list[EnrichedJob]:
        return [EnrichedJob(**job.dict(), metadata=fetch_metadata(job.id)) for job in jobs]

    def notify_matches(jobs: list[ScoredJob], threshold: float = 7.0):
        matches = [j for j in jobs if j.score >= threshold]
        if matches:
            send_notification(matches)
    ```

    **Heuristic:** "A boolean parameter is a code smell for a function that wants to be two functions."

    **Exception:** Simple flags like `verbose=True` or `dry_run=True` are fine — they modify *how*
    the function runs, not *what* it does. The smell is when the flag changes the fundamental behavior.
    """)
    return


@app.cell
def smell3_ambiguous_none(mo):
    mo.md("""
    ### Smell 3: Returning None Ambiguously

    When a function returns `None` for both "not found" and "error", the caller can't distinguish
    between the two cases. Silent errors masquerade as expected empty results.

    ---

    **BAD — `None` means "not found" *and* "error" — indistinguishable, plus SQL injection:**

    ```python
    def find_job(job_id: str) -> dict | None:
        try:
            result = db.query(f"SELECT * FROM jobs WHERE id = '{job_id}'")
            if result:
                return result[0]
            return None  # not found
        except Exception:
            return None  # error — caller can't tell the difference!
    ```

    **GOOD — explicit error handling, unambiguous return, parameterized query:**

    ```python
    class JobNotFoundError(Exception):
        pass

    def find_job(job_id: str) -> Job:
        "\""Raises JobNotFoundError if not found, propagates DB errors."\""
        result = db.query("SELECT * FROM jobs WHERE id = %s", (job_id,))
        if not result:
            raise JobNotFoundError(f"Job {job_id} not found")
        return Job(**result[0])
    ```

    **Heuristic:** "If a function returns `None`, make sure the caller can always distinguish
    'no result' from 'something went wrong.'"

    **Bonus:** The GOOD version also fixes SQL injection — parameterized queries, always.
    Raw f-strings in SQL are a critical security vulnerability.
    """)
    return


@app.cell
def smell4_magic_numbers(mo):
    mo.md("""
    ### Smell 4: Hardcoded Magic Numbers and Strings

    Raw numbers and strings scattered through logic are opaque. The reader has to guess what
    `7.5`, `3`, and `14` mean — and the next engineer has to update them in multiple places
    while hoping they found every occurrence.

    ---

    **BAD — numbers with no name or meaning:**

    ```python
    if score > 7.5 and len(skills_matched) >= 3:
        if days_since_posted < 14:
            send_to_slack("#job-alerts", format_job(job))
    ```

    **GOOD — named constants, self-documenting:**

    ```python
    MIN_SCORE_THRESHOLD = 7.5
    MIN_SKILLS_REQUIRED = 3
    MAX_POSTING_AGE_DAYS = 14
    SLACK_CHANNEL = "#job-alerts"

    if (score > MIN_SCORE_THRESHOLD
            and len(skills_matched) >= MIN_SKILLS_REQUIRED
            and days_since_posted < MAX_POSTING_AGE_DAYS):
        send_to_slack(SLACK_CHANNEL, format_job(job))
    ```

    **Heuristic:** "If a reviewer would ask 'why this number?', it needs a name."

    **Edge case:** `0` and `1` are usually obvious. `range(len(items))` doesn't need a constant.
    The test is whether the *value* is meaningful to the domain, not whether it happens to be a literal.

    The runnable demo below shows how named constants make the logic readable and changeable in one place.
    """)
    return


@app.cell
def demo_magic_numbers():
    MIN_SCORE_THRESHOLD = 7.5
    MIN_SKILLS_REQUIRED = 3
    MAX_POSTING_AGE_DAYS = 14
    SLACK_CHANNEL = "#job-alerts"

    test_jobs = [
        {"title": "ML Engineer",   "score": 8.2, "skills_matched": ["python", "pytorch", "sql"], "days_since_posted": 3},
        {"title": "Data Scientist","score": 6.1, "skills_matched": ["python", "r"],              "days_since_posted": 5},
        {"title": "Senior MLE",    "score": 9.0, "skills_matched": ["python", "mlflow", "aws"],  "days_since_posted": 20},
    ]

    qualifying = [
        j for j in test_jobs
        if (j["score"] > MIN_SCORE_THRESHOLD
            and len(j["skills_matched"]) >= MIN_SKILLS_REQUIRED
            and j["days_since_posted"] < MAX_POSTING_AGE_DAYS)
    ]

    for job in qualifying:
        print(f"[{SLACK_CHANNEL}] Qualifying: {job['title']} — score {job['score']}")

    print(f"\n{len(qualifying)}/{len(test_jobs)} jobs qualify "
          f"(score > {MIN_SCORE_THRESHOLD}, skills >= {MIN_SKILLS_REQUIRED}, "
          f"posted < {MAX_POSTING_AGE_DAYS}d ago)")
    return


@app.cell
def part2_header(mo):
    mo.md("""
    ## Part 2: Class & Module-Level Smells
    """)
    return


@app.cell
def smell5_god_class(mo):
    mo.md("""
    ### Smell 5: God Class

    A class that does everything: scraping, parsing, embedding, scoring, storing, notifying,
    exporting. It grows without bounds because everything "kinda belongs here." It becomes
    impossible to test, reuse, or deploy any component independently.

    ---

    **BAD — one class, ten responsibilities:**

    ```python
    class JobSearchEngine:
        def scrape_indeed(self): ...
        def scrape_linkedin(self): ...
        def parse_html(self, html): ...
        def compute_embeddings(self, text): ...
        def store_in_db(self, job): ...
        def score_job(self, job): ...
        def generate_cover_letter(self, job): ...
        def send_slack_notification(self, job): ...
        def export_to_csv(self): ...
        def generate_report(self): ...
    ```

    **GOOD — Single Responsibility Principle, thin pipeline coordinator:**

    ```python
    class IndeedScraper:
        def scrape(self, query: str) -> list[RawJob]: ...

    class JobEmbedder:
        def embed(self, text: str) -> np.ndarray: ...

    class JobScorer:
        def score(self, job: Job, resume: Resume) -> ScoredJob: ...

    class SlackNotifier:
        def notify(self, jobs: list[ScoredJob]): ...

    class JobSearchPipeline:
        "\""Orchestrates components — thin coordinator, no business logic."\""
        def __init__(self, scraper, embedder, scorer, notifier): ...

        def run(self, query: str):
            raw_jobs = self.scraper.scrape(query)
            scored = [self.scorer.score(j, self.resume) for j in raw_jobs]
            self.notifier.notify([j for j in scored if j.score > MIN_SCORE_THRESHOLD])
    ```

    **Heuristic:** "If you can't describe the class's purpose in one sentence without *and*, split it."

    **Connection to Canopy:** The scraper, scorer, and Slack notifier should be separate services
    with a pipeline coordinator — not one class that imports `requests`, `openai`, and `slack_sdk`.
    The pipeline class is allowed to *coordinate*; it should not contain *logic*.
    """)
    return


@app.cell
def smell6_tight_coupling(mo):
    mo.md("""
    ### Smell 6: Inappropriate Intimacy (Tight Coupling)

    When class A reaches into class B's private attributes or internal methods, changing B breaks A.
    The classes are coupled at the implementation level, not the interface level. The test:
    can you swap `IndeedScraper` for `LinkedInScraper` without touching `JobScorer`?

    ---

    **BAD — scorer reaches into scraper's private state:**

    ```python
    class JobScorer:
        def score(self, scraper: IndeedScraper) -> float:
            raw_html = scraper._cached_html                      # accessing private state!
            salary = scraper._parse_salary_from_html(raw_html)  # calling private method!
            return self._compute(salary, scraper.last_response.headers['date'])
    ```

    **GOOD — communicate through data, not object internals:**

    ```python
    class JobScorer:
        def score(self, job: Job) -> float:
            return self._compute(job.salary, job.posted_date)
    ```

    **Heuristic:** "If class A needs to know class B's internal structure to work, they're too
    coupled. Pass data, not objects."

    **Why the underscore matters:** In Python, `_name` is a convention for "private — don't touch
    this from outside the class." If you find yourself accessing `_something` from another class,
    that's a coupling smell — the data should be surfaced through a proper public interface.
    """)
    return


@app.cell
def smell7_primitive_obsession(mo):
    mo.md("""
    ### Smell 7: Primitive Obsession

    Passing 8 loose primitives (strings, ints, bools) instead of a domain object means no
    validation, no type safety, no IDE autocomplete, and functions with 10-argument signatures
    nobody can remember.

    ---

    **BAD — primitives everywhere, no validation, no structure:**

    ```python
    def process_job(title: str, company: str, salary_min: int, salary_max: int,
                    location: str, remote: bool, skills: list[str],
                    posted_date: str, source: str, url: str) -> dict:
        return {
            'title': title,
            'score': compute_score(title, skills),
            'salary_range': f"${salary_min}-${salary_max}",
        }
    ```

    **GOOD — Pydantic models encode meaning, enforce invariants, and serve as documentation:**

    ```python
    from pydantic import BaseModel, field_validator, HttpUrl
    from datetime import date

    class SalaryRange(BaseModel):
        min_usd: int
        max_usd: int

        @field_validator('max_usd')
        @classmethod
        def max_gte_min(cls, v, info):
            if v < info.data.get('min_usd', 0):
                raise ValueError('max_usd must be >= min_usd')
            return v

    class Job(BaseModel):
        title: str
        company: str
        salary: SalaryRange
        location: str
        remote: bool
        skills: list[str]
        posted_date: date
        source: str
        url: HttpUrl

    class ScoredJob(Job):
        score: float
        matched_skills: list[str]
    ```

    **Heuristic:** "If you're passing more than 3 related primitives together, they want to be an object."

    **Connection:** This is the structured outputs pattern from the LLM serving notebook — Pydantic
    models as contracts between components. The schema IS the documentation.
    The runnable demo below shows Pydantic catching invalid data at construction time.
    """)
    return


@app.cell
def demo_primitive_obsession():
    from pydantic import BaseModel, field_validator

    class SalaryRange(BaseModel):
        min_usd: int
        max_usd: int

        @field_validator('max_usd')
        @classmethod
        def max_gte_min(cls, v, info):
            if v < info.data.get('min_usd', 0):
                raise ValueError(
                    f'max_usd ({v}) must be >= min_usd ({info.data.get("min_usd")})'
                )
            return v

    class Job(BaseModel):
        title: str
        company: str
        salary: SalaryRange
        skills: list[str]
        remote: bool

    # Valid construction
    job = Job(
        title="ML Engineer",
        company="Acme",
        salary=SalaryRange(min_usd=120_000, max_usd=160_000),
        skills=["python", "pytorch", "sql"],
        remote=True,
    )
    print(f"Valid job: {job.title} at {job.company}")
    print(f"  Salary: ${job.salary.min_usd:,} – ${job.salary.max_usd:,}")
    print(f"  Skills: {job.skills}")

    # Pydantic catches the invalid salary range at construction — not buried in downstream logic
    try:
        bad = Job(
            title="Bad Job",
            company="Acme",
            salary=SalaryRange(min_usd=200_000, max_usd=100_000),
            skills=["python"],
            remote=False,
        )
    except Exception as e:
        print(f"\nCaught invalid salary at construction:\n  {e}")
    return


@app.cell
def part3_header(mo):
    mo.md("""
    ## Part 3: Architecture-Level Smells
    """)
    return


@app.cell
def smell8_config_in_code(mo):
    mo.md("""
    ### Smell 8: Configuration Hardcoded in Code

    API keys, URLs, thresholds, and environment-specific values baked into source. Two problems:
    (1) they change between dev/staging/prod and require code changes to deploy,
    (2) they leak to version control when the repo is pushed or shared.

    ---

    **BAD — hardcoded secrets and config:**

    ```python
    class JobScraper:
        def __init__(self):
            self.api_key = "sk-abc123..."             # leaked to git
            self.base_url = "https://api.indeed.com/v2"
            self.max_results = 50
            self.timeout = 30
            self.slack_webhook = "https://hooks.slack.com/..."
    ```

    **GOOD — config from environment, validated at startup:**

    ```python
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
        indeed_api_key: str                           # required — fails fast if missing
        indeed_base_url: str = "https://api.indeed.com/v2"
        max_results: int = 50
        request_timeout: int = 30
        slack_webhook_url: str

        class Config:
            env_file = ".env"

    settings = Settings()  # raises ValidationError at startup if required vars missing
    ```

    **Heuristic:** "Anything that changes between environments (dev/staging/prod) is configuration,
    not code."

    **Security angle:** API keys in source = leaked on GitHub when the repo goes public, or when
    a contributor's machine is compromised. Always use environment variables or a secrets manager.
    `.env` in `.gitignore`, always. `pydantic_settings` fails fast at startup rather than at the
    first API call 30 minutes into a run.
    """)
    return


@app.cell
def smell9_error_boundaries(mo):
    mo.md("""
    ### Smell 9: Missing Error Boundaries

    When multiple independent operations are chained without isolation, the first failure kills
    everything downstream. One flaky API shouldn't prevent three working ones from delivering
    results. Partial results are almost always better than no results.

    ---

    **BAD — one failure kills the entire pipeline:**

    ```python
    def daily_briefing():
        arxiv_papers = fetch_arxiv()    # if this raises, nothing below runs
        reddit_posts  = fetch_reddit()
        tweets        = fetch_twitter()
        summaries = summarize_all(arxiv_papers + reddit_posts + tweets)
        send_to_slack(summaries)
    ```

    **GOOD — each source is independent, failures are contained:**

    ```python
    def daily_briefing():
        content = []
        sources_failed = []

        for source_name, fetcher in [
            ("arXiv",   fetch_arxiv),
            ("Reddit",  fetch_reddit),
            ("Twitter", fetch_twitter),
        ]:
            try:
                items = fetcher()
                content.extend(items)
                logger.info(f"{source_name}: fetched {len(items)} items")
            except Exception as e:
                sources_failed.append(source_name)
                logger.error(f"{source_name} failed: {e}")

        if not content:
            send_alert("All sources failed", sources_failed)
            return

        summaries = summarize_all(content)
        send_to_slack(
            summaries,
            note=f"Sources unavailable: {sources_failed}" if sources_failed else None,
        )
    ```

    **Heuristic:** "If one component's failure shouldn't kill the system, wrap it in an error boundary."

    **Connection:** The daily briefing agent follows this exact pattern — each source independent,
    partial results sent rather than nothing. Twitter being down shouldn't stop the arXiv digest.
    """)
    return


@app.cell
def smell10_sync_bottleneck(mo):
    mo.md("""
    ### Smell 10: Synchronous Everything When Async Is Needed

    Sequential API calls when the calls are completely independent. Three 2-second requests take
    6 seconds in series — they could take 2 seconds in parallel. At scale, this determines whether
    a system meets its SLA or times out.

    ---

    **BAD — sequential I/O, 3× slower than necessary:**

    ```python
    def fetch_all_sources():
        arxiv   = requests.get("https://arxiv.org/api/...").json()     # 2s
        reddit  = requests.get("https://reddit.com/r/...").json()      # 2s
        twitter = requests.get("https://api.twitter.com/...").json()   # 2s
        return arxiv + reddit + twitter                                 # 6 seconds total
    ```

    **GOOD — concurrent fetching, all three in ~2s total:**

    ```python
    import asyncio
    import httpx

    async def fetch_all_sources():
        async with httpx.AsyncClient(timeout=10) as client:
            results = await asyncio.gather(
                client.get("https://arxiv.org/api/..."),
                client.get("https://reddit.com/r/..."),
                client.get("https://api.twitter.com/..."),
                return_exceptions=True,  # one failure does not cancel the others
            )

        content = []
        for source, result in zip(["arXiv", "Reddit", "Twitter"], results):
            if isinstance(result, Exception):
                logger.error(f"{source} failed: {result}")
            else:
                content.extend(result.json())
        return content
    ```

    **Heuristic:** "If operations are independent and I/O-bound, run them concurrently."

    **Note:** `return_exceptions=True` combines error boundaries with async — the standard
    production pattern. Without it, a single exception propagates and cancels all in-flight requests.
    This one flag merges Smell 9 and Smell 10 fixes into a single `gather` call.
    """)
    return


@app.cell
def smell11_leaky_abstractions(mo):
    mo.md("""
    ### Smell 11: Leaky Abstractions / Wrong Layer

    Business logic in route handlers, database queries in services, presentation logic in domain
    models. Each layer doing work that belongs to another. The system works, but adding any feature
    requires understanding and touching the entire stack.

    ---

    **BAD — route handler does everything: parsing, DB access, LLM call, persistence:**

    ```python
    @app.post("/api/jobs/score")
    async def score_job(request: Request):
        data = await request.json()
        job    = db.execute("SELECT * FROM jobs WHERE id = ?", data['job_id'])
        resume = db.execute("SELECT * FROM resumes WHERE user_id = ?", data['user_id'])
        prompt = f"Score this job: {job['title']} against resume: {resume['text']}"
        response = openai.chat.completions.create(model="gpt-4o", messages=[...])
        score = float(response.choices[0].message.content)
        db.execute("UPDATE jobs SET score = ? WHERE id = ?", score, data['job_id'])
        return {"score": score, "job": job['title']}
    ```

    **GOOD — strict layering, each layer has one concern:**

    ```python
    # routes/jobs.py — parse request, delegate, return response
    @app.post("/api/jobs/score")
    async def score_job(request: ScoreRequest, scorer: JobScorer = Depends()):
        result = await scorer.score(request.job_id, request.user_id)
        return ScoreResponse(score=result.score, job_title=result.job.title)

    # services/scorer.py — business logic only
    class JobScorer:
        def __init__(self, job_repo: JobRepository, llm: LLMClient):
            self.job_repo = job_repo
            self.llm = llm

        async def score(self, job_id: str, user_id: str) -> ScoreResult:
            job    = await self.job_repo.get(job_id)
            resume = await self.job_repo.get_resume(user_id)
            score  = await self.llm.score_match(job, resume)
            await self.job_repo.update_score(job_id, score)
            return ScoreResult(job=job, score=score)

    # repositories/jobs.py — data access only, no business logic
    class JobRepository:
        async def get(self, job_id: str) -> Job: ...
        async def update_score(self, job_id: str, score: float): ...
    ```

    **Heuristic:** "Routes parse requests and return responses. Services contain business logic.
    Repositories handle data access. Each layer only talks to the one directly below it."
    """)
    return


@app.cell
def part4_header(mo):
    mo.md("""
    ## Part 4: Edge Cases & Subtle Smells
    """)
    return


@app.cell
def smell12_silent_failures(mo):
    mo.md("""
    ### Smell 12: Silent Failures

    `except: pass` swallows every exception — including unexpected bugs in your own code and
    corrupted state. Errors become invisible. The system *appears* to work while silently
    producing wrong results. This is harder to debug than a crash.

    ---

    **BAD — errors disappear into the void:**

    ```python
    try:
        result = llm.score(job_description)
    except Exception:
        pass  # ¯\_(ツ)_/¯
    ```

    **GOOD — log, handle specifically, or propagate — never swallow:**

    ```python
    try:
        result = llm.score(job_description)
    except RateLimitError:
        logger.warning("LLM rate limited, retrying in 60s")
        await asyncio.sleep(60)
        result = llm.score(job_description)
    except LLMError as e:
        logger.error(f"LLM scoring failed: {e}", extra={"job_id": job.id})
        result = ScoreResult(score=None, error=str(e))
    ```

    **Heuristic:** "Bare `except: pass` is the single worst line of code you can write."

    **The exception hierarchy — in order of preference:**
    1. Handle specifically (known errors you can recover from)
    2. Log and propagate (unexpected errors — let the caller decide)
    3. Log and return a sentinel value (when partial failure is acceptable)
    4. **Never:** silently swallow
    """)
    return


@app.cell
def smell13_test_hostile(mo):
    mo.md("""
    ### Smell 13: Test-Hostile Code (Hard Dependencies)

    A class that creates its own HTTP clients, database connections, and LLM clients inside
    `__init__` or inline can't be tested without the real dependencies. Every test requires
    an API key, a running database, and internet access.

    ---

    **BAD — hard dependencies, requires the real internet to test:**

    ```python
    class JobScorer:
        def score(self, job_id: str) -> float:
            job = requests.get(f"https://api.jobs.com/{job_id}").json()
            llm_response = openai.chat.completions.create(...)
            db.execute("INSERT INTO scores ...")
            slack.post("New score: ...")
            return float(llm_response.choices[0].message.content)
    ```

    **GOOD — dependency injection, testable with in-memory fakes:**

    ```python
    class JobScorer:
        def __init__(self, job_client: JobClient, llm: LLMClient, db: Database):
            self.job_client = job_client
            self.llm = llm
            self.db = db

        async def score(self, job_id: str) -> float:
            job   = await self.job_client.get(job_id)
            score = await self.llm.score_match(job)
            await self.db.save_score(job_id, score)
            return score

    # In tests — no network, no DB, no API key:
    scorer = JobScorer(
        job_client=FakeJobClient({"123": mock_job}),
        llm=FakeLLM(fixed_score=8.5),
        db=InMemoryDB(),
    )
    result = await scorer.score("123")
    assert result == 8.5
    ```

    **Heuristic:** "If you can't test a class without the internet, a database, and an API key,
    it has too many hard dependencies."

    **The rule:** *Receive* dependencies, don't *create* them. Classes should accept what they need
    via constructor injection. The caller (or a DI framework) decides which implementation to use.
    """)
    return


@app.cell
def smell14_mutable_defaults(mo):
    mo.md("""
    ### Smell 14: Mutable Default Arguments (Python-Specific Trap)

    Default arguments are evaluated **once** when the function is *defined* — not on each call.
    A mutable default (list, dict, set) is shared across every call that uses the default.
    This is Python's most common "spot the bug" interview question.

    ---

    **BAD — default list is shared, accumulates across calls:**

    ```python
    def add_skill(skill: str, skills: list = []) -> list:
        skills.append(skill)
        return skills

    add_skill("python")   # → ["python"]             ✓ first call looks fine
    add_skill("sql")      # → ["python", "sql"]      ✗ BUG: default list persists!
    add_skill("pytorch")  # → ["python", "sql", "pytorch"]  keeps growing
    ```

    **GOOD — `None` sentinel, fresh list created inside on each call:**

    ```python
    def add_skill(skill: str, skills: list | None = None) -> list:
        if skills is None:
            skills = []
        skills.append(skill)
        return skills
    ```

    **Heuristic:** "Never use mutable defaults (list, dict, set) in Python function signatures."

    This comes up in interviews as a "spot the bug" question surprisingly often.
    The runnable demo below makes the bug observable — all three return values are the same object.
    """)
    return


@app.cell
def demo_mutable_defaults():
    def add_skill_buggy(skill: str, skills: list = []) -> list:
        skills.append(skill)
        return skills

    def add_skill_fixed(skill: str, skills: list | None = None) -> list:
        if skills is None:
            skills = []
        skills.append(skill)
        return skills

    print("=== BUGGY version (shared mutable default) ===")
    r1 = add_skill_buggy("python")
    r2 = add_skill_buggy("sql")
    r3 = add_skill_buggy("pytorch")
    print(f"Call 1 result: {r1}")
    print(f"Call 2 result: {r2}")
    print(f"Call 3 result: {r3}")
    print(f"All the same object (r1 is r2 is r3): {r1 is r2 and r2 is r3}")

    print("\n=== FIXED version (None sentinel) ===")
    f1 = add_skill_fixed("python")
    f2 = add_skill_fixed("sql")
    f3 = add_skill_fixed("pytorch")
    print(f"Call 1 result: {f1}")
    print(f"Call 2 result: {f2}")
    print(f"Call 3 result: {f3}")
    print(f"All different objects: {f1 is not f2 and f2 is not f3}")
    return


@app.cell
def smell15_stringly_typed(mo):
    mo.md("""
    ### Smell 15: Stringly-Typed Code

    Using raw strings for values that can only be one of N options — status, action, type, state.
    Typos are silent (the `if` branch just never matches), IDEs can't autocomplete, and you have
    to grep the entire codebase to find all the valid values.

    ---

    **BAD — magic strings, typo-prone, no IDE help:**

    ```python
    def process(job: dict, action: str):
        if action == "scroe":    # typo — silent failure, no match
            return score_job(job)
        elif action == "notify":
            return notify(job)
    ```

    **GOOD — Enum, typos caught at definition time, IDE autocompletes:**

    ```python
    from enum import Enum

    class Action(str, Enum):
        SCORE   = "score"
        NOTIFY  = "notify"
        ARCHIVE = "archive"

    def process(job: Job, action: Action):
        handlers = {
            Action.SCORE:   score_job,
            Action.NOTIFY:  notify,
            Action.ARCHIVE: archive,
        }
        return handlers[action](job)
    ```

    **Heuristic:** "If a string can only be one of N values, it should be an Enum."

    `str, Enum` (inheriting from both) means the enum values serialize/deserialize as plain strings —
    compatible with JSON APIs and Pydantic models. The runnable demo below shows the typo caught at
    construction time and the dispatch table replacing the if/elif chain.
    """)
    return


@app.cell
def demo_stringly_typed():
    from enum import Enum

    class Action(str, Enum):
        SCORE   = "score"
        NOTIFY  = "notify"
        ARCHIVE = "archive"

    print("Valid enum values:", [a.value for a in Action])
    print(f"Action.SCORE == 'score': {Action.SCORE == 'score'}")
    print(f"Serializes as plain string: {str(Action.SCORE)!r}")

    # Typo from the BAD example → raised immediately, not a silent wrong-branch bug
    try:
        bad = Action("scroe")
    except ValueError as e:
        print(f"\nCaught typo at construction: {e}")

    # Dispatch table replaces if/elif — adding a new action is one line in the Enum + one in the dict
    def score_job(job):  return f"Scored:   {job['title']}"
    def notify(job):     return f"Notified: {job['title']}"
    def archive(job):    return f"Archived: {job['title']}"

    handlers = {
        Action.SCORE:   score_job,
        Action.NOTIFY:  notify,
        Action.ARCHIVE: archive,
    }

    test_job = {"title": "ML Engineer @ Acme"}
    for action in Action:
        print(handlers[action](test_job))
    return


@app.cell
def cheat_sheet(mo):
    mo.md("""
    ## Clean Code Heuristics Cheat Sheet

    | # | Heuristic | Smell it fixes |
    |---|-----------|----------------|
    | 1 | Functions should do one thing | God functions |
    | 2 | Name things by what they DO, not what they ARE | Unclear variable names |
    | 3 | If you need a comment to explain WHAT, the code isn't clear enough | Over-commented obvious code |
    | 4 | Comments should explain WHY, not WHAT | Missing intent / context |
    | 5 | Don't repeat yourself — but don't over-abstract either | DRY gone wrong, premature abstraction |
    | 6 | Prefer explicit over clever | Unreadable one-liners |
    | 7 | Errors are data, not surprises — handle them explicitly | Silent failures, bare except |
    | 8 | Make illegal states unrepresentable | Primitive obsession, invalid combinations |
    | 9 | Dependencies flow inward (routes → services → repos) | Leaky abstractions |
    | 10 | Code for the reader, not the writer | Write-once spaghetti |

    ---

    ### The Rule of Three

    - **First time:** just write it.
    - **Second time:** notice the duplication, maybe copy-paste.
    - **Third time:** now refactor into a shared abstraction.

    > *"Premature abstraction is worse than duplication. Wait for the pattern to emerge."*

    The cost of the wrong abstraction is higher than the cost of a little duplication.
    Three similar lines are better than one leaky abstraction applied in three places.
    """)
    return


@app.cell
def architecture_summary(mo):
    mo.md("""
    ## Architecture Smells Summary

    | Smell | Symptom | Fix |
    |-------|---------|-----|
    | God class | One class with 10+ methods across unrelated concerns | Split by single responsibility |
    | Tight coupling | Changing A requires touching B's internals | Communicate through data / interfaces |
    | No error boundaries | One source failure cascades and kills everything | Per-component try/except, partial results |
    | Config in code | API keys, URLs, thresholds hardcoded in source | `pydantic_settings`, env vars, `.env` |
    | Wrong layer | Business logic in routes, DB calls scattered everywhere | Strict: route → service → repo |
    | Sync bottleneck | Sequential I/O when operations are independent | `asyncio.gather` with `return_exceptions=True` |
    | No dependency injection | Classes instantiate their own dependencies | Inject via constructor, test with fakes |
    | Stringly-typed | Magic strings for states, actions, types | `str, Enum` + Pydantic validators |
    """)
    return


@app.cell
def flashcards(mo):
    mo.md("""
    ## Flashcard Summary

    **Q: What's a code smell?**
    A: A surface indicator of a deeper design problem. The code works but is fragile, hard to test,
    or hard to extend. It's a symptom, not the bug itself.

    **Q: Name 3 function-level smells.**
    A: (1) Functions that do too much, (2) boolean parameters that change behavior, (3) returning
    `None` ambiguously for both "not found" and "error".

    **Q: Name 3 architecture smells.**
    A: (1) God class, (2) tight coupling / inappropriate intimacy, (3) no error boundaries.

    **Q: What's the Single Responsibility Principle?**
    A: A class or function should have one reason to change. If it does two things, split it.

    **Q: Why is dependency injection important?**
    A: It makes code testable. Without it, you need real APIs, databases, and API keys to run
    `pytest`. With it, you swap in fakes and test everything in memory.

    **Q: What's wrong with bare `except: pass`?**
    A: It swallows all errors silently — including bugs in your own code. The system appears to
    work while producing wrong results. Errors become undebuggable.

    **Q: Mutable default argument bug?**
    A: Default lists and dicts are created once at function definition and shared across all calls.
    Use `None` as the sentinel and create a fresh container inside the function body.

    **Q: Primitive obsession fix?**
    A: Group related primitives into Pydantic models or dataclasses. You get validation, type
    safety, IDE autocomplete, and self-documenting code at the same time.

    **Q: Rule of three?**
    A: Don't abstract until you've seen the pattern three times. First time: write it. Second time:
    note the duplication. Third time: refactor. Premature abstraction costs more than duplication.

    **Q: What's the layered architecture rule?**
    A: Routes parse requests and return responses. Services contain business logic. Repositories
    handle data access. Each layer only talks to the one directly below it.
    """)
    return


@app.cell
def interview_talking_points(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    **"How do you write clean code?"**

    > "I follow three core principles: single responsibility (functions and classes do one thing),
    > explicit error handling (no silent failures — every exception is logged or propagated), and
    > dependency injection (everything testable without hitting real APIs or databases).
    > I use Pydantic for data contracts and type hints throughout. But I don't over-engineer —
    > I follow the rule of three for abstractions and avoid optimizing for hypothetical requirements."

    ---

    **"What do you look for in code reviews?"**

    > "I scan for: unclear function boundaries (doing too much), missing error handling,
    > hardcoded config and secrets, tight coupling between modules, and test-hostile patterns —
    > classes that create their own dependencies. I also watch for Python-specific traps like
    > mutable defaults and bare `except` clauses. The smell I care about most is whether a change
    > to one part of the system would require touching five other files."

    ---

    **"How does this apply to your projects?"**

    > "In Canopy, I structured the codebase with separate services: `IndeedScraper`, `JobScorer`,
    > and `SlackNotifier`, coordinated by a `JobSearchPipeline` class that contains no business
    > logic — just orchestration. Each service receives its dependencies via constructor injection,
    > so I can test the scorer with a `FakeLLMClient` and an `InMemoryDB` without any network calls.
    > LLM responses go through Pydantic validation before entering the pipeline — invalid JSON from
    > the LLM raises at the boundary, not silently corrupts downstream state."

    ---

    **"What's your refactoring approach?"**

    > "I don't refactor for the sake of it. I refactor when a smell is actively blocking a feature
    > or causing bugs. My process: write a characterization test first to catch regressions, extract
    > the new abstraction, verify the test still passes. Small, safe steps. The rule of three applies
    > here too — I wait until I've seen the pattern three times before abstracting, because the wrong
    > abstraction costs more than a little duplication."
    """)
    return


if __name__ == "__main__":
    app.run()
