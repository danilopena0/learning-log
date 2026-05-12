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
    # General Software Engineering Ops

    | Field  | Value |
    |--------|-------|
    | Date   | 2026-05-11 |
    | Track  | System Design |
    | Time   | 75 min |
    | Topics | Git Workflows · Testing Pyramid · CI/CD · Docker · Observability · IaC · Linux · API Design · Secrets |
    """)
    return


@app.cell
def why_swe_ops(mo):
    mo.md("""
    ## Why SWE Ops

    Writing code that works locally is the easy part. SWE Ops is everything that gets code
    reliably into production and keeps it running — reproducibly, safely, and observably.

    **The gap between "it works on my machine" and production:**
    - Code must be tested, reviewed, and deployed without breaking other things
    - Multiple engineers must collaborate on the same codebase without stepping on each other
    - Production failures must be detectable, diagnosable, and recoverable
    - Infrastructure must be reproducible across environments (dev, staging, prod)
    - Secrets must never be exposed; access must be audited and rotated

    > **"A senior engineer isn't just someone who writes good code — it's someone who ships
    > reliably, debugs with data, and leaves the system more observable than they found it."**
    """)
    return


@app.cell
def git_workflows_concept(mo):
    mo.md("""
    ## Git Workflows & Branching Strategies

    ### The Three Main Strategies

    | Strategy | How it works | Best for |
    |----------|-------------|---------|
    | **Trunk-Based Development** | All devs commit directly to `main` (or short-lived feature branches < 1 day). Feature flags gate incomplete work. | High-deployment-frequency teams; CI is fast; strong automated testing culture |
    | **GitHub Flow** | `main` is always deployable. Work on a named branch, open a PR, merge when approved. Deploy from `main`. | Most product teams; good balance of velocity and review |
    | **GitFlow** | `main` + `develop` + `feature/` + `release/` + `hotfix/` branches. Heavy process, explicit release cycles. | Libraries / versioned software with discrete releases |

    **The trend:** teams are moving from GitFlow → GitHub Flow → Trunk-Based as deployment
    automation matures. GitFlow was invented for a world with quarterly releases. Trunk-Based
    Development is optimized for continuous delivery.

    ---

    ### Trunk-Based vs GitHub Flow: The Core Tradeoff

    ```
    GitHub Flow:
    main ──────────────────────────────────────────► (always deployable)
              │                    │
              └── feature/auth ────┘  (< 2 days ideally)
                    PR → Review → Merge

    Trunk-Based:
    main ──────────────────────────────────────────► (commit directly)
         ↑       ↑       ↑       ↑
         dev1   dev2   dev1   dev3   (small, frequent commits)
         Feature flags hide incomplete work from users
    ```

    **GitHub Flow wins when:** teams are small-medium, review culture is strong,
    feature branches stay short-lived.

    **Trunk-Based wins when:** CI is fast (<5 min), teams are large, and you need to
    avoid the "merge hell" of long-lived branches diverging.
    """)
    return


@app.cell
def git_best_practices(mo):
    mo.md("""
    ### Git Best Practices

    **Commit message conventions — Conventional Commits:**
    ```
    <type>(<scope>): <short description>

    <optional body>

    <optional footer>
    ```

    | Type | When to use |
    |------|------------|
    | `feat` | New feature |
    | `fix` | Bug fix |
    | `refactor` | Code change that's not a feature or fix |
    | `test` | Adding or modifying tests |
    | `chore` | Build process, tooling, deps — no production code change |
    | `docs` | Documentation only |
    | `perf` | Performance improvement |
    | `ci` | CI configuration changes |

    Examples:
    ```
    feat(auth): add OAuth2 login with Google
    fix(api): return 404 instead of 500 when user not found
    refactor(db): extract connection pooling into separate module
    ```

    ---

    ### Rebase vs Merge — When to Use Each

    | Operation | What it does | When to use |
    |-----------|-------------|------------|
    | `git merge` | Creates a merge commit. Preserves full history including branch divergence. | Integrating feature branches — the explicit merge commit is valuable documentation |
    | `git rebase` | Replays commits on top of target branch. Linear history, no merge commit. | Cleaning up local WIP commits before a PR; keeping feature branch current with main |
    | `git squash merge` | Collapses all branch commits into one commit on main. | When branch history is messy (lots of "wip" commits) and you want clean main history |

    **The golden rule:** never rebase a branch others are working on. Rebase rewrites
    history — if someone else has pulled the branch, their history diverges from yours.
    Only rebase your own local commits or when collaborators agree.

    ---

    ### PR Workflow Checklist

    ```
    Before opening a PR:
    ☐ All tests pass locally
    ☐ Linting clean (ruff, eslint, etc.)
    ☐ Type checks pass (mypy, tsc)
    ☐ Self-reviewed the diff — no debug prints, commented-out code, TODO left in
    ☐ PR description: what changed, why, how to test it, screenshots (for UI)
    ☐ Linked to the ticket or issue

    Reviewer checklist:
    ☐ Does it solve the stated problem?
    ☐ Any edge cases or error handling missing?
    ☐ Are there test cases for the new behavior?
    ☐ Would this be hard to maintain in 6 months?
    ☐ Approve or request changes — not "LGTM" with unresolved concerns
    ```
    """)
    return


@app.cell
def testing_pyramid(mo):
    mo.md("""
    ## The Testing Pyramid

    ```
                     ┌─────────────────┐
                     │    E2E Tests     │  ← Few, slow, expensive, high-confidence
                     │  (Playwright,    │     Test real user flows end-to-end
                     │   Cypress)       │     Catch integration failures
                     ├─────────────────┤
                    /│  Integration      │\\
                   / │  Tests            │ \\
                  /  │  (API tests,      │  \\
                 /   │   DB tests)       │   \\
                ├────┤───────────────────┤────┤
               /     │   Unit Tests      │     \\
              /      │   (pytest,        │      \\
             /       │    jest, go test) │       \\
            ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
            Many · Fast · Cheap · Narrow scope
    ```

    **The ratio (rough guideline):** 70% unit · 20% integration · 10% E2E

    ---

    ### Unit Tests

    - Test a single function or class in isolation
    - Mock all external dependencies (DB, HTTP, filesystem)
    - Should be deterministic, fast (<1ms each), and cheap to write
    - **What makes a good unit test:** tests behavior (what), not implementation (how).
      Refactoring shouldn't break unit tests if the behavior is unchanged.

    **Anti-pattern:** testing implementation details. If your test breaks when you rename
    a private method or change an internal data structure, the test is too tightly coupled.

    ---

    ### Integration Tests

    - Test interactions between components: service → DB, service → cache, service → external API
    - Use real dependencies when feasible (test DB, real Redis locally via Docker)
    - Slower than unit tests but catch a different class of bugs: schema mismatches,
      transaction behavior, serialization issues, connection pooling bugs

    **My project example — RAG app:**
    > Integration test: send a query to the retrieval endpoint, verify the returned chunks
    > are non-empty and have the expected metadata shape. This catches ChromaDB schema
    > issues that unit tests (with mocked vector store) would miss entirely.

    ---

    ### E2E Tests

    - Test the full stack from user's perspective: browser → frontend → API → DB
    - Tools: Playwright (Python/JS), Cypress (JS-only), Selenium (older)
    - Expensive to write and maintain. Flaky if not careful (timing issues, test data pollution)
    - Best for: critical user journeys — login, checkout, core feature flows

    **Keep E2E tests minimal:** cover the 3–5 flows that would cause the most damage if broken.
    Don't write E2E tests for every feature — that's what unit + integration tests are for.

    ---

    ### Contract Testing

    - Verifies that two services agree on the shape of their API contract
    - **Consumer-Driven Contract Testing (CDCT)**: the consumer defines what it expects;
      the provider verifies it can satisfy those expectations
    - Tool: **Pact** (language-agnostic, widely used in microservices)
    - Catches API breaking changes before they reach integration or E2E tests — much faster feedback

    **When it matters:** when you own multiple services that talk to each other and different
    teams own different services. One team changing a response field breaks another team's
    service silently. Contract tests surface this in CI before deployment.
    """)
    return


@app.cell
def cicd_pipeline(mo):
    mo.md("""
    ## CI/CD Pipelines for Software

    ### CI — Continuous Integration

    **Goal:** catch bugs and integration issues as early as possible, on every commit.

    ```
    [Push / PR opened]
         │
         ├── [Lint + Format Check]
         │     ruff, eslint, gofmt — fast, catches style issues
         │     Fail fast: don't waste time running tests on unformatted code
         │
         ├── [Type Check]
         │     mypy, tsc, pyright — catches type errors statically
         │
         ├── [Unit Tests]
         │     Fastest tests first. If they fail, no point running integration tests.
         │     Must be < 2 minutes total to maintain fast feedback loops.
         │
         ├── [Integration Tests]
         │     Spin up test DB + dependencies via Docker Compose or testcontainers.
         │     Run against real infrastructure, not mocks.
         │
         ├── [Security Scan]
         │     Bandit (Python), Trivy (containers), Dependabot (dependency vulns)
         │     Block on high/critical severity findings.
         │
         ├── [Build / Package]
         │     Build the artifact (Docker image, wheel, binary).
         │     Run ONLY after tests pass — don't waste build time on broken code.
         │
         └── [E2E Smoke Tests] (optional in CI, often deferred to CD)
               Run critical-path tests against the built artifact.
    ```

    **The "fail fast" principle:** order pipeline stages from cheapest to most expensive.
    A lint failure shouldn't wait for integration tests to be detected — that wastes
    minutes per PR across hundreds of PRs per week.

    ---

    ### CD — Continuous Delivery vs Deployment

    | Term | Meaning |
    |------|---------|
    | **Continuous Delivery** | Every merge to main produces a deployable artifact. Deployment to production is a human decision (button click). |
    | **Continuous Deployment** | Every merge to main that passes CI is deployed to production automatically. No human gate. |

    Most teams do Continuous Delivery. Continuous Deployment requires very high test
    coverage and confidence, and is more common in mature DevOps organizations.

    ---

    ### Deployment Strategies

    | Strategy | How it works | Risk | Rollback speed |
    |----------|-------------|------|---------------|
    | **Recreate** | Stop old, start new. Downtime during swap. | High — users see downtime | Instant (restart old) |
    | **Rolling** | Replace instances one by one. No downtime but both versions run simultaneously during deploy. | Medium — old+new serve traffic together | Moderate — re-roll forward |
    | **Blue-Green** | Two identical environments (blue=live, green=standby). Deploy to green, test, then switch traffic. | Low — instant cutover with no version mixing | Instant — flip traffic back to blue |
    | **Canary** | Route small % to new version, expand gradually. Same as MLOps canary pattern. | Low | Instant — route 100% back to old |

    **Blue-Green is ideal when:**
    - You have resources to run two full environments
    - You need guaranteed instant rollback
    - Your deployments have migrations that might be risky

    **Canary is ideal when:**
    - You want to validate in production traffic before full rollout
    - Business metric validation during rollout is valuable
    - You want automated rollback triggers on error rate / latency

    ---

    ### Database Migrations in CI/CD

    Migrations are the hardest part of deploying web applications. The challenge:
    the database schema must be updated in sync with the application code.

    **Safe migration pattern (backward-compatible migrations):**
    ```
    Phase 1 (current deploy): Add new column as nullable. App still writes old schema.
    Phase 2 (next deploy): App writes both old and new columns.
    Phase 3 (after backfill): Drop old column. App only writes new column.
    ```

    **Never do:** add a NOT NULL column without a default in a single deploy.
    The migration will fail on a live database with millions of rows, or lock the table.

    **Tools:** Alembic (Python/SQLAlchemy), Flyway, Liquibase, Django migrations.

    **Best practice:** migrations run automatically at deploy time (not manually).
    Migration state is tracked in the DB. CI runs migration against a test DB as a check.
    """)
    return


@app.cell
def docker_containerization(mo):
    mo.md("""
    ## Docker & Containerization

    ### Why Containers

    - **Reproducibility**: "works on my machine" disappears. The container bundles everything:
      OS libraries, runtime, dependencies, app code. Same image runs in dev, CI, and prod.
    - **Isolation**: processes can't interfere with each other; each container has its own
      filesystem, network namespace, and process space.
    - **Portability**: any system with Docker installed runs the same container.

    ---

    ### Dockerfile Best Practices

    ```dockerfile
    # 1. Pin base image versions — avoid :latest
    FROM python:3.12-slim

    # 2. Set working directory early
    WORKDIR /app

    # 3. Copy dependency files FIRST, before source code
    #    Dependencies change less often than code — this maximizes layer cache hits
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # 4. Copy source code last
    COPY . .

    # 5. Use non-root user — security best practice
    RUN adduser --disabled-password --gecos '' appuser
    USER appuser

    # 6. Expose port explicitly (documentation, not enforcement)
    EXPOSE 8000

    # 7. CMD vs ENTRYPOINT:
    #    ENTRYPOINT = the command that always runs; CMD = default args
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    ```

    **The layer caching rule:** copy files that change infrequently (requirements, package.json)
    before files that change frequently (source code). Docker invalidates the cache at the
    first changed layer — everything below it rebuilds. This can mean the difference between
    a 5-second and a 5-minute build.

    ---

    ### Multi-Stage Builds

    ```dockerfile
    # Stage 1: Build
    FROM python:3.12 AS builder
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --user -r requirements.txt   # install to user's home, not system

    # Stage 2: Runtime (much smaller — no build tools, no pip cache)
    FROM python:3.12-slim AS runtime
    WORKDIR /app
    COPY --from=builder /root/.local /root/.local  # copy only installed packages
    COPY . .
    CMD ["python", "main.py"]
    ```

    **Why multi-stage:** build tools (gcc, make, pip) can be 500MB+. The final image
    only needs the runtime. Multi-stage builds produce images 5–10x smaller — faster
    to pull, less attack surface.

    ---

    ### Docker Compose for Local Dev

    ```yaml
    # docker-compose.yml
    services:
      app:
        build: .
        ports:
          - "8000:8000"
        environment:
          - DATABASE_URL=postgresql://user:pass@db:5432/mydb
          - REDIS_URL=redis://redis:6379
        depends_on:
          db:
            condition: service_healthy  # wait for health check, not just container start

      db:
        image: postgres:16
        environment:
          - POSTGRES_USER=user
          - POSTGRES_PASSWORD=pass
          - POSTGRES_DB=mydb
        healthcheck:
          test: ["CMD-SHELL", "pg_isready -U user -d mydb"]
          interval: 5s
          retries: 5

      redis:
        image: redis:7-alpine
    ```

    **Key distinction:** `depends_on: service_started` (container running) vs
    `service_healthy` (container passed health check). A Postgres container starts
    in ~1 second but isn't accepting connections for 3–5 seconds. Use `service_healthy`.

    ---

    ### Container Security Checklist

    | Practice | Why |
    |----------|-----|
    | Run as non-root user | If the process is compromised, attacker gets limited OS access |
    | Use slim/distroless base images | Less attack surface — fewer packages to exploit |
    | Never store secrets in the image | `docker history` reveals every layer; secrets in ENV vars are visible |
    | Scan images for vulnerabilities | Trivy, Snyk, Docker Scout — automate in CI |
    | Use read-only filesystem | `docker run --read-only` — app can't modify its own filesystem |
    | Set resource limits | `--memory`, `--cpus` — prevent one container from starving others |
    """)
    return


@app.cell
def observability_concept(mo):
    mo.md("""
    ## Observability: The Three Pillars

    **Monitoring asks:** is the system healthy right now?
    **Observability asks:** why is the system behaving the way it is?

    Observability requires three complementary data types:

    ```
    ┌──────────────────────────────────────────────────────────┐
    │                    OBSERVABILITY                         │
    │                                                          │
    │  LOGS              METRICS            TRACES             │
    │  ─────             ───────            ──────             │
    │  Timestamped       Numeric            Distributed        │
    │  events from       measurements       request paths      │
    │  your code         over time          across services    │
    │                                                          │
    │  "User 123 login   "p99 latency       "Request A hit     │
    │   failed at 14:32" = 450ms at 14:30"  svc1→svc2→db"     │
    │                                                          │
    │  Loki, ELK         Prometheus +       Jaeger, Tempo,     │
    │  CloudWatch Logs   Grafana, Datadog   Zipkin, OTEL       │
    └──────────────────────────────────────────────────────────┘
    ```

    ---

    ### Logs

    **Structured logging > plain text logging:**

    ```python
    # Bad — hard to query at scale
    logger.info(f"User {user_id} logged in from {ip}")

    # Good — machine-parseable, filterable, aggregatable
    logger.info("user.login", extra={
        "user_id": user_id,
        "ip": ip,
        "latency_ms": latency,
        "event": "user.login",
        "level": "info"
    })
    ```

    Structured logs emit JSON. Log aggregation platforms (Loki, Elasticsearch, CloudWatch)
    can filter and aggregate on any field. `level=error AND user_id=123` becomes a one-line query.

    **Log levels:**
    | Level | When to use |
    |-------|------------|
    | `DEBUG` | Verbose diagnostics for development — turned off in prod |
    | `INFO` | Normal operations — logins, requests, background job completions |
    | `WARNING` | Unexpected but recoverable state — retry succeeded, fallback used |
    | `ERROR` | Something failed — the request failed, the job failed, should alert on-call |
    | `CRITICAL` | System-level failures — DB unreachable, OOM — page immediately |

    **Don't log sensitive data:** passwords, tokens, PII, credit card numbers.
    Log `user_id`, not `email`. Log `payment_id`, not card number. Check your log
    aggregation platform's data retention policies too.

    ---

    ### Metrics

    **The four golden signals** (Google SRE Book):

    | Signal | What it measures | Example |
    |--------|-----------------|---------|
    | **Latency** | How long requests take. Distinguish successful vs error latency — errors can be fast and misleading. | p50, p95, p99 response time |
    | **Traffic** | How much demand the system is receiving. | Requests/sec, messages/sec |
    | **Errors** | Rate of failed requests. | 5xx rate, exception rate |
    | **Saturation** | How "full" the system is — the resource closest to being exhausted. | CPU %, memory %, queue depth, disk I/O wait |

    **Percentiles vs averages:**
    > Never alert on average latency — it masks tail latency problems.
    > p99 = 1 in 100 requests is slower than this threshold.
    > For user-facing services, alert on p95 or p99, not mean.
    > A 500ms mean with a 10s p99 means 1% of your users wait 10 seconds.

    **SLOs, SLAs, SLIs:**
    - **SLI** (Service Level Indicator): a metric you measure. "p99 latency"
    - **SLO** (Service Level Objective): a target for an SLI. "p99 latency < 500ms, 99.9% of time"
    - **SLA** (Service Level Agreement): a contractual SLO with financial consequences. "If we miss SLO, customer gets credits"
    - **Error budget**: 100% - SLO target. If SLO is 99.9% availability, error budget = 0.1% = 43.8 min/month of allowed downtime.

    ---

    ### Distributed Tracing

    **The problem it solves:** a slow request in a microservices system — which service
    is the bottleneck? Logs tell you what happened in each service. Traces tell you the
    full path of a request across ALL services with timing for each hop.

    ```
    Request → API Gateway → Auth Service → Product Service → DB
    Trace shows:
    │ API Gateway       5ms  │
    │   Auth Service   45ms  │  ← bottleneck
    │   Product Service 8ms  │
    │     DB query      3ms  │
    Total: 61ms

    Without tracing: "API is slow." With tracing: "Auth service's token validation is slow."
    ```

    **OpenTelemetry (OTel):** the open standard for instrumentation. Vendor-neutral.
    Instrument your code once; export to Jaeger, Tempo, Datadog, Honeycomb, or any OTel
    backend. Avoids vendor lock-in on instrumentation.

    ---

    ### Alerting Best Practices

    **Alert on symptoms, not causes:**
    - Bad: alert when CPU > 80% (cause — may not affect users)
    - Good: alert when p99 latency > 1s for 5 minutes (symptom — users are experiencing it)

    **Every alert should be actionable.** If receiving an alert doesn't change what you do,
    it's not an alert — it's noise. Noisy alerts desensitize on-call engineers and lead to
    alert fatigue (ignoring pages).

    **Runbook-driven alerts:** every alert links to a runbook (step-by-step diagnosis and
    remediation guide). New engineers can handle on-call without tribal knowledge.
    """)
    return


@app.cell
def infrastructure_as_code(mo):
    mo.md("""
    ## Infrastructure as Code (IaC)

    ### Why IaC

    - **Reproducibility**: `terraform apply` creates identical infrastructure in dev, staging, and prod.
    - **Version control**: infrastructure changes are tracked in git — who changed what, when, why.
    - **Audit trail**: every change is a commit with a diff. Required for compliance.
    - **Disaster recovery**: if a region goes down, you can provision everything in a new region
      in minutes, not days.

    > **"ClickOps" (configuring infra via the AWS console) creates snowflake servers —
    > environments that are unique, undocumented, and impossible to reproduce.
    > IaC is the cure."**

    ---

    ### Terraform Concepts

    ```hcl
    # main.tf — provision an EC2 instance
    provider "aws" {
      region = var.aws_region
    }

    resource "aws_instance" "web" {
      ami           = "ami-0c55b159cbfafe1f0"
      instance_type = "t3.micro"

      tags = {
        Name        = "web-server"
        Environment = var.environment
      }
    }

    output "public_ip" {
      value = aws_instance.web.public_ip
    }
    ```

    **Core workflow:**
    ```
    terraform init     # download providers and modules
    terraform plan     # show what will change — always read this before apply
    terraform apply    # make the changes (prompts for confirmation)
    terraform destroy  # tear everything down
    ```

    **State:** Terraform tracks what it created in a state file (`terraform.tfstate`).
    The state is the source of truth for what exists. In teams, state must be stored remotely
    (S3 + DynamoDB for locking) — not locally. Never commit state files to git (they contain secrets).

    ---

    ### IaC Principles

    | Principle | What it means |
    |-----------|--------------|
    | **Idempotency** | Running the same IaC multiple times produces the same result. Applying twice shouldn't create two EC2 instances. |
    | **Immutable infrastructure** | Don't modify servers in place — replace them. Update the AMI, apply, Terraform destroys old and creates new. |
    | **Least privilege** | IAM roles and policies grant only the permissions the service actually needs. Not `*:*` on all resources. |
    | **DRY via modules** | Reusable Terraform modules for common patterns (VPC, ECS cluster, RDS). Don't copy-paste infrastructure blocks. |

    **Drift detection:** the real infrastructure can drift from the IaC definition (someone
    clicked in the console, or a cloud event changed something). `terraform plan` will show
    drift — unexpected changes between state and reality. Treat drift as a bug and fix it.

    ---

    ### IaC Tools Landscape

    | Tool | Scope | Language | Notes |
    |------|-------|----------|-------|
    | **Terraform** | Multi-cloud infra provisioning | HCL | Industry standard. Declarative. Huge ecosystem. |
    | **Pulumi** | Same as Terraform | Python, TypeScript, Go | "IaC in a real programming language" — conditional logic is easier |
    | **AWS CDK** | AWS-only, higher level | Python, TypeScript | Great for AWS-heavy orgs; generates CloudFormation |
    | **Ansible** | Configuration management | YAML | Manage what's already running: install packages, configure services |
    | **Helm** | Kubernetes app packaging | YAML templates | Package and deploy Kubernetes applications |

    Terraform = provision infrastructure. Ansible = configure what's on it. Kubernetes + Helm = deploy apps into the cluster.
    """)
    return


@app.cell
def linux_fundamentals(mo):
    mo.md("""
    ## Linux Fundamentals for SWEs

    ### Process Management

    ```bash
    # View running processes
    ps aux | grep python          # find python processes
    top / htop                    # live resource usage

    # Process signals
    kill -15 <pid>                # SIGTERM: graceful shutdown (ask nicely)
    kill -9 <pid>                 # SIGKILL: force terminate (no cleanup)

    # Background processes
    ./server &                    # run in background
    nohup ./server > out.log &   # run after logout; output to file

    # Useful: what process is using port 8000?
    lsof -i :8000
    ss -tulpn | grep 8000
    ```

    **SIGTERM vs SIGKILL:** always try SIGTERM first. It allows the process to clean up
    (flush buffers, close DB connections, finish in-flight requests). SIGKILL kills the
    process immediately — data loss and connection leaks are possible.

    ---

    ### File System & Permissions

    ```bash
    # Permissions: rwxrwxrwx = owner | group | others
    ls -la file.py
    # -rw-r--r-- 1 ubuntu ubuntu 4096 Jan 1 00:00 file.py
    #  ↑↑↑ ↑↑↑ ↑↑↑
    #  own grp oth

    chmod 755 script.sh           # rwxr-xr-x (owner can execute, group/others can read+execute)
    chmod +x script.sh            # add execute permission for everyone
    chown ubuntu:ubuntu file.py   # change owner and group

    # Find files
    find . -name "*.log" -mtime +7  # .log files older than 7 days
    find . -type f -size +100M      # files larger than 100MB
    ```

    | Permission | Octal | Meaning |
    |-----------|-------|---------|
    | `rwx` | 7 | Read + Write + Execute |
    | `rw-` | 6 | Read + Write |
    | `r-x` | 5 | Read + Execute |
    | `r--` | 4 | Read only |

    ---

    ### Networking

    ```bash
    # DNS and connectivity
    curl -I https://api.example.com     # HTTP headers only
    curl -v https://api.example.com     # verbose: see request + response headers
    dig api.example.com                  # DNS resolution trace
    nslookup api.example.com             # simpler DNS lookup

    # Network connections
    ss -s                                # socket statistics summary
    netstat -an | grep ESTABLISHED       # active connections

    # Ports and firewalls
    telnet host port                     # test TCP connectivity
    nc -zv host port                     # netcat: test if port is open (faster)
    ```

    **Diagnosing "connection refused" vs "connection timed out":**
    - **Connection refused**: the host is reachable, but nothing is listening on that port.
      Check if the service is running and bound to the right interface.
    - **Connection timed out**: packets aren't reaching the destination.
      Check firewalls, security groups, routing.

    ---

    ### Useful Shell Patterns

    ```bash
    # Searching and filtering
    grep -rn "TODO" ./src              # recursive grep with line numbers
    grep -v "debug" app.log            # exclude lines matching pattern
    grep "ERROR" app.log | tail -100   # last 100 error log lines

    # Log inspection
    tail -f app.log                    # follow a log file in real-time
    journalctl -u myservice -f         # follow systemd service logs
    journalctl -u myservice --since "1 hour ago"

    # System resources
    df -h                              # disk usage (human-readable)
    du -sh /var/log/*                  # size of each item in /var/log
    free -m                            # memory usage in MB
    uptime                             # load average (1, 5, 15 min)

    # Text processing
    sort file.txt | uniq -c | sort -rn  # frequency count of unique lines
    awk '{print $2}' file.txt           # print second column
    cut -d',' -f1,3 data.csv            # CSV columns 1 and 3
    ```

    **Load average interpretation:**
    > `uptime` shows 3 numbers: load average over 1, 5, 15 minutes.
    > On a 4-core machine, a load of 4.0 means all cores fully utilized.
    > A load > number-of-cores means processes are waiting for CPU.
    """)
    return


@app.cell
def api_design(mo):
    mo.md("""
    ## API Design Best Practices

    ### REST Resource Naming

    ```
    ✅ Good — resources are nouns, HTTP verbs express actions
    GET    /users                    list all users
    GET    /users/{id}               get one user
    POST   /users                    create user
    PUT    /users/{id}               replace user (full update)
    PATCH  /users/{id}               partial update
    DELETE /users/{id}               delete user

    GET    /users/{id}/orders        get orders for a user (nested resource)
    POST   /users/{id}/orders        create order for a user

    ❌ Bad — verbs in the URL (RPC-style, not REST)
    POST   /createUser
    GET    /getUserById?id=123
    POST   /deleteUser
    ```

    ---

    ### HTTP Status Codes to Know

    | Code | Name | When to use |
    |------|------|-------------|
    | 200 | OK | Successful GET, PUT, PATCH |
    | 201 | Created | Successful POST that created a resource. Include `Location` header pointing to new resource. |
    | 204 | No Content | Successful DELETE (nothing to return) |
    | 400 | Bad Request | Client sent invalid input. Return validation error details in the body. |
    | 401 | Unauthorized | Authentication required. Client needs to provide credentials. |
    | 403 | Forbidden | Authenticated but not authorized. You know who they are, but they can't do this. |
    | 404 | Not Found | Resource doesn't exist. Also use when you don't want to reveal whether a resource exists (security). |
    | 409 | Conflict | Request conflicts with current state (e.g., duplicate email on registration). |
    | 422 | Unprocessable Entity | Semantically invalid input (correct format, but fails business rules). |
    | 429 | Too Many Requests | Rate limit exceeded. Include `Retry-After` header. |
    | 500 | Internal Server Error | Unexpected server error. Don't leak stack traces to clients. |

    ---

    ### API Versioning Strategies

    | Strategy | Example | Pros | Cons |
    |----------|---------|------|------|
    | **URL path** | `/v1/users`, `/v2/users` | Most explicit; easy to route in nginx/API gateway | Feels ugly; version in URL is "impure REST" |
    | **Header** | `Accept: application/vnd.myapi.v2+json` | URL stays clean | Less discoverable, harder to test in browser |
    | **Query param** | `/users?version=2` | Easy to test | Easy to ignore; accidental caching |

    > **URL path versioning is the pragmatic choice.** Despite theoretical objections, it's
    > the most widely used, easiest to understand, easiest to route, and easiest to document.

    **When to version:** when a breaking change is unavoidable — removing or renaming a field,
    changing types, altering required fields. Never break existing clients without a migration path.

    ---

    ### Pagination

    **Offset-based pagination:**
    ```
    GET /users?limit=20&offset=0   # page 1
    GET /users?limit=20&offset=20  # page 2
    ```
    Simple. But: slow on large offsets (DB scans all preceding rows),
    and results are inconsistent if records are added/deleted between pages.

    **Cursor-based pagination (better for large datasets):**
    ```
    GET /users?limit=20
    Response: { "data": [...], "next_cursor": "eyJpZCI6MTAwfQ==" }
    GET /users?limit=20&cursor=eyJpZCI6MTAwfQ==  # next page
    ```
    Efficient: uses an index on the cursor field. Consistent: new records don't shift pages.
    Downside: can't jump to page 5 directly.

    **Use cursor pagination for:** large datasets, real-time feeds, anything with frequent
    inserts/deletes. Use offset pagination for: small datasets, admin UIs where random-access
    page jumps are needed.

    ---

    ### Idempotency

    An operation is **idempotent** if calling it multiple times has the same effect as calling it once.

    | Method | Idempotent? | Why |
    |--------|------------|-----|
    | GET | Yes | Read-only, no side effects |
    | DELETE | Yes | Deleting an already-deleted resource returns 404, not a second deletion |
    | PUT | Yes | Replacing a resource with the same data has the same result |
    | POST | No | Calling POST /users twice creates two users |
    | PATCH | Usually no | Depends on the operation (increment vs set) |

    **Idempotency keys for non-idempotent operations:**
    Client sends a unique `Idempotency-Key` header with POST requests. Server stores the key
    and result. If the same key is received again (network retry), return the stored result
    instead of processing twice. Critical for payment APIs — you never want to charge twice
    because of a network hiccup.
    """)
    return


@app.cell
def secrets_management(mo):
    mo.md("""
    ## Secrets Management

    ### The Problem

    Secrets (API keys, DB passwords, tokens, credentials) need to be available to your
    app at runtime but must never be:
    - Committed to git (even in private repos — git history is forever)
    - Baked into Docker images
    - Logged (even accidentally)
    - Visible to unauthorized people

    ---

    ### What NOT to Do

    ```python
    # 🚫 Hardcoded in source code
    api_key = "sk-abc123..."

    # 🚫 In a .env file committed to git
    # .env: DATABASE_URL=postgresql://user:pass@host/db

    # 🚫 In a Dockerfile ENV instruction (visible in docker history)
    # ENV DATABASE_URL=postgresql://user:pass@host/db

    # 🚫 Logged accidentally
    logger.info(f"Connecting to DB: {database_url}")
    ```

    ---

    ### Secrets Hierarchy: Dev → Prod

    | Environment | Approach |
    |-------------|----------|
    | **Local dev** | `.env` file (gitignored). `.env.example` with dummy values is committed as documentation. |
    | **CI/CD** | Secrets stored in the CI platform's secret store (GitHub Actions Secrets, GitLab CI Variables). Injected as env vars at runtime. |
    | **Production** | Dedicated secrets manager: AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager. App fetches secrets at startup. Secrets rotated automatically. Access audited. |

    ---

    ### HashiCorp Vault Concepts

    - **Secrets engines**: plugins that store or generate secrets. `kv` for static secrets,
      `database` for dynamic DB credentials, `aws` for temporary IAM credentials.
    - **Dynamic secrets**: Vault generates a DB credential on demand with a TTL. The credential
      expires after use. No long-lived passwords. Compromise radius is one short-lived credential.
    - **Policies**: define who can read/write which paths. Least-privilege access.
    - **Audit log**: every secret access is logged. Required for compliance.

    **Dynamic credentials pattern:**
    ```
    App starts → requests DB credentials from Vault
    Vault creates a Postgres user with 1-hour TTL
    App uses those credentials to connect
    TTL expires → Vault revokes the user → credentials useless
    ```

    If an attacker steals the credentials, they expire in ≤1 hour. Compare to a static
    password that never changes: an attacker has it forever.

    ---

    ### Secret Rotation

    **Why rotate:** credentials stolen in a breach become useless after rotation.
    Rotation limits exposure windows.

    **Rotation checklist:**
    - API keys: rotate on a schedule (quarterly) or immediately on suspected compromise
    - DB passwords: automate with Vault's database secrets engine
    - JWT signing keys: rotate with key versioning — validate old tokens during rotation window
    - SSH keys: rotate on personnel changes (someone leaves the team)

    > **"The question isn't if a secret will be compromised — it's when. Rotation limits the
    > damage window. Audit logs tell you what was accessed. Least-privilege limits the blast radius."**
    """)
    return


@app.cell
def full_architecture_diagram(mo):
    mo.md("""
    ## Full SWE Ops Architecture

    ```
    ┌────────────────────────────────────────────────────────────────────┐
    │                          DEVELOPMENT                               │
    │                                                                    │
    │  [Feature Branch]                                                  │
    │  Conventional commits · Short-lived branches · PR with description │
    └──────────────────────────────┬─────────────────────────────────────┘
                                   │ push
    ┌──────────────────────────────▼─────────────────────────────────────┐
    │                          CI PIPELINE                               │
    │                                                                    │
    │  Lint → Type Check → Unit Tests → Integration Tests               │
    │  Security Scan → Build Docker Image → E2E Smoke Tests             │
    │                                                                    │
    │  All gates green? → Artifact pushed to registry                   │
    └──────────────────────────────┬─────────────────────────────────────┘
                                   │ merge to main
    ┌──────────────────────────────▼─────────────────────────────────────┐
    │                     CD PIPELINE / DEPLOYMENT                       │
    │                                                                    │
    │  DB migrations (backward-compatible) → Deploy (Blue-Green/Canary) │
    │  → Smoke tests → Expand traffic                                   │
    │  Automated rollback if error rate / latency spikes                │
    │  IaC in Terraform — infrastructure is versioned alongside code    │
    └──────────────────────────────┬─────────────────────────────────────┘
                                   │ serving
    ┌──────────────────────────────▼─────────────────────────────────────┐
    │                        PRODUCTION                                  │
    │                                                                    │
    │  Secrets from Vault (dynamic credentials, short TTL)               │
    │  Containers: non-root, read-only FS, resource limits               │
    │                                                                    │
    │  OBSERVABILITY:                                                     │
    │  Structured logs → Loki       p50/p99 latency → Grafana           │
    │  Traces → Jaeger              Error rate / saturation → Alerting   │
    │                                                                    │
    │  Four Golden Signals: Latency · Traffic · Errors · Saturation     │
    └────────────────────────────────────────────────────────────────────┘
    ```
    """)
    return


@app.cell
def cheat_sheet(mo):
    mo.md("""
    ## Cheat Sheet

    ### Git
    | Command | What it does |
    |---------|-------------|
    | `git rebase main` | Replay my branch commits on top of latest main |
    | `git rebase -i HEAD~3` | Interactive rebase: squash/reorder last 3 commits |
    | `git stash` / `git stash pop` | Temporarily shelve / restore uncommitted changes |
    | `git bisect start` | Binary search commits to find which one introduced a bug |
    | `git log --oneline --graph` | Visual branch history |
    | `git blame file.py` | See who last changed each line |

    ### Linux
    | Command | What it does |
    |---------|-------------|
    | `lsof -i :8000` | What process is using port 8000 |
    | `kill -15 pid` | Graceful shutdown (SIGTERM) |
    | `tail -f app.log` | Follow log file in real-time |
    | `df -h` | Disk usage by mount point |
    | `du -sh dir/` | Size of directory |
    | `ss -tulpn` | All listening ports with process names |
    | `grep -rn "term" ./src` | Recursive grep with line numbers |

    ### Docker
    | Command | What it does |
    |---------|-------------|
    | `docker build -t name:tag .` | Build image |
    | `docker run -p 8080:8000 name` | Run with port mapping |
    | `docker exec -it container bash` | Shell into running container |
    | `docker logs -f container` | Follow container logs |
    | `docker system prune` | Remove unused images/containers/volumes |
    | `docker compose up -d` | Start all services detached |
    | `docker compose down -v` | Stop and remove including volumes |

    ### HTTP Status Quick Reference
    | Range | Meaning |
    |-------|---------|
    | 2xx | Success |
    | 3xx | Redirect |
    | 4xx | Client error |
    | 5xx | Server error |
    | 400 | Bad request | 401 | Unauthenticated | 403 | Unauthorized | 404 | Not found |
    | 409 | Conflict | 422 | Validation error | 429 | Rate limited |
    """)
    return


@app.cell
def interview_talking_points(mo):
    mo.md("""
    ## Interview Talking Points

    ---

    ### "How do you approach deploying to production safely?"

    > Use a staged rollout: canary to 1%, monitor for 10–15 minutes (error rate, latency, p99),
    > expand to 25%, then 50%, then 100%. Automated rollback triggers fire if error rate or
    > latency spikes beyond baseline — rollback is a config change, not a redeployment.
    > DB migrations run first and are backward-compatible — I never deploy a migration and
    > app code together if the old code can't handle the new schema.

    ---

    ### "What does observability mean to you?"

    > Observability is the ability to understand why a system is behaving the way it is,
    > not just whether it's up. That requires three things: structured logs with consistent
    > fields so I can filter by `user_id`, `request_id`, `level`; metrics on the four golden
    > signals — latency (p99, not mean), traffic, error rate, saturation; and distributed
    > traces so that when a request is slow, I can see which hop in the call chain is the
    > bottleneck. Monitoring tells me something is wrong. Observability tells me why.

    ---

    ### "How do you handle secrets in production?"

    > I never hardcode secrets or commit them to git — even in private repos.
    > In local dev: gitignored `.env` file. In CI: the platform's secret store injected
    > as environment variables. In production: a secrets manager like Vault or AWS Secrets
    > Manager. Ideally with dynamic credentials: Vault generates a DB credential with a
    > short TTL on demand, so a stolen credential expires quickly. Every secret access
    > is audited. Rotation is automated where possible.

    ---

    ### "What's your testing strategy?"

    > Testing pyramid: heavy on unit tests (fast, deterministic, mock externals),
    > moderate integration tests against real infrastructure using testcontainers or
    > a test DB (catch schema mismatches and transaction bugs unit tests miss),
    > minimal E2E tests covering only the 3–5 critical user paths.
    > All run in CI on every PR. Unit tests must complete in < 2 minutes — if they're
    > slow, engineers skip them. Fast feedback is the whole point.

    ---

    ### "How do you keep infrastructure consistent across environments?"

    > Terraform. All infrastructure is defined as code, versioned in git, and applied
    > through CI. Dev, staging, and prod are separate Terraform workspaces — same modules,
    > different variable values (instance sizes, replica counts). State lives in S3 with
    > DynamoDB locking. Nobody touches the AWS console to make permanent changes;
    > if they do, `terraform plan` will show drift and we fix it.

    ---

    ### One-Sentence Definitions

    - **Trunk-based development**: all engineers commit to main (or branches < 1 day); feature flags gate incomplete work
    - **Canary deploy**: route small % of real traffic to new version, monitor, expand gradually
    - **Blue-green deploy**: two identical environments; flip traffic from old to new atomically
    - **Idempotency**: calling an operation multiple times produces the same result as calling it once
    - **p99 latency**: 99% of requests complete faster than this threshold
    - **Error budget**: the allowable downtime implied by an SLO target (100% - SLO%)
    - **Cursor-based pagination**: use an opaque cursor for stable, index-efficient pagination at scale
    - **Dynamic secrets**: short-lived credentials generated on demand; expire automatically
    - **IaC drift**: real infrastructure diverged from the Terraform definition
    - **Contract test**: verifies that two services agree on the API contract between them
    """)
    return


if __name__ == "__main__":
    app.run()
