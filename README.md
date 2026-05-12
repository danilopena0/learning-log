# Learning Log

Daily interview-prep study log. Each notebook covers one topic with concept explanations,
visualizations, code, and interview talking points.

Notebooks are written in **[marimo](https://marimo.io/)** — run any `.py` file with
`marimo edit <file>` to get an interactive notebook in the browser.

---

## Structure

```
learning-log/
├── dsa/            # Data Structures & Algorithms
├── ml/             # Machine Learning Theory
├── system-design/  # System Design & SWE Ops
└── ai-eng/         # AI Engineering
```

---

## Notebooks

### DSA

| Notebook | Topics | Date |
|----------|--------|------|
| [arrays-hashmaps.py](dsa/arrays-hashmaps.py) | Arrays, Hash Maps, Two Sum, Valid Anagram | 2026-03-30 |
| [sliding-window-two-pointers.py](dsa/sliding-window-two-pointers.py) | Sliding Window, Two Pointers, Best Time to Buy/Sell Stock, Container With Most Water | 2026-04-02 |

### ML Theory

| Notebook | Topics | Date |
|----------|--------|------|
| [bias-variance-regularization.py](ml/bias-variance-regularization.py) | Bias-Variance Tradeoff, Overfitting, L1/L2/Elastic Net | 2026-03-31 |
| [regression-gradient-descent.py](ml/regression-gradient-descent.py) | Linear/Logistic Regression, Gradient Descent, Loss Functions | 2026-04-02 |
| [tree-models-boosting.py](ml/tree-models-boosting.py) | Decision Trees, Random Forest, XGBoost, Feature Importance | 2026-04-05 |
| [evaluation-metrics.py](ml/evaluation-metrics.py) | Precision/Recall, F1, AUC-ROC, Confusion Matrix | 2026-04-07 |
| [validation-leakage.py](ml/validation-leakage.py) | Train/Val/Test Splits, Cross-Validation, Data Leakage | 2026-04-08 |
| [model-interpretability.py](ml/model-interpretability.py) | SHAP, LIME, Feature Importance, Partial Dependence | 2026-04-10 |
| [feature-engineering.py](ml/feature-engineering.py) | Encoding, Scaling, Imputation, Feature Creation | 2026-04-12 |
| [dimensionality-reduction.py](ml/dimensionality-reduction.py) | PCA, t-SNE, UMAP, Curse of Dimensionality | 2026-04-14 |
| [recommender-systems.py](ml/recommender-systems.py) | Collaborative Filtering, Matrix Factorization, Content-Based | 2026-04-17 |
| [statistical-tests.py](ml/statistical-tests.py) | A/B Testing, Hypothesis Tests, p-values, Power Analysis | 2026-04-19 |
| [transformer-attention.py](ml/transformer-attention.py) | Attention Mechanism, Transformers, BERT, GPT | 2026-04-21 |

### System Design

| Notebook | Topics | Date |
|----------|--------|------|
| [rag-pipeline.py](system-design/rag-pipeline.py) | RAG Architecture, Chunking, Vector Retrieval, Reranking | 2026-04-15 |
| [fraud-detection.py](system-design/fraud-detection.py) | Fraud Detection Systems, Feature Engineering, Real-Time Scoring | 2026-04-18 |
| [mlops-cicd.py](system-design/mlops-cicd.py) | MLOps Maturity, Model Versioning, CI/CD for ML, A/B Testing, Canary Deploys | 2026-04-23 |
| [training-platform.py](system-design/training-platform.py) | Distributed Training, GPU Scheduling, Experiment Management | 2026-04-25 |
| [ml-template-recsys.py](system-design/ml-template-recsys.py) | ML System Design Template, Recommender Systems at Scale | 2026-04-28 |
| [code-architecture-smells.py](system-design/code-architecture-smells.py) | Code Smells, Refactoring Patterns, Architecture Anti-Patterns | 2026-05-02 |
| [ai-agent-briefing.py](system-design/ai-agent-briefing.py) | AI Agent Architecture, Tool Use, Memory, Orchestration | 2026-05-05 |
| [swe-ops.py](system-design/swe-ops.py) | Git Workflows, Testing Pyramid, CI/CD, Docker, Observability, IaC, Linux, API Design, Secrets | 2026-05-11 |

### AI Engineering

| Notebook | Topics | Date |
|----------|--------|------|
| [vector-databases.py](ai-eng/vector-databases.py) | HNSW, IVF, Distance Metrics, Hybrid Search, Quantization | 2026-04-20 |
| [prompt-engineering.py](ai-eng/prompt-engineering.py) | Few-Shot, Chain-of-Thought, System Prompts, Prompt Patterns | 2026-04-22 |
| [rag-advanced.py](ai-eng/rag-advanced.py) | HyDE, Query Rewriting, Reranking, Multi-Hop Retrieval | 2026-04-27 |
| [llm-serving-patterns.py](ai-eng/llm-serving-patterns.py) | Batching, KV Cache, Quantization, Speculative Decoding | 2026-04-29 |
| [llm-evaluation.py](ai-eng/llm-evaluation.py) | LLM-as-Judge, RAGAS, Benchmark Design, Eval Pipelines | 2026-05-01 |
| [agent-architectures.py](ai-eng/agent-architectures.py) | ReAct, Tool Use, Memory, Multi-Agent, Orchestration | 2026-05-04 |
| [agent-evals.py](ai-eng/agent-evals.py) | Agent Evaluation, Trajectory Evals, Task Success Metrics | 2026-05-07 |
| [fine-tuning.py](ai-eng/fine-tuning.py) | LoRA, QLoRA, PEFT, Instruction Tuning, RLHF | 2026-05-09 |

---

## Running a Notebook

```bash
# Install marimo once
pip install marimo

# Open any notebook interactively
marimo edit dsa/arrays-hashmaps.py
marimo edit ml/bias-variance-regularization.py
```

---

## Notebook Format

Each notebook follows a consistent 6-section structure:

1. **Header** — date, track, time, topics covered
2. **Concept Overview** — explain as if teaching a junior dev
3. **Visualizations** — charts and diagrams for the core ideas
4. **Code / Problem Walkthroughs** — implementations with complexity analysis
5. **Flashcard Summary** — Q&A table for rapid review
6. **Interview Talking Points** — scripted answers and follow-up handling
