# Learning Log

Daily interview-prep study log. Each notebook covers one topic with concept explanations,
visualizations, code, and interview talking points.

Notebooks are written in **[marimo](https://marimo.io/)** — run any `.py` file with
`marimo edit <file>` to get an interactive notebook in the browser.

---

## Structure

```
learning-log/
├── dsa/          # Data Structures & Algorithms
└── ml/           # Machine Learning Theory
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
