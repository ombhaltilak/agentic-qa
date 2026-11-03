# Testing a real RAG: "Attention Is All You Need" assistant

`rag.py` is the same pipeline as `research_paper_rag.ipynb` (PyPDF → 1000/200 chunks → FAISS k=3 →
gpt-4o-mini, three-sentence answers), packaged so `agentic-qa` can test it. The FAISS index is cached in
`.cache/`, so later runs don't re-embed the paper.

| File | What it is |
|---|---|
| `golden.jsonl` | 20 questions: 16 with reference answers checked against the PDF (with page citations), 4 the paper cannot answer |
| `policy.yaml` | The bot's rules: no fabricated references, admit unknowns, stay on topic, ≤ 3 sentences (soft) |

Run from the project root (needs `OPENAI_API_KEY`; optional `OPENCODE_API_KEY` for the free Jev judge).

## 1. Baseline

```bash
agentic-qa run --target examples.research_paper_rag.rag:rag_chain \
  --name "Attention paper RAG" --domain "ML research papers" \
  --description "$(python -c 'from examples.research_paper_rag.rag import DESCRIPTION; print(DESCRIPTION)')" \
  --dataset examples/research_paper_rag/golden.jsonl \
  --policy  examples/research_paper_rag/policy.yaml \
  --iterations 3 --tests 8 --output baseline.json
```

You get pass rate, **hallucination rate** on the golden set, per-rule policy violations, per-category
pass rates, p50/p95 latency and the QA run's cost. Reports go to `reports/`.

## 2. Change something, then compare

Try one change at a time, for example `RAG_TOP_K=5` or a different `chunk_size` in `rag.py`
(delete `.cache/` after changing chunking), or edit `SYSTEM_PROMPT`:

```bash
RAG_TOP_K=5 agentic-qa run --target examples.research_paper_rag.rag:rag_chain \
  --replay baseline.json --policy examples/research_paper_rag/policy.yaml --output top5.json
agentic-qa diff baseline.json top5.json --markdown diff.md
```

`--replay` re-asks exactly the same questions, so the diff is apples to apples. It shows newly failing
tests, fixed tests, and changes in hallucination rate, latency and cost.

## 3. Gate it

```bash
agentic-qa run --target examples.research_paper_rag.rag:rag_chain \
  --dataset examples/research_paper_rag/golden.jsonl --no-generate \
  --policy examples/research_paper_rag/policy.yaml \
  --max-hallucination 0.10 --max-policy-violations 0 --fail-under 0.8
```

## Growing the golden set

The 16 reference answers were checked against the PDF text, but review them yourself and add more,
aiming for 30 or more. Good additions are questions that need two chunks combined (for example "How did the base and big
models differ in training time and steps?"), questions about tables (Table 3 variants), and more unanswerable
questions that sound plausible.
