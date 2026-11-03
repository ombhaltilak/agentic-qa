# Changelog

## 0.3.0

### Judging you can trust
- Layered judge: deterministic assertions (PII, injection compliance, prompt leaks, …) override model verdicts.
- Jev decision model as a cheap probabilistic judge (`jev`, `cascade`, `ensemble`, `auto` backends; free `jev-1.13-free` with automatic fallback to paid).
- Blind review of generated tests: a reviewer model checks each test's expectation without seeing the answer. Wrong expectations give an `invalid` verdict, excluded from all metrics.
- The judge sees the retrieved contexts (up to 2,000 characters per chunk), judges in batches of 8, and re-asks for any test it skipped.
- `agentic-qa label`, `calibrate`, `rejudge` and `agreement`: judge accuracy, failure precision and recall with 95% intervals.

### Your rules
- `policy.yaml`: must / must_not rules used as regex checks, judge criteria, Jev questions and red-team targets.
- `citations:` rules verify every citation against the retrieved contexts.
- The red team is assigned the least-tested rules every iteration, so every rule gets attacked.

### Ground truth & CI
- Golden datasets with reference answers; abstention-aware hallucination rate. Correct answers that only break a policy rule count as correct, but still fail.
- Default quality gate: any critical/high failure exits 1. The CLI always prints which gates it checked; `--no-gates` turns this off.
- `agentic-qa diff` compares two runs (newly failing, fixed, metric deltas) and can write Markdown for a PR comment.
- Per-system regression memory with suspected / confirmed / flaky / fixed / invalid states; `agentic-qa memory list | invalidate`.
- Latency (p50/p95) and QA token cost in every run.

### Connect anything
- HTTP adapter: nested JSON paths, body templates, `${ENV}` headers. Also an OpenAI-compatible chat adapter and a LangChain Runnable adapter.
- Examples: `examples/research_paper_rag/` (LangChain + FAISS over a PDF) and `examples/youtube_rag/` (Flask API, timestamp-citation policy).

### Fixes
- Executor no longer merges an adapter's `sources` labels into the retrieved contexts.
- Coverage is measured from the attack taxonomy, not estimated by an LLM.
