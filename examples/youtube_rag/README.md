# Testing the YouTube RAG assistant

Target: a Flask YouTube RAG backend (`server.py`): `POST /chat` → `{"answer", "sources", "contexts"}`,
Qwen2.5-7B via Hugging Face, FAISS + CrossEncoder rerank. `contexts` (the retrieved chunks, as
`"[M:SS - M:SS]: text"`) must be returned for the citation check and grounded judging.

## 1. Start the backend

```bash
cd path/to/youtube-rag/backend
python server.py            # serves on http://localhost:7860
```

No Hugging Face inference credits? The backend has an opt-in OpenAI mode (same retrieval, reranking
and prompts, only the LLM changes):

```bash
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python server.py   # gpt-4o-mini; OPENAI_MODEL to change
```

In this mode the `X-HF-Token` header is not needed.

## 2. Index a video

```bash
python -m examples.youtube_rag.index_video --video <VIDEO_ID> --server http://localhost:7860
```

This posts the transcript to `/save_transcript` and saves `transcript_<VIDEO_ID>.json`.
Use that file to write `golden.jsonl` (question + reference answer + timestamp), and add a few
questions the video does not answer (`"answerable": false`) to measure hallucination.

## 3. Run Agentic QA

```bash
agentic-qa run \
  --endpoint http://localhost:7860/chat \
  --body-template '{"question": "{input}", "mode": "concise"}' \
  --output-path answer \
  --contexts-path contexts \
  --header 'X-HF-Token: ${HUGGINGFACEHUB_API_TOKEN}' \
  --description "Answers questions about one YouTube video from its transcript, citing [MM:SS - MM:SS] timestamps" \
  --policy examples/youtube_rag/policy.yaml \
  --dataset examples/youtube_rag/golden.jsonl \
  --workers 1 \
  --output qa_run_youtube.json
```

`--contexts-path contexts` reads the chunks the backend returns, so the judge checks grounding
against what the model actually saw, and the citation check can verify every timestamp.

`--workers 1` because the Hugging Face free tier rate-limits and the backend reloads the
CrossEncoder on every request.

## What the policy checks

| Rule | How |
|---|---|
| Cites `[MM:SS - MM:SS]` timestamps (skipped when it says "not in the video") | regex, hard |
| Every cited range exists in the retrieved chunks (no invented timestamps) | citation check, hard |
| No outside knowledge | judge |
| Refuses harmful requests | judge (red team attacks it) |
| No leaked `[NO_SOURCES]` / `REJECT` tokens | regex, hard |
| Doesn't stop mid-sentence | regex, report only |

## What the runs found

Video: 3Blue1Brown, *But what is a neural network?* (`aircAruvnKk`); backend in OpenAI mode (gpt-4o-mini).

- **Invented timestamps.** In about a quarter of answers the model cites ranges such as `[02:33 - 03:00]` or
  `[13:32 - 13:50]` that are not chunk boundaries (the real chunks are `2:33-5:54`, `13:32-16:48`, …), even
  when the answer itself is right. The `citations_exist` check catches every one. Fix in the backend: number
  the chunks in the prompt, have the model cite numbers, and map numbers to ranges in code.
- **Golden set:** 85–90% of answers correct; all 4 unanswerable questions answered "not in the video".
- **Guardrail bug, low impact.** When the query rewriter returns `REJECT`, `server.py` falls back to the raw
  question instead of refusing. In practice the grounded answer prompt then replies "not in the video", so
  no harmful answer was produced, but the guardrail itself never blocks anything.

`labels_claude.jsonl` holds pass/fail labels for one 38-test run, written by an AI assistant. Use them with
`agentic-qa agreement`, or label the outputs yourself with `agentic-qa label`.

## Troubleshooting

- **HTTP 400 from `/chat`**: the token has a trailing `\r` (a `.env` saved with Windows line endings).
  Load it with `export HUGGINGFACEHUB_API_TOKEN=$(grep ^HUGGINGFACEHUB_API_TOKEN .env | cut -d= -f2- | tr -d '\r"')`.
- **401 "Invalid username or password"**: the HF token is expired or revoked. Create a new *Read* token
  with "Make calls to Inference Providers" enabled at https://huggingface.co/settings/tokens.
