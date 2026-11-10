# RoboResearch AI backend

## Local setup

From this directory, create and activate a virtual environment, install the
dependencies, copy `.env.example` to `.env`, and add the needed API keys:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python -m uvicorn main:app --reload
```

Set `GROQ_API_KEY` for Groq chat and `GEMINI_API_KEY` for Gemini PDF embeddings.
For local chat, set `CHAT_PROVIDER=ollama`; for local embeddings, set
`EMBEDDING_PROVIDER=ollama`. Ollama must be running with the selected models.

## API contract

- `GET /` returns application name, status, and version.
- `GET /api/health` returns `{"status":"healthy"}`.
- `GET /api/models` returns `{"models":[...],"default_model":"...","provider":"groq|ollama"}`.
  Groq returns the configured model without making an API call. Ollama lists
  installed chat models.
- `POST /api/chat` accepts `multipart/form-data`: required `messages` is a JSON
  string array of `{ "role": "user"|"assistant", "content": "..." }`;
  optional `model` is a string and optional `file` is the PDF itself. The
  response is `{ "reply": "...", "model": "...", "sources": [...] }`.
  Each source has `filename`, `page_number`, `similarity`, and `excerpt`.
- `POST /api/chat/stream` accepts the same multipart fields and returns
  `application/x-ndjson`, one JSON event per line. Token events are
  `{ "type": "token", "content": "..." }`. Completion events are
  `{ "type": "done", "model": "...", "finish_reason": "...",
  "eval_count": 0, "prompt_eval_count": 0, "truncated": false,
  "sources": [...] }`. Error events are
  `{ "type": "error", "message": "..." }`.

There is no standalone PDF upload endpoint. Attach the PDF to the chat request;
it is read, extracted, embedded, retrieved, and discarded in memory. The upload
limit is 20 MiB. Conversation history belongs in browser `localStorage`.

## Frontend integration status

`frontend/src/App.tsx` sends multipart form data to `/api/chat/stream`, retains
the selected PDF `File` in component memory, and attaches it to requests until
the user removes it or switches conversations. Conversation history remains in
browser `localStorage`; attachments are not persisted.
