# Company Chatbot — Simple GenAI Project

A small, working implementation of the workflow you outlined:
Chat ID → Conversation ID → Context → Retrieval → LLM → Response → Store → Search.

No vector DB, no auth system, no build tooling for the frontend — just enough to see
every piece of the pipeline clearly. Swap pieces out (keyword search → embeddings,
SQLite → Postgres, React CDN → Vite) once each concept makes sense.

## Structure

```
company-chatbot/
├── backend/
│   ├── main.py                 FastAPI app — all endpoints
│   ├── database.py             SQLite: conversations + messages
│   ├── knowledge.py            Keyword-based retrieval (RAG-lite)
│   ├── llm.py                  Builds the prompt, calls the LLM
│   ├── company_knowledge.txt   Sample company info (edit this)
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── index.html              React (via CDN, no build step) — chat UI + sidebar
```

## 1. Backend setup

```bash
cd backend
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your Anthropic API key
```

Load the key into your shell (or use `python-dotenv` if you prefer — not included
by default to keep the file count small):


```

Run the API:

```bash
uvicorn main:app --reload --port 8000
```

Check it's alive: open http://localhost:8000 → `{"status": "ok", ...}`.
Interactive API docs: http://localhost:8000/docs

## 2. Frontend setup

No install needed — it's a single HTML file using React from a CDN.

```bash
cd frontend
python3 -m http.server 5500
```

Open http://localhost:5500 in your browser. Make sure the backend is running on
port 8000 (the frontend calls `http://localhost:8000` directly — see `API_BASE`
at the top of `index.html` if you need to change it).

## 3. Try it

1. Click **+ New Chat**.
2. Ask: *"What are the working hours?"*
3. Ask a follow-up: *"And how many casual leaves do I get?"* — notice the AI keeps
   context from the previous turn.
4. Click **+ New Chat** again and ask something unrelated — it won't remember the
   first conversation. That's the conversation-ID boundary from the workflow doc.
5. Use the sidebar search box to find an old chat by keyword.

## How each workflow step maps to code

| Workflow step | File |
|---|---|
| Create/get chat ID | `database.create_conversation` / `get_conversation` |
| Retrieve chat history | `database.get_messages(limit=HISTORY_WINDOW)` in `main.py` |
| Retrieve company knowledge | `knowledge.retrieve_relevant_knowledge` |
| Build context + prompt | `llm.build_system_prompt` |
| Call LLM | `llm.generate_response` |
| Save message | `database.add_message` |
| Search old chats | `database.search_conversations` |

## Next upgrades (one at a time, as the doc suggests)

- **Embeddings + vector DB**: replace `knowledge.py`'s keyword overlap with an
  embedding model + a vector store (e.g. Chroma) for semantic search over
  `company_knowledge.txt`.
- **Conversation summaries**: once `HISTORY_WINDOW` messages isn't enough context,
  summarize older messages instead of dropping them.
- **Multi-user auth**: `user_id` is already threaded through the DB — add real
  auth and stop defaulting to `"default_user"`.
- **Streaming responses**: swap `client.messages.create` for `client.messages.stream`
  in `llm.py` and stream tokens to the frontend.

## Notes

- SQLite file (`backend/chatbot.db`) is created automatically on first run.
- CORS is wide open (`allow_origins=["*"]`) since this is a local learning
  project — restrict it before deploying anywhere real.
- Model used: `claude-sonnet-4-5` (see `MODEL` in `llm.py`) — change it there if
  you want a different one.




