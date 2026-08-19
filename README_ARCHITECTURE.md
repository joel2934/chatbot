## Company Chatbot — Architecture Explained (Beginner Friendly)

This document explains the pieces of the project, how they interact, and how to run and extend the code. It is written for someone new to GenAI projects and web apps.

**Project Overview:**
- **Purpose:** A simple Retrieval-Augmented Generation (RAG) demo: upload documents, retrieve relevant information, and use a chat LLM to answer questions using both conversation history and uploaded company knowledge.
- **Main pieces:** backend API (FastAPI + SQLite), a tiny frontend (single HTML using React via CDN), and a local file-based knowledge store.

**High-level Flow:**
- **User** sends a message from the frontend.
- **Frontend** calls backend endpoints to create/list/open conversations and to send messages.
- **Backend (`main.py`)** builds context: recent messages + retrieved knowledge chunks.
- **Knowledge retrieval (`knowledge.py`)** finds relevant text chunks from `company_knowledge.txt` (keyword overlap).
- **LLM layer (`llm.py`)** builds a system prompt that includes the retrieved knowledge and conversation history, then calls the configured LLM (OpenRouter/OpenAI client) to generate an answer.
- **Backend** saves messages to the SQLite DB and returns assistant replies to the frontend.

**Files & Components (what they do):**
- **`backend/main.py`**: FastAPI application — endpoints for conversations, messages, file uploads, and listing documents. Orchestrates retrieval + LLM calls.
- **`backend/database.py`**: Lightweight SQLite helpers — conversations, messages, and a `documents` table (stores uploaded file metadata).
- **`backend/knowledge.py`**: Simple RAG-lite retriever that splits `company_knowledge.txt` into chunks and picks top overlaps by keywords. Also contains `append_text_to_knowledge()` used when documents are uploaded.
- **`backend/llm.py`**: Builds the system prompt and calls the LLM client. This is the layer you swap when changing LLM providers.
- **`backend/company_knowledge.txt`**: Text file that stores company information and ingested uploaded text. The retriever reads this file.
- **`backend/uploads/`**: Directory where uploaded files (PDF/TXT) are saved.
- **`frontend/index.html`**: Single-file React UI (no build) — lists conversations, opens chats, sends messages, and lets the user upload files from within each chat.

**How document upload works (end-to-end):**
- Upload from the chat view sends a multipart POST to `/upload`. The request can include `conversation_id`.
- Backend saves the file into `backend/uploads/` and extracts text (PDF via `PyPDF2`, TXT via utf-8 decoding).
- Extracted text is appended to `backend/company_knowledge.txt` via `knowledge.append_text_to_knowledge()` so future retrievals include the uploaded content.
- Backend also records metadata in the `documents` table in `backend/chatbot.db` (filename, path, optional conversation association, upload time).
- The frontend refreshes the document list for the active conversation after upload.

**Why this is RAG (Retrieval-Augmented Generation):**
- The system retrieves small, human-readable pieces of company-wide text and places them in the LLM system prompt. The LLM uses these chunks to ground its answers.
- This avoids putting the entire company corpus in the prompt (costly/slow) and lets your app pick the most relevant snippets.

**Running locally (quick):**
1. Backend
```bash
cd backend
python -m pip install -r requirements.txt
cp .env.example .env           # then add your API key
uvicorn main:app --reload --port 8000
```
2. Frontend
```bash
cd frontend
python -m http.server 5500
# open http://localhost:5500 in your browser
```

**Design trade-offs & notes:**
- **No vector DB by design:** This project uses keyword overlap — simpler to understand and run locally, but less semantically powerful than embeddings + FAISS/Chroma.
- **Knowledge store is a file:** `company_knowledge.txt` is easy to inspect and edit. For larger projects use a document store + vector index.
- **Security & CORS:** CORS is wide open for local testing. Close it for production and add auth.
- **Idempotency & duplicates:** Appending uploaded text directly can create duplicates or long files. Consider chunking, deduplication, or storing original chunks in a vector DB for production.

**Next improvements (recommended path for learning):**
- Replace `knowledge.py` with an embeddings-based retriever (OpenAI, Cohere, or local embedding model) + a vector store (FAISS, Chroma).
- Add document chunking (overlap + size limit) and store chunks separately with ids and metadata.
- Add a preview/download endpoint for uploaded files and show them in the UI.
- Add per-conversation document scoping: currently uploaded text is appended to the global file; you can instead index per-conversation collections.
- Add streaming LLM outputs to show partial responses in real time.

**Troubleshooting tips:**
- If uploads appear to disappear after reload: check `backend/uploads/` and `backend/chatbot.db` for a `documents` entry. Ensure backend process has write permission.
- If LLM calls fail: check `.env` for the API key and confirm the `llm.py` client config matches your provider.
- If retrieved answers seem irrelevant: tune `knowledge.retrieve_relevant_knowledge()` or switch to embeddings.

**Where to look in code for specific tasks:**
- Add new endpoint: `backend/main.py`
- Change retrieval behavior: `backend/knowledge.py`
- Swap LLM provider / model config: `backend/llm.py`
- Change DB schema: `backend/database.py` (remember to handle migrations or re-create the DB during development)
- Frontend UI changes: `frontend/index.html` (React via CDN; no build system)

If you'd like, I can also add a small `docs/quickstart.md` with screenshots and common troubleshooting commands, or implement embeddings + FAISS next — which would you prefer?
