"""
main.py
FastAPI app wiring together: conversations, messages, context
retrieval, knowledge retrieval, the LLM call, and search.

Run with:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database as db
import knowledge
import llm

import os
from pathlib import Path
import io
from datetime import datetime
from PyPDF2 import PdfReader

app = FastAPI(title="Company Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a local learning project; tighten for real deployments
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()

HISTORY_WINDOW = 6  # last N messages used as context (see knowledge.py note on upgrading this)


class NewConversation(BaseModel):
    title: str | None = "New Chat"


class NewMessage(BaseModel):
    content: str


@app.get("/")
def root():
    return {"status": "ok", "message": "Company Chatbot API is running"}


# ---------- document upload & listing ----------

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), conversation_id: str | None = Form(None)):
    """Accept PDF or TXT files, extract text, and append to knowledge store."""
    filename = file.filename
    dest = UPLOAD_DIR / filename

    contents = await file.read()

    # Save raw file
    with open(dest, "wb") as f:
        f.write(contents)

    text = ""
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(io.BytesIO(contents))
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            text = "\n\n".join(pages)
        elif suffix in {".txt"}:
            text = contents.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or TXT.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {e}")

    # Append extracted text to knowledge store
    knowledge.append_text_to_knowledge(text, source=filename)

    # record document metadata in the database
    try:
        db.add_document(filename, str(dest), conversation_id)
    except Exception as e:
        # don't fail the whole upload if DB write fails, but log
        print("Warning: could not write document metadata:", e)

    return {"status": "ok", "filename": filename}


@app.get("/documents")
def list_documents(conversation_id: str | None = None):
    """Return documents, optionally filtered by `conversation_id`."""
    try:
        docs = db.list_documents(conversation_id)
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- conversations ----------

@app.post("/conversations")
def create_conversation(body: NewConversation):
    return db.create_conversation(title=body.title or "New Chat")


@app.get("/conversations")
def list_conversations():
    return db.list_conversations()


@app.get("/conversations/search")
def search_conversations(q: str):
    return db.search_conversations(q)


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    convo = db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    convo["messages"] = db.get_messages(conversation_id)
    return convo

@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    convo = db.get_conversation(conversation_id)

    if not convo:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    db.delete_conversation(conversation_id)

    return {
        "message": "Conversation deleted successfully",
        "conversation_id": conversation_id
    }

# ---------- messages ----------

@app.post("/conversations/{conversation_id}/messages")
def send_message(conversation_id: str, body: NewMessage):
    convo = db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    question = body.content.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    # 1. Retrieve context: prior messages in this conversation
    prior_messages = db.get_messages(conversation_id, limit=HISTORY_WINDOW)
    history = [{"role": m["role"], "content": m["content"]} for m in prior_messages]

    # 2. Retrieve relevant company knowledge (RAG-lite)
    relevant_knowledge = knowledge.retrieve_relevant_knowledge(question)

    # 3. Save the user's message
    db.add_message(conversation_id, "user", question)

    # 4. Auto-title a fresh conversation from its first message
    if convo["title"] == "New Chat" and not prior_messages:
        db.update_conversation_title(conversation_id, question[:50])

    # 5. Call the LLM with context + knowledge + question
    try:
        print("\n========== LLM REQUEST ==========")
        print("Question:", question)
        print("History:", history)
        print("Knowledge:", relevant_knowledge)

        answer = llm.generate_response(
            question,
            history,
            relevant_knowledge
        )

        print("========== LLM RESPONSE ==========")
        print(answer)

    except Exception as e:
        print("\n========== LLM ERROR ==========")
        print(repr(e))

        raise HTTPException(
            status_code=500,
            detail=f"LLM call failed: {str(e)}"
        )

    # 6. Save the assistant's reply
    saved_reply = db.add_message(conversation_id, "assistant", answer)

    return {
        "user_message": question,
        "assistant_message": saved_reply,
        "used_knowledge": relevant_knowledge,
    }
