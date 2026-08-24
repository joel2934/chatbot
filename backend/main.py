"""
main.py

FastAPI backend for the Company Chatbot.

Architecture:

    Frontend
        ↓
    FastAPI
        ↓
    Conversation History
        ↓
    Traditional RAG Retriever
        ↓
    ChromaDB
        ↓
    Retrieved Policy Context
        ↓
    OpenAI Gateway
        ↓
    gpt-4o-mini
        ↓
    Answer
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathlib import Path
import io

from PyPDF2 import PdfReader

import database as db
import llm

from rag.retriever import retrieve_with_sources


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Company Chatbot API",
    description="Company chatbot with Traditional RAG using ChromaDB",
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

db.init_db()


# Number of previous conversation messages sent to the LLM
HISTORY_WINDOW = 6


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# REQUEST MODELS
# ============================================================

class NewConversation(BaseModel):
    title: str | None = "New Chat"


class NewMessage(BaseModel):
    content: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Company Chatbot API is running",
        "rag": "Traditional RAG",
        "vector_database": "ChromaDB",
        "embedding_model": "text-embedding-3-small",
        "llm": "gpt-4o-mini",
    }


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str | None = Form(None),
):
    """
    Upload a PDF or TXT document.

    The uploaded file is saved locally and its metadata is
    recorded in SQLite.

    NOTE:
    The initial four policy PDFs are ingested into ChromaDB
    using rag/ingest.py.

    This endpoint currently handles file storage and metadata.
    New documents should be added to the RAG vector database
    through the ingestion pipeline.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    filename = Path(file.filename).name

    suffix = Path(filename).suffix.lower()

    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use PDF or TXT.",
        )

    dest = UPLOAD_DIR / filename

    try:

        contents = await file.read()

        # ----------------------------------------------------
        # Save uploaded file
        # ----------------------------------------------------

        with open(dest, "wb") as f:
            f.write(contents)

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        text = ""

        if suffix == ".pdf":

            reader = PdfReader(io.BytesIO(contents))

            pages = []

            for page in reader.pages:

                try:
                    page_text = page.extract_text() or ""
                    pages.append(page_text)

                except Exception:
                    pages.append("")

            text = "\n\n".join(pages)

        elif suffix == ".txt":

            text = contents.decode(
                "utf-8",
                errors="ignore",
            )

        # ----------------------------------------------------
        # Validate extracted content
        # ----------------------------------------------------

        if not text.strip():

            raise HTTPException(
                status_code=400,
                detail="No readable text could be extracted from the document.",
            )

        # ----------------------------------------------------
        # Save document metadata
        # ----------------------------------------------------

        try:

            db.add_document(
                filename,
                str(dest),
                conversation_id,
            )

        except Exception as e:

            print(
                "Warning: could not write document metadata:",
                e,
            )

        # ----------------------------------------------------
        # NOTE ABOUT RAG
        # ----------------------------------------------------

        print("\n========== DOCUMENT UPLOAD ==========")
        print("Filename:", filename)
        print("Saved to:", dest)
        print("Extracted characters:", len(text))
        print(
            "NOTE: Run the RAG ingestion pipeline "
            "to add this document to ChromaDB."
        )

        return {
            "status": "ok",
            "filename": filename,
            "path": str(dest),
            "characters_extracted": len(text),
            "message": (
                "Document uploaded successfully. "
                "Run the RAG ingestion pipeline to add it to ChromaDB."
            ),
        }

    except HTTPException:
        raise

    except Exception as e:

        print("Document upload error:", repr(e))

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}",
        )


# ============================================================
# DOCUMENT LISTING
# ============================================================

@app.get("/documents")
def list_documents(
    conversation_id: str | None = None,
):
    """
    Return uploaded document metadata.
    """

    try:

        docs = db.list_documents(
            conversation_id,
        )

        return docs

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# CONVERSATIONS
# ============================================================

@app.post("/conversations")
def create_conversation(
    body: NewConversation,
):
    """
    Create a new conversation.
    """

    return db.create_conversation(
        title=body.title or "New Chat"
    )


@app.get("/conversations")
def list_conversations():
    """
    Return all conversations.
    """

    return db.list_conversations()


@app.get("/conversations/search")
def search_conversations(
    q: str,
):
    """
    Search conversations by title or message content.
    """

    return db.search_conversations(q)


@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
):
    """
    Return a conversation and all its messages.
    """

    convo = db.get_conversation(
        conversation_id
    )

    if not convo:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    convo["messages"] = db.get_messages(
        conversation_id
    )

    return convo


# ============================================================
# DELETE CONVERSATION
# ============================================================

@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
):
    """
    Delete a conversation and its messages.
    """

    convo = db.get_conversation(
        conversation_id
    )

    if not convo:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    db.delete_conversation(
        conversation_id
    )

    return {
        "message": "Conversation deleted successfully",
        "conversation_id": conversation_id,
    }


# ============================================================
# SEND MESSAGE
# ============================================================

@app.post("/conversations/{conversation_id}/messages")
def send_message(
    conversation_id: str,
    body: NewMessage,
):
    """
    Main chatbot endpoint.

    Flow:

        User question
              ↓
        Conversation history
              ↓
        Vector RAG retrieval
              ↓
        ChromaDB
              ↓
        Relevant policy chunks
              ↓
        LLM
              ↓
        Save response
              ↓
        Return answer + sources
    """

    # ========================================================
    # 1. Validate conversation
    # ========================================================

    convo = db.get_conversation(
        conversation_id
    )

    if not convo:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    # ========================================================
    # 2. Validate question
    # ========================================================

    question = body.content.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Message content cannot be empty",
        )

    # ========================================================
    # 3. Retrieve conversation history
    # ========================================================

    prior_messages = db.get_messages(
        conversation_id,
        limit=HISTORY_WINDOW,
    )

    history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in prior_messages
    ]

    # ========================================================
    # 4. Traditional RAG retrieval
    # ========================================================

    try:

        retrieved_results = retrieve_with_sources(
            question,
            top_k=3,
        )

    except Exception as e:

        print("\n========== RAG ERROR ==========")
        print(repr(e))

        raise HTTPException(
            status_code=500,
            detail=f"RAG retrieval failed: {str(e)}",
        )

    # ========================================================
    # 5. Extract retrieved text
    # ========================================================

    relevant_knowledge = [
        result["content"]
        for result in retrieved_results
    ]

    # ========================================================
    # 6. Debug RAG results
    # ========================================================

    print("\n======================================")
    print("TRADITIONAL RAG RETRIEVAL")
    print("======================================")

    print("Question:")
    print(question)

    print("\nRetrieved chunks:")
    print(len(retrieved_results))

    for index, result in enumerate(
        retrieved_results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        print("\n--------------------------------------")
        print(f"Result {index}")
        print("--------------------------------------")

        print(
            "Source:",
            metadata.get("source"),
        )

        print(
            "Page:",
            metadata.get("page"),
        )

        print(
            "Chunk:",
            metadata.get("chunk"),
        )

        print(
            "Distance:",
            result.get("distance"),
        )

        print("Content:")
        print(result.get("content"))

    # ========================================================
    # 7. Save user message
    # ========================================================

    db.add_message(
        conversation_id,
        "user",
        question,
    )

    # ========================================================
    # 8. Auto-title conversation
    # ========================================================

    if (
        convo["title"] == "New Chat"
        and not prior_messages
    ):

        db.update_conversation_title(
            conversation_id,
            question[:50],
        )

    # ========================================================
    # 9. Generate LLM response
    # ========================================================

    try:

        print("\n======================================")
        print("LLM REQUEST")
        print("======================================")

        print("Question:", question)
        print("History:", history)
        print(
            "Retrieved knowledge chunks:",
            len(relevant_knowledge),
        )

        answer = llm.generate_response(
            question=question,
            history=history,
            knowledge_chunks=relevant_knowledge,
        )

        print("\n======================================")
        print("LLM RESPONSE")
        print("======================================")

        print(answer)

    except Exception as e:

        print("\n======================================")
        print("LLM ERROR")
        print("======================================")

        print(repr(e))

        raise HTTPException(
            status_code=500,
            detail=f"LLM call failed: {str(e)}",
        )

    # ========================================================
    # 10. Save assistant response
    # ========================================================

    saved_reply = db.add_message(
        conversation_id,
        "assistant",
        answer,
    )

    # ========================================================
    # 11. Prepare source information
    # ========================================================

    sources = []

    for result in retrieved_results:

        metadata = result.get(
            "metadata",
            {},
        )

        sources.append(
            {
                "document": metadata.get(
                    "source"
                ),
                "page": metadata.get(
                    "page"
                ),
                "chunk": metadata.get(
                    "chunk"
                ),
                "distance": result.get(
                    "distance"
                ),
            }
        )

    # ========================================================
    # 12. Return response
    # ========================================================

    return {
        "user_message": question,

        "assistant_message": saved_reply,

        "sources": sources,

        "retrieved_chunks": len(
            retrieved_results
        ),
    }