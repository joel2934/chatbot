from pathlib import Path
from .embeddings import get_embedding
import chromadb


# --------------------------------------------------
# ChromaDB configuration
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent

VECTOR_DB_PATH = BACKEND_DIR / "vector_store"

COLLECTION_NAME = "company_policies"


# --------------------------------------------------
# ChromaDB client
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)


# --------------------------------------------------
# Get / create collection
# --------------------------------------------------

def get_collection():
    """
    Return the company policy vector collection.

    Cosine similarity is used because we are working
    with semantic embeddings.
    """

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Company HR and policy documents",
            "hnsw:space": "cosine",
        },
    )


# --------------------------------------------------
# Add documents
# --------------------------------------------------

def add_documents(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
):
    """
    Add document chunks and their embeddings to ChromaDB.
    """

    if not documents:
        return

    collection = get_collection()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# --------------------------------------------------
# Search
# --------------------------------------------------

def search(
    query_embedding: list[float],
    top_k: int = 5,
):
    """
    Search the vector database for chunks that are
    semantically similar to the query.
    """

    collection = get_collection()

    # Avoid asking for more documents than exist.
    count = collection.count()

    if count == 0:
        return []

    top_k = min(top_k, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved.append(
            {
                "content": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return retrieved


# --------------------------------------------------
# Collection information
# --------------------------------------------------

def get_document_count() -> int:
    """
    Return the number of chunks currently stored.
    """

    collection = get_collection()

    return collection.count()


# --------------------------------------------------
# Reset collection
# --------------------------------------------------

def reset_collection():
    """
    Delete the existing collection.

    Useful when completely re-ingesting the PDFs.
    """

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
    except Exception:
        pass

    get_collection()