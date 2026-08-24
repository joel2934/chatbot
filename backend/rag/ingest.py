from pathlib import Path

from PyPDF2 import PdfReader

from .embeddings import get_embeddings
from .vector_store import add_documents, reset_collection


# --------------------------------------------------
# Paths
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BACKEND_DIR / "documents"


# --------------------------------------------------
# Chunking configuration
# --------------------------------------------------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# --------------------------------------------------
# PDF extraction
# --------------------------------------------------

def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns:

        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(
                f"Warning: Could not read "
                f"{pdf_path.name}, page {page_number}: {exc}"
            )
            text = ""

        text = text.strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# --------------------------------------------------
# Text chunking
# --------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Example:

        chunk 1 = words 0-800
        chunk 2 = words 680-1480
        chunk 3 = words 1360-2160

    This overlap helps preserve context between chunks.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# --------------------------------------------------
# Process one PDF
# --------------------------------------------------

def process_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract and chunk one PDF.
    """

    print(f"\nProcessing: {pdf_path.name}")

    pages = extract_pdf_pages(pdf_path)

    print(f"Pages with text: {len(pages)}")

    chunks = []

    chunk_number = 0

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        page_chunks = chunk_text(text)

        for chunk in page_chunks:

            chunks.append(
                {
                    "content": chunk,
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                }
            )

            chunk_number += 1

    print(f"Chunks created: {len(chunks)}")

    return chunks


# --------------------------------------------------
# Main ingestion pipeline
# --------------------------------------------------

def main():

    print("======================================")
    print("Traditional RAG - Document Ingestion")
    print("======================================")

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    pdf_files = sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR}"
        )

    print(f"\nPDF files found: {len(pdf_files)}")

    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    # --------------------------------------------------
    # Clear previous vector data
    # --------------------------------------------------

    print("\nResetting existing vector collection...")

    reset_collection()

    # --------------------------------------------------
    # Extract and chunk PDFs
    # --------------------------------------------------

    all_chunks = []

    for pdf_path in pdf_files:

        pdf_chunks = process_pdf(pdf_path)

        all_chunks.extend(pdf_chunks)

    if not all_chunks:
        raise ValueError(
            "No text could be extracted from the PDFs."
        )

    print("\n======================================")
    print(f"Total chunks: {len(all_chunks)}")
    print("======================================")

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------

    documents = [
        item["content"]
        for item in all_chunks
    ]

    metadatas = [
        {
            "source": item["source"],
            "page": item["page"],
            "chunk": item["chunk"],
        }
        for item in all_chunks
    ]

    ids = [
        f"{item['source']}_{item['page']}_{item['chunk']}"
        for item in all_chunks
    ]

    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------

    print("\nGenerating embeddings...")
    print("Model: text-embedding-3-small")

    # Process in batches to avoid sending a huge request.
    BATCH_SIZE = 50

    all_embeddings = []

    for start in range(
        0,
        len(documents),
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            len(documents),
        )

        batch = documents[start:end]

        print(
            f"Embedding chunks "
            f"{start + 1}-{end} "
            f"of {len(documents)}..."
        )

        batch_embeddings = get_embeddings(batch)

        all_embeddings.extend(
            batch_embeddings
        )

    # --------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------

    print("\nStoring vectors in ChromaDB...")

    add_documents(
        ids=ids,
        documents=documents,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )

    print("\n======================================")
    print("INGESTION COMPLETE")
    print("======================================")
    print(f"PDFs processed: {len(pdf_files)}")
    print(f"Chunks stored: {len(documents)}")
    print("Vector database: ChromaDB")
    print("Embedding model: text-embedding-3-small")


if __name__ == "__main__":
    main()