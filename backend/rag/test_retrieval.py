from retriever import retrieve_with_sources


def main():
    print("======================================")
    print("Traditional RAG - Retrieval Test")
    print("======================================")

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    print("\nSearching vector database...")
    print("--------------------------------------")

    results = retrieve_with_sources(
        question,
        top_k=3
    )

    if not results:
        print("No relevant documents found.")
        return

    print(f"\nRetrieved {len(results)} chunks:\n")

    for index, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"RESULT {index}")
        print(f"Source   : {metadata.get('source')}")
        print(f"Page     : {metadata.get('page')}")
        print(f"Chunk    : {metadata.get('chunk')}")
        print(f"Distance : {result['distance']}")
        print("\nContent:")
        print(result["content"])
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()