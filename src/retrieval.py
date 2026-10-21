from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingestion import load_documents, chunk_documents


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Keep the ChromaDB database inside the project's data folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "company_policies"



def build_search_index():
    """Create, reuse, or refresh the persistent policy index."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Always read the current policy files and create their chunks.
    documents = load_documents()
    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError("No policy chunks found.")

    texts = [chunk["text"] for chunk in chunks]
    sources = [chunk["source"] for chunk in chunks]
    ids = [f"policy_chunk_{i}" for i in range(len(chunks))]

    # Read the existing index for comparison.
    existing = collection.get(
        include=["documents", "metadatas"]
    )

    existing_records = sorted(
        (
            chunk_id,
            text,
            metadata.get("source", "")
        )
        for chunk_id, text, metadata in zip(
            existing["ids"],
            existing["documents"],
            existing["metadatas"]
        )
    )

    current_records = sorted(
        (chunk_id, text, source)
        for chunk_id, text, source in zip(ids, texts, sources)
    )

    # If both the content and sources match, reuse the index.
    if existing_records == current_records:
        print(
            f"Policy documents unchanged. Reusing ChromaDB index: "
            f"{collection.count()} chunks"
        )
        return collection

    # The documents changed, or the collection is empty.
    # Remove the old chunks before inserting the updated ones.
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Generate embeddings only for the updated document chunks.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": source}
            for source in sources
        ]
    )

    print(
        f"Policy index refreshed. "
        f"ChromaDB now contains {collection.count()} chunks"
    )
    return collection



def search_policies(question, collection, top_k=3, min_score=0.50):
    """Retrieve relevant policy chunks using ChromaDB similarity search."""

    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    matches = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        # For cosine distance, similarity = 1 - distance
        score = 1 - distance

        if score < min_score:
            continue

        matches.append({
            "text": text,
            "source": metadata["source"],
            "score": round(score, 4)
        })

    return matches


if __name__ == "__main__":
    collection = build_search_index()

    question = "What is the company's paid parental leave policy?"
    results = search_policies(question, collection)

    print(f"\nQuestion: {question}")

    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {result['source']}")
        print(f"Similarity score: {result['score']}")
        print(result["text"])
