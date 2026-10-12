
from sentence_transformers import SentenceTransformer, util
from src.ingestion import load_documents, chunk_documents


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_search_index():
    """Load policy documents, chunk them, and create embeddings."""
    documents = load_documents()
    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError("No policy chunks found.")

    texts = [chunk["text"] for chunk in chunks]

    # Convert each chunk into a numerical vector
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    return chunks, embeddings



def search_policies(
    question,
    chunks,
    embeddings,
    top_k=3,
    min_score=0.50
):
    """Return relevant policy chunks above a similarity threshold."""

    question_embedding = model.encode(
        question,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    scores = util.cos_sim(question_embedding, embeddings)[0]

    k = min(top_k, len(chunks))
    top_results = scores.topk(k)

    results = []

    for score, index in zip(
        top_results.values.tolist(),
        top_results.indices.tolist()
    ):
        # Ignore chunks with weak similarity scores
        if score < min_score:
            continue

        results.append({
            "text": chunks[index]["text"],
            "source": chunks[index]["source"],
            "score": round(score, 4)
        })

    return results



if __name__ == "__main__":
    chunks, embeddings = build_search_index()

    question = "What is the company's paid parental leave policy?"
    results = search_policies(question, chunks, embeddings)

    print(f"\nQuestion: {question}")

    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {result['source']}")
        print(f"Similarity score: {result['score']}")
        print(result["text"])
