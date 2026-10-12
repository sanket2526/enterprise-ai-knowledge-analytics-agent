
from pathlib import Path


def load_documents(data_dir="data/company_policies"):
    """Load policy text files and preserve their source filenames."""
    folder = Path(data_dir)
    documents = []

    for file_path in sorted(folder.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "text": text,
                "source": file_path.name
            })

    return documents


def chunk_documents(documents, max_chars=500, overlap=100):
    """Split documents into overlapping text chunks."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be between 0 and max_chars")

    chunks = []

    for document in documents:
        text = document["text"]
        start = 0

        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": document["source"]
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} | {chunk['source']} ---")
        print(chunk["text"])
