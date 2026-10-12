
import os
from dotenv import load_dotenv
from google import genai

from src.retrieval import build_search_index, search_policies

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.5-flash-lite"


def answer_question(question, chunks, embeddings):
    """Answer a policy question using retrieved document context."""

    # Step 1: Retrieve relevant policy chunks
    results = search_policies(
        question=question,
        chunks=chunks,
        embeddings=embeddings,
        top_k=3
    )

    # Stop if no chunks are sufficiently relevant
    if not results:
        return {
            "answer": (
                "I couldn't find sufficiently relevant information "
                "in the available company policies to answer this question."
            ),
            "sources": [],
            "retrieved_chunks": []
        }


    # Step 2: Combine the retrieved chunks into context
    context = "\n\n".join(
        f"Source: {result['source']}\n{result['text']}"
        for result in results
    )

    # Step 3: Ask Gemini to answer using the supplied context
    prompt = f"""
You are an enterprise policy assistant.

Answer the user's question using ONLY the policy context below.
If the context does not contain enough information, clearly say so.
Do not invent company rules.
Keep the answer concise and easy to understand.

POLICY CONTEXT:
{context}

USER QUESTION:
{question}

Answer the question and identify the source file(s) used.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return {
        "answer": response.text,
        "sources": sorted({result["source"] for result in results}),
        "retrieved_chunks": results
    }



if __name__ == "__main__":
    print("Initializing Enterprise AI Knowledge Agent...")

    # Build the search index once when the program starts
    chunks, embeddings = build_search_index()

    print("\nAgent is ready!")
    print("Ask a question about company policies.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Agent: Goodbye!")
            break

        if not question:
            continue

        try:
            result = answer_question(
                question,
                chunks,
                embeddings
            )

            print("\nAgent:", result["answer"])
            print("\nRetrieved source files:")

            if result["sources"]:
                for source in result["sources"]:
                        print("-", source)
            else:
                print("None (no sufficiently relevant chunks found)")

            print()

        except Exception as error:
            print(f"\nSomething went wrong: {error}\n")

