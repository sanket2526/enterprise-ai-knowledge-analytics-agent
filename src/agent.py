
import os

from dotenv import load_dotenv
from google import genai

from src.rag import answer_question
from src.retrieval import build_search_index
from src.tools import employee_analytics


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.5-flash-lite"


def classify_question(question):
    """Choose the best capability for a user's question."""

    prompt = f"""
Classify the user's question into exactly one category:

- policy: questions about company rules, annual leave, sick leave,
  remote work, probation, and other company policies.
- analytics: questions requiring calculations or information from
  employee records, such as departments, locations, joining dates,
  employee counts, or managers.
- other: questions unrelated to the available company policies
  or employee dataset.

Return only one word: policy, analytics, or other.

User question: {question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    category = response.text.strip().lower()

    if category not in {"policy", "analytics", "other"}:
        return "other"

    return category


def run_agent(question, chunks, embeddings):
    """Route a question to the appropriate capability."""

    category = classify_question(question)

    if category == "policy":
        result = answer_question(question, chunks, embeddings)

        return {
            "category": category,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    
    if category == "analytics":
        result = employee_analytics(question)

        if not result["success"]:
            return {
                "category": category,
                "answer": f"Analytics error: {result['error']}",
                "sources": []
            }

        if not result["rows"]:
            answer = "No matching employee records were found."
        else:
            # Convert column names into readable labels
            labels = [
                column.replace("_", " ").title()
                for column in result["columns"]
            ]

            # Format each row as a readable result
            formatted_rows = []

            for row in result["rows"]:
                details = [
                    f"{label}: {value}"
                    for label, value in zip(labels, row)
                ]
                formatted_rows.append(", ".join(details))

            answer = "\n".join(formatted_rows)

        return {
            "category": category,
            "answer": answer,
            "sources": ["employee_analytics.db"]
        }


    return {
        "category": "other",
        "answer": (
            "I can help with company policies and employee analytics. "
            "Please ask a question related to either of those areas."
        ),
        "sources": []
    }


if __name__ == "__main__":
    print("Initializing Enterprise AI Knowledge & Analytics Agent...")

    # Build the policy search index once at startup
    chunks, embeddings = build_search_index()

    print("\nAgent ready!")
    print("Ask about company policies or employee analytics.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Agent: Goodbye!")
            break

        if not question:
            continue

        try:
            result = run_agent(question, chunks, embeddings)

            print(f"\nCapability used: {result['category']}")
            print("Agent:", result["answer"])

            if result["sources"]:
                print("Sources:", ", ".join(result["sources"]))

            print()

        except Exception as error:
            print(f"\nSomething went wrong: {error}\n")
