import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.rag import answer_question
from src.retrieval import build_search_index
from src.tools import employee_analytics

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

# The persistent ChromaDB collection is assigned when the agent starts.
policy_collection = None


@tool
def search_company_policies(question: str) -> str:
    """Search company policy documents to answer questions about leave,
    remote work, probation, and other documented company policies."""

    if policy_collection is None:
        return json.dumps({
            "answer": "The policy search index has not been initialized.",
            "sources": [],
        })

    result = answer_question(question, policy_collection)

    return json.dumps({
        "answer": result["answer"],
        "sources": result["sources"],
    })


@tool
def query_employee_analytics(question: str) -> str:
    """Answer questions about employee counts, departments, locations,
    employment types, and joining dates using the employee dataset."""

    result = employee_analytics(question)

    if not result["success"]:
        return json.dumps({
            "success": False,
            "error": result["error"],
        })

    return json.dumps({
        "success": True,
        "question": result["question"],
        "columns": result["columns"],
        "rows": result["rows"],
    })


llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
)

agent = create_agent(
    model=llm,
    tools=[search_company_policies, query_employee_analytics],
    system_prompt=(
        "You are an enterprise knowledge and analytics assistant. "
        "Use search_company_policies for questions about company policies. "
        "Use query_employee_analytics for questions about employee data. "
        "Do not invent facts or results. Base your answer on tool outputs. "
        "If policy information is unavailable, clearly say: "
        "'The available company policies do not provide enough information "
        "to answer this question.' "
        "If a tool reports an error, explain it clearly. "
        "For unrelated questions, politely explain the assistant's scope. "
        "Write grammatically correct, natural sentences. "
        "Always include spaces between words and proofread your final answer."
    ),
)


def _message_text(content):
    """Convert a model message's content into readable text."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)

    return str(content)


def run_agent(question, collection=None, chunks=None, embeddings=None):
    """Run a question through the LangChain agent.

    The optional chunks and embeddings parameters temporarily preserve
    compatibility with the previous function signature.
    """

    global policy_collection

    if collection is not None:
        policy_collection = collection

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    

    messages = result["messages"]
    category = "other"
    sources = []

    # Identify which tool LangChain selected and collect its metadata.
    for message in messages:
        tool_name = getattr(message, "name", None)

        if tool_name == "search_company_policies":
            category = "policy"

            try:
                tool_result = json.loads(message.content)
                sources.extend(tool_result.get("sources", []))
            except (json.JSONDecodeError, TypeError):
                pass

        elif tool_name == "query_employee_analytics":
            category = "analytics"
            sources.append("employee_analytics.db")

    answer = _message_text(messages[-1].content)

    return {
        "category": category,
        "answer": answer,
        "sources": sorted(set(sources)),
    }


if __name__ == "__main__":
    print("Initializing Enterprise AI Knowledge & Analytics Agent...")

    policy_collection = build_search_index()

    print("Agent ready! Ask about company policies or employee analytics.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            result = run_agent(question, collection=policy_collection)

            print(f"\nCapability used: {result['category']}")
            print(f"Answer: {result['answer']}")

            if result["sources"]:
                print(f"Sources: {', '.join(result['sources'])}")

        except Exception as error:
            print(f"An error occurred: {error}")
