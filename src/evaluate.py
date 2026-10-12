
from src.retrieval import build_search_index, search_policies


TEST_CASES = [
    {
        "question": "How many days of annual leave do employees receive?",
        "expected_source": "leave_policy.txt",
    },
    {
        "question": "How many days of sick leave do employees receive?",
        "expected_source": "leave_policy.txt",
    },
    {
        "question": "How many remote working days are permitted?",
        "expected_source": "remote_work_policy.txt",
    },
    {
        "question": "How long is the probation period?",
        "expected_source": "probation_policy.txt",
    },
    {
        "question": "What is the company's paid parental leave policy?",
        "expected_source": None,
    },
]


def evaluate_retrieval():
    chunks, embeddings = build_search_index()

    correct = 0

    print("\nRAG Retrieval Evaluation")
    print("=" * 60)

    for test in TEST_CASES:
        results = search_policies(
            question=test["question"],
            chunks=chunks,
            embeddings=embeddings,
            top_k=3,
            min_score=0.50,
        )

        sources = {result["source"] for result in results}
        expected = test["expected_source"]

        if expected is None:
            passed = len(results) == 0
        else:
            passed = expected in sources

        correct += int(passed)

        best_score = (
            max(result["score"] for result in results)
            if results else None
        )

        print(f"\nQuestion: {test['question']}")
        print(f"Expected source: {expected or 'No relevant source'}")
        print(f"Retrieved sources: {sorted(sources) or 'None'}")
        print(f"Best similarity score: {best_score}")
        print(f"Result: {'PASS' if passed else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"Retrieval success: {correct}/{len(TEST_CASES)}")
    print(f"Success rate: {correct / len(TEST_CASES) * 100:.1f}%")


if __name__ == "__main__":
    evaluate_retrieval()
