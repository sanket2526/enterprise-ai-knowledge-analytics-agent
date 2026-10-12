
import unittest
from unittest.mock import patch

from src.agent import classify_question, run_agent


class TestAgentRouting(unittest.TestCase):

    @patch("src.agent.client.models.generate_content")
    def test_policy_question_routing(self, mock_generate):
        mock_generate.return_value.text = "policy"

        result = classify_question(
            "How many sick leave days do employees receive?"
        )

        self.assertEqual(result, "policy")

    @patch("src.agent.client.models.generate_content")
    def test_analytics_question_routing(self, mock_generate):
        mock_generate.return_value.text = "analytics"

        result = classify_question(
            "Which department has the most employees?"
        )

        self.assertEqual(result, "analytics")

    @patch("src.agent.answer_question")
    @patch("src.agent.classify_question")
    def test_unsupported_policy_question(
        self, mock_classify, mock_answer
    ):
        mock_classify.return_value = "policy"
        mock_answer.return_value = {
            "answer": (
                "I couldn't find sufficiently relevant information "
                "in the available company policies to answer this question."
            ),
            "sources": [],
            "retrieved_chunks": [],
        }

        result = run_agent(
            "What is the paid parental leave policy?",
            chunks=[],
            embeddings=None,
        )

        self.assertEqual(result["category"], "policy")
        self.assertEqual(result["sources"], [])
        self.assertIn("couldn't find", result["answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
