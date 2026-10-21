import unittest
from unittest.mock import patch, MagicMock

from src.agent import run_agent


def make_message(name, content):
    """Create a mock message with an explicit name and content."""
    message = MagicMock()
    message.name = name
    message.content = content
    return message


class TestAgentRouting(unittest.TestCase):

    @patch("src.agent.agent.invoke")
    def test_policy_question_routing(self, mock_invoke):
        mock_invoke.return_value = {
            "messages": [
                make_message(
                    "user",
                    "How many sick leave days do employees receive?"
                ),
                make_message(
                    "search_company_policies",
                    '{"answer": "Employees receive 10 sick leave days.", '
                    '"sources": ["leave_policy.txt"]}'
                ),
                make_message(
                    "assistant",
                    "Employees receive 10 sick leave days."
                ),
            ]
        }

        result = run_agent(
            "How many sick leave days do employees receive?"
        )

        self.assertEqual(result["category"], "policy")
        self.assertEqual(result["sources"], ["leave_policy.txt"])
        self.assertIn("10 sick leave days", result["answer"])

    @patch("src.agent.agent.invoke")
    def test_analytics_question_routing(self, mock_invoke):
        mock_invoke.return_value = {
            "messages": [
                make_message(
                    "user",
                    "Which department has the most employees?"
                ),
                make_message(
                    "query_employee_analytics",
                    '{"success": true, "rows": [["Engineering", 7]]}'
                ),
                make_message(
                    "assistant",
                    "Engineering has the most employees, with 7."
                ),
            ]
        }

        result = run_agent(
            "Which department has the most employees?"
        )

        self.assertEqual(result["category"], "analytics")
        self.assertEqual(result["sources"], ["employee_analytics.db"])
        self.assertIn("Engineering", result["answer"])

    @patch("src.agent.agent.invoke")
    def test_unsupported_policy_question(self, mock_invoke):
        mock_invoke.return_value = {
            "messages": [
                make_message(
                    "user",
                    "What is the paid parental leave policy?"
                ),
                make_message(
                    "search_company_policies",
                    '{"answer": "I couldn\'t find sufficiently relevant '
                    'information in the available company policies.", '
                    '"sources": []}'
                ),
                make_message(
                    "assistant",
                    "I couldn't find sufficiently relevant information "
                    "in the available company policies."
                ),
            ]
        }

        result = run_agent(
            "What is the paid parental leave policy?"
        )

        self.assertEqual(result["category"], "policy")
        self.assertEqual(result["sources"], [])
        self.assertIn("couldn't find", result["answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
