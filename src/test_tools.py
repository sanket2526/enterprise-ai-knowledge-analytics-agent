import unittest

from src.tools import run_sql_query


class TestSQLSafety(unittest.TestCase):

    def test_valid_select_query(self):
        """A valid SELECT query should execute successfully."""
        result = run_sql_query(
            "SELECT department, COUNT(*) AS employee_count "
            "FROM employees GROUP BY department"
        )

        self.assertIn("department", result["columns"])
        self.assertIn("employee_count", result["columns"])
        self.assertEqual(len(result["rows"]), 5)

    def test_unknown_column_rejected(self):
        """Queries using columns outside the approved schema must fail."""
        with self.assertRaisesRegex(ValueError, "not allowed"):
            run_sql_query("SELECT salary FROM employees")

    def test_delete_query_rejected(self):
        """Write operations must not be permitted."""
        with self.assertRaisesRegex(ValueError, "Only SELECT"):
            run_sql_query("DELETE FROM employees")

    def test_multiple_statements_rejected(self):
        """Multiple SQL statements must be rejected."""
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            run_sql_query(
                "SELECT employee_id FROM employees; "
                "DELETE FROM employees"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)