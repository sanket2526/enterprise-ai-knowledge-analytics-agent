import sqlite3
import re
import os
import sqlglot

from pathlib import Path
from dotenv import load_dotenv
from google import genai
from sqlglot import exp

import pandas as pd


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.5-flash-lite"

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "employee_data.csv"
DB_PATH = PROJECT_ROOT / "data" / "employee_analytics.db"


def initialize_database():
    """Load employee CSV data into a local SQLite database."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Employee data not found: {CSV_PATH}")

    # Read the employee CSV
    df = pd.read_csv(CSV_PATH)

    # Connect to SQLite and create the employees table
    with sqlite3.connect(DB_PATH) as connection:
        df.to_sql(
            "employees",
            connection,
            if_exists="replace",
            index=False
        )

    print(f"Database created: {DB_PATH}")
    print(f"Employee records loaded: {len(df)}")


def generate_sql(question):
    """Convert a business question into a SQL SELECT query."""

    prompt = f"""
You generate SQLite queries for employee analytics.

The only available table is:

employees(
    employee_id TEXT,
    department TEXT,
    location TEXT,
    joining_date TEXT,
    employment_type TEXT,
    manager_id TEXT
)

Rules:
- Return only one SQLite SELECT statement.
- Use only the employees table and its listed columns.
- Do not use JOINs, subqueries, CTEs, or multiple statements.
- Do not use INSERT, UPDATE, DELETE, DROP, or other write operations.
- For joining-date filters, use ISO dates in YYYY-MM-DD format.
- Do not include Markdown fences or explanations.
- If the question cannot be answered from these columns,
  return INVALID.

Business question: {question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    sql = response.text.strip()

    # Remove Markdown fences if the model includes them
    if sql.startswith("```"):
        sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s*```$", "", sql)

    if sql.strip().upper() == "INVALID":
        raise ValueError(
            "This question cannot be answered using the employee dataset."
        )

    return sql.strip()




def run_sql_query(query):
    """Validate and execute a restricted, read-only SQL query."""

    allowed_columns = {
        "employee_id",
        "department",
        "location",
        "joining_date",
        "employment_type",
        "manager_id",
    }

    # Parse the SQL using SQLite syntax
    try:
        statements = sqlglot.parse(query, read="sqlite")
    except sqlglot.errors.ParseError as error:
        raise ValueError("Invalid SQL syntax.") from error

    # Permit exactly one non-empty statement
    statements = [statement for statement in statements if statement is not None]

    if len(statements) != 1:
        raise ValueError("Exactly one SQL statement is allowed.")

    tree = statements[0]

    # Permit only SELECT statements
    if not isinstance(tree, exp.Select):
        raise ValueError("Only SELECT queries are allowed.")

    # Disallow subqueries and joins for this prototype
    if list(tree.find_all(exp.Subquery)):
        raise ValueError("Subqueries are not allowed.")

    if list(tree.find_all(exp.Join)):
        raise ValueError("JOIN operations are not allowed.")

    # Check all referenced tables
    tables = list(tree.find_all(exp.Table))

    if not tables or any(
        table.name.lower() != "employees"
        or table.db
        or table.catalog
        for table in tables
    ):
        raise ValueError("Only the employees table can be queried.")

    
    # Collect aliases defined in the SELECT clause
    allowed_aliases = {
        expression.alias.lower()
        for expression in tree.expressions
        if expression.alias
    }

    # Validate column references and permit SELECT aliases
    for column in tree.find_all(exp.Column):
        if column.is_star:
            continue

        if column.table and column.table.lower() != "employees":
            raise ValueError("Unexpected table qualifier.")

        column_name = column.name.lower()

        if (
            column_name not in allowed_columns
            and column_name not in allowed_aliases
        ):
            raise ValueError(
                f"Column '{column.name}' is not allowed."
            )


    # Open the existing database in read-only mode
    database_uri = f"{DB_PATH.as_uri()}?mode=ro"

    with sqlite3.connect(database_uri, uri=True) as connection:
        connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 5000)

        cursor = connection.execute(tree.sql(dialect="sqlite"))

        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

    return {
        "columns": columns,
        "rows": rows
    }




def employee_analytics(question):
    """Answer an employee analytics question using SQL."""

    try:
        # Step 1: Convert the question into SQL
        sql = generate_sql(question)

        print(f"\nGenerated SQL: {sql}")

        # Step 2: Validate and execute the query
        result = run_sql_query(sql)

        # Step 3: Return structured results
        return {
            "success": True,
            "question": question,
            "sql": sql,
            "columns": result["columns"],
            "rows": result["rows"],
            "error": None
        }

    except Exception as error:
        return {
            "success": False,
            "question": question,
            "sql": None,
            "columns": [],
            "rows": [],
            "error": str(error)
        }



if __name__ == "__main__":
    initialize_database()

    
    # Valid query: should return department counts
    result = run_sql_query(
        "SELECT department, COUNT(*) AS total "
        "FROM employees GROUP BY department"
    )
    print("\nValid query:", result["rows"])

    # Invalid query: should reject an unknown column
    try:
        run_sql_query("SELECT salary FROM employees")
    except ValueError as error:
        print("Unknown column rejected:", error)

    # Invalid query: should reject a write operation
    try:
        run_sql_query("DELETE FROM employees")
    except ValueError as error:
        print("Write operation rejected:", error)


    # Test a valid query
    result = run_sql_query(
        "SELECT department, COUNT(*) AS total "
        "FROM employees GROUP BY department"
    )
    print("\nValid query:", result["rows"])

    # Test a rejected write operation
    try:
        run_sql_query("DELETE FROM employees")
    except ValueError as error:
        print("Unsafe query rejected:", error)


    questions = [
        "Which department has the most employees, and how many?",
        "How many employees joined in 2025?",
        "Which locations have the most employees?"
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print("Question:", question)

        result = employee_analytics(question)

        if result["success"]:
            print("Columns:", result["columns"])
            print("Results:")

            for row in result["rows"]:
                print(row)
        else:
            print("Error:", result["error"])
