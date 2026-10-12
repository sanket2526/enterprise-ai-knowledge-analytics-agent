# Enterprise AI Knowledge & Analytics Agent

An AI-powered agent that answers company policy questions using Retrieval-Augmented Generation (RAG) and analyzes employee data using natural-language-to-SQL. The project demonstrates document retrieval, LLM-based reasoning, tool integration, SQL validation, and automated testing.

## Features

- **Policy Q&A:** Retrieves relevant passages from company policy documents using semantic similarity search and generates context-grounded answers.
- **Employee analytics:** Converts natural-language questions into SQL queries and executes them against a local SQLite database.
- **Intent routing:** Classifies questions and directs them to policy retrieval, employee analytics, or an appropriate unsupported-request response.
- **SQL validation:** Restricts analytics queries to approved `SELECT` statements, the `employees` table, and approved columns.
- **Source reporting:** Identifies the policy document or analytics database used to answer a question.
- **Automated tests:** Tests agent routing, unsupported policy handling, valid SQL queries, and rejection of disallowed SQL operations.

## Architecture

```text
                  User Question
                        |
                        v
                 Intent Classifier
                        |
            +-----------+-----------+
            |           |           |
            v           v           v
          Policy     Analytics    Unsupported
            |           |           |
            v           v           v
       Semantic      LLM SQL     Capability
       Retrieval    Generation    Guidance
            |           |
            v           v
       Policy Text   SQL Validation
            |           |
            v           v
       Gemini Answer  SQLite Database
            |           |
            +-----+-----+
                  |
                  v
           Answer + Sources
```

## Technology Stack

- **Python** — application logic and orchestration
- **Google Gemini API** — intent classification, answer generation, and SQL generation
- **Sentence Transformers** — semantic embeddings using `all-MiniLM-L6-v2`
- **SQLite** — local employee analytics database
- **Pandas** — employee data loading and database initialization
- **SQLGlot** — SQL parsing and validation
- **Unittest** — automated testing

## Project Structure

```text
enterprise-ai-agent/
├── data/
│   ├── company_policies/
│   ├── employee_data.csv
│   └── employee_analytics.db
├── notebooks/
│   ├── embeddings.ipynb
│   └── rag_experiments.ipynb
├── src/
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── rag.py
│   ├── tools.py
│   ├── agent.py
│   ├── evaluate.py
│   ├── test_agent.py
│   └── test_tools.py
├── requirements.txt
├── .env
└── README.md
```

## Setup and Execution

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Gemini API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

Use your own valid API key. Do not commit this file or share the key publicly.

### 4. Initialize the employee database

```bash
python -m src.tools
```

This initializes the SQLite database from `data/employee_data.csv` and runs the tool's demonstration queries.

### 5. Start the agent

```bash
python -m src.agent
```

Ask questions interactively. Type `exit` to quit.

### 6. Run automated tests

```bash
python -m unittest discover -s src -p "test_*.py" -v
```

## Example Questions

**Policy questions**
- How many annual leave days do employees receive?
- How many sick leave days do employees receive?
- How many remote working days are permitted?

**Employee analytics**
- Which department has the most employees?
- How many employees joined in 2025?
- Which locations have the most employees?

The agent should acknowledge when information is unavailable or a request falls outside its current capabilities.

## Evaluation

The current development checks include:

- Five hand-written policy retrieval evaluation cases.
- Seven automated unit tests for routing, fallback behavior, and SQL validation.
- Manual end-to-end checks for policy questions, unsupported policy questions, employee counts, department headcounts, and tied location counts.

## Limitations

- The policy documents and employee dataset are synthetic examples, not real company data.
- Analytics accuracy depends on the generated SQL and the underlying dataset.
- SQL validation provides useful restrictions but is not a complete production-grade security boundary.
- The agent does not currently perform employee attrition prediction.
- Answers depend on the available policy documents, database schema, embedding model, and Gemini responses.
- The application is a local prototype; authentication, authorization, deployment, monitoring, and production-grade access controls are outside its current scope.

