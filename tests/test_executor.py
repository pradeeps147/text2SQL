import json

import duckdb
import pytest

from src.nlq.executor import answer_question


class FakeClient:
    """Stand-in for OllamaClient that returns pre-scripted JSON responses,
    so tests don't require a running Ollama daemon."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.calls: list[dict] = []

    def generate(self, system, user, json_schema=None, temperature=0.0):
        self.calls.append({"system": system, "user": user})
        return self._responses[len(self.calls) - 1]


@pytest.fixture()
def sample_db(tmp_path):
    db_path = tmp_path / "test_shipments.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE shipments (
            shipment_id VARCHAR,
            route_id VARCHAR,
            origin VARCHAR,
            destination VARCHAR,
            carrier VARCHAR,
            ship_date DATE,
            expected_delivery_date DATE,
            actual_delivery_date DATE,
            status VARCHAR,
            delay_days INTEGER,
            is_delayed BOOLEAN,
            source_file VARCHAR,
            ingested_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        INSERT INTO shipments VALUES
        ('SHP1', 'NEW-HOU', 'New York', 'Houston', 'Swift Freight',
         '2025-06-01', '2025-06-05', '2025-06-07', 'DELAYED', 2, true,
         'a.csv', now()),
        ('SHP2', 'NEW-HOU', 'New York', 'Houston', 'Swift Freight',
         '2025-06-02', '2025-06-06', '2025-06-06', 'DELIVERED', 0, false,
         'a.csv', now())
        """
    )
    con.close()
    return db_path


def test_valid_sql_succeeds_on_first_try(sample_db):
    response = json.dumps(
        {"sql": "SELECT route_id, COUNT(*) AS cnt FROM shipments GROUP BY route_id", "explanation": "counts by route"}
    )
    client = FakeClient([response])

    result = answer_question("How many shipments per route?", sample_db, client=client)

    assert result.succeeded
    assert result.dataframe is not None
    assert len(result.attempts) == 1
    assert client.calls[0]["system"].count("shipments") >= 1


def test_disallowed_table_triggers_retry_then_succeeds(sample_db):
    bad_response = json.dumps({"sql": "SELECT * FROM information_schema.tables", "explanation": "oops"})
    good_response = json.dumps({"sql": "SELECT * FROM shipments", "explanation": "fixed"})
    client = FakeClient([bad_response, good_response])

    result = answer_question("Show me everything", sample_db, client=client)

    assert result.succeeded
    assert len(result.attempts) == 2
    assert result.attempts[0]["error"] is not None
    assert result.attempts[1]["error"] is None
    # second call's user prompt should include the guardrail error as feedback
    assert "not allowed" in client.calls[1]["user"].lower()


def test_model_declines_returns_explanation(sample_db):
    response = json.dumps({"sql": "", "explanation": "The schema has no pricing data to answer this."})
    client = FakeClient([response])

    result = answer_question("What was the total revenue?", sample_db, client=client)

    assert not result.succeeded
    assert "pricing" in result.error.lower()


def test_execution_error_retries_with_feedback(sample_db):
    # First query references a real column name but with a typo in a function
    # that will fail at execution time (not caught by static validation).
    bad_response = json.dumps({"sql": "SELECT NOT_A_REAL_FUNC(origin) FROM shipments", "explanation": "x"})
    good_response = json.dumps({"sql": "SELECT origin FROM shipments", "explanation": "fixed"})
    client = FakeClient([bad_response, good_response])

    result = answer_question("weird question", sample_db, client=client)

    assert result.succeeded
    assert len(result.attempts) == 2
    assert result.attempts[0]["error"] is not None


def test_successful_query_is_synthesized_into_a_natural_answer(sample_db):
    sql_response = json.dumps(
        {
            "sql": "SELECT ROUND(AVG(CASE WHEN is_delayed THEN 1 ELSE 0 END) * 100, 1) AS delay_rate_pct FROM shipments",
            "explanation": "Calculates the delayed share of shipments.",
        }
    )
    answer_response = json.dumps({"answer": "The shipment delay rate is 50%."})
    client = FakeClient([sql_response, answer_response])

    result = answer_question("What is the shipment delay rate?", sample_db, client=client)

    assert result.succeeded
    assert result.answer == "The shipment delay rate is 50%."
    assert result.sql_explanation == "Calculates the delayed share of shipments."
    assert len(client.calls) == 2
    assert "Result rows" in client.calls[1]["user"]
    assert result.retrieved_context
