import importlib

import pandas as pd
from fastapi.testclient import TestClient

from src.api.app import app
from src.nlq.executor import QueryResult


client = TestClient(app)
api_module = importlib.import_module("src.api.app")


def test_health_and_rag_endpoints():
    health = client.get("/api/v1/health")
    rag = client.get("/api/v1/rag")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert rag.status_code == 200
    assert len(rag.json()["steps"]) >= 5


def test_log_monitor_endpoint_returns_structured_records():
    client.get("/api/v1/health")
    response = client.get("/api/v1/logs?limit=20")

    assert response.status_code == 200
    assert response.json()["count"] >= 1
    assert {"timestamp", "level", "message"} <= set(response.json()["records"][0])


def test_multi_file_upload_endpoint_forwards_complete_batch(monkeypatch):
    captured = {}

    def fake_process(payloads):
        captured["names"] = [payload.filename for payload in payloads]
        return {"files_loaded": captured["names"], "accepted_rows": 2}

    monkeypatch.setattr(api_module, "process_uploaded_csvs", fake_process)
    response = client.post(
        "/api/v1/datasets/upload",
        files=[
            ("files", ("a.csv", b"shipment_id\nS1\n", "text/csv")),
            ("files", ("b.csv", b"shipment_id\nS2\n", "text/csv")),
        ],
    )

    assert response.status_code == 200
    assert captured["names"] == ["a.csv", "b.csv"]


def test_chat_endpoint_returns_answer_sql_rows_and_rag_evidence(tmp_path, monkeypatch):
    active_db = tmp_path / "active.duckdb"
    active_db.touch()

    class FakeOllamaClient:
        def __init__(self, model):
            self.model = model

        def check_connection(self):
            return None

    result = QueryResult(
        question="How many?",
        sql="SELECT COUNT(*) AS shipment_count FROM shipments LIMIT 1000",
        dataframe=pd.DataFrame({"shipment_count": [2]}),
        answer="There are 2 shipments.",
        sql_explanation="Counts active shipments.",
        retrieved_context=[{"document_id": "metric.delay", "title": "Metrics", "content": "x", "score": 1.0}],
        attempts=[{"attempt": 1, "sql": "SELECT COUNT(*) FROM shipments", "error": None}],
    )
    monkeypatch.setattr(api_module, "DEFAULT_DB_PATH", active_db)
    monkeypatch.setattr(api_module, "build_llm_client", lambda model: FakeOllamaClient(model))
    monkeypatch.setattr(api_module, "answer_question", lambda *args, **kwargs: result)

    response = client.post("/api/v1/chat", json={"question": "How many shipments?", "model": "test"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "There are 2 shipments."
    assert payload["rows"] == [{"shipment_count": 2}]
    assert payload["retrieved_context"][0]["document_id"] == "metric.delay"
    assert payload["request_id"] == response.headers["X-Request-ID"]


def test_admin_key_protects_upload_and_logs(monkeypatch):
    monkeypatch.setattr(api_module, "ADMIN_KEY", "top-secret")

    assert client.get("/api/v1/logs").status_code == 401
    assert client.get("/api/v1/logs", headers={"X-Tenarai-Admin-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/logs", headers={"X-Tenarai-Admin-Key": "top-secret"}).status_code == 200
