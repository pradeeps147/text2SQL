from src.nlq.retrieval import describe_rag_pipeline, retrieve_context


def test_delay_route_question_retrieves_relevant_business_context():
    context = retrieve_context("Which route had the highest delay rate last quarter?")
    ids = {item.document_id for item in context}

    assert "metric.delay" in ids
    assert "dimension.route" in ids
    assert "time.relative" in ids


def test_rag_description_exposes_the_full_grounding_flow():
    description = describe_rag_pipeline()

    assert description["knowledge_documents"] >= 1
    assert any("DuckDB" in step for step in description["steps"])
    assert any("plain-language" in step for step in description["steps"])
