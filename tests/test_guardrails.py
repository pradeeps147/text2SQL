from src.nlq.guardrails import validate_sql

ALLOWED_TABLES = {"shipments"}
ALLOWED_COLUMNS = {
    "shipment_id",
    "route_id",
    "origin",
    "destination",
    "carrier",
    "ship_date",
    "expected_delivery_date",
    "actual_delivery_date",
    "status",
    "delay_days",
    "is_delayed",
    "source_file",
    "ingested_at",
}


def _validate(sql: str):
    return validate_sql(sql, ALLOWED_TABLES, ALLOWED_COLUMNS, max_rows=1000)


class TestValidSql:
    def test_simple_select_passes(self):
        result = _validate("SELECT origin, destination FROM shipments LIMIT 10")
        assert result.is_valid is True
        assert "shipments" in result.sql.lower()

    def test_aggregate_group_by_passes(self):
        result = _validate(
            "SELECT route_id, COUNT(*) AS cnt FROM shipments GROUP BY route_id ORDER BY cnt DESC"
        )
        assert result.is_valid is True

    def test_missing_limit_is_added(self):
        result = _validate("SELECT * FROM shipments")
        assert result.is_valid is True
        assert "LIMIT 1000" in result.sql.upper()

    def test_oversized_limit_is_capped(self):
        result = _validate("SELECT * FROM shipments LIMIT 999999")
        assert result.is_valid is True
        assert "LIMIT 1000" in result.sql.upper()

    def test_subquery_and_union_pass(self):
        result = _validate(
            "SELECT origin FROM shipments WHERE ship_date > (SELECT MAX(ship_date) FROM shipments) "
            "UNION SELECT destination FROM shipments LIMIT 5"
        )
        assert result.is_valid is True


class TestRejectedSql:
    def test_multi_statement_injection_rejected(self):
        result = _validate("SELECT * FROM shipments; DROP TABLE shipments;")
        assert result.is_valid is False
        assert "one sql statement" in result.error.lower()

    def test_drop_table_alone_rejected(self):
        result = _validate("DROP TABLE shipments")
        assert result.is_valid is False

    def test_insert_rejected(self):
        result = _validate("INSERT INTO shipments (shipment_id) VALUES ('X')")
        assert result.is_valid is False

    def test_update_rejected(self):
        result = _validate("UPDATE shipments SET status = 'DELIVERED'")
        assert result.is_valid is False

    def test_delete_rejected(self):
        result = _validate("DELETE FROM shipments")
        assert result.is_valid is False

    def test_pragma_rejected(self):
        result = _validate("PRAGMA database_list")
        assert result.is_valid is False

    def test_attach_rejected(self):
        result = _validate("ATTACH 'evil.db' AS evil")
        assert result.is_valid is False

    def test_information_schema_table_rejected(self):
        result = _validate("SELECT table_name FROM information_schema.tables")
        assert result.is_valid is False
        assert "not allowed" in result.error.lower()

    def test_duckdb_system_table_function_rejected(self):
        result = _validate("SELECT * FROM duckdb_tables()")
        assert result.is_valid is False

    def test_read_csv_filesystem_access_rejected(self):
        result = _validate("SELECT * FROM read_csv('/etc/passwd')")
        assert result.is_valid is False

    def test_disallowed_column_rejected(self):
        result = _validate("SELECT ssn FROM shipments")
        assert result.is_valid is False
        assert "column" in result.error.lower()

    def test_empty_sql_rejected(self):
        result = _validate("")
        assert result.is_valid is False

    def test_unparseable_sql_rejected(self):
        result = _validate("SELEC * FORM shipments")
        assert result.is_valid is False
