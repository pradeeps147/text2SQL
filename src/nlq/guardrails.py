"""Defensive validation/sandboxing for LLM-generated SQL before it ever
touches DuckDB.

Strategy is allowlist-first (safer than trying to blocklist every dangerous
keyword):
  1. Exactly one statement (blocks `SELECT ...; DROP TABLE ...`).
  2. The statement must parse as a SELECT/UNION (blocks INSERT/UPDATE/DELETE/
     DROP/ALTER/CREATE/ATTACH/PRAGMA/COPY/etc. outright, since those parse to
     a different node type or fail to parse as a Select).
  3. Every referenced table must be in an explicit allowlist (blocks reads
     from information_schema, duckdb_* system views, read_csv()/read_parquet()
     table functions pointed at the filesystem, etc.).
  4. Every referenced column must be in an explicit allowlist (defense in
     depth even though there's currently one table).
  5. A LIMIT clause is enforced (added if missing, capped if too large).
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

DEFAULT_MAX_ROWS = 1000


@dataclass
class ValidationResult:
    is_valid: bool
    sql: str = ""
    error: str = ""


def validate_sql(
    raw_sql: str,
    allowed_tables: set[str],
    allowed_columns: set[str],
    max_rows: int = DEFAULT_MAX_ROWS,
    dialect: str = "duckdb",
) -> ValidationResult:
    raw_sql = (raw_sql or "").strip()
    if not raw_sql:
        return ValidationResult(is_valid=False, error="Empty SQL.")

    try:
        statements = [s for s in sqlglot.parse(raw_sql, dialect=dialect) if s is not None]
    except Exception as exc:
        return ValidationResult(is_valid=False, error=f"SQL failed to parse: {exc}")

    if len(statements) != 1:
        return ValidationResult(
            is_valid=False,
            error=f"Expected exactly one SQL statement, found {len(statements)}.",
        )

    root = statements[0]

    if not isinstance(root, (exp.Select, exp.Union)):
        return ValidationResult(
            is_valid=False,
            error=f"Only SELECT statements are allowed; got {type(root).__name__}.",
        )

    allowed_tables_lower = {t.lower() for t in allowed_tables}
    for table in root.find_all(exp.Table):
        table_name = (table.name or "").lower()
        if not table_name:
            return ValidationResult(
                is_valid=False,
                error="Table-valued function calls (e.g. read_csv(), duckdb_tables()) are not allowed in FROM/JOIN.",
            )
        if table_name not in allowed_tables_lower:
            return ValidationResult(
                is_valid=False,
                error=f"Table '{table.name}' is not allowed. Allowed tables: {sorted(allowed_tables_lower)}",
            )

    # Aliases defined in the SELECT list (e.g. `COUNT(*) AS cnt`) are legal to
    # reference in ORDER BY / HAVING even though they aren't real columns.
    output_aliases = {
        alias.output_name.lower()
        for alias in root.find_all(exp.Alias)
        if alias.output_name
    }

    allowed_columns_lower = {c.lower() for c in allowed_columns} | output_aliases
    for column in root.find_all(exp.Column):
        col_name = (column.name or "").lower()
        if col_name and col_name not in allowed_columns_lower:
            return ValidationResult(
                is_valid=False,
                error=f"Column '{column.name}' is not allowed. Allowed columns: {sorted(allowed_columns_lower)}",
            )

    root = _enforce_limit(root, max_rows)

    return ValidationResult(is_valid=True, sql=root.sql(dialect=dialect))


def _enforce_limit(root: exp.Expression, max_rows: int) -> exp.Expression:
    target = root
    # For UNION, the LIMIT applies to the whole union expression in sqlglot.
    existing_limit = target.args.get("limit")
    if existing_limit is None:
        return target.limit(max_rows)

    try:
        current = int(existing_limit.expression.this)
    except (AttributeError, ValueError, TypeError):
        return target.limit(max_rows)

    if current > max_rows:
        return target.limit(max_rows)
    return target
