from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DB_PATH = Path("data/course_outcome_agent.db")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                course_code TEXT,
                course_name TEXT,
                report_type TEXT NOT NULL,
                audience TEXT NOT NULL,
                model TEXT NOT NULL,
                config_json TEXT NOT NULL,
                uploaded_row_counts_json TEXT NOT NULL,
                co_attainment_json TEXT NOT NULL,
                outcome_attainment_json TEXT NOT NULL,
                gap_analysis_json TEXT NOT NULL,
                diagnostics_json TEXT NOT NULL,
                llm_report TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS uploaded_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                dataset_key TEXT NOT NULL,
                file_name TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                columns_json TEXT NOT NULL,
                data_json TEXT NOT NULL
            );
            """
        )


def _records(df: pd.DataFrame) -> str:
    return df.to_json(orient="records")


def save_uploaded_dataset(dataset_key: str, file_name: str, df: pd.DataFrame) -> int:
    init_db()
    if df is None:
        raise ValueError("Dataset cannot be None.")
    with connect() as connection:
        connection.execute("DELETE FROM uploaded_data WHERE dataset_key = ?", (str(dataset_key),))
        cursor = connection.execute(
            """
            INSERT INTO uploaded_data (
                created_at,
                dataset_key,
                file_name,
                row_count,
                columns_json,
                data_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                str(dataset_key),
                str(file_name),
                int(len(df)),
                json.dumps(list(df.columns)),
                _records(df),
            ),
        )
        return int(cursor.lastrowid)


def list_uploaded_datasets(limit: int = 10) -> pd.DataFrame:
    init_db()
    with connect() as connection:
        return pd.read_sql_query(
            """
            SELECT id, created_at, dataset_key, file_name, row_count
            FROM uploaded_data
            ORDER BY id DESC
            LIMIT ?
            """,
            connection,
            params=(limit,),
        )


def get_uploaded_dataset(dataset_id: int) -> dict[str, Any] | None:
    init_db()
    with connect() as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM uploaded_data WHERE id = ?", (dataset_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["columns"] = json.loads(data["columns_json"])
    data["data"] = pd.read_json(data["data_json"], orient="records")
    return data


def save_run(
    *,
    course_code: str,
    course_name: str,
    report_type: str,
    audience: str,
    model: str,
    config: dict[str, Any],
    uploaded_row_counts: dict[str, int],
    co_attainment: pd.DataFrame,
    outcome_attainment: pd.DataFrame,
    gap_analysis: pd.DataFrame,
    diagnostics: pd.DataFrame,
    llm_report: str,
) -> int:
    init_db()
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO runs (
                created_at,
                course_code,
                course_name,
                report_type,
                audience,
                model,
                config_json,
                uploaded_row_counts_json,
                co_attainment_json,
                outcome_attainment_json,
                gap_analysis_json,
                diagnostics_json,
                llm_report
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                course_code,
                course_name,
                report_type,
                audience,
                model,
                json.dumps(config),
                json.dumps(uploaded_row_counts),
                _records(co_attainment),
                _records(outcome_attainment),
                _records(gap_analysis),
                _records(diagnostics),
                llm_report,
            ),
        )
        return int(cursor.lastrowid)


def list_runs(limit: int = 25) -> pd.DataFrame:
    init_db()
    with connect() as connection:
        return pd.read_sql_query(
            """
            SELECT
                id,
                created_at,
                course_code,
                course_name,
                report_type,
                audience,
                model
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            connection,
            params=(limit,),
        )


def delete_run(run_id: int) -> bool:
    init_db()
    with connect() as connection:
        cursor = connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cursor.rowcount > 0


def get_run(run_id: int) -> dict[str, Any] | None:
    init_db()
    with connect() as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    for key in [
        "config_json",
        "uploaded_row_counts_json",
        "co_attainment_json",
        "outcome_attainment_json",
        "gap_analysis_json",
        "diagnostics_json",
    ]:
        data[key] = json.loads(data[key])
    return data
