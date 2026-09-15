from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:  # pragma: no cover - optional integration dependency
    gspread = None
    Credentials = None


DB_PATH = Path("data/course_outcome_agent.db")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "").strip()


def get_google_sheet_client():
    if gspread is None or Credentials is None:
        return None
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
    if not sheet_id:
        return None
    if service_account_json:
        try:
            import json as _json

            creds = Credentials.from_service_account_info(
                _json.loads(service_account_json),
                scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive",
                ],
            )
            return gspread.authorize(creds)
        except Exception:
            return None
    if service_account_file and Path(service_account_file).exists():
        try:
            creds = Credentials.from_service_account_file(
                service_account_file,
                scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive",
                ],
            )
            return gspread.authorize(creds)
        except Exception:
            return None
    return None


def ensure_google_sheet(sheet_name: str, headers: list[str] | None = None) -> Any | None:
    client = get_google_sheet_client()
    if client is None:
        return None
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    if not sheet_id:
        return None
    try:
        workbook = client.open_by_key(sheet_id)
        existing = [ws.title for ws in workbook.worksheets()]
        if sheet_name in existing:
            worksheet = workbook.worksheet(sheet_name)
        else:
            worksheet = workbook.add_worksheet(title=sheet_name, rows=1000, cols=20)
        if headers and not worksheet.get_all_values():
            worksheet.append_row(headers, value_input_option="RAW")
        return worksheet
    except Exception:
        return None


def get_google_sheet_rows(sheet_name: str) -> list[list[str]]:
    client = get_google_sheet_client()
    if client is None:
        return []
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    if not sheet_id:
        return []
    try:
        workbook = client.open_by_key(sheet_id)
        worksheet = workbook.worksheet(sheet_name)
        return worksheet.get_all_values()
    except Exception:
        return []


def append_google_sheet_rows(sheet_name: str, rows: list[list[Any]]) -> bool:
    if not rows:
        return False
    worksheet = ensure_google_sheet(sheet_name)
    if worksheet is None:
        return False
    try:
        worksheet.append_rows(rows, value_input_option="RAW")
        return True
    except Exception:
        return False


def sync_google_user_login(username: str, role: str) -> bool:
    if not username or not role:
        return False
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row = [timestamp, str(username).strip(), str(role).strip(), "login"]
    return append_google_sheet_rows("Users", [row])


def sync_google_uploaded_dataset(dataset_key: str, file_name: str, df: pd.DataFrame) -> bool:
    if df is None:
        return False
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row = [timestamp, str(dataset_key), str(file_name), int(len(df)), json.dumps(list(df.columns))]
    return append_google_sheet_rows("UploadedData", [row])


def sync_google_run(run_id: int, course_code: str, course_name: str, report_type: str, audience: str, llm_report: str) -> bool:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row = [timestamp, int(run_id), str(course_code), str(course_name), str(report_type), str(audience), llm_report[:2000]]
    return append_google_sheet_rows("Reports", [row])


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
    sync_google_uploaded_dataset(dataset_key, file_name, df)
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
        run_id = int(cursor.lastrowid)
    sync_google_run(run_id, course_code, course_name, report_type, audience, llm_report)
    return run_id


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
