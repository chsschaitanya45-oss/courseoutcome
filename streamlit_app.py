from __future__ import annotations

import hashlib
import os
from dataclasses import asdict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

from attainment import (
    AttainmentConfig,
    build_gap_analysis,
    compute_co_attainment,
    compute_outcome_attainment,
    compute_question_diagnostics,
    create_excel_report,
    normalize_columns,
    report_context,
    validate_required_columns,
)
from database import DB_PATH, delete_run, get_run, init_db, list_runs, save_run


REQUIRED_COLUMNS = {
    "courses": ["co", "co_statement"],
    "mapping": ["co", "outcome_type", "outcome", "strength"],
    "marks": ["student_id", "assessment", "question", "co", "max_marks", "marks_obtained"],
    "indirect": ["co", "survey_percent"],
}


def get_allowed_roles() -> list[str]:
    return ["Faculty", "Head of department", "NBA coordinator", "IQAC"]


def role_can_access_feature(role: str, feature: str) -> bool:
    role_name = (role or "").strip()
    feature_name = (feature or "").strip().lower()
    access_map = {
        "Faculty": {"data_upload", "dashboard_view", "report_generation", "history_view"},
        "Head of department": {"dashboard_view", "report_generation", "history_view"},
        "NBA coordinator": {"dashboard_view", "report_generation", "history_view"},
        "IQAC": {"dashboard_view", "report_generation", "history_view"},
    }
    return feature_name in access_map.get(role_name, set())


def get_role_credentials_map() -> dict[str, tuple[str, str]]:
    return {
        "Faculty": ("AUTH_FACULTY_USERNAME", "AUTH_FACULTY_PASSWORD"),
        "Head of department": ("AUTH_HOD_USERNAME", "AUTH_HOD_PASSWORD"),
        "NBA coordinator": ("AUTH_NBA_USERNAME", "AUTH_NBA_PASSWORD"),
        "IQAC": ("AUTH_IQAC_USERNAME", "AUTH_IQAC_PASSWORD"),
    }


def hash_password(password: str) -> str:
    return hashlib.sha256(str(password).strip().encode("utf-8")).hexdigest()


def validate_role_credentials(username: str, password: str) -> bool:
    if not username or not password:
        return False
    normalized_username = str(username).strip().lower()
    normalized_password = str(password).strip()
    for role, (username_key, password_key) in get_role_credentials_map().items():
        env_user = os.getenv(username_key, "").strip().lower()
        env_pass = os.getenv(password_key, "").strip()
        if env_user and env_pass:
            expected_hash = hash_password(env_pass)
            if normalized_username == env_user and hash_password(normalized_password) == expected_hash:
                return True
            if normalized_username == env_user and normalized_password == env_pass:
                return True
    return False


def infer_role_by_username(username: str) -> str:
    normalized_username = str(username).strip().lower()
    for role, (username_key, _) in get_role_credentials_map().items():
        if os.getenv(username_key, "").strip().lower() == normalized_username:
            return role
    return "Unauthorized"


def require_authentication() -> None:
    if st.session_state.get("authenticated_user"):
        return

    st.title("Access required", icon=":material/lock:")
    st.caption("Only authorized Faculty, HOD, NBA coordinator, and IQAC users can access this data.")

    with st.form("role_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if validate_role_credentials(username, password):
                st.session_state.authenticated_user = username
                st.session_state.authenticated_role = infer_role_by_username(username)
                st.success(f"Access granted for {st.session_state.authenticated_role}.")
                st.rerun()
            else:
                st.error("Invalid username or password. Only approved institutional roles can access this application.")
    st.stop()


def logout_current_user() -> None:
    st.session_state.authenticated_user = ""
    st.session_state.authenticated_role = ""
    st.rerun()


def filter_runs_for_role(runs: pd.DataFrame, role: str) -> pd.DataFrame:
    if runs is None or runs.empty:
        return runs.copy()
    if "audience" not in runs.columns:
        return runs.copy()
    target = (role or "").strip()
    if target == "Faculty":
        return runs[runs["audience"].str.contains("Faculty", case=False, na=False)].copy()
    if target == "Head of department":
        return runs[runs["audience"].str.contains("Head of department|HOD", case=False, na=False)].copy()
    if target == "NBA coordinator":
        return runs[runs["audience"].str.contains("NBA|coordinator", case=False, na=False)].copy()
    if target == "IQAC":
        return runs[runs["audience"].str.contains("IQAC", case=False, na=False)].copy()
    return runs.head(0).copy()


def get_selected_history_id(runs: pd.DataFrame, selected_id: int | None = None) -> int | None:
    if selected_id is not None:
        return int(selected_id)
    if runs is None or runs.empty:
        return None
    return int(runs.iloc[0]["id"]) if "id" in runs.columns else None


def init_state() -> None:
    for key in ["courses", "mapping", "marks", "indirect"]:
        if key not in st.session_state:
            st.session_state[key] = None
    if "generated_report" not in st.session_state:
        st.session_state.generated_report = ""
    if "authenticated_user" not in st.session_state:
        st.session_state.authenticated_user = ""
    if "authenticated_role" not in st.session_state:
        st.session_state.authenticated_role = ""


def load_csv_or_excel(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    if uploaded_file.name.lower().endswith(".csv"):
        return normalize_columns(pd.read_csv(uploaded_file))
    return normalize_columns(pd.read_excel(uploaded_file))


def uploaded_data_ready() -> bool:
    return all(st.session_state[key] is not None for key in REQUIRED_COLUMNS)


def validate_inputs() -> list[str]:
    errors = []
    labels = {
        "courses": "CO statements",
        "mapping": "CO-PO/PSO mapping",
        "marks": "Question-wise marks",
        "indirect": "Indirect survey",
    }
    for key, columns in REQUIRED_COLUMNS.items():
        data = st.session_state[key]
        if data is None:
            errors.append(f"{labels[key]} file has not been uploaded.")
        else:
            errors.extend(validate_required_columns(data, columns, labels[key]))
            if data.empty:
                errors.append(f"{labels[key]} file has no records.")
    return errors


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No attainment gaps were detected."
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def ai_report(context: str, audience: str, report_type: str, model: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not available in .env.local.")

    report_type_clean = (report_type or "Action taken report").strip()
    report_type_lower = report_type_clean.lower()

    if "gap" in report_type_lower:
        report_instructions = """
        - Explain the difference between expected/target attainment and actual attainment.
        - Identify what was below target, how far below it was, and why the gap exists.
        - Explain the root cause using direct assessment evidence, student performance, and question-level trends.
        - State what action should be taken to close the gap.
        - Begin with the course metadata: Course Code, Course Name, Department, Semester, Academic Year, Faculty, Number of Students, and Target Attainment Level.
        - Include register numbers or student IDs for below-target students.
        - Add a clear HOD verification statement for all students below target.
        """
        required_sections = """
        1. Course metadata
        2. Executive summary
        3. Gap analysis by CO
        4. Expected/target attainment vs actual attainment
        5. Why the gap exists
        6. What action should be taken
        7. HOD verification requirement
        """
    elif "action" in report_type_lower:
        report_instructions = """
        - This is the follow-up to the Gap Analysis Report.
        - Explain what the faculty/institution did after identifying the gap.
        - Describe the corrective action, responsible faculty, date/period, action type, evidence, and reassessment result.
        - Show the improvement achieved after the intervention.
        - State whether the action is completed or ongoing.
        - Include a clear note that below-target students require HOD review before closure.
        """
        required_sections = """
        1. Course metadata
        2. Gap summary from the earlier analysis
        3. Action taken details
        4. Action type and responsible faculty
        5. Dates/period and evidence of implementation
        6. Reassessment and result
        7. Status: completed or ongoing
        """
    else:
        report_instructions = """
        - This is a concise audit-ready NBA evidence note, not a gap or action report.
        - Answer: What evidence proves the CO/PO attainment process was actually carried out?
        - Include the actual evidence sources: direct assessment, indirect survey, CO-PO/PSO mapping, question-wise marks, gap analysis, and ATR.
        - Explain where the attainment numbers came from and how they were calculated.
        - Mention the evidence file or attachment identifiers where relevant.
        - Keep the report concise, traceable, and accreditation-ready.
        """
        required_sections = """
        1. Basic course information
        2. CO attainment evidence
        3. Direct assessment evidence
        4. CO-PO/PSO mapping evidence
        5. Indirect assessment evidence
        6. Gap analysis evidence
        7. Action taken evidence
        8. Improvement evidence
        9. Evidence files / attachments
        """

    client = genai.Client(api_key=api_key)
    prompt = f"""
You are a Course Outcome attainment agent for engineering accreditation.

Generate a clean, professional {report_type_clean} for {audience} using only the supplied data.

Required sections:
{required_sections}

Mandatory opening fields for all reports:
- Course Code
- Course Name
- Department
- Semester
- Academic Year
- Faculty
- Number of Students
- Target Attainment Level

Writing rules:
- Use clear Markdown headings and concise bullet points.
- Keep the tone formal, academic, and audit-ready.
- Do not include raw JSON or code blocks.
- Mention register numbers/student IDs for below-target students whenever applicable.
- {report_instructions}
- For Gap Analysis Report: explain expected/target attainment versus actual attainment, identify why the gap exists, and specify what action should be taken.
- For Action Taken Report: explain what the faculty/institution did after the gap was identified and what the result was.
- For NBA Evidence Note: answer what evidence proves that the CO/PO attainment process was actually carried out.

Data:
{context}
"""
    response = client.models.generate_content(model=model, contents=prompt)
    if hasattr(response, "text") and response.text:
        return response.text
    if hasattr(response, "candidates"):
        text_parts = []
        for candidate in response.candidates:
            for part in getattr(candidate, "content", []).parts:
                if hasattr(part, "text"):
                    text_parts.append(part.text)
        if text_parts:
            return "".join(text_parts)
    return str(response)


def friendly_llm_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "api key" in message or "authentication" in message or "forbidden" in message:
        return "The API request was not authenticated. Create or update the GEMINI_API_KEY in .env.local."
    if "quota" in message or "limit" in message or "429" in message or "rate" in message:
        return "The request reached Gemini, but the API quota or rate limit was reached. Try again later."
    if "invalid" in message or "bad request" in message or "safety" in message:
        return f"The API request was rejected. Check the selected model and request settings. Details: {exc}"
    if "connection" in message or "timed out" in message or "network" in message:
        return "The app could not connect to Gemini. Check internet access, firewall/proxy settings, and retry."
    return f"Report generation failed: {exc}"


def report_template(report_type: str, course_code: str, course_name: str, audience: str, gap_analysis: pd.DataFrame) -> str:
    report_type = (report_type or "Action taken report").strip()
    gap_rows = gap_analysis if isinstance(gap_analysis, pd.DataFrame) else pd.DataFrame()
    gap_summary = "\n".join(
        [
            "| CO | Gap | Students below target | Probable cause |",
            "| --- | --- | --- | --- |",
            *[
                f"| {row['co']} | {row.get('gap', '')} | {row.get('students_below_target', '')} | {row.get('probable_cause', '')} |"
                for _, row in gap_rows.iterrows()
            ],
        ]
    ) if not gap_rows.empty else "No major attainment gap was found in the current dataset."

    if "gap" in report_type.lower():
        return f"""# Gap Analysis Report

## Course Metadata
- Course Code: {course_code or 'Not available'}
- Course Name: {course_name or 'Not available'}
- Department: To be entered by faculty
- Semester: To be entered by faculty
- Academic Year: To be entered by faculty
- Faculty: {audience}
- Number of Students: To be entered by faculty
- Target Attainment Level: 2.00

## 1. Executive Summary
This report explains the difference between expected/target attainment and actual attainment for the course. It identifies where the course outcome fell below the target, why the gap exists, and what action should be taken to close the gap.

## 2. Gap Analysis by CO
{gap_summary}

## 3. Root Cause Analysis
- Compare the target attainment with actual attainment for each CO.
- Identify weak areas from question-wise marks and assessment performance.
- Explain why the gap exists in terms of direct assessment weakness, low question performance, or student difficulty in the relevant learning outcome.

## 4. Required Action
- Recommend remedial teaching, tutorial support, assignment reinforcement, or reassessment.
- State clearly the corrective action that should be taken to improve the attainment.
- Record that below-target students require HOD verification before closure.
"""

    if "action" in report_type.lower():
        return f"""# Action Taken Report (ATR)

## Course Metadata
- Course Code: {course_code or 'Not available'}
- Course Name: {course_name or 'Not available'}
- Department: To be entered by faculty
- Semester: To be entered by faculty
- Academic Year: To be entered by faculty
- Faculty: {audience}
- Number of Students: To be entered by faculty
- Target Attainment Level: 2.00

## 1. Gap Summary from Earlier Analysis
This ATR is the follow-up to the Gap Analysis Report. It records what the faculty/institution did after identifying the gap and what the result was.

## 2. Action Taken Details
- COs below target: {', '.join(str(item) for item in gap_rows['co'].tolist()) if not gap_rows.empty else 'None'}
- Action Type: Remedial / Tutorial / Assignment / Lab / Reassessment
- Responsible Faculty: To be entered
- Date/Period: To be entered
- Evidence: Attendance sheet, tutorial material, assignment, practice questions, reassessment results

## 3. Reassessment and Improvement
- Compare the attainment before and after the action.
- Document the improvement achieved.
- State whether the action is completed or ongoing.

## 4. Final Status
The action taken is recorded as completed or ongoing, with the requirement that below-target students are reviewed by the HOD before closure.
"""

    return f"""# NBA Evidence Note

## Course Metadata
- Course Code: {course_code or 'Not available'}
- Course Name: {course_name or 'Not available'}
- Department: To be entered by faculty
- Semester: To be entered by faculty
- Academic Year: To be entered by faculty
- Faculty: {audience}
- Number of Students: To be entered by faculty
- Target Attainment Level: 2.00

## 1. Purpose
This document answers the audit question: What evidence proves that the CO/PO attainment process was actually carried out?

## 2. CO Attainment Evidence
- Direct assessment marks and indirect survey data were used according to the approved institutional methodology.
- CO attainment values and status were calculated from the assessment and survey evidence.

## 3. Direct Assessment Evidence
- Question paper
- CO-wise question mapping
- Student marks
- Answer scripts
- Internal assessment marks
- End-semester examination marks
- Assignment/lab marks

## 4. CO-PO/PSO Mapping Evidence
- CO-PO/PSO mapping was established using the approved correlation matrix.
- The mapping forms the basis for outcome attainment analysis.

## 5. Indirect Assessment Evidence
- Student course outcome survey data was used as indirect evidence for CO attainment.

## 6. Gap Analysis Evidence
- Where COs were below target, the evidence note references the corresponding gap analysis and question-wise weakness.

## 7. Action Taken Evidence
- Corrective teaching, tutorials, assignments, or reassessment records provide the evidence for the action taken.

## 8. Improvement Evidence
- Reassessment and post-action results show whether the gap was reduced or closed.

## 9. Evidence Files
- EV-001 Course syllabus
- EV-002 CO statements
- EV-003 Question paper
- EV-004 CO-question mapping
- EV-005 Student marks
- EV-006 CO attainment calculation
- EV-007 CO-PO/PSO mapping
- EV-008 Student survey
- EV-009 Gap analysis
- EV-010 Action Taken Report
- EV-011 Reassessment
"""


def get_active_course(courses: pd.DataFrame, selected_course_code: str | None = None) -> tuple[str, str]:
    if courses is None or courses.empty:
        return "", ""
    if selected_course_code:
        selected = courses[courses["course_code"].astype(str).str.strip().str.lower() == str(selected_course_code).strip().lower()]
        if not selected.empty:
            first = selected.iloc[0]
            return str(first.get("course_code", "")), str(first.get("course_name", ""))
    first = courses.iloc[0]
    return str(first.get("course_code", "")), str(first.get("course_name", ""))


def course_identity(courses: pd.DataFrame) -> tuple[str, str]:
    return get_active_course(courses)


def main() -> None:
    st.set_page_config(
        page_title="Course outcome agent",
        page_icon=":material/analytics:",
        layout="wide",
    )

    load_dotenv(".env.local")
    init_db()
    init_state()
    require_authentication()

    st.title("Course outcome agent", icon=":material/analytics:")
    st.caption(f"Signed in as: {st.session_state.authenticated_role} ({st.session_state.authenticated_user})")
    st.caption(
        "Upload institutional academic data, calculate attainment, generate LLM analysis, and store the report in the database."
    )

    with st.sidebar:
        st.write(f"Role: {st.session_state.authenticated_role}")
        if st.button("Logout", type="secondary"):
            logout_current_user()

        st.header("Rubric settings", icon=":material/tune:")
        mark_threshold_percent = st.slider("Marks threshold", 0, 100, 60, 5)
        direct_weight = st.number_input("Direct weight", min_value=0.0, max_value=100.0, value=80.0, step=5.0)
        indirect_weight = st.number_input("Indirect weight", min_value=0.0, max_value=100.0, value=20.0, step=5.0)
        target_level = st.number_input("Target attainment level", min_value=0.0, max_value=3.0, value=2.0, step=0.1)
        st.caption("Institutional attainment levels")
        level_1_min_percent = st.slider("Level 1 minimum", 0, 100, 50, 5)
        level_2_min_percent = st.slider("Level 2 minimum", 0, 100, 60, 5)
        level_3_min_percent = st.slider("Level 3 minimum", 0, 100, 70, 5)
        model_name = st.text_input("Gemini model", value=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
        st.caption(f"Database: {DB_PATH}")

    config = AttainmentConfig(
        mark_threshold_percent=mark_threshold_percent,
        direct_weight=direct_weight,
        indirect_weight=indirect_weight,
        target_level=target_level,
        level_1_min_percent=level_1_min_percent,
        level_2_min_percent=level_2_min_percent,
        level_3_min_percent=level_3_min_percent,
    )

    if not (level_1_min_percent <= level_2_min_percent <= level_3_min_percent):
        st.warning("Attainment level thresholds should increase from level 1 to level 3.", icon=":material/warning:")

    if role_can_access_feature(st.session_state.authenticated_role, "data_upload"):
        data_tab, dashboard_tab, report_tab, history_tab = st.tabs(
            [
                ":material/upload: Upload data",
                ":material/query_stats: Attainment dashboard",
                ":material/article: LLM reports",
                ":material/storage: Database history",
            ]
        )
    else:
        dashboard_tab, report_tab, history_tab = st.tabs(
            [
                ":material/query_stats: Attainment dashboard",
                ":material/article: LLM reports",
                ":material/storage: Database history",
            ]
        )
        st.info("You are signed in as a reviewer role. Data upload is restricted to Faculty members.", icon=":material/info:")

    if role_can_access_feature(st.session_state.authenticated_role, "data_upload"):
        with data_tab:
            st.subheader("Manual data upload", icon=":material/upload:")
            st.caption("No default records are loaded. Upload all required CSV/XLSX files before processing.")

            with st.expander("Required file columns", icon=":material/table_chart:", expanded=True):
                st.table(
                    {
                        "CO statements": "co, co_statement, optional course_code, optional course_name",
                        "CO-PO/PSO mapping": "co, outcome_type, outcome, strength",
                        "Question-wise marks": "student_id, assessment, question, co, max_marks, marks_obtained, optional cohort",
                        "Indirect survey": "co, survey_percent",
                    },
                    border="horizontal",
                    width="content",
                )

            upload_cols = st.columns(4)
            uploaders = {
                "courses": upload_cols[0].file_uploader("CO statements", type=["csv", "xlsx"], key="courses_file"),
                "mapping": upload_cols[1].file_uploader("CO-PO/PSO mapping", type=["csv", "xlsx"], key="mapping_file"),
                "marks": upload_cols[2].file_uploader("Question-wise marks", type=["csv", "xlsx"], key="marks_file"),
                "indirect": upload_cols[3].file_uploader("Indirect survey", type=["csv", "xlsx"], key="indirect_file"),
            }

            for key, uploaded in uploaders.items():
                loaded = load_csv_or_excel(uploaded)
                if loaded is not None:
                    st.session_state[key] = loaded
                    st.toast(f"{uploaded.name} uploaded", icon=":material/check:")

            preview_cols = st.columns(4)
            for idx, key in enumerate(["courses", "mapping", "marks", "indirect"]):
                data = st.session_state[key]
                with preview_cols[idx].container(border=True):
                    st.metric(key.replace("_", " ").capitalize(), "Not uploaded" if data is None else f"{len(data)} rows")

            if uploaded_data_ready():
                preview_tab_1, preview_tab_2, preview_tab_3, preview_tab_4 = st.tabs(
                    ["CO statements", "Mapping", "Question marks", "Indirect survey"]
                )
                with preview_tab_1:
                    st.dataframe(st.session_state.courses, hide_index=True)
                with preview_tab_2:
                    st.dataframe(st.session_state.mapping, hide_index=True)
                with preview_tab_3:
                    st.dataframe(st.session_state.marks, hide_index=True)
                with preview_tab_4:
                    st.dataframe(st.session_state.indirect, hide_index=True)

    errors = validate_inputs()

    if errors:
        with dashboard_tab:
            st.info("Upload all required files to calculate attainment.", icon=":material/info:")
            for error in errors:
                st.caption(error)
        with report_tab:
            st.info("Upload valid files first. The LLM report is generated only from uploaded data.", icon=":material/info:")
        with history_tab:
            runs = list_runs()
            st.subheader("Saved database records", icon=":material/storage:")
            if runs.empty:
                st.caption("No generated reports have been stored yet.")
            else:
                st.write("All authenticated users can open and delete saved reports.")
                for _, row in runs.iterrows():
                    cols = st.columns([2, 2, 2, 2, 1, 1])
                    cols[0].write(row["course_code"] if "course_code" in row and row["course_code"] is not None else "-")
                    cols[1].write(row["course_name"] if "course_name" in row and row["course_name"] is not None else "-")
                    cols[2].write(row["report_type"] if "report_type" in row and row["report_type"] is not None else "-")
                    cols[3].write(row["audience"] if "audience" in row and row["audience"] is not None else "-")
                    if cols[4].button("Open", key=f"open_saved_record_{row['id']}_error", use_container_width=True):
                        st.session_state["selected_saved_record_id"] = int(row["id"])
                    if cols[5].button("Delete", key=f"delete_saved_record_{row['id']}_error", use_container_width=True):
                        if delete_run(int(row["id"])):
                            st.session_state["selected_saved_record_id"] = None
                            st.success(f"Record #{row['id']} deleted successfully.")
                            st.rerun()
                        else:
                            st.error(f"Unable to delete record #{row['id']}.")

            default_selected = st.session_state.get("selected_saved_record_id")
            selected_id = get_selected_history_id(runs, default_selected)
            if selected_id is not None:
                saved = get_run(int(selected_id))
                if saved:
                    st.subheader(f"Opened record #{selected_id}", divider="rainbow")
                    st.text_area("Saved LLM report", saved["llm_report"], height=300)
                    saved_gap = pd.DataFrame(saved["gap_analysis_json"])
                    if not saved_gap.empty:
                        st.dataframe(saved_gap, hide_index=True)
                else:
                    st.warning("The selected record could not be opened.")
        st.stop()

    co_attainment = compute_co_attainment(st.session_state.courses, st.session_state.marks, st.session_state.indirect, config)
    outcome_attainment = compute_outcome_attainment(co_attainment, st.session_state.mapping)
    diagnostics = compute_question_diagnostics(st.session_state.marks, config)
    gap_analysis = build_gap_analysis(co_attainment, diagnostics, st.session_state.marks, config)
    below_count = int((co_attainment["status"] == "Below target").sum())
    avg_co = co_attainment["combined_level"].mean()
    avg_outcome = outcome_attainment["attainment_level"].mean()
    evidence_rows = len(diagnostics)
    course_options = []
    if st.session_state.courses is not None and not st.session_state.courses.empty:
        course_options = [
            str(code).strip()
            for code in st.session_state.courses.get("course_code", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        ]
    selected_course_code = st.session_state.get("selected_course_code")
    if not course_options:
        selected_course_code = None
    elif selected_course_code not in course_options:
        selected_course_code = course_options[0]
        st.session_state["selected_course_code"] = selected_course_code
    course_code, course_name = get_active_course(st.session_state.courses, selected_course_code)

    with dashboard_tab:
        with st.container(horizontal=True):
            st.metric("Average CO level", f"{avg_co:.2f}", border=True)
            st.metric("Average PO/PSO level", f"{avg_outcome:.2f}", border=True)
            st.metric("COs below target", str(below_count), border=True)
            st.metric("Evidence links", str(evidence_rows), border=True)

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.subheader("CO attainment", icon=":material/bar_chart:")
                chart_data = co_attainment[["co", "combined_level", "target_level"]].melt(
                    "co", var_name="Measure", value_name="Level"
                )
                st.bar_chart(chart_data, x="co", y="Level", color="Measure")
        with right:
            with st.container(border=True):
                st.subheader("PO and PSO attainment", icon=":material/account_tree:")
                st.bar_chart(outcome_attainment, x="outcome", y="attainment_level", color="outcome_type")

        with st.container(border=True):
            st.subheader("CO attainment sheet", icon=":material/table_chart:")
            st.dataframe(
                co_attainment,
                hide_index=True,
                column_config={
                    "combined_level": st.column_config.ProgressColumn("Combined level", min_value=0, max_value=3, format="%.2f"),
                    "target_level": st.column_config.NumberColumn("Target", format="%.2f"),
                    "gap": st.column_config.NumberColumn("Gap", format="%.2f"),
                },
            )

        with st.container(border=True):
            st.subheader("Traceability by question", icon=":material/visibility:")
            st.dataframe(
                diagnostics,
                hide_index=True,
                column_config={
                    "average_percent": st.column_config.NumberColumn("Average %", format="%.2f"),
                    "threshold_crossing_percent": st.column_config.ProgressColumn(
                        "Students crossing threshold", min_value=0, max_value=100, format="%.2f%%"
                    ),
                },
            )

    with report_tab:
        st.subheader("LLM analysis and reports", icon=":material/article:")
        if gap_analysis.empty:
            st.success("All COs have met the configured target.", icon=":material/check_circle:")
        else:
            st.dataframe(gap_analysis, hide_index=True)

        if course_options:
            selected_course_code = st.selectbox(
                "Course",
                course_options,
                index=course_options.index(selected_course_code) if selected_course_code in course_options else 0,
            )
            st.session_state["selected_course_code"] = selected_course_code
            course_code, course_name = get_active_course(st.session_state.courses, selected_course_code)

        report_choice = st.segmented_control(
            "Report type",
            ["Gap analysis report", "Action taken report", "NBA evidence note"],
            default="Action taken report",
        )
        audience = st.selectbox(
            "Audience",
            ["Faculty", "Head of department", "NBA coordinator", "IQAC"],
        )

        context = report_context(co_attainment, outcome_attainment, gap_analysis)
        fallback_report = report_template(report_choice, course_code, course_name, audience, gap_analysis)

        if st.button("Generate, analyze, and save", icon=":material/auto_awesome:", type="primary"):
            try:
                with st.skeleton(height=260):
                    report_text = ai_report(context, audience, report_choice, model_name)
                run_id = save_run(
                    course_code=course_code,
                    course_name=course_name,
                    report_type=report_choice,
                    audience=audience,
                    model=model_name,
                    config=asdict(config),
                    uploaded_row_counts={
                        "courses": len(st.session_state.courses),
                        "mapping": len(st.session_state.mapping),
                        "marks": len(st.session_state.marks),
                        "indirect": len(st.session_state.indirect),
                    },
                    co_attainment=co_attainment,
                    outcome_attainment=outcome_attainment,
                    gap_analysis=gap_analysis,
                    diagnostics=diagnostics,
                    llm_report=report_text,
                )
                st.session_state.generated_report = report_text
                st.session_state.saved_run_id = run_id
                st.success(f"LLM report generated and stored in database record #{run_id}.", icon=":material/check_circle:")
            except Exception as exc:
                st.error(friendly_llm_error(exc), icon=":material/error:")

        report_text = st.session_state.generated_report or fallback_report
        st.text_area("Generated report", report_text, height=360)

        with st.container(horizontal=True):
            st.download_button(
                "Download Excel evidence",
                data=create_excel_report(co_attainment, outcome_attainment, gap_analysis, diagnostics),
                file_name="course_outcome_attainment_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                icon=":material/download:",
            )
            st.download_button(
                "Download report text",
                data=report_text,
                file_name="action_taken_report.md",
                mime="text/markdown",
                icon=":material/description:",
            )

    with history_tab:
        st.subheader("Saved database records", icon=":material/storage:")
        runs = list_runs()
        if runs.empty:
            st.caption("No saved reports have been created yet.")
        else:
            st.write("All authenticated users can open and delete saved reports.")
            for _, row in runs.iterrows():
                cols = st.columns([2, 2, 2, 2, 1, 1])
                cols[0].write(row["course_code"] if "course_code" in row and row["course_code"] is not None else "-")
                cols[1].write(row["course_name"] if "course_name" in row and row["course_name"] is not None else "-")
                cols[2].write(row["report_type"] if "report_type" in row and row["report_type"] is not None else "-")
                cols[3].write(row["audience"] if "audience" in row and row["audience"] is not None else "-")
                if cols[4].button("Open", key=f"open_saved_record_{row['id']}", use_container_width=True):
                    st.session_state["selected_saved_record_id"] = int(row["id"])
                if cols[5].button("Delete", key=f"delete_saved_record_{row['id']}", use_container_width=True):
                    if delete_run(int(row["id"])):
                        st.session_state["selected_saved_record_id"] = None
                        st.success(f"Record #{row['id']} deleted successfully.")
                        st.rerun()
                    else:
                        st.error(f"Unable to delete record #{row['id']}.")

            default_selected = st.session_state.get("selected_saved_record_id")
            selected_id = get_selected_history_id(runs, default_selected)
            if selected_id is not None:
                saved = get_run(int(selected_id))
                if saved:
                    st.subheader(f"Opened record #{selected_id}", divider="rainbow")
                    st.text_area("Saved LLM report", saved["llm_report"], height=300)
                    saved_gap = pd.DataFrame(saved["gap_analysis_json"])
                    if not saved_gap.empty:
                        st.dataframe(saved_gap, hide_index=True)
                else:
                    st.warning("The selected record could not be opened.")
            else:
                st.info("Select a saved record to open it.")


if __name__ == "__main__":
    main()
