import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

import database
from database import list_uploaded_datasets, save_uploaded_dataset

MODULE_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"

spec = importlib.util.spec_from_file_location("streamlit_app_under_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GeminiAiReportTests(unittest.TestCase):
    def test_ai_report_uses_gemini(self):
        os.environ["GEMINI_API_KEY"] = "test-key"

        fake_client = MagicMock()
        fake_response = MagicMock()
        fake_response.text = "Gemini report"
        fake_client.models.generate_content.return_value = fake_response
        module.genai = MagicMock()
        module.genai.Client.return_value = fake_client

        result = module.ai_report("context", "faculty", "summary", "gemini-3.6-flash")

        self.assertEqual(result, "Gemini report")
        module.genai.Client.assert_called_once_with(api_key="test-key")
        fake_client.models.generate_content.assert_called_once()
        call_kwargs = fake_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-3.6-flash")
        self.assertIn("context", call_kwargs["contents"])
        self.assertIn("faculty", call_kwargs["contents"])

    def test_ai_report_rejects_retired_model_name(self):
        os.environ["GEMINI_API_KEY"] = "test-key"

        fake_client = MagicMock()
        fake_response = MagicMock()
        fake_response.text = "Should not be used"
        fake_client.models.generate_content.return_value = fake_response
        module.genai = MagicMock()
        module.genai.Client.return_value = fake_client

        with self.assertRaisesRegex(RuntimeError, r"retired|unsupported|gemini-3\.6-flash"):
            module.ai_report("context", "faculty", "summary", "gemini-2.0-flash")

        fake_client.models.generate_content.assert_not_called()

    def test_generate_dashboard_summary_uses_llm(self):
        os.environ["GEMINI_API_KEY"] = "test-key"

        fake_client = MagicMock()
        fake_response = MagicMock()
        fake_response.text = "Dashboard insight"
        fake_client.models.generate_content.return_value = fake_response
        module.genai = MagicMock()
        module.genai.Client.return_value = fake_client

        result = module.generate_dashboard_summary("context", "Faculty", "gemini-3.6-flash")

        self.assertEqual(result, "Dashboard insight")
        module.genai.Client.assert_called_once_with(api_key="test-key")
        fake_client.models.generate_content.assert_called_once()
        call_kwargs = fake_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-3.6-flash")
        self.assertIn("Dashboard Analysis", call_kwargs["contents"])
        self.assertIn("context", call_kwargs["contents"])

    def test_gap_analysis_report_has_required_metadata_and_gap_logic(self):
        os.environ["GEMINI_API_KEY"] = "test-key"

        fake_client = MagicMock()
        fake_response = MagicMock()
        fake_response.text = "Gap analysis report"
        fake_client.models.generate_content.return_value = fake_response
        module.genai = MagicMock()
        module.genai.Client.return_value = fake_client

        module.ai_report("context", "faculty", "Gap analysis report", "gemini-3.6-flash")

        prompt = fake_client.models.generate_content.call_args.kwargs["contents"]
        self.assertIn("Course Code", prompt)
        self.assertIn("Course Name", prompt)
        self.assertIn("Department", prompt)
        self.assertIn("Semester", prompt)
        self.assertIn("Academic Year", prompt)
        self.assertIn("Faculty", prompt)
        self.assertIn("Number of Students", prompt)
        self.assertIn("Target Attainment Level", prompt)
        self.assertIn("expected/target attainment", prompt.lower())
        self.assertIn("actual attainment", prompt.lower())
        self.assertIn("why the gap exists", prompt.lower())
        self.assertIn("what action should be taken", prompt.lower())

    def test_uploaded_dataset_is_saved_to_database(self):
        df = pd.DataFrame({"co": ["CO1"], "co_statement": ["Intro to data"]})
        save_uploaded_dataset("courses", "courses.csv", df)
        saved = list_uploaded_datasets(limit=5)
        self.assertTrue((saved["dataset_key"] == "courses").any())
        self.assertTrue((saved["file_name"] == "courses.csv").any())

    def test_google_sheets_sync_helpers_are_optional_and_safe(self):
        os.environ.pop("GOOGLE_SHEET_ID", None)
        os.environ.pop("GOOGLE_SERVICE_ACCOUNT_JSON", None)
        os.environ.pop("GOOGLE_SERVICE_ACCOUNT_FILE", None)

        self.assertIsNone(database.get_google_sheet_client())
        self.assertEqual(database.get_google_sheet_rows("Users"), [])

    def test_role_based_auth_allows_only_allowed_roles(self):
        os.environ["AUTH_FACULTY_USERNAME"] = "faculty"
        os.environ["AUTH_FACULTY_PASSWORD"] = "faculty123"
        os.environ["AUTH_HOD_USERNAME"] = "hod"
        os.environ["AUTH_HOD_PASSWORD"] = "hod123"
        os.environ["AUTH_NBA_USERNAME"] = "nba"
        os.environ["AUTH_NBA_PASSWORD"] = "nba123"
        os.environ["AUTH_IQAC_USERNAME"] = "iqac"
        os.environ["AUTH_IQAC_PASSWORD"] = "iqac123"

        self.assertTrue(module.validate_role_credentials("faculty", "faculty123"))
        self.assertTrue(module.validate_role_credentials("hod", "hod123"))
        self.assertTrue(module.validate_role_credentials("nba", "nba123"))
        self.assertTrue(module.validate_role_credentials("iqac", "iqac123"))
        self.assertFalse(module.validate_role_credentials("student", "student123"))
        self.assertFalse(module.validate_role_credentials("faculty", "wrongpass"))
        self.assertEqual(module.get_allowed_roles(), ["Faculty", "Head of department", "NBA coordinator", "IQAC"])

    def test_main_requires_authentication_before_rendering(self):
        original_st = module.st
        original_require_auth = module.require_authentication
        original_load_dotenv = module.load_dotenv
        original_init_db = module.init_db
        original_init_state = module.init_state

        module.st = MagicMock()
        module.st.session_state = {}
        module.require_authentication = MagicMock(side_effect=RuntimeError("auth stop"))
        module.load_dotenv = MagicMock()
        module.init_db = MagicMock()
        module.init_state = MagicMock()

        try:
            with self.assertRaisesRegex(RuntimeError, "auth stop"):
                module.main()
            module.require_authentication.assert_called_once()
        finally:
            module.st = original_st
            module.require_authentication = original_require_auth
            module.load_dotenv = original_load_dotenv
            module.init_db = original_init_db
            module.init_state = original_init_state

    def test_render_auth_controls_shows_logout_for_authenticated_user(self):
        original_st = module.st
        original_logout_current_user = module.logout_current_user
        module.st = MagicMock()
        module.st.session_state = {"authenticated_user": "faculty", "authenticated_role": "Faculty"}
        module.st.sidebar.button.return_value = True
        module.logout_current_user = MagicMock()

        try:
            module.render_auth_controls()
            module.st.sidebar.button.assert_called_once_with("Logout", icon=":material/logout:")
            module.logout_current_user.assert_called_once()
        finally:
            module.st = original_st
            module.logout_current_user = original_logout_current_user

    def test_role_permissions_are_role_specific(self):
        self.assertTrue(module.role_can_access_feature("Faculty", "data_upload"))
        self.assertTrue(module.role_can_access_feature("Head of department", "report_generation"))
        self.assertFalse(module.role_can_access_feature("Head of department", "data_upload"))
        self.assertTrue(module.role_can_access_feature("IQAC", "history_view"))

    def test_dashboard_reports_are_filtered_by_audience_role(self):
        runs = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "audience": ["Faculty", "Head of department", "NBA coordinator", "IQAC"],
                "course_name": ["A", "B", "C", "D"],
            }
        )

        self.assertEqual(
            list(module.filter_runs_for_role(runs, "Head of department")["audience"]),
            ["Head of department"],
        )
        self.assertEqual(
            list(module.filter_runs_for_role(runs, "NBA coordinator")["audience"]),
            ["NBA coordinator"],
        )
        self.assertEqual(
            list(module.filter_runs_for_role(runs, "IQAC")["audience"]),
            ["IQAC"],
        )
        self.assertEqual(
            list(module.filter_runs_for_role(runs, "Faculty")["audience"]),
            ["Faculty"],
        )

    def test_course_selection_uses_selected_course_not_first_row(self):
        courses = pd.DataFrame(
            {
                "course_code": ["BDA", "DBMS"],
                "course_name": ["Big Data Analytics", "Database Management System"],
            }
        )

        self.assertEqual(module.get_active_course(courses, "DBMS"), ("DBMS", "Database Management System"))
        self.assertEqual(module.get_active_course(courses, "BDA"), ("BDA", "Big Data Analytics"))
