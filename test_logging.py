"""
Unit & Integration Test Suite for AI API Usage Logging, Cost Estimation, and Export.
Runs offline using mock objects to avoid making any billable API calls or external requests.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import api_logger
from api_logger import JSONLApiLogger, calculate_cost, log_api_call, sanitize_text
from api_usage_summary import generate_summary
from export_logs import export_data, filter_records, read_jsonl_records


class TestApiUsageLogging(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_log_file = os.path.join(self.test_dir, "test_api_usage.jsonl")
        self.logger = JSONLApiLogger(log_file_path=self.test_log_file)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cost_calculation(self):
        # Test Gemini 2.0 Flash cost calculation
        cost_info = calculate_cost("gemini-2.0-flash", input_tokens=1000, output_tokens=500, cached_input_tokens=200)
        self.assertEqual(cost_info["cost_status"], "estimated")
        self.assertEqual(cost_info["currency"], "USD")
        # 800 non-cached input tokens @ $0.10/M = 0.000080
        # 200 cached tokens @ $0.025/M = 0.000005
        # 500 output tokens @ $0.40/M = 0.000200
        # Total = 0.000285
        self.assertAlmostEqual(cost_info["total_request_cost"], 0.000285, places=6)

        # Test unknown model produces unavailable
        unknown_cost = calculate_cost("unregistered-model-xyz", input_tokens=1000, output_tokens=500)
        self.assertEqual(unknown_cost["cost_status"], "unavailable")
        self.assertIsNone(unknown_cost["total_request_cost"])

    def test_sanitization(self):
        raw_err = "Request failed for key AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q and Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6"
        clean = sanitize_text(raw_err)
        self.assertNotIn("AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q", clean)
        self.assertIn("[REDACTED_API_KEY]", clean)
        self.assertIn("[REDACTED_TOKEN]", clean)

    def test_log_api_call_and_jsonl_output(self):
        test_txt_dir = os.path.join(self.test_dir, "txt_logs")
        text_logger = api_logger.TextApiLogger(log_dir=test_txt_dir)

        with patch.object(api_logger, "get_logger", return_value=self.logger), \
             patch.object(api_logger, "get_text_logger", return_value=text_logger):
            success = log_api_call(
                request_id="req_test_001",
                parent_request_id="chat_test_001",
                session_id="9c965954-7af1-468c-9213-8345ea927cf2",
                user_query="Tell me about Flutter development",
                bot_response="MultiQoS is a top Flutter app development company.",
                provider="google",
                model_name="gemini-2.0-flash",
                feature_name="chat_response",
                operation_type="chat_completion",
                endpoint="socketio.message",
                start_time=api_logger.time.time() - 0.25,
                status="success",
                input_tokens=150,
                output_tokens=50,
            )
            self.assertTrue(success)

        # Verify JSONL content
        self.assertTrue(os.path.exists(self.test_log_file))
        records, valid_count, malformed_count = read_jsonl_records(self.test_log_file)
        self.assertEqual(valid_count, 1)
        self.assertEqual(malformed_count, 0)
        rec = records[0]
        self.assertEqual(rec["request_id"], "req_test_001")
        self.assertEqual(rec["parent_request_id"], "chat_test_001")
        self.assertEqual(rec["input_tokens"], 150)
        self.assertEqual(rec["output_tokens"], 50)
        self.assertEqual(rec["total_tokens"], 200)
        self.assertEqual(rec["cost_status"], "estimated")
        self.assertEqual(rec["user_query"], "Tell me about Flutter development")
        self.assertEqual(rec["bot_response"], "MultiQoS is a top Flutter app development company.")
        self.assertIsNotNone(rec["total_request_cost"])

        # Verify TXT file output: YYYY_MM_DD_HHMMSS_<uuid[:10]>.txt
        # session_id "9c965954-7af1-468c-9213-8345ea927cf2" clean first 10 is "9c9659547a"
        today_prefix = api_logger.datetime.now().strftime("%Y_%m_%d")
        import glob
        matches = glob.glob(os.path.join(test_txt_dir, f"{today_prefix}_*_9c9659547a.txt"))
        self.assertTrue(len(matches) > 0, f"Expected matching txt log for 9c9659547a in {test_txt_dir}")
        expected_txt_path = matches[0]

        with open(expected_txt_path, "r", encoding="utf-8") as f:
            txt_content = f.read()

        self.assertIn("9c965954-7af1-468c-9213-8345ea927cf2", txt_content)
        self.assertIn("USER QUERY", txt_content)
        self.assertIn("Tell me about Flutter development", txt_content)
        self.assertIn("BOT RESPONSE", txt_content)
        self.assertIn("MultiQoS is a top Flutter app development company.", txt_content)
        self.assertIn("TOKEN USAGE", txt_content)
        self.assertIn("Total Tokens     : 200", txt_content)
        self.assertIn("gemini-2.0-flash", txt_content)

    def test_malformed_json_handling_and_summary(self):
        # Inject one valid and one malformed line
        with open(self.test_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "request_id": "req_002",
                "parent_request_id": "chat_002",
                "timestamp": "2026-10-22T10:00:00+00:00",
                "provider": "google",
                "model_name": "gemini-2.0-flash",
                "feature_name": "welcome_greeting",
                "status": "failure",
                "retry_count": 2,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "total_request_cost": None,
                "latency_ms": 500,
            }) + "\n")
            f.write("THIS_IS_CORRUPTED_JSON_LINE\n")

        records, valid, malformed = read_jsonl_records(self.test_log_file)
        self.assertEqual(valid, 1)
        self.assertEqual(malformed, 1)

        summary = generate_summary(records)
        self.assertEqual(summary["total_calls"], 1)
        self.assertEqual(summary["failed_calls"], 1)
        self.assertEqual(summary["retry_count"], 2)
        self.assertEqual(summary["failure_rate_pct"], 100.0)

    def test_csv_export(self):
        records = [
            {
                "request_id": "r1",
                "parent_request_id": "p1",
                "timestamp": "2026-10-22T10:00:00+00:00",
                "feature_name": "chat_response",
                "provider": "google",
                "status": "success",
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "total_request_cost": 0.0001,
            }
        ]
        out_csv = os.path.join(self.test_dir, "export.csv")
        export_data(records, "csv", out_csv)
        self.assertTrue(os.path.exists(out_csv))
        with open(out_csv, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("request_id", content)
            self.assertIn("r1", content)


if __name__ == "__main__":
    unittest.main()
