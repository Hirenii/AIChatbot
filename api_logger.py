"""
Internal AI API Usage and Cost Monitoring Logger.

Provides thread-safe and process-safe structured logging of all external AI provider calls
to local JSON Lines (.jsonl) files with automatic rotation, sanitized errors, and centralized pricing.
"""

import glob
import json
import logging
import os
import re
import textwrap
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Safe internal fallback logger for operational/diagnostic warnings
op_logger = logging.getLogger("api_logger_internal")
if not op_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[API_LOGGER %(levelname)s] %(asctime)s - %(message)s"))
    op_logger.addHandler(_h)
    op_logger.setLevel(logging.WARNING)

# Base project path to resolve reliable absolute defaults
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = os.getenv("API_LOG_DIR", os.path.join(BASE_DIR, "logs"))
DEFAULT_LOG_FILE = os.getenv("API_LOG_FILE", os.path.join(DEFAULT_LOG_DIR, "api_usage_logs.jsonl"))
MAX_LOG_SIZE_BYTES = int(os.getenv("API_LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB default rotation
MAX_BACKUP_COUNT = int(os.getenv("API_LOG_BACKUP_COUNT", "10"))
CURRENT_ENV = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "staging"))

# Centralized pricing per 1 Million tokens (USD).
# Explicitly labeled as estimates. Updates can be maintained here or via environment.
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Gemini models (Official Google AI Studio rates per 1M tokens)
    "gemini-1.5-flash": {
        "input_per_million": 0.075,
        "output_per_million": 0.30,
        "cached_input_per_million": 0.01875,
    },
    "gemini-1.5-flash-8b": {
        "input_per_million": 0.0375,
        "output_per_million": 0.15,
        "cached_input_per_million": 0.01,
    },
    "gemini-1.5-pro": {
        "input_per_million": 1.25,
        "output_per_million": 5.00,
        "cached_input_per_million": 0.3125,
    },
    "gemini-2.0-flash": {
        "input_per_million": 0.10,
        "output_per_million": 0.40,
        "cached_input_per_million": 0.025,
    },
    "gemini-2.0-flash-lite": {
        "input_per_million": 0.075,
        "output_per_million": 0.30,
        "cached_input_per_million": 0.01875,
    },
    "gemini-2.5-flash": {
        "input_per_million": 0.10,
        "output_per_million": 0.40,
        "cached_input_per_million": 0.025,
    },
    "gemini-3.5-flash-lite": {
        # Configured alias in current codebase; map to lite tier
        "input_per_million": 0.075,
        "output_per_million": 0.30,
        "cached_input_per_million": 0.01875,
    },
    "gemini-3.5-flash": {
        "input_per_million": 0.10,
        "output_per_million": 0.40,
        "cached_input_per_million": 0.025,
    },
}

_write_lock = threading.Lock()


def get_iso_timestamp() -> str:
    """Returns the current timezone-aware ISO 8601 formatted timestamp string."""
    return datetime.now(timezone.utc).astimezone().isoformat()


def sanitize_text(text: Optional[str]) -> Optional[str]:
    """Sanitizes error messages or text to prevent accidental leakage of API keys or credentials."""
    if not text:
        return text
    # Mask common API key patterns (Google AI keys: AIzaSy..., Bearer tokens, general secrets)
    text = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", str(text))
    text = re.sub(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{20,}", r"\1[REDACTED_TOKEN]", text)
    text = re.sub(r"(key|api_key|token|secret)=([a-zA-Z0-9_\-\.]+)", r"\1=[REDACTED]", text, flags=re.IGNORECASE)
    return text


def calculate_cost(
    model_name: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cached_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calculates estimated cost based on centralized model pricing.
    Returns cost metrics or explicit null/unavailable markers if pricing is unknown.
    """
    cost_data: Dict[str, Any] = {
        "input_cost": None,
        "output_cost": None,
        "cached_input_cost": None,
        "other_cost": 0.0,
        "total_request_cost": None,
        "currency": "USD",
        "pricing_version": "2026-v1",
        "cost_status": "unavailable",
    }

    if not model_name:
        return cost_data

    # Standardize model key (strip prefixes like models/ or version suffixes if necessary)
    lookup_key = model_name.replace("models/", "").lower().strip()
    pricing = MODEL_PRICING.get(lookup_key)

    if not pricing:
        # Check prefix match for fine-tuned or date-tagged model names
        for m_key, m_val in MODEL_PRICING.items():
            if lookup_key.startswith(m_key):
                pricing = m_val
                break

    if not pricing:
        cost_data["cost_status"] = "unavailable"
        return cost_data

    if input_tokens is None and output_tokens is None:
        cost_data["cost_status"] = "unavailable"
        return cost_data

    in_cost = 0.0
    out_cost = 0.0
    cached_cost = 0.0

    # Account for cached input tokens if present
    cached_tok = cached_input_tokens or 0
    non_cached_input = max(0, (input_tokens or 0) - cached_tok)

    if input_tokens is not None:
        in_cost = (non_cached_input / 1_000_000.0) * pricing.get("input_per_million", 0.0)
        cost_data["input_cost"] = round(in_cost, 7)

    if cached_tok > 0 and "cached_input_per_million" in pricing:
        cached_cost = (cached_tok / 1_000_000.0) * pricing.get("cached_input_per_million", 0.0)
        cost_data["cached_input_cost"] = round(cached_cost, 7)

    if output_tokens is not None:
        out_cost = (output_tokens / 1_000_000.0) * pricing.get("output_per_million", 0.0)
        cost_data["output_cost"] = round(out_cost, 7)

    total = in_cost + cached_cost + out_cost
    cost_data["total_request_cost"] = round(total, 7)
    cost_data["cost_status"] = "estimated"
    return cost_data


class JSONLApiLogger:
    """Safe, rotating JSON Lines logger for recording AI provider API calls."""

    def __init__(self, log_file_path: Optional[str] = None):
        self.log_file = log_file_path or DEFAULT_LOG_FILE
        self.log_dir = os.path.dirname(self.log_file)
        self._ensure_dir()

    def _ensure_dir(self):
        try:
            if self.log_dir and not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, mode=0o750, exist_ok=True)
        except Exception as e:
            op_logger.warning("Failed to create log directory '%s': %s", self.log_dir, e)

    def _rotate_if_needed(self):
        """Rotates the log file if it exceeds MAX_LOG_SIZE_BYTES, retaining historical backups."""
        try:
            if not os.path.exists(self.log_file):
                return
            if os.path.getsize(self.log_file) < MAX_LOG_SIZE_BYTES:
                return

            for i in range(MAX_BACKUP_COUNT - 1, 0, -1):
                sfn = f"{self.log_file}.{i}"
                dfn = f"{self.log_file}.{i + 1}"
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)

            backup_1 = f"{self.log_file}.1"
            if os.path.exists(backup_1):
                os.remove(backup_1)
            os.rename(self.log_file, backup_1)
        except Exception as e:
            op_logger.warning("Error during log rotation for '%s': %s", self.log_file, e)

    def log(self, record: Dict[str, Any]) -> bool:
        """
        Thread-safe append of a structured dictionary record as a JSON line.
        Guarantees that logging errors NEVER raise exceptions to caller.
        """
        try:
            self._ensure_dir()
            json_line = json.dumps(record, ensure_ascii=False) + "\n"

            with _write_lock:
                self._rotate_if_needed()
                # Use append mode with restricted file permissions
                flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
                mode = 0o640
                fd = os.open(self.log_file, flags, mode)
                try:
                    with os.fdopen(fd, "a", encoding="utf-8") as f:
                        f.write(json_line)
                        f.flush()
                except Exception:
                    # If fdopen raises, ensure fd is closed
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    raise
            return True
        except Exception as e:
            op_logger.warning("Failed to append API usage record: %s", e)
            return False


class TextApiLogger:
    """
    Logger that writes human-readable interaction logs to individual .txt files.
    The filename format is: YYYY_MM_DD_<first_10_chars_of_uuid>.txt
    Each file records details including user question, bot response, token metrics, costs, and metadata.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self._ensure_dir()

    def _ensure_dir(self):
        try:
            if self.log_dir and not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, mode=0o755, exist_ok=True)
        except Exception as e:
            op_logger.warning("Failed to create text log directory '%s': %s", self.log_dir, e)

    def get_filename_for_record(self, record: Dict[str, Any]) -> str:
        """
        Determines filename matching: YYYY_MM_DD_HHMMSS_<uuid[:10]>.txt
        Uses session_id if available, falling back to request_id.
        If an existing file for this session/day already exists in log_dir, preserves it
        so all logs for that session remain sequentially in the same file.
        """
        # UUID identifier (session_id preferred, then parent_request_id, then request_id)
        raw_id = record.get("session_id") or record.get("parent_request_id") or record.get("request_id") or "anonymous"
        clean_id = re.sub(r"[^a-zA-Z0-9]", "", str(raw_id))
        uuid_part = clean_id[:10] if len(clean_id) >= 10 else clean_id.ljust(10, "0")

        # Parse timestamp
        ts_str = record.get("timestamp") or get_iso_timestamp()
        try:
            if "T" in ts_str:
                clean_ts = ts_str.replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean_ts)
                date_prefix = dt.strftime("%Y_%m_%d")
                time_prefix = dt.strftime("%H%M%S")
            elif len(ts_str) >= 19:
                date_prefix = ts_str[:10].replace("-", "_")
                time_prefix = ts_str[11:19].replace(":", "")
            else:
                date_prefix = ts_str[:10].replace("-", "_")
                time_prefix = datetime.now().strftime("%H%M%S")
        except Exception:
            date_prefix = datetime.now().strftime("%Y_%m_%d")
            time_prefix = datetime.now().strftime("%H%M%S")

        # Check if an existing file for this date and uuid_part already exists
        if os.path.exists(self.log_dir):
            try:
                # Look for matching pattern: <date_prefix>_*_<uuid_part>.txt or <date_prefix>_<uuid_part>.txt
                pattern = f"{date_prefix}_*_{uuid_part}.txt"
                matches = glob.glob(os.path.join(self.log_dir, pattern))
                if matches:
                    matches.sort()
                    return os.path.basename(matches[0])
                # Check legacy non-timestamp format too
                legacy = os.path.join(self.log_dir, f"{date_prefix}_{uuid_part}.txt")
                if os.path.exists(legacy):
                    return f"{date_prefix}_{uuid_part}.txt"
            except Exception:
                pass

        return f"{date_prefix}_{time_prefix}_{uuid_part}.txt"

    def format_entry(self, record: Dict[str, Any], log_number: int = 1) -> str:
        """Formats the interaction in the requested structured card presentation."""
        # 1. Timestamp formatting: YYYY-MM-DD HH:MM:SS
        raw_ts = record.get("timestamp") or ""
        ts_display = raw_ts
        try:
            # Parse ISO timestamp or date
            if "T" in raw_ts:
                clean_ts = raw_ts.replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean_ts)
                ts_display = dt.strftime("%Y-%m-%d %H:%M:%S")
            elif len(raw_ts) >= 19:
                ts_display = raw_ts[:19].replace("T", " ")
        except Exception:
            ts_display = raw_ts

        # 2. Session, Request, Parent IDs
        session_id = record.get("session_id") or "N/A"
        request_id = record.get("request_id") or "N/A"
        parent_request_id = record.get("parent_request_id") or "N/A"
        feature = record.get("feature_name") or "chat_response"
        model = record.get("model_name") or "gemini-3.5-flash-lite"
        provider = (record.get("provider") or "Google").capitalize()

        # 3. Status and Response Time
        status_raw = str(record.get("status") or "Success").capitalize()
        http_status = record.get("http_status")
        if http_status:
            status_display = f"{status_raw} (HTTP {http_status})"
        else:
            status_display = status_raw

        latency_ms = record.get("latency_ms")
        if latency_ms is not None:
            resp_time_sec = latency_ms / 1000.0
            resp_time_display = f"{resp_time_sec:.2f} seconds"
        else:
            resp_time_display = "N/A"

        # 4. User Query and Bot Response
        user_query = (record.get("user_query") or "").strip()
        user_text = user_query if user_query else "(No message)"

        bot_response = record.get("bot_response")
        error_msg = record.get("error_message")
        if bot_response:
            bot_text = bot_response.strip()
            # If bot_response has markdown link syntax like: [Text](url), convert to cleaner URL presentation if present
            # or preserve formatting cleanly
            def _clean_markdown_link(m):
                link_title = m.group(1).strip()
                link_url = m.group(2).strip()
                return f"Learn more:\n{link_url}"
            # Check for patterns like "To know more, check out this link: [Title](URL)"
            bot_text = re.sub(
                r"To know more, check out this link:\s*\[([^\]]+)\]\(([^)]+)\)",
                r"Learn more:\n\2",
                bot_text,
            )
            bot_text = re.sub(
                r"To know more, check out this link:\s*([^\n\s]+)",
                r"Learn more:\n\1",
                bot_text,
            )
        elif str(record.get("status")).lower() == "success":
            bot_text = "(No response generated)"
        else:
            bot_text = f"FAILED: {error_msg or 'Error occurred'}"

        # 5. Token Usage
        in_tokens = record.get("input_tokens")
        out_tokens = record.get("output_tokens")
        tot_tokens = record.get("total_tokens")
        in_tokens_display = f"{in_tokens:,}" if in_tokens is not None else "N/A"
        out_tokens_display = f"{out_tokens:,}" if out_tokens is not None else "N/A"
        tot_tokens_display = f"{tot_tokens:,}" if tot_tokens is not None else "N/A"

        # 6. Cost Breakdown
        in_cost = record.get("input_cost")
        out_cost = record.get("output_cost")
        tot_cost = record.get("total_request_cost")
        currency = record.get("currency", "USD")

        in_cost_str = f"${in_cost:.7f}" if in_cost is not None else "$0.0000000"
        out_cost_str = f"${out_cost:.7f}" if out_cost is not None else "$0.0000000"
        tot_cost_str = f"${tot_cost:.7f} {currency} (estimated)" if tot_cost is not None else f"$0.0000000 {currency} (estimated)"

        # 7. System Information
        env_raw = record.get("environment") or "Staging"
        env_display = env_raw.capitalize() if isinstance(env_raw, str) else str(env_raw)
        endpoint = record.get("endpoint") or "socketio.message"
        op_type = record.get("operation_type") or "chat_completion"
        retry_count = record.get("retry_count", 0)
        rate_limit_val = record.get("rate_limit_status")
        rate_limit_display = "Yes" if rate_limit_val else "No"

        header_title = f"API REQUEST LOG - {log_number}".center(80).rstrip()

        lines = [
            "=" * 80,
            header_title,
            "=" * 80,
            "",
            "REQUEST DETAILS",
            "---------------",
            f"Timestamp        : {ts_display}",
            f"Session ID       : {session_id}",
            f"Request ID       : {request_id}",
            f"Parent Request   : {parent_request_id}",
            f"Feature          : {feature}",
            f"Model            : {model}",
            f"Provider         : {provider}",
            f"Status           : {status_display}",
            f"Response Time    : {resp_time_display}",
            "",
            "",
            "USER QUERY",
            "----------",
            user_text,
            "",
            "",
            "BOT RESPONSE",
            "------------",
            bot_text,
            "",
            "",
            "TOKEN USAGE",
            "-----------",
            f"Input Tokens     : {in_tokens_display}",
            f"Output Tokens    : {out_tokens_display}",
            f"Total Tokens     : {tot_tokens_display}",
            "",
            "",
            "COST BREAKDOWN",
            "--------------",
            f"Input Cost       : {in_cost_str}",
            f"Output Cost      : {out_cost_str}",
            f"Total Cost       : {tot_cost_str}",
            "",
            "",
            "SYSTEM INFORMATION",
            "------------------",
            f"Environment      : {env_display}",
            f"Endpoint         : {endpoint}",
            f"Operation Type   : {op_type}",
            f"Retry Count      : {retry_count}",
            f"Rate Limit Hit   : {rate_limit_display}",
        ]

        if record.get("error_code") or record.get("error_message"):
            lines.extend([
                "",
                "",
                "ERROR DETAILS",
                "-------------",
                f"Error Code       : {record.get('error_code')}",
                f"Error Message    : {record.get('error_message')}",
            ])

        lines.extend(["", ""])
        return "\n".join(lines)

    def log(self, record: Dict[str, Any]) -> bool:
        """
        Thread-safe append of a human-readable text entry to the corresponding session file.
        Automatically calculates log entry index and manages dividers.
        Guarantees logging never raises exceptions to the caller.
        """
        try:
            self._ensure_dir()
            filename = self.get_filename_for_record(record)
            file_path = os.path.join(self.log_dir, filename)

            with _write_lock:
                # Count existing logs to determine log_number
                log_count = 0
                file_content = ""
                end_marker = "=" * 80 + "\n" + "END OF LOG".center(80).rstrip() + "\n" + "=" * 80 + "\n"
                
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            file_content = f.read()
                        # Count occurrences of "API REQUEST LOG -"
                        log_count = len(re.findall(r"API REQUEST LOG\s*-\s*\d+", file_content))
                    except Exception:
                        log_count = 0

                next_log_number = log_count + 1
                formatted_entry = self.format_entry(record, log_number=next_log_number)

                # If file exists and ends with the END OF LOG footer, strip it before appending
                if file_content:
                    # Strip any trailing END OF LOG block and whitespace
                    content_clean = re.sub(
                        r"={40,}\s*\n\s*END OF LOG\s*\n={40,}\s*$",
                        "",
                        file_content.rstrip()
                    ).rstrip()
                    # Reconstruct with the new entry and end footer
                    new_full_content = (content_clean + "\n\n\n" if content_clean else "") + formatted_entry + "\n" + end_marker
                    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                    mode = 0o644
                    fd = os.open(file_path, flags, mode)
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as f:
                            f.write(new_full_content)
                            f.flush()
                    except Exception:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                        raise
                else:
                    new_full_content = formatted_entry + "\n" + end_marker
                    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                    mode = 0o644
                    fd = os.open(file_path, flags, mode)
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as f:
                            f.write(new_full_content)
                            f.flush()
                    except Exception:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                        raise

            return True
        except Exception as e:
            op_logger.warning("Failed to write text log record: %s", e)
            return False


# Singleton instances
_logger_instance: Optional[JSONLApiLogger] = None
_text_logger_instance: Optional[TextApiLogger] = None


def get_logger() -> JSONLApiLogger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = JSONLApiLogger()
    return _logger_instance


def get_text_logger() -> TextApiLogger:
    global _text_logger_instance
    if _text_logger_instance is None:
        _text_logger_instance = TextApiLogger()
    return _text_logger_instance


def log_api_call(
    *,
    request_id: str,
    parent_request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_query: Optional[str] = None,
    bot_response: Optional[str] = None,
    provider: str = "google",
    model_name: str = "gemini-3.5-flash-lite",
    feature_name: str = "chat_response",
    operation_type: str = "chat_completion",
    endpoint: Optional[str] = "socketio.message",
    start_time: float,
    status: str = "success",
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cached_input_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
    usage_source: str = "provider_reported",
    usage_complete: bool = True,
    http_status: Optional[int] = 200,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    retry_count: int = 0,
    rate_limit_status: bool = False,
    prompt_version: Optional[str] = "1.0",
    application_version: Optional[str] = "1.0.0",
    request_type: Optional[str] = "general_query",
    user_type: Optional[str] = "website_visitor",
    tags: Optional[list] = None,
    latency_ms: Optional[int] = None,
    time_to_first_token_ms: Optional[int] = None,
) -> bool:
    """
    Main helper to construct and write API usage records.
    Writes human-readable text logs (year_date_uuid[10].txt) containing user question,
    bot response, token usage, cost, and all request details, and maintains structured JSONL logs.
    """
    try:
        now_ts = get_iso_timestamp()

        if latency_ms is None:
            latency_ms = int((time.time() - start_time) * 1000)

        # Cost calculation
        costs = calculate_cost(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

        record: Dict[str, Any] = {
            # A. Request identification
            "request_id": request_id,
            "parent_request_id": parent_request_id,
            "session_id": session_id,
            "timestamp": now_ts,
            "environment": CURRENT_ENV,
            "feature_name": feature_name,
            "operation_type": operation_type,
            "endpoint": endpoint,

            # B. User Query and Bot Response
            "user_query": user_query,
            "bot_response": bot_response,

            # C. Provider & model details
            "provider": provider,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "application_version": application_version,

            # D. Token usage
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens if total_tokens is not None else ((input_tokens + output_tokens) if (input_tokens is not None and output_tokens is not None) else None),
            "cached_input_tokens": cached_input_tokens,
            "reasoning_tokens": reasoning_tokens,
            "usage_source": usage_source,
            "usage_complete": usage_complete,

            # E. Cost tracking
            "input_cost": costs["input_cost"],
            "output_cost": costs["output_cost"],
            "cached_input_cost": costs["cached_input_cost"],
            "other_cost": costs["other_cost"],
            "total_request_cost": costs["total_request_cost"],
            "currency": costs["currency"],
            "pricing_version": costs["pricing_version"],
            "cost_status": costs["cost_status"],

            # F. Performance and errors
            "latency_ms": latency_ms,
            "time_to_first_token_ms": time_to_first_token_ms,
            "retry_count": retry_count,
            "status": status,
            "http_status": http_status,
            "error_code": sanitize_text(error_code),
            "error_message": sanitize_text(error_message),
            "rate_limit_status": rate_limit_status,

            # G. Classification
            "request_type": request_type,
            "user_type": user_type,
            "tags": tags or [],
        }

        # 1. Write human-readable text file (year_date_uuid[10].txt)
        txt_success = get_text_logger().log(record)

        # 2. Write structured JSONL file for analytics/reporting
        jsonl_success = get_logger().log(record)

        return txt_success or jsonl_success
    except Exception as e:
        op_logger.warning("Error creating log record: %s", e)
        return False
