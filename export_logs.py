#!/usr/bin/env python3
"""
Command-line utility to export JSONL API usage logs into CSV or Excel format.

Usage:
    python export_logs.py --format csv
    python export_logs.py --format excel
    python export_logs.py --start-date 2026-10-01 --end-date 2026-10-22 --format excel
    python export_logs.py --feature chat_response --format csv
    python export_logs.py --provider google --format excel
    python export_logs.py --status failure --format csv
    python export_logs.py --format excel --output exports/api_usage_report.xlsx
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Try importing pandas and openpyxl, provide clear instructions if missing
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import openpyxl
except ImportError:
    openpyxl = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_FILE = os.getenv("API_LOG_FILE", os.path.join(BASE_DIR, "logs", "api_usage_logs.jsonl"))
DEFAULT_EXPORT_DIR = os.getenv("API_EXPORT_DIR", os.path.join(BASE_DIR, "exports"))


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parses date string (YYYY-MM-DD or ISO format) into a datetime object."""
    if not date_str:
        return None
    date_str = date_str.strip()
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    try:
        # Fallback to fromisoformat
        return datetime.fromisoformat(date_str)
    except Exception:
        raise ValueError(f"Could not parse date '{date_str}'. Expected format: YYYY-MM-DD (e.g. 2026-10-22)")


def read_jsonl_records(file_path: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Reads records from a JSONL file without loading the entire contents as a single blob.
    Returns (valid_records, valid_count, malformed_count).
    """
    valid_records: List[Dict[str, Any]] = []
    valid_count = 0
    malformed_count = 0

    if not os.path.exists(file_path):
        return valid_records, 0, 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                if isinstance(record, dict):
                    valid_records.append(record)
                    valid_count += 1
                else:
                    malformed_count += 1
                    sys.stderr.write(f"Warning: Line {line_num} is not a valid JSON object. Skipped.\n")
            except Exception as e:
                malformed_count += 1
                sys.stderr.write(f"Warning: Line {line_num} malformed JSON ({e}). Skipped.\n")

    return valid_records, valid_count, malformed_count


def filter_records(
    records: List[Dict[str, Any]],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    feature: Optional[str] = None,
    provider: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Applies filters to the records list."""
    filtered = []

    for r in records:
        # Feature filter
        if feature and r.get("feature_name", "").lower() != feature.lower():
            continue

        # Provider filter
        if provider and r.get("provider", "").lower() != provider.lower():
            continue

        # Status filter
        if status and r.get("status", "").lower() != status.lower():
            continue

        # Date range filter
        ts_str = r.get("timestamp")
        if start_date or end_date:
            if not ts_str:
                continue
            try:
                # Parse timestamp and make comparison timezone-naive or matching
                record_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                # Strip tz info for simpler comparison if user provided naive start/end
                if start_date and start_date.tzinfo is None and record_dt.tzinfo is not None:
                    record_dt_comp = record_dt.replace(tzinfo=None)
                elif start_date and start_date.tzinfo is not None and record_dt.tzinfo is None:
                    record_dt_comp = record_dt.astimezone(start_date.tzinfo)
                else:
                    record_dt_comp = record_dt

                if start_date:
                    start_comp = start_date.replace(tzinfo=None) if record_dt_comp.tzinfo is None else start_date
                    if record_dt_comp < start_comp:
                        continue

                if end_date:
                    end_comp = end_date.replace(tzinfo=None) if record_dt_comp.tzinfo is None else end_date
                    # If end_date is date-only (00:00:00), treat as end of day
                    if end_comp.hour == 0 and end_comp.minute == 0 and end_comp.second == 0:
                        end_comp = end_comp.replace(hour=23, minute=59, second=59)
                    if record_dt_comp > end_comp:
                        continue
            except Exception:
                continue

        filtered.append(r)

    return filtered


def export_data(
    records: List[Dict[str, Any]],
    export_format: str,
    output_path: str,
    overwrite: bool = False,
) -> str:
    """Exports structured records into CSV or Excel using pandas."""
    if pd is None:
        raise RuntimeError("pandas is required for export. Install with: pip install pandas")

    if export_format == "excel" and openpyxl is None:
        raise RuntimeError("openpyxl is required for Excel export. Install with: pip install openpyxl")

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Destination file '{output_path}' already exists. Use --overwrite to replace it."
        )

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, mode=0o750, exist_ok=True)

    df = pd.DataFrame(records)

    # Flatten tags list to comma-separated string for tabular representation
    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(lambda t: ", ".join(t) if isinstance(t, list) else str(t or ""))

    if export_format == "csv":
        df.to_csv(output_path, index=False, encoding="utf-8")
    elif export_format == "excel":
        df.to_excel(output_path, index=False, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported format: {export_format}")

    # Set safe file permissions
    try:
        os.chmod(output_path, 0o640)
    except Exception:
        pass

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export AI API usage JSONL logs to CSV or Excel format with filtering options."
    )
    parser.add_argument(
        "--format",
        choices=["csv", "excel"],
        default="csv",
        help="Export format (default: csv).",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_LOG_FILE,
        help=f"Path to input .jsonl log file (default: {DEFAULT_LOG_FILE}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for exported file. Defaults to 'exports/api_usage_report_<timestamp>.<ext>'.",
    )
    parser.add_argument(
        "--start-date",
        help="Filter records on or after this date (format: YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        help="Filter records on or before this date (format: YYYY-MM-DD).",
    )
    parser.add_argument(
        "--feature",
        help="Filter records by feature name (e.g., chat_response, welcome_greeting).",
    )
    parser.add_argument(
        "--provider",
        help="Filter records by AI provider (e.g., google).",
    )
    parser.add_argument(
        "--status",
        help="Filter records by status (e.g., success, failure, timeout).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination file if it already exists.",
    )

    args = parser.parse_args()

    input_file = os.path.abspath(args.input)
    if not os.path.exists(input_file):
        print(f"Log file does not exist yet at '{input_file}'.")
        print("Once the chatbot runs and receives queries, API calls will be recorded here automatically.")
        sys.exit(0)

    try:
        start_dt = parse_date(args.start_date)
        end_dt = parse_date(args.end_date)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Reading log records from: {input_file}")
    records, valid_count, malformed_count = read_jsonl_records(input_file)
    print(f"Total lines parsed: {valid_count + malformed_count} (Valid: {valid_count}, Malformed/Skipped: {malformed_count})")

    if not records:
        print("No valid records found to export.")
        sys.exit(0)

    filtered = filter_records(
        records,
        start_date=start_dt,
        end_date=end_dt,
        feature=args.feature,
        provider=args.provider,
        status=args.status,
    )
    print(f"Records matching filters: {len(filtered)}")

    if not filtered:
        print("No records matched the specified filter criteria.")
        sys.exit(0)

    # Determine output file path
    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        ext = "xlsx" if args.format == "excel" else "csv"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(DEFAULT_EXPORT_DIR, f"api_usage_report_{ts}.{ext}")

    try:
        exported_file = export_data(
            records=filtered,
            export_format=args.format,
            output_path=out_path,
            overwrite=args.overwrite,
        )
        print(f"Successfully exported {len(filtered)} records to: {exported_file}")
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
