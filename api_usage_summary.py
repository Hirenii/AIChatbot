#!/usr/bin/env python3
"""
Command-line utility to summarize AI API usage, token consumption, estimated costs,
latency metrics, and failure rates from JSONL usage logs.

Usage:
    python api_usage_summary.py
    python api_usage_summary.py --input logs/api_usage_logs.jsonl
    python api_usage_summary.py --export-csv exports/summary.csv
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_FILE = os.getenv("API_LOG_FILE", os.path.join(BASE_DIR, "logs", "api_usage_logs.jsonl"))


def read_jsonl_records(file_path: str) -> Tuple[List[Dict[str, Any]], int]:
    valid = []
    malformed = 0
    if not os.path.exists(file_path):
        return valid, malformed

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                if isinstance(record, dict):
                    valid.append(record)
                else:
                    malformed += 1
            except Exception:
                malformed += 1
    return valid, malformed


def generate_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_calls = len(records)
    success_calls = sum(1 for r in records if r.get("status") == "success")
    failed_calls = sum(1 for r in records if r.get("status") in ["failure", "timeout", "error"])
    retried_calls = sum(r.get("retry_count", 0) for r in records)

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    missing_token_count = 0

    total_estimated_cost = 0.0
    missing_cost_count = 0

    latencies: List[int] = []

    by_provider: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0, "failures": 0})
    by_model: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0, "failures": 0})
    by_feature: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0, "failures": 0})
    by_day: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})
    by_month: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})

    # Group by parent_request_id (chatbot workflow) for average cost per interaction
    workflows: Dict[str, float] = defaultdict(float)

    for r in records:
        prov = r.get("provider") or "unknown"
        mod = r.get("model_name") or "unknown"
        feat = r.get("feature_name") or "unknown"
        status = r.get("status") or "unknown"
        is_fail = status in ["failure", "timeout", "error"]

        # Token counting
        in_t = r.get("input_tokens")
        out_t = r.get("output_tokens")
        tot_t = r.get("total_tokens")

        if in_t is not None:
            total_input_tokens += in_t
        if out_t is not None:
            total_output_tokens += out_t
        if tot_t is not None:
            total_tokens += tot_t
        elif in_t is not None and out_t is not None:
            total_tokens += (in_t + out_t)
        else:
            missing_token_count += 1

        call_toks = (tot_t or ((in_t or 0) + (out_t or 0)))

        # Cost tracking
        cost = r.get("total_request_cost")
        if cost is not None:
            total_estimated_cost += cost
            p_id = r.get("parent_request_id")
            if p_id:
                workflows[p_id] += cost
        else:
            missing_cost_count += 1

        # Latency
        lat = r.get("latency_ms")
        if lat is not None:
            latencies.append(lat)

        # Aggregations
        by_provider[prov]["calls"] += 1
        by_provider[prov]["tokens"] += call_toks
        by_provider[prov]["cost"] += (cost or 0.0)
        if is_fail:
            by_provider[prov]["failures"] += 1

        by_model[mod]["calls"] += 1
        by_model[mod]["tokens"] += call_toks
        by_model[mod]["cost"] += (cost or 0.0)
        if is_fail:
            by_model[mod]["failures"] += 1

        by_feature[feat]["calls"] += 1
        by_feature[feat]["tokens"] += call_toks
        by_feature[feat]["cost"] += (cost or 0.0)
        if is_fail:
            by_feature[feat]["failures"] += 1

        ts_str = r.get("timestamp")
        if ts_str:
            day_str = ts_str[:10]  # YYYY-MM-DD
            month_str = ts_str[:7]  # YYYY-MM
            by_day[day_str]["calls"] += 1
            by_day[day_str]["tokens"] += call_toks
            by_day[day_str]["cost"] += (cost or 0.0)

            by_month[month_str]["calls"] += 1
            by_month[month_str]["tokens"] += call_toks
            by_month[month_str]["cost"] += (cost or 0.0)

    avg_lat = statistics.mean(latencies) if latencies else 0.0
    med_lat = statistics.median(latencies) if latencies else 0.0
    fail_rate = (failed_calls / total_calls * 100) if total_calls > 0 else 0.0
    avg_workflow_cost = (statistics.mean(workflows.values())) if workflows else 0.0

    return {
        "total_calls": total_calls,
        "success_calls": success_calls,
        "failed_calls": failed_calls,
        "retry_count": retried_calls,
        "failure_rate_pct": fail_rate,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "missing_token_count": missing_token_count,
        "total_estimated_cost_usd": round(total_estimated_cost, 6),
        "missing_cost_count": missing_cost_count,
        "avg_latency_ms": round(avg_lat, 2),
        "median_latency_ms": round(med_lat, 2),
        "avg_cost_per_workflow_usd": round(avg_workflow_cost, 6),
        "workflows_count": len(workflows),
        "by_provider": dict(by_provider),
        "by_model": dict(by_model),
        "by_feature": dict(by_feature),
        "by_day": dict(sorted(by_day.items())),
        "by_month": dict(sorted(by_month.items())),
    }


def print_summary(s: Dict[str, Any], malformed_count: int):
    print("\n" + "=" * 70)
    print("           AI API USAGE & COST MONITORING SUMMARY REPORT")
    print("=" * 70)

    print("\n[1] Overall API Activity & Performance:")
    print(f"  • Total API Calls:        {s['total_calls']:,}")
    print(f"  • Successful Calls:       {s['success_calls']:,}")
    print(f"  • Failed/Timeout Calls:   {s['failed_calls']:,} ({s['failure_rate_pct']:.2f}% failure rate)")
    print(f"  • Retries Count:          {s['retry_count']:,}")
    print(f"  • Average Latency:        {s['avg_latency_ms']} ms")
    print(f"  • Median Latency:         {s['median_latency_ms']} ms")
    if malformed_count > 0:
        print(f"  • Warning: {malformed_count} malformed log records skipped in analysis.")

    print("\n[2] Token Consumption:")
    print(f"  • Total Input Tokens:     {s['total_input_tokens']:,}")
    print(f"  • Total Output Tokens:    {s['total_output_tokens']:,}")
    print(f"  • Total Tokens:           {s['total_tokens']:,}")
    if s["missing_token_count"] > 0:
        print(f"  • Note: {s['missing_token_count']} call(s) have incomplete/missing token data (e.g. failed requests).")

    print("\n[3] Estimated Costs (USD):")
    print(f"  • Total Estimated Cost:   ${s['total_estimated_cost_usd']:.6f} USD")
    if s["workflows_count"] > 0:
        print(f"  • Avg Cost per Workflow:  ${s['avg_cost_per_workflow_usd']:.6f} USD across {s['workflows_count']} chat workflows")
    if s["missing_cost_count"] > 0:
        print(f"  • Note: {s['missing_cost_count']} call(s) have uncalculated/unavailable costs (e.g. unpriced models/failures).")

    print("\n[4] Usage by AI Provider:")
    for prov, data in s["by_provider"].items():
        print(f"  • {prov:<15} Calls: {data['calls']:<6} Tokens: {data['tokens']:<10} Cost: ${data['cost']:.6f} USD  Failures: {data['failures']}")

    print("\n[5] Usage by Model:")
    for mod, data in s["by_model"].items():
        print(f"  • {mod:<25} Calls: {data['calls']:<6} Tokens: {data['tokens']:<10} Cost: ${data['cost']:.6f} USD  Failures: {data['failures']}")

    print("\n[6] Usage by Feature / Operation:")
    for feat, data in s["by_feature"].items():
        print(f"  • {feat:<22} Calls: {data['calls']:<6} Tokens: {data['tokens']:<10} Cost: ${data['cost']:.6f} USD  Failures: {data['failures']}")

    print("\n[7] Monthly Consumption:")
    for m, data in s["by_month"].items():
        print(f"  • {m:<10} Calls: {data['calls']:<6} Tokens: {data['tokens']:<10} Est. Cost: ${data['cost']:.6f} USD")

    print("\n[8] Daily Consumption (Recent):")
    recent_days = list(s["by_day"].items())[-7:]
    for d, data in recent_days:
        print(f"  • {d:<12} Calls: {data['calls']:<6} Tokens: {data['tokens']:<10} Est. Cost: ${data['cost']:.6f} USD")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate AI API usage, token, latency, and cost summary.")
    parser.add_argument("--input", default=DEFAULT_LOG_FILE, help=f"Path to .jsonl file (default: {DEFAULT_LOG_FILE})")
    parser.add_argument("--export-csv", default=None, help="Optional path to export feature & model summary table to CSV")

    args = parser.parse_args()
    log_file = os.path.abspath(args.input)

    if not os.path.exists(log_file):
        print(f"Log file not found at '{log_file}'. No records to summarize.")
        sys.exit(0)

    records, malformed_count = read_jsonl_records(log_file)
    if not records:
        print(f"No valid records found in '{log_file}'.")
        sys.exit(0)

    summary = generate_summary(records)
    print_summary(summary, malformed_count)

    if args.export_csv:
        try:
            import csv
            os.makedirs(os.path.dirname(os.path.abspath(args.export_csv)), exist_ok=True)
            with open(args.export_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["category", "name", "calls", "tokens", "estimated_cost_usd", "failures"])
                for k, v in summary["by_model"].items():
                    writer.writerow(["model", k, v["calls"], v["tokens"], round(v["cost"], 6), v["failures"]])
                for k, v in summary["by_feature"].items():
                    writer.writerow(["feature", k, v["calls"], v["tokens"], round(v["cost"], 6), v["failures"]])
            print(f"Exported category summary CSV to: {args.export_csv}")
        except Exception as e:
            print(f"Failed to export CSV: {e}")


if __name__ == "__main__":
    main()
