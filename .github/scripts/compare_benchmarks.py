#!/usr/bin/env python3
"""Compare gungraun benchmark results and optionally post a PR comment."""

import argparse
import json
import subprocess
import sys


def benchmark_key(entry):
    """Unique key for matching benchmarks across runs."""
    key = entry["module_path"]
    if entry.get("id"):
        key += "::" + entry["id"]
    return key


def display_name(entry):
    """Human-readable benchmark name."""
    name = entry.get("function_name", entry["module_path"])
    if entry.get("id"):
        name += " (" + entry["id"] + ")"
    return name


def extract_ir(entry):
    """Extract instruction count (Ir) from a benchmark entry.

    Handles both the compact format (top-level 'ir' field) and the full
    gungraun schema (nested profiles[].summaries.total.summary.Callgrind.Ir).
    """
    # Compact format: ir is pre-extracted at top level
    ir_val = entry.get("ir")
    if ir_val is not None:
        if isinstance(ir_val, dict) and "Int" in ir_val:
            return ir_val["Int"]
        if isinstance(ir_val, int):
            return ir_val

    # Full schema format
    for profile in entry.get("profiles", []):
        if profile.get("tool") != "Callgrind":
            continue
        try:
            callgrind = profile["summaries"]["total"]["summary"]["Callgrind"]
        except (KeyError, TypeError):
            continue
        ir = callgrind.get("Ir")
        if ir is None:
            continue
        metrics = ir.get("metrics", {})
        if "Both" in metrics:
            val = metrics["Both"][0]
        elif "Left" in metrics:
            val = metrics["Left"]
        else:
            continue
        if isinstance(val, dict) and "Int" in val:
            return val["Int"]
        if isinstance(val, int):
            return val
    return None


def fmt(n):
    """Format a number with thousand separators."""
    return f"{n:,}"


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("baseline", help="Baseline JSON file")
    parser.add_argument("current", help="Current JSON file")
    parser.add_argument("--pr", type=int, help="PR number for commenting")
    parser.add_argument("--repo", help="GitHub repository (owner/name)")
    args = parser.parse_args()

    try:
        with open(args.baseline) as f:
            baseline_data = json.load(f)
        with open(args.current) as f:
            current_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not read benchmark files: {e}", file=sys.stderr)
        return

    # Build lookup maps: key -> (display_name, ir)
    baseline = {}
    for entry in baseline_data:
        key = benchmark_key(entry)
        ir = extract_ir(entry)
        if ir is not None:
            baseline[key] = (display_name(entry), ir)

    current = {}
    for entry in current_data:
        key = benchmark_key(entry)
        ir = extract_ir(entry)
        if ir is not None:
            current[key] = (display_name(entry), ir)

    if not current:
        print("No benchmark results found.", file=sys.stderr)
        return

    # Build comparison table
    all_keys = sorted(set(baseline) | set(current))
    rows = []
    for key in all_keys:
        b = baseline.get(key)
        c = current.get(key)
        if b and c:
            name = c[0]
            base_ir, curr_ir = b[1], c[1]
            diff_pct = ((curr_ir - base_ir) / base_ir) * 100 if base_ir else 0
            marker = ""
            if diff_pct >= 5:
                marker = " :arrow_up:"
            elif diff_pct <= -5:
                marker = " :arrow_down:"
            rows.append((name, fmt(base_ir), fmt(curr_ir), f"{diff_pct:+.2f}%{marker}"))
        elif b is None:
            rows.append((c[0], "\u2014", fmt(c[1]), "new"))
        else:
            rows.append((b[0], fmt(b[1]), "\u2014", "removed"))

    # Format markdown
    lines = [
        "## Benchmark Comparison (Ir)",
        "",
        "| Benchmark | Baseline | Current | Change |",
        "|-----------|----------|---------|--------|",
    ]
    for name, base, curr, change in rows:
        lines.append(f"| `{name}` | {base} | {curr} | {change} |")

    big_changes = sum(
        1 for _, _, _, ch in rows if ":arrow_up:" in ch or ":arrow_down:" in ch
    )
    lines.append("")
    if big_changes:
        lines.append(f"**{big_changes} benchmark(s) changed by \u22655%.**")
    else:
        lines.append("No benchmarks changed by \u22655%.")

    body = "\n".join(lines)
    print(body)

    if args.pr and args.repo:
        try:
            subprocess.run(
                [
                    "gh", "pr", "comment", str(args.pr),
                    "--repo", args.repo,
                    "--body", body,
                ],
                check=True,
            )
            print(f"Posted comment to PR #{args.pr}", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to post PR comment: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
