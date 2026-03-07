#!/usr/bin/env python3
"""Compare gungraun benchmark results and optionally post a PR comment.

Reads a single JSON file where each entry contains ir_old/ir_new pairs
(extracted from gungraun's Both[old, new] cache output).
"""

import argparse
import json
import subprocess
import sys


def display_name(entry):
    """Human-readable benchmark name."""
    name = entry.get("function_name", entry["module_path"])
    if entry.get("id"):
        name += " (" + entry["id"] + ")"
    return name


def unwrap_ir(val):
    """Unwrap an IR value that may be a raw int or {"Int": n}."""
    if val is None:
        return None
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
    parser.add_argument("results", help="JSON file with ir_old/ir_new pairs")
    parser.add_argument("--pr", type=int, help="PR number for commenting")
    parser.add_argument("--repo", help="GitHub repository (owner/name)")
    args = parser.parse_args()

    try:
        with open(args.results) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not read benchmark file: {e}", file=sys.stderr)
        return

    rows = []
    for entry in data:
        name = display_name(entry)
        ir_old = unwrap_ir(entry.get("ir_old"))
        ir_new = unwrap_ir(entry.get("ir_new"))

        if ir_new is None:
            continue

        if ir_old is not None:
            diff_pct = ((ir_new - ir_old) / ir_old) * 100 if ir_old else 0
            marker = ""
            if diff_pct >= 5:
                marker = " :arrow_up:"
            elif diff_pct <= -5:
                marker = " :arrow_down:"
            rows.append((name, fmt(ir_old), fmt(ir_new), f"{diff_pct:+.2f}%{marker}"))
        else:
            rows.append((name, "\u2014", fmt(ir_new), "new"))

    if not rows:
        print("No benchmark results found.", file=sys.stderr)
        return

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
