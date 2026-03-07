#!/usr/bin/env python3
"""Compare hyperfine wall-clock benchmark results and optionally post a PR comment.

Modes:
  --mode pr       Parse hyperfine JSONs with two commands (main vs PR), build
                  comparison table, post PR comment.
  --mode store    Parse hyperfine JSONs with one command, aggregate into a single
                  JSON keyed by benchmark name, print to stdout for storage.
"""

import argparse
import json
import os
import subprocess
import sys


def load_hyperfine_results(results_dir):
    """Load all hyperfine JSON files from a directory.

    Returns a dict: filename_stem -> list of command results.
    """
    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        stem = fname.removesuffix(".json")
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        results[stem] = data.get("results", [])
    return results


def compare_pr(results_dir, pr_number, repo):
    """Compare main vs PR results and post a markdown table."""
    all_results = load_hyperfine_results(results_dir)

    rows = []
    for bench_name, commands in sorted(all_results.items()):
        if len(commands) < 2:
            continue
        main_median = commands[0]["median"]
        pr_median = commands[1]["median"]
        if main_median > 0:
            diff_pct = ((pr_median - main_median) / main_median) * 100
        else:
            diff_pct = 0.0

        marker = ""
        if diff_pct >= 15:
            marker = " :warning:"
        elif diff_pct <= -15:
            marker = " :rocket:"

        rows.append((
            bench_name,
            f"{main_median * 1000:.1f}ms",
            f"{pr_median * 1000:.1f}ms",
            f"{diff_pct:+.1f}%{marker}",
        ))

    lines = [
        "## Wall-Clock Benchmark Comparison",
        "",
        "| Benchmark | Main | PR | Change |",
        "|-----------|------|-----|--------|",
    ]
    for name, base, current, change in rows:
        lines.append(f"| `{name}` | {base} | {current} | {change} |")

    flagged = sum(1 for _, _, _, ch in rows if ":warning:" in ch or ":rocket:" in ch)
    lines.append("")
    if flagged:
        lines.append(f"**{flagged} benchmark(s) changed by \u226515%.**")
    else:
        lines.append("No benchmarks changed by \u226515%.")
    lines.append("")
    lines.append("_Wall-clock times from hyperfine (median of 10 runs). "
                 "Results may vary due to system load._")

    body = "\n".join(lines)
    print(body)

    if pr_number and repo:
        try:
            subprocess.run(
                [
                    "gh", "pr", "comment", str(pr_number),
                    "--repo", repo,
                    "--body", body,
                ],
                check=True,
            )
            print(f"Posted comment to PR #{pr_number}", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to post PR comment: {e}", file=sys.stderr)


def store_results(results_dir):
    """Aggregate single-command hyperfine results into one JSON object."""
    all_results = load_hyperfine_results(results_dir)

    aggregated = {}
    for bench_name, commands in sorted(all_results.items()):
        if not commands:
            continue
        aggregated[bench_name] = {
            "median": commands[0]["median"],
            "mean": commands[0]["mean"],
            "stddev": commands[0]["stddev"],
            "min": commands[0]["min"],
            "max": commands[0]["max"],
        }

    print(json.dumps(aggregated, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Compare hyperfine benchmark results")
    parser.add_argument("--mode", choices=["pr", "store"], required=True,
                        help="pr: compare and comment; store: aggregate for storage")
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing hyperfine JSON files")
    parser.add_argument("--pr", type=int, help="PR number (pr mode only)")
    parser.add_argument("--repo", help="GitHub repository owner/name (pr mode only)")
    args = parser.parse_args()

    if args.mode == "pr":
        compare_pr(args.results_dir, args.pr, args.repo)
    elif args.mode == "store":
        store_results(args.results_dir)


if __name__ == "__main__":
    main()
