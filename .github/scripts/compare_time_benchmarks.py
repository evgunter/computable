#!/usr/bin/env python3
"""Compare hyperfine wall-clock benchmark results and optionally post a PR comment.

Modes:
  --mode pr               Parse hyperfine JSONs with two commands (main vs PR),
                          build comparison table, post PR comment.
  --mode store            Parse hyperfine JSONs, aggregate into a single JSON
                          keyed by benchmark name, print to stdout for storage.
                          Use --command-index to select which command entry
                          (default -1 = last).
  --mode store-comparison Parse hyperfine JSONs with two commands (base vs head),
                          aggregate into a comparison JSON with metadata, print
                          to stdout for storage.  Requires --base-sha and
                          --head-sha.
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


def store_results(results_dir, command_index=-1):
    """Aggregate hyperfine results into one JSON object.

    *command_index* selects which command entry to use from each hyperfine
    JSON file (default ``-1``, i.e. the last command).  When hyperfine is
    run with two commands (A/B comparison on main pushes), ``-1`` picks the
    current commit (the second command).
    """
    all_results = load_hyperfine_results(results_dir)

    aggregated = {}
    for bench_name, commands in sorted(all_results.items()):
        if not commands:
            continue
        cmd = commands[command_index]
        aggregated[bench_name] = {
            "median": cmd["median"],
            "mean": cmd["mean"],
            "stddev": cmd["stddev"],
            "min": cmd["min"],
            "max": cmd["max"],
        }

    print(json.dumps(aggregated, indent=2))


def store_comparison(results_dir, base_sha, head_sha):
    """Aggregate two-command hyperfine results into a comparison JSON.

    Each hyperfine file is expected to have two command entries:
    commands[0] = base (previous commit), commands[1] = head (current commit).

    Prints JSON to stdout with the structure::

        {
            "base_sha": "...",
            "head_sha": "...",
            "benchmarks": {
                "<name>": {
                    "base": {"median": ..., "mean": ..., "stddev": ..., "min": ..., "max": ..., "times": [...]},
                    "head": {"median": ..., "mean": ..., "stddev": ..., "min": ..., "max": ..., "times": [...]}
                }
            }
        }
    """
    all_results = load_hyperfine_results(results_dir)

    benchmarks = {}
    for bench_name, commands in sorted(all_results.items()):
        if len(commands) < 2:
            continue

        def _extract(cmd):
            entry = {
                "median": cmd["median"],
                "mean": cmd["mean"],
                "stddev": cmd["stddev"],
                "min": cmd["min"],
                "max": cmd["max"],
            }
            if "times" in cmd:
                entry["times"] = cmd["times"]
            return entry

        benchmarks[bench_name] = {
            "base": _extract(commands[0]),
            "head": _extract(commands[1]),
        }

    output = {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "benchmarks": benchmarks,
    }
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Compare hyperfine benchmark results")
    parser.add_argument("--mode", choices=["pr", "store", "store-comparison"],
                        required=True,
                        help="pr: compare and comment; store: aggregate for storage; "
                             "store-comparison: aggregate A/B comparison for storage")
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing hyperfine JSON files")
    parser.add_argument("--pr", type=int, help="PR number (pr mode only)")
    parser.add_argument("--repo", help="GitHub repository owner/name (pr mode only)")
    parser.add_argument("--base-sha", help="Base commit SHA (store-comparison mode)")
    parser.add_argument("--head-sha", help="Head commit SHA (store-comparison mode)")
    parser.add_argument("--command-index", type=int, default=-1,
                        help="Which command index to store (store mode, default -1)")
    args = parser.parse_args()

    if args.mode == "pr":
        compare_pr(args.results_dir, args.pr, args.repo)
    elif args.mode == "store":
        store_results(args.results_dir, command_index=args.command_index)
    elif args.mode == "store-comparison":
        if not args.base_sha or not args.head_sha:
            parser.error("--base-sha and --head-sha are required for store-comparison mode")
        store_comparison(args.results_dir, args.base_sha, args.head_sha)


if __name__ == "__main__":
    main()
