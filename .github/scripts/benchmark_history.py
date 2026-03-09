#!/usr/bin/env python3
"""Read stored benchmark data from the benchmark-data branch and produce a
chronological summary report.

Uses ``git show`` to read per-commit result files and ``git log`` to map SHAs
to commit metadata.  Output is either Markdown (default) or CSV.
"""

import argparse
import csv
import io
import json
import subprocess
import sys


# ---------------------------------------------------------------------------
# Precision mapping
# ---------------------------------------------------------------------------
# Ir results use "precision_0" .. "precision_4".
# Time results use suffixes _1, _4, _16, _64, _256 (matching 4^N).
PRECISION_SUFFIX = {0: "_1", 1: "_4", 2: "_16", 3: "_64", 4: "_256"}

# Key benchmarks to track, expressed as (display_name, ir_key, time_key).
# ir_key  = (function_name, id) used in the Ir JSON
# time_key = key in the time-results JSON (may be None if no wall-clock data)
KEY_BENCHMARKS = []

def _add(display, func, prec, time_base):
    ir_id = f"precision_{prec}" if prec is not None else None
    time_key = (f"{time_base}{PRECISION_SUFFIX[prec]}" if prec is not None and time_base
                else time_base)
    KEY_BENCHMARKS.append((display, func, ir_id, time_key))

# sin (p0, p3, p4)
for p in (0, 3, 4):
    _add(f"sin (p{p})", "bench_sin", p, "sin")

# integer_roots (p0, p3, p4)
for p in (0, 3, 4):
    _add(f"integer_roots (p{p})", "bench_integer_roots", p, "integer_roots")

# inv (p0, p3, p4)
for p in (0, 3, 4):
    _add(f"inv (p{p})", "bench_inv", p, "inv")

# pi_refinement (p3, p4) — no wall-clock data
for p in (3, 4):
    _add(f"pi_refinement (p{p})", "bench_pi_refinement", p, None)

# complex — no precision sweep
_add("complex", "bench_complex", None, "complex")

# summation — no precision sweep
_add("summation", "bench_summation", None, "summation")

# sqrt2_plus_pi (p3, p4) — no wall-clock data
for p in (3, 4):
    _add(f"sqrt2_plus_pi (p{p})", "bench_sqrt2_plus_pi", p, None)

# sin_1pi (p3, p4) — no wall-clock data
for p in (3, 4):
    _add(f"sin_1pi (p{p})", "bench_sin_1pi", p, None)

# sin_100pi (p3, p4) — no wall-clock data
for p in (3, 4):
    _add(f"sin_100pi (p{p})", "bench_sin_100pi", p, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git(*args):
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True, text=True, check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def unwrap_ir(val):
    """Unwrap an IR value that may be a raw int or {"Int": n}."""
    if val is None:
        return None
    if isinstance(val, dict) and "Int" in val:
        return val["Int"]
    if isinstance(val, int):
        return val
    return None


def display_name(entry):
    """Human-readable benchmark name (matches compare_benchmarks.py)."""
    name = entry.get("function_name", entry["module_path"])
    if entry.get("id"):
        name += " (" + entry["id"] + ")"
    return name


def fmt_ir(n):
    """Format an instruction count with thousand separators."""
    if n is None:
        return "-"
    return f"{n:,}"


def fmt_time(seconds):
    """Format a wall-clock time in a human-friendly way."""
    if seconds is None:
        return "-"
    if seconds < 0.01:
        return f"{seconds * 1000:.3f}ms"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


def pct_change(old, new):
    """Compute percentage change, returning None if either value is missing."""
    if old is None or new is None or old == 0:
        return None
    return ((new - old) / old) * 100


def short_sha(sha):
    return sha[:7]


def short_desc(msg, max_len=30):
    if len(msg) > max_len:
        return msg[:max_len - 3] + "..."
    return msg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_files():
    """Return (ir_shas, time_shas, comparison_pairs).

    *ir_shas* and *time_shas* are sets of commit SHAs with stored results.
    *comparison_pairs* is a list of ``(base_sha, head_sha)`` tuples parsed
    from ``time-comparisons/{base}_vs_{head}.json`` filenames.
    """
    out = git("ls-tree", "-r", "--name-only", "origin/benchmark-data")
    if not out:
        print("Error: cannot list origin/benchmark-data", file=sys.stderr)
        sys.exit(1)

    ir_shas = set()
    time_shas = set()
    comparison_pairs = []
    for line in out.strip().splitlines():
        if line.startswith("results/") and line.endswith(".json"):
            name = line.removeprefix("results/").removesuffix(".json")
            if name != "latest":
                ir_shas.add(name)
        elif line.startswith("time-results/") and line.endswith(".json"):
            name = line.removeprefix("time-results/").removesuffix(".json")
            if name != "latest":
                time_shas.add(name)
        elif line.startswith("time-comparisons/") and line.endswith(".json"):
            name = line.removeprefix("time-comparisons/").removesuffix(".json")
            if "_vs_" in name:
                base, head = name.split("_vs_", 1)
                comparison_pairs.append((base, head))
    return ir_shas, time_shas, comparison_pairs


def load_commit_map():
    """Return {sha: (iso_date, subject)} from ``git log main``."""
    out = git("log", "main", "--format=%H %aI %s")
    if not out:
        return {}
    commit_map = {}
    for line in out.strip().splitlines():
        parts = line.split(" ", 2)
        if len(parts) >= 3:
            sha, date, subject = parts
            commit_map[sha] = (date, subject)
    return commit_map


def load_ir(sha):
    """Load Ir results for a commit.

    Returns {(function_name, id_or_None): ir_new_int}.
    """
    raw = git("show", f"origin/benchmark-data:results/{sha}.json")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    result = {}
    for entry in data:
        func = entry.get("function_name", entry.get("module_path", ""))
        bench_id = entry.get("id")
        ir_new = unwrap_ir(entry.get("ir_new"))
        if ir_new is not None:
            result[(func, bench_id)] = ir_new
    return result


def load_time(sha):
    """Load wall-clock results for a commit.

    Returns {bench_name: median_seconds}.
    """
    raw = git("show", f"origin/benchmark-data:time-results/{sha}.json")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return {k: v.get("median") for k, v in data.items()}


def load_comparison(base_sha, head_sha):
    """Load an A/B comparison file.

    Returns ``{bench_name: {"base": {...}, "head": {...}}}`` or ``{}``
    on failure.
    """
    raw = git("show",
              f"origin/benchmark-data:time-comparisons/{base_sha}_vs_{head_sha}.json")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data.get("benchmarks", {})


# ---------------------------------------------------------------------------
# Timeline construction
# ---------------------------------------------------------------------------

def build_timeline(commit_map, ir_shas, time_shas, comparison_pairs=None,
                   last_n=None):
    """Build a chronological list of commits with benchmark data.

    Returns [(sha, date, subject, ir_data, time_data, cmp_data)] oldest-first.

    *cmp_data* is the A/B comparison dict for this commit (as head) vs its
    predecessor (as base), or ``{}`` when no comparison file exists.
    """
    if comparison_pairs is None:
        comparison_pairs = []

    # Build a lookup: head_sha -> {base_sha: True}
    cmp_by_head = {}
    for base, head in comparison_pairs:
        cmp_by_head.setdefault(head, []).append(base)

    # Only include commits that appear in git log main
    all_shas = ir_shas | time_shas
    entries = []
    for sha in all_shas:
        if sha in commit_map:
            date, subject = commit_map[sha]
            entries.append((sha, date, subject))

    # Sort by date (oldest first)
    entries.sort(key=lambda e: e[1])

    if last_n is not None:
        entries = entries[-last_n:]

    # Build a set of SHAs in the timeline for quick predecessor lookup
    sha_list = [e[0] for e in entries]
    sha_index = {s: i for i, s in enumerate(sha_list)}

    timeline = []
    for sha, date, subject in entries:
        ir_data = load_ir(sha) if sha in ir_shas else {}
        time_data = load_time(sha) if sha in time_shas else {}

        # Find comparison data where this commit is the head and the base
        # is the previous commit in the timeline
        cmp_data = {}
        idx = sha_index.get(sha)
        if idx is not None and idx > 0:
            prev_sha = sha_list[idx - 1]
            if sha in cmp_by_head:
                for base in cmp_by_head[sha]:
                    if base == prev_sha:
                        cmp_data = load_comparison(base, sha)
                        break

        timeline.append((sha, date, subject, ir_data, time_data, cmp_data))

    return timeline


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_benchmarks(benchmarks, name_filter):
    """Filter KEY_BENCHMARKS by a substring match on display name."""
    if not name_filter:
        return benchmarks
    return [b for b in benchmarks if name_filter.lower() in b[0].lower()]


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def commit_header(sha, subject):
    return f"`{short_sha(sha)}` {short_desc(subject)}"


def markdown_report(timeline, benchmarks, ir_threshold, time_threshold):
    lines = []
    lines.append("# Benchmark History Report")
    lines.append("")

    if not timeline:
        lines.append("No benchmark data found.")
        return "\n".join(lines)

    # --- Wall-clock median table ---
    time_benchmarks = [b for b in benchmarks if b[3] is not None]
    time_commits = [(sha, subj, td) for sha, _, subj, _, td, _ in timeline if td]

    if time_benchmarks and time_commits:
        lines.append("## Wall-Clock Median Times")
        lines.append("")

        header = ["Benchmark"] + [commit_header(s, subj)
                                   for s, subj, _ in time_commits]
        sep = ["---"] * len(header)
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")

        for display, _, _, time_key in time_benchmarks:
            row = [f"`{display}`"]
            for _, _, td in time_commits:
                val = td.get(time_key)
                row.append(fmt_time(val))
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("_Wall-clock times from hyperfine (median of 10 runs). "
                      "Results may vary due to system load._")
        lines.append("")

    # --- Ir counts table ---
    ir_commits = [(sha, subj, ird) for sha, _, subj, ird, _, _ in timeline if ird]

    if benchmarks and ir_commits:
        lines.append("## Instruction Counts (Ir)")
        lines.append("")

        header = ["Benchmark"] + [commit_header(s, subj)
                                   for s, subj, _ in ir_commits]
        sep = ["---"] * len(header)
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")

        for display, func, ir_id, _ in benchmarks:
            row = [f"`{display}`"]
            for _, _, ird in ir_commits:
                val = ird.get((func, ir_id))
                row.append(fmt_ir(val))
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    # --- Notable changes ---
    lines.append("## Notable Changes")
    lines.append("")

    notable = []
    for i in range(1, len(timeline)):
        sha, _, subject, ir_data, time_data, cmp_data = timeline[i]
        _, _, _, prev_ir, prev_time, _ = timeline[i - 1]

        for display, func, ir_id, time_key in benchmarks:
            # Check Ir change
            cur_ir = ir_data.get((func, ir_id))
            old_ir = prev_ir.get((func, ir_id))
            ir_pct = pct_change(old_ir, cur_ir)
            if ir_pct is not None and abs(ir_pct) > ir_threshold:
                direction = "regression" if ir_pct > 0 else "improvement"
                notable.append((
                    sha, subject, display, "Ir",
                    f"{ir_pct:+.2f}%", direction,
                ))

            # Check wall-clock change — prefer A/B comparison data
            if time_key is not None:
                cmp_bench = cmp_data.get(time_key)
                if cmp_bench:
                    old_t = cmp_bench["base"].get("median")
                    cur_t = cmp_bench["head"].get("median")
                else:
                    cur_t = time_data.get(time_key)
                    old_t = prev_time.get(time_key)
                t_pct = pct_change(old_t, cur_t)
                if t_pct is not None and abs(t_pct) > time_threshold:
                    direction = "regression" if t_pct > 0 else "improvement"
                    source = "wall-clock (A/B)" if cmp_bench else "wall-clock"
                    notable.append((
                        sha, subject, display, source,
                        f"{t_pct:+.1f}%", direction,
                    ))

    if notable:
        lines.append("| Commit | Benchmark | Metric | Change | Type |")
        lines.append("|--------|-----------|--------|--------|------|")
        for sha, subject, bench, metric, change, direction in notable:
            lines.append(
                f"| `{short_sha(sha)}` {short_desc(subject)} "
                f"| `{bench}` | {metric} | {change} | {direction} |"
            )
    else:
        lines.append("No notable changes detected.")

    lines.append("")

    # --- Regression report ---
    lines.append("## Regression Report (Latest vs Earliest)")
    lines.append("")

    if len(timeline) < 2:
        lines.append("Need at least 2 data points for regression analysis.")
        lines.append("")
        return "\n".join(lines)

    first_sha, _, first_subj, first_ir, first_time, _ = timeline[0]
    last_sha, _, last_subj, last_ir, last_time, _ = timeline[-1]
    lines.append(
        f"Comparing `{short_sha(first_sha)}` ({short_desc(first_subj)}) "
        f"to `{short_sha(last_sha)}` ({short_desc(last_subj)})"
    )
    lines.append("")

    regressions = []
    for display, func, ir_id, time_key in benchmarks:
        # Ir regression
        first_val = first_ir.get((func, ir_id))
        last_val = last_ir.get((func, ir_id))
        ir_pct = pct_change(first_val, last_val)
        if ir_pct is not None and ir_pct > 0:
            regressions.append((display, "Ir", fmt_ir(first_val),
                                fmt_ir(last_val), f"{ir_pct:+.2f}%"))

        # Wall-clock regression
        if time_key is not None:
            first_t = first_time.get(time_key)
            last_t = last_time.get(time_key)
            t_pct = pct_change(first_t, last_t)
            if t_pct is not None and t_pct > 0:
                regressions.append((display, "wall-clock", fmt_time(first_t),
                                    fmt_time(last_t), f"{t_pct:+.1f}%"))

    if regressions:
        lines.append("| Benchmark | Metric | Earliest | Latest | Change |")
        lines.append("|-----------|--------|----------|--------|--------|")
        for bench, metric, earliest, latest, change in regressions:
            lines.append(
                f"| `{bench}` | {metric} | {earliest} | {latest} | {change} |"
            )
    else:
        lines.append("No regressions detected — all tracked benchmarks are "
                      "stable or improved.")

    lines.append("")

    # --- A/B Comparisons ---
    ab_entries = [(sha, subj, cmp)
                  for sha, _, subj, _, _, cmp in timeline if cmp]
    if ab_entries:
        lines.append("## A/B Comparisons")
        lines.append("")
        lines.append("Interleaved hyperfine runs comparing consecutive commits.")
        lines.append("")

        for sha, subject, cmp_data in ab_entries:
            lines.append(f"### `{short_sha(sha)}` {short_desc(subject)}")
            lines.append("")
            lines.append("| Benchmark | Base (median) | Head (median) | Change |")
            lines.append("|-----------|---------------|---------------|--------|")
            for bench_name in sorted(cmp_data.keys()):
                entry = cmp_data[bench_name]
                base_med = entry["base"].get("median")
                head_med = entry["head"].get("median")
                change = pct_change(base_med, head_med)
                change_str = f"{change:+.1f}%" if change is not None else "-"
                lines.append(
                    f"| `{bench_name}` | {fmt_time(base_med)} "
                    f"| {fmt_time(head_med)} | {change_str} |"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def csv_report(timeline, benchmarks):
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header
    header = ["sha", "date", "subject"]
    for display, _, _, _ in benchmarks:
        header.append(f"{display} (Ir)")
    for display, _, _, time_key in benchmarks:
        if time_key is not None:
            header.append(f"{display} (median s)")
    writer.writerow(header)

    for sha, date, subject, ir_data, time_data, _cmp in timeline:
        row = [sha, date, subject]
        for _, func, ir_id, _ in benchmarks:
            val = ir_data.get((func, ir_id))
            row.append(val if val is not None else "")
        for _, func, ir_id, time_key in benchmarks:
            if time_key is not None:
                val = time_data.get(time_key)
                row.append(val if val is not None else "")
        writer.writerow(row)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Read stored benchmark data from the benchmark-data "
                    "branch and produce a chronological summary report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                       # full markdown report\n"
            "  %(prog)s --last=5              # last 5 commits only\n"
            "  %(prog)s --benchmark=sin       # filter to sin benchmarks\n"
            "  %(prog)s --format=csv          # CSV output\n"
            "  %(prog)s --flag-threshold=3    # flag Ir changes >3%%\n"
        ),
    )
    parser.add_argument(
        "--format", choices=["markdown", "csv"], default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--last", type=int, metavar="N",
        help="Show only the last N commits",
    )
    parser.add_argument(
        "--benchmark", metavar="NAME",
        help="Filter to benchmarks matching NAME (substring, case-insensitive)",
    )
    parser.add_argument(
        "--flag-threshold", type=float, default=5.0, metavar="N",
        help="Ir change threshold for flagging notable changes (default: 5%%)",
    )
    parser.add_argument(
        "--time-threshold", type=float, default=15.0, metavar="N",
        help="Wall-clock change threshold for flagging (default: 15%%)",
    )
    args = parser.parse_args()

    ir_shas, time_shas, comparison_pairs = discover_files()
    commit_map = load_commit_map()
    timeline = build_timeline(commit_map, ir_shas, time_shas,
                              comparison_pairs=comparison_pairs,
                              last_n=args.last)

    benchmarks = filter_benchmarks(KEY_BENCHMARKS, args.benchmark)
    if not benchmarks:
        print(f"No benchmarks match filter '{args.benchmark}'", file=sys.stderr)
        sys.exit(1)

    if args.format == "csv":
        print(csv_report(timeline, benchmarks), end="")
    else:
        print(markdown_report(timeline, benchmarks,
                              args.flag_threshold, args.time_threshold))


if __name__ == "__main__":
    main()
