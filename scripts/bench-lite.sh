#!/usr/bin/env bash
# Lite benchmark mode: ~3 min hyperfine run with tiered warmup/runs.
#
# Usage:
#   scripts/bench-lite.sh                     # single run
#   scripts/bench-lite.sh /path/to/main-bins  # A/B comparison
set -euo pipefail

BASELINE_DIR="${1:-}"
RESULTS_DIR="results-lite"

SWEEP="integer_roots inv sin pi asymmetric_convergence"
NO_SWEEP="summation complex"
BITS=(1 64 256)

# Tiered hyperfine parameters based on expected runtime.
# Benchmarks >1s per run get fewer runs (low CV, still >90% power for 20% shifts).
warmup_runs() {
    local bench="$1"
    local bits="${2:-0}"
    # Known slow cases: integer_roots at bits>=64, sin at bits=256
    if [[ "$bench" == "integer_roots" && "$bits" -ge 64 ]]; then
        echo "1 3"
    elif [[ "$bench" == "sin" && "$bits" -ge 256 ]]; then
        echo "1 3"
    else
        echo "2 5"
    fi
}

# Build time-bench binaries
echo "Building time-bench binaries..."
cargo bench --features time-bench --no-run 2>&1

# Copy binaries
BIN_DIR="$(mktemp -d)"
trap 'rm -rf "$BIN_DIR"' EXIT
BENCHES="$SWEEP $NO_SWEEP"
for bench in $BENCHES; do
    bin=$(find target/release/deps -name "${bench}-*" -perm +111 ! -name "*.d" | head -1)
    if [ -z "$bin" ]; then
        echo "Warning: binary not found for $bench, skipping" >&2
        continue
    fi
    cp "$bin" "$BIN_DIR/${bench}"
done

mkdir -p "$RESULTS_DIR"

HAS_BASELINE=false
if [ -n "$BASELINE_DIR" ] && [ -d "$BASELINE_DIR" ] && [ -n "$(ls -A "$BASELINE_DIR" 2>/dev/null)" ]; then
    HAS_BASELINE=true
    echo "A/B comparison against: $BASELINE_DIR"
fi

echo "Running lite benchmarks (bits: ${BITS[*]})..."

for bench in $SWEEP; do
    [ -f "$BIN_DIR/$bench" ] || continue
    for bits in "${BITS[@]}"; do
        read -r warmup runs <<< "$(warmup_runs "$bench" "$bits")"
        echo "  $bench bits=$bits (warmup=$warmup, runs=$runs)"
        if [ "$HAS_BASELINE" = "true" ] && [ -f "$BASELINE_DIR/$bench" ]; then
            hyperfine --export-json "$RESULTS_DIR/${bench}_${bits}.json" \
                --warmup "$warmup" --runs "$runs" \
                "$BASELINE_DIR/${bench} ${bits}" "$BIN_DIR/${bench} ${bits}"
        else
            hyperfine --export-json "$RESULTS_DIR/${bench}_${bits}.json" \
                --warmup "$warmup" --runs "$runs" \
                "$BIN_DIR/${bench} ${bits}"
        fi
    done
done

for bench in $NO_SWEEP; do
    [ -f "$BIN_DIR/$bench" ] || continue
    read -r warmup runs <<< "$(warmup_runs "$bench")"
    echo "  $bench (warmup=$warmup, runs=$runs)"
    if [ "$HAS_BASELINE" = "true" ] && [ -f "$BASELINE_DIR/$bench" ]; then
        hyperfine --export-json "$RESULTS_DIR/${bench}.json" \
            --warmup "$warmup" --runs "$runs" \
            "$BASELINE_DIR/${bench}" "$BIN_DIR/${bench}"
    else
        hyperfine --export-json "$RESULTS_DIR/${bench}.json" \
            --warmup "$warmup" --runs "$runs" \
            "$BIN_DIR/${bench}"
    fi
done

echo "Results in $RESULTS_DIR/"
echo "Done."
