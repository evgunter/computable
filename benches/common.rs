#![allow(dead_code)]

use computable::{Binary, Computable, XUsize};

/// Standard precision sweep: epsilon = 2^(-bits) for each value.
const STANDARD_BITS: &[usize] = &[1, 4, 16, 64, 256];

/// Extended sweep, enabled by `BENCH_HIGH_PRECISION=1`.
const EXTENDED_BITS: &[usize] = &[1, 4, 16, 64, 256, 1024, 2048, 4096, 8192];

/// Returns the precision sweep to use. Set `BENCH_HIGH_PRECISION=1` to include
/// higher precisions (1024+).
pub fn precision_bits() -> &'static [usize] {
    if high_precision() {
        EXTENDED_BITS
    } else {
        STANDARD_BITS
    }
}

/// Whether high-precision benchmarks are enabled (via `BENCH_HIGH_PRECISION=1`).
pub fn high_precision() -> bool {
    std::env::var("BENCH_HIGH_PRECISION").is_ok()
}

/// Create a tolerance exponent for 2^(-bits) precision.
pub fn epsilon(bits: usize) -> XUsize {
    XUsize::Finite(bits)
}

/// Whether to print diagnostic info (enabled by `BENCH_VERBOSE=1`).
pub fn verbose() -> bool {
    std::env::var("BENCH_VERBOSE").is_ok()
}

/// Sums terms using a balanced reduction instead of left-associative chaining.
///
/// This keeps the computation graph shallow (O(log n) depth), avoiding deep nesting
/// that can overflow the stack or distort timing by spending most of the runtime
/// walking long expression chains.
pub fn balanced_sum(mut values: Vec<Computable>) -> Computable {
    if values.is_empty() {
        return Computable::constant(Binary::zero());
    }

    while values.len() > 1 {
        let mut next = Vec::with_capacity(values.len().div_ceil(2));
        let mut iter = values.into_iter();
        while let Some(left) = iter.next() {
            if let Some(right) = iter.next() {
                next.push(left + right);
            } else {
                next.push(left);
            }
        }
        values = next;
    }

    values
        .pop()
        .expect("values should contain at least one element")
}

/// Declares a benchmark group that works with both gungraun (valgrind) and criterion.
///
/// Two forms:
/// - **Precision sweep**: each function receives `bits: usize` and is benchmarked
///   across `[1, 4, 16, 64, 256]` bits of precision.
/// - **No sweep**: parameterless functions run once (for benchmarks where the
///   computation is exact and precision doesn't affect workload).
macro_rules! bench_group {
    // No-sweep variant: parameterless functions, run once.
    (
        name: $group:ident,
        $(
            fn $fn_name:ident() $(-> $ret:ty)? $body:block
        )+
    ) => {
        $(
            #[cfg(not(feature = "criterion-bench"))]
            #[gungraun::library_benchmark]
            fn $fn_name() $(-> $ret)? $body
        )+

        #[cfg(not(feature = "criterion-bench"))]
        library_benchmark_group!(
            name = $group,
            benchmarks = [$($fn_name),+]
        );

        #[cfg(feature = "criterion-bench")]
        fn $group(c: &mut ::criterion::Criterion) {
            let mut group = c.benchmark_group(stringify!($group));
            $(
                group.bench_function(
                    stringify!($fn_name),
                    |b| { b.iter(|| $body); },
                );
            )+
            group.finish();
        }
    };

    // Precision-sweep variant: functions take `bits` and run across the sweep.
    (
        name: $group:ident,
        $(
            fn $fn_name:ident($bits:ident) $(-> $ret:ty)? $body:block
        )+
    ) => {
        $(
            #[cfg(not(feature = "criterion-bench"))]
            #[gungraun::library_benchmark]
            #[benches::precision(args = [1_usize, 4, 16, 64, 256])]
            fn $fn_name($bits: usize) $(-> $ret)? $body
        )+

        #[cfg(not(feature = "criterion-bench"))]
        library_benchmark_group!(
            name = $group,
            benchmarks = [$($fn_name),+]
        );

        #[cfg(feature = "criterion-bench")]
        fn $group(c: &mut ::criterion::Criterion) {
            let mut group = c.benchmark_group(stringify!($group));
            $(
                for &$bits in common::precision_bits() {
                    group.bench_function(
                        &format!("{}/bits={}", stringify!($fn_name), $bits),
                        |b| { b.iter(|| $body); },
                    );
                }
            )+
            group.finish();
        }
    };
}
pub(crate) use bench_group;

/// Declares the benchmark `main` for both gungraun and criterion.
macro_rules! bench_main {
    // With valgrind_args (ignored under criterion).
    ($($group:ident),+ ; valgrind_args: [$($arg:literal),* $(,)?]) => {
        #[cfg(not(feature = "criterion-bench"))]
        main!(
            config = LibraryBenchmarkConfig::default()
                .valgrind_args([$($arg),*]);
            library_benchmark_groups = $($group),+
        );

        #[cfg(feature = "criterion-bench")]
        ::criterion::criterion_group!(benches, $($group),+);
        #[cfg(feature = "criterion-bench")]
        ::criterion::criterion_main!(benches);
    };

    // Without valgrind_args.
    ($($group:ident),+ $(,)?) => {
        #[cfg(not(feature = "criterion-bench"))]
        main!(library_benchmark_groups = $($group),+);

        #[cfg(feature = "criterion-bench")]
        ::criterion::criterion_group!(benches, $($group),+);
        #[cfg(feature = "criterion-bench")]
        ::criterion::criterion_main!(benches);
    };
}
pub(crate) use bench_main;
