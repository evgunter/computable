#![allow(dead_code)]

#[cfg(all(feature = "time-bench", feature = "criterion-bench"))]
compile_error!("features `time-bench` and `criterion-bench` cannot be enabled simultaneously");

use computable::{Binary, Computable, XUsize};

/// Standard precision sweep: epsilon = 2^(-bits) for each value.
const STANDARD_BITS: &[usize] = &[1, 4, 16, 64, 256];

/// Extended sweep, enabled by `BENCH_HIGH_PRECISION=1`.
const EXTENDED_BITS: &[usize] = &[1, 4, 16, 64, 256, 1024, 2048, 4096, 8192];

/// Lite sweep: skip intermediate precisions for fast local iteration.
const LITE_BITS: &[usize] = &[1, 64, 256];

/// Returns the precision sweep to use. `bench-lite` feature uses a reduced set;
/// `BENCH_HIGH_PRECISION=1` includes higher precisions (1024+).
pub fn precision_bits() -> &'static [usize] {
    if cfg!(feature = "bench-lite") {
        LITE_BITS
    } else if high_precision() {
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

/// Declares a benchmark group that works with gungraun (valgrind), criterion,
/// and time-bench (hyperfine wall-clock).
///
/// Three forms per variant:
/// - **gungraun** (default, no features): valgrind-based instruction counting
/// - **criterion** (`criterion-bench` feature): statistical wall-clock via criterion
/// - **time-bench** (`time-bench` feature): single-shot execution for hyperfine
macro_rules! bench_group {
    // No-sweep variant: parameterless functions, run once.
    (
        name: $group:ident,
        $(
            fn $fn_name:ident() $(-> $ret:ty)? $body:block
        )+
    ) => {
        $(
            #[cfg(not(any(feature = "criterion-bench", feature = "time-bench")))]
            #[gungraun::library_benchmark]
            fn $fn_name() $(-> $ret)? $body
        )+

        #[cfg(not(any(feature = "criterion-bench", feature = "time-bench")))]
        library_benchmark_group!(
            name = $group,
            benchmarks = [$($fn_name),+]
        );

        #[cfg(feature = "criterion-bench")]
        fn $group(c: &mut ::criterion::Criterion) {
            let mut group = c.benchmark_group(stringify!($group));
            #[cfg(feature = "bench-lite")]
            {
                group.sample_size(10);
                group.warm_up_time(::std::time::Duration::from_secs(1));
                group.measurement_time(::std::time::Duration::from_secs(1));
            }
            $(
                group.bench_function(
                    stringify!($fn_name),
                    |b| { b.iter(|| $body); },
                );
            )+
            group.finish();
        }

        #[cfg(feature = "time-bench")]
        fn $group(_bits: Option<usize>) {
            $(
                ::std::hint::black_box((|| $body)());
            )+
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
            #[cfg(not(any(feature = "criterion-bench", feature = "time-bench")))]
            #[gungraun::library_benchmark]
            #[cfg_attr(feature = "bench-lite",
                benches::precision(args = [1_usize, 64, 256]))]
            #[cfg_attr(not(feature = "bench-lite"),
                benches::precision(args = [1_usize, 4, 16, 64, 256]))]
            fn $fn_name($bits: usize) $(-> $ret)? $body
        )+

        #[cfg(not(any(feature = "criterion-bench", feature = "time-bench")))]
        library_benchmark_group!(
            name = $group,
            benchmarks = [$($fn_name),+]
        );

        #[cfg(feature = "criterion-bench")]
        fn $group(c: &mut ::criterion::Criterion) {
            let mut group = c.benchmark_group(stringify!($group));
            #[cfg(feature = "bench-lite")]
            {
                group.sample_size(10);
                group.warm_up_time(::std::time::Duration::from_secs(1));
                group.measurement_time(::std::time::Duration::from_secs(1));
            }
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

        #[cfg(feature = "time-bench")]
        fn $group(bits: Option<usize>) {
            let bits_val = bits.expect("precision sweep benchmarks require a bits argument");
            $(
                let $bits = bits_val;
                ::std::hint::black_box((|| $body)());
            )+
        }
    };
}
pub(crate) use bench_group;

/// Declares the benchmark `main` for gungraun, criterion, and time-bench.
macro_rules! bench_main {
    // With valgrind_args (ignored under criterion and time-bench).
    ($($group:ident),+ ; valgrind_args: [$($arg:literal),* $(,)?]) => {
        #[cfg(not(any(feature = "criterion-bench", feature = "time-bench")))]
        main!(
            config = LibraryBenchmarkConfig::default()
                .valgrind_args([$($arg),*]);
            library_benchmark_groups = $($group),+
        );

        #[cfg(feature = "criterion-bench")]
        ::criterion::criterion_group!(benches, $($group),+);
        #[cfg(feature = "criterion-bench")]
        ::criterion::criterion_main!(benches);

        #[cfg(feature = "time-bench")]
        fn main() {
            let bits = ::std::env::args().nth(1).map(|s| s.parse::<usize>().expect("bits argument must be a valid usize"));
            $( $group(bits); )+
        }
    };

    // Without valgrind_args.
    ($($group:ident),+ $(,)?) => {
        #[cfg(not(any(feature = "criterion-bench", feature = "time-bench")))]
        main!(
            config = LibraryBenchmarkConfig::default()
                .valgrind_args(["--fair-sched=yes"]);
            library_benchmark_groups = $($group),+
        );

        #[cfg(feature = "criterion-bench")]
        ::criterion::criterion_group!(benches, $($group),+);
        #[cfg(feature = "criterion-bench")]
        ::criterion::criterion_main!(benches);

        #[cfg(feature = "time-bench")]
        fn main() {
            let bits = ::std::env::args().nth(1).map(|s| s.parse::<usize>().expect("bits argument must be a valid usize"));
            $( $group(bits); )+
        }
    };
}
pub(crate) use bench_main;
