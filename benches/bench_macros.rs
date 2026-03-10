#[cfg(all(feature = "time-bench", feature = "criterion-bench"))]
compile_error!("features `time-bench` and `criterion-bench` cannot be enabled simultaneously");

use computable::XU;

/// Standard precision sweep: epsilon = 2^(-bits) for each value.
#[cfg(feature = "criterion-bench")]
const STANDARD_BITS: &[usize] = &[1, 4, 16, 64, 256];

/// Extended sweep, enabled by `BENCH_HIGH_PRECISION=1`.
#[cfg(feature = "criterion-bench")]
const EXTENDED_BITS: &[usize] = &[1, 4, 16, 64, 256, 1024, 2048, 4096, 8192];

/// Returns the precision sweep to use. Set `BENCH_HIGH_PRECISION=1` to include
/// higher precisions (1024+).
#[cfg(feature = "criterion-bench")]
pub fn precision_bits() -> &'static [usize] {
    if high_precision() {
        EXTENDED_BITS
    } else {
        STANDARD_BITS
    }
}

/// Whether high-precision benchmarks are enabled (via `BENCH_HIGH_PRECISION=1`).
#[cfg(feature = "criterion-bench")]
pub fn high_precision() -> bool {
    std::env::var("BENCH_HIGH_PRECISION").is_ok()
}

/// Create a tolerance exponent for 2^(-bits) precision.
#[allow(clippy::as_conversions)] // bench infrastructure: values are small constants
pub fn epsilon(bits: usize) -> XU {
    XU::Finite(bits as u32)
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
            #[benches::precision(args = [1_usize, 4, 16, 64, 256])]
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
            $(
                for &$bits in bench_macros::precision_bits() {
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
