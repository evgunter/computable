//! Display formatting utilities for binary numbers.
//!
//! Provides helper functions for displaying binary numbers in normalized form
//! with the mantissa shown in binary as X.XXXXXX * 2^N.

use std::fmt;

use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, Zero};

/// Formats a signed binary number with a normalized mantissa display.
/// Displays the mantissa in binary with a binary point after the first bit,
/// adjusting the exponent accordingly (e.g., "1.1101 * 2^5").
pub(super) fn format_binary_display(
    f: &mut fmt::Formatter<'_>,
    mantissa: &BigInt,
    exponent: i64,
) -> fmt::Result {
    if mantissa.is_zero() {
        return write!(f, "0.0");
    }

    let is_negative = mantissa.is_negative();
    let abs_mantissa = mantissa.magnitude();

    let (formatted_mantissa, adjusted_exponent) =
        format_mantissa_with_point(abs_mantissa, exponent);

    if is_negative {
        write!(f, "-{} * 2^{}", formatted_mantissa, adjusted_exponent)
    } else {
        write!(f, "{} * 2^{}", formatted_mantissa, adjusted_exponent)
    }
}

/// Formats an unsigned binary number with a normalized mantissa display.
/// Displays the mantissa in binary with a binary point after the first bit,
/// adjusting the exponent accordingly (e.g., "1.1101 * 2^5").
pub(super) fn format_ubinary_display(
    f: &mut fmt::Formatter<'_>,
    mantissa: &BigUint,
    exponent: i64,
) -> fmt::Result {
    if mantissa.is_zero() {
        return write!(f, "0.0");
    }

    let (formatted_mantissa, adjusted_exponent) = format_mantissa_with_point(mantissa, exponent);

    write!(f, "{} * 2^{}", formatted_mantissa, adjusted_exponent)
}

/// Helper function to format a mantissa with a binary point.
/// Returns the formatted mantissa string and the adjusted exponent.
fn format_mantissa_with_point(mantissa: &BigUint, exponent: i64) -> (String, i64) {
    // Get binary representation of the mantissa
    let binary_str = format!("{:b}", mantissa);
    let num_bits = binary_str.len();

    // Format as X.XXXXXX with binary point after first bit
    let formatted_mantissa = if num_bits == 1 {
        format!("{}.0", binary_str)
    } else {
        format!("{}.{}", &binary_str[0..1], &binary_str[1..])
    };

    // Adjust exponent: new_exp = old_exp + (num_bits - 1)
    let bits_minus_one = i64::try_from(num_bits)
        .unwrap_or_else(|_| {
            crate::detected_computable_would_exhaust_memory!(
                "mantissa too large in format_mantissa_with_point"
            )
        })
        .checked_sub(1_i64)
        .unwrap_or_else(|| {
            crate::detected_computable_would_exhaust_memory!(
                "exponent overflow in format_mantissa_with_point"
            )
        });
    let adjusted_exponent = exponent.checked_add(bits_minus_one).unwrap_or_else(|| {
        crate::detected_computable_would_exhaust_memory!(
            "exponent overflow in format_mantissa_with_point"
        )
    });

    (formatted_mantissa, adjusted_exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_single_bit_mantissa() {
        let mantissa = BigUint::from(1u32);
        let exponent = 0_i64;
        let (formatted, adjusted_exp) = format_mantissa_with_point(&mantissa, exponent);
        assert_eq!(formatted, "1.0");
        assert_eq!(adjusted_exp, 0_i64);
    }

    #[test]
    fn format_multi_bit_mantissa() {
        // 123 decimal = 1111011 binary (7 bits)
        let mantissa = BigUint::from(123u32);
        let exponent = 5_i64;
        let (formatted, adjusted_exp) = format_mantissa_with_point(&mantissa, exponent);
        assert_eq!(formatted, "1.111011");
        assert_eq!(adjusted_exp, 11_i64); // 5 + 6
    }

    #[test]
    fn format_binary_display_zero() {
        let mantissa = BigInt::from(0_i32);
        let exponent = 42_i64;
        let result = format!(
            "{}",
            FormatWrapper(&mantissa, exponent, format_binary_display)
        );
        assert_eq!(result, "0.0");
    }

    #[test]
    fn format_binary_display_negative() {
        let mantissa = BigInt::from(-15_i32); // -1111 in binary
        let exponent = 3_i64;
        let result = format!(
            "{}",
            FormatWrapper(&mantissa, exponent, format_binary_display)
        );
        assert_eq!(result, "-1.111 * 2^6"); // 3 + 3
    }

    #[test]
    fn demonstrate_new_format() {
        // Example: 123 (decimal) = 1111011 (binary, 7 bits)
        // Old format would be: 123 * 2^5
        // New format is: 1.111011 * 2^11 (exponent adjusted by 6)
        let mantissa = BigUint::from(123u32);
        let exponent = 5_i64;
        let (formatted, adjusted_exp) = format_mantissa_with_point(&mantissa, exponent);

        assert_eq!(formatted, "1.111011");
        assert_eq!(adjusted_exp, 11_i64);
    }

    // Helper struct for testing Display implementations
    struct FormatWrapper<'a, T>(
        &'a T,
        i64,
        fn(&mut fmt::Formatter<'_>, &T, i64) -> fmt::Result,
    );

    impl<'a, T> fmt::Display for FormatWrapper<'a, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            (self.2)(f, self.0, self.1)
        }
    }
}
