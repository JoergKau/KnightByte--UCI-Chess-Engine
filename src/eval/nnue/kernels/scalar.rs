//! The exact scalar reference: every SIMD backend must produce these values
//! bit-for-bit, and the unit tests compare the dispatched backend against it.
//!
use super::screlu_value;
use crate::types::color::Color;

/// `dst = bias + sum(rows)` with wrapping `i16` arithmetic.
pub(crate) fn refresh_rows(dst: &mut [i16], bias: &[i16], rows: &[&[i16]]) {
    for index in 0..dst.len() {
        let mut value = bias[index];
        for row in rows {
            value = value.wrapping_add(row[index]);
        }
        dst[index] = value;
    }
}
/// `dst = src - sum(subs) + sum(adds)` with wrapping `i16` arithmetic.
pub(crate) fn push_rows(dst: &mut [i16], src: &[i16], adds: &[&[i16]], subs: &[&[i16]]) {
    for index in 0..dst.len() {
        let mut value = src[index];
        for row in subs {
            value = value.wrapping_sub(row[index]);
        }
        for row in adds {
            value = value.wrapping_add(row[index]);
        }
        dst[index] = value;
    }
}
/// The SCReLU output head: `sum` accumulates the exact `i64` products.
///
/// With `exact` the clamped value is squared without the per-element `>> 8`;
/// the caller adds the bias in the same (256x) scale and applies the larger
/// final shift. The total is computed in `i64`, so both modes are exact.
pub(crate) fn evaluate_head(
    stm: Color,
    pair: &[Vec<i16>; 2],
    output_weights: &[i16],
    exact: bool,
) -> i64 {
    let l1 = pair[0].len();
    let stm_acc = &pair[stm as usize];
    let ntm_acc = &pair[(stm ^ 1) as usize];
    let mut sum = 0_i64;

    for index in 0..l1 {
        let a = stm_acc[index];
        if a > 0 {
            sum += i64::from(screlu_value(a, exact)) * i64::from(output_weights[index]);
        }
        let b = ntm_acc[index];
        if b > 0 {
            sum += i64::from(screlu_value(b, exact)) * i64::from(output_weights[l1 + index]);
        }
    }
    sum
}
