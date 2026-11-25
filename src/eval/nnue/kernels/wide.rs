//! The 128-bit, 256-bit and 512-bit x86 kernels.
//!
//! The head computes per element `v = (clamp(a)²) >> 8` exactly in 16-bit
//! lanes (the clamped square of a `QA = 255` value fits an unsigned 16-bit
//! lane), multiplies by the 16-bit output weight with `madd` and
//! accumulates into `i32` lanes that fold into an `i64` total on a fixed
//! schedule. The fold bounds keep every `i32` partial sum far below
//! overflow, so the result is bit-identical to the scalar `i64`
//! accumulation.

use std::arch::x86_64::*;

use super::screlu_value;
use crate::types::color::Color;

/// Generates one output-head kernel for a vector width.
///
/// The head needs `v = clamp(a, 0, QA)^2` per element, multiplied by the
/// output weight and summed. With `QA = 255` the clamped square is at most
/// 65,025 and fits an unsigned 16-bit lane, so `mullo_epi16` gives it
/// exactly.
///
/// The historical chain (`EXACT = false`) shifts the square right by eight
/// bits first, leaving `v <= 254` in a signed 16-bit lane, and `madd_epi16`
/// multiplies it by the weight. The exact chain (`EXACT = true`) splits the
/// square into `256 * v + r` and accumulates both dot products; the final
/// total is `256 * sum(v*w) + sum(r*w)`, which is the exact `sum(a^2*w)`
/// without a per-element truncation. Both partial sums use the same lane
/// bounds as the historical chain: a lane gains at most
/// `2 x 255 x 32768` per iteration, so folding into the `i64` total every
/// 32 iterations keeps every partial sum below 1.1 G, far under `i32`
/// overflow. A scalar loop finishes any tail narrower than one vector.
macro_rules! head_kernel {
    ($name:ident, $feature:literal, $lanes:expr, $vector:ty, $load:ident,
     $setzero:ident, $set1:ident, $max:ident, $min:ident, $mullo:ident,
     $srli:ident, $and:ident, $madd:ident, $add32:ident, $fold:ident) => {
        /// # Safety
        ///
        /// The host CPU must support the instruction set named in the
        /// attribute, and `output_weights` must hold at least twice the
        /// accumulator width.
        #[target_feature(enable = $feature)]
        pub(super) unsafe fn $name<const EXACT: bool>(
            stm: Color,
            pair: &[Vec<i16>; 2],
            output_weights: &[i16],
        ) -> i64 {
            let stm_acc = &pair[stm as usize];
            let ntm_acc = &pair[(stm ^ 1) as usize];
            let l1 = stm_acc.len();
            let zero = $setzero();
            let qa = $set1(crate::eval::nnue::net::QA as i16);
            let low_byte = $set1(255);
            let mut total = 0_i64;
            let mut correction_total = 0_i64;
            let mut lanes = zero;
            let mut correction = zero;
            let mut pending = 0_u32;
            let mut offset = 0;
            while offset + $lanes <= l1 {
                // The stm rows and the ntm rows live in the two halves of
                // the output weights; each chunk pairs its lanes with its
                // own half.
                for (accumulator, weight_base) in [(stm_acc, offset), (ntm_acc, l1 + offset)] {
                    unsafe {
                        let a = $load(accumulator.as_ptr().add(offset) as *const $vector);
                        let clamped = $min($max(a, zero), qa);
                        let square = $mullo(clamped, clamped);
                        let v = $srli::<8>(square);
                        let w = $load(output_weights.as_ptr().add(weight_base) as *const $vector);
                        lanes = $add32(lanes, $madd(v, w));
                        if EXACT {
                            let r = $and(square, low_byte);
                            correction = $add32(correction, $madd(r, w));
                        }
                    }
                }
                offset += $lanes;
                pending += 1;
                if pending == 32 {
                    total += $fold(lanes);
                    if EXACT {
                        correction_total += $fold(correction);
                    }
                    lanes = zero;
                    correction = zero;
                    pending = 0;
                }
            }
            total += $fold(lanes);
            if EXACT {
                // The correction term was accumulated separately so the
                // lane bound stays identical to the historical chain; fold
                // it into the main total on their common 256x scale.
                correction_total += $fold(correction);
                total = (total << 8) + correction_total;
            }
            for index in offset..l1 {
                let a = stm_acc[index];
                if a > 0 {
                    total += i64::from(screlu_value(a, EXACT)) * i64::from(output_weights[index]);
                }
                let b = ntm_acc[index];
                if b > 0 {
                    total +=
                        i64::from(screlu_value(b, EXACT)) * i64::from(output_weights[l1 + index]);
                }
            }
            total
        }
    };
}

head_kernel!(
    evaluate_head_sse41,
    "sse4.1",
    8,
    __m128i,
    _mm_loadu_si128,
    _mm_setzero_si128,
    _mm_set1_epi16,
    _mm_max_epi16,
    _mm_min_epi16,
    _mm_mullo_epi16,
    _mm_srli_epi16,
    _mm_and_si128,
    _mm_madd_epi16,
    _mm_add_epi32,
    fold_lanes128
);
head_kernel!(
    evaluate_head_avx2,
    "avx2",
    16,
    __m256i,
    _mm256_loadu_si256,
    _mm256_setzero_si256,
    _mm256_set1_epi16,
    _mm256_max_epi16,
    _mm256_min_epi16,
    _mm256_mullo_epi16,
    _mm256_srli_epi16,
    _mm256_and_si256,
    _mm256_madd_epi16,
    _mm256_add_epi32,
    fold_lanes
);
head_kernel!(
    evaluate_head_avx512,
    "avx512f,avx512bw,avx512vl",
    32,
    __m512i,
    _mm512_loadu_si512,
    _mm512_setzero_si512,
    _mm512_set1_epi16,
    _mm512_max_epi16,
    _mm512_min_epi16,
    _mm512_mullo_epi16,
    _mm512_srli_epi16,
    _mm512_and_si512,
    _mm512_madd_epi16,
    _mm512_add_epi32,
    fold_lanes512
);

/// Generates one fused row-update kernel: `dst = src - sum(subs) + sum(adds)`.
///
/// The shape (`A` rows added, `S` rows subtracted) is a const generic so
/// the row loops unroll. Each vector chunk is loaded once, updated in a
/// register and stored once; a scalar loop finishes any tail narrower than
/// one vector.
macro_rules! push_rows_kernel {
    ($name:ident, $feature:literal, $lanes:expr, $load:ident, $store:ident,
     $add:ident, $sub:ident, $vector:ty) => {
        /// # Safety
        ///
        /// The host CPU must support the instruction set named in the
        /// attribute, and `src` and every row must hold at least
        /// `dst.len()` elements.
        #[target_feature(enable = $feature)]
        pub(super) unsafe fn $name<const A: usize, const S: usize>(
            dst: &mut [i16],
            src: &[i16],
            adds: &[&[i16]; A],
            subs: &[&[i16]; S],
        ) {
            let n = dst.len();
            let mut offset = 0;
            while offset + $lanes <= n {
                unsafe {
                    let mut value = $load(src.as_ptr().add(offset) as *const $vector);
                    for row in subs {
                        value = $sub(value, $load(row.as_ptr().add(offset) as *const $vector));
                    }
                    for row in adds {
                        value = $add(value, $load(row.as_ptr().add(offset) as *const $vector));
                    }
                    $store(dst.as_mut_ptr().add(offset) as *mut $vector, value);
                }
                offset += $lanes;
            }
            for index in offset..n {
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
    };
}

/// Generates one full-refresh kernel: `dst = bias + sum(rows)`.
///
/// The chunk loop is outermost, so each vector chunk stays in a register
/// while every row contributes to it; a scalar loop finishes any tail
/// narrower than one vector.
macro_rules! refresh_rows_kernel {
    ($name:ident, $feature:literal, $lanes:expr, $load:ident, $store:ident,
     $add:ident, $vector:ty) => {
        /// # Safety
        ///
        /// The host CPU must support the instruction set named in the
        /// attribute, and `bias` and every row must hold at least
        /// `dst.len()` elements.
        #[target_feature(enable = $feature)]
        pub(super) unsafe fn $name(dst: &mut [i16], bias: &[i16], rows: &[&[i16]]) {
            let n = dst.len();
            let mut offset = 0;
            while offset + $lanes <= n {
                unsafe {
                    let mut value = $load(bias.as_ptr().add(offset) as *const $vector);
                    for row in rows {
                        value = $add(value, $load(row.as_ptr().add(offset) as *const $vector));
                    }
                    $store(dst.as_mut_ptr().add(offset) as *mut $vector, value);
                }
                offset += $lanes;
            }
            for index in offset..n {
                let mut value = bias[index];
                for row in rows {
                    value = value.wrapping_add(row[index]);
                }
                dst[index] = value;
            }
        }
    };
}

refresh_rows_kernel!(
    refresh_rows_sse41,
    "sse4.1",
    8,
    _mm_loadu_si128,
    _mm_storeu_si128,
    _mm_add_epi16,
    __m128i
);
refresh_rows_kernel!(
    refresh_rows_avx2,
    "avx2",
    16,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_add_epi16,
    __m256i
);
refresh_rows_kernel!(
    refresh_rows_avx512,
    "avx512f,avx512bw,avx512vl",
    32,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_add_epi16,
    __m512i
);
push_rows_kernel!(
    push_rows_sse41,
    "sse4.1",
    8,
    _mm_loadu_si128,
    _mm_storeu_si128,
    _mm_add_epi16,
    _mm_sub_epi16,
    __m128i
);
push_rows_kernel!(
    push_rows_avx2,
    "avx2",
    16,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_add_epi16,
    _mm256_sub_epi16,
    __m256i
);
push_rows_kernel!(
    push_rows_avx512,
    "avx512f,avx512bw,avx512vl",
    32,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_add_epi16,
    _mm512_sub_epi16,
    __m512i
);
/// Horizontal sum of the sixteen `i32` lanes into the `i64` total.
fn fold_lanes512(lanes: __m512i) -> i64 {
    let mut out = [0_i32; 16];
    unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, lanes) };
    out.iter().map(|value| i64::from(*value)).sum()
}

/// Horizontal sum of the eight `i32` lanes into the `i64` total.
fn fold_lanes(lanes: __m256i) -> i64 {
    let mut out = [0_i32; 8];
    unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, lanes) };
    out.iter().map(|value| i64::from(*value)).sum()
}

/// Horizontal sum of the four `i32` lanes into the `i64` total.
fn fold_lanes128(lanes: __m128i) -> i64 {
    let mut out = [0_i32; 4];
    unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, lanes) };
    out.iter().map(|value| i64::from(*value)).sum()
}
