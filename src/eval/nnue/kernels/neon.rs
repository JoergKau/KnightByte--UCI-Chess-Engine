//! The 128-bit AArch64 kernels.
//!
//! NEON is part of the AArch64 baseline, so every Apple M processor runs
//! these kernels and no runtime feature test is needed. The row kernels
//! use the same fused loop shape as the x86 kernels: one load of the
//! source chunk, the row additions/subtractions in registers, one store.
//! Wrapping `i16` addition is associative and commutative, so the result
//! is bit-identical to the scalar oracle.
//!
//! The head reproduces the scalar chain exactly: the clamped activation is
//! squared in 16-bit lanes (its maximum 255^2 = 65,025 still fits an
//! unsigned 16-bit lane), shifted right by eight bits (leaving at most
//! 254), widened with `vmlal_s16` and accumulated in `i32` lanes. The
//! lanes fold into the `i64` total every 16 vectors, which bounds every
//! partial sum far below `i32` overflow, so the total equals the scalar
//! `i64` sum exactly.

use core::arch::aarch64::*;

use super::screlu_value;
use crate::types::color::Color;

/// Number of `i16` lanes in one 128-bit NEON vector.
const LANES: usize = 8;

/// Vectors accumulated in the `i32` lanes before folding into the `i64`
/// total. Two perspectives contribute per iteration, so each lane sees at
/// most `2 * 2 * FOLD_EVERY` products of `254 * 32768`, about `2^29`:
/// far below `i32` overflow.
const FOLD_EVERY: u32 = 16;

/// `dst = src - sum(subs) + sum(adds)`, eight lanes at a time.
///
/// # Safety
///
/// `src` and every row must hold at least `dst.len()` elements.
#[target_feature(enable = "neon")]
pub(super) unsafe fn push_rows<const A: usize, const S: usize>(
    dst: &mut [i16],
    src: &[i16],
    adds: &[&[i16]; A],
    subs: &[&[i16]; S],
) {
    let n = dst.len();
    let mut offset = 0;
    while offset + LANES <= n {
        unsafe {
            let mut value = vld1q_s16(src.as_ptr().add(offset));
            for row in subs {
                value = vsubq_s16(value, vld1q_s16(row.as_ptr().add(offset)));
            }
            for row in adds {
                value = vaddq_s16(value, vld1q_s16(row.as_ptr().add(offset)));
            }
            vst1q_s16(dst.as_mut_ptr().add(offset), value);
        }
        offset += LANES;
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

/// `dst = bias + sum(rows)`, eight lanes at a time.
///
/// # Safety
///
/// `bias` and every row must hold at least `dst.len()` elements.
#[target_feature(enable = "neon")]
pub(super) unsafe fn refresh_rows(dst: &mut [i16], bias: &[i16], rows: &[&[i16]]) {
    let n = dst.len();
    let mut offset = 0;
    while offset + LANES <= n {
        unsafe {
            let mut value = vld1q_s16(bias.as_ptr().add(offset));
            for row in rows {
                value = vaddq_s16(value, vld1q_s16(row.as_ptr().add(offset)));
            }
            vst1q_s16(dst.as_mut_ptr().add(offset), value);
        }
        offset += LANES;
    }
    for index in offset..n {
        let mut value = bias[index];
        for row in rows {
            value = value.wrapping_add(row[index]);
        }
        dst[index] = value;
    }
}

/// The SCReLU output head over both perspectives.
///
/// `EXACT` selects the version-6 head: the clamped square is split into
/// `256 * v + r`, both dot products are accumulated and combined at the
/// end. The lane bounds are identical to the historical chain.
///
/// # Safety
///
/// `output_weights` must hold at least twice the accumulator width.
#[target_feature(enable = "neon")]
pub(super) unsafe fn evaluate_head<const EXACT: bool>(
    stm: Color,
    pair: &[Vec<i16>; 2],
    output_weights: &[i16],
) -> i64 {
    let stm_acc = &pair[stm as usize];
    let ntm_acc = &pair[(stm ^ 1) as usize];
    let l1 = stm_acc.len();
    let zero = vdupq_n_s16(0);
    let qa = vdupq_n_s16(crate::eval::nnue::net::QA as i16);
    let low_byte = vdupq_n_u16(255);
    let mut total = 0_i64;
    let mut correction_total = 0_i64;
    let mut lanes = vdupq_n_s32(0);
    let mut correction = vdupq_n_s32(0);
    let mut pending = 0_u32;
    let mut offset = 0;
    while offset + LANES <= l1 {
        // The two perspectives read their own half of the output weights.
        for (accumulator, weight_base) in [(stm_acc, offset), (ntm_acc, l1 + offset)] {
            unsafe {
                let a = vld1q_s16(accumulator.as_ptr().add(offset));
                let clamped = vminq_s16(vmaxq_s16(a, zero), qa);
                // The clamped square is at most 65,025, so the unsigned
                // 16-bit product is exact.
                let square = vreinterpretq_u16_s16(vmulq_s16(clamped, clamped));
                let w = vld1q_s16(output_weights.as_ptr().add(weight_base));
                let v = vreinterpretq_s16_u16(vshrq_n_u16::<8>(square));
                lanes = vmlal_s16(lanes, vget_low_s16(v), vget_low_s16(w));
                lanes = vmlal_s16(lanes, vget_high_s16(v), vget_high_s16(w));
                if EXACT {
                    let r = vreinterpretq_s16_u16(vandq_u16(square, low_byte));
                    correction = vmlal_s16(correction, vget_low_s16(r), vget_low_s16(w));
                    correction = vmlal_s16(correction, vget_high_s16(r), vget_high_s16(w));
                }
            }
        }
        offset += LANES;
        pending += 1;
        if pending == FOLD_EVERY {
            total += fold_lanes(lanes);
            if EXACT {
                correction_total += fold_lanes(correction);
            }
            lanes = vdupq_n_s32(0);
            correction = vdupq_n_s32(0);
            pending = 0;
        }
    }
    total += fold_lanes(lanes);
    if EXACT {
        correction_total += fold_lanes(correction);
        total = (total << 8) + correction_total;
    }
    for index in offset..l1 {
        let a = stm_acc[index];
        if a > 0 {
            total += i64::from(screlu_value(a, EXACT)) * i64::from(output_weights[index]);
        }
        let b = ntm_acc[index];
        if b > 0 {
            total += i64::from(screlu_value(b, EXACT)) * i64::from(output_weights[l1 + index]);
        }
    }
    total
}

/// Horizontal sum of the four `i32` lanes into the `i64` total.
fn fold_lanes(lanes: int32x4_t) -> i64 {
    let mut out = [0_i32; 4];
    unsafe { vst1q_s32(out.as_mut_ptr(), lanes) };
    out.iter().map(|value| i64::from(*value)).sum()
}
