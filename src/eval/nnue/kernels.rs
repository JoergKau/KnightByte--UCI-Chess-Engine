//! SIMD kernels for the accumulator arithmetic and the output head.
//!
//! The evaluation calls three bulk operations: the fused incremental update
//! of a child accumulator from its parent (`push_rows`), the full refresh of
//! one perspective (`refresh_rows`), and the SCReLU output head over a ready
//! accumulator pair. All three are integer-exact, so every instruction-set
//! backend must produce bit-identical results — the parity and bench tests
//! depend on it.
//!
//! The backend is selected once per process from [`crate::platform::cpu`]
//! (`Isa`), with the `KNIGHTBYTE_FORCE_ISA` override for tests and
//! CI. The x86 ladder maps `v3`, `v3-vnni` and `v4-vnni` to the 256-bit AVX2
//! kernels: the SCReLU head squares each element before the dot product,
//! which cannot use the VNNI dot-product instruction without changing the
//! per-element semantics, and Zen 4 executes 256-bit AVX2 natively. AArch64
//! (Apple Silicon) has exactly one compile-time backend: 128-bit NEON, which
//! is part of the AArch64 baseline and therefore safe on every Apple M chip.
//!
//! Selection is per process, not per call: the hot path pays one enum match
//! and then runs the chosen kernel without a feature test per node.

use std::sync::OnceLock;

use crate::eval::nnue::net::{EVAL_SCALE_MUL, EVAL_SCALE_SHIFT, QA, SCRELU_SHIFT};
use crate::types::color::Color;

#[cfg(target_arch = "x86_64")]
use crate::platform::cpu::{Isa, detect_isa};

/// The instruction-set backend chosen once per process.
///
/// Only the variants that exist on the compile target are defined, so the
/// dispatch matches stay exhaustive on every platform without a catch-all
/// arm that could silently downgrade a kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KernelKind {
    /// Pure Rust loops (the forced test fallback and non-SIMD builds).
    Scalar,
    /// 128-bit kernels over the x86-64 SSE4.1/SSE4.2 baseline.
    #[cfg(target_arch = "x86_64")]
    Sse41,
    /// 256-bit AVX2 kernels (the ladder's v3 and above).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// 512-bit AVX-512 kernels (the ladder's v4-vnni rung).
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// 128-bit NEON kernels (the AArch64 baseline, Apple Silicon).
    #[cfg(target_arch = "aarch64")]
    Neon,
}

static KERNEL: OnceLock<KernelKind> = OnceLock::new();

/// Returns the selected backend, honouring `KNIGHTBYTE_FORCE_ISA`.
fn kernel() -> KernelKind {
    *KERNEL.get_or_init(|| {
        // The override pins a backend so every variant can be
        // exercised on one machine in tests and CI. Names that do not exist
        // on this target are ignored, so the same CI script can set them.
        if let Ok(force) = std::env::var("KNIGHTBYTE_FORCE_ISA")
            && let Some(kernel) = forced_kernel(&force)
        {
            return kernel;
        }
        detected_kernel()
    })
}

/// Maps a `KNIGHTBYTE_FORCE_ISA` value to a backend on this target.
fn forced_kernel(name: &str) -> Option<KernelKind> {
    match name {
        "scalar" => Some(KernelKind::Scalar),
        #[cfg(target_arch = "x86_64")]
        "v2" | "sse41" => Some(KernelKind::Sse41),
        #[cfg(target_arch = "x86_64")]
        "v3" | "avx2" | "v3-vnni" => Some(KernelKind::Avx2),
        #[cfg(target_arch = "x86_64")]
        "v4-vnni" | "avx512" => Some(KernelKind::Avx512),
        #[cfg(target_arch = "aarch64")]
        "neon" | "aarch64" | "v8" => Some(KernelKind::Neon),
        _ => None,
    }
}

/// Maps the detected instruction set to the backend (x86-64 hosts).
#[cfg(target_arch = "x86_64")]
fn detected_kernel() -> KernelKind {
    match detect_isa() {
        Isa::X86V2 => KernelKind::Sse41,
        Isa::X86V3 | Isa::X86V3Vnni => KernelKind::Avx2,
        Isa::X86V4Vnni => KernelKind::Avx512,
        Isa::Aarch64 => KernelKind::Scalar,
    }
}

/// AArch64 hosts have exactly one backend: the AArch64 baseline includes
/// NEON, so every Apple M processor runs the same kernels.
#[cfg(target_arch = "aarch64")]
fn detected_kernel() -> KernelKind {
    KernelKind::Neon
}

/// Computes `dst = src - sum(subs) + sum(adds)` in one pass.
///
/// This is the incremental accumulator update: the parent perspective is
/// copied into the child while the changed weight rows are applied, so each
/// vector chunk is loaded once and stored once however many rows change.
/// `i16` wrapping addition is associative and commutative, so the result is
/// bit-identical to copying first and applying the rows one by one.
///
/// Every slice must be at least `dst.len()` long. The three row-count shapes
/// that moves produce (quiet move, capture or promotion, castling) have
/// specialised kernels; anything else takes the scalar path.
pub(crate) fn push_rows(dst: &mut [i16], src: &[i16], adds: &[&[i16]], subs: &[&[i16]]) {
    let n = dst.len();
    assert!(
        src.len() >= n && adds.iter().chain(subs).all(|row| row.len() >= n),
        "accumulator rows must cover the destination"
    );

    // SAFETY: the backend was selected from the detected (or forced and
    // verified) instruction set, and every slice covers `n`.
    macro_rules! by_shape {
        ($module:ident, $kernel:ident) => {
            match (adds.len(), subs.len()) {
                (1, 1) => unsafe {
                    $module::$kernel::<1, 1>(
                        dst,
                        src,
                        adds.try_into().unwrap(),
                        subs.try_into().unwrap(),
                    )
                },
                (1, 2) => unsafe {
                    $module::$kernel::<1, 2>(
                        dst,
                        src,
                        adds.try_into().unwrap(),
                        subs.try_into().unwrap(),
                    )
                },
                (2, 2) => unsafe {
                    $module::$kernel::<2, 2>(
                        dst,
                        src,
                        adds.try_into().unwrap(),
                        subs.try_into().unwrap(),
                    )
                },
                _ => scalar::push_rows(dst, src, adds, subs),
            }
        };
    }

    match kernel() {
        KernelKind::Scalar => scalar::push_rows(dst, src, adds, subs),
        #[cfg(target_arch = "x86_64")]
        KernelKind::Sse41 => by_shape!(wide, push_rows_sse41),
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx2 => by_shape!(wide, push_rows_avx2),
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx512 => by_shape!(wide, push_rows_avx512),
        #[cfg(target_arch = "aarch64")]
        KernelKind::Neon => by_shape!(neon, push_rows),
    }
}
/// Computes `dst = bias + sum(rows)` in one pass.
///
/// This is the full refresh of one perspective: every vector chunk starts from
/// the bias, collects the matching chunk of each active feature row in a
/// register and is stored once, instead of one load-modify-store pass over the
/// whole accumulator per row. Wrapping `i16` addition is associative and
/// commutative, so the result is bit-identical to adding the rows one by one.
///
/// `bias` and every row must be at least `dst.len()` long.
pub(crate) fn refresh_rows(dst: &mut [i16], bias: &[i16], rows: &[&[i16]]) {
    let n = dst.len();
    assert!(
        bias.len() >= n && rows.iter().all(|row| row.len() >= n),
        "accumulator rows must cover the destination"
    );
    match kernel() {
        KernelKind::Scalar => scalar::refresh_rows(dst, bias, rows),
        // SAFETY: the backend was selected from the detected (or forced and
        // verified) instruction set, and every slice covers `n`.
        #[cfg(target_arch = "x86_64")]
        KernelKind::Sse41 => unsafe { wide::refresh_rows_sse41(dst, bias, rows) },
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx2 => unsafe { wide::refresh_rows_avx2(dst, bias, rows) },
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx512 => unsafe { wide::refresh_rows_avx512(dst, bias, rows) },
        // SAFETY: NEON is part of the AArch64 baseline, so the selected
        // backend always runs on the host; every slice covers `n`.
        #[cfg(target_arch = "aarch64")]
        KernelKind::Neon => unsafe { neon::refresh_rows(dst, bias, rows) },
    }
}
/// Runs the SCReLU output head over a ready accumulator pair and returns the
/// scaled evaluation in centipawns from the side-to-move perspective.
///
/// `exact` selects the version-6 head: the clamped accumulator is squared in
/// full precision and accumulated before the bias is added. The historical
/// (version-5) chain shifts every squared element by eight bits first, which is
/// cheaper but truncates; both modes therefore need separate kernels.
pub(crate) fn evaluate_head(
    stm: Color,
    pair: &[Vec<i16>; 2],
    output_weights: &[i16],
    output_bias: i32,
    exact: bool,
) -> i32 {
    assert!(
        pair[1].len() == pair[0].len() && output_weights.len() >= 2 * pair[0].len(),
        "the output weights must cover both accumulators"
    );
    let sum = match kernel() {
        KernelKind::Scalar => scalar::evaluate_head(stm, pair, output_weights, exact),
        #[cfg(target_arch = "x86_64")]
        KernelKind::Sse41 => unsafe {
            if exact {
                wide::evaluate_head_sse41::<true>(stm, pair, output_weights)
            } else {
                wide::evaluate_head_sse41::<false>(stm, pair, output_weights)
            }
        },
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx2 => unsafe {
            if exact {
                wide::evaluate_head_avx2::<true>(stm, pair, output_weights)
            } else {
                wide::evaluate_head_avx2::<false>(stm, pair, output_weights)
            }
        },
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx512 => unsafe {
            if exact {
                wide::evaluate_head_avx512::<true>(stm, pair, output_weights)
            } else {
                wide::evaluate_head_avx512::<false>(stm, pair, output_weights)
            }
        },
        // SAFETY: NEON is part of the AArch64 baseline, and the accumulator
        // pair and the output weights were validated by the caller.
        #[cfg(target_arch = "aarch64")]
        KernelKind::Neon => unsafe {
            if exact {
                neon::evaluate_head::<true>(stm, pair, output_weights)
            } else {
                neon::evaluate_head::<false>(stm, pair, output_weights)
            }
        },
    };
    if exact {
        // The exact sum is scaled by 2^SCRELU_SHIFT relative to the historical
        // chain (`a^2` instead of `a^2 >> 8`), so the bias joins it in that
        // scale and the final shift grows by the same amount.
        let total = sum + (i64::from(output_bias) << SCRELU_SHIFT);
        ((total * EVAL_SCALE_MUL) >> (EVAL_SCALE_SHIFT + SCRELU_SHIFT as i64)) as i32
    } else {
        let total = sum + i64::from(output_bias);
        ((total * EVAL_SCALE_MUL) >> EVAL_SCALE_SHIFT) as i32
    }
}

/// The clamped square, before or after the historical per-element shift.
///
/// `exact = true` returns `clamp(v)^2`, which the version-6 head accumulates
/// directly; `exact = false` returns the version-5 value `clamp(v)^2 >> 8`.
#[inline(always)]
pub(crate) fn screlu_value(value: i16, exact: bool) -> i32 {
    let clamped = i32::from(value).clamp(0, QA);
    let square = clamped * clamped;
    if exact {
        square
    } else {
        square >> SCRELU_SHIFT
    }
}

pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
pub(crate) mod wide;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(test)]
mod tests;
