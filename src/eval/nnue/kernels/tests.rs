use super::{evaluate_head, push_rows, refresh_rows, scalar};
use crate::eval::nnue::net::{EVAL_SCALE_MUL, EVAL_SCALE_SHIFT};
use crate::types::color::{BLACK, WHITE};

/// A tiny deterministic generator for the kernel parity tests.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }

    /// Mixed magnitudes with saturated and negative values, so the clamps
    /// and the sign handling are exercised.
    fn i16(&mut self) -> i16 {
        let value = self.below(65536) as u16;
        (value >> 1) as i16
    }
}

#[test]
fn the_dispatched_backend_matches_the_scalar_reference() {
    let mut rng = Lcg(0x5EED);
    let l1 = 512;

    // The output head over random pairs, both perspectives and both head
    // chains (the shipped truncating one and the exact version-6 one).
    let output_weights: Vec<i16> = (0..2 * l1).map(|_| rng.i16()).collect();
    for stm in [WHITE, BLACK] {
        let pair = [
            (0..l1).map(|_| rng.i16()).collect::<Vec<i16>>(),
            (0..l1).map(|_| rng.i16()).collect::<Vec<i16>>(),
        ];
        for exact in [false, true] {
            let dispatched = evaluate_head(stm, &pair, &output_weights, -1_234, exact);
            let reference = scalar::evaluate_head(stm, &pair, &output_weights, exact);
            let scaled = if exact {
                (((reference + (i64::from(-1_234) << 8)) * EVAL_SCALE_MUL)
                    >> (EVAL_SCALE_SHIFT + 8)) as i32
            } else {
                (((reference - 1_234) * EVAL_SCALE_MUL) >> EVAL_SCALE_SHIFT) as i32
            };
            assert_eq!(
                dispatched, scaled,
                "the head diverges for side {stm}, exact {exact}: dispatched {dispatched}, reference {reference}"
            );
        }
    }
}

/// Builds `count` random rows of width `n`.
fn random_rows(rng: &mut Lcg, count: usize, n: usize) -> Vec<Vec<i16>> {
    (0..count)
        .map(|_| (0..n).map(|_| rng.i16()).collect())
        .collect()
}

#[test]
fn the_fused_row_update_matches_copy_then_apply() {
    let mut rng = Lcg(0xFACE);
    // Widths cover the production size, a multiple of 16 that is not a
    // multiple of 32, and an awkward tail.
    for n in [512, 48, 37] {
        for (add_count, sub_count) in [(1, 1), (1, 2), (2, 2), (0, 1), (3, 3)] {
            let src: Vec<i16> = (0..n).map(|_| rng.i16()).collect();
            let adds = random_rows(&mut rng, add_count, n);
            let subs = random_rows(&mut rng, sub_count, n);
            let add_refs: Vec<&[i16]> = adds.iter().map(Vec::as_slice).collect();
            let sub_refs: Vec<&[i16]> = subs.iter().map(Vec::as_slice).collect();

            // The obviously-correct oracle: copy, then one row at a time.
            let mut expected = src.clone();
            for row in &subs {
                for (value, weight) in expected.iter_mut().zip(row) {
                    *value = value.wrapping_sub(*weight);
                }
            }
            for row in &adds {
                for (value, weight) in expected.iter_mut().zip(row) {
                    *value = value.wrapping_add(*weight);
                }
            }

            let mut dispatched = vec![0_i16; n];
            push_rows(&mut dispatched, &src, &add_refs, &sub_refs);
            assert_eq!(
                dispatched, expected,
                "dispatched n={n} {add_count}/{sub_count}"
            );

            let mut reference = vec![0_i16; n];
            scalar::push_rows(&mut reference, &src, &add_refs, &sub_refs);
            assert_eq!(reference, expected, "scalar n={n} {add_count}/{sub_count}");
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn every_supported_fused_kernel_matches_the_scalar_reference() {
    use super::wide;
    let mut rng = Lcg(0xBEEF);
    for n in [512, 48, 37, 16] {
        let src: Vec<i16> = (0..n).map(|_| rng.i16()).collect();
        let rows = random_rows(&mut rng, 4, n);
        let (a1, a2) = (rows[0].as_slice(), rows[1].as_slice());
        let (s1, s2) = (rows[2].as_slice(), rows[3].as_slice());
        let mut expected = vec![0_i16; n];
        scalar::push_rows(&mut expected, &src, &[a1, a2], &[s1, s2]);

        // Each kernel runs only on a host that supports it.
        if is_x86_feature_detected!("sse4.1") {
            let mut out = vec![0_i16; n];
            unsafe { wide::push_rows_sse41::<2, 2>(&mut out, &src, &[a1, a2], &[s1, s2]) };
            assert_eq!(out, expected, "sse4.1 n={n}");
        }
        if is_x86_feature_detected!("avx2") {
            let mut out = vec![0_i16; n];
            unsafe { wide::push_rows_avx2::<2, 2>(&mut out, &src, &[a1, a2], &[s1, s2]) };
            assert_eq!(out, expected, "avx2 n={n}");
        }
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            let mut out = vec![0_i16; n];
            unsafe { wide::push_rows_avx512::<2, 2>(&mut out, &src, &[a1, a2], &[s1, s2]) };
            assert_eq!(out, expected, "avx512 n={n}");
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn every_supported_head_kernel_matches_the_scalar_reference() {
    use super::wide;
    type Head = unsafe fn(u8, &[Vec<i16>; 2], &[i16]) -> i64;
    let mut kernels: Vec<(&str, Head)> = Vec::new();
    if is_x86_feature_detected!("sse4.1") {
        kernels.push(("sse4.1", wide::evaluate_head_sse41::<false>));
        kernels.push(("sse4.1-exact", wide::evaluate_head_sse41::<true>));
    }
    if is_x86_feature_detected!("avx2") {
        kernels.push(("avx2", wide::evaluate_head_avx2::<false>));
        kernels.push(("avx2-exact", wide::evaluate_head_avx2::<true>));
    }
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
    {
        kernels.push(("avx512", wide::evaluate_head_avx512::<false>));
        kernels.push(("avx512-exact", wide::evaluate_head_avx512::<true>));
    }

    let mut rng = Lcg(0xD00D);
    // Random contents at several widths (including tails narrower than a
    // vector), then the worst case for the 32-bit lanes at the largest
    // width the loader accepts: every activation saturated and every
    // weight at an `i16` extreme, in both signs.
    struct Case {
        first: Vec<i16>,
        second: Vec<i16>,
        output_weights: Vec<i16>,
    }
    let mut cases: Vec<Case> = Vec::new();
    for l1 in [16, 48, 64, 512, 1040] {
        cases.push(Case {
            first: (0..l1).map(|_| rng.i16()).collect(),
            second: (0..l1).map(|_| rng.i16()).collect(),
            output_weights: (0..2 * l1).map(|_| rng.i16()).collect(),
        });
    }
    for weight in [i16::MAX, i16::MIN] {
        let l1 = 2048;
        cases.push(Case {
            first: vec![300; l1],
            second: vec![255; l1],
            output_weights: vec![weight; 2 * l1],
        });
    }

    for case in cases {
        let l1 = case.first.len();
        let output_weights = case.output_weights;
        let pair = [case.first, case.second];
        for stm in [WHITE, BLACK] {
            let truncating = scalar::evaluate_head(stm, &pair, &output_weights, false);
            let exact = scalar::evaluate_head(stm, &pair, &output_weights, true);
            for (name, kernel) in &kernels {
                let got = unsafe { kernel(stm, &pair, &output_weights) };
                let want = if name.ends_with("-exact") {
                    exact
                } else {
                    truncating
                };
                assert_eq!(got, want, "{name} head diverges at l1 {l1}, side {stm}");
            }
        }
    }
}

#[test]
fn the_fused_refresh_matches_adding_the_rows_one_by_one() {
    let mut rng = Lcg(0xC0DE);
    for n in [512, 48, 37] {
        for row_count in [0, 1, 2, 7, 30, 32] {
            let bias: Vec<i16> = (0..n).map(|_| rng.i16()).collect();
            let rows = random_rows(&mut rng, row_count, n);
            let refs: Vec<&[i16]> = rows.iter().map(Vec::as_slice).collect();

            let mut expected = bias.clone();
            for row in &rows {
                for (value, weight) in expected.iter_mut().zip(row) {
                    *value = value.wrapping_add(*weight);
                }
            }

            let mut dispatched = vec![0_i16; n];
            refresh_rows(&mut dispatched, &bias, &refs);
            assert_eq!(dispatched, expected, "dispatched n={n} rows={row_count}");

            let mut reference = vec![0_i16; n];
            scalar::refresh_rows(&mut reference, &bias, &refs);
            assert_eq!(reference, expected, "scalar n={n} rows={row_count}");

            #[cfg(target_arch = "x86_64")]
            {
                use super::wide;
                // Each kernel runs only on a host that supports it.
                if is_x86_feature_detected!("sse4.1") {
                    let mut out = vec![0_i16; n];
                    unsafe { wide::refresh_rows_sse41(&mut out, &bias, &refs) };
                    assert_eq!(out, expected, "sse4.1 n={n} rows={row_count}");
                }
                if is_x86_feature_detected!("avx2") {
                    let mut out = vec![0_i16; n];
                    unsafe { wide::refresh_rows_avx2(&mut out, &bias, &refs) };
                    assert_eq!(out, expected, "avx2 n={n} rows={row_count}");
                }
                if is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512vl")
                {
                    let mut out = vec![0_i16; n];
                    unsafe { wide::refresh_rows_avx512(&mut out, &bias, &refs) };
                    assert_eq!(out, expected, "avx512 n={n} rows={row_count}");
                }
            }
        }
    }
}

/// The NEON backend must match the scalar oracle at production widths,
/// at widths with a tail narrower than one vector, and in the worst case
/// for the 32-bit head lanes (saturated activations and extreme weights).
#[cfg(target_arch = "aarch64")]
#[test]
fn the_neon_kernels_match_the_scalar_reference() {
    use super::neon;

    let mut rng = Lcg(0x0A11);
    for n in [512, 48, 37, 8] {
        let src: Vec<i16> = (0..n).map(|_| rng.i16()).collect();
        let rows = random_rows(&mut rng, 4, n);
        let (a1, a2) = (rows[0].as_slice(), rows[1].as_slice());
        let (s1, s2) = (rows[2].as_slice(), rows[3].as_slice());

        let mut expected = vec![0_i16; n];
        scalar::push_rows(&mut expected, &src, &[a1, a2], &[s1, s2]);
        let mut out = vec![0_i16; n];
        unsafe { neon::push_rows::<2, 2>(&mut out, &src, &[a1, a2], &[s1, s2]) };
        assert_eq!(out, expected, "neon push_rows n={n}");

        // The full refresh over a row count that crosses the vector width.
        let mut expected = vec![0_i16; n];
        scalar::refresh_rows(&mut expected, &src, &[a1, a2, s1, s2, s1]);
        let mut out = vec![0_i16; n];
        unsafe { neon::refresh_rows(&mut out, &src, &[a1, a2, s1, s2, s1]) };
        assert_eq!(out, expected, "neon refresh_rows n={n}");
    }

    // The head: random contents first, then the extreme cases, in both the
    // historical and the exact chain.
    for l1 in [8, 16, 48, 512, 1040, 2048] {
        let pair = [
            (0..l1).map(|_| rng.i16()).collect::<Vec<i16>>(),
            (0..l1).map(|_| rng.i16()).collect::<Vec<i16>>(),
        ];
        let weights: Vec<i16> = (0..2 * l1).map(|_| rng.i16()).collect();
        for stm in [WHITE, BLACK] {
            for exact in [false, true] {
                let expected = scalar::evaluate_head(stm, &pair, &weights, exact);
                let got = unsafe {
                    if exact {
                        neon::evaluate_head::<true>(stm, &pair, &weights)
                    } else {
                        neon::evaluate_head::<false>(stm, &pair, &weights)
                    }
                };
                assert_eq!(got, expected, "neon head l1={l1} side={stm} exact={exact}");
            }
        }
    }
    for weight in [i16::MAX, i16::MIN] {
        let l1 = 2048;
        let pair = [vec![300_i16; l1], vec![255_i16; l1]];
        let weights = vec![weight; 2 * l1];
        for stm in [WHITE, BLACK] {
            for exact in [false, true] {
                let expected = scalar::evaluate_head(stm, &pair, &weights, exact);
                let got = unsafe {
                    if exact {
                        neon::evaluate_head::<true>(stm, &pair, &weights)
                    } else {
                        neon::evaluate_head::<false>(stm, &pair, &weights)
                    }
                };
                assert_eq!(
                    got, expected,
                    "neon head worst case weight={weight} side={stm} exact={exact}"
                );
            }
        }
    }
}
