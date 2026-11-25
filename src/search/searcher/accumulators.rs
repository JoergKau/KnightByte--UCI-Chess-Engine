//! The per-ply network accumulators of the search stack.
//!
//! A move writes the child's accumulator pair from the parent's in one fused
//! pass; a null move copies it. The static evaluation of a node reads the pair
//! at its ply, through the evaluation cache when that is switched on.

use super::Searcher;
use crate::board::position::Position;
use crate::search::MAX_PLY;
use crate::types::color::{BLACK, WHITE};
use crate::types::r#move::Move;

impl Searcher {
    /// Allocates the per-ply accumulator pairs over the embedded network's
    /// first-layer width.
    pub(super) fn accumulator_stack() -> Vec<[Vec<i16>; 2]> {
        let l1 = crate::eval::nnue::embedded_net().l1();
        (0..=MAX_PLY + 1)
            .map(|_| [vec![0_i16; l1], vec![0_i16; l1]])
            .collect()
    }

    /// Returns the static evaluation of `position` from the accumulator pair
    /// at `ply`, optionally via the cache.
    ///
    /// The pair at `ply` must be current for `position` (pushed by the parent
    /// node or refreshed at the search root). The evaluation reuses this
    /// searcher's [`Evaluator`] for the output head.
    pub(crate) fn evaluate_stacked(&mut self, position: &Position, ply: usize) -> i32 {
        let use_cache = self.params.use_eval_cache;
        let key = position.key();
        if use_cache && let Some(cached) = self.tables.eval_cache.probe(key) {
            return i32::from(cached);
        }
        let net = self.evaluator.net();
        let value = crate::eval::nnue::evaluate::evaluate_pair(
            net,
            position.side(),
            &self.accumulator_stack[ply],
            crate::eval::nnue::features::output_bucket(net, position),
        );
        // The store is gated by the same lever as the probe: with the cache
        // off nothing ever reads it, so writing would only cost a shared
        // cache line per node (and ping it between threads).
        if use_cache {
            self.tables
                .eval_cache
                .store(key, value.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
        }
        value
    }

    /// Full refresh of the accumulator pair at `ply` from `position`.
    ///
    /// Used at the search root and whenever a perspective's own king moved.
    pub(crate) fn refresh_accumulators(&mut self, position: &Position, ply: usize) {
        let net = self.evaluator.net();
        let pair = &mut self.accumulator_stack[ply];
        crate::eval::nnue::evaluate::refresh_into(net, position, WHITE, &mut pair[0]);
        crate::eval::nnue::evaluate::refresh_into(net, position, BLACK, &mut pair[1]);
    }

    /// Copies the validated pair of `ply` into `ply + 1` (the null-move child:
    /// same placement, only the side to move flips).
    pub(crate) fn copy_accumulators(&mut self, ply: usize) {
        let (head, tail) = self.accumulator_stack.split_at_mut(ply + 1);
        tail[0].clone_from(&head[ply]);
    }

    /// Computes the accumulator pair at `ply + 1` by applying `mv` to the
    /// validated pair at `ply`.
    ///
    /// A perspective whose own king moved falls back to a full refresh of
    /// `child`: its region and mirror may have changed, which invalidates
    /// every row. All other perspectives keep their frame (the parent's), so
    /// at most four rows change: the moved piece, the captured piece (its own
    /// square for en passant), the promotion piece and the castling rook.
    /// The delta application lives in [`crate::eval::nnue::evaluate`], next
    /// to the row kernels it calls.
    pub(crate) fn push_accumulators(
        &mut self,
        parent: &Position,
        child: &Position,
        mv: Move,
        ply: usize,
    ) {
        let net = self.evaluator.net();
        let (parent_stack, child_stack) = self.accumulator_stack.split_at_mut(ply + 1);
        crate::eval::nnue::evaluate::push_accumulators(
            net,
            parent,
            child,
            mv,
            &parent_stack[ply],
            &mut child_stack[0],
        );
    }

    /// Evaluates one child position through the incremental path.
    ///
    /// Diagnostic hook for the release microbench: it pushes the child
    /// accumulator pair from the pair at ply 0 and then runs the output head.
    /// The pair at ply 0 must already describe `position` (a completed search
    /// leaves it ready). Not used by the search itself.
    #[doc(hidden)]
    pub fn bench_incremental_eval(&mut self, position: &Position, mv: Move) -> i32 {
        let child = position.make_move(mv);
        self.push_accumulators(position, &child, mv, 0);
        let net = self.evaluator.net();
        crate::eval::nnue::evaluate::evaluate_pair(
            net,
            child.side(),
            &self.accumulator_stack[1],
            crate::eval::nnue::features::output_bucket(net, &child),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::fen::STARTPOS_FEN;
    use crate::board::movegen::legal;
    use crate::board::position::Position;

    /// A tiny deterministic generator for the accumulator walk.
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % n
        }
    }

    /// Plays `moves` from the start position with a fresh searcher, pushing
    /// the accumulator incrementally and comparing it against the full
    /// refresh after every move. Returns the castle and en-passant counts.
    fn assert_incremental_walk(moves: &[&str]) -> (u32, u32) {
        let mut searcher = Searcher::new(16);
        let mut position = Position::from_fen(STARTPOS_FEN).expect("startpos");
        searcher.refresh_accumulators(&position, 0);
        let mut castles = 0;
        let mut en_passants = 0;
        for (ply, uci) in moves.iter().enumerate() {
            let legal_moves = legal(&position);
            let mv = *legal_moves
                .as_slice()
                .iter()
                .find(|candidate| candidate.to_uci() == *uci)
                .unwrap_or_else(|| panic!("move {uci} is not legal at ply {ply}"));
            let next = position.make_move(mv);
            searcher.push_accumulators(&position, &next, mv, ply);
            let net = searcher.evaluator.net();
            let incremental = crate::eval::nnue::evaluate::evaluate_pair(
                net,
                next.side(),
                &searcher.accumulator_stack[ply + 1],
                crate::eval::nnue::features::output_bucket(net, &next),
            );
            let full = searcher.evaluator.evaluate(&next);
            assert_eq!(
                incremental, full,
                "ply {ply} after {uci}: incremental eval diverges"
            );
            if mv.flag() == crate::types::r#move::flags::CASTLE {
                castles += 1;
            }
            if mv.flag() == crate::types::r#move::flags::EN_PASSANT {
                en_passants += 1;
            }
            position = next;
        }
        (castles, en_passants)
    }

    /// The incremental accumulator must reproduce the full refresh
    /// bit-exactly at every position of every walk. The deterministic
    /// segments cover the en-passant capture and castling for both sides; the
    /// seeded random walks add broad coverage (the promotion delta has its
    /// own test below).
    #[test]
    fn the_incremental_accumulator_matches_the_full_refresh() {
        // En passant: the white e-pawn captures the d5 pawn behind it.
        let (castles, en_passants) =
            assert_incremental_walk(&["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"]);
        assert_eq!(en_passants, 1, "the en-passant capture must be walked");
        assert_eq!(castles, 0);

        // Both sides castle: the rook delta runs for white and for black.
        let (castles, en_passants) = assert_incremental_walk(&[
            "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1", "f8c5", "d2d3", "e8g8",
        ]);
        assert_eq!(castles, 2, "both sides must have castled");
        assert_eq!(en_passants, 0);

        // Seeded random walks: 8 x 80 plies over the start position. The
        // identity check at every pushed position is the point; special move
        // deltas are pinned by the deterministic segments above.
        for seed in 1..=8u64 {
            let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut searcher = Searcher::new(16);
            let mut position = Position::from_fen(STARTPOS_FEN).expect("startpos");
            searcher.refresh_accumulators(&position, 0);
            for ply in 0..80usize {
                let moves = legal(&position);
                if moves.is_empty() {
                    break;
                }
                let mv = moves.as_slice()[rng.below(moves.len() as u64) as usize];
                let next = position.make_move(mv);
                searcher.push_accumulators(&position, &next, mv, ply);
                let net = searcher.evaluator.net();
                let incremental = crate::eval::nnue::evaluate::evaluate_pair(
                    net,
                    next.side(),
                    &searcher.accumulator_stack[ply + 1],
                    crate::eval::nnue::features::output_bucket(net, &next),
                );
                let full = searcher.evaluator.evaluate(&next);
                assert_eq!(
                    incremental,
                    full,
                    "seed {seed} ply {ply} after {}: incremental eval diverges",
                    mv.to_uci()
                );
                position = next;
            }
        }
        // Castle coverage comes from the deterministic segment above; random
        // games rarely castle, so no coverage assert here.
    }

    /// The promotion delta matches the full refresh. The FEN is verified
    /// against the external verifier at test time (project rule): the white
    /// b-pawn promotes on the empty b8 square.
    #[test]
    fn the_promotion_delta_matches_the_full_refresh() {
        // 6k1/1P6/8/8/8/8/8/6K1 w - - 0 1
        const PROMO_FEN: &str = "6k1/1P6/8/8/8/8/5PPP/6K1 w - - 0 1";
        verify_fen_externally(PROMO_FEN);

        let mut searcher = Searcher::new(16);
        let position = Position::from_fen(PROMO_FEN).expect("verified FEN");
        searcher.refresh_accumulators(&position, 0);
        let moves = legal(&position);
        let mv = *moves
            .as_slice()
            .iter()
            .find(|mv| mv.to_uci() == "b7b8q")
            .expect("the promotion must be legal");
        let child = position.make_move(mv);
        searcher.push_accumulators(&position, &child, mv, 0);
        let net = searcher.evaluator.net();
        let incremental = crate::eval::nnue::evaluate::evaluate_pair(
            net,
            child.side(),
            &searcher.accumulator_stack[1],
            crate::eval::nnue::features::output_bucket(net, &child),
        );
        let full = searcher.evaluator.evaluate(&child);
        assert_eq!(
            incremental, full,
            "promotion delta diverges: incremental {incremental}, full {full}"
        );
    }

    /// Verifies one FEN through the configured external verifier, so the test
    /// FENs of this module obey the project rule without the integration-test
    /// helper module.
    fn verify_fen_externally(fen: &str) {
        use std::io::Write as _;
        use std::process::{Command, Stdio};

        let verifier = std::env::var_os("KNIGHTBYTE_FEN_VERIFIER")
            .expect("set KNIGHTBYTE_FEN_VERIFIER before running the tests");
        let mut child = Command::new(verifier)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("cannot start the verifier");
        child
            .stdin
            .as_mut()
            .expect("verifier stdin")
            .write_all(
                format!("uci\nisready\nposition fen {fen}\nd\ngo depth 1\nquit\n").as_bytes(),
            )
            .expect("write verifier commands");
        let stdout = child.wait_with_output().expect("wait for verifier").stdout;
        let stdout = String::from_utf8_lossy(&stdout).into_owned();
        assert!(!stdout.contains("Illegal"), "verifier rejected {fen}");
        assert!(stdout.contains("bestmove"), "verifier produced no bestmove");
        let reported = stdout
            .lines()
            .find_map(|line| line.strip_prefix("Fen: "))
            .expect("verifier did not report a FEN");
        let input: Vec<&str> = fen.split_whitespace().take(4).collect();
        let reported: Vec<&str> = reported.split_whitespace().take(4).collect();
        assert_eq!(input, reported, "FEN did not round-trip: {fen}");
    }

    #[test]
    fn the_eval_cache_lever_gates_the_cache_consult() {
        // Step 11: `UseEvalCache` is a pure memoisation — identical values,
        // identical node counts — so the node-count independence test cannot
        // see it. The behavioural proof: with the lever on, `evaluate`
        // returns the CACHED value (here a deliberately wrong one); with it
        // off, the cache is never consulted and the real evaluation comes
        // back.
        let mut searcher = Searcher::new(2);
        let position = Position::from_fen(STARTPOS_FEN).expect("the start FEN must parse");
        let key = position.key();
        searcher.tables.eval_cache.store(key, 12_345);
        searcher.refresh_accumulators(&position, 0);

        searcher.params_mut().use_eval_cache = true;
        assert_eq!(
            searcher.evaluate_stacked(&position, 0),
            12_345,
            "with the lever on the cache must be consulted"
        );
        searcher.params_mut().use_eval_cache = false;
        assert_ne!(
            searcher.evaluate_stacked(&position, 0),
            12_345,
            "with the lever off the cache must be ignored"
        );
    }

    #[test]
    fn the_eval_cache_is_not_written_while_the_lever_is_off() {
        // B42: the store used to run unconditionally, writing a cache that
        // nothing read. With the lever off the cache must stay untouched;
        // with it on the evaluation must be stored and then served.
        let mut searcher = Searcher::new(2);
        let position = Position::from_fen(STARTPOS_FEN).expect("the start FEN must parse");
        let key = position.key();
        searcher.refresh_accumulators(&position, 0);

        searcher.params_mut().use_eval_cache = false;
        let value = searcher.evaluate_stacked(&position, 0);
        assert_eq!(
            searcher.tables.eval_cache.probe(key),
            None,
            "an evaluation must not be stored while the lever is off"
        );

        searcher.params_mut().use_eval_cache = true;
        assert_eq!(searcher.evaluate_stacked(&position, 0), value);
        assert_eq!(
            searcher.tables.eval_cache.probe(key),
            Some(value as i16),
            "with the lever on the evaluation must be stored"
        );
    }
}
