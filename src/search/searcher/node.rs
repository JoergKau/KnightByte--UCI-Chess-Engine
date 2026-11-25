//! What every node needs around its search: the table probe helpers, the store
//! of a finished node with the correction-history update, and the small
//! predicates the pruning stages share.

use super::{MATE_LIMIT, MATE_SCORE, NO_STATIC_EVAL, Searcher};
use crate::board::position::Position;
use crate::search::history::Previous;
use crate::search::tt::{Bound, TtEntry, score_from_tt, score_to_tt};
use crate::types::color::Color;
use crate::types::r#move::Move;
use crate::types::piece::{self, EMPTY};
use crate::types::piece::{BISHOP, KNIGHT, QUEEN, ROOK};

/// Everything a finished negamax node contributes to the shared tables.
///
/// Grouping the fields keeps the store and the correction-history update in one
/// place, so the two exits of the node function cannot drift apart.
pub(crate) struct NodeResult<'a> {
    /// Position of the node.
    pub position: &'a Position,
    /// Distance from the root.
    pub ply: usize,
    /// Remaining depth after extensions.
    pub depth: i32,
    /// Score returned by the node.
    pub score: i32,
    /// Raw, correction-free static evaluation.
    pub raw_eval: i32,
    /// Static evaluation including the correction, as used by the pruners.
    pub corrected_eval: i32,
    /// Bound type the score represents.
    pub bound: Bound,
    /// Best move found (or the inherited table move on an upper bound).
    pub best_move: Move,
    /// Whether the node was in check.
    pub in_check: bool,
    /// Move that produced the node, for the countermove correction term.
    pub previous: Option<Previous>,
    /// Whether the node is a singular-verification call; its result is a
    /// bound probe, never stored.
    pub excluded: bool,
}

impl Searcher {
    /// Returns the legal moves the root is restricted to by `go searchmoves`.
    ///
    /// The result is empty when there is no restriction: either no filter was
    /// given, or none of its entries is among the `legal` moves (the
    /// unrestricted search then runs, as in the reference). The copy is fine:
    /// the root node is visited a handful of times per search, and holding a
    /// borrow of the filter across the move loop would fight the mutable
    /// searcher state.
    pub(crate) fn root_restriction(&self, legal: &[Move]) -> Vec<Move> {
        if self.root_filter.is_empty() {
            return Vec::new();
        }
        legal
            .iter()
            .filter(|mv| self.root_filter.contains(mv))
            .copied()
            .collect()
    }

    /// Stores a search result in the transposition table.
    ///
    /// The generation is stamped here so every search-code caller records the
    /// same value the ageing policy expects. When `searchstats` counting is
    /// on, the table's store outcome is folded into the searcher's counters.
    pub(crate) fn store_tt(&mut self, mut entry: TtEntry) {
        entry.generation = self.tables.tt.generation();
        let outcome = self.tables.tt.store(entry);
        if self.collect_stats {
            self.stats.record_tt_store(outcome);
        }
    }

    /// Converts a stored score back to search terms.
    pub(crate) fn probe_score(entry: &TtEntry, ply: usize) -> i32 {
        i32::from(score_from_tt(entry.score, ply as u16))
    }

    /// Returns whether a stored score is consistent with the raw static
    /// evaluation, and therefore safe to reuse as the evaluation.
    pub(crate) fn can_use_tt_score(entry: &TtEntry, raw_static_eval: i32, ply: usize) -> bool {
        let score = Self::probe_score(entry, ply);
        match entry.bound {
            Bound::Exact => true,
            Bound::Lower => score >= raw_static_eval,
            Bound::Upper => score <= raw_static_eval,
        }
    }

    /// Returns whether the static evaluation improved over the same line.
    ///
    /// The grandparent's eval is the comparison point: two plies are enough to
    /// see whether the current side's position is trending up or down. The
    /// root and the first ply have no comparison point and count as *not*
    /// improving, which is the reference's convention — pruning margins there
    /// use their not-improving scales.
    pub(crate) fn is_improving(&self, ply: usize, eval: i32) -> bool {
        self.params.use_improving_flag && ply >= 2 && eval > self.evals[ply - 2]
    }

    /// Returns whether `color` has a piece other than pawns and the king.
    pub(crate) fn has_non_pawn_material(position: &Position, color: Color) -> bool {
        let pieces = position.pieces(color, KNIGHT)
            | position.pieces(color, BISHOP)
            | position.pieces(color, ROOK)
            | position.pieces(color, QUEEN);
        pieces != 0
    }

    /// Stores the node result and folds its evaluation error into the
    /// correction history, then returns the score.
    ///
    /// A singular-verification node returns its score without storing: its
    /// window is artificial, so neither the table nor the correction history
    /// may learn from it.
    pub(crate) fn finish_node(&mut self, result: NodeResult<'_>) -> i32 {
        if result.excluded {
            return result.score;
        }
        self.store_tt(tt_entry(
            result.position,
            result.ply,
            result.depth,
            result.score,
            if result.in_check {
                NO_STATIC_EVAL
            } else {
                result.raw_eval
            },
            result.bound,
            result.best_move,
        ));

        // Reference update conditions: no mate score, not in check, and a quiet
        // best move — quiet in the correction-history sense: the destination is
        // not occupied by the opponent (no ordinary capture), and the move is
        // neither en passant nor a promotion. Bug B37 fix (reference bug 149):
        // the fixed reference tests the OPPONENT occupancy at the destination
        // (its pre-fix code tested the mover's own occupancy, which can never
        // fire, so ordinary captures wrongly passed as quiet and polluted the
        // correction tables); KnightByte's gate now classifies through
        // [`is_quiet_store_move`], which implements the corrected semantics.
        const TB_WIN_SCORE: i32 = 29_900;
        let trusted = match result.bound {
            Bound::Lower => result.raw_eval >= result.score,
            Bound::Upper => result.raw_eval <= result.score,
            Bound::Exact => false,
        };
        if self.params.use_correction_history
            && self.params.use_ref_correction_history
            && !result.in_check
            && result.score.abs() < MATE_LIMIT
            && result.score.abs() < TB_WIN_SCORE
            && (result.best_move.is_none()
                || is_quiet_store_move(result.position, result.best_move))
            && !trusted
        {
            self.correction.update(
                result.position,
                result.previous,
                result.score,
                result.corrected_eval,
                result.depth,
            );
        }
        result.score
    }
}

/// Builds a transposition-table entry, converting the score for `ply` and
/// clamping every field into its stored width.
pub(crate) fn tt_entry(
    position: &Position,
    ply: usize,
    depth: i32,
    score: i32,
    eval: i32,
    bound: Bound,
    mv: Move,
) -> TtEntry {
    TtEntry {
        key: position.key(),
        mv,
        score: score_to_tt(score.clamp(-MATE_SCORE, MATE_SCORE) as i16, ply as u16),
        eval: eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        depth: depth.clamp(0, i32::from(u8::MAX)) as u8,
        bound,
        // Replaced by `Searcher::store_tt`; kept here so the struct is complete.
        generation: 0,
    }
}

/// Returns the score of being mated in `ply` plies from the root.
#[inline]
pub(crate) const fn mated_in(ply: i32) -> i32 {
    -MATE_SCORE + ply
}

/// Returns `true` when `mv` counts as quiet in the correction-history update
/// gate.
///
/// Bug B37 fix (reference bug 149): quiet means the destination is NOT occupied
/// by the opponent (no ordinary capture) and the move is neither an en-passant
/// capture nor a promotion — the fixed reference's `isQuiet` semantics,
/// evaluated on the node position (`position.side()` is the mover at the store
/// site). Extracted from the gate so the classification is directly unit-
/// testable: the pre-fix gate omitted the capture disjunct, so ordinary
/// captures folded their eval-error diff into the correction tables.
///
/// `Move::NONE` counts as quiet (the gate's "no best move" branch).
pub(crate) fn is_quiet_store_move(position: &Position, mv: Move) -> bool {
    if mv.is_none() {
        return true;
    }
    let target = position.piece_at(mv.to());
    let capture = target != EMPTY && piece::color_of(target) != position.side();
    !capture && mv.flag() != crate::types::r#move::flags::EN_PASSANT && !mv.is_promotion()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::fen::STARTPOS_FEN;
    use crate::board::movegen::legal;
    use crate::board::position::Position;
    use crate::search::searcher::test_support::play;
    use crate::types::r#move::{Move, flags};

    #[test]
    fn the_correction_quiet_gate_accepts_quiet_moves_and_rejects_captures() {
        // Bug B37 fix (reference bug 149): an ordinary capture must not count
        // as quiet, so its eval-error diff no longer reaches the correction
        // tables. Built from the start position by play, so no new FEN
        // constant is introduced.
        let start = Position::from_fen(STARTPOS_FEN).unwrap();

        let quiet = legal(&start)
            .as_slice()
            .iter()
            .copied()
            .find(|mv| mv.to_uci() == "e2e3")
            .expect("e2e3 is legal in the start position");
        assert!(is_quiet_store_move(&start, quiet));
        assert!(is_quiet_store_move(&start, Move::NONE));

        // After 1.e4 d5 white's e4 pawn can capture the d5 pawn: a capture
        // must be rejected (the pre-fix gate accepted it).
        let after_e4_d5 = play(&start, &["e2e4", "d7d5"])[2];
        let capture = legal(&after_e4_d5)
            .as_slice()
            .iter()
            .copied()
            .find(|mv| mv.to_uci() == "e4d5")
            .expect("e4d5 is legal after 1.e4 d5");
        assert!(!is_quiet_store_move(&after_e4_d5, capture));

        // After 1.e4 a6 2.e5 d5 white's e5 pawn can capture en passant on d6:
        // also rejected, via the flag.
        let ep_position = play(&start, &["e2e4", "a7a6", "e4e5", "d7d5"])[4];
        let en_passant = legal(&ep_position)
            .as_slice()
            .iter()
            .copied()
            .find(|mv| mv.to_uci() == "e5d6")
            .expect("e5d6 en passant is legal after 1.e4 a6 2.e5 d5");
        assert_eq!(en_passant.flag(), flags::EN_PASSANT);
        assert!(!is_quiet_store_move(&ep_position, en_passant));

        // Promotion: classification only — the helper reads occupancy and
        // flags, not geometry, so the squares are arbitrary as long as the
        // target is empty (f6 here) and the flag says promotion.
        let promotion = Move::new(36, 53, flags::PROMO_QUEEN);
        assert!(!is_quiet_store_move(&ep_position, promotion));
    }
}
