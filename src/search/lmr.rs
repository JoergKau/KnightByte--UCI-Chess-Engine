//! Late move reductions.
//!
//! Moves that come late in the ordering are unlikely to be best, so they are
//! searched to a reduced depth and only searched fully if the reduced search
//! surprises. This file holds everything that decides *how much*:
//!
//! * the base table, indexed by remaining depth and move number;
//! * the schedule that adds the adjustment terms to the base and returns the
//!   reduction in plies (`lmr_reduction`, a pure function of its inputs);
//! * the gate that says which moves are reduced at all (`lmr_applies`);
//! * the depth of the re-search after a reduced search failed high.
//!
//! The schedule is fixed point over 1024ths of a ply.

use super::history::lmr_history_term_1024;
use super::params::SearchParams;
use super::searcher::Searcher;

/// One ply in the fixed-point unit of the schedule.
const PLY: i32 = 1024;

/// A mover with at most this many pieces left is in a bare endgame, where late
/// moves matter less and the reduction grows by a ply.
const BARE_ENDGAME_PIECES: u32 = 4;

/// Side of the late-move-reduction base table, indexed `[depth][move number]`.
pub(crate) const LMR_TABLE_SIZE: usize = 64;

/// The late-move-reduction base table, indexed `[depth][move number]`.
pub(crate) type LmrTable = [[i32; LMR_TABLE_SIZE]; LMR_TABLE_SIZE];

/// Builds the late-move-reduction base table:
/// `int(1.0 + ln(depth) * ln(move_number) / (divisor / 100))`, zero at either
/// index zero. The default divisor of 175 gives the reference's 1.75 exactly
/// (175 / 100 is representable), so the default table is unchanged.
pub(crate) fn build_lmr_table(divisor_hundredths: i32) -> Box<LmrTable> {
    let divisor = f64::from(divisor_hundredths) / 100.0;
    let mut table = Box::new([[0_i32; LMR_TABLE_SIZE]; LMR_TABLE_SIZE]);
    for (depth, row) in table.iter_mut().enumerate().skip(1) {
        for (moves, cell) in row.iter_mut().enumerate().skip(1) {
            let value = 1.0 + (depth as f64).ln() * (moves as f64).ln() / divisor;
            *cell = value as i32;
        }
    }
    table
}

/// Returns the depth for the full-depth zero-window re-search after a reduced
/// search failed high.
///
/// A large original reduction shaves one or two plies off the re-search - a
/// move that wanted a big reduction rarely needs full re-verification - and
/// the result is floored at the reduced depth so the re-search is never
/// shallower than the search it improves on.
pub(crate) fn compute_re_search_depth(new_depth: i32, reduction: i32, enabled: bool) -> i32 {
    if !enabled {
        return new_depth;
    }
    let mut re_search_depth = new_depth;
    if reduction >= 3 {
        re_search_depth -= 1;
    }
    if reduction >= 5 && new_depth > 2 {
        re_search_depth -= 1;
    }
    let min_depth = new_depth - reduction;
    if re_search_depth < min_depth {
        re_search_depth = min_depth;
    }
    re_search_depth
}

impl Searcher {
    /// Returns the base late-move reduction for `depth` and the 1-based
    /// `move_number`, read from the searcher's table with clamped indices.
    pub(crate) fn lmr_base_reduction(&self, depth: i32, move_number: i32) -> i32 {
        let d = depth.clamp(0, LMR_TABLE_SIZE as i32 - 1) as usize;
        let m = move_number.clamp(0, LMR_TABLE_SIZE as i32 - 1) as usize;
        self.lmr_table[d][m]
    }
}

/// The facts about one move that the reduction schedule reads.
#[derive(Clone, Copy)]
pub(crate) struct LmrInput {
    /// The base reduction from the table for this depth and move number.
    pub base: i32,
    /// Pieces the moving side has left after the move (pawns and king
    /// included).
    pub mover_pieces_left: u32,
    /// Whether the side to move is improving.
    pub improving: bool,
    /// The combined history score of the move.
    pub history_score: i32,
    /// Whether the node is predicted to fail high.
    pub cut_node: bool,
    /// Whether the node is a principal-variation node.
    pub pv_node: bool,
    /// Whether the move is a killer or the countermove of the previous move.
    pub killer_or_counter: bool,
    /// The depth the move would be searched to without a reduction; the
    /// reduction never exceeds it.
    pub new_depth: i32,
}

/// Returns whether a move is a candidate for reduction at all: the table is
/// switched on, the node is deep enough, the move is late enough, and it is
/// quiet (`noisy` is a capture or a promotion) and does not involve a check
/// (`check_involved` is the node in check or the move giving check).
pub(crate) fn lmr_applies(
    params: &SearchParams,
    depth: i32,
    move_number: i32,
    noisy: bool,
    check_involved: bool,
) -> bool {
    params.use_lmr_table
        && depth >= params.lmr_min_depth
        && move_number > params.lmr_min_moves
        && !noisy
        && !check_involved
}

/// Returns the reduction in plies of one move, between zero and
/// `input.new_depth`.
///
/// The base from the table gets a ply more in a bare endgame, a ply more when
/// not improving (with `use_term1_not_improving`), a ply more at a predicted
/// cut node, and loses what the move's history earns, the PV weight at PV
/// nodes and the killer weight for a killer or countermove.
pub(crate) fn lmr_reduction(params: &SearchParams, input: LmrInput) -> i32 {
    let mut reduction = input.base * PLY;
    if input.mover_pieces_left <= BARE_ENDGAME_PIECES {
        reduction += PLY;
    }
    if params.use_term1_not_improving && !input.improving {
        reduction += PLY;
    }
    reduction -= lmr_history_term_1024(
        input.history_score,
        params.lmr_history_divisor,
        params.use_lmr_history_precision,
    );
    if params.use_cut_node && input.cut_node {
        reduction += PLY;
    }
    if input.pv_node {
        reduction -= params.lmr_pv_weight;
    }
    if input.killer_or_counter {
        reduction -= params.lmr_killer_weight;
    }
    (reduction / PLY).clamp(0, input.new_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SearchParams {
        SearchParams::default()
    }

    fn input() -> LmrInput {
        LmrInput {
            base: 3,
            mover_pieces_left: 12,
            improving: true,
            history_score: 0,
            cut_node: false,
            pv_node: false,
            killer_or_counter: false,
            new_depth: 20,
        }
    }

    #[test]
    fn the_base_table_follows_the_log_formula() {
        let table = build_lmr_table(175);
        assert_eq!(table[0][5], 0);
        assert_eq!(table[5][0], 0);
        // int(1 + ln(12) * ln(8) / 1.75) = int(3.95) = 3.
        assert_eq!(table[12][8], 3);
        assert!(table[20][20] > table[4][4]);
        // A larger divisor reduces less.
        assert!(build_lmr_table(300)[20][20] < table[20][20]);
    }

    #[test]
    fn the_schedule_starts_from_the_base_and_adds_one_ply_per_term() {
        let p = params();
        assert_eq!(lmr_reduction(&p, input()), 3);
        let cut = LmrInput {
            cut_node: true,
            ..input()
        };
        assert_eq!(lmr_reduction(&p, cut), 4);
        let endgame = LmrInput {
            mover_pieces_left: 4,
            ..input()
        };
        assert_eq!(lmr_reduction(&p, endgame), 4);
        let mut term1 = p;
        term1.use_term1_not_improving = true;
        let worsening = LmrInput {
            improving: false,
            ..input()
        };
        assert_eq!(lmr_reduction(&term1, worsening), 4);
        assert_eq!(lmr_reduction(&p, worsening), 3, "the term ships off");
    }

    #[test]
    fn pv_nodes_and_killers_reduce_less() {
        let p = params();
        let pv = LmrInput {
            pv_node: true,
            ..input()
        };
        assert_eq!(lmr_reduction(&p, pv), 2);
        let killer = LmrInput {
            killer_or_counter: true,
            ..input()
        };
        assert_eq!(lmr_reduction(&p, killer), 2);
        let both = LmrInput {
            pv_node: true,
            killer_or_counter: true,
            ..input()
        };
        assert_eq!(lmr_reduction(&p, both), 1);
    }

    #[test]
    fn history_moves_the_reduction_both_ways() {
        let p = params();
        let good = LmrInput {
            history_score: 6_000,
            ..input()
        };
        let bad = LmrInput {
            history_score: -6_000,
            ..input()
        };
        assert!(lmr_reduction(&p, good) < lmr_reduction(&p, input()));
        assert!(lmr_reduction(&p, bad) > lmr_reduction(&p, input()));
    }

    #[test]
    fn the_reduction_stays_between_zero_and_the_unreduced_depth() {
        let p = params();
        let shallow = LmrInput {
            base: 9,
            new_depth: 2,
            ..input()
        };
        assert_eq!(lmr_reduction(&p, shallow), 2);
        let helped = LmrInput {
            base: 0,
            pv_node: true,
            killer_or_counter: true,
            history_score: 8_000,
            ..input()
        };
        assert_eq!(lmr_reduction(&p, helped), 0);
        let nothing = LmrInput {
            new_depth: 0,
            ..input()
        };
        assert_eq!(lmr_reduction(&p, nothing), 0);
    }

    #[test]
    fn only_late_quiet_moves_without_check_are_candidates() {
        let p = params();
        let (min_depth, min_moves) = (p.lmr_min_depth, p.lmr_min_moves);
        assert!(lmr_applies(&p, min_depth, min_moves + 1, false, false));
        assert!(!lmr_applies(&p, min_depth - 1, min_moves + 1, false, false));
        assert!(!lmr_applies(&p, min_depth, min_moves, false, false));
        assert!(!lmr_applies(&p, min_depth, min_moves + 1, true, false));
        assert!(!lmr_applies(&p, min_depth, min_moves + 1, false, true));
        let mut off = p;
        off.use_lmr_table = false;
        assert!(!lmr_applies(&off, min_depth, min_moves + 1, false, false));
    }

    #[test]
    fn the_re_search_depth_shaves_big_reductions_but_never_below_the_reduced_depth() {
        assert_eq!(compute_re_search_depth(10, 2, true), 10);
        assert_eq!(compute_re_search_depth(10, 3, true), 9);
        assert_eq!(compute_re_search_depth(10, 5, true), 8);
        // Depth 2 or less is only shaved once.
        assert_eq!(compute_re_search_depth(2, 5, true), 1);
        // Never shallower than the search it improves on.
        assert_eq!(compute_re_search_depth(3, 5, true), 1);
        assert_eq!(compute_re_search_depth(10, 5, false), 10);
    }
}
