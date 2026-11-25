//! Decision rules of the main search's pruning stack.
//!
//! These are pure functions of the node's state and the tunable parameters:
//! they read nothing from the searcher and change nothing, so every rule can be
//! tested on its own and the node function in [`crate::search::negamax`] only
//! orchestrates them. A rule returns a verdict; acting on it (returning a
//! score, searching a verification, skipping moves) stays with the caller.
//!
//! Constants that no campaign has ever moved live here as named constants with
//! their meaning; the ones worth tuning are fields of
//! [`crate::search::params::SearchParams`] and levers.

use super::params::SearchParams;
use super::searcher::MATE_SCORE;

/// Fixed part of the reverse-futility margin, in centipawns.
const RFP_BASE: i32 = 11;

/// Margin credit when the opponent's last move made its own static evaluation
/// worse (the sum of the two evaluations, seen from the side to move, is
/// positive): a position the opponent has just damaged is likelier to hold.
const RFP_WORSENING_BONUS: i32 = 12;

/// Extra plies of direct reverse-futility pruning when the side to move is
/// improving: a rising evaluation is trusted one search level further.
const RFP_IMPROVING_DEPTH_BONUS: i32 = 2;

/// Share, in percent, of the overshoot `eval - beta` that a direct
/// reverse-futility cutoff returns above beta. Returning only part of it keeps a
/// fail-high from handing the parent an inflated score.
const RFP_RETURN_PERCENT: i32 = 38;

/// A table entry this many plies shallower than the search no longer counts as
/// guidance for move ordering: the node is reduced as if it had no table move.
const IIR_SHALLOW_ENTRY_PLIES: i32 = 4;

/// What reverse futility pruning decides for one node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RfpVerdict {
    /// The node is not pruned; the search goes on.
    Continue,
    /// The node fails high at once with this score.
    Cutoff(i32),
    /// The margin is cleared but the node is too deep to trust the static
    /// evaluation alone: a null-window quiescence search around beta confirms
    /// it, and only a result at or above beta prunes the node.
    Verify,
}

/// Decides reverse futility pruning: a node whose static evaluation clears
/// beta by a depth-scaled margin is expected to fail high.
///
/// The margin is `RFP_BASE + rfp_scale * depth`, lowered by
/// `rfp_improving_bonus` when the side to move is improving and by
/// [`RFP_WORSENING_BONUS`] when the opponent's last move worsened its own
/// evaluation. Up to `rfp_max_depth` plies (two more when improving) a cleared
/// margin prunes directly; for `rfp_verify_band` further plies it asks for a
/// quiescence verification. The caller must exclude PV nodes, nodes in check and
/// singular-verification nodes.
pub(crate) fn rfp_verdict(
    params: &SearchParams,
    eval: i32,
    beta: i32,
    depth: i32,
    improving: bool,
    opponent_worsening: bool,
) -> RfpVerdict {
    let mut margin = RFP_BASE + params.rfp_scale * depth;
    if improving {
        margin -= params.rfp_improving_bonus;
    }
    if opponent_worsening {
        margin -= RFP_WORSENING_BONUS;
    }
    if eval - margin < beta {
        return RfpVerdict::Continue;
    }
    let direct_depth = params.rfp_max_depth
        + if improving {
            RFP_IMPROVING_DEPTH_BONUS
        } else {
            0
        };
    if depth <= direct_depth {
        RfpVerdict::Cutoff(beta + (eval - beta) * RFP_RETURN_PERCENT / 100)
    } else if depth <= direct_depth + params.rfp_verify_band {
        RfpVerdict::Verify
    } else {
        RfpVerdict::Continue
    }
}

/// Decides internal iterative reduction: whether a node is searched one ply
/// shallower because it has no good move to try first.
///
/// `expects_full_search` is true for PV nodes, the root and predicted cut
/// nodes, the nodes whose cost depends on finding the best move early. Such a
/// node is reduced when the table holds no move for it, or (with
/// `use_micro_riders`) when the table entry is more than
/// [`IIR_SHALLOW_ENTRY_PLIES`] plies shallower than this search. A reduced
/// node leaves a move behind in the table for the full-depth visit. The caller
/// must exclude nodes in check and singular-verification nodes.
pub(crate) fn iir_applies(
    params: &SearchParams,
    depth: i32,
    expects_full_search: bool,
    has_tt_move: bool,
    tt_depth: Option<i32>,
) -> bool {
    if !params.use_internal_iterative_reduction
        || depth < params.iir_min_depth
        || !expects_full_search
    {
        return false;
    }
    let shallow_entry = params.use_micro_riders
        && tt_depth.is_some_and(|entry_depth| entry_depth < depth - IIR_SHALLOW_ENTRY_PLIES);
    !has_tt_move || shallow_entry
}

/// Returns the depth reduction of the null-move search: a fixed part, a part
/// that grows with the depth, and a bonus for an evaluation far above beta
/// (capped), clamped so that the null search can never reach the current
/// node's depth.
pub(crate) fn nmp_reduction(params: &SearchParams, depth: i32, eval: i32, beta: i32) -> i32 {
    let eval_bonus = ((eval - beta) / params.nmp_eval_divisor).min(params.nmp_eval_bonus_max);
    let reduction = params.nmp_base_reduction + depth / params.nmp_depth_divisor + eval_bonus;
    reduction.min(depth - 1)
}

/// Returns the beta of the singular verification window: the stored score
/// lowered by `singular_beta_slope` centipawns per ply of remaining depth.
pub(crate) fn singular_beta(params: &SearchParams, tt_score: i32, depth: i32) -> i32 {
    tt_score - params.singular_beta_slope * depth
}

/// Returns how many plies the table move is extended by, given the score
/// `alternative` of the verification search (the alternatives to the table
/// move, searched below `beta`): one ply when they all fail low, and with
/// double extensions on a second and a third ply as the fail-low margin
/// reaches `singular_double_margin` and `singular_triple_margin`.
pub(crate) fn singular_plies(params: &SearchParams, alternative: i32, beta: i32) -> i32 {
    if alternative >= beta {
        return 0;
    }
    if !params.use_singular_double_extensions {
        return 1;
    }
    if alternative < beta - params.singular_triple_margin {
        3
    } else if alternative < beta - params.singular_double_margin {
        2
    } else {
        1
    }
}

/// Returns the window `(alpha, beta)` narrowed by mate-distance pruning at
/// `ply`: no score below being mated at this ply, none above mating at the next.
pub(crate) fn mate_distance_window(alpha: i32, beta: i32, ply: usize) -> (i32, i32) {
    let mate_value = MATE_SCORE - ply as i32;
    (alpha.max(-mate_value), beta.min(mate_value - 1))
}

/// Returns the futility margin for a remaining depth, capped at depth 4.
pub(crate) fn futility_margin(depth: i32) -> i32 {
    const MARGINS: [i32; 5] = [0, 61, 144, 268, 334];
    MARGINS[(depth as usize).min(4)]
}

/// Returns the razoring margin for a remaining depth.
pub(crate) fn razor_margin(depth: i32) -> i32 {
    if depth <= 1 { 86 } else { 353 }
}

/// Returns the number of quiet moves searched before late move pruning stops
/// the quiet stage. Indexed by remaining depth, capped at depth 4.
pub(crate) fn lmp_limit(depth: i32) -> i32 {
    const LIMITS: [i32; 5] = [0, 9, 12, 17, 24];
    LIMITS[(depth as usize).min(4)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SearchParams {
        SearchParams::default()
    }

    #[test]
    fn a_node_below_the_margin_is_not_pruned() {
        // Depth 1, not improving: margin 11 + 62 = 73.
        assert_eq!(
            rfp_verdict(&params(), 172, 100, 1, false, false),
            RfpVerdict::Continue
        );
        assert!(matches!(
            rfp_verdict(&params(), 173, 100, 1, false, false),
            RfpVerdict::Cutoff(_)
        ));
    }

    #[test]
    fn improving_and_worsening_lower_the_margin() {
        let p = params();
        // Depth 2: margin 11 + 124 = 135, minus 56 improving, minus 12 worsening.
        let beta = 0;
        let cut = |eval, improving, worsening| {
            matches!(
                rfp_verdict(&p, eval, beta, 2, improving, worsening),
                RfpVerdict::Cutoff(_)
            )
        };
        assert!(!cut(134, false, false));
        assert!(cut(135, false, false));
        assert!(!cut(78, true, false));
        assert!(cut(79, true, false));
        assert!(!cut(122, false, true));
        assert!(cut(123, false, true));
        assert!(!cut(66, true, true));
        assert!(cut(67, true, true));
    }

    #[test]
    fn a_direct_cutoff_returns_a_share_of_the_overshoot() {
        // Overshoot 1000 above beta 50: 38 percent of it is returned above beta.
        assert_eq!(
            rfp_verdict(&params(), 1050, 50, 1, false, false),
            RfpVerdict::Cutoff(50 + 380)
        );
    }

    #[test]
    fn depth_decides_between_cutoff_verification_and_nothing() {
        let p = params();
        let far = 100_000; // clears every margin
        let verdict = |depth, improving| rfp_verdict(&p, far, 0, depth, improving, false);
        // Not improving: direct up to depth 8, verified up to 16.
        assert!(matches!(verdict(8, false), RfpVerdict::Cutoff(_)));
        assert_eq!(verdict(9, false), RfpVerdict::Verify);
        assert_eq!(verdict(16, false), RfpVerdict::Verify);
        assert_eq!(verdict(17, false), RfpVerdict::Continue);
        // Improving: two plies further on both edges.
        assert!(matches!(verdict(10, true), RfpVerdict::Cutoff(_)));
        assert_eq!(verdict(11, true), RfpVerdict::Verify);
        assert_eq!(verdict(18, true), RfpVerdict::Verify);
        assert_eq!(verdict(19, true), RfpVerdict::Continue);
    }

    #[test]
    fn a_zero_verification_band_leaves_only_the_direct_cutoff() {
        let mut p = params();
        p.rfp_verify_band = 0;
        assert!(matches!(
            rfp_verdict(&p, 100_000, 0, 8, false, false),
            RfpVerdict::Cutoff(_)
        ));
        assert_eq!(
            rfp_verdict(&p, 100_000, 0, 9, false, false),
            RfpVerdict::Continue
        );
    }

    #[test]
    fn iir_reduces_a_node_that_expects_a_full_search_and_has_no_table_move() {
        let p = params();
        assert!(iir_applies(&p, 6, true, false, None));
        // A cut node without a table move is reduced too (the B2 change).
        assert!(iir_applies(&p, 6, true, false, Some(1)));
        // A node that does not expect a full search is never reduced.
        assert!(!iir_applies(&p, 6, false, false, None));
    }

    #[test]
    fn iir_keeps_a_node_with_a_usable_table_move() {
        let p = params();
        // A deep enough entry with a move: ordering is already guided.
        assert!(!iir_applies(&p, 8, true, true, Some(8)));
        assert!(!iir_applies(&p, 8, true, true, Some(4)));
        // More than four plies shallower: reduced even though it has a move.
        assert!(iir_applies(&p, 8, true, true, Some(3)));
    }

    #[test]
    fn iir_respects_its_depth_floor_and_its_switches() {
        let mut p = params();
        assert!(!iir_applies(&p, p.iir_min_depth - 1, true, false, None));
        assert!(iir_applies(&p, p.iir_min_depth, true, false, None));
        p.use_micro_riders = false;
        assert!(
            !iir_applies(&p, 8, true, true, Some(0)),
            "no shallow-entry gate"
        );
        assert!(iir_applies(&p, 8, true, false, None));
        p.use_internal_iterative_reduction = false;
        assert!(!iir_applies(&p, 8, true, false, None));
    }

    #[test]
    fn the_null_move_reduction_grows_with_depth_and_with_the_evaluation_surplus() {
        let p = params();
        // Default: 3 + depth / 4 + min((eval - beta) / 191, 4).
        assert_eq!(nmp_reduction(&p, 12, 100, 100), 3 + 3);
        assert_eq!(nmp_reduction(&p, 12, 100 + 191, 100), 3 + 3 + 1);
        assert_eq!(nmp_reduction(&p, 12, 100 + 10_000, 100), 3 + 3 + 4);
        assert!(nmp_reduction(&p, 16, 100, 100) > nmp_reduction(&p, 8, 100, 100));
    }

    #[test]
    fn the_null_move_reduction_never_reaches_the_nodes_depth() {
        let p = params();
        for depth in 1..=20 {
            assert!(
                nmp_reduction(&p, depth, 100_000, 0) < depth,
                "depth {depth}"
            );
        }
        assert_eq!(nmp_reduction(&p, 3, 0, 0), 2);
    }

    #[test]
    fn the_singular_window_sits_below_the_stored_score() {
        let p = params();
        // Slope 2 per ply: depth 10 lowers the window by 20.
        assert_eq!(singular_beta(&p, 100, 10), 80);
        assert_eq!(singular_beta(&p, -50, 8), -66);
    }

    #[test]
    fn the_singular_tiers_follow_the_fail_low_margin() {
        let p = params();
        let beta = 0;
        assert_eq!(singular_plies(&p, 0, beta), 0, "no fail-low: not singular");
        assert_eq!(singular_plies(&p, 10, beta), 0);
        assert_eq!(singular_plies(&p, -1, beta), 1);
        assert_eq!(
            singular_plies(&p, -30, beta),
            1,
            "margin 30 is not below 30"
        );
        assert_eq!(singular_plies(&p, -31, beta), 2);
        assert_eq!(singular_plies(&p, -120, beta), 2);
        assert_eq!(singular_plies(&p, -121, beta), 3);
        let mut single = p;
        single.use_singular_double_extensions = false;
        assert_eq!(singular_plies(&single, -500, beta), 1);
    }

    #[test]
    fn mate_distance_pruning_narrows_the_window_to_the_reachable_scores() {
        // Far from mate the window is untouched.
        assert_eq!(mate_distance_window(-100, 100, 5), (-100, 100));
        // At ply 5 nothing is worse than being mated now, nothing better than
        // mating on the next move.
        let mate = MATE_SCORE - 5;
        assert_eq!(
            mate_distance_window(-MATE_SCORE, MATE_SCORE, 5),
            (-mate, mate - 1)
        );
        // A window that lies beyond the reachable scores collapses.
        let (alpha, beta) = mate_distance_window(mate + 10, mate + 20, 5);
        assert!(alpha >= beta);
    }

    #[test]
    fn the_depth_tables_are_capped_and_ordered() {
        assert_eq!(futility_margin(0), 0);
        assert_eq!(futility_margin(7), futility_margin(4));
        assert!(futility_margin(1) < futility_margin(2));
        assert_eq!(razor_margin(1), 86);
        assert_eq!(razor_margin(3), 353);
        assert_eq!(lmp_limit(0), 0);
        assert_eq!(lmp_limit(9), lmp_limit(4));
        assert!(lmp_limit(1) < lmp_limit(2));
    }
}
