//! The per-move rules of the move loop.
//!
//! Before a move is made, a handful of rules may drop it or end the loop:
//! depth-scaled SEE pruning, late-move pruning, history pruning, futility and
//! bad-noisy pruning. After it is made, the negative extension may take a ply
//! off a suspicious move. When a move fails high, `Searcher::reward_cutoff`
//! credits it in the history tables.
//!
//! The rules that read only numbers (`quiet_prune`) are pure functions with
//! unit tests; the ones that need the board (SEE) are methods that apply the
//! verdict and count it.

use crate::board::position::Position;
use crate::search::history::{Continuations, Previous};
use crate::search::params::SearchParams;
use crate::search::pruning::lmp_limit;
use crate::search::searcher::Searcher;
use crate::search::see::see_ge;
use crate::types::r#move::{Move, flags};
use crate::types::piece::{KING, PAWN, Piece, color_of, piece_type};

/// Moves the SEE pruning rule gives the flat legacy thresholds for.
const LEGACY_SEE_CEILING: i32 = 4;
/// Legacy SEE threshold of a capture.
const LEGACY_SEE_CAPTURE: i32 = -50;
/// Legacy SEE threshold of a quiet move.
const LEGACY_SEE_QUIET: i32 = -100;

/// The facts about a node and one of its moves that the rules read.
#[derive(Clone, Copy)]
pub(crate) struct MoveContext {
    /// Remaining depth of the node.
    pub depth: i32,
    /// Whether the node is in check.
    pub in_check: bool,
    /// Whether the node is a principal-variation node.
    pub pv_node: bool,
    /// Whether the move is a capture (en passant included).
    pub capture: bool,
    /// The piece that moves.
    pub moving: Piece,
    /// The combined history score of the move.
    pub history_score: i32,
    /// Distance of the node from the root.
    pub ply: usize,
    /// The move that produced the node.
    pub previous: Option<Previous>,
}

/// What the pruning rules decided about a quiet move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuietPrune {
    /// No rule applies; the move is searched.
    Keep,
    /// Enough quiets were tried: skip this move and every later quiet.
    LateMoves,
    /// The move's history is far below the threshold: skip this move only.
    History,
    /// The evaluation is far below alpha: skip this move and every later
    /// quiet.
    Futility,
}

/// What the quiet-move rules read about the move and the node.
#[derive(Clone, Copy)]
pub(crate) struct QuietFacts {
    /// Remaining depth of the node.
    pub depth: i32,
    /// Whether the node is in check.
    pub in_check: bool,
    /// Whether the node is a principal-variation node.
    pub pv_node: bool,
    /// Whether the side to move is improving.
    pub improving: bool,
    /// Legal moves seen so far, this one included.
    pub legal_moves: i32,
    /// Quiet moves seen so far, this one included.
    pub quiet_move_count: i32,
    /// The combined history score of the move.
    pub history_score: i32,
    /// Whether the moving piece is a pawn.
    pub pawn_move: bool,
    /// Whether the node qualifies for futility pruning at all: not PV, not in
    /// check, shallow, and the evaluation plus the margin below alpha.
    pub futile: bool,
}

/// Decides about a quiet (neither capture nor promotion) move.
///
/// The rules are tried in a fixed order: late-move pruning, history pruning,
/// futility. The first one that applies wins; none applies to the first legal
/// move, in check, or at a PV node.
pub(crate) fn quiet_prune(params: &SearchParams, facts: QuietFacts) -> QuietPrune {
    let QuietFacts {
        depth,
        in_check,
        pv_node,
        improving,
        legal_moves,
        quiet_move_count,
        history_score,
        pawn_move,
        futile,
    } = facts;
    // Late move pruning: enough quiets have been tried that the rest cannot
    // realistically be best.
    if !in_check && !pv_node && legal_moves > 1 && depth <= params.lmp_max_depth {
        let mut limit = (lmp_limit(depth) * params.lmp_scale / 100).max(1);
        if params.use_improving_flag && !improving {
            limit = (limit / 2).max(1);
        }
        if quiet_move_count > limit {
            return QuietPrune::LateMoves;
        }
    }
    // History pruning: a quiet whose combined history is far below the
    // depth-scaled threshold is skipped; the remaining quiets are still
    // searched, one move at a time.
    if params.use_history_pruning
        && depth <= params.hist_prune_max_depth
        && !pv_node
        && !in_check
        && legal_moves > 1
        && history_score < -params.hist_prune_margin * depth
    {
        return QuietPrune::History;
    }
    // Futility: the static evaluation is so far below alpha that even the
    // margin cannot bridge the gap; quiet non-pawn moves cannot realistically
    // improve on alpha. Pawn moves are exempt.
    if !in_check && !pv_node && legal_moves > 1 && futile && !pawn_move {
        return QuietPrune::Futility;
    }
    QuietPrune::Keep
}

impl Searcher {
    /// Returns whether depth-scaled SEE pruning drops the move.
    ///
    /// The threshold scales with depth and is refunded by the move's history;
    /// the first legal move of the node is never pruned (`legal_moves_before`
    /// counts the moves before this one), and king moves are exempt on the
    /// quiet side (the exchange says little about king safety). The flat
    /// legacy thresholds remain available behind `UseRefSeeThresholds`.
    pub(crate) fn see_prunes<const STATS: bool>(
        &mut self,
        position: &Position,
        mv: Move,
        context: MoveContext,
        legal_moves_before: i32,
    ) -> bool {
        let MoveContext {
            depth,
            in_check,
            pv_node,
            capture,
            moving,
            history_score,
            ..
        } = context;
        let params = &self.params;
        let ceiling = if params.use_ref_see_thresholds {
            params.see_pruning_max_depth
        } else {
            params.see_pruning_max_depth.min(LEGACY_SEE_CEILING)
        };
        if depth > ceiling || in_check || pv_node || legal_moves_before == 0 {
            return false;
        }
        let threshold = if capture {
            if params.use_ref_see_thresholds {
                params.see_noisy_margin * depth * depth - history_score / params.see_history_divisor
            } else {
                LEGACY_SEE_CAPTURE
            }
        } else if !mv.is_promotion() && piece_type(moving) != KING {
            if params.use_ref_see_thresholds {
                params.see_quiet_margin * depth - history_score / params.see_history_divisor
            } else {
                LEGACY_SEE_QUIET
            }
        } else {
            return false;
        };
        if see_ge(position, mv, threshold) {
            return false;
        }
        if STATS {
            self.stats.see_skips += 1;
        }
        true
    }

    /// Returns the evaluation floor when bad-noisy pruning ends the move loop
    /// at this (losing) capture, or `None` when it does not apply.
    ///
    /// Off by default (`UseBadNoisyPruning`). When the evaluation plus a margin
    /// says even a winning capture cannot lift the node to alpha, the first
    /// losing capture ends the loop, which kills every remaining capture at
    /// once: the staged picker serves losing captures last, so the first one
    /// met starts that stage. The margin becomes the node's fail-soft score
    /// floor so the table still stores a meaningful upper bound.
    pub(crate) fn bad_noisy_stop(
        &self,
        position: &Position,
        mv: Move,
        context: MoveContext,
        eval: i32,
        alpha: i32,
        legal_moves: i32,
    ) -> Option<i32> {
        if !self.params.use_bad_noisy_pruning
            || context.in_check
            || context.depth >= self.params.bnp_depth
            || !context.capture
            || see_ge(position, mv, 0)
        {
            return None;
        }
        let margin = eval + 113 * context.depth + 383 * legal_moves / 133;
        (margin <= alpha).then_some(margin)
    }

    /// Returns the plies a suspicious move loses: one for a losing capture or
    /// for a quiet the history tables distrust, zero otherwise.
    ///
    /// Off by default (`UseNegativeExtensions`). It applies at depth 3 and
    /// more, at non-PV nodes, when neither the node nor the move gives check.
    pub(crate) fn negative_extension<const STATS: bool>(
        &mut self,
        position: &Position,
        mv: Move,
        context: MoveContext,
        gives_check: bool,
    ) -> i32 {
        if !self.params.use_negative_extensions
            || context.depth < 3
            || context.pv_node
            || context.in_check
            || gives_check
        {
            return 0;
        }
        let suspicious = if context.capture {
            !see_ge(position, mv, 0)
        } else {
            !mv.is_promotion()
                && self.history.butterfly(color_of(context.moving), mv)
                    < -self.params.neg_ext_history_threshold
        };
        if !suspicious {
            return 0;
        }
        if STATS {
            self.stats.negative_extensions += 1;
        }
        -1
    }

    /// Credits the move that failed high in the history tables.
    ///
    /// The branch is chosen by "not a capture", so a quiet promotion credits
    /// the quiet-move tables even though it is excluded from the failed-quiet
    /// list. A quiet move becomes a killer and the countermove of the previous
    /// move and is punished-against the quiets that failed before it; a capture
    /// is credited in the capture table.
    pub(crate) fn reward_cutoff(
        &mut self,
        position: &Position,
        mv: Move,
        context: MoveContext,
        failed_quiets: &[Move],
        continuations: &Continuations,
    ) {
        let MoveContext {
            depth,
            capture,
            moving,
            ply,
            previous,
            ..
        } = context;
        if !capture {
            self.killers.record(ply, mv);
            self.history
                .update_quiet(color_of(moving), mv, depth, failed_quiets, continuations);
            self.history.update_counter(previous, mv);
        } else {
            self.history
                .update_capture(piece_type(moving), mv, victim_type(position, mv), depth);
        }
    }

    /// Returns the combined history score of a move, computed before it is
    /// made: a capture reads the capture table, a quiet the butterfly and
    /// continuation tables.
    pub(crate) fn move_history_score(
        &self,
        position: &Position,
        mv: Move,
        capture: bool,
        moving: Piece,
        continuations: &Continuations,
    ) -> i32 {
        if capture {
            self.history
                .capture(piece_type(moving), mv, victim_type(position, mv))
                .clamp(-8_033, 8_033)
        } else {
            self.history
                .pruning_score(color_of(moving), mv, continuations)
        }
    }
}

/// Returns the piece type a capture takes (a pawn for en passant, whose
/// target square is empty).
fn victim_type(position: &Position, mv: Move) -> usize {
    if mv.flag() == flags::EN_PASSANT {
        PAWN
    } else {
        piece_type(position.piece_at(mv.to()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> QuietFacts {
        QuietFacts {
            depth: 4,
            in_check: false,
            pv_node: false,
            improving: true,
            legal_moves: 10,
            quiet_move_count: 3,
            history_score: 0,
            pawn_move: false,
            futile: false,
        }
    }

    fn verdict(facts: QuietFacts) -> QuietPrune {
        quiet_prune(&SearchParams::default(), facts)
    }

    #[test]
    fn an_ordinary_quiet_move_is_kept() {
        assert_eq!(verdict(facts()), QuietPrune::Keep);
    }

    #[test]
    fn late_move_pruning_needs_more_quiets_than_the_depth_allows() {
        let p = SearchParams::default();
        let limit = (lmp_limit(4) * p.lmp_scale / 100).max(1);
        let at_limit = QuietFacts {
            quiet_move_count: limit,
            ..facts()
        };
        assert_eq!(verdict(at_limit), QuietPrune::Keep);
        let beyond = QuietFacts {
            quiet_move_count: limit + 1,
            ..facts()
        };
        assert_eq!(verdict(beyond), QuietPrune::LateMoves);
    }

    #[test]
    fn a_node_that_is_not_improving_halves_the_late_move_limit() {
        let p = SearchParams::default();
        assert!(p.use_improving_flag, "the test assumes the shipped default");
        let limit = (lmp_limit(4) * p.lmp_scale / 100).max(1);
        let quiets = QuietFacts {
            quiet_move_count: limit / 2 + 1,
            ..facts()
        };
        assert_eq!(verdict(quiets), QuietPrune::Keep);
        let worsening = QuietFacts {
            improving: false,
            ..quiets
        };
        assert_eq!(verdict(worsening), QuietPrune::LateMoves);
    }

    #[test]
    fn history_pruning_skips_a_move_with_a_very_bad_history() {
        let p = SearchParams::default();
        assert!(
            p.use_history_pruning,
            "the test assumes the shipped default"
        );
        let depth = p.hist_prune_max_depth;
        let bad = QuietFacts {
            depth,
            history_score: -p.hist_prune_margin * depth - 1,
            ..facts()
        };
        assert_eq!(verdict(bad), QuietPrune::History);
        let borderline = QuietFacts {
            depth,
            history_score: -p.hist_prune_margin * depth,
            ..facts()
        };
        assert_eq!(verdict(borderline), QuietPrune::Keep);
    }

    #[test]
    fn futility_skips_quiet_moves_but_not_pawn_moves() {
        let futile = QuietFacts {
            futile: true,
            ..facts()
        };
        assert_eq!(verdict(futile), QuietPrune::Futility);
        let pawn = QuietFacts {
            pawn_move: true,
            ..futile
        };
        assert_eq!(verdict(pawn), QuietPrune::Keep);
    }

    #[test]
    fn no_rule_touches_the_first_move_a_check_or_a_pv_node() {
        let all = QuietFacts {
            quiet_move_count: 100,
            history_score: -1_000_000,
            futile: true,
            ..facts()
        };
        assert_ne!(verdict(all), QuietPrune::Keep);
        assert_eq!(
            verdict(QuietFacts {
                legal_moves: 1,
                ..all
            }),
            QuietPrune::Keep
        );
        assert_eq!(
            verdict(QuietFacts {
                in_check: true,
                ..all
            }),
            QuietPrune::Keep
        );
        assert_eq!(
            verdict(QuietFacts {
                pv_node: true,
                ..all
            }),
            QuietPrune::Keep
        );
    }

    #[test]
    fn the_rules_are_tried_in_a_fixed_order() {
        let both = QuietFacts {
            quiet_move_count: 100,
            history_score: -1_000_000,
            futile: true,
            ..facts()
        };
        assert_eq!(verdict(both), QuietPrune::LateMoves);
        let history_and_futility = QuietFacts {
            depth: SearchParams::default().hist_prune_max_depth,
            quiet_move_count: 1,
            ..both
        };
        assert_eq!(verdict(history_and_futility), QuietPrune::History);
    }
}
