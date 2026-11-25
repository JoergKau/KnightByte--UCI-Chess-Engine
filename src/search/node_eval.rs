//! The static evaluation of a node, as the pruning decisions see it.
//!
//! Three adjustments follow the network's raw evaluation, in this order: the
//! correction-history offset, the stored table score when it is consistent with
//! the raw evaluation, and the hindsight depth pair (the previous ply's
//! reduction and the two evaluations decide whether this node earns a ply back
//! or loses one). All of them run before any pruning decision, which therefore
//! sees the adjusted values and depth.

use crate::board::position::Position;
use crate::search::history::Previous;
use crate::search::searcher::Searcher;
use crate::search::tt::TtEntry;
use crate::types::r#move::Move;

/// What the evaluation stage reads about the node.
#[derive(Clone, Copy)]
pub(crate) struct NodeEvalQuery {
    /// Remaining depth.
    pub depth: i32,
    /// Distance from the root.
    pub ply: usize,
    /// Whether the side to move is in check.
    pub in_check: bool,
    /// Whether the node is a principal-variation node.
    pub pv_node: bool,
    /// The move that produced the node.
    pub previous: Option<Previous>,
    /// The table move, or [`Move::NONE`].
    pub tt_move: Move,
    /// The static evaluation stored with the table entry, when there is one.
    pub tt_eval: Option<i32>,
    /// The table entry of the position, when there is one.
    pub tt_probe: Option<TtEntry>,
    /// Whether the position may use the stored score (false for a position
    /// that repeats on the search path).
    pub allow_tt_cutoff: bool,
}

/// The evaluation of a node and the depth after the hindsight adjustment.
#[derive(Clone, Copy)]
pub(crate) struct NodeEval {
    /// The network's evaluation without any correction.
    pub raw_eval: i32,
    /// The evaluation the pruning decisions use.
    pub eval: i32,
    /// Remaining depth after the hindsight pair.
    pub depth: i32,
}

impl Searcher {
    /// Evaluates the node and applies the adjustments (see the module
    /// documentation).
    pub(crate) fn evaluate_node(&mut self, position: &Position, query: NodeEvalQuery) -> NodeEval {
        let NodeEvalQuery {
            mut depth,
            ply,
            in_check,
            pv_node,
            previous,
            tt_move,
            tt_eval,
            tt_probe,
            allow_tt_cutoff,
        } = query;
        let raw_eval = tt_eval.unwrap_or_else(|| self.evaluate_stacked(position, ply));
        let mut eval = raw_eval;

        // Correction history: a learned offset applied before every pruning
        // decision. The reference skips it when a usable TT move exists and
        // when the node is in check, and only corrects nodes with remaining
        // depth.
        if self.params.use_correction_history
            && self.params.use_ref_correction_history
            && depth > 0
            && !in_check
            && tt_move.is_none()
        {
            let correction = self.correction.correction(position, previous);
            eval = self.correction.corrected_eval(raw_eval, correction);
        }

        // A stored score that is consistent with the raw evaluation is more
        // informed than raw + correction, so it replaces the evaluation.
        if self.params.use_tt_score_eval
            && !in_check
            && allow_tt_cutoff
            && let Some(entry) = tt_probe
            && Self::can_use_tt_score(&entry, raw_eval, ply)
        {
            eval = Self::probe_score(&entry, ply);
        }

        // Hindsight depth pair: the previous ply's reduction and the two
        // evals decide whether this node earns the reduction back or loses
        // another ply. Runs before every pruning decision, which therefore
        // sees the adjusted depth.
        if self.params.use_hindsight_reductions && ply > 0 && depth >= 2 && !in_check && !pv_node {
            let previous_reduction = self.reduction_stack[ply - 1];
            let previous_eval = self.evals[ply - 1];
            if previous_reduction >= 2 && eval + previous_eval < 0 {
                depth += 1;
            }
            if previous_reduction >= 1 && eval + previous_eval > self.params.hindsight_margin {
                depth -= 1;
            }
        }
        NodeEval {
            raw_eval,
            eval,
            depth,
        }
    }
}
