//! The two pruning rules that decide from the static evaluation alone: reverse
//! futility and razoring.
//!
//! Both run at non-PV nodes that are not in check and not a singular
//! verification (the caller checks that). Reverse futility lets a node whose
//! evaluation is far above beta fail high at once at low depth and, at greater
//! depth, once a null-window quiescence search around beta agrees; its margins
//! are the pure rule `crate::search::pruning::rfp_verdict`. Razoring drops a
//! node whose evaluation is very low at low depth into quiescence, and a
//! quiescence score below alpha confirms that the node fails low.

use crate::board::position::Position;
use crate::search::history::Previous;
use crate::search::pruning::{RfpVerdict, razor_margin, rfp_verdict};
use crate::search::searcher::Searcher;

/// The state of a node that the evaluation-based pruning reads.
#[derive(Clone, Copy)]
pub(crate) struct EvalPruneNode {
    /// Lower bound of the node's window.
    pub alpha: i32,
    /// Upper bound of the node's window.
    pub beta: i32,
    /// Remaining depth.
    pub depth: i32,
    /// Distance from the root.
    pub ply: usize,
    /// The evaluation the pruning decisions use.
    pub eval: i32,
    /// Whether the side to move is improving.
    pub improving: bool,
    /// The move that produced the node.
    pub previous: Option<Previous>,
}

impl Searcher {
    /// Tries reverse futility and then razoring.
    ///
    /// Returns the score the node returns (0 when the search was stopped
    /// meanwhile), or `None` when neither rule ends the node.
    pub(crate) fn eval_prune<const STATS: bool>(
        &mut self,
        position: &Position,
        node: EvalPruneNode,
    ) -> Option<i32> {
        let EvalPruneNode {
            alpha,
            beta,
            depth,
            ply,
            eval,
            improving,
            previous,
        } = node;
        // Reverse futility (the rule lives in `pruning::rfp_verdict`): a
        // static evaluation far above beta fails high at once at low depth
        // and, at greater depth, once a null-window quiescence search
        // around beta agrees.
        let opponent_worsening = ply >= 1 && eval + self.evals[ply - 1] > 0;
        match rfp_verdict(
            &self.params,
            eval,
            beta,
            depth,
            improving,
            opponent_worsening,
        ) {
            RfpVerdict::Cutoff(score) => {
                if STATS {
                    self.stats.rfp_cutoffs += 1;
                }
                return Some(score);
            }
            RfpVerdict::Verify => {
                let score =
                    self.quiescence_counted::<STATS>(position, beta - 1, beta, ply, previous);
                if self.stopped {
                    return Some(0);
                }
                if score >= beta {
                    if STATS {
                        self.stats.rfp_cutoffs += 1;
                        self.stats.rfp_verified_cutoffs += 1;
                    }
                    return Some(score);
                }
            }
            RfpVerdict::Continue => {}
        }

        // Razoring: very low evaluation at low depth drops into quiescence;
        // a quiescence score below alpha confirms the node fails low.
        if depth <= self.params.razor_max_depth {
            let margin = razor_margin(depth) * self.params.razor_margin_scale / 100;
            if eval + margin <= alpha {
                let score = self.quiescence_counted::<STATS>(position, alpha, beta, ply, previous);
                if self.stopped {
                    return Some(0);
                }
                if score < alpha {
                    if STATS {
                        self.stats.razor_cutoffs += 1;
                    }
                    return Some(alpha);
                }
            }
        }
        None
    }
}
