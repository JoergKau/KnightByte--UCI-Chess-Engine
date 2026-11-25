//! Null-move pruning.
//!
//! The side to move passes; if the opponent still cannot get below beta in a
//! reduced search, the node is expected to fail high and is cut off at beta.
//! The stage runs at any non-root node that is not in check, including
//! principal-variation nodes and singular-verification nodes: the gate has no
//! node-type or exclusion condition. It carries the cut-node arm: at a
//! predicted cut node the null search runs even when a table score exists
//! below beta. A non-mate fail-high cuts fail-hard at beta; a mate line
//! through a null move is not trusted and the node keeps searching.
//!
//! The reduction is a pure rule in `crate::search::pruning::nmp_reduction`;
//! this file holds the gate and the search around it.

use crate::board::position::Position;
use crate::search::pruning::nmp_reduction;
use crate::search::searcher::{MATE_LIMIT, SearchEvent, Searcher};
use crate::search::tt::TtEntry;
use crate::types::r#move::Move;

/// The state of a node that the null-move stage reads.
#[derive(Clone, Copy)]
pub(crate) struct NullMoveNode {
    /// Upper bound of the node's window.
    pub beta: i32,
    /// Remaining depth after the adjustments made so far.
    pub depth: i32,
    /// Distance from the root.
    pub ply: usize,
    /// Whether the node is predicted to fail high.
    pub cut_node: bool,
    /// Static evaluation used by the pruning decisions.
    pub eval: i32,
    /// The table entry of the position, when there is one.
    pub tt_probe: Option<TtEntry>,
    /// Extension budget already spent on the path.
    pub extensions_used: i32,
}

impl Searcher {
    /// Tries a null move at a non-root node that is not in check (the caller
    /// checks both).
    ///
    /// Returns the score the node returns: `beta` when the null search failed
    /// high, 0 when the search was stopped meanwhile; `None` when the gate is
    /// closed or the null search proves nothing.
    pub(crate) fn null_move_cutoff<const STATS: bool>(
        &mut self,
        position: &Position,
        node: NullMoveNode,
        observer: &mut dyn FnMut(SearchEvent),
    ) -> Option<i32> {
        let NullMoveNode {
            beta,
            depth,
            ply,
            cut_node,
            eval,
            tt_probe,
            extensions_used,
        } = node;
        if depth < self.params.nmp_min_depth
            || eval < beta
            || !(tt_probe.is_none()
                || cut_node
                || tt_probe.is_some_and(|entry| Self::probe_score(&entry, ply) >= beta))
            || !Self::has_non_pawn_material(position, position.side())
        {
            return None;
        }
        if STATS {
            self.stats.record_null_move();
        }
        let reduction = nmp_reduction(&self.params, depth, eval, beta);
        let next = position.make_null();
        self.tables.tt.prefetch(next.key());
        self.copy_accumulators(ply);
        self.move_stack[ply] = Move::NONE;
        let score = -self.negamax::<STATS>(
            &next,
            -beta,
            -beta + 1,
            depth - 1 - reduction,
            ply + 1,
            None,
            !cut_node,
            Move::NONE,
            extensions_used,
            observer,
        );
        if self.stopped {
            return Some(0);
        }
        if score >= beta && score.abs() < MATE_LIMIT {
            if STATS {
                self.stats.record_null_move_cutoff();
            }
            return Some(beta);
        }
        None
    }
}
