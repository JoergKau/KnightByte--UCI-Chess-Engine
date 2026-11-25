//! Searching one move of the move loop: the principal-variation scheme.
//!
//! The first legal move gets the full window. Every later move is scouted with
//! a null window at the reduced depth, as an expected cut node. A scout that
//! beats alpha after a reduction is searched again at the full (slightly
//! shaved) depth, and one that still lands inside the window gets a
//! full-window search as a principal-variation child.

use crate::board::position::Position;
use crate::search::history::Previous;
use crate::search::lmr::compute_re_search_depth;
use crate::search::searcher::{SearchEvent, Searcher};
use crate::types::r#move::Move;

/// The move to search and the state of its parent node.
#[derive(Clone, Copy)]
pub(crate) struct ChildSearch<'a> {
    /// The position after the move.
    pub child: &'a Position,
    /// Lower bound of the parent's window.
    pub alpha: i32,
    /// Upper bound of the parent's window.
    pub beta: i32,
    /// Depth of the child without any reduction.
    pub new_depth: i32,
    /// Plies the late-move reduction takes off the scout.
    pub reduction: i32,
    /// Distance of the parent from the root.
    pub ply: usize,
    /// The move that is searched (continuation context for the child).
    pub previous: Option<Previous>,
    /// Whether the parent is a principal-variation node.
    pub pv_node: bool,
    /// Whether the parent is predicted to fail high.
    pub cut_node: bool,
    /// Whether this is the first legal move of the parent.
    pub first_move: bool,
    /// Extension budget already spent on the path.
    pub extensions_used: i32,
}

impl Searcher {
    /// Searches one move and returns its score from the parent's side. The
    /// caller checks the stop flag afterwards.
    pub(crate) fn search_child<const STATS: bool>(
        &mut self,
        search: ChildSearch,
        observer: &mut dyn FnMut(SearchEvent),
    ) -> i32 {
        let ChildSearch {
            child,
            alpha,
            beta,
            new_depth,
            reduction,
            ply,
            previous,
            pv_node,
            cut_node,
            first_move,
            extensions_used,
        } = search;
        if first_move {
            // First legal move: a principal-variation child at PV nodes,
            // otherwise the opposite node type.
            let child_cut = if pv_node { false } else { !cut_node };
            return -self.negamax::<STATS>(
                child,
                -beta,
                -alpha,
                new_depth,
                ply + 1,
                previous,
                child_cut,
                Move::NONE,
                extensions_used,
                observer,
            );
        }
        // Reduced scout of a late move: treated as an expected cut node.
        let mut score = -self.negamax::<STATS>(
            child,
            -alpha - 1,
            -alpha,
            new_depth - reduction,
            ply + 1,
            previous,
            true,
            Move::NONE,
            extensions_used,
            observer,
        );
        if score > alpha && reduction > 0 {
            if STATS {
                self.stats.record_lmr_re_search(false);
            }
            let re_search_depth =
                compute_re_search_depth(new_depth, reduction, self.params.use_re_search_reduction);
            score = -self.negamax::<STATS>(
                child,
                -alpha - 1,
                -alpha,
                re_search_depth,
                ply + 1,
                previous,
                !cut_node,
                Move::NONE,
                extensions_used,
                observer,
            );
        }
        if score > alpha && score < beta {
            if STATS {
                self.stats.record_lmr_re_search(true);
            }
            // Full-window re-search establishes the principal variation: a PV
            // child, never a cut node.
            score = -self.negamax::<STATS>(
                child,
                -beta,
                -alpha,
                new_depth,
                ply + 1,
                previous,
                false,
                Move::NONE,
                extensions_used,
                observer,
            );
        }
        score
    }
}
