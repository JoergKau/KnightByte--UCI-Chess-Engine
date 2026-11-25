//! Singular-extension verification.
//!
//! When a deep table move with a proven bound is very likely the only good
//! move, the node is forcing. The verification search runs the alternatives
//! (excluding the table move) at half depth inside a window below the stored
//! score; a wide fail-low extends the node by one, two or three plies, and the
//! extension is applied to the table move only, in the move loop.
//!
//! **Known inert mechanism (quirk Q1).** The
//! verification node probes the transposition table without excluding the
//! entry that triggered it, takes a cutoff on that very entry and therefore
//! never produces an alternative score below the singular beta: the stage
//! searches but never extends. Making it work is an experiment (it lost 18.7
//! Elo when tried on the earlier search); until then it is kept as the plumbing
//! (the excluded move, the verification window, the extension tiers) that a
//! correct implementation needs.
//!
//! The window and the tiers are pure rules in [`crate::search::pruning`].

use crate::board::position::Position;
use crate::search::history::Previous;
use crate::search::pruning::{singular_beta, singular_plies};
use crate::search::searcher::{MATE_LIMIT, SearchEvent, Searcher};
use crate::search::tt::{Bound, TtEntry};
use crate::types::r#move::Move;

/// The state of a node that the singular stage reads.
#[derive(Clone, Copy)]
pub(crate) struct SingularNode {
    /// Remaining depth after the adjustments made so far.
    pub depth: i32,
    /// Distance from the root.
    pub ply: usize,
    /// The move that produced the node (continuation context).
    pub previous: Option<Previous>,
    /// Whether the node is predicted to fail high.
    pub cut_node: bool,
    /// The table move the verification excludes.
    pub tt_move: Move,
    /// The table entry of the position, when there is one.
    pub tt_probe: Option<TtEntry>,
    /// Extension budget already spent on the path.
    pub extensions_used: i32,
}

impl Searcher {
    /// Runs the singular verification at a non-root node with a table move
    /// that is not itself a verification node (the caller checks both and
    /// the switch).
    ///
    /// Returns the plies the table move is extended by (0 when the gate is
    /// closed or the move is not singular), or `None` when the search was
    /// stopped during the verification.
    pub(crate) fn singular_extension<const STATS: bool>(
        &mut self,
        position: &Position,
        node: SingularNode,
        observer: &mut dyn FnMut(SearchEvent),
    ) -> Option<i32> {
        let SingularNode {
            depth,
            ply,
            previous,
            cut_node,
            tt_move,
            tt_probe,
            extensions_used,
        } = node;
        let entry = tt_probe?;
        if depth < self.params.singular_min_depth
            || !matches!(entry.bound, Bound::Exact | Bound::Lower)
            || Self::probe_score(&entry, ply).abs() >= MATE_LIMIT
            || i32::from(entry.depth) < depth - 3
        {
            return Some(0);
        }
        if STATS {
            self.stats.singular_verifications += 1;
        }
        let beta = singular_beta(&self.params, Self::probe_score(&entry, ply), depth);
        let alternative = self.negamax::<STATS>(
            position,
            beta - 1,
            beta,
            depth / 2,
            ply,
            previous,
            cut_node,
            tt_move,
            extensions_used,
            observer,
        );
        if self.stopped {
            return None;
        }
        Some(singular_plies(&self.params, alternative, beta))
    }
}
