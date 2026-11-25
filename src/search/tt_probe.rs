//! The transposition-table probe at the top of a node.
//!
//! The probe reads the entry of the position and does four things with it: it
//! hands the stored move and static evaluation to the rest of the node, grants
//! the table extension (off by default, lever `UseTtExtension`), takes a cutoff
//! when the entry is deep enough and its bound settles the window, and
//! otherwise narrows the window with the bound. A position that repeats on the
//! search path is refused cutoffs and extensions by the caller
//! (`allow_cutoff`): a cached score knows nothing about the repetition the
//! search is trying to detect.

use crate::board::movegen;
use crate::board::position::Position;
use crate::search::searcher::{NO_STATIC_EVAL, Searcher};
use crate::search::tt::{Bound, TtEntry};
use crate::types::r#move::Move;
use crate::types::piece::color_of;

/// What the probe needs to know about the node.
#[derive(Clone, Copy)]
pub(crate) struct TtQuery {
    /// Lower bound of the node's window.
    pub alpha: i32,
    /// Upper bound of the node's window.
    pub beta: i32,
    /// Remaining depth.
    pub depth: i32,
    /// Distance from the root.
    pub ply: usize,
    /// Whether the node is a principal-variation node.
    pub pv_node: bool,
    /// Whether the node is predicted to fail high.
    pub cut_node: bool,
    /// Whether the node is the root, which never takes a cutoff.
    pub is_root: bool,
    /// Whether the position may take a cutoff or an extension (false for a
    /// position that repeats on the search path).
    pub allow_cutoff: bool,
}

/// What the probe leaves the node with when it does not end it.
#[derive(Clone, Copy)]
pub(crate) struct TtNode {
    /// The window's lower bound after narrowing.
    pub alpha: i32,
    /// The window's upper bound after narrowing.
    pub beta: i32,
    /// The remaining depth after a table extension.
    pub depth: i32,
    /// The stored move, or [`Move::NONE`].
    pub tt_move: Move,
    /// The entry of the position, when there is one.
    pub entry: Option<TtEntry>,
    /// The stored static evaluation, when the entry holds one.
    pub eval: Option<i32>,
}

/// The verdict of the probe.
pub(crate) enum TtStep {
    /// The entry settles the node; return this score.
    Return(i32),
    /// The node goes on with this state.
    Continue(TtNode),
}

impl Searcher {
    /// Probes the table for `position` and applies the entry (see the module
    /// documentation).
    pub(crate) fn probe_table<const STATS: bool>(
        &mut self,
        position: &Position,
        query: TtQuery,
    ) -> TtStep {
        let TtQuery {
            mut alpha,
            mut beta,
            mut depth,
            ply,
            pv_node,
            cut_node,
            is_root,
            allow_cutoff: allow_tt_cutoff,
        } = query;
        let mut tt_move = Move::NONE;
        let mut tt_probe = None;
        let mut tt_eval = None;
        let tt_found = self.tables.tt.probe(position.key());
        if STATS {
            self.stats.record_tt_probe(tt_found.is_some());
        }
        if let Some(entry) = tt_found {
            // Bug B35 fix (reference bug 147): the table move is used as stored,
            // markers included. KnightByte's table packing keeps the full move,
            // so a stored castling or en-passant move compares equal to the
            // generated move it came from and is tried first, excluded in the
            // singular-verification search and eligible for a singular
            // extension. The reference strips these markers only because its
            // 16-bit packing cannot hold them; the fixed reference rebuilds
            // them from board context at probe time, and so does the stored
            // form here.
            tt_move = entry.mv;
            tt_probe = Some(entry);
            let stored_eval = i32::from(entry.eval);
            if stored_eval != NO_STATIC_EVAL {
                tt_eval = Some(stored_eval);
            }
            // Table extension (off by default, lever `UseTtExtension`): an entry
            // that is deep enough for a cutoff and looks like a plausible one
            // (`score <= alpha`, or any score at a predicted cut node), yet lies
            // OUTSIDE the window (a lower bound below beta, an upper bound above
            // alpha), says the position is more interesting than this shallow
            // window suggests: the node earns one extra ply. Exact entries are
            // excluded (they simply cut off below), and the check runs before
            // the cutoff block so alpha and beta are still the un-narrowed
            // window. A repeated position never has its depth anchored to a
            // cached entry (`allow_tt_cutoff`).
            if self.params.use_tt_extension
                && !is_root
                && allow_tt_cutoff
                && entry.bound != Bound::Exact
                && i32::from(entry.depth) >= depth + if pv_node { 2 } else { 0 }
                && (Self::probe_score(&entry, ply) <= alpha || cut_node)
                && depth <= self.params.tt_extension_max_depth
            {
                let stored = Self::probe_score(&entry, ply);
                let outside = if entry.bound == Bound::Lower {
                    stored < beta
                } else {
                    stored > alpha
                };
                if outside {
                    depth += 1;
                    if STATS {
                        self.stats.tt_extensions += 1;
                    }
                }
            }

            // A verification node would want to skip the transposition-table
            // cutoff: its entry is the one that triggered the verification, so
            // returning on it makes the verification meaningless. The reference
            // does not skip it (the verification always returns on its own
            // trigger entry, so a singular extension never fires - quirk Q1,
            // reproduced for parity), and neither does
            // KnightByte. The former `SingularExcludeTtCutoff` research lever
            // was removed after it measured no gain.
            if !is_root && allow_tt_cutoff && i32::from(entry.depth) >= depth {
                let score = Self::probe_score(&entry, ply);
                match entry.bound {
                    Bound::Exact => {
                        if STATS {
                            self.stats.tt_cutoffs += 1;
                        }
                        return TtStep::Return(score);
                    }
                    Bound::Lower => {
                        // Early transposition-table-cutoff history update:
                        // a quiet table move found at a non-PV node, whose
                        // parent found its own move quickly, earns a
                        // butterfly bonus. The reference awards it for every
                        // sufficiently deep lower-bound entry, before it
                        // decides whether the stored score reaches beta.
                        if !tt_move.is_none()
                            && !pv_node
                            && ply > 0
                            && self.move_counts[ply - 1] < 4
                            && !movegen::is_capture(position, tt_move)
                            && !tt_move.is_promotion()
                        {
                            let bonus = self.history.butterfly_bonus(depth);
                            self.history.update_butterfly_bonus(
                                color_of(position.piece_at(tt_move.from())),
                                tt_move,
                                bonus,
                            );
                        }
                        if score >= beta {
                            if STATS {
                                self.stats.tt_cutoffs += 1;
                            }
                            return TtStep::Return(score);
                        }
                        alpha = alpha.max(score);
                    }
                    Bound::Upper => {
                        if score <= alpha {
                            if STATS {
                                self.stats.tt_cutoffs += 1;
                            }
                            return TtStep::Return(score);
                        }
                        beta = beta.min(score);
                    }
                }
                if alpha >= beta {
                    if STATS {
                        self.stats.tt_cutoffs += 1;
                    }
                    return TtStep::Return(score);
                }
            }
        }
        TtStep::Continue(TtNode {
            alpha,
            beta,
            depth,
            tt_move,
            entry: tt_probe,
            eval: tt_eval,
        })
    }
}
