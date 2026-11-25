//! ProbCut: proving a fail-high with a shallow search.
//!
//! At a node that looks like a fail-high, a capture that survives a
//! quiescence check and a reduced-depth search above `beta + margin` proves
//! the cutoff without the full search. The margin shrinks when the position
//! is improving; the capture must pass a SEE filter set by how far the
//! evaluation is from the ProbCut bound. The verification search carries no
//! continuation context (`previous = None`), and a proven cutoff is stored as
//! a lower bound.
//!
//! The stage is off by default (lever `UseProbCut`). It lives in its own file
//! so that [`crate::search::negamax`] only decides whether to try it.

use crate::board::movegen;
use crate::board::position::Position;
use crate::search::searcher::{MATE_LIMIT, SearchEvent, Searcher};
use crate::search::see::see_ge;
use crate::search::tt::{Bound, TtEntry};
use crate::types::r#move::Move;
use crate::types::piece::{PAWN, piece_type};

/// The state of a node that the ProbCut stage reads.
#[derive(Clone, Copy)]
pub(crate) struct ProbCutNode {
    /// Upper bound of the node's window.
    pub beta: i32,
    /// Remaining depth after the adjustments made so far.
    pub depth: i32,
    /// Distance from the root.
    pub ply: usize,
    /// Whether the node is predicted to fail high.
    pub cut_node: bool,
    /// Whether the side to move is improving.
    pub improving: bool,
    /// Static evaluation used by the pruning decisions.
    pub eval: i32,
    /// Static evaluation as the network produced it (stored with the entry).
    pub raw_eval: i32,
    /// The table move, or [`Move::NONE`].
    pub tt_move: Move,
    /// The table entry of the position, when there is one.
    pub tt_probe: Option<TtEntry>,
    /// Extension budget already spent on the path.
    pub extensions_used: i32,
}

impl Searcher {
    /// Tries ProbCut at a non-PV node that is not in check and not a
    /// singular-verification node (the caller checks those and the switch).
    ///
    /// Returns the score the node returns when a capture proves the cutoff
    /// (or 0 when the search was stopped meanwhile), and `None` when ProbCut
    /// does not apply or proves nothing.
    pub(crate) fn probcut<const STATS: bool>(
        &mut self,
        position: &Position,
        node: ProbCutNode,
        observer: &mut dyn FnMut(SearchEvent),
    ) -> Option<i32> {
        let ProbCutNode {
            beta,
            depth,
            ply,
            cut_node,
            improving,
            eval,
            raw_eval,
            tt_move,
            tt_probe,
            extensions_used,
        } = node;
        if depth < self.params.probcut_min_depth || beta.abs() >= MATE_LIMIT {
            return None;
        }
        let mut probcut_beta = beta + self.params.probcut_margin;
        if improving {
            probcut_beta -= 63;
        }
        if eval >= probcut_beta
            && tt_probe.is_none_or(|entry| Self::probe_score(&entry, ply) >= probcut_beta - 1)
        {
            if STATS {
                self.stats.record_probcut();
            }
            let see_threshold = probcut_beta - eval;
            let probcut_depth = depth - 4;

            // Captures only, the table move first, then most valuable
            // victim by least valuable attacker.
            const MAX_CAPTURES: usize = 96;
            let mut captures = [(0_i32, Move::NONE); MAX_CAPTURES];
            let mut capture_count = 0;
            for &mv in movegen::legal(position).as_slice() {
                if !movegen::is_capture(position, mv) || capture_count == MAX_CAPTURES {
                    continue;
                }
                let victim = if mv.flag() == crate::types::r#move::flags::EN_PASSANT {
                    PAWN
                } else {
                    piece_type(position.piece_at(mv.to()))
                };
                let attacker = piece_type(position.piece_at(mv.from()));
                let order = if mv == tt_move {
                    i32::MAX
                } else {
                    victim as i32 * 16 - attacker as i32
                };
                captures[capture_count] = (order, mv);
                capture_count += 1;
            }
            // Insertion sort, descending: a handful of entries, and stable
            // so equal victims keep their generation order.
            for index in 1..capture_count {
                let item = captures[index];
                let mut slot = index;
                while slot > 0 && captures[slot - 1].0 < item.0 {
                    captures[slot] = captures[slot - 1];
                    slot -= 1;
                }
                captures[slot] = item;
            }

            for &(_, mv) in &captures[..capture_count] {
                if !see_ge(position, mv, see_threshold) {
                    continue;
                }
                let moving = position.piece_at(mv.from());
                let next = position.make_move(mv);
                self.tables.tt.prefetch(next.key());
                self.push_accumulators(position, &next, mv, ply);
                self.move_stack[ply] = mv;
                let mut cut_score = -self.quiescence::<STATS>(
                    &next,
                    -probcut_beta,
                    -probcut_beta + 1,
                    ply + 1,
                    Some((moving, mv.from(), mv.to())),
                );
                if cut_score >= probcut_beta {
                    cut_score = -self.negamax::<STATS>(
                        &next,
                        -probcut_beta,
                        -probcut_beta + 1,
                        probcut_depth,
                        ply + 1,
                        None,
                        !cut_node,
                        Move::NONE,
                        extensions_used,
                        observer,
                    );
                }
                if self.stopped {
                    return Some(0);
                }
                if cut_score >= probcut_beta {
                    if STATS {
                        self.stats.record_probcut_cutoff();
                    }
                    self.store_tt(crate::search::searcher::tt_entry(
                        position,
                        ply,
                        probcut_depth,
                        cut_score,
                        raw_eval,
                        Bound::Lower,
                        mv,
                    ));
                    return Some(cut_score);
                }
            }
        }
        None
    }
}
