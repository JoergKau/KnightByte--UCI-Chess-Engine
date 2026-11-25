//! Iterative deepening: the entry point of a search, the loop over depths with
//! aspiration windows, and the root move loop.

use super::node::mated_in;
use super::{
    DRAW_SCORE, INFINITE, MATE_LIMIT, MATE_SCORE, SearchEvent, SearchInfo, SearchResult, Searcher,
};
use crate::board::position::Position;
use crate::search::limits::Limits;
use crate::types::r#move::Move;

impl Searcher {
    /// Runs iterative deepening and returns the best move.
    ///
    /// `observer` is called after every completed iteration; pass a closure
    /// that ignores its argument when no reporting is needed. When the
    /// `searchstats` diagnostics are on, the searcher's counters are merged
    /// into the process-wide snapshot as the search ends.
    pub fn search(
        &mut self,
        position: &Position,
        limits: Limits,
        observer: &mut dyn FnMut(SearchEvent),
    ) -> SearchResult {
        if self.lmr_table_divisor != self.params.lmr_divisor {
            self.lmr_table = crate::search::lmr::build_lmr_table(self.params.lmr_divisor);
            self.lmr_table_divisor = self.params.lmr_divisor;
        }
        self.collect_stats = crate::tools::diagnostics::search_stats_enabled();
        if self.collect_stats {
            self.stats = crate::tools::diagnostics::SearchStats::new();
        }
        let result = self.search_inner(position, limits, observer);
        if self.collect_stats {
            self.stats.nodes = self.nodes;
            crate::tools::diagnostics::merge_into_last_search(&self.stats);
            self.collect_stats = false;
        }
        result
    }

    /// The iterative-deepening driver behind [`Self::search`].
    pub(super) fn search_inner(
        &mut self,
        position: &Position,
        limits: Limits,
        observer: &mut dyn FnMut(SearchEvent),
    ) -> SearchResult {
        // The search clock is started by the `init_*` call that configured the
        // time manager; it is deliberately not restarted here.
        self.limits = limits;
        self.completed_depth = 0;
        self.nodes = 0;
        self.published_nodes = 0;
        self.seldepth = 0;
        self.stopped = false;
        self.currmove_reported = false;
        self.pv.reset();
        self.killers.clear();
        // The tablebase set is looked up once per search: a `SyzygyPath`
        // reload only happens while no search runs (the UCI loop joins first).
        self.tb = crate::tb::syzygy::current();
        self.tb_probing = crate::tb::syzygy::probing_enabled();
        self.root_tb_move = Move::NONE;
        self.root_tb_wdl = None;
        self.tb_hits = 0;
        self.published_tb_hits = 0;
        if self.helper_nodes.is_none() {
            // No helpers: this searcher owns the whole count. With helpers the
            // UCI layer cleared the shared total before launching them.
            crate::search::tb::reset_helper_tb_hits();
        }
        // The learned tables start cold every search, as in the predecessor:
        // its UCI layer builds a fresh per-thread state object for each `go`,
        // so history and correction history never survive a search (B41).
        // Only the transposition table persists.
        self.history.clear();
        self.history.set_scales(
            self.params.history_bonus_scale,
            self.params.history_malus_scale,
        );
        self.correction.clear();
        self.reset_stacks();
        *self.nodes_per_move = [0; 64 * 64];
        // The predecessor's main thread advances the TT generation at the
        // start of every iterative-deepening search (NewSearch + _currentAge).
        // Keep that: the age blending in the replacement policy depends on it.
        self.tables.tt.new_generation();

        let root_moves = crate::board::movegen::legal(position);
        if root_moves.is_empty() {
            let score = if position.in_check(position.side()) {
                mated_in(0)
            } else {
                DRAW_SCORE
            };
            return SearchResult {
                best_move: Move::NONE,
                score,
                depth: 0,
                seldepth: 0,
                nodes: 0,
                pv: Vec::new(),
            };
        }

        // Step 6 (`go searchmoves`): the root is restricted to the listed
        // moves. The filter holds the listed moves that are legal here; an
        // empty list — none given, or every entry illegal — searches
        // everything (the reference ignores an illegal-only list, verified
        // 2026-11-18).
        self.root_filter = self
            .limits
            .root_moves
            .iter()
            .filter(|mv| root_moves.as_slice().contains(mv))
            .copied()
            .collect();
        // The no-search fallback best move comes from the effective root
        // list: the first restricted move when the filter is active, the
        // first legal move otherwise.
        let fallback_move = if self.root_filter.is_empty() {
            root_moves.as_slice()[0]
        } else {
            self.root_filter[0]
        };

        // DTZ root ranking. The leaf probe makes every win-preserving move
        // score the same tablebase value, so the root would otherwise keep
        // whichever move the ordering yields first. Probing the root once and
        // trying the DTZ-optimal move first is a tie-break only: the search
        // still runs and a real mate still outranks the tablebase value. Only
        // the main thread ranks (helpers search the same job but their result
        // is never played) and a ponder search never volunteers a move.
        if self.params.use_syzygy_dtz_root_ranking
            && self.tb_probing
            && !self.limits.ponder
            && !self.publishes_nodes
            && let Some(tables) = self.tb.as_ref()
            && tables.covers(position)
        {
            let result = tables.probe_root(position);
            if !result.failed
                && !result.is_checkmate
                && !result.is_stalemate
                && !result.mv.is_none()
            {
                self.root_tb_move = result.mv;
                self.root_tb_wdl = Some(result.wdl);
                self.tb_hits += 1;
            }
        }

        let max_depth = self.limits.depth_limit();
        // Reporting clock for the `info` lines: it starts with the search,
        // so every iteration publishes the elapsed time from the `go` (the
        // time manager keeps its own clock, started by the `init_*` call).
        let started = std::time::Instant::now();
        let mut best = SearchResult {
            best_move: fallback_move,
            score: 0,
            depth: 0,
            seldepth: 0,
            nodes: 0,
            pv: Vec::new(),
        };
        let mut last_score = 0;
        // The root accumulator pair is refreshed once per search: every
        // deeper ply pushes its pair incrementally from the parent's.
        self.refresh_accumulators(position, 0);

        for depth in 1..=max_depth {
            // The stop flag is honoured before each iteration too: a `stop`
            // that arrived between iterations must not start a new one. The
            // per-node polling in `should_stop` alone could miss it for up
            // to 1023 nodes into the next iteration.
            if self.stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            // The counting variant of the search is a separate instantiation:
            // the normal one contains no statistics code at all.
            let root = if self.collect_stats {
                self.search_root::<true>(position, depth, last_score, observer)
            } else {
                self.search_root::<false>(position, depth, last_score, observer)
            };
            let Some((score, pv)) = root else {
                break;
            };
            last_score = score;
            best.score = score;
            best.depth = depth;
            best.seldepth = self.seldepth;
            best.nodes = self.reported_nodes();
            best.pv = pv;
            self.completed_depth = depth;
            if let Some(&first) = best.pv.first() {
                best.best_move = first;
            }
            let time_ms = started.elapsed().as_millis() as u64;
            observer(SearchEvent::Info(&SearchInfo {
                depth: best.depth,
                seldepth: best.seldepth,
                score: best.score,
                nodes: best.nodes,
                tb_hits: (self.tb.is_some() && self.tb_probing).then(|| self.reported_tb_hits()),
                time_ms,
                nps: nps_of(best.nodes, time_ms),
                // Read once per completed iteration, not per node: the
                // occupancy is only published on the `info` line.
                hashfull: self.tables.tt.hashfull(),
                pv: best.pv.clone(),
            }));

            // Time management: record the completed depth, feed the node
            // statistics and the best-move stability, then decide whether
            // the next iteration starts. A proven mate breaks first — a
            // proven mate needs no more depth. With a `go mate N` limiter the
            // break waits for a mate within N moves (a slower proven mate
            // keeps searching, per the step-0 capture); being mated always
            // ends it.
            let mated = score <= -MATE_LIMIT;
            let our_mate_moves = (MATE_SCORE - score + 1) / 2;
            let within_mate_bound = self
                .limits
                .mate
                .is_none_or(|bound| our_mate_moves <= bound as i32);
            if score >= MATE_LIMIT && within_mate_bound || mated {
                break;
            }
            self.time_manager.set_completed_depth(depth);
            let best_index =
                usize::from(best.best_move.from()) * 64 + usize::from(best.best_move.to());
            let best_nodes = self.nodes_per_move[best_index];
            self.time_manager.set_node_stats(best_nodes, self.nodes);
            self.time_manager
                .update_pv_stability(best_index as i32, true);
            if !self
                .time_manager
                .should_continue(self.now_ms(), best_index as i32, score)
            {
                break;
            }
            // History ages by 7/8 per completed iteration from depth 4 on,
            // matching the reference's saturation regulator.
            if depth >= 4 {
                self.history.age();
            }
        }

        best.nodes = self.reported_nodes();
        best.seldepth = self.seldepth;
        best
    }

    /// Searches one iteration, optionally inside an aspiration window.
    ///
    /// Returns `None` when the iteration was interrupted by a limit; the caller
    /// then keeps the result of the previous depth.
    ///
    /// The window handling mirrors the reference: a fail-low halves the window
    /// (`beta = (alpha + beta) / 2`) and shifts alpha down by the current
    /// delta, a fail-high raises beta by the delta, and the delta widens
    /// geometrically by half of itself per cycle. A mate score or an exhausted
    /// cycle budget drops the loop back to the full window, which terminates
    /// the iteration.
    pub(super) fn search_root<const STATS: bool>(
        &mut self,
        position: &Position,
        depth: i32,
        previous_score: i32,
        observer: &mut dyn FnMut(SearchEvent),
    ) -> Option<(i32, Vec<Move>)> {
        let params = self.params;
        let mut delta = params.aspiration_base;
        let mut alpha = -INFINITE;
        let mut beta = INFINITE;
        if depth >= params.aspiration_start_depth && previous_score.abs() < MATE_LIMIT {
            // No clamping on the widening arithmetic: the full-window bounds
            // stay far outside the mate range, and the reference widens in
            // plain `int` arithmetic without bounds.
            alpha = previous_score - delta;
            beta = previous_score + delta;
        }

        let mut cycles = 0;
        loop {
            if STATS && alpha != -INFINITE {
                self.stats.aspiration_searches += 1;
            }
            let score = self.negamax::<STATS>(
                position,
                alpha,
                beta,
                depth,
                0,
                None,
                false,
                Move::NONE,
                0,
                observer,
            );
            if self.stopped {
                return None;
            }
            if STATS {
                if score <= alpha {
                    self.stats.aspiration_fails_low += 1;
                } else if score >= beta {
                    self.stats.aspiration_fails_high += 1;
                }
            }
            // Fifty-move-aware root correction, applied before the window
            // test (so a clamped score that leaves the window is re-searched,
            // exactly like the reference): the leaf probe ignores the real
            // counter, so a WDL win that the actual counter turns into a
            // cursed win or draw must not be reported as a decisive
            // tablebase win. Mate scores and genuine wins are untouched.
            let score = if self.root_tb_wdl.is_some_and(|wdl| matches!(wdl, -1..=1))
                && score.abs() >= crate::tb::syzygy::TB_WIN_SCORE_MIN
                && score.abs() < MATE_LIMIT
            {
                0
            } else {
                score
            };
            if score > alpha && score < beta {
                return Some((score, self.pv.line(0).to_vec()));
            }
            if score.abs() >= MATE_LIMIT || cycles >= params.aspiration_max_cycles {
                // One final search with the full window: any score lies
                // inside it, so the next pass terminates.
                alpha = -INFINITE;
                beta = INFINITE;
                delta = params.aspiration_base;
                continue;
            }
            if score <= alpha {
                // Fail-low: halve the window and shift alpha by the delta.
                beta = (alpha + beta) / 2;
                alpha -= delta;
            } else {
                // Fail-high: raise beta by the delta.
                beta += delta;
            }
            delta += delta / 2;
            cycles += 1;
        }
    }
}

/// Derives nodes-per-second from a node count and an elapsed time.
///
/// An elapsed time of zero (an iteration that finished inside the clock
/// resolution) counts as one millisecond, so the division can never divide by
/// zero and a completed iteration is never reported as zero nps.
#[inline]
pub(crate) fn nps_of(nodes: u64, time_ms: u64) -> u64 {
    nodes * 1000 / time_ms.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nps_arithmetic_and_the_zero_time_guard() {
        // Ordinary arithmetic: nodes per second, rounded down.
        assert_eq!(nps_of(1_500, 10), 150_000);
        assert_eq!(nps_of(47, 1), 47_000);
        // The guard: a zero elapsed time counts as one millisecond, so a
        // completed iteration is never reported as zero nps and nothing
        // divides by zero.
        assert_eq!(nps_of(47, 0), 47_000);
        assert_eq!(nps_of(0, 0), 0);
    }
}
