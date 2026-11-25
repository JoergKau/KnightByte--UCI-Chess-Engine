//! Tablebase access from the search.
//!
//! The searcher keeps one tablebase handle per search (refreshed from the
//! process-wide set in [`Searcher::search`](crate::search::searcher::Searcher::search)),
//! so a probe in the tree costs only the prober's own lock. These helpers keep
//! the decision logic out of the node functions: they answer "is this position
//! covered, and what is its exact WDL/DTZ value".

use std::sync::atomic::{AtomicU64, Ordering};

use crate::board::position::Position;

use super::searcher::Searcher;

/// Tablebase hits published by the helper threads of the current search.
///
/// Helpers flush their private counts here in blocks, next to their node
/// blocks; the main searcher reports its own exact count plus this total. The
/// UCI layer clears it before it launches a search.
static HELPER_TB_HITS: AtomicU64 = AtomicU64::new(0);

/// Clears the helpers' published tablebase hits (start of a search).
pub fn reset_helper_tb_hits() {
    HELPER_TB_HITS.store(0, Ordering::Relaxed);
}

/// Adds a helper's unpublished hits to the shared total.
pub(crate) fn publish_helper_tb_hits(hits: u64) {
    HELPER_TB_HITS.fetch_add(hits, Ordering::Relaxed);
}

/// Returns the hits the helpers have published so far.
pub(crate) fn helper_tb_hits() -> u64 {
    HELPER_TB_HITS.load(Ordering::Relaxed)
}

impl Searcher {
    /// Returns the WDL value of `position` when it is inside the loaded tables.
    ///
    /// The master switch, the loaded material cardinality, castling rights and
    /// position legality are all checked here, so callers can treat `None` as
    /// "the search must decide this node normally". Every value returned is
    /// one tablebase hit, counted for the `tbhits` field of the `info` line.
    pub(crate) fn tb_wdl(&mut self, position: &Position) -> Option<i32> {
        if !self.tb_probing {
            return None;
        }
        let wdl = self.tb.as_ref()?.probe_wdl(position);
        self.tb_hits += u64::from(wdl.is_some());
        wdl
    }

    /// Returns the DTZ value of `position` when it is inside the loaded tables.
    pub(crate) fn tb_dtz(&self, position: &Position) -> Option<i32> {
        if !self.tb_probing {
            return None;
        }
        self.tb.as_ref()?.probe_dtz(position)
    }

    /// Returns the score of a node inside the loaded tables, or `None` when
    /// the search must decide the node itself.
    ///
    /// The probe is unconditional on depth: inside the tables the value is
    /// exact, and a depth gate would let the heuristic pruning stack mask it.
    /// A PV node is graded by the distance to the next zeroing move, and a
    /// value whose distance fits in the remaining `depth` is searched so the
    /// search can reach the real mate itself.
    pub(crate) fn tb_node_score(
        &mut self,
        position: &Position,
        pv_node: bool,
        depth: i32,
    ) -> Option<i32> {
        let wdl = self.tb_wdl(position)?;
        if pv_node && !self.params.allow_syzygy_cutoff_on_pv_nodes {
            let dtz_abs = self.tb_dtz(position).map_or(0, |dtz| dtz.abs());
            if dtz_abs <= 0 || dtz_abs > depth {
                return Some(crate::tb::syzygy::wdl_to_score_graded(wdl, dtz_abs));
            }
            None
        } else {
            Some(crate::tb::syzygy::wdl_to_score(wdl))
        }
    }
}
