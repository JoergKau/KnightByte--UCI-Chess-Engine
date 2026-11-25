//! Stopping and sharing: the stop flag, the shared tables, the helper threads'
//! node counts and the periodic stop check.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::Searcher;

impl Searcher {
    /// Installs a fresh, cleared stop flag and returns its handle.
    ///
    /// The UCI loop keeps the handle and sets it on `stop` (and on the
    /// join-first fallback for protocol-breaking commands); the search polls
    /// it every 1024 nodes and before every iteration. Arming before every
    /// `go` also clears any flag left over from a previous search.
    pub fn arm_stop_flag(&mut self) -> Arc<std::sync::atomic::AtomicBool> {
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.stop_flag = Arc::clone(&flag);
        flag
    }

    /// Returns the shared tables (the step-7 handle).
    pub fn tables(&self) -> &crate::search::tables::SharedTables {
        &self.tables
    }

    /// Adopts `flag` as this searcher's stop flag instead of arming a fresh
    /// one.
    ///
    /// The worker pool uses this to give a helper the main search's flag, so
    /// the helper's iteration ends when the main thread's does.
    pub fn use_stop_flag(&mut self, flag: Arc<std::sync::atomic::AtomicBool>) {
        self.stop_flag = flag;
    }

    /// Replaces the shared tables (the step-7 handle).
    ///
    /// The worker pool swaps the shared handle in before every helper
    /// search, so all threads always probe the same table.
    pub fn set_tables(&mut self, tables: crate::search::tables::SharedTables) {
        self.tables = tables;
    }

    /// Releases the shared tables (bug B39 fix).
    ///
    /// A persistent helper must not keep the `SharedTables` clones between
    /// searches: the clones would keep the transposition table's `Arc` count
    /// above one forever, and every later `setoption Hash` resize would fail
    /// its exclusive-borrow check and kill the engine. After a search, the
    /// helper holds only its own minimal private tables until the next job
    /// arrives.
    pub fn release_tables(&mut self) {
        self.tables = crate::search::tables::SharedTables::new(1);
    }

    /// Attaches the shared node counter of a helper search.
    ///
    /// While attached, the reported node counts are this searcher's exact
    /// count plus the helpers' published blocks, and the node limit is also
    /// checked against that aggregate at the 1024-node poll.
    pub fn attach_helpers(&mut self, helper_nodes: Arc<AtomicU64>) {
        self.helper_nodes = Some(helper_nodes);
    }

    /// Detaches the shared node counter (back to the exact single-threaded
    /// reporting).
    pub fn detach_helpers(&mut self) {
        self.helper_nodes = None;
        self.publishes_nodes = false;
    }

    /// Makes this searcher a HELPER that publishes its node blocks to
    /// `counter` (the counter the main searcher reports from).
    pub fn attach_publisher(&mut self, counter: Arc<AtomicU64>) {
        self.helper_nodes = Some(counter);
        self.publishes_nodes = true;
    }

    /// Publishes this searcher's unpublished node block to the shared
    /// counter.
    ///
    /// Helpers call this from the 1024-node poll, where it publishes only
    /// full 2048-node blocks (no per-node atomic), and once with `force` when
    /// their search ends, which flushes the final partial block. The main
    /// thread never publishes (its count stays exact).
    pub fn publish_node_blocks(&mut self, force: bool) {
        if !self.publishes_nodes {
            return;
        }
        if let Some(counter) = &self.helper_nodes {
            let unpublished = self.nodes - self.published_nodes;
            if unpublished > 0 && (force || unpublished >= 2048) {
                counter.fetch_add(unpublished, Ordering::Relaxed);
                self.published_nodes = self.nodes;
            }
            if force || unpublished >= 2048 {
                let new_hits = self.tb_hits - self.published_tb_hits;
                if new_hits > 0 {
                    crate::search::tb::publish_helper_tb_hits(new_hits);
                    self.published_tb_hits = self.tb_hits;
                }
            }
        }
    }

    /// Returns the node count this searcher reports: its own exact count
    /// plus, while helpers are attached, the helpers' published blocks.
    pub fn reported_nodes(&self) -> u64 {
        self.nodes
            + self
                .helper_nodes
                .as_ref()
                .map_or(0, |counter| counter.load(Ordering::Relaxed))
    }

    /// Returns the tablebase hits this searcher reports: its own exact count
    /// plus, while helpers are attached, what the helpers have published.
    pub fn reported_tb_hits(&self) -> u64 {
        if self.helper_nodes.is_some() && !self.publishes_nodes {
            self.tb_hits + crate::search::tb::helper_tb_hits()
        } else {
            self.tb_hits
        }
    }

    /// Returns whether the search must stop before visiting another node.
    ///
    /// Two checks stay on every node, and they are plain integer comparisons:
    /// the exact `go nodes` limit against this thread's own count, and the
    /// remembered stop decision. The expensive inputs — the wall clock, the
    /// shared stop flag and the helpers' published node blocks — are touched
    /// only every 1024 nodes (the plan's step-4 rule: no clock read and no
    /// atomic access per node).
    pub(crate) fn should_stop(&mut self) -> bool {
        use std::sync::atomic::Ordering;
        if self.stopped {
            return true;
        }
        if let Some(limit) = self.limits.nodes
            && self.nodes >= limit
        {
            self.stopped = true;
            return true;
        }
        // `nodes` was incremented before this call, so the polling window
        // fires once every 1024 visited nodes.
        if self.nodes & 1023 == 0 {
            // Step 8: helpers publish their counts in blocks; with helpers
            // attached, the node limit is enforced on the aggregate here.
            if self.publishes_nodes {
                self.publish_node_blocks(false);
            } else if self.helper_nodes.is_some()
                && let Some(limit) = self.limits.nodes
                && self.reported_nodes() >= limit
            {
                self.stopped = true;
                return true;
            }
            if self.stop_flag.load(Ordering::Relaxed)
                || self.time_manager.should_stop(self.now_ms())
            {
                self.stopped = true;
            }
        }
        self.stopped
    }
}
