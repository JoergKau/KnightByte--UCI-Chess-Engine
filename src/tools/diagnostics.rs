//! Runtime diagnostics behind the hidden `searchstats` and `nnueregions` UCI
//! commands.
//!
//! Both are opt-in measurement tools for the engine's developers, off by
//! default, and neither changes what the engine plays:
//!
//! * `searchstats` counts, per search, how much of the tree is quiescence, how
//!   often the transposition table is probed and hit, and how early the cutoff
//!   move is found. Every searcher keeps plain per-thread counters, copies the
//!   switch once at the start of a search (so the node loop never reads an
//!   atomic) and merges its counters into one process-wide snapshot when its
//!   search ends.
//! * `nnueregions` counts accumulator refreshes per king region: the
//!   real-game distribution of the network's region table. Refreshes are rare
//!   (a few percent of the nodes), so the counters are atomics read on that
//!   path only.
//!
//! The command surface and the output lines follow the reference engine's
//! hidden hooks so existing analysis scripts keep working.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Number of cutoff-index buckets; the last bucket means "that index or later".
pub const CUTOFF_BUCKETS: usize = 16;

/// Number of remaining-depth buckets of the main-search node histogram:
/// depths 1, 2, 3, 4, 5-6, 7-8, 9-12 and 13 or more.
pub const DEPTH_BUCKETS: usize = 8;

/// Returns the histogram bucket of a main-search node with `depth` plies left.
#[inline]
pub fn depth_bucket(depth: i32) -> usize {
    match depth {
        ..=1 => 0,
        2 => 1,
        3 => 2,
        4 => 3,
        5..=6 => 4,
        7..=8 => 5,
        9..=12 => 6,
        _ => 7,
    }
}

/// Counters of one search (one searcher, or the merge of all of them).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchStats {
    /// Nodes visited, quiescence included.
    pub nodes: u64,
    /// Quiescence nodes.
    pub qnodes: u64,
    /// Transposition-table probes by the main search.
    pub tt_probes: u64,
    /// Probes that found an entry.
    pub tt_hits: u64,
    /// Transposition-table probes by quiescence.
    pub qtt_probes: u64,
    /// Quiescence probes that found an entry.
    pub qtt_hits: u64,
    /// Beta cutoffs in the main search.
    pub cutoffs: u64,
    /// Beta cutoffs in quiescence.
    pub qcutoffs: u64,
    /// Main-search cutoffs by the index of the cutoff move (0 = the first
    /// legal move tried).
    pub cutoff_index: [u64; CUTOFF_BUCKETS],
    /// Quiescence cutoffs by the index of the cutoff move.
    pub qcutoff_index: [u64; CUTOFF_BUCKETS],
    /// Late-move-reduction attempts (every move that reached the LMR block).
    pub lmr_attempts: u64,
    /// Sum of the reduction plies over all LMR attempts.
    pub lmr_reduction_sum: u64,
    /// LMR attempts whose reduction was greater than zero.
    pub lmr_applied: u64,
    /// Reduced zero-window re-searches after a fail-high.
    pub lmr_re_searches: u64,
    /// Full-window re-searches after a reduced or scout fail-high.
    pub lmr_full_re_searches: u64,
    /// Null-move searches started.
    pub null_move_attempts: u64,
    /// Null-move searches that cut off.
    pub null_move_cutoffs: u64,
    /// ProbCut move-loop attempts that passed the evaluation and table gate.
    pub probcut_attempts: u64,
    /// ProbCut attempts that proved a cutoff.
    pub probcut_cutoffs: u64,
    /// Aspiration-window root searches (a finite window).
    pub aspiration_searches: u64,
    /// Aspiration searches that failed low.
    pub aspiration_fails_low: u64,
    /// Aspiration searches that failed high.
    pub aspiration_fails_high: u64,
    /// Check extensions applied.
    pub check_extensions: u64,
    /// Singular-verification searches started.
    pub singular_verifications: u64,
    /// Nodes extended by one or more singular plies.
    pub singular_extensions: u64,
    /// Moves searched one ply shallower by a negative extension.
    pub negative_extensions: u64,
    /// Table extensions applied by `UseTtExtension`.
    pub tt_extensions: u64,
    /// Internal-iterative-reduction plies removed.
    pub iir_reductions: u64,
    /// Accepted table stores (new, same-key or replaced).
    pub tt_stores: u64,
    /// Stores that used an empty slot.
    pub tt_store_new: u64,
    /// Stores that refreshed a same-key entry.
    pub tt_store_same_key: u64,
    /// Stores that evicted a lower-value victim.
    pub tt_store_replaced: u64,
    /// Store requests that were refused (zero key, or a deeper same-key entry).
    pub tt_store_skipped: u64,
    /// Main-search nodes by remaining depth at entry (after the first drop
    /// into quiescence), bucketed by [`depth_bucket`].
    pub main_depth: [u64; DEPTH_BUCKETS],
    /// Main-search nodes cut off by a transposition-table entry.
    pub tt_cutoffs: u64,
    /// Nodes cut off by reverse futility pruning (direct and verified).
    pub rfp_cutoffs: u64,
    /// Of those, the nodes whose cutoff a quiescence verification confirmed.
    pub rfp_verified_cutoffs: u64,
    /// Nodes that razoring proved to fail low.
    pub razor_cutoffs: u64,
    /// Times late-move pruning ended the quiet moves of a node.
    pub lmp_skips: u64,
    /// Quiet moves or move stages skipped by futility pruning.
    pub futility_skips: u64,
    /// Quiet moves skipped by history pruning.
    pub history_skips: u64,
    /// Moves skipped by SEE pruning.
    pub see_skips: u64,
    /// Moves made in the main-search move loop.
    pub moves_made: u64,
}

impl SearchStats {
    /// Returns zeroed counters.
    pub const fn new() -> Self {
        Self {
            nodes: 0,
            qnodes: 0,
            tt_probes: 0,
            tt_hits: 0,
            qtt_probes: 0,
            qtt_hits: 0,
            cutoffs: 0,
            qcutoffs: 0,
            cutoff_index: [0; CUTOFF_BUCKETS],
            qcutoff_index: [0; CUTOFF_BUCKETS],
            lmr_attempts: 0,
            lmr_reduction_sum: 0,
            lmr_applied: 0,
            lmr_re_searches: 0,
            lmr_full_re_searches: 0,
            null_move_attempts: 0,
            null_move_cutoffs: 0,
            probcut_attempts: 0,
            probcut_cutoffs: 0,
            aspiration_searches: 0,
            aspiration_fails_low: 0,
            aspiration_fails_high: 0,
            check_extensions: 0,
            singular_verifications: 0,
            singular_extensions: 0,
            negative_extensions: 0,
            tt_extensions: 0,
            iir_reductions: 0,
            tt_stores: 0,
            tt_store_new: 0,
            tt_store_same_key: 0,
            tt_store_replaced: 0,
            tt_store_skipped: 0,
            main_depth: [0; DEPTH_BUCKETS],
            tt_cutoffs: 0,
            rfp_cutoffs: 0,
            rfp_verified_cutoffs: 0,
            razor_cutoffs: 0,
            lmp_skips: 0,
            futility_skips: 0,
            history_skips: 0,
            see_skips: 0,
            moves_made: 0,
        }
    }

    /// Adds `other` into `self`.
    pub fn add(&mut self, other: &SearchStats) {
        self.nodes += other.nodes;
        self.qnodes += other.qnodes;
        self.tt_probes += other.tt_probes;
        self.tt_hits += other.tt_hits;
        self.qtt_probes += other.qtt_probes;
        self.qtt_hits += other.qtt_hits;
        self.cutoffs += other.cutoffs;
        self.qcutoffs += other.qcutoffs;
        for index in 0..CUTOFF_BUCKETS {
            self.cutoff_index[index] += other.cutoff_index[index];
            self.qcutoff_index[index] += other.qcutoff_index[index];
        }
        self.lmr_attempts += other.lmr_attempts;
        self.lmr_reduction_sum += other.lmr_reduction_sum;
        self.lmr_applied += other.lmr_applied;
        self.lmr_re_searches += other.lmr_re_searches;
        self.lmr_full_re_searches += other.lmr_full_re_searches;
        self.null_move_attempts += other.null_move_attempts;
        self.null_move_cutoffs += other.null_move_cutoffs;
        self.probcut_attempts += other.probcut_attempts;
        self.probcut_cutoffs += other.probcut_cutoffs;
        self.aspiration_searches += other.aspiration_searches;
        self.aspiration_fails_low += other.aspiration_fails_low;
        self.aspiration_fails_high += other.aspiration_fails_high;
        self.check_extensions += other.check_extensions;
        self.singular_verifications += other.singular_verifications;
        self.singular_extensions += other.singular_extensions;
        self.negative_extensions += other.negative_extensions;
        self.tt_extensions += other.tt_extensions;
        self.iir_reductions += other.iir_reductions;
        self.tt_stores += other.tt_stores;
        self.tt_store_new += other.tt_store_new;
        self.tt_store_same_key += other.tt_store_same_key;
        self.tt_store_replaced += other.tt_store_replaced;
        self.tt_store_skipped += other.tt_store_skipped;
        for index in 0..DEPTH_BUCKETS {
            self.main_depth[index] += other.main_depth[index];
        }
        self.tt_cutoffs += other.tt_cutoffs;
        self.rfp_cutoffs += other.rfp_cutoffs;
        self.rfp_verified_cutoffs += other.rfp_verified_cutoffs;
        self.razor_cutoffs += other.razor_cutoffs;
        self.lmp_skips += other.lmp_skips;
        self.futility_skips += other.futility_skips;
        self.history_skips += other.history_skips;
        self.see_skips += other.see_skips;
        self.moves_made += other.moves_made;
    }

    /// Records a main-search node that stays in the main search, by its
    /// remaining depth.
    #[inline]
    pub fn record_main_node(&mut self, depth: i32) {
        self.main_depth[depth_bucket(depth)] += 1;
    }

    /// Returns the number of nodes that stayed in the main search after the
    /// first drop into quiescence.
    ///
    /// This is the sum of the depth histogram, not `nodes - qnodes`: razoring
    /// probes quiescence for a node that then continues in the main search,
    /// so that subtraction would under-count the main nodes.
    pub fn main_nodes(&self) -> u64 {
        self.main_depth.iter().sum()
    }

    /// Formats the second report line of `searchstats`: how the main search
    /// prunes and where its nodes sit, per remaining depth. The first line
    /// keeps the predecessor's format; this one is KnightByte's own.
    pub fn report_pruning(&self) -> String {
        let main = self.main_nodes();
        let per_node = |count: u64| {
            if main > 0 {
                count as f64 / main as f64
            } else {
                0.0
            }
        };
        let q_per_main = if main > 0 {
            self.qnodes as f64 / main as f64
        } else {
            0.0
        };
        let depths: Vec<String> = self.main_depth.iter().map(u64::to_string).collect();
        format!(
            "searchprune main {} qpermain {:.2} moves {} permain {:.2} \
             ttcut {} rfp {} rfpv {} razor {} lmp {} fut {} hist {} see {} \
             rfppermain {:.3} depth1-2-3-4-5/6-7/8-9/12-13+ {}",
            main,
            q_per_main,
            self.moves_made,
            per_node(self.moves_made),
            self.tt_cutoffs,
            self.rfp_cutoffs,
            self.rfp_verified_cutoffs,
            self.razor_cutoffs,
            self.lmp_skips,
            self.futility_skips,
            self.history_skips,
            self.see_skips,
            per_node(self.rfp_cutoffs),
            depths.join("/"),
        )
    }

    /// Records one main-search transposition-table probe.
    #[inline]
    pub fn record_tt_probe(&mut self, hit: bool) {
        self.tt_probes += 1;
        self.tt_hits += u64::from(hit);
    }

    /// Records one quiescence transposition-table probe.
    #[inline]
    pub fn record_qtt_probe(&mut self, hit: bool) {
        self.qtt_probes += 1;
        self.qtt_hits += u64::from(hit);
    }

    /// Records a main-search beta cutoff by the `legal_move_index`-th move.
    #[inline]
    pub fn record_cutoff(&mut self, legal_move_index: usize) {
        self.cutoffs += 1;
        self.cutoff_index[legal_move_index.min(CUTOFF_BUCKETS - 1)] += 1;
    }

    /// Records a quiescence beta cutoff by the `legal_move_index`-th move.
    #[inline]
    pub fn record_qcutoff(&mut self, legal_move_index: usize) {
        self.qcutoffs += 1;
        self.qcutoff_index[legal_move_index.min(CUTOFF_BUCKETS - 1)] += 1;
    }

    /// Records one late-move-reduction attempt and its reduction in plies.
    #[inline]
    pub fn record_lmr(&mut self, reduction: i32) {
        self.lmr_attempts += 1;
        self.lmr_reduction_sum += reduction.max(0) as u64;
        self.lmr_applied += u64::from(reduction > 0);
    }

    /// Records a re-search after a reduced fail-high; `full` marks the
    /// full-window PV re-search rather than the zero-window verification.
    #[inline]
    pub fn record_lmr_re_search(&mut self, full: bool) {
        if full {
            self.lmr_full_re_searches += 1;
        } else {
            self.lmr_re_searches += 1;
        }
    }

    /// Records one null-move search attempt.
    #[inline]
    pub fn record_null_move(&mut self) {
        self.null_move_attempts += 1;
    }

    /// Records one null-move cutoff.
    #[inline]
    pub fn record_null_move_cutoff(&mut self) {
        self.null_move_cutoffs += 1;
    }

    /// Records one ProbCut attempt.
    #[inline]
    pub fn record_probcut(&mut self) {
        self.probcut_attempts += 1;
    }

    /// Records one ProbCut cutoff.
    #[inline]
    pub fn record_probcut_cutoff(&mut self) {
        self.probcut_cutoffs += 1;
    }

    /// Records one accepted or refused table store.
    #[inline]
    pub fn record_tt_store(&mut self, outcome: crate::search::tt::TtStoreOutcome) {
        use crate::search::tt::TtStoreOutcome;
        match outcome {
            TtStoreOutcome::Empty => {
                self.tt_stores += 1;
                self.tt_store_new += 1;
            }
            TtStoreOutcome::SameKey => {
                self.tt_stores += 1;
                self.tt_store_same_key += 1;
            }
            TtStoreOutcome::Replaced => {
                self.tt_stores += 1;
                self.tt_store_replaced += 1;
            }
            TtStoreOutcome::Skipped => self.tt_store_skipped += 1,
        }
    }

    /// Formats the one-line report of `searchstats`.
    pub fn report(&self) -> String {
        let percent = |part: u64, whole: u64| {
            if whole > 0 {
                100.0 * part as f64 / whole as f64
            } else {
                0.0
            }
        };
        let avg_reduction = if self.lmr_attempts > 0 {
            self.lmr_reduction_sum as f64 / self.lmr_attempts as f64
        } else {
            0.0
        };
        format!(
            "searchstats nodes {} qnodes {} qshare {:.1}% \
             tt {} hits {} hitrate {:.1}% \
             qtt {} hits {} hitrate {:.1}% \
             cutoffs {} first {:.1}% second {:.1}% third {:.1}% \
             qcutoffs {} first {:.1}% second {:.1}% \
             lmr {} avgred {:.1} applied {} res {} full {} \
             nmp {} cut {} probcut {} cut {} \
             asp {} low {} high {} \
             ext chk {} sing {} singv {} neg {} tt {} iir {} \
             store {} new {} same {} repl {} skip {}",
            self.nodes,
            self.qnodes,
            percent(self.qnodes, self.nodes),
            self.tt_probes,
            self.tt_hits,
            percent(self.tt_hits, self.tt_probes),
            self.qtt_probes,
            self.qtt_hits,
            percent(self.qtt_hits, self.qtt_probes),
            self.cutoffs,
            percent(self.cutoff_index[0], self.cutoffs),
            percent(self.cutoff_index[1], self.cutoffs),
            percent(self.cutoff_index[2], self.cutoffs),
            self.qcutoffs,
            percent(self.qcutoff_index[0], self.qcutoffs),
            percent(self.qcutoff_index[1], self.qcutoffs),
            self.lmr_attempts,
            avg_reduction,
            self.lmr_applied,
            self.lmr_re_searches,
            self.lmr_full_re_searches,
            self.null_move_attempts,
            self.null_move_cutoffs,
            self.probcut_attempts,
            self.probcut_cutoffs,
            self.aspiration_searches,
            self.aspiration_fails_low,
            self.aspiration_fails_high,
            self.check_extensions,
            self.singular_extensions,
            self.singular_verifications,
            self.negative_extensions,
            self.tt_extensions,
            self.iir_reductions,
            self.tt_stores,
            self.tt_store_new,
            self.tt_store_same_key,
            self.tt_store_replaced,
            self.tt_store_skipped,
        )
    }
}

impl Default for SearchStats {
    fn default() -> Self {
        Self::new()
    }
}

static SEARCH_STATS_ENABLED: AtomicBool = AtomicBool::new(false);
static LAST_SEARCH: Mutex<SearchStats> = Mutex::new(SearchStats::new());

/// Turns search counting on or off (it applies from the next search on).
pub fn set_search_stats_enabled(enabled: bool) {
    SEARCH_STATS_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Returns whether searches are being counted.
pub fn search_stats_enabled() -> bool {
    SEARCH_STATS_ENABLED.load(Ordering::Relaxed)
}

/// Forgets the snapshot of the last counted search.
pub fn clear_last_search() {
    *LAST_SEARCH.lock().expect("the stats lock must not poison") = SearchStats::new();
}

/// Adds one searcher's counters to the snapshot of the current search.
pub fn merge_into_last_search(stats: &SearchStats) {
    LAST_SEARCH
        .lock()
        .expect("the stats lock must not poison")
        .add(stats);
}

/// Returns a copy of the snapshot of the last counted search.
pub fn last_search() -> SearchStats {
    *LAST_SEARCH.lock().expect("the stats lock must not poison")
}

/// Upper bound of the network's region count (the loader enforces it).
const MAX_REGIONS: usize = 32;

static REGION_COUNTS: [AtomicU64; MAX_REGIONS] = [const { AtomicU64::new(0) }; MAX_REGIONS];
static REGION_COUNTING: AtomicBool = AtomicBool::new(false);

/// Starts or stops counting accumulator refreshes per king region.
pub fn set_region_counting(enabled: bool) {
    REGION_COUNTING.store(enabled, Ordering::Relaxed);
}

/// Returns whether refreshes are being counted.
pub fn region_counting() -> bool {
    REGION_COUNTING.load(Ordering::Relaxed)
}

/// Clears every region counter.
pub fn reset_regions() {
    for counter in &REGION_COUNTS {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Counts one accumulator refresh of `region`; a no-op unless counting is on.
///
/// `region` is below `MAX_REGIONS` because the network loader rejects any
/// larger region table.
#[inline]
pub fn record_region(region: usize) {
    if REGION_COUNTING.load(Ordering::Relaxed) {
        REGION_COUNTS[region % MAX_REGIONS].fetch_add(1, Ordering::Relaxed);
    }
}

/// Returns the first `regions` counters and their total.
pub fn region_snapshot(regions: usize) -> (Vec<u64>, u64) {
    let counts: Vec<u64> = REGION_COUNTS
        .iter()
        .take(regions.min(MAX_REGIONS))
        .map(|counter| counter.load(Ordering::Relaxed))
        .collect();
    let total = counts.iter().sum();
    (counts, total)
}

/// Formats the histogram line of `nnueregions`.
pub fn region_report(counts: &[u64], total: u64, counting: bool) -> String {
    let mut line = String::from("nnueregions");
    for (region, count) in counts.iter().enumerate() {
        line.push_str(&format!(" r{region}={count}"));
    }
    line.push_str(&format!(" total {total}"));
    if counting {
        line.push_str(" (counting)");
    }
    line
}

/// Formats the per-region shares line, or `None` when nothing was counted.
pub fn region_shares(counts: &[u64], total: u64) -> Option<String> {
    if total == 0 {
        return None;
    }
    let shares: Vec<String> = counts
        .iter()
        .enumerate()
        .map(|(region, &count)| format!("r{region} {:.3}%", 100.0 * count as f64 / total as f64))
        .collect();
    Some(format!(
        "info string nnueregions shares: {}",
        shares.join(" ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cutoffs_land_in_their_bucket_and_the_last_bucket_collects_the_rest() {
        let mut stats = SearchStats::new();
        stats.record_cutoff(0);
        stats.record_cutoff(0);
        stats.record_cutoff(2);
        stats.record_cutoff(15);
        stats.record_cutoff(400);
        assert_eq!(stats.cutoffs, 5);
        assert_eq!(stats.cutoff_index[0], 2);
        assert_eq!(stats.cutoff_index[2], 1);
        assert_eq!(stats.cutoff_index[15], 2);
    }

    #[test]
    fn depth_buckets_cover_every_remaining_depth_once() {
        let expected = [
            (-3, 0),
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 4),
            (6, 4),
            (7, 5),
            (8, 5),
            (9, 6),
            (12, 6),
            (13, 7),
            (60, 7),
        ];
        for (depth, bucket) in expected {
            assert_eq!(depth_bucket(depth), bucket, "depth {depth}");
        }
    }

    #[test]
    fn main_nodes_are_the_histogram_sum_not_nodes_minus_qnodes() {
        // Razoring probes quiescence for a node that stays in the main
        // search, so `nodes - qnodes` would under-count the main nodes.
        let mut stats = SearchStats::new();
        for depth in [1, 1, 2, 5, 20] {
            stats.record_main_node(depth);
        }
        stats.nodes = 9;
        stats.qnodes = 6;
        assert_eq!(stats.main_nodes(), 5);
        assert_eq!(stats.main_depth, [2, 1, 0, 0, 1, 0, 0, 1]);
    }

    #[test]
    fn the_pruning_counters_merge_and_report() {
        let mut first = SearchStats::new();
        first.record_main_node(1);
        first.moves_made = 4;
        first.rfp_cutoffs = 1;
        first.tt_cutoffs = 2;
        let mut second = SearchStats::new();
        second.record_main_node(2);
        second.moves_made = 2;
        second.razor_cutoffs = 1;
        second.lmp_skips = 3;
        second.futility_skips = 4;
        second.history_skips = 5;
        second.see_skips = 6;
        first.add(&second);
        assert_eq!(first.main_nodes(), 2);
        assert_eq!(first.moves_made, 6);
        assert_eq!(
            (first.rfp_cutoffs, first.razor_cutoffs, first.tt_cutoffs),
            (1, 1, 2)
        );
        assert_eq!(
            (
                first.lmp_skips,
                first.futility_skips,
                first.history_skips,
                first.see_skips
            ),
            (3, 4, 5, 6)
        );
        first.qnodes = 3;
        let line = first.report_pruning();
        assert!(
            line.starts_with("searchprune main 2 qpermain 1.50 moves 6 permain 3.00 "),
            "{line}"
        );
        assert!(line.contains("ttcut 2 rfp 1 rfpv 0 razor 1 lmp 3 fut 4 hist 5 see 6 "));
        assert!(line.ends_with("1/1/0/0/0/0/0/0"), "{line}");
        assert!(
            SearchStats::new()
                .report_pruning()
                .starts_with("searchprune main 0 qpermain 0.00 moves 0 permain 0.00 ")
        );
    }

    #[test]
    fn merging_adds_every_counter() {
        let mut first = SearchStats::new();
        first.nodes = 10;
        first.record_tt_probe(true);
        first.record_qcutoff(1);
        first.record_lmr(2);
        first.record_null_move();
        first.record_null_move_cutoff();
        first.record_probcut();
        first.record_probcut_cutoff();
        first.aspiration_fails_low = 1;
        first.check_extensions = 1;
        first.tt_store_new = 1;
        let mut second = SearchStats::new();
        second.nodes = 5;
        second.record_tt_probe(false);
        second.record_qtt_probe(true);
        second.record_lmr(0);
        second.record_null_move();
        second.record_probcut();
        second.aspiration_fails_high = 1;
        second.tt_store_replaced = 1;
        first.add(&second);
        assert_eq!(first.nodes, 15);
        assert_eq!((first.tt_probes, first.tt_hits), (2, 1));
        assert_eq!((first.qtt_probes, first.qtt_hits), (1, 1));
        assert_eq!(first.qcutoff_index[1], 1);
        assert_eq!((first.lmr_attempts, first.lmr_reduction_sum), (2, 2));
        assert_eq!(first.lmr_applied, 1);
        assert_eq!((first.null_move_attempts, first.null_move_cutoffs), (2, 1));
        assert_eq!((first.probcut_attempts, first.probcut_cutoffs), (2, 1));
        assert_eq!(
            (first.aspiration_fails_low, first.aspiration_fails_high),
            (1, 1)
        );
        assert_eq!(
            (
                first.check_extensions,
                first.tt_store_new,
                first.tt_store_replaced
            ),
            (1, 1, 1)
        );
    }

    #[test]
    fn lmr_and_tt_store_helpers_count_what_their_names_say() {
        use crate::search::tt::TtStoreOutcome;
        let mut stats = SearchStats::new();
        stats.record_lmr(3);
        stats.record_lmr(0);
        stats.record_lmr_re_search(false);
        stats.record_lmr_re_search(true);
        assert_eq!((stats.lmr_attempts, stats.lmr_reduction_sum), (2, 3));
        assert_eq!(stats.lmr_applied, 1);
        assert_eq!((stats.lmr_re_searches, stats.lmr_full_re_searches), (1, 1));

        stats.record_tt_store(TtStoreOutcome::Empty);
        stats.record_tt_store(TtStoreOutcome::SameKey);
        stats.record_tt_store(TtStoreOutcome::Replaced);
        stats.record_tt_store(TtStoreOutcome::Skipped);
        assert_eq!(stats.tt_stores, 3);
        assert_eq!(stats.tt_store_new, 1);
        assert_eq!(stats.tt_store_same_key, 1);
        assert_eq!(stats.tt_store_replaced, 1);
        assert_eq!(stats.tt_store_skipped, 1);
    }

    #[test]
    fn the_report_has_the_documented_shape_and_no_nan() {
        let empty = SearchStats::new().report();
        assert!(!empty.contains("NaN"));
        assert!(empty.starts_with("searchstats nodes 0 qnodes 0 qshare 0.0% tt 0 hits 0"));

        let mut stats = SearchStats::new();
        stats.nodes = 200;
        stats.qnodes = 50;
        stats.record_tt_probe(true);
        stats.record_tt_probe(false);
        stats.record_cutoff(0);
        assert_eq!(
            stats.report(),
            "searchstats nodes 200 qnodes 50 qshare 25.0% tt 2 hits 1 hitrate 50.0% \
             qtt 0 hits 0 hitrate 0.0% cutoffs 1 first 100.0% second 0.0% third 0.0% \
             qcutoffs 0 first 0.0% second 0.0% lmr 0 avgred 0.0 applied 0 res 0 full 0 \
             nmp 0 cut 0 probcut 0 cut 0 asp 0 low 0 high 0 ext chk 0 sing 0 \
             singv 0 neg 0 tt 0 iir 0 store 0 new 0 same 0 repl 0 skip 0"
        );
    }

    #[test]
    fn region_lines_follow_the_reference_format() {
        let counts = [3, 1, 0, 0];
        assert_eq!(
            region_report(&counts, 4, false),
            "nnueregions r0=3 r1=1 r2=0 r3=0 total 4"
        );
        assert_eq!(
            region_report(&counts, 4, true),
            "nnueregions r0=3 r1=1 r2=0 r3=0 total 4 (counting)"
        );
        assert_eq!(
            region_shares(&counts, 4).as_deref(),
            Some("info string nnueregions shares: r0 75.000% r1 25.000% r2 0.000% r3 0.000%")
        );
        assert!(region_shares(&counts, 0).is_none());
    }
}
