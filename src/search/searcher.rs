//! The search state: everything that survives between iterations and between
//! searches.
//!
//! [`Searcher`] owns the transposition table (shared through
//! [`crate::search::tables::SharedTables`]), the history tables, killers, the
//! principal-variation table, the per-ply stacks and the clock, so a search
//! needs no allocation after the constructor. Its behaviour is spread over
//! small files, each an `impl Searcher` block:
//!
//! | file | holds |
//! | :--- | :--- |
//! | `searcher.rs` (this file) | the state, its construction and reset, and the settings |
//! | `searcher/iterative.rs` | `search`: iterative deepening, aspiration windows, the root loop |
//! | `searcher/accumulators.rs` | the per-ply network accumulators and the static evaluation |
//! | `searcher/control.rs` | the stop flag, shared tables, helper node counts and the periodic stop check |
//! | `searcher/rules.rs` | repetition, fifty-move and insufficient-material draws |
//! | `searcher/node.rs` | table probe helpers and the store of a finished node |
//!
//! The node function itself is in [`crate::search::negamax`], quiescence in
//! [`crate::search::quiescence`]. A search runs to completion and reports every
//! finished iteration through a callback.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use crate::eval::nnue::Evaluator;
use crate::search::MAX_PLY;
use crate::search::correction::CorrectionHistory;
use crate::search::history::History;
use crate::search::killers::Killers;
use crate::search::limits::Limits;
use crate::search::move_picker::PickerState;
use crate::search::params::SearchParams;
use crate::search::pv::PvTable;
use crate::search::time_manager::{TimeControl, TimeManager};
use crate::types::r#move::Move;

mod accumulators;
mod control;
mod iterative;
mod node;
mod rules;
#[cfg(test)]
mod test_support;

pub(crate) use self::node::{NodeResult, mated_in, tt_entry};

/// Score of a forced mate at the root.
///
/// Mate scores decrease by one per ply, so a mate in one ply is
/// [`MATE_SCORE`] - 1. Values at or above [`MATE_LIMIT`] are treated as mates.
pub const MATE_SCORE: i32 = 31_000;

/// Boundary above which a score counts as a mate distance.
pub const MATE_LIMIT: i32 = 30_000;

/// Marker for "this node has no usable static evaluation".
///
/// The table's static-eval field is 16 bits; this sentinel cannot collide with
/// a real evaluation after clamping, and quiescence stores it for in-check
/// nodes where no stand-pat exists.
pub const NO_STATIC_EVAL: i32 = i16::MIN as i32;

/// A score larger than any real evaluation.
pub const INFINITE: i32 = 32_000;

/// Score of a drawn position.
pub const DRAW_SCORE: i32 = 0;

/// What the search reports to its observer.
///
/// Two kinds exist, matching the reference's output exactly: a completed
/// iteration (one `info depth …` line) and the single `info currmove` report
/// of iteration 1's first root move (the step-0 transcripts show one such
/// line per search, before the depth-1 line — and none when the search never
/// reached iteration 1).
#[derive(Clone, Debug)]
pub enum SearchEvent<'a> {
    /// A completed iteration.
    Info(&'a SearchInfo),
    /// The first root move of iteration 1 is about to be searched: the move
    /// as the picker yields it, its move number, and the node count at that
    /// moment (1).
    CurrMove {
        /// The move the root's picker yields first.
        mv: Move,
        /// The move's index in the root move loop, counting from one.
        number: u64,
        /// The node count when the move begins.
        nodes: u64,
    },
}

/// Per-depth report emitted after every completed iteration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchInfo {
    /// Completed depth in plies.
    pub depth: i32,
    /// Deepest ply reached by any line of this iteration.
    pub seldepth: usize,
    /// Score in centipawns from the side to move; mate distances use
    /// [`MATE_SCORE`].
    pub score: i32,
    /// Nodes visited so far.
    pub nodes: u64,
    /// Tablebase hits so far, or `None` while no tablebase is in use (the
    /// `info` line then carries no `tbhits` field).
    pub tb_hits: Option<u64>,
    /// Milliseconds elapsed since the search started.
    pub time_ms: u64,
    /// Nodes per second, derived from `nodes` and `time_ms` (guarded against
    /// a zero elapsed time).
    pub nps: u64,
    /// Table occupancy in permill, read once per iteration.
    pub hashfull: u16,
    /// Principal variation of the completed iteration.
    pub pv: Vec<Move>,
}

/// Final result of one search.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchResult {
    /// Best move found, or [`Move::NONE`] when no move is legal.
    pub best_move: Move,
    /// Score of the best move in centipawns from the side to move.
    pub score: i32,
    /// Last completed depth in plies.
    pub depth: i32,
    /// Deepest ply reached.
    pub seldepth: usize,
    /// Number of visited nodes.
    pub nodes: u64,
    /// Principal variation, which may be shorter than `depth` in a forced line.
    pub pv: Vec<Move>,
}

/// Shared search state plus the iterative-deepening driver.
pub struct Searcher {
    pub(crate) tables: crate::search::tables::SharedTables,
    pub(crate) history: History,
    pub(crate) correction: CorrectionHistory,
    pub(crate) killers: Killers,
    pub(crate) pv: PvTable,
    pub(crate) params: SearchParams,
    pub(crate) limits: Limits,
    pub(crate) nodes: u64,
    pub(crate) seldepth: usize,
    pub(crate) stopped: bool,
    /// Position key per ply, used for repetition detection.
    pub(crate) keys: Box<[u64]>,
    /// Keys of the positions that preceded the root in the game, oldest first
    /// (the root itself is `keys[0]`). A repetition of one of them counts like
    /// a repetition inside the search.
    pub(crate) game_keys: Vec<u64>,
    /// Static evaluation per ply, used by the improving heuristic.
    pub(crate) evals: Box<[i32]>,
    /// The move played at each ply (`move_stack[ply]` is the move the node at
    /// `ply` made, leading to `ply + 1`). Indexes the continuation history.
    pub(crate) move_stack: Box<[Move]>,
    /// Legal moves searched so far at each ply; the parent's count gates the
    /// early transposition-table-cutoff history bonus in the child.
    pub(crate) move_counts: Box<[i32]>,
    /// The late-move reduction applied at each ply; the child's hindsight
    /// pair reads the parent's slot.
    pub(crate) reduction_stack: Box<[i32]>,
    /// One staged-picker state per ply (the reference's per-thread per-ply
    /// picker buffers).
    pub(crate) picker: Box<[PickerState]>,
    /// Stateful time manager: allocation, the stop decisions and the
    /// stability trackers.
    pub(crate) time_manager: TimeManager,
    /// Nodes spent per root move, indexed `from * 64 + to`; the time
    /// manager's node-fraction term reads the best move's share.
    pub(crate) nodes_per_move: Box<[u64; 64 * 64]>,
    /// Per-move time overhead subtracted from every budget, in milliseconds.
    pub(crate) move_overhead_ms: u64,
    /// The reusable full-refresh evaluator over the embedded net (its two
    /// accumulator buffers are allocated once per searcher, not per node).
    pub(crate) evaluator: crate::eval::nnue::evaluate::Evaluator<'static>,
    /// Start of the current search, used for the time budgets.
    pub(crate) start: std::time::Instant,
    /// Cooperative stop flag shared with the UCI loop. The loop sets it on
    /// `stop` (and on the join-first fallback); the search polls it every
    /// 1024 nodes and before every iteration. A searcher created without the
    /// UCI layer owns a flag that is never set.
    pub(crate) stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Step 6 (`go searchmoves`): the legal subset of the listed root moves
    /// for the current search. Empty means no restriction. Set once per
    /// search in [`Searcher::search`], read at every root node.
    pub(crate) root_filter: Vec<Move>,
    /// Tablebase set for this search, or `None` when no `SyzygyPath` is set.
    /// Taken once per search so a probe costs no global lock.
    pub(crate) tb: Option<Arc<crate::tb::syzygy::Syzygy>>,
    /// Whether tablebase probing is enabled for this search. Snapshotted from
    /// the process-wide switch at the start, so the node path never reads an
    /// atomic.
    pub(crate) tb_probing: bool,
    /// DTZ-optimal root move of the current search (root ranking, main thread
    /// only); [`Move::NONE`] when the root is not in the tables.
    pub(crate) root_tb_move: Move,
    /// Fifty-move-aware WDL class of the root, when the root probe succeeded.
    pub(crate) root_tb_wdl: Option<i32>,
    /// Tablebase hits of this searcher in the current search: every WDL value
    /// the tables returned for a node, plus the successful root probe.
    pub(crate) tb_hits: u64,
    /// Part of `tb_hits` already flushed to the shared helper total (only
    /// helpers flush).
    pub(crate) published_tb_hits: u64,
    /// Step 8 (Lazy SMP): the shared node counter of a multi-threaded search.
    /// The main thread only READS it (its own count stays exact, which is
    /// what keeps the `Threads=1` node pins bit-identical); each helper
    /// thread WRITES its blocks to it (see `publishes_nodes`). `None` = no
    /// helpers (the single-threaded path).
    pub(crate) helper_nodes: Option<Arc<AtomicU64>>,
    /// Whether this searcher is a helper that publishes its node blocks to
    /// `helper_nodes` (bug B44: the main searcher used to publish its own
    /// count into the counter it reports from, and the helpers never
    /// published at all).
    pub(crate) publishes_nodes: bool,
    /// This searcher's node count already published to the shared counter
    /// (the block frontier; only helpers publish).
    pub(crate) published_nodes: u64,
    /// Whether iteration 1's `info currmove` report was emitted (step 10:
    /// exactly one per search).
    pub(crate) currmove_reported: bool,
    /// Number of completed iterations; the first one always runs to the end.
    pub(crate) completed_depth: i32,
    /// Incremental NNUE accumulator pair per search ply: the pair at `ply`
    /// holds the white/black accumulators of the position searched there.
    /// Pushed from the parent pair for every made move; a perspective whose
    /// own king moved falls back to a full refresh (the feature rows depend
    /// on the own king's region and mirror).
    accumulator_stack: Vec<[Vec<i16>; 2]>,
    /// Whether this search counts the `searchstats` diagnostics. Copied once
    /// per search from the process-wide switch, so the node path reads a
    /// plain field and never an atomic.
    pub(crate) collect_stats: bool,
    /// This searcher's `searchstats` counters for the current search.
    pub(crate) stats: crate::tools::diagnostics::SearchStats,
    /// The late-move-reduction base table for the current `LmrDivisor`.
    pub(crate) lmr_table: Box<crate::search::lmr::LmrTable>,
    /// The divisor `lmr_table` was built from; a changed lever rebuilds it at
    /// the start of the next search.
    lmr_table_divisor: i32,
}

impl Searcher {
    /// Creates a search state with a transposition table of `tt_megabytes`.
    ///
    /// Every per-ply array is heap allocated: a search stack sized for
    /// [`MAX_PLY`] must not live on the main thread's stack on platforms where
    /// that stack is only one megabyte.
    pub fn new(tt_megabytes: usize) -> Self {
        Self {
            tables: crate::search::tables::SharedTables::new(tt_megabytes),
            history: History::new(),
            correction: CorrectionHistory::new(),
            killers: Killers::new(),
            pv: PvTable::new(),
            params: SearchParams::default(),
            limits: Limits::default(),
            nodes: 0,
            seldepth: 0,
            stopped: false,
            keys: vec![0; MAX_PLY + 1].into_boxed_slice(),
            game_keys: Vec::new(),
            evals: vec![0; MAX_PLY + 1].into_boxed_slice(),
            move_stack: vec![Move::NONE; MAX_PLY + 1].into_boxed_slice(),
            move_counts: vec![0; MAX_PLY + 1].into_boxed_slice(),
            reduction_stack: vec![0; MAX_PLY + 1].into_boxed_slice(),
            picker: (0..=MAX_PLY).map(|_| PickerState::new()).collect(),
            time_manager: TimeManager::new(),
            nodes_per_move: Box::new([0; 64 * 64]),
            move_overhead_ms: crate::search::time_manager::DEFAULT_MOVE_OVERHEAD_MS,
            evaluator: Evaluator::new(crate::eval::nnue::embedded_net()),
            start: std::time::Instant::now(),
            stop_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            root_filter: Vec::new(),
            tb: None,
            tb_probing: false,
            root_tb_move: Move::NONE,
            root_tb_wdl: None,
            tb_hits: 0,
            published_tb_hits: 0,
            helper_nodes: None,
            publishes_nodes: false,
            published_nodes: 0,
            currmove_reported: false,
            completed_depth: 0,
            accumulator_stack: Self::accumulator_stack(),
            collect_stats: false,
            stats: crate::tools::diagnostics::SearchStats::new(),
            lmr_table: crate::search::lmr::build_lmr_table(175),
            lmr_table_divisor: 175,
        }
    }

    /// Forgets every position-dependent table.
    ///
    /// `ucinewgame` and each benchmark position call this, which is what makes
    /// benchmark node counts reproducible from run to run.
    pub fn clear(&mut self) {
        self.tables.tt.clear();
        self.tables.eval_cache.clear();
        self.history.clear();
        self.correction.clear();
        self.killers.clear();
        self.pv.reset();
    }

    /// Clears the per-ply stacks and the countermove table at the start of a
    /// search.
    ///
    /// The history tables deliberately persist between searches; everything
    /// that describes the previous search's path does not. The stacks are read
    /// before they are written for the first time (a null-move child reads its
    /// parent's slot), so a stale value left by an earlier search would leak
    /// into this one.
    pub(super) fn reset_stacks(&mut self) {
        self.keys.fill(0);
        self.evals.fill(0);
        self.move_stack.fill(Move::NONE);
        self.move_counts.fill(0);
        self.reduction_stack.fill(0);
        self.history.clear_counters();
    }

    /// Sets the positions that preceded the search root in the game, oldest
    /// first and excluding the root itself.
    ///
    /// The `position ... moves ...` command supplies them. Without them the
    /// search only sees repetitions along its own path and would walk into (or
    /// miss) a repetition of a position from earlier in the game. Only the
    /// positions since the last irreversible move can repeat, so a long history
    /// is trimmed to the last 100 plies.
    pub fn set_game_history(&mut self, keys: &[u64]) {
        let keep = keys.len().min(100);
        self.game_keys.clear();
        self.game_keys.extend_from_slice(&keys[keys.len() - keep..]);
    }

    /// Returns the keys of the positions that preceded the last search's
    /// root, oldest first (see [`Self::set_game_history`]).
    ///
    /// The ponder path reads them to extend the history by one position
    /// when the ponder move is applied.
    pub fn game_history(&self) -> &[u64] {
        &self.game_keys
    }

    /// Returns the current search parameters.
    pub fn params(&self) -> &SearchParams {
        &self.params
    }

    /// Returns mutable access to the search parameters.
    pub fn params_mut(&mut self) -> &mut SearchParams {
        &mut self.params
    }

    /// Returns the transposition-table occupancy in permill.
    pub fn hashfull(&self) -> u16 {
        self.tables.tt.hashfull()
    }

    /// Resizes the transposition table to `megabytes`.
    ///
    /// The exclusive borrow holds because `setoption` arrives when no search
    /// is active (the UCI loop's join-first rule), so no other thread holds
    /// a handle to the shared tables.
    pub fn set_hash(&mut self, megabytes: usize) {
        self.tables.resize(megabytes.clamp(1, 1024));
    }

    /// Sets the per-move time overhead in milliseconds.
    pub fn set_move_overhead(&mut self, milliseconds: u64) {
        self.move_overhead_ms = milliseconds;
    }

    /// Initializes the time manager for a clock-controlled search.
    ///
    /// The search clock starts here, when the `go` command is handled. The
    /// manager stores its limits as absolute milliseconds on that clock, so
    /// the clock must not be restarted afterwards: restarting it in
    /// [`Searcher::search`] shifted every limit by the time since the previous
    /// search began, which made timed searches overrun by about that much.
    pub fn init_clock(&mut self, control: &TimeControl) {
        self.sync_time_levers();
        self.start = std::time::Instant::now();
        self.time_manager
            .init(control, self.move_overhead_ms, self.now_ms());
    }

    /// Initializes the time manager for a fixed `movetime` search (the clock
    /// starts here, see [`Searcher::init_clock`]).
    pub fn init_movetime(&mut self, milliseconds: u64) {
        self.sync_time_levers();
        self.start = std::time::Instant::now();
        self.time_manager
            .set_fixed_time(milliseconds, self.move_overhead_ms, self.now_ms());
    }

    /// Switches the time manager to infinite mode.
    pub fn init_infinite(&mut self) {
        self.sync_time_levers();
        self.start = std::time::Instant::now();
        self.time_manager.set_infinite();
    }

    /// Pushes the time-management levers from the parameters into the
    /// manager. The clock-imbalance scales must be live before
    /// [`TimeManager::init`] runs, because the allocation reads them.
    pub(super) fn sync_time_levers(&mut self) {
        self.time_manager.set_levers(
            self.params.use_stable_time_management,
            self.params.use_clock_imbalance,
            self.params.clock_imbalance_ahead_scale,
            self.params.clock_imbalance_behind_scale,
        );
    }

    /// Milliseconds elapsed since the search's start.
    pub(crate) fn now_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Returns the configured per-move overhead.
    pub fn move_overhead(&self) -> u64 {
        self.move_overhead_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_history_is_trimmed_to_the_last_hundred_plies() {
        let mut searcher = Searcher::new(1);
        let keys: Vec<u64> = (0..250).collect();
        searcher.set_game_history(&keys);
        assert_eq!(searcher.game_keys.len(), 100);
        assert_eq!(searcher.game_keys[0], 150);
        assert_eq!(searcher.game_keys[99], 249);
        searcher.set_game_history(&[]);
        assert!(searcher.game_keys.is_empty());
    }
}
