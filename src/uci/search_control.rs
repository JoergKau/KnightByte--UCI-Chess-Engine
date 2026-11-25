//! Running a search from the command loop.
//!
//! `go` starts the search on a worker thread with an explicit stack, `stop` and
//! `ponderhit` act on it, and every command that needs the searcher or the
//! position first joins a running search. The worker prints the only `bestmove`.

use std::io;
use std::sync::Arc;

use super::EngineExtras;
use super::diagnostics::handle_perft_depth;
use super::go::{GoLimits, parse_go};
use super::info::format_info;
use crate::board::movegen::legal;
use crate::board::position::Position;
use crate::eval::nnue::evaluate_position;
use crate::output::Output;
use crate::search::limits::Limits;
use crate::search::pool::{SearchJob, StopFlag, WorkerPool};
use crate::search::searcher::{SearchEvent, SearchResult, Searcher};
use crate::search::time_manager::TimeControl;
use crate::tb::syzygy;

/// A search worker's join handle: the searcher comes back with the finished
/// search result.
pub(super) type WorkerHandle = std::thread::JoinHandle<(Searcher, SearchResult)>;

/// A search running on the worker thread.
pub(super) struct ActiveSearch {
    /// The flag the UCI loop sets on `stop`; the search polls it every 1024
    /// nodes and before every iteration.
    pub(super) stop: StopFlag,
    /// The worker. It prints exactly one `bestmove` (unless pondering) and
    /// hands the searcher and the search result back through its return
    /// value.
    pub(super) worker: WorkerHandle,
    /// True while this is a ponder search: it answers nothing itself; the
    /// UCI loop decides between `bestmove 0000` (quirk Q5) and the fresh
    /// post-ponder search.
    pub(super) ponder: bool,
    /// The limiters the `go` carried. A ponder search stores them until the
    /// `ponderhit` starts the clock.
    pub(super) limits: GoLimits,
}

/// Stops a running search and takes the searcher back.
///
/// Setting the flag ends the search at the next 1024-node poll; the worker
/// prints its one `bestmove` and returns the searcher — except for a ponder
/// search, which answers nothing itself: a `stop` (or a protocol-breaking
/// command) without `ponderhit` ends it with the null move, exactly like
/// the reference (quirk Q5). With no search running this is a no-op.
///
/// The pool's idle barrier is waited on in every case: the helpers release
/// their shared-table handle at their next poll after the stop, which can
/// lag the main worker's join (the bug-B40 window).
pub(super) fn stop_search(
    active: &mut Option<ActiveSearch>,
    searcher: &mut Option<Searcher>,
    pool: &WorkerPool,
    output: &mut Output,
) {
    if let Some(job) = active.take() {
        use std::sync::atomic::Ordering;
        job.stop.store(true, Ordering::Relaxed);
        let (state, result) = job
            .worker
            .join()
            .expect("the search worker must return the searcher");
        *searcher = Some(state);
        if job.ponder {
            // A stopped ponder search is answered like any stopped search: with
            // the best move of the last completed iteration (the first legal
            // move when none completed). The null move is only the answer for
            // a position without legal moves.
            let answer = if result.best_move.is_none() {
                "0000".to_owned()
            } else {
                result.best_move.to_uci()
            };
            let _ = output.line(&format!("bestmove {answer}"));
        }
    }
    pool.wait_until_idle();
}

/// Stack size of the search worker, in bytes.
///
/// The Windows main thread has one megabyte, and the recursion depth must
/// never depend on the platform. The performance audit's figure is used:
/// 16 MB covers the deepest legal search stacks with a wide margin.
pub(super) const SEARCH_STACK_BYTES: usize = 16 * 1024 * 1024;

/// Searches the current position and prints the UCI transcript.
///
/// The search runs on a dedicated worker thread with an explicit 16 MB
/// stack; the searcher moves into the worker and comes back through the
/// join handle, so its learned tables survive between searches. The UCI
/// loop keeps reading commands while the worker runs.
pub(super) fn handle_go(
    line: &str,
    position: Option<Position>,
    state: &mut Option<Searcher>,
    active: &mut Option<ActiveSearch>,
    pool: &mut WorkerPool,
    extras: &mut EngineExtras,
    output: &mut Output,
) -> io::Result<()> {
    let limits = parse_go(line);

    // `go perft N` runs the node counter instead of a search and prints no
    // `bestmove`, exactly like the reference's HandleGo branch.
    if limits.perft_depth > 0 {
        return handle_perft_depth(limits.perft_depth, position.as_ref(), output);
    }

    let Some(position) = position else {
        output.line("info string no position set")?;
        return answer_without_search(None, output);
    };

    // Opening book: when enabled, inside the book and within the book depth,
    // play the book move immediately. Skipped for `go searchmoves` (the
    // restriction must be honoured) and for pondering (the search predicts the
    // opponent's move).
    if !limits.ponder
        && limits.search_moves.is_empty()
        && extras.book_enabled
        && extras.book.is_loaded()
        && (extras.book_depth == 0 || extras.book_plies < extras.book_depth)
    {
        let book_move = extras.book.pick_book_move(&position, &mut extras.rng);
        if !book_move.is_none() {
            output.line(&format!("info string book move: {}", book_move.to_uci()))?;
            return output.line(&format!("bestmove {}", book_move.to_uci()));
        }
    }

    // Direct tablebase play: a root position inside the tables is answered
    // from the DTZ probe without searching. Default off; the sound path is the
    // search's leaf probe, which keeps repetition handling and finds mates.
    if extras.syzygy_root_probe
        && !limits.ponder
        && limits.search_moves.is_empty()
        && let Some(tables) = syzygy::current()
        && tables.covers(&position)
    {
        let result = tables.probe_root(&position);
        if result.is_checkmate || result.is_stalemate {
            return output.line("bestmove 0000");
        }
        if !result.failed && !result.mv.is_none() {
            output.line(&format!(
                "info string tablebase move: {} (wdl {}, dtz {})",
                result.mv.to_uci(),
                result.wdl,
                result.dtz
            ))?;
            return output.line(&format!("bestmove {}", result.mv.to_uci()));
        }
    }

    // Single-legal-move shortcut (formerly quirk Q8): with exactly one legal
    // move there is nothing to consider, so a TIME-MANAGED search (a clock or
    // `go movetime`) answers at once with a depth-0 info line carrying the
    // static evaluation and runs no search. Searches without a budget —
    // `go infinite`, `go depth/nodes/mate`, and pondering — run normally:
    // an infinite search must wait for `stop`, and an analysis search must
    // report its iterations.
    if !limits.ponder && !limits.infinite && limits.has_time_control() {
        let root_moves = legal(&position);
        if root_moves.len() == 1 {
            let only = root_moves.as_slice()[0];
            output.line(&format!(
                "info depth 0 score cp {} pv {}",
                evaluate_position(&position),
                only.to_uci()
            ))?;
            return output.line(&format!("bestmove {}", only.to_uci()));
        }
    }

    // The worker owns the searcher for the duration of the search; it comes
    // back through the join handle.
    let mut searcher = state.take().expect("a finished search restored the state");

    // While pondering the search has NO time limit: the clock limiters are
    // stored with the job and only start when the `ponderhit` arrives (the
    // step-0 capture, scenario b). Depth and node limiters still apply.
    let search_limits = init_search_limits(&mut searcher, &limits, &position, limits.ponder);

    *active = Some(spawn_active_search(
        searcher,
        position,
        search_limits,
        limits.ponder,
        limits,
        pool,
    )?);
    Ok(())
}

/// Arms the stop flag, launches the helpers and spawns the main
/// worker; returns the active search.
///
/// With `Threads > 1` every helper gets the same job the main worker runs —
/// the same root, the same limits minus the time budgets (only the main
/// thread runs the time manager and reports) — plus the shared tables and a
/// shared node counter. The main thread's result is the answer.
pub(super) fn spawn_active_search(
    mut searcher: Searcher,
    position: Position,
    search_limits: Limits,
    ponder: bool,
    go_limits: GoLimits,
    pool: &mut WorkerPool,
) -> io::Result<ActiveSearch> {
    use std::sync::atomic::AtomicU64;
    // A fresh flag per `go`: the UCI loop keeps the handle for `stop` and
    // the join-first fallback; every search polls it every 1024 nodes. A
    // ponder search answers no bestmove itself; the loop decides its ending.
    let stop = searcher.arm_stop_flag();
    // `searchstats` reports the last counted search: start a fresh snapshot
    // before any thread of this search can merge into it.
    if crate::tools::diagnostics::search_stats_enabled() {
        crate::tools::diagnostics::clear_last_search();
    }
    crate::search::tb::reset_helper_tb_hits();
    if pool.threads() > 1 {
        let helper_limits = Limits {
            soft_time_ms: None,
            hard_time_ms: None,
            ..search_limits.clone()
        };
        let counter = Arc::new(AtomicU64::new(0));
        pool.launch(SearchJob {
            position,
            limits: helper_limits,
            params: *searcher.params(),
            tables: searcher.tables().clone(),
            stop: Arc::clone(&stop),
            helper_nodes: Arc::clone(&counter),
        });
        searcher.attach_helpers(counter);
    }
    let worker = spawn_search_worker(searcher, position, search_limits, ponder, &stop)?;
    Ok(ActiveSearch {
        stop,
        worker,
        ponder,
        limits: go_limits,
    })
}

/// Spawns the main search worker and returns its join handle.
///
/// The worker owns its output writer (both writers lock standard output per
/// line, so replies never interleave mid-line); the first failed write sets
/// the stop flag, so a GUI that walks away cannot leave the search spinning.
/// When the search ends without a completed iteration, the worker prints the
/// interrupted-search depth-0 line (static evaluation and first legal move —
/// the step-0 capture, scenario b; the reference prints it for ponder
/// searches too, and only the bestmove line differs). The final `bestmove`
/// is printed by the worker unless `ponder` is set: a ponder search's ending
/// belongs to the UCI loop (`bestmove 0000` on `stop`, the fresh
/// post-ponder search on `ponderhit`).
pub(super) fn spawn_search_worker(
    mut searcher: Searcher,
    position: Position,
    search_limits: Limits,
    ponder: bool,
    stop: &StopFlag,
) -> io::Result<WorkerHandle> {
    let worker_stop = Arc::clone(stop);
    let worker = std::thread::Builder::new()
        .name("search".to_owned())
        .stack_size(SEARCH_STACK_BYTES)
        .spawn(move || {
            let mut output = Output::new();
            let mut write_failed = false;
            let result = searcher.search(&position, search_limits, &mut |event| {
                if write_failed {
                    return;
                }
                let line_text = match event {
                    SearchEvent::Info(info) => format_info(info),
                    // The reference's single `info currmove` line of
                    // iteration 1 (the step-0 transcripts: one per search,
                    // before the depth-1 line).
                    SearchEvent::CurrMove { mv, number, nodes } => format!(
                        "info currmove {} currmovenumber {number} nodes {nodes}",
                        mv.to_uci()
                    ),
                };
                if output.line(&line_text).is_err() {
                    write_failed = true;
                    worker_stop.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            });
            // The helpers stop at their next 1024-node poll once the
            // flag is set; the main thread's result is the answer.
            worker_stop.store(true, std::sync::atomic::Ordering::Relaxed);
            searcher.detach_helpers();
            if result.depth == 0 {
                // The interrupted-search answer (also for a ponder search):
                // the depth-0 line reports the static evaluation and the
                // first legal move.
                if let Some(first) = legal(&position).as_slice().first().copied() {
                    let _ = output.line(&format!(
                        "info depth 0 score cp {} pv {}",
                        evaluate_position(&position),
                        first.to_uci()
                    ));
                }
            }
            if !ponder {
                if result.depth == 0 {
                    let first = legal(&position).as_slice().first().copied();
                    let _ = output.line(&format!(
                        "bestmove {}",
                        first.map_or_else(|| "0000".to_owned(), |mv| mv.to_uci())
                    ));
                } else {
                    let _ = output.line(&format!("bestmove {}", result.best_move.to_uci()));
                }
            }
            (searcher, result)
        })?;
    Ok(worker)
}

/// Promotes a ponder search to a normal one.
///
/// The ponder worker is joined silently (it answers no bestmove itself); its
/// best move — the ponder move — is applied to the position, the game
/// history grows by one position, and a fresh normal search starts with the
/// clock beginning at this moment (the step-0 capture, scenario b: the time
/// spent pondering is not deducted). A `ponderhit` with no ponder search
/// running is ignored (scenario c), and so is one arriving during a normal
/// search.
///
/// When the ponder search completed no iteration it has no best move to
/// apply (verified against the reference binary, 2026-11-18: a burst
/// `go ponder` + `ponderhit` re-searches the same position and answers its
/// first legal move); the fresh search then runs on the unchanged position.
pub(super) fn handle_ponder_hit(
    active: &mut Option<ActiveSearch>,
    state: &mut Option<Searcher>,
    position: &mut Option<Position>,
    pool: &mut WorkerPool,
    extras: &mut EngineExtras,
    output: &mut Output,
) -> io::Result<()> {
    use std::sync::atomic::Ordering;
    if !active.as_ref().is_some_and(|job| job.ponder) {
        return Ok(());
    }
    let job = active.take().expect("checked above");
    job.stop.store(true, Ordering::Relaxed);
    let (mut searcher, result) = job
        .worker
        .join()
        .expect("the search worker must return the searcher");

    let Some(position) = position.as_ref().copied() else {
        // No position: the null move is the protocol's only answer.
        *state = Some(searcher);
        return output.line("bestmove 0000");
    };
    // The ponder move is the ponder search's best move — its last completed
    // iteration's first PV move. With no completed iteration there is
    // nothing to apply: the fresh search runs on the unchanged position.
    let ponder_move = (result.depth > 0).then_some(result.best_move);
    let pondered = match ponder_move {
        Some(ponder_move) => {
            // The game history grows by the position the ponder move was
            // applied to, so repetitions across the promotion are seen.
            let mut game = searcher.game_history().to_vec();
            game.push(position.key());
            searcher.set_game_history(&game);
            extras.book_plies = extras.book_plies.saturating_add(1);
            position.make_move(ponder_move)
        }
        None => position,
    };

    // The clock starts at the ponderhit moment; the depth and node limiters
    // carry over.
    let search_limits = init_search_limits(&mut searcher, &job.limits, &pondered, false);
    *active = Some(spawn_active_search(
        searcher,
        pondered,
        search_limits,
        false,
        job.limits,
        pool,
    )?);
    Ok(())
}

/// Initialises the time manager from the parsed `go` limiters and returns
/// the search limits.
///
/// While `ponder` is set the clock limiters are deliberately not started:
/// the ponder search runs unbounded until `ponderhit` or `stop`, and the
/// clock begins only when the fresh post-ponder search is started (which
/// calls this helper again with `ponder == false` — its `init_clock` starts
/// the clock at that moment, as the step-0 capture shows).
pub(super) fn init_search_limits(
    searcher: &mut Searcher,
    limits: &GoLimits,
    position: &Position,
    ponder: bool,
) -> Limits {
    if ponder {
        searcher.init_infinite();
    } else if limits.movetime > 0 {
        searcher.init_movetime(limits.movetime);
    } else if limits.has_time_control() {
        let white_to_move = position.side_to_move_is_white();
        searcher.init_clock(&TimeControl {
            time_left_ms: if white_to_move {
                limits.white_time
            } else {
                limits.black_time
            },
            opponent_time_ms: if white_to_move {
                limits.black_time
            } else {
                limits.white_time
            },
            increment_ms: if white_to_move {
                limits.white_inc
            } else {
                limits.black_inc
            },
            moves_to_go: (limits.moves_to_go > 0).then_some(limits.moves_to_go),
        });
    } else if limits.infinite || limits.depth == 0 && limits.nodes == 0 {
        searcher.init_infinite();
    }

    let mut search_limits = Limits {
        ponder,
        ..Limits::default()
    };
    if limits.depth > 0 {
        search_limits.depth = Some(limits.depth);
    }
    if limits.nodes > 0 {
        search_limits.nodes = Some(limits.nodes);
    }
    // `go mate N` stops at the first completed iteration proving a
    // mate within N moves; `go searchmoves` restricts the root to the listed
    // moves that are legal here (an illegal-only list is ignored, verified
    // against the reference).
    if limits.mate > 0 {
        search_limits.mate = Some(limits.mate);
    }
    if !limits.search_moves.is_empty() {
        let all = legal(position);
        search_limits.root_moves = limits
            .search_moves
            .iter()
            .filter_map(|text| {
                all.as_slice()
                    .iter()
                    .find(|mv| mv.to_uci() == text.as_str())
                    .copied()
            })
            .collect();
    }
    search_limits
}

/// Answers a `go` that cannot be searched (an unsupported limiter, a missing
/// position) with a move anyway.
///
/// A GUI waits for a `bestmove` after every `go`; staying silent would leave it
/// waiting until its clock runs out, which is a lost game. The first legal move
/// is a legal, if unconsidered, answer; with no legal move or no position the
/// null move `0000` is the protocol's way to say so.
pub(super) fn answer_without_search(
    position: Option<&Position>,
    output: &mut Output,
) -> io::Result<()> {
    let fallback = position
        .and_then(|position| legal(position).as_slice().first().copied())
        .map_or_else(|| "0000".to_owned(), |mv| mv.to_uci());
    output.line(&format!("bestmove {fallback}"))
}

#[cfg(test)]
mod tests {

    use crate::search::searcher::Searcher;

    #[test]
    fn a_mate_limiter_has_no_depth_cap() {
        use crate::uci::position_cmd::start_position;
        // `go mate N` without another limiter searches without a
        // depth cap until a mate within N is proven or a `stop` arrives
        // (the step-0 capture, scenario b). Other limits still apply.
        let mut searcher = Searcher::new(2);
        let position = start_position();
        let limits = super::init_search_limits(
            &mut searcher,
            &super::parse_go("go mate 3"),
            &position,
            false,
        );
        assert_eq!(limits.mate, Some(3));
        assert_eq!(limits.depth, None);
        assert_eq!(
            limits.depth_limit(),
            crate::search::MAX_PLY as i32,
            "a mate limiter must not cap the depth"
        );
        // A depth limiter given alongside the mate limiter still applies.
        let limits = super::init_search_limits(
            &mut searcher,
            &super::parse_go("go mate 3 depth 5"),
            &position,
            false,
        );
        assert_eq!(limits.mate, Some(3));
        assert_eq!(limits.depth, Some(5));
        assert_eq!(limits.depth_limit(), 5);
    }

    #[test]
    fn searchmoves_resolves_legal_moves_into_the_root_filter() {
        use crate::uci::position_cmd::start_position;
        // The UCI layer resolves the listed moves against the root
        // position's legal moves; an illegal-only list resolves to an empty
        // filter, which the searcher treats as no restriction.
        let mut searcher = Searcher::new(2);
        let position = start_position();
        let limits = super::init_search_limits(
            &mut searcher,
            &super::parse_go("go depth 3 searchmoves e2e4 a1a1 e7e5"),
            &position,
            false,
        );
        // e2e4 is legal; a1a1 is move-shaped but not legal; e7e5 is not a
        // White move. Only the legal ones enter the filter.
        assert_eq!(limits.root_moves.len(), 1);
        assert_eq!(limits.root_moves[0].to_uci(), "e2e4");
        let limits = super::init_search_limits(
            &mut searcher,
            &super::parse_go("go depth 3 searchmoves e7e5"),
            &position,
            false,
        );
        assert!(
            limits.root_moves.is_empty(),
            "an illegal-only list must resolve to no restriction"
        );
    }
}
