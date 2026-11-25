//! UCI command loop.
//!
//! The loop owns the board and the search state and dispatches every command.
//! The commands themselves live in the sibling files:
//!
//! | file | holds |
//! | :--- | :--- |
//! | `parser.rs` | the first word of a line into a command |
//! | `handshake.rs` | the answer to `uci` and `isready` |
//! | `setoption.rs` | `setoption`: validation and application |
//! | `position_cmd.rs` | `position` |
//! | `go.rs` | the `go` limiters |
//! | `search_control.rs` | starting, stopping and pondering a search |
//! | `info.rs` | the `info` line |
//! | `diagnostics.rs` | `perft`, `nnueeval`, `nnueregions`, `searchstats`, `d`, `eval`, `checkmoves` |

use std::io::{self, BufRead};

use self::diagnostics::{
    handle_check_moves, handle_display, handle_eval, handle_nnue_eval, handle_nnue_regions,
    handle_perft, handle_search_stats,
};
use self::handshake::{eval_source_line, handshake};
use self::position_cmd::{parse_position, start_position};
use self::search_control::{ActiveSearch, handle_go, handle_ponder_hit, stop_search};
use self::setoption::{apply_preset_params, handle_set_option};
use crate::board::position::Position;
use crate::output::Output;
use crate::search::searcher::Searcher;
use crate::tb::book::{BookRng, CtgBook};
use crate::tools::bench;

mod banner;
mod diagnostics;
mod go;
mod handshake;
mod info;
pub mod parser;
mod position_cmd;
mod search_control;
mod setoption;

/// Transposition-table size for the command loop, in megabytes.
const DEFAULT_TT_MEGABYTES: usize = 256;

/// Default maximum book depth in plies (0 = no limit).
const DEFAULT_BOOK_DEPTH: u32 = 16;

/// Optional engine features that live outside the search state.
///
/// The opening book keeps its loaded index between commands, and the book
/// depth is counted from the start of the current game position (`position`
/// commands reset it). `SyzygyRootProbe` is the UCI-level direct-play switch;
/// the tablebase tables themselves live in the process-wide set.
struct EngineExtras {
    /// Loaded opening book (empty when none is loaded).
    book: CtgBook,
    /// `Own Book` option.
    book_enabled: bool,
    /// `Book Depth` option, in plies from the game start.
    book_depth: u32,
    /// Plies applied by the last `position` command.
    book_plies: u32,
    /// `SyzygyRootProbe` option.
    syzygy_root_probe: bool,
    /// Weighted book-move selection source.
    rng: BookRng,
}

impl EngineExtras {
    /// Creates the default feature state: book off, no book loaded.
    fn new() -> Self {
        Self {
            book: CtgBook::default(),
            book_enabled: false,
            book_depth: DEFAULT_BOOK_DEPTH,
            book_plies: 0,
            syzygy_root_probe: false,
            rng: BookRng::from_entropy(),
        }
    }
}

/// Runs the UCI event loop over standard input without arguments.
///
/// # Errors
///
/// Returns an I/O error if reading standard input or writing standard output
/// fails. A disconnected GUI is treated as end of input, which lets the
/// process exit cleanly.
pub fn run() -> io::Result<()> {
    run_with_argv(&[])
}

/// Runs the UCI event loop with the process arguments applied.
///
/// The reference's argv surface is reproduced: `param:Name=Value` entries
/// preload validated search-parameter overrides before the first command, and
/// every other argument is joined into a one-shot UCI command line that runs
/// once and exits (a `quit` is appended). With no arguments the loop reads
/// standard input.
///
/// # Errors
///
/// Returns an I/O error when reading or writing fails, and a fail-fast error
/// for a malformed or unknown `param:` override — the reference aborts the
/// process on those rather than silently searching with default parameters.
pub fn run_with_argv(argv: &[String]) -> io::Result<()> {
    // A person at a terminal sees the banner; a GUI or match runner (redirected
    // standard input) does not, so the protocol streams stay clean.
    banner::print_startup_banner();

    // The `param:` prefix is matched case-insensitively, like the reference.
    let is_param = |argument: &String| argument.to_ascii_lowercase().starts_with("param:");
    let params: Vec<String> = argv
        .iter()
        .filter(|argument| is_param(argument))
        .map(|argument| argument["param:".len()..].to_owned())
        .collect();
    let rest: Vec<&String> = argv.iter().filter(|argument| !is_param(argument)).collect();

    if rest.is_empty() {
        run_from(&mut io::stdin().lock(), &params)
    } else {
        // One-shot mode: the arguments are one UCI line, then the engine exits.
        let command_line = rest
            .iter()
            .map(|argument| argument.as_str())
            .collect::<Vec<&str>>()
            .join(" ");
        let script = format!("{command_line}\nquit\n");
        run_from(&mut script.as_bytes(), &params)
    }
}

/// Runs the UCI event loop over `reader` with preset parameter overrides.
///
/// A `go` runs the search on a worker thread (with an explicit 16 MB stack,
/// so the recursion depth never depends on the platform's main-thread stack);
/// the loop keeps reading standard input while it runs. `stop` sets a shared
/// flag and joins, `quit` stops, joins and exits, and every command that
/// touches the searcher or the position first joins a running search (the
/// fallback for GUIs that break the protocol; it never prints a second
/// `bestmove` because the worker prints the only one).
fn run_from<R: BufRead>(reader: &mut R, params: &[String]) -> io::Result<()> {
    let mut output = Output::new();
    // Reference parity (the predecessor's bug-143 fix): the engine starts on
    // the start position, so a `go` before any `position` command searches
    // something sensible instead of answering a null bestmove. The option
    // stays in the type because the answer path below remains as robustness.
    let mut position: Option<Position> = Some(start_position());
    // The searcher lives here between searches. While one runs on the
    // worker, it is `None`: it comes back through the join handle.
    let mut searcher: Option<Searcher> = Some(Searcher::new(DEFAULT_TT_MEGABYTES));
    apply_preset_params(
        params,
        searcher.as_mut().expect("no search started yet"),
        &mut output,
    )?;
    let mut active: Option<ActiveSearch> = None;
    // The persistent helper pool. `Threads = 1` keeps it empty; a
    // `setoption Threads` grows and shrinks it without touching the main
    // searcher's state.
    let mut pool = crate::search::pool::WorkerPool::new();
    // The optional book and root-probe state; the book starts unloaded and
    // disabled (`Own Book` off), and no directory is scanned at startup.
    let mut extras = EngineExtras::new();

    let mut debug_mode = false;
    for line in reader.lines() {
        let line = line?;
        let command = line.trim();
        if command.is_empty() {
            continue;
        }

        match parser::parse(command) {
            parser::Command::Uci => handshake(&extras, &mut output)?,
            // The eval-source line is restated on every `isready` (the step-0
            // transcript shows it before each `readyok`, including during a
            // search): a GUI that attaches late still learns which evaluation
            // the engine will use.
            parser::Command::IsReady => {
                output.line(&eval_source_line())?;
                output.line("readyok")?;
            }
            // Reference parity (the predecessor's bug-143 fix): a new game
            // resets to the start position, so `go` after `ucinewgame` never
            // searches the previous game's position. The game history is
            // reset with it: the start position has no earlier positions.
            parser::Command::UciNewGame => {
                stop_search(&mut active, &mut searcher, &pool, &mut output);
                let state = searcher.as_mut().expect("restored by the join");
                state.clear();
                position = Some(start_position());
                state.set_game_history(&[]);
                extras.book_plies = 0;
            }
            parser::Command::Bench => {
                stop_search(&mut active, &mut searcher, &pool, &mut output);
                // The benchmark runs with the levers as currently configured, so a
                // lever that does nothing prints the default total.
                let params = *searcher.as_ref().expect("restored by the join").params();
                bench::run(command, params, &mut output)?
            }
            parser::Command::SetOption => {
                stop_search(&mut active, &mut searcher, &pool, &mut output);
                handle_set_option(
                    command,
                    searcher.as_mut().expect("restored by the join"),
                    &mut pool,
                    &mut extras,
                    &mut output,
                )?
            }
            parser::Command::Position => {
                stop_search(&mut active, &mut searcher, &pool, &mut output);
                let state = searcher.as_mut().expect("restored by the join");
                match parse_position(command) {
                    Ok((next, earlier_keys)) => {
                        state.set_game_history(&earlier_keys);
                        position = Some(next);
                        extras.book_plies = earlier_keys.len() as u32;
                    }
                    // Reference parity (the predecessor's bug-144/145
                    // decision): a rejected `position` command is reported
                    // loudly and keeps the previous position, so the engine
                    // never silently searches a position the GUI did not ask
                    // for.
                    Err(error) => output.line(&format!("info string {}", error.message()))?,
                }
            }
            parser::Command::Go => {
                stop_search(&mut active, &mut searcher, &pool, &mut output);
                handle_go(
                    command,
                    position,
                    &mut searcher,
                    &mut active,
                    &mut pool,
                    &mut extras,
                    &mut output,
                )?
            }
            // The flag is set and the worker joined: its one `bestmove` is
            // already on the wire (or, for a ponder search, the Q5 null
            // move is printed by the join). With no search running a `stop`
            // is a no-op, exactly like the reference.
            parser::Command::Stop => stop_search(&mut active, &mut searcher, &pool, &mut output),
            // The ponder promotion: the ponder search's best move is applied
            // to the position and a fresh search starts with the clock
            // beginning at this moment (the step-0 capture, scenario b). A
            // `ponderhit` with no ponder search running is ignored.
            parser::Command::PonderHit => handle_ponder_hit(
                &mut active,
                &mut searcher,
                &mut position,
                &mut pool,
                &mut extras,
                &mut output,
            )?,
            parser::Command::Debug => {
                // The flag exists for reference parity; its consumers arrive
                // with the search instrumentation.
                debug_mode = command
                    .split_whitespace()
                    .nth(1)
                    .is_some_and(|value| value.eq_ignore_ascii_case("on"));
            }
            parser::Command::Perft => {
                stop_search(&mut active, &mut searcher, &pool, &mut output);
                handle_perft(command, position.as_ref(), &mut output)?
            }
            parser::Command::NnueEval => handle_nnue_eval(position.as_ref(), &mut output)?,
            parser::Command::Display => handle_display(position.as_ref(), &mut output)?,
            parser::Command::Eval => handle_eval(position.as_ref(), &mut output)?,
            parser::Command::CheckMoves => handle_check_moves(position.as_ref(), &mut output)?,
            parser::Command::NnueRegions => handle_nnue_regions(command, &mut output)?,
            parser::Command::SearchStats => {
                // A finished search may still have helpers merging their
                // counters; wait for them before reading the snapshot.
                if active.is_none() {
                    pool.wait_until_idle();
                }
                handle_search_stats(command, &mut output)?
            }
            // `quit` stops and joins first: the worker finishes its
            // `bestmove`, the helper pool shuts down, and the process exits
            // cleanly.
            parser::Command::Quit => {
                stop_search(&mut active, &mut searcher, &pool, &mut output);
                pool.shutdown();
                break;
            }
            parser::Command::Unknown => {
                // Reference parity: the report names the lowercased command
                // word, not the whole line.
                let word = command.split_whitespace().next().unwrap_or_default();
                output.line(&format!(
                    "info string Unknown command: {}",
                    word.to_ascii_lowercase()
                ))?;
            }
        }
    }
    let _ = debug_mode;
    // End of input behaves like `quit`: a still-running search is stopped,
    // joined and answers its `bestmove`, and the helper pool is shut down,
    // before the process exits.
    stop_search(&mut active, &mut searcher, &pool, &mut output);
    pool.shutdown();

    Ok(())
}
