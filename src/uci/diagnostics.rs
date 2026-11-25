//! The diagnostic commands: `perft`, `nnueeval`, `nnueregions`, `searchstats`,
//! `d`, `eval` and `checkmoves`.

use std::io;
use std::time::Instant;

use crate::board::movegen::{gives_check, legal};
use crate::board::perft::perft;
use crate::board::position::Position;
use crate::eval::nnue::evaluate_position;
use crate::output::Output;
use crate::tools::net::source_name;

/// Handles the diagnostic `perft <depth>` command.
pub(super) fn handle_perft(
    line: &str,
    position: Option<&Position>,
    output: &mut Output,
) -> io::Result<()> {
    let depth = line
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(1);
    handle_perft_depth(depth, position, output)
}

/// Counts perft nodes for the current position at `depth`.
///
/// The reference prints three lines (`Nodes`, `Time`, `NPS`) for both the
/// `perft` command and `go perft`; the elapsed-milliseconds guard keeps a
/// fast count from dividing by zero, the same rule the search's nps uses.
pub(super) fn handle_perft_depth(
    depth: u32,
    position: Option<&Position>,
    output: &mut Output,
) -> io::Result<()> {
    match position {
        Some(position) => {
            let started = Instant::now();
            let nodes = perft(position, depth);
            let millis = started.elapsed().as_millis().max(1);
            let nps = nodes as u128 * 1000 / millis;
            output.line(&format!("Nodes: {nodes}"))?;
            output.line(&format!("Time: {millis}ms"))?;
            output.line(&format!("NPS: {nps}"))
        }
        None => output.line("info string no position set"),
    }
}

/// Handles the diagnostic `nnueeval` command.
pub(super) fn handle_nnue_eval(position: Option<&Position>, output: &mut Output) -> io::Result<()> {
    match position {
        Some(position) => output.line(&format!("nnueeval {}", evaluate_position(position))),
        None => output.line("nnueeval none"),
    }
}

/// Handles the hidden `nnueregions` command: the real-game king-region
/// histogram of accumulator refreshes.
///
/// `reset` clears the counters and starts counting, `off` stops counting
/// (the counts stay readable), and any other form only prints the histogram.
pub(super) fn handle_nnue_regions(line: &str, output: &mut Output) -> io::Result<()> {
    use crate::tools::diagnostics;
    match line.split_whitespace().nth(1) {
        Some("reset") => {
            diagnostics::reset_regions();
            diagnostics::set_region_counting(true);
            output.line("info string nnueregions: counters cleared, counting STARTED")?;
        }
        Some("off") => {
            diagnostics::set_region_counting(false);
            output.line("info string nnueregions: counting STOPPED")?;
        }
        _ => {}
    }
    let regions = crate::eval::nnue::embedded_net().region_count();
    let (counts, total) = diagnostics::region_snapshot(regions);
    output.line(&diagnostics::region_report(
        &counts,
        total,
        diagnostics::region_counting(),
    ))?;
    if let Some(shares) = diagnostics::region_shares(&counts, total) {
        output.line(&shares)?;
    }
    Ok(())
}

/// Handles the hidden `searchstats` command: the counters of the last search
/// that ran with counting on.
///
/// `reset` clears the snapshot and counts the following searches, `off` stops
/// counting, and any other form only prints the snapshot.
pub(super) fn handle_search_stats(line: &str, output: &mut Output) -> io::Result<()> {
    use crate::tools::diagnostics;
    match line.split_whitespace().nth(1) {
        Some("reset") => {
            diagnostics::set_search_stats_enabled(true);
            diagnostics::clear_last_search();
            output.line("info string searchstats: counters cleared, counting STARTED")?;
        }
        Some("off") => {
            diagnostics::set_search_stats_enabled(false);
            output.line("info string searchstats: counting STOPPED")?;
        }
        _ => {}
    }
    let stats = diagnostics::last_search();
    output.line(&stats.report())?;
    output.line(&stats.report_pruning())
}

/// Handles the diagnostic `d` command: position, FEN and zobrist key.
pub(super) fn handle_display(position: Option<&Position>, output: &mut Output) -> io::Result<()> {
    match position {
        Some(position) => {
            output.line("info string")?;
            output.line(&format!("info string FEN: {}", position.to_fen()))?;
            output.line(&format!("info string Zobrist: 0x{:016X}", position.key()))
        }
        None => output.line("info string No position set"),
    }
}

/// Handles the diagnostic `eval` command: the static evaluation in centipawns.
pub(super) fn handle_eval(position: Option<&Position>, output: &mut Output) -> io::Result<()> {
    match position {
        Some(position) => output.line(&format!(
            "info string Evaluation: {} cp (source: nnue net {})",
            evaluate_position(position),
            source_name()
        )),
        None => output.line("info string No position set"),
    }
}

/// Handles the diagnostic `checkmoves` command.
///
/// Every legal move is printed with its gives-check verdict, one line each,
/// terminated by `checkmoves done` — the shape the reference's external
/// validation against an authoritative oracle expects.
pub(super) fn handle_check_moves(
    position: Option<&Position>,
    output: &mut Output,
) -> io::Result<()> {
    let Some(position) = position else {
        output.line("info string No position set")?;
        return output.line("checkmoves done");
    };
    for &mv in legal(position).as_slice() {
        output.line(&format!(
            "checkmove {} givescheck={}",
            mv.to_uci(),
            gives_check(position, mv)
        ))?;
    }
    output.line("checkmoves done")
}
