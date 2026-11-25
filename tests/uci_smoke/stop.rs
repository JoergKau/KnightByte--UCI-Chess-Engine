//! Stopping a search: latency, repeated cycles, `quit`, `isready` during a search, a
//! single legal move, and commands that touch the state while a search runs.

use crate::helpers::{run_engine, run_engine_with_args, run_interactively, spawn_with_reader};
use std::io::Write;
use std::time::{Duration, Instant};

#[test]
fn stop_answers_go_infinite_within_100_ms() {
    let lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (mut child, reader) = spawn_with_reader(&lines);
    let mut stdin = child.stdin.take().expect("stdin");
    stdin
        .write_all(b"position startpos\ngo infinite\n")
        .expect("write go");
    stdin.flush().expect("flush go");
    // Let the search reach a couple of iterations before stopping.
    std::thread::sleep(Duration::from_millis(300));

    let stopped = Instant::now();
    stdin.write_all(b"stop\n").expect("write stop");
    stdin.flush().expect("flush stop");

    // The 100 ms budget is the plan's release-mode requirement: the stop
    // flag is polled every 1024 nodes, which costs about 3 ms at release
    // speed (measured 0.1-3 ms on the release binary). A debug build runs
    // ~100x slower and the full test suite runs several engines in
    // parallel, so the budget scales with the build; the mechanism (a stop
    // ends the search promptly) is what is under test.
    let budget = if cfg!(debug_assertions) {
        Duration::from_secs(10)
    } else {
        Duration::from_millis(100)
    };
    let deadline = stopped + budget;
    let mut bestmove_time = None;
    while Instant::now() < deadline {
        let found = lines
            .lock()
            .expect("reader lock")
            .iter()
            .find(|(_, line)| line.starts_with("bestmove "))
            .map(|(instant, _)| *instant);
        if let Some(instant) = found {
            bestmove_time = Some(instant);
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let bestmove_time = bestmove_time.expect("no bestmove within the stop budget of stop");
    assert!(
        bestmove_time.duration_since(stopped) < budget,
        "the bestmove took {:?} after stop",
        bestmove_time.duration_since(stopped)
    );

    stdin.write_all(b"quit\n").expect("write quit");
    drop(stdin);
    assert!(child.wait().expect("wait").success(), "engine failed");
    reader.join().expect("reader");
    // Exactly one bestmove for the one go.
    assert_eq!(
        lines
            .lock()
            .expect("reader lock")
            .iter()
            .filter(|(_, line)| line.starts_with("bestmove "))
            .count(),
        1
    );
}

#[test]
fn two_hundred_go_stop_cycles_answer_exactly_one_bestmove_each() {
    // Every go gets exactly one bestmove, whether the stop arrives before
    // depth 1 finishes (a depth-0 answer) or during a later iteration. The
    // cycles walk depths 1 to 6, as the step-4 test prescription asks.
    let mut script = String::from("position startpos\n");
    let mut cycles = 0;
    for depth in 1..=6 {
        for _ in 0..(200 / 6 + usize::from(depth <= 2)) {
            script.push_str(&format!("go depth {depth}\nstop\n"));
            cycles += 1;
        }
    }
    script.push_str("quit\n");

    let stdout = run_engine_with_args(&[], script.as_bytes()).0;
    assert_eq!(cycles, 200);
    assert_eq!(
        stdout
            .lines()
            .filter(|line| line.starts_with("bestmove "))
            .count(),
        200,
        "every go/stop cycle must answer exactly one bestmove"
    );
}

#[test]
fn quit_during_a_search_exits_cleanly() {
    let stdout = run_engine(b"position startpos\ngo depth 30\nquit\n");
    assert!(
        stdout
            .lines()
            .filter(|line| line.starts_with("bestmove "))
            .count()
            == 1,
        "quit during a search must still answer the go exactly once"
    );
}

#[test]
fn isready_is_answered_during_a_search() {
    // The step-0 capture (isready_during_search.txt): readyok comes at once,
    // preceded by the eval-source line, while the search runs.
    let stdout = run_engine(b"position startpos\ngo depth 20\nisready\nquit\n");
    assert!(stdout.contains("readyok"), "readyok missing");
    assert!(stdout.contains("eval-source"), "eval-source line missing");
    assert!(
        stdout.lines().any(|line| line.starts_with("bestmove ")),
        "missing bestmove"
    );
}

/// The only legal move of this position is Kb1 (the FEN is verified against
/// the external oracle at test time, per the project rules).
const FORCED_MOVE_FEN: &str = "7k/8/8/8/8/1p6/r7/K7 w - - 0 1";

#[test]
fn a_single_legal_move_is_answered_at_once_under_a_clock() {
    // Single-legal-move shortcut (formerly quirk Q8): with a clock or
    // `go movetime` there is nothing to consider, so the move is answered with
    // a depth-0 info line carrying the static evaluation and no search runs.
    crate::common::verify_fen(FORCED_MOVE_FEN);
    let stdout = run_interactively(
        format!("position fen {FORCED_MOVE_FEN}\ngo wtime 60000 btime 60000\n").as_bytes(),
    );
    let depth_lines: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with("info depth "))
        .collect();
    assert_eq!(depth_lines.len(), 1, "exactly the depth-0 line: {stdout}");
    assert!(
        depth_lines[0].starts_with("info depth 0 score cp "),
        "the depth-0 line carries the static evaluation: {}",
        depth_lines[0]
    );
    assert!(
        depth_lines[0].ends_with("pv a1b1"),
        "the only move: {stdout}"
    );
    assert!(
        stdout.contains("bestmove a1b1"),
        "the only move is the bestmove: {stdout}"
    );
}

#[test]
fn a_single_legal_move_is_searched_without_a_time_budget() {
    // Quirk Q8 fix (2026-11-19): a fixed-depth analysis of a forced move
    // reports its iterations instead of the instant depth-0 answer.
    crate::common::verify_fen(FORCED_MOVE_FEN);
    let stdout =
        run_interactively(format!("position fen {FORCED_MOVE_FEN}\ngo depth 4\n").as_bytes());
    assert!(
        stdout.contains("info depth 1 "),
        "iteration lines expected: {stdout}"
    );
    assert!(
        !stdout.contains("info depth 0 "),
        "no instant depth-0 answer without a time budget: {stdout}"
    );
    assert!(stdout.contains("bestmove a1b1"), "{stdout}");
}

#[test]
fn go_infinite_on_a_single_legal_move_does_not_answer_instantly() {
    // Quirk Q8 fix (2026-11-19): `go infinite` must keep searching, not hand
    // back a bestmove at once (UCI: do not leave an infinite search until
    // told to). The search here ends by itself only because the position is a
    // forced loss; what matters is that iterations are reported first.
    crate::common::verify_fen(FORCED_MOVE_FEN);
    let stdout =
        run_interactively(format!("position fen {FORCED_MOVE_FEN}\ngo infinite\n").as_bytes());
    assert!(
        stdout.contains("info depth 1 "),
        "iteration lines expected: {stdout}"
    );
    assert!(
        !stdout.contains("info depth 0 "),
        "no instant depth-0 answer for an infinite search: {stdout}"
    );
}

#[test]
fn a_stop_before_depth_1_finishes_answers_depth_zero() {
    // The step-0 capture (stop.txt, scenario b): a stop that arrives before
    // depth 1 completes gets `info depth 0 score cp <static-eval> pv
    // <first-legal-move>` and the first legal move as the bestmove — the
    // start position evaluates 27 and generates a2a3 first.
    let stdout = run_engine(b"position startpos\ngo depth 1\nstop\nquit\n");
    let depth_lines: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with("info depth "))
        .collect();
    assert_eq!(
        depth_lines,
        vec!["info depth 0 score cp 27 pv a2a3"],
        "the stop-during-depth-1 answer must match the capture: {stdout}"
    );
    assert!(stdout.contains("bestmove a2a3"), "missing bestmove");
}

#[test]
fn a_command_touching_the_state_joins_the_running_search() {
    // The fallback for GUIs that break the protocol: `position` during a
    // running search stops and joins it first (one bestmove on the wire),
    // then applies. The `d` output proves the new position is live.
    let stdout =
        run_engine(b"position startpos\ngo depth 30\nposition startpos moves e2e4\nd\nquit\n");
    assert_eq!(
        stdout
            .lines()
            .filter(|line| line.starts_with("bestmove "))
            .count(),
        1,
        "the joined search answers once; no second bestmove"
    );
    assert!(
        stdout
            .lines()
            .any(|line| line.starts_with("info string FEN: rnbqkbnr/pppppppp/8/8/4P3")),
        "the position command was not applied after the join: {stdout}"
    );
}
