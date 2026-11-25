//! `go mate N` and `go searchmoves`.

use crate::helpers::{bestmove_of, field, nodes_at_depth, run_interactively, spawn_with_reader};
use std::io::Write;
use std::time::Duration;

/// The mate-in-one position of the step-0 capture (`a1a8` is the mate).
const MATE_IN_ONE_FEN: &str = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1";

#[test]
fn go_mate_stops_at_the_first_iteration_proving_the_mate() {
    // The step-0 capture (scenario a): `go mate 1` stops at the first
    // completed iteration whose score proves a mate within one move —
    // depth 1, `score mate 1 pv a1a8`. The node count is not pinned here: it
    // follows the search, and the bench pins are the one place that records
    // it. The FEN is verified against the external oracle at test time, per
    // the project rules.
    crate::common::verify_fen(MATE_IN_ONE_FEN);
    let stdout =
        run_interactively(&format!("position fen {MATE_IN_ONE_FEN}\ngo mate 1\n").into_bytes());
    let depth_lines: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with("info depth "))
        .collect();
    assert_eq!(depth_lines.len(), 1, "only depth 1 runs: {stdout}");
    let line = depth_lines[0];
    assert!(
        line.starts_with("info depth 1 seldepth 1 score mate 1 nodes "),
        "the mate-in-one answer must be a depth-1 mate score: {line}"
    );
    assert!(field(line, "nodes") > 0, "{line}");
    assert!(line.ends_with("pv a1a8"), "the mate move is the PV: {line}");
    assert!(stdout.contains("bestmove a1a8"), "missing bestmove");
}

#[test]
fn go_mate_without_a_provable_mate_searches_unbounded() {
    // The step-0 capture (scenario b): with no provable mate in N the
    // search runs on without a depth cap until stopped; a stop answers the
    // last completed iteration's bestmove (a normal search's ending — not
    // the Q5 null move). The load-tolerant core: no bestmove while the
    // search runs (a depth-capped `go` would have answered by itself; the
    // depth-cap lift is pinned deterministically by
    // `a_mate_limiter_has_no_depth_cap` in the UCI module), and one real
    // bestmove after the stop.
    let lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (mut child, reader) = spawn_with_reader(&lines);
    let mut stdin = child.stdin.take().expect("stdin");
    stdin
        .write_all(b"position startpos\ngo mate 3\n")
        .expect("write go mate");
    stdin.flush().expect("flush");
    std::thread::sleep(Duration::from_millis(600));
    assert!(
        !lines
            .lock()
            .expect("reader lock")
            .iter()
            .any(|(_, line)| line.starts_with("bestmove ")),
        "the mate search must run until stopped"
    );
    stdin.write_all(b"stop\nquit\n").expect("write stop");
    drop(stdin);
    assert!(child.wait().expect("wait").success(), "engine failed");
    reader.join().expect("reader");
    let captured = lines.lock().expect("reader lock").clone();
    let bestmoves: Vec<&str> = captured
        .iter()
        .filter(|(_, line)| line.starts_with("bestmove "))
        .map(|(_, line)| line.as_str())
        .collect();
    assert_eq!(bestmoves.len(), 1, "exactly one bestmove: {captured:?}");
    assert_ne!(bestmoves[0], "bestmove 0000", "a real move is answered");
}

#[test]
fn searchmoves_restricts_the_root_moves() {
    // The step-0 capture (scenario c, cold table): the restricted search is
    // a normal search over the listed root moves only. Its trees are smaller
    // than the unrestricted search's at every depth; no node count, score or
    // timing field is pinned (they follow the search; the bench pins are the
    // one place that records them).
    let plain = run_interactively(b"position startpos\ngo depth 5\n");
    let stdout = run_interactively(b"position startpos\ngo depth 5 searchmoves e2e4\n");
    let depth_lines: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with("info depth "))
        .collect();
    assert!(depth_lines.len() >= 5, "depth 5 not reached: {stdout}");
    for depth in 1..=5 {
        assert!(
            nodes_at_depth(&stdout, depth) < nodes_at_depth(&plain, depth),
            "the root was not restricted at depth {depth}: {stdout}"
        );
    }
    // Every iteration's PV starts with the listed move, and the bestmove is
    // the listed move.
    assert!(
        depth_lines
            .iter()
            .all(|line| line.contains(" pv e2e4") || line.contains(" pv e2e4 ")),
        "every PV starts with e2e4: {stdout}"
    );
    assert!(stdout.contains("bestmove e2e4"), "missing bestmove");
}

#[test]
fn searchmoves_with_two_moves_stays_inside_the_list() {
    // Two listed moves: the answer and the first move of every iteration's
    // principal variation must be one of them. (No node count is compared: the
    // shape of a two-move tree against the unrestricted one follows the
    // pruning rules and proves nothing about the restriction.)
    let stdout = run_interactively(b"position startpos\ngo depth 5 searchmoves e2e4 d2d4\n");
    let bestmove = bestmove_of(&stdout);
    assert!(
        bestmove == "e2e4" || bestmove == "d2d4",
        "the bestmove must be a listed move: {bestmove}"
    );
    let depth_lines: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with("info depth "))
        .collect();
    assert!(depth_lines.len() >= 5, "depth 5 not reached: {stdout}");
    assert!(
        depth_lines
            .iter()
            .all(|line| line.contains(" pv e2e4") || line.contains(" pv d2d4")),
        "every PV starts with a listed move: {stdout}"
    );
}

#[test]
fn searchmoves_without_moves_or_with_illegal_moves_is_ignored() {
    // The step-0 capture (scenario e) and the reference probe (2026-11-18):
    // an empty `searchmoves` and a list whose every entry is illegal both
    // run the normal unrestricted search: same node count and same best move
    // as a plain `go depth 5` on the same position.
    let plain = run_interactively(b"position startpos\ngo depth 5\n");
    for go in ["go depth 5 searchmoves", "go depth 5 searchmoves e7e5"] {
        let stdout = run_interactively(&format!("position startpos\n{go}\n").into_bytes());
        assert_eq!(
            nodes_at_depth(&stdout, 5),
            nodes_at_depth(&plain, 5),
            "the unrestricted depth-5 search must run: {stdout}"
        );
        assert_eq!(bestmove_of(&stdout), bestmove_of(&plain), "{go}");
    }
}
