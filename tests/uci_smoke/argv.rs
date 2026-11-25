//! The command-line surface: a one-shot command line and `param:` overrides.

use crate::helpers::run_engine_with_args;

#[test]
fn argv_runs_a_one_shot_command_line() {
    // The reference's argv surface: all non-param arguments are joined into ONE
    // UCI command line, run once, and the engine exits afterwards. A burst
    // `go depth 1` + `quit` stops the search before an iteration completes —
    // verified against the reference binary (2026-11-18): it answers the same
    // depth-0 line.
    let (stdout, success) = run_engine_with_args(&["go depth 1"], b"");
    assert!(success, "one-shot run failed");
    assert!(
        stdout
            .lines()
            .filter(|line| line.starts_with("bestmove "))
            .count()
            == 1,
        "exactly one bestmove expected: {stdout}"
    );
    assert!(
        stdout.contains("info depth 0 score cp "),
        "the stopped one-shot answers the depth-0 line: {stdout}"
    );
}

#[test]
fn argv_param_overrides_are_announced_and_applied() {
    // `param:Name=Value` preloads a search-parameter override and announces it.
    let (stdout, success) = run_engine_with_args(&["param:AspirationBase=33", "uci"], b"");
    assert!(success, "param run failed");
    assert!(
        stdout.contains("info string param-override: AspirationBase=33 (applies to every go)"),
        "missing the param-override announcement"
    );
    assert!(stdout.contains("uciok"), "missing handshake");
}

#[test]
fn argv_param_overrides_fail_fast_on_unknown_names() {
    // The reference aborts on an unknown override rather than silently
    // searching with default parameters.
    let (_stdout, success) = run_engine_with_args(&["param:NoSuchLever=5", "uci"], b"");
    assert!(!success, "an unknown param override must fail fast");
}
