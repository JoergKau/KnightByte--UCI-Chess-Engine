//! The persistent helper pool (Lazy SMP): resizing between searches.

use crate::helpers::{run_engine, run_interactively, spawn_with_reader};
use std::io::Write;
use std::time::{Duration, Instant};

#[test]
fn threads_above_one_searches_with_helpers_and_resizes_between_searches() {
    // Step 8: with Threads > 1 every search runs on the helpers too, while
    // the main thread's result stays the answer. Multi-thread node counts
    // are nondeterministic by design (plan 7.3), so the test asserts the
    // protocol shape: every search gets exactly one real bestmove, across
    // pool resizes.
    let lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (mut child, reader) = spawn_with_reader(&lines);
    let mut stdin = child.stdin.take().expect("stdin");

    let mut searches = 0;
    for threads in ["4", "2", "1"] {
        stdin
            .write_all(
                format!("setoption name Threads value {threads}\nposition startpos\ngo depth 6\n")
                    .as_bytes(),
            )
            .expect("write search commands");
        stdin.flush().expect("flush");
        searches += 1;
        // Wait for this search's bestmove (the depth-6 debug search takes
        // about a second; the deadline is generous).
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let bestmoves = lines
                .lock()
                .expect("reader lock")
                .iter()
                .filter(|(_, line)| line.starts_with("bestmove "))
                .count();
            if bestmoves >= searches || Instant::now() > deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    stdin.write_all(b"quit\n").expect("write quit");
    drop(stdin);
    assert!(child.wait().expect("wait").success(), "engine failed");
    reader.join().expect("reader");

    let captured = lines.lock().expect("reader lock").clone();
    let bestmoves: Vec<&str> = captured
        .iter()
        .filter(|(_, line)| line.starts_with("bestmove "))
        .map(|(_, line)| line.as_str())
        .collect();
    assert_eq!(
        bestmoves.len(),
        searches,
        "one bestmove per go: {captured:?}"
    );
    for bestmove in bestmoves {
        assert_ne!(bestmove, "bestmove 0000", "a legal move is answered");
    }
}

#[test]
fn threads_resizing_is_safe_when_no_search_runs() {
    // The stress core of step 9, in miniature: repeated resize cycles with
    // searches interleaved must never panic and always answer.
    let stdout = run_engine(
        b"setoption name Threads value 8\n\
          setoption name Threads value 1\n\
          setoption name Threads value 4\n\
          setoption name Threads value 256\n\
          setoption name Threads value 1\n\
          quit\n",
    );
    assert!(
        stdout.is_empty(),
        "a silent option change answered: {stdout}"
    );
}

#[test]
fn a_hash_resize_after_a_helper_search_survives() {
    // Bug B39 regression (found by the step-9 stress walk): helpers used to
    // keep their SharedTables clones between searches, which blocked the
    // exclusive resize borrow and killed the engine on the next
    // `setoption Hash`. The full sequence must keep the engine alive and
    // searching.
    let stdout = run_interactively(
        b"setoption name Threads value 4\nposition startpos\ngo depth 3\nsetoption name Hash value 16\n",
    );
    assert!(stdout.contains("bestmove "), "the first search answered");
    let stdout = run_engine(
        b"setoption name Threads value 4\nposition startpos\ngo depth 3\nsetoption name Hash value 16\nisready\nquit\n",
    );
    assert!(
        stdout.contains("readyok"),
        "the engine died on the resize after a helper search: {stdout}"
    );
}
