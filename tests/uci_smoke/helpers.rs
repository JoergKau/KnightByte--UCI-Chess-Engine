//! Process helpers: run the engine on a script, interactively or in two phases, and read
//! numbers out of its output.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Runs the engine on `input` and returns everything it printed.
pub fn run_engine(input: &[u8]) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_knightbyte"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn knightbyte");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(input)
        .expect("write UCI commands");
    let output = child.wait_with_output().expect("wait for knightbyte");
    assert!(output.status.success(), "engine exited with failure");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Sends `commands`, waits until the search answers its `bestmove`, then
/// quits, and returns everything the engine printed.
///
/// This is the GUI pattern: `quit` stops a running search (the reference
/// does the same — a burst `go depth 6` + `quit` gets the depth-0 answer),
/// so a test that wants the full search must wait for the `bestmove` before
/// quitting.
pub fn run_interactively(commands: &[u8]) -> String {
    let lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (mut child, reader) = spawn_with_reader(&lines);
    let mut stdin = child.stdin.take().expect("stdin");
    stdin.write_all(commands).expect("write UCI commands");
    stdin.flush().expect("flush UCI commands");

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let done = lines
            .lock()
            .expect("reader lock")
            .iter()
            .any(|(_, line)| line.starts_with("bestmove "));
        if done || Instant::now() > deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    stdin.write_all(b"quit\n").expect("write quit");
    drop(stdin);
    assert!(child.wait().expect("wait").success(), "engine failed");
    reader.join().expect("reader");
    lines
        .lock()
        .expect("reader lock")
        .iter()
        .map(|(_, line)| line.clone())
        .collect::<Vec<String>>()
        .join("\n")
}

/// Sends `first`, waits for the search's `bestmove`, sends `second`, quits and
/// returns everything the engine printed. For commands that read the result of
/// a finished search.
pub fn run_in_two_phases(first: &[u8], second: &[u8]) -> Vec<String> {
    let lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (mut child, reader) = spawn_with_reader(&lines);
    let mut stdin = child.stdin.take().expect("stdin");
    stdin.write_all(first).expect("write first commands");
    stdin.flush().expect("flush");
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline
        && !lines
            .lock()
            .expect("reader lock")
            .iter()
            .any(|(_, line)| line.starts_with("bestmove "))
    {
        std::thread::sleep(Duration::from_millis(1));
    }
    stdin.write_all(second).expect("write second commands");
    stdin.write_all(b"quit\n").expect("write quit");
    drop(stdin);
    assert!(child.wait().expect("wait").success(), "engine failed");
    reader.join().expect("reader");
    let collected = lines.lock().expect("reader lock");
    collected.iter().map(|(_, line)| line.clone()).collect()
}

/// Returns the `nodes` of the `info depth <depth>` line in an engine transcript.
pub fn nodes_at_depth(stdout: &str, depth: u32) -> u64 {
    let prefix = format!("info depth {depth} ");
    let line = stdout
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no depth {depth} line in {stdout}"));
    field(line, "nodes")
}

/// Returns the move of the `bestmove` line in an engine transcript.
pub fn bestmove_of(stdout: &str) -> &str {
    stdout
        .lines()
        .find_map(|line| line.strip_prefix("bestmove "))
        .unwrap_or_else(|| panic!("no bestmove in {stdout}"))
}

/// Returns the number after `key` in a space-separated report line.
pub fn field(line: &str, key: &str) -> u64 {
    let mut words = line.split_whitespace();
    while let Some(word) = words.next() {
        if word == key {
            return words
                .next()
                .and_then(|value| value.trim_end_matches('%').parse().ok())
                .unwrap_or_else(|| panic!("no number after {key} in {line}"));
        }
    }
    panic!("no {key} in {line}");
}

/// Runs the engine on `input` with `args` and returns (stdout, success).
pub fn run_engine_with_args(args: &[&str], input: &[u8]) -> (String, bool) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_knightbyte"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn knightbyte");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(input)
        .expect("write UCI commands");
    let output = child.wait_with_output().expect("wait for knightbyte");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        output.status.success(),
    )
}

/// Spawns the engine and returns it with a timestamped line reader running.
///
/// The reader records every output line with the instant it arrived, which
/// is what the stop-latency test measures against.
pub fn spawn_with_reader(
    lines: &std::sync::Arc<std::sync::Mutex<Vec<(Instant, String)>>>,
) -> (std::process::Child, std::thread::JoinHandle<()>) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_knightbyte"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn knightbyte");
    let stdout = child.stdout.take().expect("stdout");
    let reader_lines = std::sync::Arc::clone(lines);
    let reader = std::thread::spawn(move || {
        use std::io::BufRead;
        for line in std::io::BufReader::new(stdout).lines() {
            match line {
                Ok(line) => reader_lines
                    .lock()
                    .expect("reader lock")
                    .push((Instant::now(), line)),
                Err(_) => break,
            }
        }
    });
    (child, reader)
}
