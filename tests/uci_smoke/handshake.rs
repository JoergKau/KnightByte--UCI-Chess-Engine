//! The handshake and `setoption` validation against the step-0 transcripts of the
//! predecessor.

use crate::helpers::run_engine;

/// The step-0 handshake transcript, with its header comments.
const HANDSHAKE_TRANSCRIPT: &str = include_str!("../golden/transcripts/handshake.txt");

/// The ten public options KnightByte advertises. The reference capture also
/// lists `EvalFile`, `Contempt` and its own research levers; KnightByte removed
/// those two public options and hides its sixty levers, so only these ten are
/// compared.
const PUBLIC_OPTION_NAMES: [&str; 10] = [
    "Hash",
    "Threads",
    "Move Overhead",
    "Clear Hash",
    "Own Book",
    "Book File",
    "Book Depth",
    "SyzygyPath",
    "SyzygyRootProbe",
    "Syzygy50MoveRule",
];

/// Returns the option name of an `option name <name> type ...` line.
fn option_name(line: &str) -> Option<&str> {
    line.strip_prefix("option name ")
        .and_then(|rest| rest.split(" type ").next())
}

#[test]
fn the_handshake_matches_the_step_zero_transcript() {
    // Every engine line of the captured handshake appears, in order, in
    // KnightByte's reply, except the option lines the catalogue deliberately
    // does not advertise (see PUBLIC_OPTION_NAMES). KnightByte hides its
    // research levers, so the expected lines are matched as an ordered
    // subsequence.
    // Placeholders: the identity lines and the eval-source net name are
    // checked structurally.
    let expected: Vec<String> = HANDSHAKE_TRANSCRIPT
        .lines()
        .filter(|line| {
            !line.is_empty()
                && !line.starts_with('#')
                && !line.starts_with('>')
                && !line.starts_with('<')
                && option_name(line).is_none_or(|name| PUBLIC_OPTION_NAMES.contains(&name))
        })
        .map(|line| match line {
            "id name <engine>" => "id name KnightByte".to_owned(),
            "id author <author>" => "id author JoergKau".to_owned(),
            _ => line.replace("<nnue net>", "nnue net jokausch.nnue"),
        })
        .collect();

    let stdout = run_engine(b"uci\nisready\nucinewgame\nisready\nquit\n");
    let actual: Vec<&str> = stdout.lines().collect();
    let mut cursor = 0;
    for want in &expected {
        let found = actual[cursor..]
            .iter()
            .position(|line| {
                line == want
                    || (want == "id name KnightByte" && line.starts_with("id name KnightByte "))
            })
            .unwrap_or_else(|| panic!("handshake line '{want}' missing or out of order"));
        cursor += found + 1;
    }

    // The advertised surface: exactly the ten public options. The sixty
    // research levers stay settable through `setoption` but are hidden, and the
    // removed `EvalFile` / `Contempt` options must not appear.
    let options: Vec<&str> = actual.iter().filter_map(|line| option_name(line)).collect();
    assert_eq!(
        options.len(),
        PUBLIC_OPTION_NAMES.len(),
        "only the public options are advertised"
    );
    for removed in ["EvalFile", "Contempt"] {
        assert!(
            !options.contains(&removed),
            "{removed} must not be advertised"
        );
    }
    for hidden in ["AspirationBase", "RfpMaxDepth", "UseStableTimeManagement"] {
        assert!(
            !options.contains(&hidden),
            "research lever {hidden} must stay hidden"
        );
    }
    let mut unique = options.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), options.len(), "an option is advertised twice");
}

#[test]
fn setoption_validation_matches_the_step_zero_transcript() {
    // The captured transcript replays ten setoption scenarios; only the
    // unknown name produces a line, everything else is silent (accepted or
    // rejected with the previous value kept). This is the exact behaviour
    // step 2 implements.
    let stdout = run_engine(
        b"setoption name Hash value 32\n\
          setoption name Hash value 999999\n\
          setoption name Hash value 0\n\
          setoption name NoSuchOption value 5\n\
          setoption name Own Book value true\n\
          setoption name Own Book value false\n\
          setoption name Own Book value maybe\n\
          setoption name Clear Hash\n\
          setoption name Threads value 0\n\
          setoption name Threads value 999\n\
          setoption name AspirationBase value 60\n\
          quit\n",
    );
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines,
        vec!["info string Unknown option: NoSuchOption"],
        "the setoption replay must stay silent except for the unknown name"
    );
}
