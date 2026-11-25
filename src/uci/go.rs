//! Parsing of the `go` command line into the limiters a search runs under.
//!
//! Only the tokens are interpreted here; resolving them against the position
//! (the clock budget, `searchmoves`) happens in [`super::search_control`].

/// Depth used when `go` names no limiter (reference parity: a bare `go`
/// searches depth 6; the old KnightByte default of 8 was a recorded deviation).
pub(super) const DEFAULT_GO_DEPTH: u32 = 6;

/// Every limiter a `go` command can carry.
///
/// Unset counters are zero, mirroring the reference's "0 = off" convention:
/// the default-depth rule then reads like the reference's, and unknown tokens
/// are silently ignored, so a GUI that sends an extra keyword never loses its
/// search.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct GoLimits {
    /// `go depth N` (0 = not given).
    pub(super) depth: u32,
    /// `go nodes N`.
    pub(super) nodes: u64,
    /// `go movetime N`.
    pub(super) movetime: u64,
    /// `go wtime N`.
    pub(super) white_time: u64,
    /// `go btime N`.
    pub(super) black_time: u64,
    /// `go winc N`.
    pub(super) white_inc: u64,
    /// `go binc N`.
    pub(super) black_inc: u64,
    /// `go movestogo N` (0 = not given).
    pub(super) moves_to_go: u32,
    /// `go mate N` (0 = not given).
    pub(super) mate: u32,
    /// `go infinite`.
    pub(super) infinite: bool,
    /// `go ponder`.
    pub(super) ponder: bool,
    /// `go perft N` (0 = not given): run perft instead of a search.
    pub(super) perft_depth: u32,
    /// `go searchmoves` root-move restriction.
    pub(super) search_moves: Vec<String>,
}

impl GoLimits {
    /// True when a clock limiter (not `movestogo` alone) is present.
    ///
    /// The reference's `HasTimeControl` counts only `movetime` and the two
    /// clocks: `go movestogo 40` alone carries no clock, so it falls through
    /// to the default-depth rule.
    pub(super) fn has_time_control(&self) -> bool {
        self.movetime > 0 || self.white_time > 0 || self.black_time > 0
    }

    /// Applies the reference's default: a `go` with no limiter at all searches
    /// depth 6.
    pub(super) fn apply_default_depth(&mut self) {
        if self.depth == 0
            && !self.has_time_control()
            && !self.infinite
            && self.nodes == 0
            && self.perft_depth == 0
            && self.mate == 0
        {
            self.depth = DEFAULT_GO_DEPTH;
        }
    }
}

/// True when `token` looks like a UCI move (4-5 characters, board squares).
///
/// `go searchmoves` consumes exactly these tokens, so a following limiter ends
/// the list. The fifth character (a promotion piece) is not validated, like the
/// reference.
pub(super) fn is_move_token(token: &str) -> bool {
    let bytes = token.as_bytes();
    let is_square =
        |file: u8, rank: u8| (b'a'..=b'h').contains(&file) && (b'1'..=b'8').contains(&rank);
    (bytes.len() == 4 || bytes.len() == 5)
        && is_square(bytes[0], bytes[1])
        && is_square(bytes[2], bytes[3])
}

/// Parses the limiters of a `go` command into a [`GoLimits`].
///
/// Token matching is case-insensitive and unknown tokens are ignored, exactly
/// like the reference: accepting more input never rejects a valid search. A
/// limiter's value token is consumed even when it does not parse, and the
/// limiter is then simply not set.
pub(super) fn parse_go(line: &str) -> GoLimits {
    let mut limits = GoLimits::default();
    let mut tokens = line
        .split_whitespace()
        .map(str::to_ascii_lowercase)
        .peekable();
    tokens.next(); // the "go" word

    while let Some(token) = tokens.next() {
        match token.as_str() {
            "depth" => {
                if let Some(value) = tokens.next() {
                    limits.depth = value.parse().unwrap_or(0);
                }
            }
            "nodes" => {
                if let Some(value) = tokens.next() {
                    limits.nodes = value.parse().unwrap_or(0);
                }
            }
            "movetime" => {
                if let Some(value) = tokens.next() {
                    limits.movetime = value.parse().unwrap_or(0);
                }
            }
            "wtime" => {
                if let Some(value) = tokens.next() {
                    limits.white_time = value.parse().unwrap_or(0);
                }
            }
            "btime" => {
                if let Some(value) = tokens.next() {
                    limits.black_time = value.parse().unwrap_or(0);
                }
            }
            "winc" => {
                if let Some(value) = tokens.next() {
                    limits.white_inc = value.parse().unwrap_or(0);
                }
            }
            "binc" => {
                if let Some(value) = tokens.next() {
                    limits.black_inc = value.parse().unwrap_or(0);
                }
            }
            "movestogo" => {
                if let Some(value) = tokens.next() {
                    limits.moves_to_go = value.parse().unwrap_or(0);
                }
            }
            "mate" => {
                if let Some(value) = tokens.next() {
                    limits.mate = value.parse().unwrap_or(0);
                }
            }
            "perft" => {
                if let Some(value) = tokens.next() {
                    limits.perft_depth = value.parse().unwrap_or(0);
                }
            }
            "infinite" => limits.infinite = true,
            "ponder" => limits.ponder = true,
            "searchmoves" => {
                // Consume exactly the move-shaped tokens that follow; the first
                // non-move token stays for the main loop, so `go searchmoves
                // e2e4 depth 3` parses both limiters.
                while let Some(next) = tokens.peek() {
                    if !is_move_token(next) {
                        break;
                    }
                    limits.search_moves.push(tokens.next().expect("peeked"));
                }
            }
            // Unknown tokens are ignored (reference parity).
            _ => {}
        }
    }
    limits.apply_default_depth();
    limits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_and_nodes_limiters_parse() {
        let limits = parse_go("go depth 5");
        assert_eq!(limits.depth, 5);
        let limits = parse_go("go nodes 1234");
        assert_eq!(limits.nodes, 1_234);
    }

    #[test]
    fn movetime_becomes_a_fixed_budget() {
        let limits = parse_go("go movetime 100");
        assert_eq!(limits.movetime, 100);
    }

    #[test]
    fn clocks_parse_into_their_fields() {
        let limits = parse_go("go wtime 1000 btime 2000 winc 10 binc 20 movestogo 30");
        assert_eq!(limits.white_time, 1_000);
        assert_eq!(limits.black_time, 2_000);
        assert_eq!(limits.white_inc, 10);
        assert_eq!(limits.black_inc, 20);
        assert_eq!(limits.moves_to_go, 30);
        assert!(limits.has_time_control());
    }

    #[test]
    fn movestogo_alone_is_no_clock_and_defaults_to_depth_six() {
        // Reference parity: `HasTimeControl` does not count `movestogo`, so the
        // default-depth rule fires.
        let limits = parse_go("go movestogo 40");
        assert_eq!(limits.moves_to_go, 40);
        assert!(!limits.has_time_control());
        assert_eq!(limits.depth, 6);
    }

    #[test]
    fn bare_go_defaults_to_depth_six() {
        let limits = parse_go("go");
        assert_eq!(limits.depth, 6);
    }

    #[test]
    fn unknown_tokens_are_ignored() {
        // The reference silently skips what it does not know; a GUI that sends
        // an extra keyword never loses its search.
        let limits = parse_go("go nonsense depth 4 more-nonsense");
        assert_eq!(limits.depth, 4);
    }

    #[test]
    fn limiter_values_are_consumed_even_when_unparsable() {
        // `go depth` with an unparsable value: the value token is consumed and
        // the limiter stays unset, so the default-depth rule fires.
        let limits = parse_go("go depth notanumber");
        assert_eq!(limits.depth, 6);
        let limits = parse_go("go depth");
        assert_eq!(limits.depth, 6);
    }

    #[test]
    fn tokens_are_case_insensitive() {
        let limits = parse_go("GO DEPTH 5 NODES 100 INFINITE");
        assert_eq!(limits.depth, 5);
        assert_eq!(limits.nodes, 100);
        assert!(limits.infinite);
    }

    #[test]
    fn mate_and_ponder_parse() {
        let limits = parse_go("go mate 3");
        assert_eq!(limits.mate, 3);
        assert_eq!(
            limits.depth, 0,
            "a mate limiter suppresses the default depth"
        );
        let limits = parse_go("go ponder");
        assert!(limits.ponder);
        assert_eq!(
            limits.depth, 6,
            "ponder alone does not suppress the default depth"
        );
    }

    #[test]
    fn searchmoves_consumes_move_tokens_until_a_limiter_follows() {
        let limits = parse_go("go searchmoves e2e4 d2d4 depth 3");
        assert_eq!(limits.search_moves, ["e2e4", "d2d4"]);
        assert_eq!(limits.depth, 3);
        // A promotion move is move-shaped; a token that is not a move ends the
        // list and is parsed as a limiter again.
        let limits = parse_go("go searchmoves a7a8q depth 2");
        assert_eq!(limits.search_moves, ["a7a8q"]);
        assert_eq!(limits.depth, 2);
        let limits = parse_go("go searchmoves");
        assert!(limits.search_moves.is_empty());
    }

    #[test]
    fn perft_is_a_go_limiter() {
        let limits = parse_go("go perft 3");
        assert_eq!(limits.perft_depth, 3);
        assert_eq!(limits.depth, 0);
    }

    #[test]
    fn infinite_and_explicit_limiters_suppress_the_default() {
        let limits = parse_go("go infinite");
        assert!(limits.infinite);
        assert_eq!(limits.depth, 0);
        let limits = parse_go("go nodes 1000");
        assert_eq!(limits.depth, 0);
        let limits = parse_go("go movetime 500");
        assert_eq!(limits.depth, 0);
    }

    #[test]
    fn default_construction_matches_the_reference_convention() {
        // Zero means "off" everywhere, which is what makes the default-depth
        // rule read like the reference's.
        let limits = GoLimits::default();
        assert_eq!(limits.depth, 0);
        assert_eq!(limits.nodes, 0);
        assert!(!limits.infinite);
        assert!(!limits.ponder);
    }
}
