//! The `position` command: FEN or `startpos` plus a list of moves.

use crate::board::fen::STARTPOS_FEN;
use crate::board::movegen::legal;
use crate::board::position::Position;

/// Why a `position` command was rejected.
///
/// The message texts mirror the fixed reference exactly (the predecessor's
/// bug-144/145 decision, commit bffe98a): a rejected command is reported
/// loudly and the previous position stays, so the engine never silently
/// searches a position the GUI did not ask for.
pub(super) enum PositionError {
    /// The FEN or the command shape is unusable.
    InvalidPosition(String),
    /// A move in the `moves` tail is not legal.
    IllegalMove(String),
}

impl PositionError {
    /// Returns the `info string` text the engine reports for this rejection.
    pub(super) fn message(&self) -> String {
        match self {
            Self::InvalidPosition(detail) => {
                format!("Error: invalid position, previous position kept: {detail}")
            }
            Self::IllegalMove(mv) => format!(
                "Error: illegal move {mv}, position command rejected, previous position kept"
            ),
        }
    }
}

/// Parses `position startpos|fen <fields>` with an optional `moves` tail.
///
/// Returns the final position and the keys of every position that preceded it
/// in the game (the starting position and each intermediate one, oldest
/// first), which the search needs to recognise repetitions of earlier game
/// positions.
pub(super) fn parse_position(line: &str) -> Result<(Position, Vec<u64>), PositionError> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let (mut position, next) = match tokens.get(1).copied() {
        Some("startpos") => (start_position(), 2),
        Some("fen") if tokens.len() >= 8 => {
            let fen = tokens[2..8].join(" ");
            (
                Position::from_fen(&fen)
                    .map_err(|error| PositionError::InvalidPosition(error.to_string()))?,
                8,
            )
        }
        _ => {
            return Err(PositionError::InvalidPosition(
                "command expects 'startpos' or 'fen <six fields>'".to_owned(),
            ));
        }
    };

    let mut earlier_keys = Vec::new();
    match tokens.get(next).copied() {
        Some("moves") => {
            for text in &tokens[next + 1..] {
                earlier_keys.push(position.key());
                position = apply_uci_move(&position, text)?;
            }
        }
        Some(unexpected) => {
            return Err(PositionError::InvalidPosition(format!(
                "unexpected token '{unexpected}' in position command"
            )));
        }
        None => {}
    }
    Ok((position, earlier_keys))
}

/// Returns a fresh start position.
///
/// A `position startpos` command and the `ucinewgame` reset both build the
/// position from the built-in constant, which always parses.
pub(super) fn start_position() -> Position {
    Position::from_fen(STARTPOS_FEN).expect("the built-in start position must parse")
}

/// Applies one UCI move string to `position`.
pub(super) fn apply_uci_move(position: &Position, text: &str) -> Result<Position, PositionError> {
    let moves = legal(position);
    for &mv in moves.as_slice() {
        if mv.to_uci() == text {
            return Ok(position.make_move(mv));
        }
    }
    Err(PositionError::IllegalMove(text.to_owned()))
}
