//! Helpers shared by the tests of the searcher's files.

use crate::board::movegen::legal;
use crate::board::position::Position;

/// Plays the UCI moves in order and returns every position reached, starting
/// with the given one.
pub(super) fn play(start: &Position, moves: &[&str]) -> Vec<Position> {
    let mut positions = vec![*start];
    for text in moves {
        let current = *positions.last().expect("at least the start");
        let mv = legal(&current)
            .as_slice()
            .iter()
            .copied()
            .find(|mv| mv.to_uci() == *text)
            .unwrap_or_else(|| panic!("{text} is not legal"));
        positions.push(current.make_move(mv));
    }
    positions
}
