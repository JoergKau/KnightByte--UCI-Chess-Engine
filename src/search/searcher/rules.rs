//! Draw rules: repetition, the fifty-move rule and insufficient material.

use super::Searcher;
use crate::board::position::Position;
use crate::types::color::{BLACK, Color, WHITE};
use crate::types::piece::{BISHOP, KNIGHT, PAWN, QUEEN, ROOK};

impl Searcher {
    /// Returns whether the current position repeats a position on the path or
    /// has exhausted the fifty-move clock.
    ///
    /// Twofold repetition scores as a draw because the search only needs to know
    /// that the side to move cannot make progress, which is exactly what the
    /// second occurrence proves. The lever keeps this switchable for parity
    /// work. Insufficient material (K vs K, K+N vs K, K+B vs K, and KB vs KB
    /// with same-coloured bishops) draws regardless of the lever.
    pub(crate) fn is_draw(&self, position: &Position, ply: usize) -> bool {
        if position.halfmove_clock() >= 100 || Self::insufficient_material(position) {
            return true;
        }
        if !self.params.use_twofold_repetition_draw {
            return false;
        }
        self.repetition_count(position, ply) > 0
    }

    /// Returns how many earlier positions with the same side to move equal
    /// `position`, looking back along the search path and then into the game
    /// history, as far as the fifty-move clock allows (a capture or pawn move
    /// makes every earlier position unreachable).
    pub(crate) fn repetition_count(&self, position: &Position, ply: usize) -> u32 {
        let key = position.key();
        let clock = usize::from(position.halfmove_clock());
        let reachable = clock.min(ply + self.game_keys.len());
        let mut count = 0;
        let mut back = 2;
        while back <= reachable {
            let found = if back <= ply {
                self.keys[ply - back] == key
            } else {
                // `beyond` plies before the root: 1 is the position just before it.
                let beyond = back - ply;
                self.game_keys[self.game_keys.len() - beyond] == key
            };
            if found {
                count += 1;
            }
            back += 2;
        }
        count
    }

    /// Returns whether neither side can deliver checkmate: king against
    /// king, king plus one knight against king, king plus one bishop against
    /// king, or king plus bishop against king plus bishop with
    /// same-coloured bishops. Any pawn, rook or queen on the board makes the
    /// position sufficient by definition (the reference's early gate —
    /// without it, K+P vs K+B would falsely read as KB vs KB).
    pub(crate) fn insufficient_material(position: &Position) -> bool {
        for piece in [PAWN, ROOK, QUEEN] {
            if position.pieces(WHITE, piece) != 0 || position.pieces(BLACK, piece) != 0 {
                return false;
            }
        }
        let knights = |color: Color| position.pieces(color, KNIGHT).count_ones();
        let bishops = |color: Color| position.pieces(color, BISHOP).count_ones();
        let white_knights = knights(WHITE);
        let black_knights = knights(BLACK);
        let white_bishops = bishops(WHITE);
        let black_bishops = bishops(BLACK);

        // King vs king.
        if white_knights == 0 && black_knights == 0 && white_bishops == 0 && black_bishops == 0 {
            return true;
        }
        // King + knight vs king.
        if (white_knights == 1 && white_bishops == 0 && black_knights == 0 && black_bishops == 0)
            || (black_knights == 1
                && black_bishops == 0
                && white_knights == 0
                && white_bishops == 0)
        {
            return true;
        }
        // King + bishop vs king.
        if (white_bishops == 1 && white_knights == 0 && black_knights == 0 && black_bishops == 0)
            || (black_bishops == 1
                && black_knights == 0
                && white_knights == 0
                && white_bishops == 0)
        {
            return true;
        }
        // King + bishop vs king + bishop on the same square colour.
        if white_knights == 0 && black_knights == 0 && white_bishops == 1 && black_bishops == 1 {
            let white_bishop = crate::board::bitboard::lsb(position.pieces(WHITE, BISHOP));
            let black_bishop = crate::board::bitboard::lsb(position.pieces(BLACK, BISHOP));
            let same_color = ((white_bishop as i32 / 8 + white_bishop as i32 % 8)
                - (black_bishop as i32 / 8 + black_bishop as i32 % 8))
                % 2
                == 0;
            return same_color;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::fen::STARTPOS_FEN;

    use crate::board::position::Position;
    use crate::search::searcher::test_support::play;

    #[test]
    fn a_repetition_of_an_earlier_game_position_is_seen_by_the_search() {
        // The game returns to the start position after four plies, then the
        // search's root is that position and its path plays g1f3 g8f6 again.
        // The node two plies below the root equals the game position two plies
        // before the root, which only the game history can reveal.
        let start = Position::from_fen(STARTPOS_FEN).unwrap();
        let game = play(&start, &["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]);
        let root = game[4];
        let node = game[6];

        let mut searcher = Searcher::new(1);
        searcher.keys[0] = root.key();
        searcher.keys[1] = game[5].key();

        assert_eq!(searcher.repetition_count(&node, 2), 0, "no history yet");
        assert!(!searcher.is_draw(&node, 2));

        let earlier: Vec<u64> = game[..4].iter().map(|position| position.key()).collect();
        searcher.set_game_history(&earlier);
        assert_eq!(searcher.repetition_count(&node, 2), 1);
        assert!(searcher.is_draw(&node, 2));
    }

    #[test]
    fn history_beyond_an_irreversible_move_cannot_repeat() {
        // A pawn move resets the fifty-move clock, so nothing before it counts
        // even when the key happens to match.
        let start = Position::from_fen(STARTPOS_FEN).unwrap();
        let game = play(&start, &["g1f3", "g8f6", "f3g1", "f6g8", "e2e4"]);
        let node = game[5];
        assert_eq!(node.halfmove_clock(), 0);

        let mut searcher = Searcher::new(1);
        searcher.keys[0] = node.key();
        let earlier: Vec<u64> = game[..5].iter().map(|position| position.key()).collect();
        searcher.set_game_history(&earlier);
        assert_eq!(searcher.repetition_count(&node, 0), 0);
    }
}
