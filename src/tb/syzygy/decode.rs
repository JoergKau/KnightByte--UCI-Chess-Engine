//! Probing a loaded table: the index of a position, the block lookup and the
//! decompression of the stored value.

use super::index::{EncShape, FILE_ENC, PIECE_ENC, leading_pawn};
use super::pairs::{DTZ, WDL};
use super::pos::{Pos, TB_PIECES, TB_WHITE};
use super::table::{TableEntry, TableValue};

/// Returns the number of table parts of one kind.
pub(super) fn num_tables(has_pawns: bool) -> usize {
    if has_pawns { 4 } else { 1 }
}

/// Returns the first encoding-info slot for one kind.
pub(super) fn first_ei_index(has_pawns: bool, kind: usize) -> usize {
    if has_pawns {
        if kind == WDL { 0 } else { 20 }
    } else if kind == WDL {
        0
    } else {
        4
    }
}

/// Probes an already-mapped table.
pub(super) fn probe_loaded(
    entry: &TableEntry,
    key: u64,
    pos: &Pos,
    s: i32,
    kind: usize,
) -> TableValue {
    let loaded = if kind == WDL {
        entry.wdl.as_ref().expect("checked by the caller")
    } else {
        entry.dtz.as_ref().expect("checked by the caller")
    };

    let (flip, bside) = if !entry.symmetric {
        let flip = key != entry.key;
        (flip, pos.turn == flip)
    } else {
        (!pos.turn, false)
    };

    let ei_base = first_ei_index(entry.has_pawns, kind);
    let mut p = [0_i32; TB_PIECES];
    let mut flags = 0_u8;

    if !entry.has_pawns {
        if kind == DTZ {
            flags = loaded.dtz_flags[0];
            if ((flags & 1 != 0) != bside) && !entry.symmetric {
                return TableValue {
                    value: 0,
                    success: -1,
                };
            }
        }
        let ei = &loaded.ei[ei_base + if kind == DTZ { 0 } else { usize::from(bside) }];
        let mut i = 0;
        while i < entry.num {
            i = fill_squares(pos, &ei.pieces, flip, 0, &mut p, i);
        }
        let index = super::index::encode(&mut p, ei, &entry_shape(entry), PIECE_ENC);
        let Some(pairs) = ei.precomp.as_ref() else {
            return TableValue::FAILED;
        };
        let Some((w0, w1)) = super::pairs::decompress_pairs(pairs, index) else {
            return TableValue::FAILED;
        };

        if kind == WDL {
            return TableValue::ok(i32::from(w0) - 2);
        }
        let mut value = i32::from(w0) + (i32::from(w1 & 0x0f) << 8);
        if flags & 2 != 0 {
            let m = super::index::WDL_TO_MAP[(s + 2) as usize];
            let map_index = loaded.dtz_map_idx[m];
            if flags & 16 == 0 {
                let Some(byte) = loaded
                    .file
                    .u8(loaded.dtz_map_off + u64::from(map_index) + value as u64)
                else {
                    return TableValue::FAILED;
                };
                value = i32::from(byte);
            } else {
                let Some(word) = loaded
                    .file
                    .u16(loaded.dtz_map_off + u64::from(map_index) + 2 + 2 * value as u64)
                else {
                    return TableValue::FAILED;
                };
                value = i32::from(word);
            }
        }
        if (flags & super::index::PA_FLAGS[(s + 2) as usize] == 0) || (s & 1) != 0 {
            value *= 2;
        }
        return TableValue::ok(value);
    }

    // Pawn tables.
    let mut i = fill_squares(
        pos,
        &loaded.ei[ei_base].pieces,
        flip,
        if flip { 0x38 } else { 0 },
        &mut p,
        0,
    );
    let shape = entry_shape(entry);
    let t = leading_pawn(&mut p, entry.pawns0, FILE_ENC);
    if kind == DTZ {
        flags = loaded.dtz_flags[t];
        if ((flags & 1 != 0) != bside) && !entry.symmetric {
            return TableValue {
                value: 0,
                success: -1,
            };
        }
    }
    let ei = &loaded.ei[if kind == WDL {
        ei_base + t + 4 * usize::from(bside)
    } else {
        ei_base + t
    }];
    while i < entry.num {
        i = fill_squares(
            pos,
            &ei.pieces,
            flip,
            if flip { 0x38 } else { 0 },
            &mut p,
            i,
        );
    }
    let index = super::index::encode(&mut p, ei, &shape, FILE_ENC);
    let Some(pairs) = ei.precomp.as_ref() else {
        return TableValue::FAILED;
    };
    let Some((w0, w1)) = super::pairs::decompress_pairs(pairs, index) else {
        return TableValue::FAILED;
    };
    if kind == WDL {
        return TableValue::ok(i32::from(w0) - 2);
    }

    let mut value = i32::from(w0) + (i32::from(w1 & 0x0f) << 8);
    if flags & 2 != 0 {
        let m = super::index::WDL_TO_MAP[(s + 2) as usize];
        let map_index = loaded.dtz_map_idx[t * 4 + m];
        if flags & 16 == 0 {
            let Some(byte) = loaded
                .file
                .u8(loaded.dtz_map_off + u64::from(map_index) + value as u64)
            else {
                return TableValue::FAILED;
            };
            value = i32::from(byte);
        } else {
            let Some(word) = loaded
                .file
                .u16(loaded.dtz_map_off + u64::from(map_index) + 2 + 2 * value as u64)
            else {
                return TableValue::FAILED;
            };
            value = i32::from(word);
        }
    }
    if (flags & super::index::PA_FLAGS[(s + 2) as usize] == 0) || (s & 1) != 0 {
        value *= 2;
    }
    TableValue::ok(value)
}

/// Returns the shape descriptor of an entry.
pub(super) fn entry_shape(entry: &TableEntry) -> EncShape {
    EncShape {
        num: entry.num,
        kk_enc: entry.kk_enc,
        pawns0: entry.pawns0,
        pawns1: entry.pawns1,
    }
}

/// Fills `p` with the squares of one piece group.
pub(super) fn fill_squares(
    pos: &Pos,
    pieces: &[u8; TB_PIECES],
    flip: bool,
    mirror: usize,
    p: &mut [i32; TB_PIECES],
    mut i: usize,
) -> usize {
    let mut color = super::pos::color_of_piece(pieces[i] as usize);
    if flip {
        color = if color == TB_WHITE { 0 } else { 1 };
    }
    let mut board = pos.pieces_by_type(color, super::pos::type_of_piece(pieces[i] as usize) as i32);
    while board != 0 {
        let square = board.trailing_zeros() as usize;
        p[i] = (square ^ mirror) as i32;
        i += 1;
        board &= board - 1;
    }
    i
}
