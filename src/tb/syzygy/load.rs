//! Mapping a table file and parsing its header into the decompression tables.

use std::path::Path;

use super::decode::{first_ei_index, num_tables};
use super::index::{EncInfo, EncShape, FILE_ENC, PIECE_ENC, init_enc_info};
use super::pairs::{Bytes, DTZ, WDL, setup_pairs};
use super::table::{Loaded, MAGIC_DTZ, MAGIC_WDL, TableEntry};

/// Reads the file header of one table kind.
pub(super) fn load_file(path: &Path, entry: &TableEntry, kind: usize) -> Option<Loaded> {
    let file = Bytes::open(path).ok()?;
    let magic = file.u32(0)?;
    let expected = if kind == WDL { MAGIC_WDL } else { MAGIC_DTZ };
    if magic != expected {
        return None;
    }

    let mut loaded = Loaded {
        file: file.clone(),
        ei: vec![EncInfo::default(); 24],
        dtz_map_off: 0,
        dtz_map_idx: [0; 16],
        dtz_flags: [0; 4],
    };

    let split = kind != DTZ && (file.u8(4)? & 0x01) != 0;
    let data_start = 5_u64;
    let mut data_off = data_start;
    let num = num_tables(entry.has_pawns);
    let ei_base = first_ei_index(entry.has_pawns, kind);
    let enc = if !entry.has_pawns {
        PIECE_ENC
    } else {
        // Pawn tables (both WDL and DTZ) are split by pawn file. The rank
        // encoding exists only for the unused DTM tables.
        FILE_ENC
    };
    let shape = EncShape {
        num: entry.num,
        kk_enc: entry.kk_enc,
        pawns0: entry.pawns0,
        pawns1: entry.pawns1,
    };

    let mut tb_size = [[0_u64; 2]; 6];
    for t in 0..num {
        let (first, ei) = init_enc_info(&file, data_off, 0, t, enc, &shape)?;
        tb_size[t][0] = first;
        loaded.ei[ei_base + t] = ei;
        if split {
            let (second, ei) = init_enc_info(&file, data_off, 4, t, enc, &shape)?;
            tb_size[t][1] = second;
            loaded.ei[ei_base + num + t] = ei;
        }
        data_off += (entry.num + 1 + usize::from(entry.has_pawns && entry.pawns1 != 0)) as u64;
    }
    data_off += data_off & 1;

    let mut sizes = [[[0_u64; 3]; 2]; 6];
    for t in 0..num {
        let (pairs, size, flags) = setup_pairs(&file, &mut data_off, tb_size[t][0], kind)?;
        loaded.ei[ei_base + t].precomp = Some(pairs);
        sizes[t][0] = size;
        if kind == DTZ {
            if !entry.has_pawns {
                loaded.dtz_flags[0] = flags;
            } else {
                loaded.dtz_flags[t] = flags;
            }
        }
        if split {
            let (pairs, size, _) = setup_pairs(&file, &mut data_off, tb_size[t][1], kind)?;
            loaded.ei[ei_base + num + t].precomp = Some(pairs);
            sizes[t][1] = size;
        } else if kind != DTZ {
            loaded.ei[ei_base + num + t].precomp = None;
        }
    }

    if kind == DTZ {
        loaded.dtz_map_off = data_off;
        for t in 0..num {
            let flags = if entry.has_pawns {
                loaded.dtz_flags[t]
            } else {
                loaded.dtz_flags[0]
            };
            if flags & 2 != 0 {
                if flags & 16 == 0 {
                    for i in 0..4 {
                        let base = if entry.has_pawns { t * 4 } else { 0 };
                        loaded.dtz_map_idx[base + i] = (data_off + 1 - loaded.dtz_map_off) as u16;
                        data_off += 1 + u64::from(file.u8(data_off)?);
                    }
                } else {
                    data_off += data_off & 1;
                    for i in 0..4 {
                        let base = if entry.has_pawns { t * 4 } else { 0 };
                        loaded.dtz_map_idx[base + i] = (data_off - loaded.dtz_map_off) as u16;
                        data_off += 2 + 2 * u64::from(file.u16(data_off)?);
                    }
                }
            }
        }
        data_off += data_off & 1;
    }

    for t in 0..num {
        let pairs = loaded.ei[ei_base + t].precomp.as_mut()?;
        pairs.index_table = data_off;
        data_off += sizes[t][0][0];
        if split {
            let pairs = loaded.ei[ei_base + num + t].precomp.as_mut()?;
            pairs.index_table = data_off;
            data_off += sizes[t][1][0];
        }
    }
    for t in 0..num {
        let pairs = loaded.ei[ei_base + t].precomp.as_mut()?;
        pairs.size_table = data_off;
        data_off += sizes[t][0][1];
        if split {
            let pairs = loaded.ei[ei_base + num + t].precomp.as_mut()?;
            pairs.size_table = data_off;
            data_off += sizes[t][1][1];
        }
    }
    for t in 0..num {
        data_off = (data_off + 0x3f) & !0x3f_u64;
        let pairs = loaded.ei[ei_base + t].precomp.as_mut()?;
        pairs.data = data_off;
        data_off += sizes[t][0][2];
        if split {
            data_off = (data_off + 0x3f) & !0x3f_u64;
            let pairs = loaded.ei[ei_base + num + t].precomp.as_mut()?;
            pairs.data = data_off;
            data_off += sizes[t][1][2];
        }
    }
    if data_off > file.len() as u64 {
        return None;
    }

    Some(loaded)
}
