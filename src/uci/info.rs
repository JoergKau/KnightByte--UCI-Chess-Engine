//! Formatting of the `info` line the search reports after each iteration.

use crate::search::searcher::{MATE_LIMIT, MATE_SCORE, SearchInfo};

/// Formats one `info` line for a completed iteration.
///
/// Field order and set mirror the captured reference lines exactly:
/// `depth`, `seldepth`, `score`, `nodes`, `nps`, `hashfull`, `time`, `pv`.
/// The reference never prints `tbhits`. KnightByte adds `tbhits <n>` after
/// `hashfull` only while a tablebase set is in use, so the line of a search
/// without tablebases is byte-identical to the reference's.
pub(super) fn format_info(info: &SearchInfo) -> String {
    let pv = info
        .pv
        .iter()
        .map(|mv| mv.to_uci())
        .collect::<Vec<String>>()
        .join(" ");
    let tb_hits = info
        .tb_hits
        .map_or_else(String::new, |hits| format!(" tbhits {hits}"));
    format!(
        "info depth {} seldepth {} score {} nodes {} nps {} hashfull {}{} time {} pv {}",
        info.depth,
        info.seldepth,
        format_score(info.score),
        info.nodes,
        info.nps,
        info.hashfull,
        tb_hits,
        info.time_ms,
        pv
    )
}

/// Formats a score as `cp` or as a mate distance in moves.
pub(super) fn format_score(score: i32) -> String {
    if score.abs() >= MATE_LIMIT {
        let plies = MATE_SCORE - score.abs();
        let moves = (plies + 1) / 2;
        if score > 0 {
            format!("mate {moves}")
        } else {
            format!("mate -{moves}")
        }
    } else {
        format!("cp {score}")
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn scores_format_as_cp_and_mate_distances() {
        use crate::search::searcher::{MATE_LIMIT, MATE_SCORE};
        // Centipawn scores, both signs.
        assert_eq!(super::format_score(45), "cp 45");
        assert_eq!(super::format_score(-123), "cp -123");
        // Mate distances round up to whole moves: two plies are mate in one.
        assert_eq!(super::format_score(MATE_SCORE - 2), "mate 1");
        assert_eq!(super::format_score(-(MATE_SCORE - 2)), "mate -1");
        // An odd ply count still rounds up (three plies are mate in two).
        assert_eq!(super::format_score(MATE_SCORE - 3), "mate 2");
        // Just below the mate limit is still a centipawn score.
        assert_eq!(super::format_score(MATE_LIMIT - 1), "cp 29999");
    }

    #[test]
    fn the_info_line_carries_every_reference_field_in_order() {
        use crate::search::searcher::{MATE_SCORE, SearchInfo};
        // A captured-shaped iteration: the line must read exactly like the
        // reference's `depth seldepth score nodes nps hashfull time pv`.
        let info = SearchInfo {
            depth: 2,
            seldepth: 3,
            score: 46,
            nodes: 213,
            tb_hits: None,
            time_ms: 2,
            nps: 106_500,
            hashfull: 4,
            pv: vec![],
        };
        assert_eq!(
            super::format_info(&info),
            "info depth 2 seldepth 3 score cp 46 nodes 213 nps 106500 hashfull 4 time 2 pv "
        );
        // A mate score keeps the same field set.
        let mate = SearchInfo {
            score: MATE_SCORE - 2,
            ..info.clone()
        };
        let line = super::format_info(&mate);
        assert!(line.contains("score mate 1"), "mate score format: {line}");
        assert!(line.contains("hashfull 4"), "hashfull missing: {line}");
        assert!(line.contains("time 2"), "time missing: {line}");

        // `tbhits` joins the line, between `hashfull` and `time`, only while a
        // tablebase set is in use; zero hits still print.
        let with_tables = SearchInfo {
            tb_hits: Some(7),
            ..info.clone()
        };
        assert_eq!(
            super::format_info(&with_tables),
            "info depth 2 seldepth 3 score cp 46 nodes 213 nps 106500 hashfull 4 tbhits 7 time 2 pv "
        );
        let none_yet = SearchInfo {
            tb_hits: Some(0),
            ..info
        };
        assert!(super::format_info(&none_yet).contains("hashfull 4 tbhits 0 time 2"));
    }
}
