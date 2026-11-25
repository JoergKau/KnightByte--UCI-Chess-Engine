//! The answer to `uci` and `isready`: identity, the public option catalogue
//! and the status lines of the book, the tablebases and the evaluation source.

use std::io;

use super::EngineExtras;
use crate::output::Output;
use crate::search::options::{OptionKind, PUBLIC_OPTIONS};
use crate::tb::book::CtgBook;
use crate::tb::syzygy;
use crate::tools::net::source_name;

/// Sends the UCI identity block, the public option catalogue and `uciok`.
///
/// The three trailing `info string` lines mirror the reference's handshake:
/// the book state, the tablebase state and the evaluation source. Only the
/// public options are advertised; the research levers are hidden from the
/// handshake but stay fully settable through `setoption` and `param:`.
pub(super) fn handshake(extras: &EngineExtras, output: &mut Output) -> io::Result<()> {
    for line in handshake_lines(&extras.book) {
        output.line(&line)?;
    }
    Ok(())
}

/// Builds every line of the UCI handshake reply.
pub(super) fn handshake_lines(book: &CtgBook) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("id name {}", crate::build_info::identity()));
    lines.push(format!("id author {}", crate::build_info::ENGINE_AUTHOR));
    for option in PUBLIC_OPTIONS {
        let mut line = format!(
            "option name {} type {}",
            option.name,
            kind_name(option.kind)
        );
        // Reference parity: a string option prints its (possibly empty)
        // default, a button prints none.
        if option.kind != OptionKind::Button {
            line.push_str(&format!(" default {}", option.default));
        }
        if let (Some(min), Some(max)) = (option.min, option.max) {
            line.push_str(&format!(" min {min} max {max}"));
        }
        lines.push(line);
    }
    // Reference parity: the book, tablebase and eval-source state lines close
    // the handshake, before `uciok`.
    lines.push(book_status_line(book));
    lines.push(syzygy_status_line());
    lines.push(eval_source_line());
    lines.push("uciok".to_owned());
    lines
}

/// The `info string` describing the loaded opening book.
pub(super) fn book_status_line(book: &CtgBook) -> String {
    if book.is_loaded() {
        format!(
            "info string opening book loaded: {} ({} positions)",
            book.book_path(),
            book.entry_count()
        )
    } else {
        "info string no opening book loaded".to_owned()
    }
}

/// The `info string` describing the loaded tablebases.
pub(super) fn syzygy_status_line() -> String {
    let (tables, max_pieces) = syzygy::status();
    if max_pieces > 0 {
        format!("info string syzygy tablebases: {tables} WDL tables, up to {max_pieces} pieces")
    } else {
        "info string syzygy tablebases: none (set SyzygyPath)".to_owned()
    }
}

/// The `info string` naming the evaluation source (the embedded net file).
///
/// The handshake and every `isready` reply carry it.
pub(super) fn eval_source_line() -> String {
    format!("info string eval-source: nnue net {}", source_name())
}

/// Returns the UCI type word for an option kind.
pub(super) fn kind_name(kind: OptionKind) -> &'static str {
    match kind {
        OptionKind::Spin => "spin",
        OptionKind::Check => "check",
        OptionKind::String => "string",
        OptionKind::Button => "button",
    }
}

#[cfg(test)]
mod tests {
    use super::EngineExtras;
    use super::*;

    #[test]
    fn the_handshake_carries_the_reference_info_lines() {
        let lines = handshake_lines(&EngineExtras::new().book);
        let last = lines.last().map(String::as_str).expect("non-empty");
        assert_eq!(last, "uciok");
        assert!(lines.contains(&"info string no opening book loaded".to_owned()));
        assert!(lines.contains(&"info string syzygy tablebases: none (set SyzygyPath)".to_owned()));
        assert!(lines.contains(&eval_source_line()));
    }

    #[test]
    fn only_public_options_are_advertised() {
        let lines = handshake_lines(&EngineExtras::new().book);
        let advertised: Vec<&str> = lines
            .iter()
            .filter_map(|line| line.strip_prefix("option name "))
            .filter_map(|rest| rest.split(" type ").next())
            .collect();
        assert_eq!(advertised.len(), PUBLIC_OPTIONS.len());
        for lever in crate::search::lever_table::LEVERS {
            assert!(
                !advertised.contains(&lever.name),
                "lever {} must stay hidden",
                lever.name
            );
        }
    }
}
