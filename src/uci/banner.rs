//! Interactive start-up banner.
//!
//! The banner is meant for a person at a terminal. It is written to standard
//! error and only when standard input is a terminal, so a GUI or a match runner
//! never sees it on either stream: the protocol stream on standard output stays
//! clean, and redirected standard error stays empty.

use std::io::{self, IsTerminal, Write};

use crate::build_info::{DISPLAY_VERSION, ENGINE_AUTHOR, ENGINE_NAME, ENGINE_URL};

/// Writes the banner to standard error when standard input is a terminal.
///
/// The function never fails the process: a closed or unwritable standard error
/// must not stop the engine from serving the UCI protocol.
pub fn print_startup_banner() {
    if !io::stdin().is_terminal() {
        return;
    }
    let mut stderr = io::stderr().lock();
    let _ = writeln!(stderr, "{}", banner_text());
    let _ = stderr.flush();
}

/// Returns the exact banner block, one line per row.
///
/// The block is built from the constants so the banner and the handshake can
/// never drift apart. Every line is 62 columns wide.
pub fn banner_text() -> String {
    let width = 62;
    let body = width - 1;
    let rule = "*".repeat(width);
    let middle = |text: &str| {
        let content = format!("*   {text}");
        format!("{content:<body$}*")
    };
    [
        rule.clone(),
        middle(&format!(
            "{ENGINE_NAME} {DISPLAY_VERSION} by {ENGINE_AUTHOR}"
        )),
        middle(ENGINE_URL),
        rule,
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_banner_is_a_four_line_block_of_equal_width() {
        let text = banner_text();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4);
        for line in &lines {
            assert_eq!(line.chars().count(), 62);
            assert!(line.starts_with('*'));
            assert!(line.ends_with('*'));
        }
        assert_eq!(lines[0], "*".repeat(62));
        assert_eq!(lines[3], "*".repeat(62));
    }

    #[test]
    fn the_banner_names_the_engine_version_and_author() {
        let text = banner_text();
        let wanted = format!("{ENGINE_NAME} {DISPLAY_VERSION} by {ENGINE_AUTHOR}");
        assert!(text.contains(&wanted));
        assert!(text.contains(crate::build_info::ENGINE_URL));
    }
}
