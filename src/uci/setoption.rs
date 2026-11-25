//! The `setoption` command: validating a name and a value and applying it to
//! the searcher, the worker pool or the optional features.
//!
//! Only an unknown option name is reported; every value-level rejection is
//! silent and keeps the previous value, as in the step-0 capture of the
//! predecessor.

use std::io;

use super::EngineExtras;
use super::handshake::{book_status_line, syzygy_status_line};
use crate::output::Output;
use crate::search::lever_table::LeverSpec;
use crate::search::options::{OptionKind, UciOption, lever, public, set};
use crate::search::pool::WorkerPool;
use crate::search::searcher::Searcher;
use crate::tb::book::find_default_book_path;
use crate::tb::syzygy;

/// Applies `param:Name=Value` startup overrides to the searcher.
///
/// The store and the validation mirror the reference: the name must be a known
/// search lever, a check lever accepts only `true`/`false`, a spin lever only
/// an integer, and a malformed or unknown spec is a fail-fast error (the
/// reference aborts the process rather than searching with default parameters).
/// Every accepted override is announced, because it changes the search.
pub(super) fn apply_preset_params(
    specs: &[String],
    searcher: &mut Searcher,
    output: &mut Output,
) -> io::Result<()> {
    for spec in specs {
        let Some((name, raw)) = spec.split_once('=') else {
            return Err(io::Error::other(format!(
                "param override '{spec}' is not Name=Value"
            )));
        };
        let Some(lever_spec) = crate::search::options::lever(name) else {
            return Err(io::Error::other(format!(
                "param override: unknown search option '{name}'"
            )));
        };
        let parsed = match lever_spec.kind {
            OptionKind::Check => {
                if raw != "true" && raw != "false" {
                    return Err(io::Error::other(format!(
                        "param override: '{raw}' is not true/false for {name}"
                    )));
                }
                i64::from(raw == "true")
            }
            _ => match raw.parse::<i64>() {
                Ok(value) => value,
                Err(_) => {
                    return Err(io::Error::other(format!(
                        "param override: '{raw}' is not an integer for {name}"
                    )));
                }
            },
        };
        crate::search::options::set(searcher.params_mut(), lever_spec.name, parsed);
        output.line(&format!(
            "info string param-override: {}={raw} (applies to every go)",
            lever_spec.name
        ))?;
    }
    Ok(())
}

/// Parses and applies `setoption name <name> [value <value>]`.
///
/// Every name is validated against the catalogue (public options and levers);
/// the reply behaviour mirrors the step-0 capture: an unknown name is reported
/// with an `info string`, every value-level rejection (out-of-range spin,
/// non-boolean check value, missing value) is silent and keeps the previous
/// value, and a button acts without a value.
pub(super) fn handle_set_option(
    line: &str,
    searcher: &mut Searcher,
    pool: &mut WorkerPool,
    extras: &mut EngineExtras,
    output: &mut Output,
) -> io::Result<()> {
    if let Some(reply) = apply_set_option(line, searcher, pool, extras) {
        output.line(&reply)?;
    }
    Ok(())
}

/// Applies one `setoption` command and returns the optional reply line.
///
/// `None` means the command was accepted silently or rejected silently —
/// both leave no trace on the wire, exactly like the reference.
pub(super) fn apply_set_option(
    line: &str,
    searcher: &mut Searcher,
    pool: &mut WorkerPool,
    extras: &mut EngineExtras,
) -> Option<String> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.get(1).copied() != Some("name") {
        return Some("info string setoption expects 'name <name>'".to_owned());
    }
    let value_index = tokens.iter().position(|token| *token == "value");
    let (name, value) = match value_index {
        Some(index) => (tokens[2..index].join(" "), tokens[index + 1..].join(" ")),
        None => (tokens[2..].join(" "), String::new()),
    };
    if name.is_empty() {
        return Some("info string setoption needs a name".to_owned());
    }

    if let Some(spec) = public(&name) {
        // The book and tablebase options act outside the search state and
        // report their resulting state, so a failure is never silent.
        match spec.name {
            "Own Book" => {
                if value == "true" || value == "false" {
                    extras.book_enabled = value == "true";
                }
                return None;
            }
            "Book File" => return Some(load_book(&value, extras)),
            "Book Depth" => {
                if spin_value_is_valid(spec.min, spec.max, &value) {
                    extras.book_depth = value.parse::<u32>().expect("validated above");
                }
                return None;
            }
            "SyzygyPath" => {
                syzygy::initialize(&value);
                return Some(syzygy_status_line());
            }
            "SyzygyRootProbe" => {
                if value == "true" || value == "false" {
                    extras.syzygy_root_probe = value == "true";
                }
                return None;
            }
            "Syzygy50MoveRule" => {
                if value == "true" || value == "false" {
                    syzygy::set_use_rule50(value == "true");
                }
                return None;
            }
            _ => {}
        }
        apply_public_option(spec, &value, searcher, pool);
        return None;
    }
    if let Some(spec) = lever(&name) {
        apply_lever(spec, &value, searcher);
        return None;
    }
    // Reference parity: an unknown name is the one reported rejection.
    Some(format!("info string Unknown option: {name}"))
}

/// Loads (or auto-detects) the opening book and returns the diagnostic line.
///
/// An empty value requests auto-detection next to the executable or in a
/// `books/` subdirectory; a named file that cannot be read is reported and the
/// previous book is dropped rather than silently kept.
pub(super) fn load_book(value: &str, extras: &mut EngineExtras) -> String {
    let path = if value.is_empty() {
        match find_default_book_path() {
            Some(path) => path.to_string_lossy().into_owned(),
            None => {
                extras.book.clear();
                return "info string no opening book loaded".to_owned();
            }
        }
    } else {
        value.to_owned()
    };
    if extras.book.load(&path) {
        book_status_line(&extras.book)
    } else {
        extras.book.clear();
        format!("info string book file not loaded: {path}")
    }
}

/// True when `value` is a valid spin value for `spec`.
///
/// A value that does not parse, or that falls outside the declared range, is
/// rejected: the previous value is kept and nothing is printed.
pub(super) fn spin_value_is_valid(
    spec_min: Option<i64>,
    spec_max: Option<i64>,
    value: &str,
) -> bool {
    match value.parse::<i64>() {
        Ok(parsed) => {
            let min = spec_min.unwrap_or(i64::MIN);
            let max = spec_max.unwrap_or(i64::MAX);
            min <= parsed && parsed <= max
        }
        Err(_) => false,
    }
}

/// Applies one public option with full validation.
///
/// The functional options resize the table (`Hash`), clear it (`Clear Hash`)
/// or resize the helper pool (`Threads`); the book and tablebase options load
/// or switch their feature and report the resulting state. An
/// out-of-range or unparsable spin value is rejected silently and the
/// previous value is kept; a check option accepts only `true`/`false`.
pub(super) fn apply_public_option(
    spec: &UciOption,
    value: &str,
    searcher: &mut Searcher,
    pool: &mut WorkerPool,
) {
    match spec.kind {
        OptionKind::Button => {
            // A button acts without a value; a stray value token is ignored.
            if spec.name == "Clear Hash" {
                searcher.clear();
            }
        }
        OptionKind::Check => {
            // Accepted only when well-formed; the inert check options store
            // nothing yet, so acceptance is the whole effect.
        }
        OptionKind::String => {
            // Accepted, inert until the book/tablebase phases land.
        }
        OptionKind::Spin => {
            if !spin_value_is_valid(spec.min, spec.max, value) {
                return;
            }
            match spec.name {
                "Hash" => {
                    searcher.set_hash(value.parse::<usize>().expect("validated above"));
                }
                "Move Overhead" => {
                    searcher.set_move_overhead(value.parse::<u64>().expect("validated above"));
                }
                // `Threads` resizes the persistent helper pool. The
                // main thread's state is not rebuilt; `Threads = 1` runs the
                // single-threaded path with no helpers.
                "Threads" => {
                    pool.set_threads(value.parse::<usize>().expect("validated above"));
                }
                _ => {}
            }
        }
    }
}

/// Applies one research lever with full validation.
///
/// A check lever accepts only `true`/`false`; a spin lever only an integer in
/// its declared range. Anything else is rejected silently and the previous
/// value is kept. Every lever of [`crate::search::lever_table`] drives a
/// search parameter.
pub(super) fn apply_lever(spec: &LeverSpec, value: &str, searcher: &mut Searcher) {
    let parsed = match spec.kind {
        OptionKind::Check => {
            if value != "true" && value != "false" {
                return;
            }
            i64::from(value == "true")
        }
        _ => {
            if !spin_value_is_valid(Some(spec.min), Some(spec.max), value) {
                return;
            }
            value.parse::<i64>().expect("validated above")
        }
    };
    set(searcher.params_mut(), spec.name, parsed);
}

#[cfg(test)]
mod tests {
    use super::EngineExtras;
    use super::*;
    use crate::search::options::{get, lever, public};
    use crate::search::pool::WorkerPool;
    use crate::search::searcher::Searcher;

    /// Applies one `setoption` line with fresh feature state.
    fn apply_option(line: &str, searcher: &mut Searcher, pool: &mut WorkerPool) -> Option<String> {
        apply_set_option(line, searcher, pool, &mut EngineExtras::new())
    }

    /// Applies `line` to a fresh 2 MB searcher and returns (reply, value read
    /// back through the lever accessor).
    fn applied(name: &str, value: Option<&str>) -> (Option<String>, i64) {
        let mut searcher = Searcher::new(2);
        let mut pool = WorkerPool::new();
        let spec = lever(name).expect("test names a lever");
        let line = match value {
            Some(value) => format!("setoption name {name} value {value}"),
            None => format!("setoption name {name}"),
        };
        let reply = apply_option(&line, &mut searcher, &mut pool);
        let state = get(searcher.params(), spec);
        (reply, state)
    }

    /// Applies a sequence of `setoption name <name> value <value>` commands to
    /// one searcher and returns the last (reply, lever state).
    fn applied_after(name: &str, values: &[&str]) -> (Option<String>, i64) {
        let mut searcher = Searcher::new(2);
        let mut pool = WorkerPool::new();
        let spec = lever(name).expect("test names a lever");
        let mut reply = None;
        for value in values {
            let line = if value.is_empty() {
                format!("setoption name {name}")
            } else {
                format!("setoption name {name} value {value}")
            };
            reply = apply_option(&line, &mut searcher, &mut pool);
        }
        (reply, get(searcher.params(), spec))
    }

    #[test]
    fn an_unknown_option_name_is_reported() {
        let mut searcher = Searcher::new(2);
        let mut pool = WorkerPool::new();
        assert_eq!(
            apply_option(
                "setoption name NoSuchOption value 5",
                &mut searcher,
                &mut pool,
            ),
            Some("info string Unknown option: NoSuchOption".to_owned())
        );
    }

    #[test]
    fn a_spin_value_above_the_maximum_is_rejected_and_the_previous_kept() {
        let (_, set) = applied("AspirationBase", Some("60"));
        assert_eq!(set, 60, "a valid value is applied");
        // 999999 is far above AspirationBase's maximum of 100: rejected, and
        // the previous value survives.
        let (reply, kept) = applied_after("AspirationBase", &["60", "999999"]);
        assert_eq!(reply, None, "a range rejection is silent");
        assert_eq!(kept, 60, "the previous value was not kept");
    }

    #[test]
    fn a_spin_value_below_the_minimum_is_rejected() {
        let (reply, kept) = applied_after("AspirationBase", &["60", "0"]);
        assert_eq!(reply, None);
        assert_eq!(kept, 60, "the previous value was not kept");
    }

    #[test]
    fn an_unparsable_spin_value_is_rejected() {
        let (reply, kept) = applied_after("AspirationBase", &["60", "not-a-number"]);
        assert_eq!(reply, None);
        assert_eq!(kept, 60, "the previous value was not kept");
    }

    #[test]
    fn a_missing_spin_value_is_rejected() {
        let (reply, kept) = applied_after("AspirationBase", &["60", ""]);
        assert_eq!(reply, None);
        assert_eq!(kept, 60, "the previous value was not kept");
    }

    #[test]
    fn a_check_option_accepts_only_true_and_false() {
        // Valid values are applied.
        let (reply, state) = applied("UseQsNarrowing", Some("true"));
        assert_eq!(reply, None);
        assert_eq!(state, 1);
        // Anything else is rejected silently and the previous value is kept.
        let (reply, kept) = applied_after("UseQsNarrowing", &["true", "maybe"]);
        assert_eq!(reply, None, "a check rejection is silent");
        assert_eq!(kept, 1, "the previous value was not kept");
        // `false` is accepted again.
        let (reply, state) = applied("UseQsNarrowing", Some("false"));
        assert_eq!(reply, None);
        assert_eq!(state, 0);
    }

    #[test]
    fn a_button_acts_without_a_value() {
        let mut searcher = Searcher::new(2);
        let mut pool = WorkerPool::new();
        // Both forms act silently: the bare button and one with a stray value.
        assert_eq!(
            apply_option("setoption name Clear Hash", &mut searcher, &mut pool),
            None
        );
        assert_eq!(
            apply_option(
                "setoption name Clear Hash value 5",
                &mut searcher,
                &mut pool
            ),
            None
        );
    }

    #[test]
    fn hash_values_outside_the_declared_range_are_rejected() {
        let mut searcher = Searcher::new(2);
        let mut pool = WorkerPool::new();
        for bad in ["0", "999999", "abc"] {
            assert_eq!(
                apply_option(
                    &format!("setoption name Hash value {bad}"),
                    &mut searcher,
                    &mut pool,
                ),
                None,
                "Hash value {bad} must be rejected silently"
            );
        }
        // A valid value is accepted silently.
        assert_eq!(
            apply_option("setoption name Hash value 32", &mut searcher, &mut pool),
            None
        );
    }

    #[test]
    fn threads_is_validated_but_the_engine_stays_single_threaded() {
        let mut searcher = Searcher::new(2);
        let mut pool = WorkerPool::new();
        // Below the minimum and above the maximum: rejected silently.
        for bad in ["0", "999"] {
            assert_eq!(
                apply_option(
                    &format!("setoption name Threads value {bad}"),
                    &mut searcher,
                    &mut pool,
                ),
                None,
                "Threads value {bad} must be rejected silently"
            );
        }
        // In range: accepted silently. `Threads = 1` keeps the pool empty;
        // a bigger value grows it (the pool behaviour itself is covered by
        // the step-8 pool tests).
        for good in ["1", "256"] {
            assert_eq!(
                apply_option(
                    &format!("setoption name Threads value {good}"),
                    &mut searcher,
                    &mut pool,
                ),
                None,
                "Threads value {good} must be accepted"
            );
        }
        assert_eq!(pool.threads(), 256, "the accepted value reached the pool");
    }

    #[test]
    fn a_ported_feature_lever_takes_effect() {
        // `UseProbCut` ships off; switching it on reaches the search state.
        let spec = lever("UseProbCut").expect("the catalogue holds the ported lever");
        assert_eq!(spec.default, 0);
        let (reply, state) = applied("UseProbCut", Some("true"));
        assert_eq!(reply, None);
        assert_eq!(state, 1, "the lever must change the search parameters");
    }

    #[test]
    fn levers_the_catalogue_dropped_are_unknown_options() {
        // The reference's inert, duplicated and negative levers, and the two
        // removed public options, are not advertised and not accepted.
        let mut searcher = Searcher::new(2);
        let mut pool = WorkerPool::new();
        for name in [
            "UseLmrV2",
            "UseSeeReach",
            "UseFormulaRfp",
            "EvalFile",
            "Contempt",
        ] {
            let reply = apply_option(
                &format!("setoption name {name} value 1"),
                &mut searcher,
                &mut pool,
            );
            assert!(
                reply
                    .as_deref()
                    .is_some_and(|reply| reply.contains("Unknown option")),
                "{name} must be unknown, got {reply:?}"
            );
        }
    }

    #[test]
    fn public_options_outside_the_search_are_known_and_validated() {
        let mut searcher = Searcher::new(2);
        let mut pool = WorkerPool::new();
        // The book and tablebase names are advertised. The functional
        // book/tablebase options report their resulting state. None of them may
        // be reported as unknown.
        for line in [
            "setoption name Own Book value true",
            "setoption name Book Depth value 8",
            "setoption name SyzygyRootProbe value false",
            "setoption name Syzygy50MoveRule value false",
        ] {
            let reply = apply_option(line, &mut searcher, &mut pool);
            assert!(
                reply
                    .as_deref()
                    .is_none_or(|reply| !reply.contains("Unknown option")),
                "{line} must be a known option"
            );
        }
        // A missing book file is reported instead of silently ignored.
        let reply = apply_option(
            "setoption name Book File value book.ctg",
            &mut searcher,
            &mut pool,
        );
        assert!(
            reply
                .as_deref()
                .is_some_and(|reply| reply.contains("book file not loaded")),
            "a missing book must be reported: {reply:?}"
        );
        // A tablebase path with no tables reports "none".
        let reply = apply_option(
            "setoption name SyzygyPath value C:\\definitely-no-tables",
            &mut searcher,
            &mut pool,
        );
        assert!(
            reply
                .as_deref()
                .is_some_and(|reply| reply.contains("syzygy tablebases: none")),
            "an empty tablebase path must be reported: {reply:?}"
        );
        // The spin/check values are still validated.
        assert_eq!(
            apply_option(
                "setoption name Book Depth value 200",
                &mut searcher,
                &mut pool
            ),
            None,
            "an out-of-range book depth must be rejected silently, not applied"
        );
    }

    #[test]
    fn a_malformed_setoption_is_reported() {
        let mut searcher = Searcher::new(2);
        let mut pool = WorkerPool::new();
        assert!(apply_option("setoption value 5", &mut searcher, &mut pool).is_some());
        assert!(apply_option("setoption", &mut searcher, &mut pool).is_some());
    }

    #[test]
    fn every_public_option_is_resolvable_and_validated() {
        // The public spin ranges are consistent with the validation helper:
        // the default is in range, one step out is not.
        for spec in crate::search::options::PUBLIC_OPTIONS {
            assert!(public(spec.name).is_some());
            if let (Some(min), Some(max)) = (spec.min, spec.max) {
                assert!(min <= max, "{}: inverted range", spec.name);
            }
        }
    }
}
