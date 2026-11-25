//! KnightByte UCI entry point.

use std::process::ExitCode;

/// Runs the UCI event loop and maps I/O failures to a non-zero exit code.
///
/// The process arguments reproduce the reference's startup surface:
/// `param:Name=Value` entries preload validated search-parameter overrides, and
/// any other argument becomes a one-shot UCI command line that runs once and
/// exits. With no arguments the engine reads standard input interactively.
fn main() -> ExitCode {
    // A dedicated build must not start on a CPU it was not compiled for.
    if let Err(message) = knightbyte::platform::cpu::check_build_requirements() {
        eprintln!("fatal: {message}");
        return ExitCode::FAILURE;
    }
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // Research builds accept `--net <path>` before the UCI surface. The switch is
    // not compiled into a release build and is never advertised as an option, so
    // a shipped binary has exactly one network. Match harnesses use it to hand
    // two different nets to two processes of the same binary.
    #[cfg(feature = "research-net")]
    let argv = {
        let mut argv = argv;
        if let Some(index) = argv
            .iter()
            .position(|argument| argument == knightbyte::tools::net::RESEARCH_NET_ARG)
        {
            if index + 1 >= argv.len() {
                eprintln!("fatal: --net needs a path");
                return ExitCode::FAILURE;
            }
            let path = argv.remove(index + 1);
            argv.remove(index);
            knightbyte::tools::net::set_research_path(path.into());
        }
        argv
    };
    match knightbyte::uci::run_with_argv(&argv) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("fatal: {error}");
            ExitCode::FAILURE
        }
    }
}
