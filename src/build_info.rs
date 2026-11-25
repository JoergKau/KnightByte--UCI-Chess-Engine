//! Build identity: crate version, public release version, git commit and
//! build target.
//!
//! The values come from `build.rs` at compile time. The banner and the UCI
//! handshake share the name, author and URL constants, so the two can never
//! drift apart. The commit details stay available for logs and bug reports.

/// Crate version from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Public release version (`major.minor`) shown by the banner and `id name`.
pub const DISPLAY_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION_MAJOR"),
    ".",
    env!("CARGO_PKG_VERSION_MINOR")
);
/// Engine name shown by the banner and `id name`.
pub const ENGINE_NAME: &str = "KnightByte";
/// Author shown by the banner and `id author`.
pub const ENGINE_AUTHOR: &str = "JoergKau";
/// Public home of the engine shown by the banner.
pub const ENGINE_URL: &str = "https://github.com/JoergKau/KnightByte--UCI-Chess-Engine";
/// Short git commit the binary was built from (`unknown` outside a checkout).
pub const GIT_HASH: &str = env!("KNIGHTBYTE_GIT_HASH");
/// Commit date (`YYYY-MM-DD`), not the build time, so tagged builds reproduce.
pub const GIT_DATE: &str = env!("KNIGHTBYTE_GIT_DATE");
/// `clean` or `dirty`: whether tracked files differed at build time.
pub const GIT_DIRTY: &str = env!("KNIGHTBYTE_GIT_DIRTY");
/// Rust target triple the binary was compiled for.
pub const BUILD_TARGET: &str = env!("KNIGHTBYTE_BUILD_TARGET");
/// Cargo profile (`debug` or `release`).
pub const BUILD_PROFILE: &str = env!("KNIGHTBYTE_BUILD_PROFILE");

/// The UCI `id name` value, for example `KnightByte 1.0`.
pub fn identity() -> String {
    format!("{ENGINE_NAME} {DISPLAY_VERSION}")
}

/// One human-readable line with every build fact, for logs and bug reports.
///
/// Example: `version 1.0.0, commit c1b88fb12345 (2026-11-23, clean), target
/// aarch64-apple-darwin, profile release`.
pub fn details() -> String {
    format!(
        "version {VERSION}, commit {GIT_HASH} ({GIT_DATE}, {GIT_DIRTY}), target {BUILD_TARGET}, profile {BUILD_PROFILE}"
    )
}
