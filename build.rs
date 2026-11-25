//! Build metadata for the UCI identity line.
//!
//! A crate version alone cannot tell two builds apart: the campaign bake, a
//! local debug build and a release binary all say `0.1.0`. The UCI identity
//! therefore carries the git commit, whether the worktree was dirty, the target
//! and the Cargo profile. The commit date is used instead of the build time so a
//! clean tagged build stays reproducible.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");

    let hash = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let date = git(&["show", "-s", "--format=%cs", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|status| if status.is_empty() { "clean" } else { "dirty" })
        .unwrap_or("unknown");

    println!("cargo:rustc-env=KNIGHTBYTE_GIT_HASH={hash}");
    println!("cargo:rustc-env=KNIGHTBYTE_GIT_DATE={date}");
    println!("cargo:rustc-env=KNIGHTBYTE_GIT_DIRTY={dirty}");
    println!(
        "cargo:rustc-env=KNIGHTBYTE_BUILD_TARGET={}",
        std::env::var("TARGET").unwrap_or_else(|_| "unknown".into())
    );
    println!(
        "cargo:rustc-env=KNIGHTBYTE_BUILD_PROFILE={}",
        std::env::var("PROFILE").unwrap_or_else(|_| "unknown".into())
    );
}

/// Runs one `git` query and trims its output; `None` when git is unavailable or
/// the directory is not a checkout, so a source tarball still builds.
fn git(arguments: &[&str]) -> Option<String> {
    let output = Command::new("git").args(arguments).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    // Keep the empty string: `git status --porcelain` prints nothing for a clean
    // tree, and the caller must distinguish that from "git unavailable".
    Some(text.trim().to_owned())
}
