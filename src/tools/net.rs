//! Embedded network asset and the optional research override.
//!
//! The production network is compiled into the binary so the release artifact
//! has no external data dependency. The checksum test protects against an
//! accidental file replacement during development.
//!
//! A build with the `research-net` Cargo feature reads `KNIGHTBYTE_NET` at
//! startup and evaluates a candidate file instead. That is a build-time switch,
//! not a UCI option: released binaries keep exactly one network and advertise
//! no way to load another.

#[cfg(feature = "research-net")]
use std::path::PathBuf;
#[cfg(feature = "research-net")]
use std::sync::Mutex;
use std::sync::OnceLock;

/// Size of the embedded network file in bytes.
pub const NET_SIZE: usize = 6_294_672;

/// Name under which the embedded network announces itself in UCI output
/// (`eval` and the eval-source line): the file it was built from.
pub const NET_SOURCE_NAME: &str = "jokausch.nnue";

/// SHA-256 digest of the embedded network file.
pub const NET_SHA256: &str = "0a70169ce7d75a9a5a392cd293746fc1f5191a75dc183b5b45fb69d869ce2968";

/// Environment variable that selects a research network.
#[cfg(feature = "research-net")]
pub const RESEARCH_NET_ENV: &str = "KNIGHTBYTE_NET";

/// Command-line switch that selects a research network (before the UCI loop).
#[cfg(feature = "research-net")]
pub const RESEARCH_NET_ARG: &str = "--net";

#[cfg(feature = "research-net")]
static RESEARCH_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

static SOURCE_NAME: OnceLock<String> = OnceLock::new();

/// Returns the embedded network bytes.
pub fn bytes() -> &'static [u8] {
    include_bytes!("../../net/jokausch.nnue")
}

/// Records the research-network path from a process argument.
#[cfg(feature = "research-net")]
pub fn set_research_path(path: PathBuf) {
    *RESEARCH_PATH.lock().expect("research path lock") = Some(path);
}

/// Returns the research-network path: the process argument wins over the
/// environment variable, so a match harness can pass two different nets to two
/// engine processes on the same machine.
#[cfg(feature = "research-net")]
pub fn research_path() -> Option<PathBuf> {
    if let Some(path) = RESEARCH_PATH
        .lock()
        .expect("research path lock")
        .as_ref()
        .cloned()
    {
        return Some(path);
    }
    std::env::var_os(RESEARCH_NET_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Returns the name the evaluation reports as its source.
///
/// In a normal build this is the embedded file's name. A research build reports
/// the override file, so a match log or a `bench` run says which network was
/// actually used.
pub fn source_name() -> &'static str {
    SOURCE_NAME.get_or_init(|| {
        #[cfg(feature = "research-net")]
        if let Some(path) = research_path() {
            return path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
        }
        NET_SOURCE_NAME.to_owned()
    })
}

#[cfg(test)]
mod tests {
    use super::{NET_SHA256, NET_SIZE, bytes};
    use sha2::{Digest, Sha256};

    #[test]
    fn embedded_network_has_expected_size() {
        assert_eq!(bytes().len(), NET_SIZE);
    }

    #[test]
    fn embedded_network_has_expected_checksum() {
        let digest = Sha256::digest(bytes());
        let checksum: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(checksum, NET_SHA256);
    }
}
