//! End-to-end smoke tests for the UCI lifecycle, one module per area.

mod argv;
#[path = "../common/mod.rs"]
mod common;
mod go_limits;
mod handshake;
mod helpers;
mod lifecycle;
mod ponder;
mod stats;
mod stop;
mod threads;
