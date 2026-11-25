# KnightByte

KnightByte is a UCI chess engine written in Rust and a port of my C#
(.NET 11.0) engine PawnForgeCS. The main differences are summarised in
[`docs/knightbyte-vs-pawnforge.md`](docs/knightbyte-vs-pawnforge.md). It
combines a modern alpha-beta search with a fully embedded, self-designed and
self-trained neural network evaluation, Syzygy tablebase probing, and
multi-threaded Lazy SMP search.

Windows 11 x64 is the primary platform. macOS arm64 and Linux x64 are fully
supported from the same source.

A detailed capability overview is in
[`KnightByte-feature-list.md`](KnightByte-feature-list.md).

## Neural network

The evaluation is a fully self-designed, self-trained NNUE in the v5 net format:
768 input slots (2 colours × 6 piece types × 64 squares) bucketed into 8 king
regions with a mirrored per-perspective mapping, L1 = 512, and a single output
head over the concatenated 2 × 512 SCReLU activations — 3,147,265 parameters in
a 6.3 MB embedded file (`net/jokausch.nnue`, description
`v5_poolv9_512_auxwdl-150`). Updates are fully incremental with a lazy own-king
refresh; cold-start and incremental evaluations are bit-identical. A strict
structure report on the shipped net passes: checksum OK, no genuinely dead input
rows, and the two output halves are distinct.

## Training

Positions are harvested from selected public game archives and re-scored with
Stockfish 19 to produce teacher centipawn labels; the pool of record contains
406,449,912 positions (~12.1 GiB), including curated endgame shards. A full run
takes roughly 3–5 hours on a single laptop GPU (150 epochs, batch 16,384, a
linear score loss against pure teacher labels), followed by material-ordering,
region-audit, teacher-correlation, side-to-move flip, bit-exactness and 600-game
match gates before promotion. Only the resulting network is shipped with the
engine.

## Credits

In no particular order, these engines have been notable sources of ideas or
inspiration:

- [Hobbes 4.0](https://github.com/kelseyde/hobbes-chess-engine)
- [Calvin 6.6.1](https://github.com/kelseyde/calvin-chess-engine)
- [Pedantic 2.1.0](https://github.com/JoAnnP38/PedanticRF)

Tools used during development:

- [Fathom](https://github.com/jdart1/Fathom)
- [Fastchess](https://github.com/Disservin/fastchess)
- [Cutechess](https://github.com/cutechess/cutechess)

Opening positions from:

- [UHO_Lichess_4852_v1.epd](https://github.com/official-stockfish/books)

## Strength

KnightByte 1.0 is the first stable release. No CCRL rating exists yet; the
figures below are local matches against established engines and are not a CCRL
rating.

I compared it against engines listed on the CCRL. The first match was on
Windows 11 with an AMD Zen 4 CPU, at CCRL 2'+1" conditions, against Pedantic
2.1.0 (CCRL Elo 3041 as of 2026-11-24):

```
Score of KnightByte vs Pedantic: 33 - 13 - 44  [0.611] 90
...      KnightByte playing White: 22 - 7 - 17  [0.663] 46
...      KnightByte playing Black: 11 - 6 - 27  [0.557] 44
...      White vs Black: 28 - 18 - 44  [0.556] 90
Elo difference: 78.5 +/- 51.7, LOS: 99.8 %, DrawRatio: 48.9 %
SPRT: llr 0 (0.0%), lbound -inf, ubound inf
```

The second match was against Pedantic 2.1.0 (CCRL Elo 3357 as of 2026-11-24)
at CCRL 40/15 conditions:

```
Score of KnightByte vs Pedantic: 74 - 37 - 89  [0.593] 200
...      KnightByte playing White: 39 - 16 - 45  [0.615] 100
...      KnightByte playing Black: 35 - 21 - 44  [0.570] 100
...      White vs Black: 60 - 51 - 89  [0.522] 200
Elo difference: 65.0 +/- 36.1, LOS: 100.0 %, DrawRatio: 44.5 %
SPRT: llr 0 (0.0%), lbound -inf, ubound inf
```

Both matches finished with **Timeouts: 0 / Crashes: 0**.

**KnightByte's strength should be around 3400 Elo.**

## Requirements

- Windows 11 x64 is the primary build and test platform (the CCRL candidate).
- macOS on Apple Silicon (M1–M5); Intel Macs are not supported.
- Linux on x86-64 with AVX2 (Intel Haswell 2013 and newer, AMD Zen and newer);
  Linux on ARM is not supported.
- Rust stable 1.100.0, pinned in `rust-toolchain.toml`.
- Optional: a Syzygy tablebase directory for endgame probing.
- Optional: an external UCI verifier for the FEN-verified tests (the suite runs
  without it; those checks are skipped).

## Build

```bash
cargo build --release
```

The binary is written to `target/release/knightbyte` (`.exe` on Windows). This
is the portable development build.

### Release builds

The shipped artifacts are profile-guided and built for a fixed instruction-set
baseline on each platform.

| Platform | File | Baseline |
| --- | --- | --- |
| Windows x64, `v3` (default) | `knightbyte-v3.exe` | AVX2, BMI2, FMA |
| Windows x64, `v2` | `knightbyte-v2.exe` | SSE4.2, POPCNT |
| Windows x64, `v4-vnni` | `knightbyte-v4-vnni.exe` | AVX-512 + VNNI |
| macOS arm64 | `knightbyte` | `apple-m1` (M1–M5) |
| Linux x64 | `knightbyte` | `x86-64-v3` (AVX2) |

Inside each Windows executable the SIMD kernels and the sliding-attack method
are still chosen once at start-up. A build started on a CPU that lacks the
instructions it was compiled for prints which features are missing and exits; it
does not crash. The executables are plain files (no installer, no packer) and
need no Visual C++ runtime.

The macOS artifact is an `aarch64-apple-darwin` binary built with
`-Ctarget-cpu=apple-m1`, the baseline of every Apple M processor. NEON is part
of the AArch64 baseline, so there is no per-chip variant. The Linux artifact is
an `x86_64-unknown-linux-gnu` binary built with `-Ctarget-cpu=x86-64-v3`; a CPU
without AVX2, BMI2 and FMA gets a clear message naming the missing features
instead of an illegal-instruction crash.

Profile-guided optimization (PGO) gives about +6–7.5 % on Apple Silicon with the
node pins unchanged. The profile is generated on the build host and is never
committed.

The release scripts build, verify and package an artifact on their own host:

```bat
python tools\build_release.py
python tools\build_release.py --variants v2 v3
```

```bash
python3 tools/build_release_macos.py
python3 tools/build_release_linux.py
```

Each script builds with PGO and verifies the four bench node pins and the
forced-backend identity, then writes the archive and its `.sha256` checksum.
`python tools/version.py` updates the version in `Cargo.toml`/`Cargo.lock` and
creates the release tag.

A source build can use the same baselines explicitly:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
rustup target add aarch64-apple-darwin
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release --target aarch64-apple-darwin
```

## Run

```bash
./target/release/knightbyte
```

On an interactive terminal the engine prints a short banner to standard error
and then reads UCI commands from standard input. With redirected input (a GUI or
a match runner) no banner is printed, so the protocol streams stay clean.

A quick handshake check:

```bash
printf 'uci\nisready\nquit\n' | ./target/release/knightbyte
```

The engine speaks UCI on stdin/stdout and can be used with any UCI-compatible
GUI or match runner. It implements the standard surface (`uci`, `isready`,
`ucinewgame`, `position`, `go`, `stop`, `ponderhit`, `setoption`, `quit`) with
the depth, movetime, clock, movestogo, nodes, mate, infinite, ponder and
searchmoves limiters. Besides the protocol it understands `bench [depth]`,
`perft <depth>`, `nnueeval` and `d`, plus two hidden developer diagnostics:
`searchstats [reset|off]` (node, quiescence, table-probe and cutoff statistics)
and `nnueregions [reset|off]` (per-king-region accumulator refreshes).

## UCI options

The engine advertises ten public options:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `Hash` | spin | 256 | Transposition table size in MB (1–1024) |
| `Threads` | spin | 1 | Search threads (Lazy SMP, 1–256) |
| `Move Overhead` | spin | 50 | Safety margin in ms for GUI/match lag |
| `Clear Hash` | button | — | Clears the transposition table and search history |
| `Own Book` | check | false | Enables the opening book |
| `Book File` | string | — | Path to a `.ctg` book; empty looks next to the executable and in `books/` |
| `Book Depth` | spin | 16 | Maximum book ply (0–100) |
| `SyzygyPath` | string | — | Tablebase directory (or `;`/`:`-separated list) |
| `SyzygyRootProbe` | check | false | Direct DTZ-optimal root probing |
| `Syzygy50MoveRule` | check | true | Honour the 50-move rule in tablebase scores |

The search also has **sixty research levers** named after the feature they
control (`Nmp...`, `Rfp...`, `Lmr...`, `Qs...`). They are deliberately absent
from the `uci` handshake, so GUI option dialogs show only the ten public options
above, but they are fully functional through
`setoption name <Name> value <Value>` and through the `param:Name=Value`
command-line override. Their defaults are the shipped configuration; changing
one is a tuning experiment. A name that is neither a public option nor one of
the sixty levers is reported as an unknown option.

Pondering is not advertised. The network is embedded, so there is no `EvalFile`
option, and the engine has no `Contempt` setting. The `info` lines carry
`tbhits <n>` between `hashfull` and `time` only while a tablebase set is in use;
without tablebases the line is exactly `depth seldepth score nodes nps hashfull
time pv`.

## Tournament use

The shipped defaults are tournament-friendly:

- The **Windows x64** release zip is the build intended for CCRL and Windows
  match runners.
- The release gate ran 500 games with no time loss, illegal move, disconnect or
  crash; the UCI smoke tests and the multi-thread stress walk are part of the
  test suite.
- `Own Book` defaults to **OFF**; no position learning is performed. A tester's
  own opening book and ponder settings can be used unchanged.
- Suggested settings: `Hash` 128–256 MB, `Threads` 1. Optional `SyzygyPath`
  enables 3-4-5 piece tablebases; `SyzygyRootProbe` stays off by default.
- The Windows zip contains the three executables and a short readme; the macOS
  and Linux archives contain the binary and the full documentation set.

## Testing

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

Every FEN used by a test must be verified by an external UCI verifier at test
time. Set `KNIGHTBYTE_FEN_VERIFIER` to the verifier binary before running the
tests:

```bash
KNIGHTBYTE_FEN_VERIFIER=/path/to/uci-verifier cargo test
```

The verifier is a local test tool and is not distributed with KnightByte.

The deterministic benchmark identity check:

```bash
printf 'uci\nbench 16\nquit\n' | ./target/release/knightbyte
```

The fixed-depth node pins of the shipped build are `bench 6` **7,553**,
`bench 12` **203,939**, `bench 16` **1,285,264** and `bench 18` **3,668,723**.
They are the single machine-readable pin set in `tests/golden/bench.json`; a
search change that moves any of them is a different engine and must be reviewed
as such.

## Attribution

KnightByte's Syzygy tablebase probing is based on the Fathom project. The
required attribution and license text are in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md). The neural network is
self-designed and self-trained for this engine.

## License

MIT — see [`LICENSE`](LICENSE). Third-party attributions and license texts are
in [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).
