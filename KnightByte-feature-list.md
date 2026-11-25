# KnightByte — Feature List

KnightByte 1.0 is a Rust UCI chess engine with a fully self-contained neural
evaluation, a modern alpha-beta search, and Syzygy tablebase support.

## Engine

- **Language / runtime**: Rust (stable 1.100, edition 2024), one self-contained
  binary per platform. No managed runtime and no installer.
- **Search**: iterative deepening with aspiration windows, principal variation
  search, transposition table, and Lazy SMP multi-threading.
- **Evaluation**: fully self-designed embedded neural network
  (`net/jokausch.nnue`) — see the network and training sections below.
- **Threads**: shared lockless transposition table; per-thread histories,
  killers and evaluator state.

## Neural network (fully self-designed)

- **Architecture** — the v5 net format, feature scheme 2: 768 input slots
  (2 colours × 6 piece types × 64 squares), bucketed into 8 king regions with a
  mirrored per-perspective mapping, L1 = 512 (6,144 inputs = 8 regions × 768
  slots), and one global output head over the concatenated 2 × 512 SCReLU
  activations. **3,147,265 parameters** in a 6,294,672-byte file.
- **Self-designed and self-trained** — the v5 architecture, feature mapping,
  quantized file format and training pipeline are defined in this project; the
  shipped weights are trained entirely from this project's own data.
- **Shipped net** — `jokausch.nnue`, description header
  `v5_poolv9_512_auxwdl-150`; checksum verified, quantization constants pinned
  per build.
- **Runtime** — fully incremental accumulator with lazy own-king refresh and
  one accumulator pair per ply; cold-start and incremental evaluations are
  bit-identical; hardware kernels for x86-64 (SSE4.1, AVX2, AVX-512) and
  AArch64 (NEON), plus a scalar reference path, all bit-identical.
- **Structure report** — checksum OK, no genuinely dead input rows, and the two
  output halves are distinct.

## Training workflow

- **Data** — positions are harvested from selected public game archives,
  validated, and re-scored with a strong reference engine to obtain teacher
  centipawn labels. The shipped net's pool of record holds **406,449,912
  positions (~12.1 GiB)**, including curated endgame shards.
- **Trainer** — a custom network definition implements the v5 architecture,
  feature scheme, king-region buckets, quantization and export. A full run
  trains on a single laptop GPU in roughly 3–5 hours at 4.4–6.5 M positions/s,
  using batch 16,384, a 150-epoch cosine schedule, and a linear score loss
  against pure teacher labels.
- **Promotion gates** — material-ordering gate, region audit with negative
  controls, teacher-correlation and side-to-move-flip checks, bit-exactness
  tests, and a 600-game match as the only strength arbiter.

## Search techniques

- Transposition table: lockless XOR-verified 16-byte entries, bucket-aligned to
  one cache line, static-eval caching, PV-flag propagation, cache prefetching.
- Move ordering: TT move, staged move picker, captures with SEE, killers,
  countermoves, butterfly/continuation history, gravity-bounded history updates.
- Pruning and reductions:
  - Reverse futility pruning (per-depth linear margin, improving easing,
    verification band).
  - Razoring (per-depth margins, not-improving easing).
  - Null-move pruning with a cut-node gate and mate clamp.
  - Late move pruning, history pruning and SEE-based move-loop pruning.
  - Futility pruning in the move loop.
  - Late move reductions with PV, killer and history terms; re-search shaving.
  - ProbCut, singular extensions, check extensions, negative extensions,
    TT-score extension and a per-path extension cap.
  - Internal iterative reduction at PV, root and predicted cut nodes.
  - Quiescence search: TT probe, delta pruning, recapture-only narrowing,
    promotion cutoff, capture SEE ordering.
- Selectivity controls: twofold repetition detection, correction history,
  evaluation cache.

## Endgame tablebases

- Syzygy WDL + DTZ probing (3-4-5 piece, up to 7-man capable), material-gated
  and depth-unconditional in search and quiescence.
- Optional root probe (`SyzygyRootProbe`) for direct DTZ-optimal root play.
- 50-move-rule–aware WDL scoring and DTZ root ranking.

## Opening book

- Own-book support with `.ctg` (ChessBase) reading, configurable file and depth.
  `Own Book` defaults OFF (tournament rule friendly).

## UCI interface

- UCI handshake plus `isready`, `ucinewgame`, `position`, `go`, `stop`,
  `ponderhit`, `setoption`, `quit`.
- The engine always holds a position: it starts from the start position, and
  `ucinewgame` resets it to the start position.
- A rejected `position` command (invalid FEN or an illegal move) is reported
  with an `info string` error line and leaves the previous position in place.
- `go` supports depth, movetime, wtime/btime/winc/binc, movestogo, nodes, mate,
  infinite, ponder and searchmoves.
- `bench` command for deterministic node-count benchmarking.
- **Public options (10)**: `Hash`, `Threads`, `Move Overhead`, `Clear Hash`,
  `Own Book`, `Book File`, `Book Depth`, `SyzygyPath`, `SyzygyRootProbe`,
  `Syzygy50MoveRule`.
- **Research levers (60)**: hidden from the `uci` handshake but fully settable
  through `setoption name <Name> value <Value>` and `param:Name=Value`. Every
  one changes the search; a name outside the catalogue is rejected.
- Diagnostics: `perft`, `nnueeval`, `d`, `eval`, `checkmoves`, and the hidden
  `searchstats` / `nnueregions` counters.

## Portability and quality

- Windows 11 (x64) is the primary platform and the intended CCRL build; macOS
  (Apple Silicon) and Linux (x64) are built and tested from the same source as
  single binaries. Intel Macs and Linux on ARM are not supported.
- Windows ships three profile-guided variants: `v3` (AVX2 default), `v2`
  (SSE4.2 fallback) and `v4-vnni` (AVX-512 VNNI). macOS is built for
  `apple-m1` (M1–M5); Linux is built for `x86-64-v3` (AVX2).
- **Release builds use PGO on every platform**; the profile is generated on the
  build host and never committed.
- Make-free legality: checkers and pinned pieces are computed once per node and
  every pseudo-legal move is tested without being made.
- BMI2 `pext` slider attacks with a software fallback; TT bucket prefetch after
  every make and null move.
- Fused per-ply NNUE accumulator: one load/store pass per make, bit-identical to
  the cold-start accumulator.
- Allocation-free capture ordering and a shared evaluation cache.
- Deterministic fixed-depth node counts:
  `bench 6` **7,553**, `bench 12` **203,939**, `bench 16` **1,285,264**,
  `bench 18` **3,668,723**.
- 17 test targets: perft and legal move generation, search behaviour, NNUE
  loader/accumulator/evaluator, vector-width parity, tablebases, UCI plumbing,
  book parsing, allocation audit and a multi-thread stress walk.
- Every FEN used by a test is verified by an external UCI verifier at test time.
- Zero build errors and zero warnings; clippy clean.

## Version 1.0

- First stable release: **KnightByte 1.0**.
- **+20.17 ± 18.74 Elo** against the previous build in a 500-game match
  (LOS 98.3 %, clean), and **+2.0 plies in the same time** (time to depth 18
  ×0.57).
- **2.1–2.25×** the predecessor's speed on the identical search tree; 7.1×
  thread scaling at 8 threads.
- The differences to the frozen C# predecessor are summarised in
  [`docs/knightbyte-vs-pawnforge.md`](docs/knightbyte-vs-pawnforge.md).

## License

MIT. See `LICENSE` and `THIRD-PARTY-NOTICES.md`.
