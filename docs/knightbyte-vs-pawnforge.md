# KnightByte versus PawnForgeCS — short summary

**KnightByte 1.0** is the Rust migration of PawnForgeCS 2.0 (C#/.NET 11). Both
engines share the same network and the same search design; PawnForgeCS is frozen,
and KnightByte continues on its own.

## What is the same

- **Network and evaluation**: the same self-designed v5 network
  (`jokausch.nnue`, 3,147,265 parameters) with bit-identical evaluations.
- **Core search**: iterative deepening, aspiration windows, PVS, a lockless
  transposition table, staged move picking, quiescence, Lazy SMP, Syzygy
  WDL/DTZ probing and the `.ctg` opening book.
- **UCI surface** and the fixes both engines received.

## What KnightByte adds

| Area | KnightByte 1.0 | PawnForgeCS 2.0 |
| :--- | :--- | :--- |
| **Search strength** | tuned defaults (+33 Elo) and the new pruning stack (+20 Elo): reverse futility with a linear margin and verified band, internal iterative reduction at PV, root and cut nodes | the reference defaults |
| **Time to depth** | ×0.57 time to depth 18 and +2.0 plies in 2 s against the previous build | the reference search |
| **Node counting** | every node counted once (bug B56 fixed) | a leaf handed to quiescence is counted twice, so totals and NPS are about 1.4× too high |
| **Speed** | 2.1–2.25× PawnForgeCS's nodes per second on the identical tree; 7.1× at 8 threads | .NET JIT with tiered PGO and Server GC; 5.0–5.4× at 8 threads |
| **Levers** | 10 public options and exactly 60 hidden research levers, each proven to change the search | about 80 levers with a sampled effect test |
| **Code layout** | search rules in small stage files, every pure rule unit-tested, no source file over 800 lines | one large negamax file |

KnightByte also removed the forced-reply extension (it never fired), the
migration trace ledger and a number of levers whose features measured flat or
negative.

## What is not known

- No direct match of KnightByte 1.0 against PawnForgeCS 2.0 has been played.
- No absolute rating exists for either engine; all Elo figures are self-play
  with an error bar of about ±19.
- The release gate of the first KnightByte release scored **75.1 %** against
  PawnForgeCS over 500 games, including the equal-time speed advantage.
