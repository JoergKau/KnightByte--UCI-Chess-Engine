"""The `bench` node pins, read from their single home: tests/golden/bench.json.

The PGO and release scripts import this module, so a search change re-pins in
one place. `tests/common/mod.rs` reads the same file for the Rust tests.
"""

import json
from pathlib import Path

BENCH_JSON = Path(__file__).resolve().parent.parent / "tests" / "golden" / "bench.json"


def load():
    """Returns {depth: total nodes} for the recorded bench depths."""
    data = json.loads(BENCH_JSON.read_text(encoding="utf8"))
    return {int(depth): int(entry["total_nodes"]) for depth, entry in data.items()}


def describe():
    """Returns the pins as 'a / b / c / d' with thousands separators, by depth."""
    pins = load()
    return " / ".join(f"{pins[depth]:,}" for depth in sorted(pins))
