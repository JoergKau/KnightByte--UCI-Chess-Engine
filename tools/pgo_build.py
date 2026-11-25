#!/usr/bin/env python3
"""Build a profile-guided (PGO) release binary of the engine.

1. builds an instrumented binary (-Cprofile-generate),
2. runs the deterministic bench workloads to collect profiles,
3. merges them with llvm-profdata,
4. rebuilds with -Cprofile-use.

The profile describes ONE host and ONE build configuration: generate it on the
machine (and with the target-cpu / feature set) the binary is built for, and
never reuse or check in the .profdata files. The host must be able to *execute*
the variant because the profile is collected by running the instrumented
binary; the build fails before compiling when rustc reports that CPU features
the variant requires are missing. PGO must not change the search:
the four bench node pins are verified on the result and the build fails if any
of them moves.

Requires the llvm-tools component:  rustup component add llvm-tools-preview

Usage:
    python tools/pgo_build.py
    python tools/pgo_build.py --target-cpu x86-64-v3 --work-dir target/pgo-work/v3 \\
        --output target/pgo/knightbyte-v3.exe
    python tools/pgo_build.py --target-cpu x86-64-v4 --target-features +avx512vnni
    python3 tools/pgo_build.py --target-cpu apple-m1 --output target/pgo/knightbyte-macos

`--target-cpu` builds a dedicated variant (for example x86-64-v3); without it
the executable is the portable build whose SIMD backend is chosen at run time.
The instrumented and the profile-use builds always use the same flags.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pins

REPO = Path(__file__).resolve().parent.parent

# Node pins of the four-position bench (depths 6, 12, 16, 18), read from
# tests/golden/bench.json, their single home.
PINS = pins.load()


def host_triple_and_sysroot():
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    version = subprocess.check_output(["rustc", "-vV"], text=True)
    host = next(line.split()[1] for line in version.splitlines() if line.startswith("host:"))
    return host, Path(sysroot)


def rust_flags(host, target_cpu, target_features):
    """The code-generation flags shared by both builds of one variant."""
    flags = []
    # The C runtime is linked statically on Windows: a plain Windows 10 machine
    # has no VCRUNTIME140.dll unless the Visual C++ redistributable was
    # installed, and the engine must start without it. (RUSTFLAGS replaces the
    # flags of .cargo/config.toml, so the flag is repeated here.)
    if host.endswith("windows-msvc"):
        flags.append("-Ctarget-feature=+crt-static")
    if target_cpu:
        flags.append(f"-Ctarget-cpu={target_cpu}")
    if target_features:
        flags.append(f"-Ctarget-feature={target_features}")
    return " ".join(flags)


def cargo_build(target_dir, rustflags):
    environment = dict(os.environ, RUSTFLAGS=rustflags)
    subprocess.run(
        ["cargo", "build", "--release", "--target-dir", str(target_dir)],
        cwd=REPO,
        env=environment,
        check=True,
    )


def feature_set(target_cpu, extra_features=""):
    """Returns the target-feature names rustc enables for a configuration.

    `target_cpu="native"` asks rustc for the features of the machine it runs
    on; the answer comes from the same LLVM CPU model that will compile the
    engine, so it matches what the generated code may use. An empty CPU means
    the target's generic baseline.
    """
    command = ["rustc", "--print", "cfg"]
    if target_cpu:
        command.append(f"-Ctarget-cpu={target_cpu}")
    if extra_features:
        command.append(f"-Ctarget-feature={extra_features}")
    output = subprocess.check_output(command, text=True)
    return {
        line.split('"')[1]
        for line in output.splitlines()
        if line.startswith('target_feature="')
    }


def missing_host_features(target_cpu, target_features=""):
    """Returns the features a variant needs but this machine lacks.

    The profile is collected by *executing* the instrumented build, so a
    variant compiled for an instruction set the host does not provide cannot be
    profiled here, even though cargo can cross-compile it. The list is empty
    when the host can run the variant; a portable build (no `--target-cpu`) is
    never rejected.

    `target_features` is the same comma-separated list that is handed to
    rustc, so `+avx512vnni` is part of the required set.

    Only the named x86-64 baseline CPUs are checked. The Apple Silicon build
    uses `apple-m1`, whose LLVM feature names are not a stable subset across
    M-series chips, and the macOS release script already verifies the host
    architecture; leaving that target alone avoids a false rejection there.
    """
    if not target_cpu or not target_cpu.startswith("x86-64"):
        return []
    return sorted(feature_set(target_cpu, target_features) - feature_set("native"))


# The Windows status code for an instruction the CPU cannot execute. It is
# reported unsigned on current Python versions and signed on older ones.
STATUS_ILLEGAL_INSTRUCTION = 0xC000001D


def bench_nodes(binary, depth):
    """Runs `bench <depth>` through standard input and returns the node total."""
    result = subprocess.run(
        [str(binary)],
        input=f"bench {depth}\nquit\n",
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Nodes:"):
            return int(line.split(":")[1].strip())
    raise RuntimeError(f"bench {depth} printed no node total")


def build_pgo(
    work_dir="target/pgo-work/default",
    output="target/pgo/knightbyte.exe",
    target_cpu="",
    target_features="",
    keep_work=True,
    verify_pins=True,
):
    """Builds the PGO binary, verifies the pins and copies it to `output`.

    The scratch folder holds two complete cargo builds (about 120 MB). It lives
    under target/ (git-ignored, removed by `cargo clean`) and is deleted after a
    successful build unless `keep_work` is set.
    """
    # Fail before the first compile when the host cannot run the result: the
    # profile is collected by executing the instrumented binary.
    missing = missing_host_features(target_cpu, target_features)
    if missing:
        raise SystemExit(
            f"this machine cannot run a build compiled for {target_cpu} "
            f"(missing {', '.join(missing)}). "
            "The PGO profile is collected by running the instrumented build, "
            "so build this variant on a CPU that supports it."
        )

    host, sysroot = host_triple_and_sysroot()
    suffix = ".exe" if os.name == "nt" else ""
    profdata = sysroot / "lib" / "rustlib" / host / "bin" / f"llvm-profdata{suffix}"
    if not profdata.exists():
        raise SystemExit("llvm-profdata not found; run: rustup component add llvm-tools-preview")

    flags = rust_flags(host, target_cpu, target_features)
    print(f"rustflags: {flags}")

    work = (REPO / work_dir).resolve()
    raw = work / "raw"
    merged = work / "merged.profdata"
    shutil.rmtree(work, ignore_errors=True)
    raw.mkdir(parents=True)

    print("== 1/4 instrumented build")
    cargo_build(work / "gen", f"{flags} -Cprofile-generate={raw}".strip())

    print("== 2/4 collecting profiles")
    instrumented = work / "gen" / "release" / f"knightbyte{suffix}"
    try:
        for depth in (12, 16):
            bench_nodes(instrumented, depth)
    except subprocess.CalledProcessError as error:
        if error.returncode & 0xFFFFFFFF == STATUS_ILLEGAL_INSTRUCTION:
            raise SystemExit(
                "the instrumented build stopped with STATUS_ILLEGAL_INSTRUCTION: "
                f"this machine cannot execute the {target_cpu or 'portable'} build. "
                "Collect the profile on a CPU that supports the variant."
            ) from None
        raise

    print("== 3/4 merging")
    subprocess.run([str(profdata), "merge", "-o", str(merged), str(raw)], check=True)

    print("== 4/4 profile-use build")
    cargo_build(work / "use", f"{flags} -Cprofile-use={merged}".strip())

    binary = work / "use" / "release" / f"knightbyte{suffix}"
    print("== verifying the node pins")
    for depth, expected in sorted(PINS.items()):
        nodes = bench_nodes(binary, depth)
        if nodes != expected:
            if verify_pins:
                raise SystemExit(
                    f"PGO build moved the bench-{depth} pin: {nodes} (expected {expected}) - do not ship"
                )
            print(f"  bench {depth}: {nodes:,} (old pin {expected:,}; new default set)")
        else:
            print(f"  bench {depth}: {nodes:,} (pin exact)")

    destination = Path(output)
    if not destination.is_absolute():
        destination = REPO / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, destination)
    print(f"PGO binary: {destination} ({'pins exact' if verify_pins else 'measurement build, pins not enforced'})")
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work-dir", default="target/pgo-work/default", help="scratch directory (under target/)")
    parser.add_argument("--clean", action="store_true", help="delete the scratch directory after a successful build")
    default_output = "target/pgo/knightbyte.exe" if os.name == "nt" else "target/pgo/knightbyte"
    parser.add_argument("--output", default=default_output, help="where the binary is copied")
    parser.add_argument("--target-cpu", default="", help="for example x86-64-v3 (empty: portable build)")
    parser.add_argument("--target-features", default="", help="for example +avx512vnni")
    parser.add_argument(
        "--skip-pin-check",
        action="store_true",
        help="measurement build only: report the bench totals instead of failing when they move "
             "(use when the defaults themselves changed and have not been re-pinned yet)",
    )
    args = parser.parse_args()
    build_pgo(args.work_dir, args.output, args.target_cpu, args.target_features,
              keep_work=not args.clean, verify_pins=not args.skip_pin_check)


if __name__ == "__main__":
    sys.exit(main())
