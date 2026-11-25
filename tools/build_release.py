#!/usr/bin/env python3
"""Build, verify and package the Windows x64 release artifacts.

Before anything is built the lever effect gate runs (every research lever
must change the search). Then, for every variant, the script

  1. builds a profile-guided executable with tools/pgo_build.py (its own
     profile, the C runtime linked statically; the four bench node pins are
     verified),
  2. runs the file: `uci` must answer `uciok`, `bench 6` must report the pinned
     node count, and a run forced to the scalar kernels must report the same,
  3. checks the import table: only Windows system DLLs, and no Visual C++
     runtime (a plain Windows 10 machine does not have it),
  4. records the SHA-256 of the file.

It then writes the archive and its .sha256 checksum. The zip contains the
executables and a short README with the variant table; the full documentation
lives in the repository.

Profiles describe one host: run this on the machine whose CPU supports the most
capable variant (the v4-vnni build is profiled by running it). A machine
without AVX-512 can build `v2` and `v3` only; the script checks every requested
variant against the CPU before it builds anything and names the missing
features, so an unsupported variant cannot fail halfway through the profile
run.

Usage:
    python tools/build_release.py
    python tools/build_release.py --variants v2 v3 --out-dir dist
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

if sys.platform != "win32":
    sys.exit(
        "build_release.py builds the Windows x64 release.\n"
        "On macOS use: python3 tools/build_release_macos.py\n"
        "On Linux use: python3 tools/build_release_linux.py"
    )

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pgo_build  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Variant table: the executable's compile-time baseline.
VARIANTS = {
    "v2": {"cpu": "x86-64-v2", "features": ""},
    "v3": {"cpu": "x86-64-v3", "features": ""},
    "v4-vnni": {"cpu": "x86-64-v4", "features": "+avx512vnni"},
}

ALLOWED_DLLS = {
    "kernel32.dll",
    "ntdll.dll",
    "bcryptprimitives.dll",
    "api-ms-win-core-synch-l1-2-0.dll",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bench6_nodes(exe, environment=None):
    result = subprocess.run(
        [str(exe)],
        input="bench 6\nquit\n",
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, **(environment or {})),
    )
    for line in result.stdout.splitlines():
        if line.startswith("Nodes:"):
            return int(line.split(":")[1].strip())
    raise RuntimeError("bench 6 printed no node total")


def imported_dlls(exe):
    output = subprocess.run(
        [sys.executable, str(REPO / "tools" / "pe_imports.py"), str(exe)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted({line.split(" ")[0].lower() for line in output.splitlines() if line and not line[0].isspace()})


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def check_variants_for_host(variants):
    """Rejects variants this machine cannot profile before anything is built.

    PGO step 2 collects the profile by running the instrumented executable, so
    the host must support the variant's instruction set as well as compile it.
    The check runs before the lever gate and before `dist/` is replaced; the
    message names the missing features and the command for the variants that
    this machine can build.
    """
    unsupported = {
        variant: pgo_build.missing_host_features(VARIANTS[variant]["cpu"], VARIANTS[variant]["features"])
        for variant in variants
    }
    unsupported = {variant: missing for variant, missing in unsupported.items() if missing}
    if not unsupported:
        return

    supported = [variant for variant in variants if variant not in unsupported]
    lines = [
        "this machine cannot collect a PGO profile for every requested variant:",
        *(f"  {variant}: missing {', '.join(missing)}" for variant, missing in unsupported.items()),
        "A profile is collected by running the instrumented build, so a variant",
        "can only be built on a CPU that supports it.",
    ]
    if supported:
        lines.extend(
            [
                "Build the variants this machine supports with:",
                f"  python tools\\build_release.py --variants {' '.join(supported)}",
            ]
        )
    if "v4-vnni" in unsupported:
        lines.append("The v4-vnni variant needs a CPU with AVX-512 VNNI.")
    else:
        lines.append("Run the remaining variants on a host that supports them.")
    raise SystemExit("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default="dist")
    parser.add_argument("--variants", nargs="+", default=["v2", "v3", "v4-vnni"], choices=sorted(VARIANTS))
    parser.add_argument("--skip-lever-gate", action="store_true", help="skip the lever effect gate")
    parser.add_argument("--keep-work", action="store_true", help="keep the scratch builds under target/pgo-work")
    args = parser.parse_args()

    check_variants_for_host(args.variants)

    cargo_toml = (REPO / "Cargo.toml").read_text(encoding="utf8")
    version = re.search(r'^version\s*=\s*"([^"]+)"', cargo_toml, re.M).group(1)
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    stem = f"knightbyte-{version}-windows-x64"

    out = (REPO / args.out_dir).resolve()
    staging = out / stem
    shutil.rmtree(out, ignore_errors=True)
    staging.mkdir(parents=True)

    if not args.skip_lever_gate:
        print("=== lever effect gate")
        subprocess.run(["cargo", "test", "--release", "--test", "lever_effect"], cwd=REPO, check=True)

    for variant in args.variants:
        spec = VARIANTS[variant]
        print(f"=== variant {variant} (target-cpu {spec['cpu']} {spec['features']})")
        exe = staging / f"knightbyte-{variant}.exe"
        pgo_build.build_pgo(
            work_dir=f"target/pgo-work/{variant}",
            keep_work=args.keep_work,
            output=str(exe),
            target_cpu=spec["cpu"],
            target_features=spec["features"],
        )

        # Behaviour checks on the packaged file itself.
        uci = subprocess.run([str(exe)], input="uci\nquit\n", capture_output=True, text=True, check=True).stdout
        if "uciok" not in uci:
            raise SystemExit(f"{variant} did not answer uciok")
        nodes = bench6_nodes(exe)
        if nodes != pgo_build.PINS[6]:
            raise SystemExit(f"{variant} bench 6 reported {nodes} nodes")
        # The slowest backend must search the identical tree (kernel-identity gate).
        scalar = bench6_nodes(exe, {"KNIGHTBYTE_FORCE_ISA": "scalar"})
        if scalar != nodes:
            raise SystemExit(f"{variant} forced scalar reported {scalar} nodes")

        dlls = imported_dlls(exe)
        unexpected = [dll for dll in dlls if dll not in ALLOWED_DLLS]
        if unexpected:
            raise SystemExit(f"{variant} imports unexpected DLLs: {', '.join(unexpected)}")

        digest = sha256(exe)
        print(f"  ok: {nodes} nodes, forced scalar identical, imports clean, sha256 {digest}")

    # The Windows zip carries only the executables and a short README; the full
    # documentation stays in the repository.
    baselines = {
        "v3": "AVX2, BMI2, FMA",
        "v2": "SSE4.2, POPCNT",
        "v4-vnni": "AVX-512 + VNNI",
    }
    table_order = ("v3", "v2", "v4-vnni")
    rows = "\n".join(
        f"| Windows x64, `{variant}`{' (default)' if variant == 'v3' else ''} "
        f"| `knightbyte-{variant}.exe` | {baselines[variant]} |"
        for variant in table_order
        if variant in args.variants
    )
    readme = (
        f"# KnightByte {version} — Windows x64\n\n"
        "KnightByte is a UCI chess engine. All executables below are the same\n"
        "engine; they differ only in the instruction set they were compiled for.\n\n"
        "| Platform | File | Baseline |\n"
        "| --- | --- | --- |\n"
        f"{rows}\n\n"
        "Run the build with the highest baseline your CPU supports; `v3` is the\n"
        "default.\n\n"
        "KnightByte is MIT licensed. Source and documentation:\n"
        "https://github.com/JoergKau/KnightByte--UCI-Chess-Engine\n"
    )
    with open(staging / "README.md", "w", encoding="utf8", newline="\n") as handle:
        handle.write(readme)

    archive = out / f"{stem}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(staging / "README.md", "README.md")
        for variant in table_order:
            if variant in args.variants:
                exe = staging / f"knightbyte-{variant}.exe"
                bundle.write(exe, exe.name)
    archive_hash = sha256(archive)
    with open(out / f"{stem}.zip.sha256", "w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{archive_hash} *{archive.name}\n")

    print()
    print(f"Release: {archive}")
    print(f"sha256 : {archive_hash}")
    if dirty:
        print("WARNING: the working tree has uncommitted changes", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
