"""version.py - bump the engine version, tag the commit and push it.

The version lives in `Cargo.toml`; the banner and the UCI `id name` use its
`major.minor` part (for example `KnightByte 1.0`), while the full semver stays
available for internal builds. A release is therefore a version commit plus the
matching annotated tag, pushed together.

USAGE
  python tools/version.py show
  python tools/version.py bump patch          # 1.0.0 -> 1.0.1
  python tools/version.py bump minor          # 1.0.1 -> 1.1.0
  python tools/version.py bump major          # 1.1.0 -> 2.0.0
  python tools/version.py set 1.0.0
  python tools/version.py bump patch --push   # commit + tag + push branch/tag
  python tools/version.py bump patch --push -m "search bake"

The default is to update Cargo.toml/Cargo.lock and commit them (and create the
tag) but not push. `--push` pushes the branch and the tag to origin.

Safety rules:
  * the worktree must be clean apart from Cargo.toml/Cargo.lock;
  * the target tag must not exist;
  * the version must be valid semver.
Commit your actual changes first, then run this. The version commit stays
separate, which makes the release line readable.
"""
import argparse
import os
import re
import subprocess
import sys

CARGO = "Cargo.toml"
LOCK = "Cargo.lock"


def run(*args, check=True):
    result = subprocess.run(args, capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.exit(f"{' '.join(args)} failed:\n{result.stderr.strip()}")
    return result


def git(*args, check=True):
    return run("git", *args, check=check)


def current_version():
    text = open(CARGO, encoding="utf-8").read()
    section = re.search(r"^\[package\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if not section:
        sys.exit("Cargo.toml has no [package] section")
    match = re.search(r'^version\s*=\s*"([^"]+)"', section.group(1), re.M)
    if not match:
        sys.exit("Cargo.toml [package] has no version")
    return match.group(1)


def parse_version(text):
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", text)
    if not match:
        sys.exit(f"{text!r} is not a semver version")
    return tuple(int(part) for part in match.groups())


def bump(text, kind):
    major, minor, patch = parse_version(text)
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def write_version(version):
    text = open(CARGO, encoding="utf-8").read()
    section = re.search(r"^\[package\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if not section:
        sys.exit("Cargo.toml has no [package] section")
    body = section.group(1)
    new_body = re.sub(
        r'^version\s*=\s*"[^"]+"', f'version = "{version}"', body, count=1, flags=re.M
    )
    start, end = section.span(1)
    open(CARGO, "w", encoding="utf-8", newline="\n").write(text[:start] + new_body + text[end:])

    if not os.path.exists(LOCK):
        return
    lock = open(LOCK, encoding="utf-8").read()
    block = re.search(
        r'(\[\[package\]\]\nname = "knightbyte"\nversion = ")([^"]+)(")',
        lock,
        re.M,
    )
    if block:
        lock = lock[: block.start(2)] + version + lock[block.end(2) :]
        open(LOCK, "w", encoding="utf-8", newline="\n").write(lock)


def clean_worktree():
    result = git("status", "--porcelain")
    dirty = [line for line in result.stdout.splitlines() if line.strip()]
    allowed = {"M  Cargo.toml", "M  Cargo.lock", " M Cargo.toml", " M Cargo.lock",
               "M Cargo.toml", "M Cargo.lock"}
    unexpected = [line for line in dirty if line not in allowed]
    if unexpected:
        sys.exit(
            "worktree is not clean; commit your changes first, then bump the version:\n"
            + "\n".join(unexpected)
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="print current version, commit and tags")
    bump_cmd = sub.add_parser("bump", help="bump major/minor/patch")
    bump_cmd.add_argument("kind", choices=("major", "minor", "patch"))
    set_cmd = sub.add_parser("set", help="set an explicit version")
    set_cmd.add_argument("version")
    for parser in (bump_cmd, set_cmd):
        parser.add_argument("--push", action="store_true", help="push branch and tag to origin")
        parser.add_argument("--no-commit", action="store_true", help="only edit the files")
        parser.add_argument("--no-tag", action="store_true", help="do not create a tag")
        parser.add_argument("-m", "--message", default="", help="extra commit/tag message")
        parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    args = ap.parse_args()

    if args.command == "show":
        tag = git("describe", "--tags", "--always", check=False).stdout.strip()
        print(f"version: {current_version()}")
        print(f"commit:  {git('rev-parse', '--short=12', 'HEAD').stdout.strip()}")
        print(f"describe:{' ' if tag else ' '}{tag or '(no tag)'}")
        return

    old = current_version()
    new = bump(old, args.kind) if args.command == "bump" else args.version
    parse_version(new)
    tag_name = f"v{new}"
    if not args.no_tag and git("rev-parse", "-q", "--verify", f"refs/tags/{tag_name}", check=False).returncode == 0:
        sys.exit(f"tag {tag_name} already exists")

    print(f"version: {old} -> {new}   tag: {'(none)' if args.no_tag else tag_name}")
    if args.dry_run:
        return
    clean_worktree()
    write_version(new)
    print(f"updated {CARGO}" + (f" and {LOCK}" if os.path.exists(LOCK) else ""))

    if args.no_commit:
        return
    git("add", CARGO, LOCK)
    subject = f"Release {new}"
    if args.message:
        subject += f": {args.message}"
    git("commit", "-m", subject)
    if not args.no_tag:
        annotation = args.message or f"KnightByte {new}"
        git("tag", "-a", tag_name, "-m", annotation)
    if args.push:
        branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        git("push", "origin", branch)
        if not args.no_tag:
            git("push", "origin", tag_name)
        print(f"pushed {branch} and {tag_name}")
    else:
        print(f"committed {subject}" + ("" if args.no_tag else f" and tagged {tag_name}"))
        print("push with: git push origin HEAD" + ("" if args.no_tag else f" {tag_name}"))


if __name__ == "__main__":
    main()
