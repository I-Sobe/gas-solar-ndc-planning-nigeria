"""
make_results_manifest.py -- hash manifest of a results tree.

TASK 00 (docs/task_log_00_branch_provenance.md), step 4. Produces a
Markdown manifest of every file under a results root: relative path,
SHA-256, byte size, and modification time (UTC). Used once now, to
freeze the pre-Phase-B state (`docs/results_manifest_pre_phase_b.md`),
and again after Phase B, against the same root, so the two manifests
can be diffed to show exactly what changed.

Takes the results root and the output path as positional arguments.
Neither is hardcoded, so the same script serves both runs.

Tag SHA in the header: a manifest is typically produced BEFORE the
tag it precedes exists yet (the commit order puts the tag last, after
the manifest is already committed), so the tag usually cannot be
resolved the run that matters most. The script looks up the tag named
by `--tag` (default `pre-phase-b`, TASK 00's own tag) via
`git rev-parse` and falls back to reporting current HEAD, labelled as
such, when that tag is not yet present -- this makes the script
correct both for a pre-tag run and for any later run made after the
tag exists, with no special-casing by the caller. `--tag` must be
passed explicitly for any tag other than `pre-phase-b` (e.g. TASK 15's
`phase-b-start`) -- the default exists only to keep TASK 00's original
invocation unchanged, not because `pre-phase-b` is this script's only
valid tag.
"""
import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git(args, cwd):
    try:
        out = subprocess.run(
            ["git"] + args, cwd=cwd, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def _resolve_tag_sha(repo_root, tag_name):
    sha = _git(["rev-parse", tag_name], repo_root)
    if sha:
        return f"{sha} ({tag_name})"
    head = _git(["rev-parse", "HEAD"], repo_root)
    return f"{head} ({tag_name} tag not yet created; HEAD at generation time shown instead)"


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(results_root: Path, repo_root: Path):
    rows = []
    for path in sorted(results_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(results_root).as_posix()
        stat = path.stat()
        mtime_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        rows.append(
            {
                "path": rel,
                "sha256": _sha256_of(path),
                "bytes": stat.st_size,
                "mtime_utc": mtime_utc,
            }
        )
    rows.sort(key=lambda r: r["path"])
    return rows


def write_manifest(rows, results_root, output_path, repo_root, tag_name):
    total_bytes = sum(r["bytes"] for r in rows)
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tag_sha = _resolve_tag_sha(repo_root, tag_name)

    lines = [
        f"# Results manifest -- {results_root.as_posix()}",
        "",
        f"- File count: {len(rows)}",
        f"- Total size: {total_bytes} bytes",
        f"- Tag SHA: {tag_sha}",
        f"- Generated (UTC): {generated_utc}",
        "",
        "| path | sha256 | bytes | mtime_utc |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['path']} | {r['sha256']} | {r['bytes']} | {r['mtime_utc']} |")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Write a SHA-256 hash manifest for every file under a results root."
    )
    parser.add_argument("results_root", type=Path, help="Directory to manifest (e.g. results/)")
    parser.add_argument("output_path", type=Path, help="Markdown file to write the manifest to")
    parser.add_argument(
        "--tag",
        default="pre-phase-b",
        help="Git tag this manifest precedes (default: pre-phase-b, TASK 00's tag). "
        "Pass the tag explicitly for any other run, e.g. --tag phase-b-start.",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    if not results_root.is_dir():
        raise SystemExit(f"Not a directory: {results_root}")

    repo_root = Path(__file__).resolve().parents[1]

    rows = build_manifest(results_root, repo_root)
    write_manifest(rows, results_root, args.output_path, repo_root, args.tag)
    print(f"Wrote {len(rows)} entries to {args.output_path}")


if __name__ == "__main__":
    main()
