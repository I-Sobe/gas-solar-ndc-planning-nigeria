"""
Output-naming helpers: short, deterministic path components for a full
arm tag that exceeds the filesystem's ~255-byte per-path-component limit.

Lifted from `scripts/26_run_f3_identification.py`'s hash8 scheme
(`docs/f3_identification.md`, "Path-length fix") once the same failure
hit `01b_run_baseline_sliced.py`, `02_run_ndc_caps.py` and
`03_run_ndc_eaas.py` (`docs/output_naming.md`). Semantics are copied
EXACTLY -- same hash algorithm, same length, same INDEX.md line format --
so hashes already written by `26` remain valid and comparable.

Scope:
- Deriving a short, deterministic path component from a long arm tag.
- Recording the hash8 -> full-tag mapping so a results directory stays
  navigable without opening every output file.

Non-scope (by design):
- Deciding WHAT the arm tag contains -- callers build that string
  themselves from their own required-arg discipline.
- Writing the full tag into the output file's own content (JSON/CSV) --
  callers do that too; this module only names the path.
"""

import hashlib
from pathlib import Path


def hash8_of(arm_tag: str) -> str:
    """
    First 8 hex characters of the SHA-1 digest of the FULL arm tag.

    Deterministic: the same arm_tag string always yields the same hash8,
    so re-running a configuration overwrites its prior output rather than
    accumulating a new file/directory alongside it.
    """
    return hashlib.sha1(arm_tag.encode("utf-8")).hexdigest()[:8]


def om_tag_suffix(om_case: str, sep: str = "__") -> str:
    """
    Tag suffix that puts a NON-central O&M arm into the hashed arm tag.

    Empty for `om_central`, so every hash8 written before this existed (all of
    them om_central) is unchanged; `f"{sep}{om_case}"` for om_none / om_low /
    om_high, so the three O&M arms can no longer share one path (TASK 48 B.1:
    Phase C Group 1's om_low result was overwritten by om_high at hash 02c17f55).
    `sep` matches the runner's own tag separator ("__" in 01b_, "_" in 02_/03_).
    """
    return "" if om_case == "om_central" else f"{sep}{om_case}"


def capital_window_tag_suffix(capital_window_case: str, sep: str = "__") -> str:
    """
    Tag suffix for TASK 51's capital_window_case, mirroring om_tag_suffix:
    empty for the default 'advance' (so every hash written before the arm
    existed stays valid), f"{sep}{case}" otherwise.
    """
    return "" if capital_window_case == "advance" else f"{sep}{capital_window_case}"


def short_tag(arm_tag: str, prefix: str) -> str:
    """
    `f"{prefix}_{hash8}"` -- a short, filesystem-safe stand-in for a full
    arm tag, for use as a DIRECTORY component (pattern A in
    docs/output_naming.md). `prefix` should be short and stable (e.g. an
    NDC case name), not itself derived from the full tag.

    The full arm_tag is NOT recoverable from this string alone -- callers
    must still write it into the output file's own content (as
    `"arm_tag"` in a JSON summary, or a header comment in a CSV) and call
    `append_index()` so the mapping is recorded.
    """
    return f"{prefix}_{hash8_of(arm_tag)}"


def append_index(results_dir, hash8: str, arm_tag: str) -> None:
    """
    Append one `` `hash8` -> `arm_tag` `` line to `results_dir/INDEX.md`,
    so the directory is navigable without opening every output file.

    Append-only (a hash8 is never overwritten once recorded) and
    deduplicated (re-running the same configuration, and so producing the
    same line again, does not grow the file). Line format is copied
    verbatim from `26_run_f3_identification.py`'s original implementation
    so `26`'s existing `INDEX.md` entries and any new ones interleave
    without a format seam.
    """
    index_path = Path(results_dir) / "INDEX.md"
    index_line = f"- `{hash8}` -> `{arm_tag}`\n"
    existing = index_path.read_text() if index_path.exists() else ""
    if index_line not in existing:
        with open(index_path, "a") as f:
            f.write(index_line)
