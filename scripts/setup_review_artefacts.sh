#!/usr/bin/env bash
#
# setup_review_artefacts.sh
#
# Copies the frozen PRISMA systematic-review artefacts into the thesis
# repository under data/review/ and docs/review/, records a SHA-256
# manifest, writes a provenance README, and stages everything for commit.
#
# It does NOT commit. Review `git diff --cached --stat` first, per the
# project's review-before-commit convention.
#
# Run from the repository root:
#     cd ~/thesis-code
#     chmod +x scripts/setup_review_artefacts.sh
#     ./scripts/setup_review_artefacts.sh
#
set -euo pipefail

# ---------------------------------------------------------------------
# 1. EDIT THIS LINE ONLY.
#    Absolute path to the folder holding the nine source files.
#    From WSL, a Windows folder looks like:
#       /mnt/c/Users/<you>/Downloads/review_artefacts
# ---------------------------------------------------------------------
SRC="/mnt/c/Users/USER/Downloads/review_artefacts"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/review"
DOCS_DIR="${REPO_ROOT}/docs/review"
MANIFEST="${DATA_DIR}/MANIFEST.sha256"

say()  { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------
# 2. Preconditions
# ---------------------------------------------------------------------
say "=== Preconditions ==="

[[ -d "${SRC}" ]] || fail "source folder not found: ${SRC}
       Edit SRC on line 20 of this script."

git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || fail "not a git repository: ${REPO_ROOT}"

say "repo   : ${REPO_ROOT}"
say "source : ${SRC}"
say "branch : $(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  say ""
  say "NOTE: working tree is not clean. Files staged by this script will"
  say "      sit alongside your existing changes."
  if [[ -t 0 ]]; then
    say "      Continue? [y/N]"
    read -r reply
    [[ "${reply}" =~ ^[Yy]$ ]] || fail "aborted by user"
  else
    say "      (non-interactive run; continuing)"
  fi
fi

# ---------------------------------------------------------------------
# 3. File map:  <source filename> | <destination relative to repo root>
#    Marked REQUIRED or OPTIONAL. A missing REQUIRED file aborts the run
#    before anything is copied, so a partial run cannot leave a
#    half-populated folder.
# ---------------------------------------------------------------------
FILES=(
  "SLR_Protocol_v5_SSA_PREREG.md|docs/review/protocol_v5_ssa_prereg.md|REQUIRED"
  "Stage1_SLR_Protocol.md|docs/review/protocol_stage1.md|REQUIRED"
  "OSF_amendment_screening.md|docs/review/amendment_screening.md|REQUIRED"
  "screening_rulebook.md|docs/review/screening_rulebook.md|REQUIRED"
  "Stage2_Search_Strings_REBUILT.md|docs/review/search_strings_stage2.md|REQUIRED"
  "SSA_extraction_workbook_v5_Results.xlsx|data/review/extraction_workbook_v5.xlsx|REQUIRED"
  "SSA_fulltext_working_sheet_AutoRecovered_.xlsx|data/review/fulltext_screening_sheet.xlsx|REQUIRED"
  "Limitations_register.docx|docs/review/limitations_register.docx|REQUIRED"
  "Final.docx|docs/review/manuscript_final.docx|REQUIRED"
  "HANDOVER_chapter_2.md|docs/review/handover_chapter_2.md|OPTIONAL"
  "OSF_amendment_endogeneity.md|docs/review/amendment_endogeneity.md|OPTIONAL"
)

say ""
say "=== Checking sources ==="
missing=0
for entry in "${FILES[@]}"; do
  IFS='|' read -r src dest req <<< "${entry}"
  if [[ -f "${SRC}/${src}" ]]; then
    say "  found    ${src}"
  elif [[ "${req}" == "REQUIRED" ]]; then
    say "  MISSING: ${src}"
    missing=$(( missing + 1 ))
  else
    say "  skipped  ${src}  (optional, not present)"
  fi
done

[[ ${missing} -eq 0 ]] || fail "${missing} required file(s) missing. Nothing copied."

# ---------------------------------------------------------------------
# 4. Copy
# ---------------------------------------------------------------------
say ""
say "=== Copying ==="
mkdir -p "${DATA_DIR}" "${DOCS_DIR}"

copied=0
for entry in "${FILES[@]}"; do
  IFS='|' read -r src dest req <<< "${entry}"
  [[ -f "${SRC}/${src}" ]] || continue
  cp -p "${SRC}/${src}" "${REPO_ROOT}/${dest}"
  say "  copied -> ${dest}"
  copied=$(( copied + 1 ))
done

# ---------------------------------------------------------------------
# 5. Manifest
# ---------------------------------------------------------------------
say ""
say "=== Manifest ==="
{
  printf '# SHA-256 manifest for the frozen PRISMA review artefacts\n'
  printf '# Generated: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf '# Verify with:  sha256sum -c data/review/MANIFEST.sha256\n'
  printf '# (run from the repository root)\n\n'
} > "${MANIFEST}"

cd "${REPO_ROOT}"
for entry in "${FILES[@]}"; do
  IFS='|' read -r src dest req <<< "${entry}"
  [[ -f "${dest}" ]] || continue
  sha256sum "${dest}" >> "${MANIFEST}"
done
say "  MANIFEST written (${copied} files) -> data/review/MANIFEST.sha256"

# ---------------------------------------------------------------------
# 6. Provenance README
# ---------------------------------------------------------------------
cat > "${DATA_DIR}/README.md" <<'EOF'
# data/review — frozen evidence from the systematic review

These are the evidence artefacts behind the thesis's Chapter 2 and the
standalone review manuscript. They are frozen: nothing here is edited by
analysis code. Scripts read from these files and write derived tables to
`docs/`.

| File | What it is |
|---|---|
| `extraction_workbook_v5.xlsx` | Per-study extraction, codebook, synthesis matrix, Results panels, and a changelog of every correction applied to v4. |
| `fulltext_screening_sheet.xlsx` | Full-text screening decisions, exclusion reasons, PDF retrieval status, and stage tallies. |
| `MANIFEST.sha256` | Checksums for every artefact, in `data/` and `docs/review/` alike. |

Protocol, amendments, screening rulebook, search strings, the limitations
register and the manuscript live in `docs/review/`.

## Corpus, as at this freeze

- Included studies: **75**
- Financing treatment: exogenous 65, absent 10, endogenous 0
- Dimensional coverage: availability 75, deliverability 59, access 44,
  affordability 75, acceptability 50
- Deliverability/access co-occurrence: both 39, deliverability-only 20,
  access-only 5, neither 11
- Enacted-not-named: 38 of the 39 studies representing both dimensions

Any script that recomputes these and disagrees should report the
disagreement rather than correct either side. The counts above are the
reconciliation target, not an assumption.

## Verifying integrity

    sha256sum -c data/review/MANIFEST.sha256

Re-run `scripts/setup_review_artefacts.sh` after replacing any artefact,
so the manifest tracks the current files.
EOF
say "  README written -> data/review/README.md"

# ---------------------------------------------------------------------
# 7. Size check — warn if anything is large enough to want Git LFS
# ---------------------------------------------------------------------
say ""
say "=== Size check ==="
total_kb=0
for entry in "${FILES[@]}"; do
  IFS='|' read -r src dest req <<< "${entry}"
  [[ -f "${dest}" ]] || continue
  kb=$(( $(stat -c%s "${dest}") / 1024 ))
  total_kb=$(( total_kb + kb ))
  if [[ ${kb} -gt 5120 ]]; then
    say "  WARNING: ${dest} is ${kb} KB — consider Git LFS"
  fi
done
say "  total staged payload: ${total_kb} KB"
[[ ${total_kb} -lt 10240 ]] \
  && say "  (well inside normal git limits; no LFS needed)"

# ---------------------------------------------------------------------
# 8. Stage, do not commit
# ---------------------------------------------------------------------
say ""
say "=== Staging ==="
git add data/review docs/review
git status --short -- data/review docs/review

say ""
say "=== Done ==="
say "Nothing has been committed. Review, then commit:"
say ""
say "    git diff --cached --stat"
say "    git commit -m 'docs(review): add frozen PRISMA review artefacts and manifest'"
say ""
