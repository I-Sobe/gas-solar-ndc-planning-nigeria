# Thesis correction — standing rules

- Read the actual file before writing any edit. Never edit against grep output.
- Accuracy over agreement. Push back; don't concede to be agreeable.
- Results produced during correction are VERIFICATION artefacts, not thesis findings.
- Git: explicit `git add <paths>`, never `-A`. Read `git diff --cached` before commit.
- When a change alters the MEANING of a quantity, rename it so stale readers fail loudly.
- Before editing N consumers of a changed quantity, inventory all of them first.
- CSV runners need FLAT fields; JSON runners nest.
- genset_case is REQUIRED in load_econ (no default).
- Python 3.10, conda env `nigeria-energy`, Pyomo + CBC/HiGHS.
- Deferred-fix docket runs D1–D24; see docs/phase_2_6_handoff.md.
- Every negative ("X appears nowhere") reports the extent of the search: documents, sheets or page ranges, terms, and whether scans were read visually or by extracted text. See docs/literature_search_coverage.md.
- The first act of every task is to commit its own brief (TASK.md) to docs/tasks/TASK_<n>.md. Never reconstruct a missing brief from memory; see docs/tasks/README.md.
- Vintage-dispatch collapse check (TASK 76 C.1; supersedes any earlier "exactly"): with the flag on and the new-build efficiency set equal to gas_eta, (1) the decision-variable block is identical, (2) the objective is within 1e-7 relative of 24,986,620,949.18249 (CBC's own tolerance), (3) the tightened legacy-stream row's slack is positive everywhere at the REAL efficiency, and at e = gas_eta is at or above -1e-6 TWh_e (TASK 78 A.1). At equal efficiencies the two streams are interchangeable, so the split is undetermined and the legacy row sits at its bound without binding; zero slack there is expected, not a failure. An 8 USD objective difference from a different CBC vertex is not a failure either.
- Discovery is closed (TASK 80). A defect found during steps 2 to 8 is recorded in the register and NOT fixed, unless leaving it would make a reported result false; if it meets that bar, stop and report rather than repairing it in place.
