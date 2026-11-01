# tests/

There is no pytest suite here, on purpose.

`tests/test_run.py` (removed, TASK 53 2.11) was a bare module-level script with no assertions, it
failed at import (`generate_epsilon_pareto` no longer exists) and, had it run, would have tested nothing. A file named `test_*` that tests nothing is worse
than no file.

The project's test suite is **`scripts/52_invariants.py`** (87 invariants at TASK 53, Stages A, B, C; `--list`,
`--stage {A,B,C}`, `--only ID`, `--detail DIR`), reduced instances solved in seconds, with independent
recomputation. The regression demonstration for a change is **`scripts/53_epoch_demo.py`**: both anchors
with their LP hashes, the commands the next user will issue, and the suite tally.
