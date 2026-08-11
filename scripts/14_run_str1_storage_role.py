"""
14_run_str1_storage_role.py  —  STR-1  [SUSPENDED]
==================================================

STATUS: SUSPENDED pending Phase 2 (intra-annual time structure).

STR-1 asks what role battery storage plays in the optimal system. It cannot
be answered in the current model, and this runner is intentionally disabled.

WHY SUSPENDED
-------------
At annual resolution the storage energy constraint is
    discharge[t] <= RTE * charge[t]          (same year, no inter-year carry)
and charge is debited in the electricity balance. Storage's net contribution
to any year is therefore
    discharge - charge <= (RTE - 1) * charge <= -0.10 * charge <= 0.
Storage can only destroy energy at a capital cost, so storage_add = 0 is
strictly optimal in every scenario. Empirically confirmed: all runs return
zero storage capacity and zero discharge.

The real source of storage value — shifting midday solar to evening peak —
does not exist in the model because there is no intra-annual time structure
for it to live in. The earlier solar-surplus coupling constraint
(charge <= surplus_frac * solar_generation) that made storage appear active
has been removed as unsourced and structurally unjustified at annual
resolution.

Additionally, the previous sweep axes were no-ops:
  - storage_solar_surplus_frac : no longer read by build_model
  - STORAGE_OM_PER_MWH_YR      : never entered the objective
so all swept arms solved identical LPs and any resulting plots were artifacts.

RE-ENABLE WHEN
--------------
Phase 2 introduces representative intra-annual slices
({wet, dry} x {day, evening-peak}); storage then shifts day->evening within a
season and acquires a genuine physical role. At that point storage_deployable_
hours becomes derivable from the slice structure rather than an asserted
parameter, and STR-1 is rebuilt against the time-sliced model.
"""

import sys


def main():
    print(
        "STR-1 is SUSPENDED. Storage has no valid role at annual resolution "
        "(discharge - charge <= -0.10*charge <= 0, so storage_add = 0 is "
        "always optimal). Re-enable after Phase 2 adds intra-annual time "
        "slices. No results were produced. See module docstring for detail."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()