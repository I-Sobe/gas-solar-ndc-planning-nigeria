"""
Battery Storage Module (Planning-Level Energy Abstraction)

Scope
-----
Stateful battery storage model for long-term planning studies with
annual time-step resolution. Storage is represented as an energy buffer
with simplified charge/discharge behavior.

Modeling assumptions
--------------------
- Annual time-step resolution
- Power limits are annualized (MW × 8760 → MWh/year)
- Round-trip efficiency is applied symmetrically:
    √η on charge, √η on discharge
- Minimum state of charge (SOC) is fixed at 0 MWh
- No degradation, ramping, or intra-annual dynamics

Boundary discipline
-------------------
- This class is STATEFUL: SOC is mutated during charge/discharge.
- Callers MUST reset or reinstantiate storage between scenarios.
- Not safe for reuse across parallel experiments.

Non-scope
---------
- Hourly or sub-hourly dispatch
- Battery chemistry or degradation modeling
- Unit commitment or reserve provision
"""

from math import sqrt
from src.utils import assert_non_negative


class BatteryStorage:
    """
    Battery energy storage system (stateful, energy-level abstraction).
    """

    def __init__(
        self,
        energy_capacity_mwh,
        power_capacity_mw,
        round_trip_efficiency=0.9,
        initial_soc=0.0,
    ):
        """
        Parameters
        ----------
        energy_capacity_mwh : float
            Maximum stored energy (MWh)
        power_capacity_mw : float
            Maximum charge/discharge power (MW)
        round_trip_efficiency : float
            Round-trip efficiency (0–1)
        initial_soc : float
            Initial state of charge (MWh)
        """

        assert_non_negative(
            [energy_capacity_mwh, power_capacity_mw, initial_soc],
            "storage parameters",
        )

        if not (0 < round_trip_efficiency <= 1):
            raise ValueError("round_trip_efficiency must be in (0, 1]")

        if initial_soc > energy_capacity_mwh:
            raise ValueError("initial_soc cannot exceed energy_capacity_mwh")

        self.energy_capacity = energy_capacity_mwh
        self.power_capacity = power_capacity_mw

        self.eta_rt = round_trip_efficiency
        self.eta_c = sqrt(round_trip_efficiency)
        self.eta_d = sqrt(round_trip_efficiency)

        self.initial_soc = initial_soc
        self.soc = initial_soc

    # --------------------------------------------------------
    # State management
    # --------------------------------------------------------

    def reset(self):
        """
        Reset state of charge (SOC) to the initial value.

        Must be called between independent scenario runs
        if the same storage instance is reused.
        """
        self.soc = self.initial_soc

    # --------------------------------------------------------
    # Charge / discharge interface
    # --------------------------------------------------------

    def charge(self, energy_mwh):
        """
        Charge the battery (annual time-step).

        Parameters
        ----------
        energy_mwh : float
            Energy available for charging over the year (MWh)

        Returns
        -------
        float
            Energy actually stored in SOC (MWh),
            after efficiency and capacity limits.
        """

        assert_non_negative([energy_mwh], "charge energy")

        # Annualized power constraint (MW × 8760 → MWh/year)
        max_charge = self.power_capacity * 8760
        charge_energy = min(energy_mwh, max_charge)

        # Apply charging efficiency
        effective_energy = charge_energy * self.eta_c

        available_space = self.energy_capacity - self.soc
        stored = min(effective_energy, available_space)

        self.soc += stored

        # Invariant check
        if not (0.0 <= self.soc <= self.energy_capacity):
            raise RuntimeError("SOC invariant violated after charging")

        return stored

    def discharge(self, energy_mwh):
        """
        Discharge the battery (annual time-step).

        Parameters
        ----------
        energy_mwh : float
            Energy requested from storage (MWh)

        Returns
        -------
        float
            Energy actually delivered to the system (MWh),
            after efficiency and SOC constraints.
        """

        assert_non_negative([energy_mwh], "discharge energy")

        # Annualized power constraint (MW × 8760 → MWh/year)
        max_discharge = self.power_capacity * 8760
        requested = min(energy_mwh, max_discharge)

        # Account for discharge efficiency
        required_soc = requested / self.eta_d
        withdrawn = min(required_soc, self.soc)

        self.soc -= withdrawn
        delivered = withdrawn * self.eta_d

        # Invariant check
        if not (0.0 <= self.soc <= self.energy_capacity):
            raise RuntimeError("SOC invariant violated after discharging")

        return delivered
