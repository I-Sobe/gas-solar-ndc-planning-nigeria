"""
Battery Storage Module

Defines state-of-charge (SOC) update equations, charge/discharge
constraints, efficiencies, and storage operational limits.

Functions:
    update_soc(soc, charge, discharge, eta_c, eta_d)
    enforce_storage_limits(soc, soc_min, soc_max)
"""
"""
storage.py
Battery energy storage model (energy-level abstraction)
"""

import numpy as np
from src.utils import validate_non_negative


class BatteryStorage:
    """
    Battery energy storage system model.
    """

    def __init__(
        self,
        energy_capacity_mwh,
        power_capacity_mw,
        round_trip_efficiency=0.9,
        initial_soc=0.0
    ):
        """
        Parameters
        ----------
        energy_capacity_mwh : float
            Maximum stored energy (MWh)
        power_capacity_mw : float
            Maximum charge/discharge power (MW)
        round_trip_efficiency : float
            Fractional efficiency (0–1)
        initial_soc : float
            Initial state of charge (MWh)
        """

        validate_non_negative(
            [energy_capacity_mwh, power_capacity_mw, initial_soc],
            "storage parameters"
        )

        if not (0 < round_trip_efficiency <= 1):
            raise ValueError("round_trip_efficiency must be in (0,1]")

        self.energy_capacity = energy_capacity_mwh
        self.power_capacity = power_capacity_mw
        self.eta = round_trip_efficiency
        self.soc = initial_soc

    def reset(self):
        """Reset state of charge to zero."""
        self.soc = 0.0

    def charge(self, energy_mwh):
        """
        Charge the battery.

        Parameters
        ----------
        energy_mwh : float
            Energy available for charging (MWh)

        Returns
        -------
        float
            Energy actually stored (MWh)
        """

        validate_non_negative([energy_mwh], "charge energy")

        max_charge = self.power_capacity * 8760
        charge_energy = min(energy_mwh, max_charge)

        effective_energy = charge_energy * self.eta

        available_space = self.energy_capacity - self.soc
        stored = min(effective_energy, available_space)

        self.soc += stored
        return stored

    def discharge(self, energy_mwh):
        """
        Discharge the battery.

        Parameters
        ----------
        energy_mwh : float
            Energy required from storage (MWh)

        Returns
        -------
        float
            Energy actually delivered (MWh)
        """

        validate_non_negative([energy_mwh], "discharge energy")

        max_discharge = self.power_capacity * 8760
        discharge_energy = min(energy_mwh, max_discharge)

        delivered = min(discharge_energy, self.soc)

        self.soc -= delivered
        return delivered
