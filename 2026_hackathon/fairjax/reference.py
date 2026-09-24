"""A standard FaIR configuration, used to validate the port and to seed notebooks.

This is deliberately the *stock* FaIR setup -- all 64 default species, RCMIP
emissions, the default species configs -- so that agreement with
``fair.FAIR.run()`` is a meaningful check rather than a check on a toy subset.
"""

import numpy as np

__all__ = ["build_fair", "OCEAN_HEAT_CAPACITY", "OCEAN_HEAT_TRANSFER"]

# a three-layer ocean roughly in the middle of the AR6 calibrated range
OCEAN_HEAT_CAPACITY = np.array([2.8, 9.5, 63.5])
OCEAN_HEAT_TRANSFER = np.array([1.1, 1.6, 0.9])
DEEP_OCEAN_EFFICACY = 1.28


def build_fair(
    scenarios=("ssp245",),
    configs=("default",),
    start=1750,
    end=2100,
    timestep=1,
    ch4_method="thornhill2021",
):
    """Build and configure (but do not run) a stock ``fair.FAIR`` object."""
    from fair import FAIR
    from fair.interface import fill, initialise
    from fair.io import read_properties

    f = FAIR(ch4_method=ch4_method)
    f.define_time(start, end, timestep)
    f.define_scenarios(list(scenarios))
    f.define_configs(list(configs))
    species, properties = read_properties()
    f.define_species(species, properties)
    f.allocate()
    f.fill_species_configs()
    f.fill_from_rcmip()

    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    # gas_partitions is live state, not a time series: FaIR overwrites it during
    # run(). Set it explicitly so the initial condition is unambiguous.
    f.gas_partitions.data[:] = 0.0

    fill(f.climate_configs["ocean_heat_capacity"], OCEAN_HEAT_CAPACITY)
    fill(f.climate_configs["ocean_heat_transfer"], OCEAN_HEAT_TRANSFER)
    fill(f.climate_configs["deep_ocean_efficacy"], DEEP_OCEAN_EFFICACY)
    fill(f.climate_configs["gamma_autocorrelation"], 2.0)
    fill(f.climate_configs["stochastic_run"], False)
    return f
