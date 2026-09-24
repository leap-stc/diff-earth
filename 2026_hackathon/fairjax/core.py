"""The FaIR time loop as a differentiable ``lax.scan``.

The model is written for a *single* scenario and a *single* config. Ensembles
are produced with :func:`run_ensemble`, which ``vmap``s over configs and then
over scenarios. This is the main structural departure from FaIR, which carries
``(time, scenario, config, specie, gasbox)`` through every expression.

Scope relative to FaIR v2.2.4
-----------------------------
Implemented: the forward (emissions-driven) gas cycle with the state-dependent
``alpha`` feedback, Thornhill methane lifetime, Meinshausen 2020 greenhouse gas
forcing, ERFari, ERFaci, EESC, ozone, contrails, LAPSI, stratospheric water
vapour, land use, the temperature-forcing feedback, the Cummins energy balance
model, TOA imbalance and ocean heat content.

Not implemented: concentration-driven (inverse) mode, stochastic internal
variability, and the ``leach2021`` / ``etminan2016`` / ``myhre1998`` forcing
schemes.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import ebm, physics
from .earth import EARTH_RADIUS, SECONDS_PER_YEAR

__all__ = ["Params", "SpeciesMap", "InitialState", "Output", "run", "run_ensemble"]


class Params(NamedTuple):
    """Differentiable model parameters for one config.

    Species-axis arrays have shape ``(n_species,)``; ``partition_fraction`` and
    ``unperturbed_lifetime`` have shape ``(n_species, n_gasbox)``; ocean arrays
    have shape ``(n_layer,)``.
    """

    # gas cycle
    baseline_concentration: jnp.ndarray
    baseline_emissions: jnp.ndarray
    concentration_per_emission: jnp.ndarray
    g0: jnp.ndarray
    g1: jnp.ndarray
    iirf_0: jnp.ndarray
    iirf_airborne: jnp.ndarray
    iirf_temperature: jnp.ndarray
    iirf_uptake: jnp.ndarray
    partition_fraction: jnp.ndarray
    unperturbed_lifetime: jnp.ndarray
    ch4_lifetime_chemical_sensitivity: jnp.ndarray
    lifetime_temperature_sensitivity: jnp.ndarray
    # forcing
    forcing_scale: jnp.ndarray  # includes the tropospheric adjustment
    forcing_efficacy: jnp.ndarray
    forcing_temperature_feedback: jnp.ndarray
    forcing_reference_concentration: jnp.ndarray
    greenhouse_gas_radiative_efficiency: jnp.ndarray
    erfari_radiative_efficiency: jnp.ndarray
    aci_scale: jnp.ndarray  # scalar
    aci_shape: jnp.ndarray
    ozone_radiative_efficiency: jnp.ndarray
    contrails_radiative_efficiency: jnp.ndarray
    lapsi_radiative_efficiency: jnp.ndarray
    h2o_stratospheric_factor: jnp.ndarray
    land_use_cumulative_emissions_to_forcing: jnp.ndarray
    cl_atoms: jnp.ndarray
    br_atoms: jnp.ndarray
    fractional_release: jnp.ndarray
    # climate
    ocean_heat_capacity: jnp.ndarray
    ocean_heat_transfer: jnp.ndarray
    deep_ocean_efficacy: jnp.ndarray  # scalar
    gamma_autocorrelation: jnp.ndarray  # scalar
    forcing_4co2: jnp.ndarray  # scalar; only used by the ECS/TCR diagnostics


class SpeciesMap(NamedTuple):
    """Which species does what. Fixed by the configuration, never differentiated.

    Multi-species sets are float 0/1 masks so they can multiply cleanly; the
    singleton species are integer indices paired with a ``_on`` flag that is 0.0
    when the species is absent from the configuration.
    """

    ghg: jnp.ndarray
    ghg_forward: jnp.ndarray
    minor_ghg: jnp.ndarray
    halogen: jnp.ndarray
    aerosol_from_emissions: jnp.ndarray
    aerosol_from_concentration: jnp.ndarray
    prescribed_forcing: jnp.ndarray
    i_co2: int
    i_ch4: int
    i_n2o: int
    i_cfc11: int
    i_ari: int
    i_aci: int
    i_ozone: int
    i_contrails: int
    i_lapsi: int
    i_h2ostrat: int
    i_landuse: int
    i_eesc: int
    ari_on: float
    aci_on: float
    ozone_on: float
    contrails_on: float
    lapsi_on: float
    h2ostrat_on: float
    landuse_on: float
    eesc_on: float


class InitialState(NamedTuple):
    """Model state on the first timebound."""

    concentration: jnp.ndarray
    forcing: jnp.ndarray
    temperature: jnp.ndarray  # (n_layer,)
    gas_partitions: jnp.ndarray  # (n_species, n_gasbox)
    airborne_emissions: jnp.ndarray
    cumulative_emissions: jnp.ndarray
    ocean_heat_content_change: jnp.ndarray  # scalar


class Output(NamedTuple):
    """Model output. Time axis is timebounds (``n_timepoints + 1``)."""

    concentration: jnp.ndarray
    forcing: jnp.ndarray
    forcing_sum: jnp.ndarray
    temperature: jnp.ndarray
    airborne_emissions: jnp.ndarray
    cumulative_emissions: jnp.ndarray
    alpha_lifetime: jnp.ndarray
    toa_imbalance: jnp.ndarray
    ocean_heat_content_change: jnp.ndarray


@partial(jax.jit, static_argnames=("timestep", "iirf_max", "ch4_method",
                                   "br_cl_ods_potential"))
def run(
    params,
    smap,
    emissions,
    forcing_prescribed,
    init,
    timestep=1.0,
    iirf_max=100.0,
    ch4_method="thornhill2021",
    br_cl_ods_potential=45.0,
):
    """Run FaIR for one scenario and one config.

    Parameters
    ----------
    params : Params
    smap : SpeciesMap
    emissions : (n_timepoints, n_species) array
    forcing_prescribed : (n_timebounds, n_species) array
        Forcing for species driven directly by forcing (solar, volcanic). Slots
        belonging to species whose forcing the model computes are ignored.
    init : InitialState
    """
    n_timepoints, n_species = emissions.shape

    # --- precompute things that do not depend on the timestep ----------------
    matrix = ebm.eb_matrix(
        params.ocean_heat_capacity,
        params.ocean_heat_transfer,
        params.deep_ocean_efficacy,
        params.gamma_autocorrelation,
        timestep,
    )
    matrix_d, forcing_vector_d = ebm.discretise(matrix, params.gamma_autocorrelation)

    # cumulative emissions are a pure prefix sum, so they come out of the loop
    cumulative_emissions = jnp.concatenate(
        [
            init.cumulative_emissions[None, :],
            init.cumulative_emissions[None, :]
            + jnp.cumsum(emissions, axis=0) * timestep,
        ]
    )

    # nonlinear GHG forcing is measured relative to the baseline state
    ghg_forcing_offset = physics.meinshausen2020(
        params.baseline_concentration,
        params.forcing_reference_concentration,
        params.forcing_scale,
        params.greenhouse_gas_radiative_efficiency,
        smap.i_co2,
        smap.i_ch4,
        smap.i_n2o,
        smap.minor_ghg,
    )

    computed_forcing = (
        smap.ghg
        + _onehot(n_species, smap.i_ari, smap.ari_on)
        + _onehot(n_species, smap.i_aci, smap.aci_on)
        + _onehot(n_species, smap.i_ozone, smap.ozone_on)
        + _onehot(n_species, smap.i_contrails, smap.contrails_on)
        + _onehot(n_species, smap.i_lapsi, smap.lapsi_on)
        + _onehot(n_species, smap.i_h2ostrat, smap.h2ostrat_on)
        + _onehot(n_species, smap.i_landuse, smap.landuse_on)
    )
    forcing_defined = jnp.clip(computed_forcing + smap.prescribed_forcing, 0.0, 1.0)

    # FaIR computes ERFari and ERFaci *before* it updates EESC within the same
    # timestep, so the EESC concentration slot is still unwritten (NaN) and drops
    # silently out of their nansums. Ozone, computed after the EESC update, does
    # see it. Reproduce that by giving the two aerosol terms a mask with EESC
    # removed -- this is an ordering artefact of the reference implementation,
    # not a modelling choice, but the port has to match it.
    aerosol_from_concentration_pre_eesc = smap.aerosol_from_concentration * (
        1.0 - _onehot(n_species, smap.i_eesc, smap.eesc_on)
    )

    # initial EBM state: element 0 is the Cummins stochastic forcing, which FaIR
    # seeds with the initial forcing sum
    state_0 = jnp.concatenate(
        [jnp.sum(init.forcing)[None], init.temperature]
    )

    def step(carry, xs):
        gas_partitions, airborne, concentration, state = carry
        emis, cumulative_now, cumulative_next, prescribed = xs
        temperature = state[1]

        # 1. lifetime scaling from the state-dependent gas cycle feedback
        alpha = physics.calculate_alpha(
            airborne,
            cumulative_now,
            temperature,
            params.g0,
            params.g1,
            params.iirf_0,
            params.iirf_airborne,
            params.iirf_temperature,
            params.iirf_uptake,
            iirf_max,
        )

        # 2. methane lifetime responds to other species, not just to itself
        if ch4_method == "thornhill2021":
            alpha_ch4 = physics.calculate_alpha_ch4(
                emis,
                concentration,
                temperature,
                params.baseline_emissions,
                params.baseline_concentration,
                params.ch4_lifetime_chemical_sensitivity,
                params.lifetime_temperature_sensitivity,
                smap.aerosol_from_emissions,
                smap.aerosol_from_concentration,
            )
            alpha = alpha.at[smap.i_ch4].set(alpha_ch4)

        # 3. emissions to concentrations
        concentration_new, gas_partitions_new, airborne_new = physics.step_concentration(
            emis,
            gas_partitions,
            alpha,
            params.baseline_emissions,
            params.baseline_concentration,
            params.concentration_per_emission,
            params.unperturbed_lifetime,
            params.partition_fraction,
            timestep,
        )
        forward = smap.ghg_forward
        concentration = jnp.where(forward, concentration_new, concentration)
        gas_partitions = jnp.where(
            forward[:, None], gas_partitions_new, gas_partitions
        )
        airborne = jnp.where(forward, airborne_new, airborne)

        # 5. greenhouse gas concentrations to forcing
        forcing = smap.ghg * (
            physics.meinshausen2020(
                concentration,
                params.forcing_reference_concentration,
                params.forcing_scale,
                params.greenhouse_gas_radiative_efficiency,
                smap.i_co2,
                smap.i_ch4,
                smap.i_n2o,
                smap.minor_ghg,
            )
            - ghg_forcing_offset
        )

        # 6. aerosol-radiation interactions
        forcing = forcing.at[smap.i_ari].add(
            smap.ari_on
            * physics.erfari(
                emis,
                concentration,
                params.baseline_emissions,
                params.baseline_concentration,
                params.forcing_scale[smap.i_ari],
                params.erfari_radiative_efficiency,
                smap.aerosol_from_emissions,
                aerosol_from_concentration_pre_eesc,
            )
        )

        # 7. aerosol-cloud interactions
        forcing = forcing.at[smap.i_aci].add(
            smap.aci_on
            * physics.erfaci(
                emis,
                concentration,
                params.baseline_emissions,
                params.baseline_concentration,
                params.forcing_scale[smap.i_aci],
                params.aci_scale,
                params.aci_shape,
                smap.aerosol_from_emissions,
                aerosol_from_concentration_pre_eesc,
            )
        )

        # 8. equivalent effective stratospheric chlorine, used by ozone below and
        #    by the methane lifetime on the next timestep
        concentration = concentration.at[smap.i_eesc].add(
            smap.eesc_on
            * (
                physics.eesc(
                    concentration,
                    params.fractional_release,
                    params.cl_atoms,
                    params.br_atoms,
                    smap.i_cfc11,
                    smap.halogen,
                    br_cl_ods_potential,
                )
                - concentration[smap.i_eesc]
            )
        )

        # 9. ozone
        forcing = forcing.at[smap.i_ozone].add(
            smap.ozone_on
            * physics.ozone(
                emis,
                concentration,
                params.baseline_emissions,
                params.baseline_concentration,
                params.forcing_scale[smap.i_ozone],
                params.ozone_radiative_efficiency,
                smap.aerosol_from_emissions,
                smap.aerosol_from_concentration,
            )
        )

        # 10-13. linear forcing terms
        forcing = forcing.at[smap.i_contrails].add(
            smap.contrails_on
            * physics.linear_forcing(
                emis,
                0.0,
                params.forcing_scale[smap.i_contrails],
                params.contrails_radiative_efficiency,
            )
        )
        forcing = forcing.at[smap.i_lapsi].add(
            smap.lapsi_on
            * physics.linear_forcing(
                emis,
                params.baseline_emissions,
                params.forcing_scale[smap.i_lapsi],
                params.lapsi_radiative_efficiency,
            )
        )
        forcing = forcing.at[smap.i_h2ostrat].add(
            smap.h2ostrat_on
            * physics.linear_forcing(
                concentration,
                params.baseline_concentration,
                params.forcing_scale[smap.i_h2ostrat],
                params.h2o_stratospheric_factor,
            )
        )
        forcing = forcing.at[smap.i_landuse].add(
            smap.landuse_on
            * physics.linear_forcing(
                cumulative_next,
                0.0,
                params.forcing_scale[smap.i_landuse],
                params.land_use_cumulative_emissions_to_forcing,
            )
        )

        # species driven by prescribed forcing (solar, volcanic) pass through
        forcing = jnp.where(computed_forcing > 0, forcing, prescribed)

        # 14. temperature-forcing feedback
        forcing = forcing_defined * (
            forcing + temperature * params.forcing_temperature_feedback
        )

        # 15-16. sum forcings and advance the energy balance model
        forcing_sum = jnp.sum(forcing)
        forcing_efficacy_sum = jnp.sum(forcing * params.forcing_efficacy)
        state = ebm.step_temperature(
            state, matrix_d, forcing_vector_d, forcing_efficacy_sum
        )

        carry = (gas_partitions, airborne, concentration, state)
        return carry, (concentration, forcing, forcing_sum, state, airborne, alpha)

    carry_0 = (
        init.gas_partitions,
        init.airborne_emissions,
        init.concentration,
        state_0,
    )
    xs = (
        emissions,
        cumulative_emissions[:-1],
        cumulative_emissions[1:],
        forcing_prescribed[1:],
    )
    _, traced = jax.lax.scan(step, carry_0, xs)
    concentration, forcing, forcing_sum, state, airborne, alpha = traced

    # stitch the initial timebound back onto the front
    concentration = jnp.concatenate([init.concentration[None], concentration])
    forcing = jnp.concatenate([init.forcing[None], forcing])
    forcing_sum = jnp.concatenate([jnp.sum(init.forcing)[None], forcing_sum])
    state = jnp.concatenate([state_0[None], state])
    airborne = jnp.concatenate([init.airborne_emissions[None], airborne])
    alpha = jnp.concatenate([alpha, jnp.full((1, n_species), jnp.nan)])

    # 17-18. TOA imbalance and ocean heat content, both diagnostic
    toa = ebm.toa_imbalance(
        state, forcing_sum, params.ocean_heat_transfer, params.deep_ocean_efficacy
    )
    ohc = init.ocean_heat_content_change + (
        (jnp.cumsum(toa) - toa[0])
        * timestep
        * EARTH_RADIUS**2
        * 4
        * jnp.pi
        * SECONDS_PER_YEAR
    )

    return Output(
        concentration=concentration,
        forcing=forcing,
        forcing_sum=forcing_sum,
        temperature=state[:, 1:],
        airborne_emissions=airborne,
        cumulative_emissions=cumulative_emissions,
        alpha_lifetime=alpha,
        toa_imbalance=toa,
        ocean_heat_content_change=ohc,
    )


def _onehot(n, index, value):
    return jnp.zeros(n).at[index].add(value)


def run_ensemble(params, smap, emissions, forcing_prescribed, init, **kwargs):
    """Run every scenario against every config.

    ``params`` and ``init`` lead with a config axis; ``emissions`` and
    ``forcing_prescribed`` lead with a scenario axis. Output arrays come back
    shaped ``(scenario, config, time, ...)``.
    """
    per_config = jax.vmap(
        lambda p, i, e, f: run(p, smap, e, f, i, **kwargs),
        in_axes=(0, 0, None, None),
    )
    per_scenario = jax.vmap(
        lambda e, f: per_config(params, init, e, f), in_axes=(0, 0)
    )
    return per_scenario(emissions, forcing_prescribed)
