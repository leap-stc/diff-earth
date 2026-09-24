"""Gas cycle and radiative forcing formulae, ported from FaIR v2.2.4 to JAX.

Every function here takes and returns arrays over the *species* axis only --
one scenario, one config. Scenario and config ensembles are handled by ``vmap``
in :mod:`fairjax.core`, which is both faster to read and easier to get right
than FaIR's five-dimensional broadcasting.

Porting notes
-------------
FaIR uses NaN as a sentinel meaning "this property does not apply to this
species", then ``np.nansum``s over the species axis. That is fine for a forward
model and fatal for a gradient: ``0 * nan == nan`` in the backward pass, so a
single NaN anywhere poisons the whole Jacobian. Here, every array is sanitised
to a finite value at extraction time (see :mod:`fairjax.params`) and the
species that do not apply are removed by *multiplicative masks* instead.
"""

import jax.numpy as jnp

__all__ = [
    "calculate_alpha",
    "calculate_alpha_ch4",
    "step_concentration",
    "meinshausen2020",
    "erfari",
    "erfaci",
    "eesc",
    "ozone",
    "linear_forcing",
]


def calculate_alpha(
    airborne_emissions,
    cumulative_emissions,
    temperature,
    g0,
    g1,
    iirf_0,
    iirf_airborne,
    iirf_temperature,
    iirf_uptake,
    iirf_max,
):
    """Lifetime scaling factor from the state-dependent carbon/gas cycle feedback.

    This is the one genuine nonlinearity in the FaIR gas cycle: the airborne
    fraction depends on how much has already been taken up and on temperature.
    FaIR v2.x computes it in closed form, which is why no implicit solve or
    root-find appears anywhere in this port.
    """
    iirf = (
        iirf_0
        + iirf_uptake * (cumulative_emissions - airborne_emissions)
        + iirf_temperature * temperature
        + iirf_airborne * airborne_emissions
    )
    # FaIR clamps with arithmetic rather than a branch. Keeping the same form
    # gives the correct subgradient for free (d/diirf = 1 below the cap, 0 above).
    iirf = (iirf > iirf_max) * iirf_max + iirf * (iirf < iirf_max)

    # Very long-lived gases (in the default species set, only CF4) overflow the
    # exponential. FaIR lets it happen and repairs the NaN afterwards; we cannot,
    # because NaN in the forward pass is NaN in the backward pass whatever the
    # `where` says. So we sanitise the *input* to the unsafe branch as well as
    # its output -- the "double where" -- which keeps the gradient finite.
    ratio = iirf / g1
    overflows = ratio > 709.0  # jnp.exp overflows float64 above ~709.78
    return jnp.where(overflows, 1.0, g0 * jnp.exp(jnp.where(overflows, 0.0, ratio)))


def calculate_alpha_ch4(
    emissions,
    concentration,
    temperature,
    baseline_emissions,
    baseline_concentration,
    chemical_sensitivity,
    temperature_sensitivity,
    emissions_mask,
    concentration_mask,
):
    """Thornhill (2021) multi-species methane lifetime scaling."""
    from_emissions = jnp.sum(
        emissions_mask
        * jnp.log1p(
            emissions_mask
            * (emissions - baseline_emissions)
            * chemical_sensitivity
        )
    )
    from_concentration = jnp.sum(
        concentration_mask
        * jnp.log1p(
            concentration_mask
            * (concentration - baseline_concentration)
            * chemical_sensitivity
        )
    )
    log_scaling = (
        from_emissions
        + from_concentration
        + jnp.log1p(temperature * temperature_sensitivity)
    )
    return jnp.exp(log_scaling)


def step_concentration(
    emissions,
    gasboxes,
    alpha_lifetime,
    baseline_emissions,
    baseline_concentration,
    concentration_per_emission,
    lifetime,
    partition_fraction,
    timestep,
):
    """Advance the multi-box gas cycle one timestep (emissions -> concentration).

    ``gasboxes``, ``lifetime`` and ``partition_fraction`` carry a trailing
    gasbox axis; everything else is per-species.
    """
    decay_rate = timestep / (alpha_lifetime[..., None] * lifetime)
    decay_factor = jnp.exp(-decay_rate)

    gasboxes_new = (
        partition_fraction
        * (emissions - baseline_emissions)[..., None]
        / decay_rate
        * (1 - decay_factor)
        * timestep
        + gasboxes * decay_factor
    )
    airborne_new = jnp.sum(gasboxes_new, axis=-1)
    concentration_new = (
        baseline_concentration + concentration_per_emission * airborne_new
    )
    return concentration_new, gasboxes_new, airborne_new


def meinshausen2020(
    concentration,
    reference_concentration,
    forcing_scale,
    radiative_efficiency,
    i_co2,
    i_ch4,
    i_n2o,
    minor_ghg_mask,
    a1=-2.4785e-07,
    b1=0.00075906,
    c1=-0.0021492,
    d1=5.2488,
    a2=-0.00034197,
    b2=0.00025455,
    c2=-0.00024357,
    d2=0.12173,
    a3=-8.9603e-05,
    b3=-0.00012462,
    d3=0.045194,
):
    """Meinshausen et al. (2020) greenhouse gas concentration-to-forcing.

    The CO2 band is piecewise in three concentration regimes. FaIR selects them
    with ``.nonzero()`` fancy-index assignment, which JAX cannot trace; here it
    becomes nested ``jnp.where``. The regimes are disjoint, so the nesting order
    does not change the result. Note the branches meet continuously but not
    smoothly -- gradients are kinked at ``co2_base`` and at ``ca_max``.
    """
    co2 = concentration[i_co2]
    ch4 = concentration[i_ch4]
    n2o = concentration[i_n2o]
    co2_base = reference_concentration[i_co2]
    ch4_base = reference_concentration[i_ch4]
    n2o_base = reference_concentration[i_n2o]

    ca_max = co2_base - b1 / (2 * a1)
    alpha_p = jnp.where(
        co2 <= co2_base,
        d1,
        jnp.where(
            co2 <= ca_max,
            d1 + a1 * (co2 - co2_base) ** 2 + b1 * (co2 - co2_base),
            d1 - b1**2 / (4 * a1),
        ),
    )
    alpha_n2o = c1 * jnp.sqrt(n2o)

    erf = jnp.zeros_like(concentration)
    erf = erf.at[i_co2].set(
        (alpha_p + alpha_n2o) * jnp.log(co2 / co2_base) * forcing_scale[i_co2]
    )
    erf = erf.at[i_ch4].set(
        (a3 * jnp.sqrt(ch4) + b3 * jnp.sqrt(n2o) + d3)
        * (jnp.sqrt(ch4) - jnp.sqrt(ch4_base))
        * forcing_scale[i_ch4]
    )
    erf = erf.at[i_n2o].set(
        (a2 * jnp.sqrt(co2) + b2 * jnp.sqrt(n2o) + c2 * jnp.sqrt(ch4) + d2)
        * (jnp.sqrt(n2o) - jnp.sqrt(n2o_base))
        * forcing_scale[i_n2o]
    )
    # minor GHGs are linear in concentration; the 0.001 converts ppt to ppb
    erf = erf + minor_ghg_mask * (
        (concentration - reference_concentration)
        * radiative_efficiency
        * 0.001
        * forcing_scale
    )
    return erf


def erfari(
    emissions,
    concentration,
    baseline_emissions,
    baseline_concentration,
    forcing_scale,
    radiative_efficiency,
    emissions_mask,
    concentration_mask,
):
    """Aerosol-radiation interactions: linear in each precursor.

    FaIR writes the emissions term into a per-species array and then writes the
    concentration term over the top of it, so a species that appears in both
    index sets (methane, for one) is counted *once*, using its concentration.
    Summing both terms here would double count it.
    """
    emissions_only = emissions_mask * (1 - concentration_mask)
    from_emissions = jnp.sum(
        emissions_only * (emissions - baseline_emissions) * radiative_efficiency
    )
    from_concentration = jnp.sum(
        concentration_mask
        * (concentration - baseline_concentration)
        * radiative_efficiency
    )
    return (from_emissions + from_concentration) * forcing_scale


def erfaci(
    emissions,
    concentration,
    baseline_emissions,
    baseline_concentration,
    forcing_scale,
    scale,
    sensitivity,
    emissions_mask,
    concentration_mask,
):
    """Aerosol-cloud interactions: ``F = beta * log(1 + sum_i s_i A_i)``."""

    def radiative_effect(emis, conc):
        total = jnp.sum(emissions_mask * sensitivity * emis) + jnp.sum(
            concentration_mask * sensitivity * conc
        )
        return scale * jnp.log(1 + total)

    delta = radiative_effect(emissions, concentration) - radiative_effect(
        baseline_emissions, baseline_concentration
    )
    return delta * forcing_scale


def eesc(
    concentration,
    fractional_release,
    cl_atoms,
    br_atoms,
    i_cfc11,
    halogen_mask,
    br_cl_ratio,
):
    """Equivalent effective stratospheric chlorine, in CFC-11 equivalents."""
    # the division and re-multiplication by cfc11_fr cancel algebraically; they
    # are kept so this reproduces FaIR's floating-point result exactly
    cfc11_fr = fractional_release[i_cfc11]
    per_species = (
        cl_atoms * concentration * fractional_release / cfc11_fr
        + br_cl_ratio * br_atoms * concentration * fractional_release / cfc11_fr
    )
    return jnp.sum(halogen_mask * per_species) * cfc11_fr


def ozone(
    emissions,
    concentration,
    baseline_emissions,
    baseline_concentration,
    forcing_scale,
    radiative_efficiency,
    emissions_mask,
    concentration_mask,
):
    """Thornhill (2021) / Skeie (2020) ozone forcing from precursors."""
    from_concentration = jnp.sum(
        concentration_mask
        * (concentration - baseline_concentration)
        * radiative_efficiency
    )
    from_emissions = jnp.sum(
        emissions_mask * (emissions - baseline_emissions) * radiative_efficiency
    )
    return (from_concentration + from_emissions) * forcing_scale


def linear_forcing(driver, baseline_driver, forcing_scale, radiative_efficiency):
    """Generic linear driver-to-forcing relationship (contrails, LAPSI, ...).

    No species mask is needed: ``radiative_efficiency`` is already zero for every
    species the term does not apply to, which is exactly how FaIR uses it (there
    the zeros are NaNs and the sum is a ``nansum``). Masking further would be
    wrong -- land use, for instance, is driven by cumulative *CO2 AFOLU*
    emissions, which is not a greenhouse gas species.
    """
    return jnp.sum((driver - baseline_driver) * radiative_efficiency) * forcing_scale
