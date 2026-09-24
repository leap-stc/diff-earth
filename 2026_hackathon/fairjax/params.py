"""Bridge from a configured ``fair.FAIR`` object to JAX pytrees.

FaIR's front end -- species properties, RCMIP/SSP scenario loading, the AR6
calibration files, unit handling -- is good, and none of it needs to be
differentiable. So we keep it: configure a ``FAIR`` object exactly as you
normally would, then call :func:`from_fair` to lift it into arrays that
:mod:`fairjax.core` can run and differentiate.

The important job here is *sanitisation*. FaIR marks inapplicable species
properties with NaN. Those NaNs are harmless in a forward pass under
``np.nansum`` but they destroy gradients, so every array is filled with a
neutral finite value on the way out and the species that do not apply are
selected with masks instead.
"""

import numpy as np

from .core import InitialState, Params, SpeciesMap

__all__ = ["from_fair"]

# fill values chosen so that a masked-out species contributes exactly zero and
# never produces a division by zero or a log of zero
_FILL = {
    "g0": 1.0,
    "g1": 1.0,
    "unperturbed_lifetime": 1.0,
    "forcing_reference_concentration": 1.0,
    "fractional_release": 1.0,
    "forcing_scale": 0.0,
    "forcing_efficacy": 0.0,
}


def _clean(array, name):
    out = np.nan_to_num(np.asarray(array, dtype=np.float64), nan=_FILL.get(name, 0.0))
    return out


def from_fair(f):
    """Lift a configured ``FAIR`` instance into JAX-ready arrays.

    Returns ``(params, smap, emissions, forcing_prescribed, init)`` where
    ``params`` and ``init`` carry a leading config axis and ``emissions`` and
    ``forcing_prescribed`` carry a leading scenario axis.

    The ``FAIR`` object must be fully configured (species defined, emissions and
    configs filled) but does not need to have been run.
    """
    f._check_properties()
    f._make_indices()

    props = f.properties_df

    # FaIR sums the two CO2 emissions species into the single "CO2" species as
    # the first act of run(); do the same so results line up.
    if (
        f._co2_indices.sum() + f._co2_ffi_indices.sum() + f._co2_afolu_indices.sum()
        == 3
    ):
        f.emissions[..., f._co2_indices] = (
            f.emissions[..., f._co2_ffi_indices].data
            + f.emissions[..., f._co2_afolu_indices].data
        )

    # a few species configs (molecular weight, concentration_per_emission,
    # cl_atoms, br_atoms) carry no config axis in FaIR. Broadcast them so every
    # parameter has the same leading axis and vmap can treat them alike.
    n_config = f.species_configs["g0"].shape[0]

    def species_config(name):
        array = f.species_configs[name]
        out = _clean(array.data, name)
        if "config" not in array.dims:
            out = np.broadcast_to(out, (n_config,) + out.shape).copy()
        return out

    def climate_config(name):
        return _clean(f.climate_configs[name].data, name)

    params = Params(
        baseline_concentration=species_config("baseline_concentration"),
        baseline_emissions=species_config("baseline_emissions"),
        concentration_per_emission=species_config("concentration_per_emission"),
        g0=species_config("g0"),
        g1=species_config("g1"),
        iirf_0=species_config("iirf_0"),
        iirf_airborne=species_config("iirf_airborne"),
        iirf_temperature=species_config("iirf_temperature"),
        iirf_uptake=species_config("iirf_uptake"),
        partition_fraction=species_config("partition_fraction"),
        unperturbed_lifetime=species_config("unperturbed_lifetime"),
        ch4_lifetime_chemical_sensitivity=species_config(
            "ch4_lifetime_chemical_sensitivity"
        ),
        lifetime_temperature_sensitivity=species_config(
            "lifetime_temperature_sensitivity"
        ),
        # FaIR folds the tropospheric adjustment into the forcing scale before
        # the loop; we do the same so the parameter means the same thing here.
        forcing_scale=_clean(
            f.species_configs["forcing_scale"].data
            * (1 + f.species_configs["tropospheric_adjustment"].data),
            "forcing_scale",
        ),
        forcing_efficacy=species_config("forcing_efficacy"),
        forcing_temperature_feedback=species_config("forcing_temperature_feedback"),
        forcing_reference_concentration=species_config(
            "forcing_reference_concentration"
        ),
        greenhouse_gas_radiative_efficiency=species_config(
            "greenhouse_gas_radiative_efficiency"
        ),
        erfari_radiative_efficiency=species_config("erfari_radiative_efficiency"),
        aci_scale=species_config("aci_scale"),
        aci_shape=species_config("aci_shape"),
        ozone_radiative_efficiency=species_config("ozone_radiative_efficiency"),
        contrails_radiative_efficiency=species_config(
            "contrails_radiative_efficiency"
        ),
        lapsi_radiative_efficiency=species_config("lapsi_radiative_efficiency"),
        h2o_stratospheric_factor=species_config("h2o_stratospheric_factor"),
        land_use_cumulative_emissions_to_forcing=species_config(
            "land_use_cumulative_emissions_to_forcing"
        ),
        cl_atoms=species_config("cl_atoms"),
        br_atoms=species_config("br_atoms"),
        fractional_release=species_config("fractional_release"),
        ocean_heat_capacity=climate_config("ocean_heat_capacity"),
        ocean_heat_transfer=climate_config("ocean_heat_transfer"),
        deep_ocean_efficacy=climate_config("deep_ocean_efficacy"),
        gamma_autocorrelation=climate_config("gamma_autocorrelation"),
        forcing_4co2=climate_config("forcing_4co2"),
    )

    def single(mask, label):
        """Index of a species that must appear at most once."""
        where = np.asarray(mask).nonzero()[0]
        if len(where) > 1:
            raise ValueError(f"expected at most one {label} species, found {len(where)}")
        return (int(where[0]), 1.0) if len(where) else (0, 0.0)

    def required(mask, label):
        where = np.asarray(mask).nonzero()[0]
        if len(where) != 1:
            raise ValueError(f"fairjax requires exactly one {label} species")
        return int(where[0])

    i_ari, ari_on = single(f._ari_indices, "ari")
    i_aci, aci_on = single(f._aci_indices, "aci")
    i_ozone, ozone_on = single(f._ozone_indices, "ozone")
    i_contrails, contrails_on = single(f._contrails_indices, "contrails")
    i_lapsi, lapsi_on = single(f._lapsi_indices, "lapsi")
    i_h2ostrat, h2ostrat_on = single(f._h2ostrat_indices, "h2o stratospheric")
    i_landuse, landuse_on = single(f._landuse_indices, "land use")
    i_eesc, eesc_on = single(f._eesc_indices, "eesc")
    i_cfc11, _ = single(f._cfc11_indices, "cfc-11")

    if f._ghg_inverse_indices.any():
        raise NotImplementedError(
            "fairjax implements the forward (emissions-driven) gas cycle only; "
            f"{int(f._ghg_inverse_indices.sum())} species are concentration-driven"
        )

    smap = SpeciesMap(
        ghg=f._ghg_indices.astype(np.float64),
        ghg_forward=f._ghg_forward_indices.astype(np.float64),
        minor_ghg=f._minor_ghg_indices.astype(np.float64),
        halogen=f._halogen_indices.astype(np.float64),
        aerosol_from_emissions=(
            f._aerosol_chemistry_from_emissions_indices.astype(np.float64)
        ),
        aerosol_from_concentration=(
            f._aerosol_chemistry_from_concentration_indices.astype(np.float64)
        ),
        prescribed_forcing=np.asarray(
            props["input_mode"] == "forcing", dtype=np.float64
        ),
        i_co2=required(f._co2_indices, "co2"),
        i_ch4=required(f._ch4_indices, "ch4"),
        i_n2o=required(f._n2o_indices, "n2o"),
        i_cfc11=i_cfc11,
        i_ari=i_ari,
        i_aci=i_aci,
        i_ozone=i_ozone,
        i_contrails=i_contrails,
        i_lapsi=i_lapsi,
        i_h2ostrat=i_h2ostrat,
        i_landuse=i_landuse,
        i_eesc=i_eesc,
        ari_on=ari_on,
        aci_on=aci_on,
        ozone_on=ozone_on,
        contrails_on=contrails_on,
        lapsi_on=lapsi_on,
        h2ostrat_on=h2ostrat_on,
        landuse_on=landuse_on,
        eesc_on=eesc_on,
    )

    # emissions and prescribed forcing must not vary by config: the config axis
    # is what we vmap parameters over
    emissions = _clean(
        _require_config_invariant(f.emissions.data, "emissions"), "emissions"
    )
    forcing_prescribed = _clean(
        _require_config_invariant(f.forcing.data, "forcing"), "forcing"
    )

    init = InitialState(
        concentration=_clean(f.concentration.data[0, 0], "concentration"),
        forcing=_clean(f.forcing.data[0, 0], "forcing"),
        temperature=_clean(f.temperature.data[0, 0], "temperature"),
        gas_partitions=_clean(f.gas_partitions.data[0], "gas_partitions"),
        airborne_emissions=_clean(
            f.airborne_emissions.data[0, 0], "airborne_emissions"
        ),
        cumulative_emissions=_clean(
            f.cumulative_emissions.data[0, 0], "cumulative_emissions"
        ),
        ocean_heat_content_change=_clean(
            f.ocean_heat_content_change.data[0, 0], "ocean_heat_content_change"
        ),
    )
    return params, smap, emissions, forcing_prescribed, init


def _require_config_invariant(array, name):
    """Move (time, scenario, config, specie) to (scenario, time, specie).

    Raises if the array actually differs between configs, since ``fairjax``
    vmaps parameters over the config axis and drives every config with the same
    forcing.
    """
    data = np.asarray(array, dtype=np.float64)
    reference = data[:, :, 0:1, :]
    if not np.allclose(np.nan_to_num(data), np.nan_to_num(reference), equal_nan=True):
        raise ValueError(
            f"{name} varies between configs, which fairjax does not support; "
            "run one config group at a time"
        )
    return np.moveaxis(data[:, :, 0, :], 0, 1)
