"""fairjax must reproduce stock FaIR. This is the contract the port is built to.

If any of these fail, the port is wrong -- not FaIR. Run with ``pytest -q``.
"""

import numpy as np
import pytest

import fairjax
from fairjax.reference import build_fair

RTOL = 1e-10


def _config(tree, index=0):
    """Select one config out of a pytree whose leaves lead with a config axis."""
    return type(tree)(*[np.asarray(leaf)[index] for leaf in tree])


@pytest.fixture(scope="module")
def run_pair():
    """Run stock FaIR and fairjax on the same configuration.

    Note the ordering: ``from_fair`` is called *before* ``f.run()``, because
    FaIR keeps ``gas_partitions`` as live state and overwrites it during the run.
    """
    f = build_fair(scenarios=("ssp119", "ssp245", "ssp585"))
    params, smap, emissions, forcing, init = fairjax.from_fair(f)
    f.run(progress=False)
    ported = [
        fairjax.run(_config(params), smap, emissions[i], forcing[i], _config(init))
        for i in range(emissions.shape[0])
    ]
    return f, ported


@pytest.mark.parametrize(
    "field",
    [
        "concentration",
        "forcing",
        "forcing_sum",
        "airborne_emissions",
        "toa_imbalance",
        "ocean_heat_content_change",
    ],
)
def test_matches_fair(run_pair, field):
    reference, ported = run_pair
    for i, out in enumerate(ported):
        expected = np.nan_to_num(getattr(reference, field).data[:, i, 0, ...])
        got = np.nan_to_num(np.asarray(getattr(out, field)))
        np.testing.assert_allclose(got, expected, rtol=RTOL, atol=1e-9)


def test_temperature_matches_fair(run_pair):
    reference, ported = run_pair
    for i, out in enumerate(ported):
        expected = reference.temperature.data[:, i, 0, :]
        np.testing.assert_allclose(
            np.asarray(out.temperature), expected, rtol=RTOL, atol=1e-12
        )


def test_ensemble_matches_single_runs():
    """vmap over configs and scenarios must agree with looping over them."""
    f = build_fair(scenarios=("ssp126", "ssp585"), configs=("a", "b", "c"))
    # give the configs genuinely different climates
    f.climate_configs["ocean_heat_transfer"].data[:, 0] = [0.8, 1.1, 1.5]
    f.climate_configs["deep_ocean_efficacy"].data[:] = [1.0, 1.28, 1.6]
    params, smap, emissions, forcing, init = fairjax.from_fair(f)

    ensemble = fairjax.run_ensemble(params, smap, emissions, forcing, init)

    for scenario in range(emissions.shape[0]):
        for config in range(3):
            one = fairjax.run(
                _config(params, config),
                smap,
                emissions[scenario],
                forcing[scenario],
                _config(init, config),
            )
            np.testing.assert_allclose(
                np.asarray(ensemble.temperature[scenario, config]),
                np.asarray(one.temperature),
                # vmap lets XLA reassociate the arithmetic, so this is a
                # floating-point-equivalence check, not a bit-for-bit one
                rtol=1e-9,
                atol=1e-14,
            )
    # and the configs must actually differ, or the test proves nothing
    spread = np.ptp(np.asarray(ensemble.temperature[0, :, -1, 0]))
    assert spread > 0.1, f"configs are too similar to be a real check ({spread=})"
