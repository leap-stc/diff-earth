"""The point of the port: gradients that exist, are finite, and are correct.

A forward model that merely runs under JAX is not enough. FaIR's NaN-as-mask
convention makes it very easy to build something that reproduces FaIR exactly
and then hands back an all-NaN Jacobian, so the finiteness check below is doing
real work.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fairjax
from fairjax import diagnostics
from fairjax.reference import build_fair


@pytest.fixture(scope="module")
def model():
    f = build_fair(scenarios=("ssp245",))
    params, smap, emissions, forcing, init = fairjax.from_fair(f)
    one = lambda tree: type(tree)(*[jnp.asarray(leaf)[0] for leaf in tree])
    p, i0 = one(params), one(init)
    e, fo = jnp.asarray(emissions[0]), jnp.asarray(forcing[0])

    def warming_2100(q):
        return fairjax.run(q, smap, e, fo, i0).temperature[-1, 0]

    return p, warming_2100


def test_gradient_is_finite_everywhere(model):
    """No NaN or inf in any parameter's gradient."""
    params, warming_2100 = model
    grad = jax.grad(warming_2100)(params)
    bad = {
        name: np.asarray(value)
        for name, value in grad._asdict().items()
        if not np.all(np.isfinite(np.asarray(value)))
    }
    assert not bad, f"non-finite gradients for: {sorted(bad)}"


def test_gradient_is_nonzero_for_the_parameters_that_matter(model):
    params, warming_2100 = model
    grad = jax.grad(warming_2100)(params)
    for name in [
        "ocean_heat_transfer",
        "ocean_heat_capacity",
        "forcing_scale",
        "iirf_0",
        "iirf_temperature",
        "partition_fraction",
    ]:
        assert np.any(np.asarray(getattr(grad, name)) != 0), f"{name} gradient is all zero"


@pytest.mark.parametrize(
    "field,index,step",
    [
        ("ocean_heat_transfer", 0, 1e-3),
        ("ocean_heat_transfer", 2, 1e-3),
        ("ocean_heat_capacity", 0, 1e-2),
        ("deep_ocean_efficacy", (), 1e-3),
    ],
)
def test_gradient_matches_finite_difference(model, field, index, step):
    """Central differences, at a step large enough to stay out of the roundoff floor.

    A 350-step scan through ``expm`` amplifies cancellation error, so steps
    below about 1e-4 are dominated by noise rather than truncation -- the
    finite difference is the less accurate of the two here, not the gradient.
    """
    params, warming_2100 = model
    analytic = np.asarray(getattr(jax.grad(warming_2100)(params), field))[index]

    base = np.asarray(getattr(params, field))

    def perturbed(delta):
        values = base.copy()
        values[index] += delta
        return float(warming_2100(params._replace(**{field: jnp.asarray(values)})))

    numeric = (perturbed(step) - perturbed(-step)) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-3)


def test_hessian_is_finite_and_symmetric():
    """Second derivatives underpin the identifiability analysis, so check them."""
    f = build_fair(scenarios=("ssp245",))
    params, smap, emissions, forcing, init = fairjax.from_fair(f)
    one = lambda tree: type(tree)(*[jnp.asarray(leaf)[0] for leaf in tree])
    p, i0 = one(params), one(init)
    e, fo = jnp.asarray(emissions[0]), jnp.asarray(forcing[0])

    def warming(vector):
        q = p._replace(
            ocean_heat_transfer=vector[:3],
            deep_ocean_efficacy=vector[3],
        )
        return fairjax.run(q, smap, e, fo, i0).temperature[-1, 0]

    vector = jnp.concatenate([p.ocean_heat_transfer, p.deep_ocean_efficacy[None]])
    hessian = np.asarray(jax.hessian(warming)(vector))
    assert np.all(np.isfinite(hessian))
    np.testing.assert_allclose(hessian, hessian.T, rtol=1e-8, atol=1e-10)


def test_ecs_and_tcr_match_fair():
    f = build_fair()
    params, smap, _, _, _ = fairjax.from_fair(f)
    one = lambda tree: type(tree)(*[jnp.asarray(leaf)[0] for leaf in tree])
    p = one(params)
    f.run(progress=False)
    np.testing.assert_allclose(diagnostics.ecs(p), f.ebms["ecs"].data[0], rtol=1e-12)
    np.testing.assert_allclose(diagnostics.tcr(p), f.ebms["tcr"].data[0], rtol=1e-5)


def test_ecs_is_differentiable():
    f = build_fair()
    params, _, _, _, _ = fairjax.from_fair(f)
    one = lambda tree: type(tree)(*[jnp.asarray(leaf)[0] for leaf in tree])
    p = one(params)
    grad = jax.grad(diagnostics.ecs)(p)
    # ECS = F_2x / kappa_0, so only these two entries should move it
    assert np.isfinite(grad.ocean_heat_transfer[0]) and grad.ocean_heat_transfer[0] != 0
    assert np.isfinite(grad.forcing_4co2) and grad.forcing_4co2 != 0
    np.testing.assert_allclose(np.asarray(grad.ocean_heat_transfer)[1:], 0, atol=1e-15)
