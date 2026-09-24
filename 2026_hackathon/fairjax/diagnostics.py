"""Emergent climate parameters, computed without an eigendecomposition.

FaIR derives ECS and TCR from an eigendecomposition of the energy balance
matrix (``scipy.linalg.eig``). That route does not port cleanly: JAX's ``eig``
is CPU-only and its gradient is ill-conditioned when eigenvalues are close
together, which happens routinely across a calibrated parameter ensemble.

Both quantities have equivalent definitions that need only arithmetic and a
scan, so we use those instead. Each is differentiable, which is what makes
"fit ECS to a target" a one-line optimisation rather than an ensemble search.
"""

import jax
import jax.numpy as jnp

from . import ebm
from .constants import DOUBLING_TIME_1PCT

__all__ = ["ecs", "tcr", "equilibrium_temperature"]


def equilibrium_temperature(params, forcing):
    """Steady-state warming for a sustained forcing.

    At equilibrium every ocean layer reaches the same temperature, so the
    interior heat-exchange terms cancel and the balance reduces to
    ``kappa_0 * T = F``. No matrix solve required.
    """
    return forcing / params.ocean_heat_transfer[0]


def ecs(params, forcing_2co2_4co2_ratio=0.5):
    """Equilibrium climate sensitivity (K)."""
    return equilibrium_temperature(
        params, params.forcing_4co2 * forcing_2co2_4co2_ratio
    )


def tcr(params, forcing_2co2_4co2_ratio=0.5, n_steps=700):
    """Transient climate response (K).

    Computed the way TCR is defined: integrate the energy balance model under a
    forcing ramping linearly to 2xCO2 over the 1%/yr doubling time (69.66 yr)
    and read off the warming at the end of the ramp.

    ``n_steps`` sets the sub-annual resolution of the ramp. The default agrees
    with FaIR's analytic ``ebms["tcr"]`` to about 1e-6 K; ``n_steps=70`` is ~10x
    cheaper and still agrees to 1e-4 K. Note that this is the *ramp* TCR, which
    differs by a few tenths of a percent from what a 1-year-timestep
    ``fairjax.run`` produces, because the annual discretisation and the Cummins
    forcing filter both matter at dt = 1 yr.
    """
    forcing_2co2 = params.forcing_4co2 * forcing_2co2_4co2_ratio
    timestep = DOUBLING_TIME_1PCT / n_steps

    matrix = ebm.eb_matrix(
        params.ocean_heat_capacity,
        params.ocean_heat_transfer,
        params.deep_ocean_efficacy,
        params.gamma_autocorrelation,
        timestep,
    )
    matrix_d, forcing_vector_d = ebm.discretise(matrix, params.gamma_autocorrelation)

    ramp = forcing_2co2 * jnp.arange(1, n_steps + 1) / n_steps

    def step(state, forcing):
        return ebm.step_temperature(state, matrix_d, forcing_vector_d, forcing), None

    state_0 = jnp.zeros(params.ocean_heat_capacity.shape[0] + 1)
    state, _ = jax.lax.scan(step, state_0, ramp)
    return state[1]
