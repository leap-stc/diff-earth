"""Cummins et al. (2020) n-layer energy balance model, in JAX.

Mirrors ``fair.energy_balance_model`` for the deterministic case. The stochastic
terms of Cummins are dropped: the state vector keeps its leading "stochastic
forcing" element so that the matrices are shape-identical to FaIR's, but the
noise realisation is always zero.

Everything here is differentiable with respect to ``ocean_heat_capacity``,
``ocean_heat_transfer``, ``deep_ocean_efficacy`` and ``gamma_autocorrelation``.
"""

import jax
import jax.numpy as jnp

__all__ = ["eb_matrix", "discretise", "step_temperature", "toa_imbalance"]


def eb_matrix(
    ocean_heat_capacity,
    ocean_heat_transfer,
    deep_ocean_efficacy,
    gamma_autocorrelation,
    timestep=1.0,
):
    """Build the (n_layer+1, n_layer+1) continuous-time EBM matrix.

    Row/column 0 is the Cummins stochastic-forcing state; rows 1: are the ocean
    layers. Identical in construction to ``EnergyBalanceModel._eb_matrix``.
    """
    capacity = ocean_heat_capacity / timestep
    transfer = ocean_heat_transfer
    n_box = capacity.shape[0]

    # deep-ocean efficacy multiplies the heat exchange into the bottom layer
    epsilon = jnp.ones(n_box).at[n_box - 2].set(deep_ocean_efficacy)

    matrix = jnp.zeros((n_box, n_box))
    matrix = matrix.at[0, 0].set(
        -(transfer[0] + epsilon[0] * transfer[1]) / capacity[0]
    )
    matrix = matrix.at[0, 1].set(epsilon[0] * transfer[1] / capacity[0])

    # n_box is static, so this unrolls at trace time
    for row in range(1, n_box - 1):
        matrix = matrix.at[row, row - 1].set(transfer[row] / capacity[row])
        matrix = matrix.at[row, row].set(
            -(transfer[row] + epsilon[row] * transfer[row + 1]) / capacity[row]
        )
        matrix = matrix.at[row, row + 1].set(
            epsilon[row] * transfer[row + 1] / capacity[row]
        )

    matrix = matrix.at[n_box - 1, n_box - 2].set(transfer[-1] / capacity[-1])
    matrix = matrix.at[n_box - 1, n_box - 1].set(-transfer[-1] / capacity[-1])

    # prepend the stochastic row and column (Cummins eqs. 13-14)
    full = jnp.zeros((n_box + 1, n_box + 1))
    full = full.at[1:, 1:].set(matrix)
    full = full.at[0, 0].set(-gamma_autocorrelation)
    full = full.at[1, 0].set(1.0 / capacity[0])
    return full


def discretise(matrix, gamma_autocorrelation):
    """Discretise the continuous EBM over one timestep.

    Returns ``(eb_matrix_d, forcing_vector_d)`` such that

        state[t+1] = eb_matrix_d @ state[t] + forcing_vector_d * forcing[t]

    ``expm`` and ``solve`` are both differentiable in JAX, which is the whole
    reason the EBM ports across without special handling.
    """
    n_matrix = matrix.shape[0]
    matrix_d = jax.scipy.linalg.expm(matrix)

    forcing_vector = jnp.zeros(n_matrix).at[0].set(gamma_autocorrelation)
    forcing_vector_d = jax.scipy.linalg.solve(
        matrix, (matrix_d - jnp.identity(n_matrix)) @ forcing_vector
    )
    return matrix_d, forcing_vector_d


def step_temperature(state, matrix_d, forcing_vector_d, forcing):
    """Advance the EBM state by one timestep."""
    return matrix_d @ state + forcing_vector_d * forcing


def toa_imbalance(state, forcing, ocean_heat_transfer, deep_ocean_efficacy):
    """Top-of-atmosphere energy imbalance from the EBM state.

    ``state`` has the stochastic element at index 0, so ocean layer *i* lives at
    ``state[..., i + 1]`` -- matching FaIR's ``calculate_toa_imbalance_postrun``.
    """
    return (
        forcing
        - ocean_heat_transfer[..., 0] * state[..., 1]
        + (1 - deep_ocean_efficacy)
        * ocean_heat_transfer[..., -1]
        * (state[..., -2] - state[..., -1])
    )
