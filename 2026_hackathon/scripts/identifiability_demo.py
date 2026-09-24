"""Anchor result for the hackathon: what can a warming record actually constrain?

Fit a handful of FaIR parameters to a target global-mean temperature record,
then take the Hessian of the misfit at the optimum and look at its eigenvectors.
Small eigenvalues are directions in parameter space the data cannot see -- the
combinations you are free to move without changing the fit at all.

Here the target is FaIR itself run with known parameters, so the "truth" is
known and the degeneracies found are properties of the *observing system*, not
of the target model. Swap ``target`` for a CESM2 record and the same code
answers the same question about CESM2.

Run: python scripts/identifiability_demo.py
"""

import jax
import jax.numpy as jnp
import numpy as np

import fairjax
from fairjax.reference import build_fair

# the parameters we try to learn, and the scale on which they vary
KNOBS = [
    ("ocean_heat_transfer", 0, "kappa_1  surface heat uptake"),
    ("ocean_heat_transfer", 1, "kappa_2  thermocline exchange"),
    ("ocean_heat_transfer", 2, "kappa_3  deep ocean exchange"),
    ("deep_ocean_efficacy", (), "epsilon  deep ocean efficacy"),
    ("forcing_scale", "CO2", "scale    CO2 forcing"),
    ("forcing_scale", "Aerosol-cloud interactions", "scale    aerosol-cloud forcing"),
]


def build():
    f = build_fair(scenarios=("ssp245",), start=1750, end=2100)
    params, smap, emissions, forcing, init = fairjax.from_fair(f)
    one = lambda tree: type(tree)(*[jnp.asarray(leaf)[0] for leaf in tree])
    names = list(f.properties_df.index)
    knobs = [
        (field, names.index(index) if isinstance(index, str) else index, label)
        for field, index, label in KNOBS
    ]
    return one(params), smap, jnp.asarray(emissions[0]), jnp.asarray(forcing[0]), one(init), knobs, f


def main():
    base, smap, emissions, forcing, init, knobs, f = build()

    def with_knobs(vector):
        params = base
        for value, (field, index, _) in zip(vector, knobs):
            current = getattr(params, field)
            updated = current.at[index].set(value) if np.shape(current) else value
            params = params._replace(**{field: updated})
        return params

    def gsat(vector):
        return fairjax.run(with_knobs(vector), smap, emissions, forcing, init).temperature[:, 0]

    theta_true = jnp.array(
        [float(np.asarray(getattr(base, field))[index]) if np.shape(getattr(base, field))
         else float(getattr(base, field)) for field, index, _ in knobs]
    )
    # a "truth" that is not the default, so recovering it means something
    theta_true = theta_true * jnp.array([1.15, 0.9, 1.1, 0.85, 1.08, 1.2])

    timebounds = np.arange(1750, 2101)
    observed = np.asarray(gsat(theta_true))
    # only the historical record is observable, and only to ~0.05 K
    window = (timebounds >= 1850) & (timebounds <= 2024)
    sigma = 0.05
    rng = np.random.default_rng(0)
    target = observed[window] + rng.normal(0, sigma, window.sum())

    def misfit(vector):
        return jnp.sum(((gsat(vector)[window] - target) / sigma) ** 2) / 2

    # --- fit -----------------------------------------------------------------
    import optax

    theta = theta_true * jnp.array([0.85, 1.1, 0.9, 1.2, 0.95, 0.8])  # start away from truth
    optimiser = optax.adam(2e-3)
    state = optimiser.init(theta)
    loss_and_grad = jax.jit(jax.value_and_grad(misfit))
    for step in range(1500):
        loss, grad = loss_and_grad(theta)
        updates, state = optimiser.update(grad, state)
        theta = optax.apply_updates(theta, updates)
        if step % 300 == 0:
            print(f"  step {step:5d}  misfit {float(loss):12.3f}")
    print(f"  step {1500:5d}  misfit {float(misfit(theta)):12.3f}\n")

    print(f"{'parameter':34s} {'truth':>10s} {'fitted':>10s} {'error':>9s}")
    for (field, index, label), true, fit in zip(knobs, theta_true, theta):
        print(f"{label:34s} {float(true):10.4f} {float(fit):10.4f} "
              f"{100*abs(float(fit)-float(true))/abs(float(true)):8.1f}%")

    # --- identifiability -----------------------------------------------------
    # Use the Fisher information J^T J / sigma^2 rather than the raw Hessian of
    # the misfit. It is positive semi-definite by construction, so it stays
    # meaningful even if the optimiser has not fully converged, and its inverse
    # is the Cramer-Rao bound on the parameter covariance -- exactly the
    # "what can this record constrain" question. The Jacobian costs one
    # forward-mode pass per parameter, so six passes here.
    jacobian = np.asarray(jax.jacfwd(lambda v: gsat(v)[window])(theta))
    scale = np.asarray(theta)
    jacobian = jacobian * scale[None, :]  # relative units: d(GSAT) / d(ln theta)
    fisher = jacobian.T @ jacobian / sigma**2
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)

    print("\nFisher information eigenvalues (relative units), best-constrained first:")
    print("  a direction with eigenvalue L is pinned to about 1/sqrt(L), fractionally")
    for value, vector in zip(eigenvalues[::-1], eigenvectors.T[::-1]):
        constraint = f"{1 / np.sqrt(value):7.1%}" if value > 1e-12 else "   inf "
        leading = np.argsort(-np.abs(vector))[:3]
        combo = "  ".join(f"{vector[i]:+.2f} {knobs[i][2].split()[0]}" for i in leading)
        print(f"  lambda = {value:11.4g}   pinned to {constraint}   {combo}")

    print(f"\ncondition number {eigenvalues[-1] / max(eigenvalues[0], 1e-30):.3g}")
    print("\nflattest direction -- the combination this record cannot see:")
    for i in np.argsort(-np.abs(eigenvectors[:, 0])):
        print(f"    {eigenvectors[i, 0]:+.3f}  {knobs[i][2]}")

    print("\npredicted vs actual recovery error, per parameter:")
    predicted = np.sqrt(np.diag(np.linalg.pinv(fisher)))
    actual = np.abs(np.asarray(theta) - np.asarray(theta_true)) / np.asarray(theta_true)
    print(f"  {'parameter':34s} {'Cramer-Rao':>11s} {'actual':>9s}")
    for (_, _, label), p_err, a_err in zip(knobs, predicted, actual):
        print(f"  {label:34s} {p_err:10.1%} {a_err:8.1%}")


if __name__ == "__main__":
    main()
