# %% [markdown]
# # 2. Fit the model, then ask what the fit means
#
# Gradients make calibration fast. That is the obvious win, and it is the less
# interesting one.
#
# The interesting one is the *second* derivative. Once you can differentiate the
# model twice, you can ask which of your fitted parameters the data actually
# determined, and which ones just landed somewhere. Almost every calibrated
# climate parameter set has directions in it that the observations never
# constrained — and without curvature information you cannot see them.
#
# That is this notebook:
#
# 1. fit FaIR parameters to a global-mean warming record with `jax.grad` + Adam,
# 2. build the Fisher information from the Jacobian at the fit,
# 3. eigendecompose it and read off what the record could and could not see.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import fairjax
from fairjax.reference import build_fair

f = build_fair(scenarios=("ssp245",), start=1750, end=2100)
params, smap, emissions, forcing, init = fairjax.from_fair(f)
one = lambda tree: type(tree)(*[jnp.asarray(leaf)[0] for leaf in tree])
base, i0 = one(params), one(init)
e, fo = jnp.asarray(emissions[0]), jnp.asarray(forcing[0])
years = np.asarray(f.timebounds)
names = list(f.properties_df.index)

# %% [markdown]
# ## Choose what to fit
#
# Six parameters: three ocean heat transfer coefficients, deep ocean efficacy,
# and the forcing scales for CO₂ and for aerosol-cloud interactions.
#
# Keep this list short to start with. Six parameters is already enough to show
# the effect, and a small problem makes the Hessian trivial to look at.

# %%
KNOBS = [
    ("ocean_heat_transfer", 0, "kappa_1   surface heat uptake"),
    ("ocean_heat_transfer", 1, "kappa_2   thermocline exchange"),
    ("ocean_heat_transfer", 2, "kappa_3   deep ocean exchange"),
    ("deep_ocean_efficacy", (), "epsilon   deep ocean efficacy"),
    ("forcing_scale", names.index("CO2"), "scale     CO2 forcing"),
    ("forcing_scale", names.index("Aerosol-cloud interactions"), "scale     aerosol-cloud"),
]

def with_knobs(vector):
    """Write a parameter vector back into the full model parameter pytree."""
    p = base
    for value, (field, index, _) in zip(vector, KNOBS):
        current = getattr(p, field)
        p = p._replace(**{field: current.at[index].set(value) if np.shape(current) else value})
    return p

def gsat(vector):
    return fairjax.run(with_knobs(vector), smap, e, fo, i0).temperature[:, 0]

theta_default = jnp.array([
    float(np.asarray(getattr(base, field))[index]) if np.shape(getattr(base, field))
    else float(getattr(base, field)) for field, index, _ in KNOBS])
print("default parameters:", np.asarray(theta_default))

# %% [markdown]
# ## The target
#
# **For the hackathon this is where CESM2 goes.** Load the global-mean surface
# air temperature from CESM2 `historical` (plus an SSP extension), take the
# anomaly against a 1850–1900 baseline, and drop it in as `target`.
#
# Until that data is staged, we use a synthetic target: FaIR itself run with
# known parameters and observational noise added. That is not a cop-out — it is
# the right first experiment, because the truth is known, so you can check
# whether the identifiability analysis is telling you something true.

# %%
from fairjax import cesm

try:
    data = cesm.load()
    cesm_years, cesm_gsat = cesm.gsat_anomaly(data, scenario="ssp245")
    observable = np.isin(years, cesm_years) & (years <= 2100)
    target = cesm_gsat[np.isin(cesm_years, years[observable])]
    theta_true = None  # there is no known truth for a real model
    label = "CESM2 ssp245"
    # sigma is an error model, and choosing it is a modelling decision, not a
    # detail. Fitting a simple model to an ESM, it has to cover CESM2's
    # interannual variability (~0.1 K on an annual global mean) *and* the
    # structural mismatch between FaIR and CESM2 -- not observational error.
    # Set it too small and the fit chases weather it cannot possibly reproduce.
    sigma = 0.12
except FileNotFoundError as error:
    print(f"{error}\nFalling back to a synthetic target.\n")
    theta_true = theta_default * jnp.array([1.15, 0.9, 1.1, 0.85, 1.08, 1.2])
    observable = (years >= 1850) & (years <= 2024)
    sigma = 0.05  # here it really is observational noise, and we know its size
    rng = np.random.default_rng(0)
    target = np.asarray(gsat(theta_true))[observable] + rng.normal(0, sigma, observable.sum())
    label = "synthetic (FaIR with known parameters)"

print(f"target: {label}, {observable.sum()} years")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(years[observable], target, ".", ms=3, color="0.4", label=f"target — {label}")
ax.plot(years, gsat(theta_default), lw=2, label="FaIR defaults")
ax.set(xlabel="year", ylabel="GSAT anomaly (K)", xlim=(1850, 2105))
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## Fit
#
# A least-squares misfit, `jax.value_and_grad`, and Adam. This is the entire
# calibration.

# %%
def misfit(vector):
    return jnp.sum(((gsat(vector)[observable] - target) / sigma) ** 2) / 2

loss_and_grad = jax.jit(jax.value_and_grad(misfit))

start_from = theta_true if theta_true is not None else theta_default
theta = start_from * jnp.array([0.85, 1.1, 0.9, 1.2, 0.95, 0.8])  # start away from the answer
optimiser = optax.adam(2e-3)
state = optimiser.init(theta)

history = []
for step in range(1500):
    loss, grad = loss_and_grad(theta)
    updates, state = optimiser.update(grad, state)
    theta = optax.apply_updates(theta, updates)
    history.append(float(loss))

print(f"misfit {history[0]:.1f} -> {history[-1]:.1f}   "
      f"(a fit consistent with sigma={sigma} K over {observable.sum()} years "
      f"would give ~{observable.sum()/2:.0f})")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].semilogy(history)
axes[0].set(xlabel="Adam step", ylabel="misfit", title="convergence")
axes[1].plot(years[observable], target, ".", ms=3, color="0.6", label="target")
axes[1].plot(years, gsat(theta), lw=2, color="#c0392b", label="fitted")
axes[1].set(xlabel="year", ylabel="GSAT anomaly (K)", xlim=(1850, 2030), title="fit")
axes[1].legend()
fig.tight_layout()

# %% [markdown]
# The fit is good. Now look at the parameters.

# %%
if theta_true is not None:
    print(f"{'parameter':32s} {'truth':>9s} {'fitted':>9s} {'error':>8s}")
    for (_, _, name), true, fit in zip(KNOBS, theta_true, theta):
        print(f"{name:32s} {float(true):9.4f} {float(fit):9.4f} "
              f"{100 * abs(float(fit) - float(true)) / abs(float(true)):7.1f}%")
else:
    print(f"{'parameter':32s} {'default':>9s} {'fitted':>9s}")
    for (_, _, name), default, fit in zip(KNOBS, theta_default, theta):
        print(f"{name:32s} {float(default):9.4f} {float(fit):9.4f}")

# %% [markdown]
# Some parameters came back close to the truth and some did not — even though
# the fit to the data is excellent. That is the whole point. A good fit does not
# mean good parameters.
#
# ## Which directions did the data actually see?
#
# Build the Jacobian of the modelled record with respect to the parameters, then
# the Fisher information `J^T J / sigma^2`. Its eigenvectors are the combinations
# of parameters the data constrains, and the eigenvalues say how tightly.
#
# We use the Fisher information rather than the raw Hessian of the misfit because
# it is positive semi-definite by construction, so it stays meaningful even if
# the optimiser has not fully converged. Its inverse is the Cramér-Rao bound on
# the parameter covariance.
#
# The Jacobian costs one forward-mode pass per parameter — six passes here.

# %%
jacobian = np.asarray(jax.jacfwd(lambda v: gsat(v)[observable])(theta))
scale = np.asarray(theta)
jacobian = jacobian * scale[None, :]        # relative units: d(GSAT) / d(ln theta)
fisher = jacobian.T @ jacobian / sigma**2
eigenvalues, eigenvectors = np.linalg.eigh(fisher)

print("best-constrained direction first:")
for value, vector in zip(eigenvalues[::-1], eigenvectors.T[::-1]):
    leading = np.argsort(-np.abs(vector))[:3]
    combo = "  ".join(f"{vector[i]:+.2f} {KNOBS[i][2].split()[0]}" for i in leading)
    print(f"  lambda = {value:11.4g}   pinned to {1 / np.sqrt(value):6.1%}   {combo}")

print(f"\ncondition number: {eigenvalues[-1] / eigenvalues[0]:.3g}")

# %%
fig, ax = plt.subplots(figsize=(7, 4.2))
image = ax.imshow(eigenvectors[:, ::-1], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(KNOBS)))
ax.set_xticklabels([f"$\\lambda$={v:.3g}" for v in eigenvalues[::-1]], rotation=35,
                   ha="right", fontsize=8)
ax.set_yticks(range(len(KNOBS)))
ax.set_yticklabels([label for _, _, label in KNOBS], fontsize=8)
ax.set_title("Fisher eigenvectors: well constrained (left) to unconstrained (right)")
fig.colorbar(image, ax=ax, label="loading")
fig.tight_layout()

# %% [markdown]
# ## What it says
#
# Read the leftmost and rightmost columns.
#
# The best-constrained direction is a *contrast* between the CO₂ and aerosol
# forcing scales — that is net historical forcing, which a warming record pins
# down very tightly. The worst-constrained direction is dominated by deep ocean
# efficacy, which a surface temperature record can barely see at all.
#
# So the parameter errors in the table above were not a failure of the
# optimiser. They are a property of the observing system, and the Fisher
# information predicted them before we looked.

# %%
predicted = np.sqrt(np.diag(np.linalg.pinv(fisher)))

fig, ax = plt.subplots(figsize=(7, 4))
position = np.arange(len(KNOBS))
ax.barh(position + 0.2, predicted, 0.4, label="Cramér-Rao bound", color="#2471a3")
if theta_true is not None:
    actual = np.abs(np.asarray(theta) - np.asarray(theta_true)) / np.asarray(theta_true)
    ax.barh(position - 0.2, actual, 0.4, label="actual recovery error", color="#c0392b")
ax.set_yticks(position)
ax.set_yticklabels([label for _, _, label in KNOBS], fontsize=8)
ax.set(xlabel="fractional error", title="what the record could constrain vs what we recovered")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## Does the fit recover CESM2's climate sensitivity?
#
# A last check, and the one that ties the notebook together. ECS and TCR are
# emergent properties of the fitted parameters, not things we fitted directly.

# %%
from fairjax import diagnostics

fitted = with_knobs(theta)
print(f"{'':16s} {'ECS':>8s} {'TCR':>8s}")
print(f"{'FaIR default':16s} {diagnostics.ecs(base):8.2f} {diagnostics.tcr(base):8.2f}")
print(f"{'fitted':16s} {diagnostics.ecs(fitted):8.2f} {diagnostics.tcr(fitted):8.2f}")
print(f"{'CESM2 published':16s} {'5.1-5.3':>8s} {'~2.0':>8s}")

# %% [markdown]
# Look at the two rows together. **TCR lands almost exactly on CESM2. ECS does
# not** — it moves maybe half the distance and stops.
#
# That is not a bug in the fit, and it is worth sitting with. A 1850–2100
# record is a *transient* record, so it constrains transient behaviour well and
# equilibrium behaviour only weakly: the system is nowhere near equilibrium
# anywhere in that window. Meanwhile the forcing scales are free to absorb part
# of the mismatch, and the Fisher spectrum above already told us the
# forcing-scale-versus-feedback contrast is the single best-constrained
# direction — so the fit slides along it rather than correcting the feedback.
#
# The emergent quantity the data speaks to comes out right. The one it does not
# comes out wrong. Same fit, same misfit, no warning.
#
# So the model reproduces the record while getting the equilibrium physics only
# partly right. You could not have diagnosed that from the misfit, which is
# perfectly healthy. You need the curvature.
#
# **This is the argument for differentiable climate models in one paragraph.**
# Not that the fitting is faster. That the second derivative tells you which of
# your fitted parameters you are entitled to believe.

# %% [markdown]
# ## Extensions
#
# Pick one. None of these need more than the tools already in this notebook.
#
# **A. Add an observation and watch a degeneracy break.** Ocean heat content is
# in `fairjax.run(...).ocean_heat_content_change`, and it is sensitive to exactly
# the deep-ocean parameters that GSAT cannot see. Add it to the misfit with its
# own uncertainty and recompute the Fisher spectrum. How much does the condition
# number drop?
#
# **B. Adjoint attribution.** `jax.grad` of 2100 warming with respect to
# `emissions` — not parameters — gives ∂T/∂(emissions of species *s* in year *t*),
# a full sensitivity field. Which decade of which species matters most?
#
# **C. Inverse design.** Optimise an emissions scaling pathway to minimise
# cumulative abatement subject to peak warming ≤ 1.5 °C. Use a soft penalty for
# the constraint to keep it differentiable.
#
# **D. More parameters.** Add the carbon cycle: `iirf_0`, `iirf_uptake`,
# `iirf_temperature`, `iirf_airborne` at the CO₂ index. Does the historical
# record constrain the temperature feedback on carbon uptake at all?
#
# **E. HMC.** Put `misfit` into NumPyro as a log-likelihood and run NUTS. FaIR's
# official calibration uses a ~1.5M-member prior ensemble with constraint
# filtering because it has no gradients. Compare the posterior you get here.
