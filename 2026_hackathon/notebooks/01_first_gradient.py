# %% [markdown]
# # 1. Your first gradient through a climate model
#
# `fairjax` is FaIR v2.2.4 rewritten in JAX. It gives the same numbers as stock
# FaIR to a relative tolerance of `1e-10`, and it is differentiable.
#
# By the end of this notebook you will have:
#
# 1. run FaIR and `fairjax` side by side and confirmed they agree,
# 2. taken a gradient of 2100 warming with respect to every model parameter,
# 3. run a 100-member ensemble with `vmap`.
#
# Nothing here needs a GPU.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import fairjax
from fairjax import diagnostics
from fairjax.reference import build_fair

print("float64 enabled:", jax.config.jax_enable_x64)

# %% [markdown]
# ## Build a model
#
# `fairjax` does not replace FaIR's front end. Configure a `fair.FAIR` object
# exactly as you normally would, then lift it into JAX arrays.
#
# `build_fair` is a convenience wrapper around the stock setup: all 64 default
# species, RCMIP emissions, a three-layer ocean.

# %%
f = build_fair(scenarios=("ssp119", "ssp245", "ssp585"))
params, smap, emissions, forcing, init = fairjax.from_fair(f)

print(f"{emissions.shape[0]} scenarios, {emissions.shape[1]} timepoints, "
      f"{emissions.shape[2]} species")

# %% [markdown]
# `from_fair` must be called **before** `f.run()`. FaIR stores `gas_partitions`
# as live state rather than a time series, so running first overwrites the
# initial condition with the final one. This is the kind of thing a port finds.
#
# `params` is a `NamedTuple` — an ordinary JAX pytree. That is what makes
# `jax.grad` work on it without any special handling.

# %%
one = lambda tree: type(tree)(*[jnp.asarray(leaf)[0] for leaf in tree])
p, i0 = one(params), one(init)

for name in ["ocean_heat_transfer", "forcing_scale", "partition_fraction"]:
    print(f"{name:24s} {np.shape(getattr(p, name))}")

# %% [markdown]
# ## Check it against stock FaIR
#
# Never trust a port you have not diffed.

# %%
f.run(progress=False)

outputs = [
    fairjax.run(p, smap, jnp.asarray(emissions[i]), jnp.asarray(forcing[i]), i0)
    for i in range(3)
]

for i, scenario in enumerate(f.scenarios):
    reference = f.temperature.data[:, i, 0, 0]
    ported = np.asarray(outputs[i].temperature[:, 0])
    print(f"{scenario:8s} 2100 warming  FaIR {reference[-1]:.6f} K   "
          f"fairjax {ported[-1]:.6f} K   max |diff| {np.abs(reference - ported).max():.2e} K")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
years = f.timebounds
for i, scenario in enumerate(f.scenarios):
    axes[0].plot(years, f.temperature.data[:, i, 0, 0], lw=3, alpha=0.35, label=f"FaIR {scenario}")
    axes[0].plot(years, outputs[i].temperature[:, 0], lw=1, ls="--", label=f"fairjax {scenario}")
    axes[1].semilogy(years, np.abs(f.temperature.data[:, i, 0, 0]
                                   - np.asarray(outputs[i].temperature[:, 0])) + 1e-18,
                     label=scenario)
axes[0].set(xlabel="year", ylabel="GSAT anomaly (K)", title="FaIR vs fairjax")
axes[0].legend(fontsize=8)
axes[1].set(xlabel="year", ylabel="|difference| (K)", title="agreement")
axes[1].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## The gradient
#
# Here is the whole point. Define a scalar you care about, and differentiate it
# with respect to the entire parameter set.

# %%
e245, fo245 = jnp.asarray(emissions[1]), jnp.asarray(forcing[1])

def warming_2100(q):
    return fairjax.run(q, smap, e245, fo245, i0).temperature[-1, 0]

gradient = jax.grad(warming_2100)(p)

print(f"2100 warming under ssp245: {warming_2100(p):.4f} K\n")
print("d(2100 warming) / d(ocean heat transfer) =", np.asarray(gradient.ocean_heat_transfer))
print("d(2100 warming) / d(deep ocean efficacy) =", float(gradient.deep_ocean_efficacy))

# %% [markdown]
# That single backward pass filled in a derivative for *every* parameter, not
# just the ones printed. Let us look at the forcing scales — one per species.

# %%
names = list(f.properties_df.index)
scale_gradient = np.asarray(gradient.forcing_scale)
order = np.argsort(-np.abs(scale_gradient))[:12]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh([names[j][:38] for j in order][::-1], scale_gradient[order][::-1],
        color=["#c0392b" if v > 0 else "#2471a3" for v in scale_gradient[order][::-1]])
ax.set(xlabel="d(2100 warming) / d(forcing scale)  [K]",
       title="Which species' forcing uncertainty matters most")
ax.axvline(0, color="k", lw=0.8)
fig.tight_layout()

# %% [markdown]
# ## How much did that cost?
#
# Reverse-mode autodiff gives you the whole gradient for a small constant
# multiple of one forward run, no matter how many parameters there are.

# %%
import time

run_jit = jax.jit(warming_2100)
grad_jit = jax.jit(jax.grad(warming_2100))
run_jit(p).block_until_ready()
jax.block_until_ready(grad_jit(p))

t0 = time.perf_counter()
for _ in range(20):
    run_jit(p).block_until_ready()
forward = (time.perf_counter() - t0) / 20

t0 = time.perf_counter()
for _ in range(20):
    jax.block_until_ready(grad_jit(p))
backward = (time.perf_counter() - t0) / 20

n_parameters = sum(np.size(leaf) for leaf in p)
print(f"forward run          {forward * 1e3:7.2f} ms")
print(f"gradient             {backward * 1e3:7.2f} ms   ({backward / forward:.1f}x forward)")
print(f"parameters           {n_parameters}")
print(f"finite differences   {2 * n_parameters * forward:7.1f} s   "
      f"({2 * n_parameters * forward / backward:.0f}x slower)")

# %% [markdown]
# ## Ensembles with `vmap`
#
# `fairjax.run` handles one scenario and one config. Ensembles are a `vmap`,
# which is how the port stays readable — no five-dimensional broadcasting.

# %%
n_members = 100
rng = np.random.default_rng(0)

ensemble = p._replace(
    ocean_heat_transfer=jnp.asarray(
        np.asarray(p.ocean_heat_transfer) * rng.lognormal(0, 0.18, (n_members, 3))),
    deep_ocean_efficacy=jnp.asarray(
        float(p.deep_ocean_efficacy) * rng.lognormal(0, 0.12, n_members)),
    **{name: jnp.broadcast_to(jnp.asarray(getattr(p, name)),
                              (n_members,) + np.shape(getattr(p, name)))
       for name in p._fields if name not in ("ocean_heat_transfer", "deep_ocean_efficacy")},
)
init_ensemble = type(i0)(*[jnp.broadcast_to(jnp.asarray(leaf), (n_members,) + np.shape(leaf))
                           for leaf in i0])

member = jax.vmap(lambda q, s: fairjax.run(q, smap, e245, fo245, s), in_axes=(0, 0))
spread = member(ensemble, init_ensemble)
print("ensemble temperature shape:", spread.temperature.shape)

# %%
gsat = np.asarray(spread.temperature[:, :, 0])
fig, ax = plt.subplots(figsize=(7, 4))
ax.fill_between(years, *np.percentile(gsat, [5, 95], axis=0), alpha=0.25,
                color="#2471a3", label="5-95%")
ax.fill_between(years, *np.percentile(gsat, [25, 75], axis=0), alpha=0.35,
                color="#2471a3", label="25-75%")
ax.plot(years, np.median(gsat, axis=0), color="#154360", label="median")
ax.set(xlabel="year", ylabel="GSAT anomaly (K)",
       title=f"{n_members}-member ocean parameter ensemble, ssp245")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## ECS and TCR are differentiable too
#
# FaIR derives these by eigendecomposing the energy balance matrix. That does
# not port well — JAX's `eig` is CPU-only and its gradient is ill-conditioned
# when eigenvalues are close. Both have equivalent definitions needing only
# arithmetic, so `fairjax.diagnostics` uses those instead.

# %%
print(f"ECS  FaIR {f.ebms['ecs'].data[0]:.6f} K   fairjax {diagnostics.ecs(p):.6f} K")
print(f"TCR  FaIR {f.ebms['tcr'].data[0]:.6f} K   fairjax {diagnostics.tcr(p):.6f} K")
print()
print("d(ECS)/d(kappa_1)  =", float(jax.grad(diagnostics.ecs)(p).ocean_heat_transfer[0]))
print("d(TCR)/d(kappa_1)  =", float(jax.grad(diagnostics.tcr)(p).ocean_heat_transfer[0]))

# %% [markdown]
# ## Try it
#
# - Differentiate a different target: peak warming, the year warming crosses
#   1.5 °C (careful — that one is not smooth), 2100 ocean heat content.
# - Which *carbon cycle* parameter does 2100 CO₂ concentration care about most?
#   Look at `gradient.iirf_0`, `iirf_temperature`, `iirf_uptake`.
# - Compare the gradient under ssp119 and ssp585. Does the ranking change?
#
# Then move on to notebook 2.
