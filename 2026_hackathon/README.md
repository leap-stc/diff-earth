# Differentiable FaIR — 2026 annual meeting hackathon

`fairjax` is a port of the [FaIR](https://github.com/OMS-NetZero/FAIR) simple climate
model (v2.2.4) to JAX. It reproduces stock FaIR to a relative tolerance of `1e-10`
and is differentiable end to end, so `jax.grad`, `jax.hessian`, `jax.vmap` and
`jax.jit` all work on the full model.

```
forward run, 1750-2100, 64 species     3.4 ms
gradient w.r.t. every parameter       12.5 ms
```

A gradient of 2100 warming with respect to all ~2000 model parameters costs about
four forward runs. The same information by finite differences would cost ~2000.

## Quick start

```bash
conda env create -f environment.yml
conda activate fair-jax
pip install -e .
pytest -q                                  # 17 tests, ~17 s
python scripts/identifiability_demo.py     # the anchor result, ~30 s
```

```python
import jax, jax.numpy as jnp, fairjax
from fairjax.reference import build_fair

f = build_fair(scenarios=("ssp245",))          # a stock FaIR object
params, smap, emissions, forcing, init = fairjax.from_fair(f)

one = lambda tree: type(tree)(*[jnp.asarray(x)[0] for x in tree])
p, i0 = one(params), one(init)
e, fo = jnp.asarray(emissions[0]), jnp.asarray(forcing[0])

warming = lambda q: fairjax.run(q, smap, e, fo, i0).temperature[-1, 0]
print(warming(p))                               # 3.4334 K
print(jax.grad(warming)(p).ocean_heat_transfer) # [-2.29, -0.15, -0.34] K per W m-2 K-1
```

**`from_fair` must be called before `f.run()`.** FaIR keeps `gas_partitions` as live
state rather than a time series and overwrites it during the run.

## What the hackathon is

Half a day, ~10 people, mixed JAX experience. The model is already ported, so the
session is about using it.

| Time | |
| --- | --- |
| 0:00–0:25 | Live intro: `jax.grad` on the FaIR energy balance model. Everyone runs one cell and sees `dT/dκ`. |
| 0:25–0:50 | Notebook 1: run `fairjax`, check it against stock FaIR, `vmap` an ensemble. |
| 0:50–2:20 | Notebook 2 in pairs: fit to CESM2, then the identifiability analysis. Extensions at the bottom for fast pairs. |
| 2:20–3:00 | Share-out. |

One anchor question everyone does, with optional extensions rather than parallel
tracks — with 10 people and 3 hours, five tracks means five unfinished things.

### The anchor question

*What can a global-mean temperature record actually constrain?*

Fit a handful of FaIR parameters to a target warming record, then eigendecompose the
Fisher information at the fit. Large eigenvalues are directions the data pins down;
small ones are combinations you can move freely without changing the fit.
`scripts/identifiability_demo.py` runs this end to end against a synthetic target and
finds a condition number of ~6×10⁴:

```
lambda =   3.965e+04   pinned to    0.5%   +0.72 CO2_scale  -0.59 aerosol_scale  -0.34 kappa_1
lambda =      0.6484   pinned to  124.2%   +0.87 epsilon    -0.41 kappa_3        +0.22 kappa_2
```

The contrast between CO₂ and aerosol forcing — net historical forcing — is pinned to
half a percent. Deep-ocean efficacy is not constrained at all. Swap the synthetic
target for a CESM2 record and the same code answers the same question about CESM2.

This is worth doing because it cannot fail, and because it is the honest reason to
want gradients: not that optimisation is faster, but that the *second* derivative
tells you which of your fitted parameters mean anything.

### Extensions

- **Add an observation and watch a degeneracy break.** Ocean heat content is
  sensitive to exactly the deep-ocean parameters GSAT cannot see. Verified: adding
  OHC to the misfit drops the condition number from 5.6×10⁴ to 8.4×10³ and tightens
  the worst-constrained direction from 120% to 31%. This is the best first extension
  — it works, and it makes the point that identifiability is a property of the
  *observing system*, not the model.
- **Adjoint attribution.** One backward pass gives ∂(2100 warming)/∂(emissions of
  species *s* in year *t*) — a full sensitivity field over species × time.
- **Inverse design.** Solve for the emissions pathway minimising cumulative abatement
  subject to peak warming ≤ 1.5 °C.
- **HMC.** Put the model in NumPyro and run NUTS. FaIR's official calibration uses a
  ~1.5M-member prior ensemble with constraint filtering precisely because it has no
  gradients.

### Fitting to CESM2

FaIR is global-mean and emissions-driven, so the targets are CESM2-derived global
timeseries:

| Experiment | Target | Constrains |
| --- | --- | --- |
| `abrupt-4xCO2`, `piControl` | Gregory regression → λ, F₄ₓ | energy balance model |
| `1pctCO2` | TCR | ocean heat uptake |
| `historical`, `ssp126/245/370/585` | GSAT, TOA net flux, OHC | forcing scales, aerosols |
| `esm-hist`, `1pctCO2-bgc` | airborne fraction | carbon cycle `r0, rU, rT, rA` |

**Stage this data before the session.** `scripts/stage_cesm.py` pulls CESM2 global
annual means from the public Pangeo CMIP6 catalog on GCS and writes a single small
netCDF to `data/`. Run it ahead of time and commit the result — ten people
simultaneously pulling CMIP6 zarr is the most common way these events lose their
first hour. (Needs `gcsfs`, `zarr`, `intake-esm`, which are not core dependencies.)

Three things the staging script had to learn the hard way, all worth knowing before
anyone fits to this data. The staged file has them fixed; they are recorded because
the same traps are waiting in any CMIP6 analysis.

- **CESM2's SSP runs have no `r1i1p1f1` member.** The available ones are `r4`, `r10`,
  `r11`. Querying for `r1i1p1f1` returns nothing and silently drops the experiment,
  leaving a target that stops in 2014. Worse, `sorted()` on member ids is
  lexicographic — `"r10i1p1f1" < "r1i1p1f1"` — so a naive "first member" pick is not
  the first member. The script now intersects members across each *splice group*
  (`historical` + its SSP extensions are one continuous simulation, so mixing
  realisations would join two different draws of internal variability) and sorts
  numerically. The chosen members are recorded in the file's `members` attribute.

- **Gregory regression needs TOA flux *anomalies*.** CESM2's control carries a
  +0.70 W m⁻² residual imbalance, which lands directly in the regression intercept:

  ```
  raw flux          lambda = 0.636   F_4x = 7.26   ECS = 5.71 K     <- wrong
  drift corrected   lambda = 0.636   F_4x = 6.56   ECS = 5.16 K
  published CESM2   lambda 0.63-0.68               ECS 5.15-5.3 K
  ```

  The feedback parameter is unaffected — only the intercept moves — so the error is
  invisible unless you check ECS against the literature. `cesm.toa_imbalance()`
  drift-corrects by default and raises rather than silently returning a raw flux if
  the piControl radiation is missing.

- **One year axis per experiment, not per variable.** Giving each variable its own
  `year_*` dimension makes `d["piControl_rsdt"] - d["piControl_rsut"]` broadcast to
  2-D instead of differencing — silently, and with a plausible-looking result.

**Caveat worth putting on slide 3:** CESM2's effective radiative forcing is not
directly diagnosable from standard CMIP output. Either use CESM2's RFMIP `piClim-*`
runs, or accept that forcing error is absorbed into the fitted feedback parameters —
which is part of what the identifiability analysis measures. Naming this up front
turns a gotcha into the framing.

Nothing here needs a GPU. Tiny matrices, ~350 timesteps, `vmap` over configs on a
laptop CPU. Say so, or someone will spend the morning on CUDA.

## What was involved in the port

FaIR turned out to be an unusually good target.

**No implicit solvers and no root-finding.** FaIR 1.6 solved for the carbon-cycle
lifetime scaling by Newton iteration on iIRF100, which would have forced implicit
differentiation. v2.x replaced it with a closed form, so every timestep is a smooth
explicit map:

```python
alpha = g0 * exp(iirf / g1)
```

**Already array-shaped.** FaIR carries `(time, scenario, config, specie, gasbox)`
through every expression, so the vectorisation work was already done. `fairjax`
actually goes the other way: the core runs *one* scenario and *one* config over the
species axis only, and `vmap` rebuilds the ensemble. That is both faster to read and
easier to get right.

**Small.** The numerical core is ~700 lines; the other ~1400 lines of `fair.py` are
xarray and pandas plumbing that never needed to be differentiable. `fairjax` keeps
FaIR's front end — species properties, RCMIP loading, unit handling, the AR6
calibration files — and lifts only the maths. `fairjax/params.py` is that bridge.

### The parts that actually bit

1. **NaN as a semantic mask.** FaIR marks inapplicable species properties with NaN and
   `np.nansum`s over the species axis. That is fine forward and fatal backward:
   `0 * nan == nan`, so a single NaN anywhere returns an all-NaN Jacobian. Every array
   is sanitised to a finite value in `params.py` and species are removed by
   multiplicative masks instead. **This was most of the debugging.**
2. **Fancy-index assignment.** `forcing/ghg.py` selects three CO₂ regimes with
   `.nonzero()` and assigns into slices. Became nested `jnp.where`. The regimes are
   disjoint so ordering does not matter, but they meet continuously and not smoothly —
   gradients are kinked at the regime boundaries.
3. **Overflow that FaIR repairs after the fact.** CF₄'s `g0` underflows to exactly zero
   while its exponent overflows, giving `0 * inf = nan`; FaIR patches the NaN
   afterwards. We cannot, so the *input* to the unsafe branch is sanitised as well as
   its output — the "double where" (`physics.calculate_alpha`).
4. **`float64` is mandatory.** Enabled on import. In float32 a 350-year carbon cycle
   drifts far enough to swamp the comparison against FaIR.
5. **An ordering artefact.** FaIR computes ERFari and ERFaci *before* it writes EESC in
   the same timestep, so the EESC slot is still NaN and silently drops out of their
   `nansum`s — while ozone, computed afterwards, does see it. Reproducing FaIR meant
   reproducing that. It is documented at the point where it matters in `core.py`.
6. **Live state masquerading as a field.** `gas_partitions` has no time axis; `run()`
   overwrites it. Reading initial conditions after a run silently gives you the final
   carbon-cycle state.
7. **`eig` on a non-symmetric matrix.** FaIR derives ECS and TCR by eigendecomposing
   the energy balance matrix. JAX's `eig` is CPU-only and its gradient is
   ill-conditioned with near-degenerate eigenvalues. Both have equivalent definitions
   that need only arithmetic: ECS is `F_2x / kappa_1` exactly, and TCR is a 70-year
   ramp integration. See `diagnostics.py` — they agree with FaIR to 1e-12 and 1e-6.

### Scope

Implemented: forward (emissions-driven) gas cycle with the state-dependent `alpha`
feedback, Thornhill methane lifetime, Meinshausen 2020 GHG forcing, ERFari, ERFaci,
EESC, ozone, contrails, LAPSI, stratospheric water vapour, land use, the
temperature-forcing feedback, the Cummins energy balance model, TOA imbalance, ocean
heat content. All 64 default species.

Not implemented: concentration-driven (inverse) mode, stochastic internal variability,
and the `leach2021` / `etminan2016` / `myhre1998` forcing schemes. None are needed for
the science above; each is 20–60 lines against an existing failing test if wanted.

## Layout

```
fairjax/
  core.py         the time loop as a lax.scan, plus Params / SpeciesMap / Output
  physics.py      gas cycle and forcing formulae, species axis only
  ebm.py          Cummins n-layer energy balance model
  diagnostics.py  ECS and TCR without an eigendecomposition
  params.py       the bridge from a configured fair.FAIR object
  reference.py    a stock FaIR configuration, for validation and notebooks
tests/
  test_against_fair.py   agreement with stock FaIR at rtol 1e-10
  test_gradients.py      finiteness, finite-difference agreement, Hessian symmetry
scripts/
  identifiability_demo.py
```

`tests/test_against_fair.py` is the contract. If it fails, the port is wrong, not FaIR.

# Differentiable coupled model

[Notebook 03](notebooks/03_coupled_model.ipynb) uses JAX-GCM v2.0.1 (SPEEDY)
to change soil boundary conditions and atmospheric initial conditions, then
compute forward (JVP) and reverse (VJP) sensitivities. Runs last seven days;
sensitivities use six hours. Soil temperature and wetness are prescribed.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once
(Linux/macOS):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

From the repository root:

```bash
cd 2026_hackathon
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

JupyterLab is included. Open the notebook using this environment in your editor
or Jupyter. The first run compiles; clear outputs before saving.
