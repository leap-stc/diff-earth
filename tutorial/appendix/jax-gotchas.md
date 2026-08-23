# JAX gotchas cheat sheet

A quick-reference list of the ways JAX will surprise you when you arrive from NumPy or Fortran. Chapter 1.1 walks through most of these with examples; this page is the condensed version to keep open while you work. See also the official [JAX sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) page.

| Gotcha | Symptom | Fix |
|---|---|---|
| Arrays are immutable | `TypeError` on `x[i] = v` | `x = x.at[i].set(v)` |
| float32 by default | Gradients disagree with finite differences; drifting budgets | `jax.config.update("jax_enable_x64", True)` at import time |
| Python `if` on traced values | `TracerBoolConversionError` under `jit`/`grad` | `jnp.where`, `lax.cond`, `lax.select` |
| Python `for` loops under `jit` | Minutes-long compile times | `lax.scan` (time stepping), `lax.fori_loop` |
| Side effects in jitted functions | `print` fires once (at trace time), not per call | `jax.debug.print`; keep functions pure |
| Implicit global RNG | No `np.random.seed` equivalent | Explicit keys: `jax.random.key`, `jax.random.split` |
| Recompilation | First call after a shape/dtype change is slow | Keep shapes static; pad instead of resizing |
| `nan` gradients from safe-looking code | `grad(jnp.sqrt)(0.0)`, `jnp.where` with `nan` in the untaken branch | Clamp inputs; "double-`where`" trick; `jax.debug_nans` |
| Integer/weak-type promotion | Silent dtype surprises in mixed expressions | Be explicit with dtypes in model constants |

*(Starter list — group members: add gotchas as you hit them, one row each.)*
