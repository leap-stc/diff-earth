# JAX gotchas cheat sheet

A quick-reference list of the ways JAX will surprise you when you arrive from NumPy or Fortran. Chapter 1.1 walks through most of these with examples; this page is the condensed version to keep open while you work. See also the official [JAX sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) page.

| Gotcha | Symptom | Fix | Shown in |
|---|---|---|---|
| Arrays are immutable | `TypeError` on `x[i] = v` | `x = x.at[i].set(v)` | 1.1 |
| float32 by default | Gradients disagree with finite differences; drifting budgets | `jax.config.update("jax_enable_x64", True)` before creating arrays | 3.5 |
| Python `if` on traced values | `TracerBoolConversionError` under `jit` | `jnp.where`, `lax.cond`, `lax.select` | 1.1 |
| Python `for` loops under `jit` | Minutes-long compile times | `lax.scan` (time stepping), `lax.fori_loop` | 1.1 |
| Side effects in jitted functions | `print` fires once (at trace time), not per call | `jax.debug.print`; keep functions pure | 1.1, 3.5 |
| Implicit global RNG | No `np.random.seed` equivalent | Explicit keys: `jax.random.key`, `jax.random.split` | 1.1 |
| Recompilation | First call after a shape/dtype change is slow | Keep shapes static; pad instead of resizing | 1.1 |
| `nan` gradients from safe-looking code | `jnp.where` with a non-differentiable untaken branch | The "double-`where`" idiom; `jax_debug_nans` | 3.3, 3.5 |
| Thresholds/switches in physics | Loss varies but gradient is exactly zero | Smooth, reformulate, or `custom_vjp` | 3.3 |
| Long-rollout `grad` runs out of memory | Adjoint memory grows with steps | `jax.checkpoint`, nested-scan checkpointing | 3.2 |
| Integer/weak-type promotion | Silent dtype surprises in mixed expressions | Be explicit with dtypes in model constants | — |

*(Starter list — group members: add gotchas as you hit them, one row each.)*
