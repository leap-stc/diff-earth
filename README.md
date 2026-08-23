# Differentiable Earth system science research group
Apply differentiable Earth system model components as instruments for gradient-based science. 

### What differentiability makes possible:

*Learn from data at scale*: learn parameters, structures, and ML closures 

*Embed machine learning in physics*: train learned components while coupled 

*Ask scientific questions in new ways*: sensitivity, attribution, causality

## Tutorial

[`tutorial/`](tutorial/) contains **Differentiable Earth System Modeling with JAX**, a Jupyter Book for scientists new to differentiable code — from your first `jax.grad` on a radiative-balance formula, through adjoint sensitivity maps, 4D-Var, and online-trained ML closures on a 1-D energy balance model, to the hard parts: gradients through chaos, checkpointing, non-smooth physics, and stiff-solver adjoints. All chapters are executed notebooks that run on a laptop.

Read it online at <https://leap-stc.github.io/diff-earth/> (published by the `deploy-book` workflow; enable **Settings → Pages → Source: GitHub Actions** once). To build it locally:

```bash
conda env create -f tutorial/environment.yml
conda activate diff-earth-tutorial
jupyter-book build tutorial/
```

Inspired by the M2LInES [L96 demo book](https://m2lines.github.io/L96_demo/intro.html).

