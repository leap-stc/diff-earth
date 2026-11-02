# Differentiable Earth system science research group
Apply differentiable Earth system model components as instruments for gradient-based science. 

### What differentiability makes possible:

*Learn from data at scale*: learn parameters, structures, and ML closures 

*Embed machine learning in physics*: train learned components while coupled 

*Ask scientific questions in new ways*: sensitivity, attribution, causality

## Tutorial

[`tutorial/`](tutorial/) contains a tutorial Jupyter Book for **Differentiable Earth System Modeling with JAX**, designed for scientists new to differentiable code. It covers a brief introduction to JAX applying `jax.grad` on a simple radiative-balance formula. We demonstrate applications including adjoint sensitivity maps, data assimilation, and hybrid modeling with online-training of ML closures on a 1-D energy balance model. We also introduce the more challenging aspects of using these tools in practice. All chapters are executed notebooks that run on a laptop.

Read it online at <https://leap-stc.github.io/diff-earth/> (published by the `deploy-book` workflow; enable **Settings → Pages → Source: GitHub Actions** once).

Inspired by the M2LInES [L96 demo book](https://m2lines.github.io/L96_demo/intro.html).
