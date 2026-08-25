# Differentiable Earth System Modeling with JAX

Climate models written in differentiable code let us ask scientific questions in new ways: every model output comes with its sensitivities to every input, parameter, and forcing — the adjoint is generated mechanically. This is an introductory guide for researchers working in Earth system science on how to build, and do science with, differentiable models in [JAX](https://github.com/jax-ml/jax).

## Tutorial outline

1. **Part 0 — Why differentiable modeling?** No code. What a gradient buys you, and which science questions become tractable when your model is differentiable.
2. **Part 1 — JAX fundamentals.** From an algebraic radiative-equilibrium formula to a time-stepped zero-dimensional energy balance model with an ice–albedo tipping point. Gradients here are well behaved.
3. **Part 2 — Gradient-based science.** The Budyko–Sellers one-dimensional energy balance model: adjoint sensitivity maps, calibration / optimization, 4D-Var, and a neural-network closure trained *online*. 
4. **Part 3 — The hard parts.** Differentiating through chaos with Lorenz-96, what starts to break: exploding gradients, adjoint memory, non-smooth physics, unstable adjoints of stiff solvers. Each chapter demonstrates the failure *and* at least one working fix, ending with the verification habits that make gradient-based results trustworthy.

## How to use this book

Each chapter is a self-contained Jupyter notebook that runs in minutes on an ordinary laptop — no GPUs required. The notebooks are stored with their outputs, so the figures and numbers on these pages are real executed results; to run or modify them yourself, see the [setup appendix](appendix/setup.md) or use the launch button at the top of any notebook page to open it in Colab.
