# Differentiable Earth System Modeling with JAX

```{warning}
🚧 This tutorial is under construction. Chapter stubs below outline what each notebook will cover; content is being filled in by the group.
```

Climate models written in differentiable code let us ask scientific questions in new ways: every model output comes with its sensitivities to every input, parameter, and forcing — the adjoint that once took person-decades to hand-code, now generated mechanically. This tutorial teaches research scientists and PhD students in Earth system science how to build and — more importantly — how to *reason about* differentiable models in [JAX](https://github.com/jax-ml/jax).

## Who this is for

Earth system scientists with some machine learning experience. We assume you are comfortable with climate models, NumPy, and the basics of training a neural network. We do **not** assume prior experience with JAX, automatic differentiation, or differentiable programming.

## The arc of the book

The book follows one coherent model hierarchy, adding exactly one difficulty at a time:

1. **Part 0 — Why differentiable modeling?** No code. What a gradient buys you, and which science questions become tractable when your model is differentiable.
2. **Part 1 — JAX fundamentals.** From an algebraic radiative-equilibrium formula (your first climate gradient, no time stepping) to a time-stepped zero-dimensional energy balance model. Gradients here are perfectly behaved.
3. **Part 2 — Gradient-based science.** The Budyko–Sellers one-dimensional energy balance model: adjoint sensitivity maps, calibration as optimization, 4D-Var in ~20 lines, and a neural-network closure trained *online* through the solver. Still non-chaotic — the workflow stays clean.
4. **Part 3 — The hard parts.** Lorenz-96 enters as the chaos stress test, and everything that worked beautifully in Part 2 starts to break: exploding gradients, memory blowup, non-smooth physics, adjoint instability. Each chapter demonstrates the failure and at least one working fix.

## How to use this book

Each chapter is a self-contained Jupyter notebook, runnable in minutes on a laptop or in Colab (use the 🚀 launch button at the top of each notebook page). See the [setup appendix](appendix/setup.md) to build the environment locally.

## Acknowledgments

The structure of this book is inspired by the excellent [Learning Machine Learning with Lorenz-96](https://m2lines.github.io/L96_demo/intro.html) book from the M2LInES collaboration.
