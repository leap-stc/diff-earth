# 0.1 The adjoint for free

```{admonition} Outline
:class: tip
- Recall what tangent linear and adjoint models are used for 
- Explain what automatic differentiation is 
- Map forward-mode differentiation to the tangent linear model and reverse mode to the adjoint model
- State what automatic differentiation does *not* solve (the subject of Part 3)
```

## Before automatic differentiation

Many questions in Earth system science are questions about derivatives, whether or not they are phrased that way. *How much does a forecast change if the initial state is nudged here?* *Which observations would most improve the analysis?* *How sensitive is process A to process B?* Each asks for the derivative of a model output with respect to model inputs.

Answering such questions typically required strategically designed ensembles of simulations, or alternatively required building two companion models by hand. The **tangent linear model** (TLM) propagates a small perturbation forward through the dynamics: given a nudge to the initial state, it returns the resulting nudge to the forecast. The **adjoint model** runs the same linearized dynamics in reverse: given a target quantity (e.g., a forecast error, a regional average) it returns that quantity's sensitivity to *every* input at once. The adjoint is what made variational data assimilation (4D-Var) operational at forecast centers in the 1990s {cite}`talagrand1987variational`, powered decades of sensitivity studies {cite}`errico1997adjoint`, and underlies ocean state estimation efforts such as ECCO.

An adjoint model is not a small modification of the forward model; it is a second model, line by line, that must be updated every time the forward model changes. Source-to-source transformation tools (TAF, Tapenade, OpenAD) automated parts of the work for Fortran codes, but building and maintaining an adjoint remained a specialist, multi-year undertaking. That is why, of the Earth system models in a CMIP ensemble, none have a complete full complexity operational adjoint.

## Automatic differentiation 

**Automatic differentiation** (AD, or "autodiff") computes exact derivatives of a program by applying the chain rule mechanically to every elementary operation the program performs {cite}`baydin2018automatic,griewank2008evaluating`. Every model, however complex, is ultimately a composition of additions, multiplications, and elementary functions, each with a known derivative; AD composes those known derivatives in the order dictated by the code.

AD is not *symbolic* differentiation: no giant algebraic expression is ever built, the derivative program has the same structure and roughly the same cost as the original. And AD is not *numerical* (finite-difference) differentiation: no step size is chosen, so there is no truncation or cancellation error — the result is exact to floating-point precision. 

## Forward mode is the TLM; reverse mode is the adjoint

AD comes in two modes, and they are precisely the two companion models the community used to build by hand:

| | Forward mode (`jax.jvp`) | Reverse mode (`jax.grad`, `jax.vjp`) |
|---|---|---|
| Climate-community name | tangent linear model | adjoint model |
| Propagates | one input perturbation, forward | one output sensitivity, backward |
| Cost per pass | ~1 model run | ~a few model runs |
| Yields per pass | effect on all outputs | sensitivity to **all inputs** |
| Cheap when | few inputs matter | few outputs matter (a scalar diagnostic) |
| Memory | like the forward model | must store or recompute the trajectory |

A scalar diagnostic of a model with a million inputs — a mean temperature as a function of a forcing field, a bias as a function of neural-network weights — costs *one* reverse-mode pass instead of a million perturbation runs. In a differentiable programming framework such as JAX, the adjoint is generated mechanically from the model code, is exact for the model as written, and never falls out of date when the model changes.

## When adjoints are free

When every model you write carries its exact adjoint, categories of work change size. Sensitivity analysis no longer require ensembles (Chapter 2.1). Variational data assimilation fits in twenty lines (Chapter 2.3). Calibration scales from a handful of parameters to the thousands of weights of an embedded neural network, because gradient descent prices all parameters at a few model runs (Chapters 2.2, 2.4). Whole research designs that were previously uneconomical — training a subgrid closure *through* the host model against observable climate statistics — become routine mechanics.

## What does not change

The mathematics of the Earth system is not simplified by differentiating it. Chaotic dynamics make long-window derivatives exact but meaningless (Chapter 3.1). Reverse mode's memory cost grows with simulation length until it is managed deliberately (Chapter 3.2). Thresholds in physics such as convective triggers or freezing points have derivatives that are zero or undefined, and AD faithfully reports them (Chapter 3.3). And the derivative of a numerical solution has its own stability theory, with real failure modes (Chapter 3.4).

For broad surveys of differentiable programming in Earth system science see {cite}`gelbrecht2023differentiable` and {cite}`shen2023differentiable`.
