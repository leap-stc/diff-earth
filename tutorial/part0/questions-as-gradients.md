# 0.2 Science questions, reframed as gradients

```{admonition} Learning goals
:class: tip
- Recognize when a science question is secretly a gradient computation
- Map the group's three research pillars to concrete gradient expressions
- Know which chapter of this book demonstrates each question type
```

*(Prose chapter — no code. To be written; section skeleton below.)*

## A gallery

| Science question | Gradient expression | Chapter |
|---|---|---|
| Which forcing pattern most efficiently warms the Arctic? | ∂(Arctic temperature)/∂(forcing at every latitude) — one adjoint pass | 2.1 |
| What parameter values best match observations? | ∇ of a model–data misfit, fed to an optimizer | 2.2 |
| What initial state is consistent with these observations? | 4D-Var: ∇ of a cost function with respect to the initial state | 2.3 |
| What should the subgrid closure be? | ∇ of a rollout loss with respect to neural-network weights, *through the solver* | 2.4 |
| How sensitive is the climate (not the weather) of my model? | A time-averaged, ensemble-averaged gradient — and here be dragons | 3.1 |

## The three pillars, mapped to this book

- **Learn from data at scale** — parameters, structures, and ML closures: Chapters 2.2, 2.4
- **Embed machine learning in physics** — train learned components while coupled: Chapter 2.4, with the stability caveats of Part 3
- **Ask scientific questions in new ways** — sensitivity, attribution, causality: Chapters 2.1, 2.3, 3.1

## What gradients don't give you

A gradient is local. It will not find distant minima, it says nothing about structural error by itself, and in chaotic systems the naive gradient can be exactly wrong for the question you meant to ask {cite}`metz2021gradients`. Part 3 is about earning the right to trust these numbers.
