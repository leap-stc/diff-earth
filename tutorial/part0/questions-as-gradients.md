# 0.2 Science questions, reframed as gradients

A derivative is simply a response of one quantity to an infinitesimal change in another. The reach of differentiable modeling comes from how many scientific questions have exactly that shape when stated precisely. This chapter has a few examples — each row of the table below is a real question type, its gradient formulation, and the chapter where this book computes it on a working model.

## A gallery

| Science question | As a gradient | Chapter |
|---|---|---|
| Which forcing pattern most efficiently warms the Arctic? | ∂(Arctic mean T) / ∂(forcing at every latitude): one reverse-mode pass yields the whole map | 2.1 |
| What parameter values best match observations? | ∇ of a model–data misfit with respect to the parameters, driving an optimizer | 2.2 |
| What was the state of the system, given sparse observations? | ∇ of the 4D-Var cost function with respect to the initial state — the adjoint method | 2.3 |
| What should the subgrid closure be, given only observable outputs? | ∇ of a simulated-climate misfit with respect to neural-network weights, *through the solver* | 2.4 |
| How sensitive is the model's climate to a parameter? | a gradient of long-time statistics: exact autodiff fails and we must use modified estimators | 3.1 |
| How does the equilibrium respond to a parameter, without re-running the spin-up? | the implicit-function-theorem gradient at the fixed point | 3.2 |
| How close is the system to a tipping point? | the divergence of ∂(state)/∂(forcing) as stability is lost — with sharp limits  | 1.3 |

Two structural aspects: First, most scientific diagnostics are *scalars* (a mean, a misfit, an index), and reverse-mode differentiation prices the sensitivity of one scalar to *any number of inputs* at a few model runs. So "with respect to everything" is affordable by default. Second, the gradient of a composition is composable: physics, numerics, and neural networks differentiated together, jointly, through one another. That is what "training a closure through the host model" or "end-to-end" means mechanically.

## The focal areas to apply these techniques:

**Learn from data at scale.** Parameters, initial states, and learned components are all just inputs with gradients. Calibration (2.2), state estimation (2.3), and closure learning (2.4) are the same computation at increasing input dimension and gradient descent is the only known method whose cost does not grow with that dimension.

**Embed machine learning in physics.** A neural network inside a differentiable model is trainable *in place*, against what is actually observable, with the physics participating in every gradient (2.4). The alternative, training offline against diagnosed quantities and hoping the coupled system behaves, needs data that rarely exist and offers no guarantee the coupled feedbacks are right. Part 3 addresses associated challenges: rollout lengths are limited by chaos (3.1) and memory (3.2).

**Ask scientific questions in new ways.** An adjoint sensitivity map (2.1) answers "what matters most for this outcome?" at every input simultaneously, the natural mathematical object for attribution and mechanism questions. Growing gradients flag approaching instability (1.3). And equilibrium sensitivities come directly from the steady-state equations (3.2), no perturbation ensemble required.

## What gradients do not give you

A gradient is local, linear, and only as meaningful as the function it differentiates.

*Local*: it describes infinitesimal response at one point — it cannot see other basins of attraction, and near a tipping point it diverges without saying where the cliff is (Chapter 1.3 shows both properties in one table).

*Linear*: finite perturbations, regime changes, and state-dependence are outside its contract; it is the first term of a Taylor series, not a substitute for the response surface.

*As meaningful as the function*: if the quantity being differentiated is a finite-window statistic of a chaotic flow, the exact derivative reflects the window's particular weather, not the climate. Chapter 3.1 demonstrates that and {cite}`metz2021gradients` documents this challenge across fields.
