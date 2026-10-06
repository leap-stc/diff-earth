# Differentiable Earth system modeling hackathon

LEAP annual meeting, 2026.

The [tutorial](https://leap-stc.github.io/diff-earth/) demonstrates gradient-based methods on small models where everything works: a few state variables, smooth physics, short runs, and gradients we can check by hand. However, Earth system models are chaotic, they have threshold physics, and their gradients are expensive to store.

The goal of this hackathon is to exercise the tutorial's methods on an intermediate complexity model. We will work through the core of the tutorial together then repeat some of its examples in [JAX-GCM](https://github.com/climate-analytics-lab/jax-gcm) (JCM), a differentiable implementation of the SPEEDY atmospheric general circulation model. SPEEDY has a full three-dimensional dynamical core and simplified physics, and simulates a week of weather in seconds on a laptop. It sits between the tutorial's toy models and a full GCM, which makes it a good place to see the problems from Part 3 of the tutorial start to emerge. 

For this hackathon we will

- work selected examples from the tutorial together. 
- repeat simple exercises with JCM.
- stress test JCM to explore the limitations of differentiable modeling, and identify solutions.
- identify how these methods may be useful for your own research.

## Initial set up

Install JAX and JCM. 

1. Clone this repository.
2. Follow the setup instructions in [`notebooks/SETUP.md`](notebooks/SETUP.md). They cover conda and uv, and how to select the kernel in VS Code or JupyterLab. The same environment runs every tutorial chapter.
3. Run [`notebooks/01_jcm_sensitivities.ipynb`](notebooks/01_jcm_sensitivities.ipynb) from top to bottom. If every cell finishes, you are ready. 

## Part 1: the tutorial

We will work through five of the chapters in the tutorial together. The tutorial notebooks are in [`tutorial/`](../tutorial/) and run in the hackathon environment; you can also read them [online](https://leap-stc.github.io/diff-earth/).

| Chapter | Key point |
|---|---|
| [1.2 First climate gradient](https://leap-stc.github.io/diff-earth/part1/first-climate-gradient.html) | `jax.grad`, `jvp`, and `vjp`; always compare with a finite difference |
| [1.3 Energy balance model](https://leap-stc.github.io/diff-earth/part1/energy-balance-model.html) | Differentiating through a time-stepping loop |
| [2.1 Sensitivity maps](https://leap-stc.github.io/diff-earth/part2/sensitivity-maps.html) | One adjoint pass gives the sensitivity of a scalar to every input |
| [2.2 Calibration as optimization](https://leap-stc.github.io/diff-earth/part2/calibration.html) | Calibrate simple model parameters with gradient descent |
| [3.1 Gradients through chaos](https://leap-stc.github.io/diff-earth/part3/gradients-through-chaos.html) | Long-window gradients in chaotic systems grow without bound, and how to ask better questions |
| [3.2 Memory and cost](https://leap-stc.github.io/diff-earth/part3/memory-and-cost.html) | Why reverse mode stores the trajectory and how its memory scales |

If you are new to JAX, skim [1.1 Coding in JAX](https://leap-stc.github.io/diff-earth/part1/thinking-in-jax.html) first.

## Part 2: Get JCM running and repeat

[`notebooks/01_jcm_sensitivities.ipynb`](notebooks/01_jcm_sensitivities.ipynb) runs JCM for seven days and then computes three kinds of sensitivity:

- the response of evaporation to drier soil, from two nonlinear runs and from a forward-mode derivative (JVP);
- the response of evaporation to a warmer initial atmosphere, from two nonlinear runs;
- the sensitivity of mean land evaporation to five surface-flux parameters, from one reverse-mode derivative (VJP).

## Part 3: Choose your own challenge

Choose one aspect of applying differentiable models either from the tutorial or from your own research. Set up an example with JCM, and asks whether the result from the simple models in the tutorial still holds. Start from a copy of `01_jcm_sensitivities.ipynb` and keep your changes in your copy.

The challenges are open ended, take this in whatever direction you like, but here are some ideas to get you started. 

### Challenge 1: Do you trust this gradient?

*Tutorial: [1.2](https://leap-stc.github.io/diff-earth/part1/first-climate-gradient.html), [3.5](https://leap-stc.github.io/diff-earth/part3/trusting-gradients.html). Start from section 2b. Recommended if you are new to JAX.*

Compare the JVP from section 2b with centered finite differences from two model runs at `soil_scale = 1 ± h`, for step sizes `h` from 1e-1 down to 1e-6. Plot the difference between the two estimates against `h`.

- For which step sizes do the two agree?
- Where does the finite difference break down, and why? JAX computes in single precision by default.
- Does the agreement depend on location, for example wet versus dry regions?

### Challenge 2: One adjoint pass, one map

*Tutorial: [2.1](https://leap-stc.github.io/diff-earth/part2/sensitivity-maps.html). Start from section 2b.*

Section 2b scales the soil moisture everywhere by one number. Replace that number with a field, one scaling factor per grid point. Define a scalar target, for example mean evaporation over one region after six hours, and compute its gradient with respect to the whole field in one reverse pass.

- Where does soil moisture matter for evaporation in your region, and is any of it non-local?
- Check one grid point against a JVP or a finite difference.
- How many forward runs would the same map cost with forward mode?

### Challenge 3: Gradients through chaos

*Tutorial: [3.1](https://leap-stc.github.io/diff-earth/part3/gradients-through-chaos.html). Start from sections 3a and 3b.*

Choose a scalar output, for example global mean near-surface temperature at the end of the run. Compute its gradient with respect to the initial temperature field for windows of 6 hours, 1 day, 2 days, and 4 days. Plot the norm of the gradient against window length.

- Does the gradient norm grow exponentially? What growth rate does it imply, and how does that compare with the predictability time of midlatitude weather?
- At what window does the gradient stop being useful for prediction?
- Does a parameter gradient (section 3b) behave the same way as an initial-condition gradient?

### Challenge 4: Memory and cost

*Tutorial: [3.2](https://leap-stc.github.io/diff-earth/part3/memory-and-cost.html). Start from section 3b.*

Time the VJP in section 3b and measure its memory for windows of 6 hours, 12 hours, 1 day, 2 days, and longer. For memory, `jax.jit(f).lower(*args).compile().memory_analysis().temp_size_in_bytes` reports the working memory the compiled function needs, and your operating system's activity monitor shows the peak.

- How do run time and memory scale with window length? How does that compare with the cost of the forward run alone?
- At what window does your laptop run out of memory?
- Rematerialization (`jax.checkpoint`) trades compute for memory. How much does it help here?

### Challenge 5: Non-smooth physics

*Tutorial: [3.3](https://leap-stc.github.io/diff-earth/part3/non-smooth-physics.html). Start from section 3b.*

SPEEDY's convection and large-scale condensation schemes switch on at relative humidity thresholds. Choose one of these parameters, for example `parameters.convection.rhbl` (the boundary-layer relative humidity threshold for convection, default 0.9) or `parameters.condensation.rhlsc`. Sweep it across a range of values, and plot both the objective J and its derivative from JAX.

- Is the derivative from JAX consistent with the slope of the curve?
- Do you find places where J changes but the derivative is zero, or where the derivative is noisy?
- Which of the fixes from Chapter 3.3 would apply to this scheme?

### Extension: Calibration

*Tutorial: [2.2](https://leap-stc.github.io/diff-earth/part2/calibration.html). Start from section 3b.*

Run a twin experiment: make synthetic observations with `chl` increased by 20%, then recover it from the default value by gradient descent on the misfit over six-hour windows. Does the recovered value match the truth? Is the loss landscape as well behaved as in the tutorial?

## Results and discussion

Share one finding or figure from your exploration and we can discuss as a group:

- Which lessons from the tutorial held in JCM, and which did not?
- What would it take to use these methods in your own research: in a full GCM, with longer windows, or with observations?

## Practical notes

- **Compilation.** The first call to a model run or derivative compiles it, which takes up to a minute. Later calls with the same shapes reuse the compiled code. Building a new model inside a function you differentiate is fine, but changing array shapes triggers a new compilation.
- **Start short.** Begin with six-hour windows and double them. Long reverse-mode windows can exhaust a laptop's memory.
- **Precision.** JAX uses single precision (float32) by default. Keep this in mind when you compare with finite differences.

## Contents

```
2026_hackathon/
├── README.md                       this file
├── environment.yml                 conda environment
├── requirements.txt                pinned packages (used by conda and uv)
└── notebooks/
    ├── SETUP.md                   environment setup instructions
    └── 01_jcm_sensitivities.ipynb  the JCM walkthrough
```
