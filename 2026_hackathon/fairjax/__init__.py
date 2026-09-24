"""fairjax: the FaIR simple climate model in JAX, end-to-end differentiable.

A port of FaIR v2.2.4 (https://github.com/OMS-NetZero/FAIR) that reproduces the
reference implementation to floating-point tolerance while supporting
``jax.grad``, ``jax.hessian``, ``jax.vmap`` and ``jax.jit``.

Typical use: configure a ``fair.FAIR`` object as normal, lift it, run it.

    >>> from fair import FAIR
    >>> import fairjax
    >>> f = FAIR()                      # ... configure as usual ...
    >>> params, smap, emis, forc, init = fairjax.from_fair(f)
    >>> out = fairjax.run(params[0], smap, emis[0], forc[0], init[0])

Float64 is enabled on import. FaIR integrates a carbon cycle over centuries;
in float32 the drift is large enough to swamp the comparison against the
reference implementation.
"""

import jax

jax.config.update("jax_enable_x64", True)

from .core import (  # noqa: E402
    InitialState,
    Output,
    Params,
    SpeciesMap,
    run,
    run_ensemble,
)
from .params import from_fair  # noqa: E402

__all__ = [
    "InitialState",
    "Output",
    "Params",
    "SpeciesMap",
    "run",
    "run_ensemble",
    "from_fair",
]
__version__ = "0.1.0"
