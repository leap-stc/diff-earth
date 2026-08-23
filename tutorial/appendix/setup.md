# Setup

## Local installation

From the repository root:

```bash
conda env create -f tutorial/environment.yml
conda activate diff-earth-tutorial
jupyter lab
```

All notebooks run on CPU in minutes; no GPU is required anywhere in this book.

```{note}
Enable double precision at the top of every notebook — climate quantities and
gradient checks both suffer in float32, and JAX defaults to float32:

    import jax
    jax.config.update("jax_enable_x64", True)
```

## Running in Colab

Every notebook page has a 🚀 launch button that opens it in Google Colab. JAX is preinstalled there; the first cell of each notebook installs the few extras (`optax`, `equinox`, `diffrax`) when needed.

## Building the book

```bash
conda activate diff-earth-tutorial
jupyter-book build tutorial/
open tutorial/_build/html/index.html
```

While chapters are stubs, notebook execution is off in `_config.yml` (`execute_notebooks: "off"`). Once chapters have real content, switch it to `"cache"` so the published book always reflects executed output.
