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
Every notebook enables double precision in its first cell:

    jax.config.update("jax_enable_x64", True)

JAX defaults to single precision, which Chapter 3.5 shows is not enough to even
*verify* a gradient through a long rollout. Keep this line at the top of your own
notebooks too — it must run before any arrays are created.
```

## Running in Colab

Every notebook page has a launch button that opens it in Google Colab. JAX is preinstalled there; add one cell at the top to install the few extras used in Parts 2 and 3:

```
%pip install -q optax equinox diffrax
```

## Building the book

The notebooks are committed **with their executed outputs**, so building the website does not re-run them:

```bash
conda activate diff-earth-tutorial
jupyter-book build tutorial/
open tutorial/_build/html/index.html
```

If you edit a notebook, re-run it top to bottom (`Run > Run All Cells`) before committing, so the published outputs match the code. The GitHub Actions workflow in `.github/workflows/deploy-book.yml` rebuilds and publishes the site on every push to `main`; enabling it requires setting the repository's **Settings → Pages → Source** to "GitHub Actions" once.
