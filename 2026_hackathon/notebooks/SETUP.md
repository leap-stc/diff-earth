# Setup

One environment runs every hackathon notebook in this folder and every chapter of the tutorial. Set it up once, before the hackathon, with either conda or uv.

## Option 1: conda

From the repository root:

```bash
conda env create -f 2026_hackathon/environment.yml
conda activate diff-earth-hackathon
```

Then register the environment as a Jupyter kernel so it appears in the kernel dropdown:

```bash
python -m ipykernel install --user --name diff-earth-hackathon --display-name "Python (diff-earth-hackathon)"
```

## Option 2: uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once (Linux and macOS):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Then, from the repository root:

```bash
cd 2026_hackathon
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name diff-earth-hackathon --display-name "Python (diff-earth-hackathon)"
```

## Selecting the kernel

**In VS Code:** click **Select Kernel** (top right of the notebook) → **Jupyter Kernel...** → **Python (diff-earth-hackathon)**. If it is not listed, run **Developer: Reload Window** from the Command Palette.

**In JupyterLab:** with the environment active, run `jupyter lab` and choose **Kernel → Change Kernel... → Python (diff-earth-hackathon)**.


## install JCM
```bash
git clone https://github.com/climate-analytics-lab/jax-gcm.git
cd jax-gcm
git checkout 06fd4771bccf6f8df3a1f032ca0c6e4e7957cf2d
pip install -e .
```

## Check that it works

Run [`01_jcm_sensitivities.ipynb`](01_jcm_sensitivities.ipynb) from top to bottom. The first model run compiles, which takes about a minute; after that each run takes a few seconds. If every cell finishes, you are ready for the hackathon. Please clear the outputs before committing any changes.
