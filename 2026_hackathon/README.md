# 2026 annual meeting hackathon

## Differentiable coupled model

[Notebook 03](notebooks/03_coupled_model.ipynb) uses JAX-GCM v2.0.1 (SPEEDY)
to change soil boundary conditions and atmospheric initial conditions, then
compute forward (JVP) and reverse (VJP) sensitivities. Runs last seven days;
sensitivities use six hours. Soil temperature and wetness are prescribed.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once
(Linux/macOS):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

From the repository root:

```bash
cd 2026_hackathon
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

JupyterLab is included. Open the notebook using this environment in your editor
or Jupyter. The first run compiles; clear outputs before saving.
