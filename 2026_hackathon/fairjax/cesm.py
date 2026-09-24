"""Load the pre-staged CESM2 global-mean targets.

The data file is produced by ``scripts/stage_cesm.py``, which must be run ahead
of time -- see the README. Nothing here touches the network.
"""

import pathlib

import numpy as np

__all__ = ["load", "gsat_anomaly", "toa_imbalance", "DEFAULT_PATH"]

DEFAULT_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "cesm2_targets.nc"


def load(path=None):
    """Open the staged CESM2 dataset, with a useful error if it is missing."""
    import xarray as xr

    path = pathlib.Path(path or DEFAULT_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/stage_cesm.py` first "
            "(needs gcsfs, zarr and intake-esm)."
        )
    return xr.open_dataset(path)


def _series(dataset, name):
    variable = dataset[name]
    return np.asarray(dataset[variable.dims[0]].values), np.asarray(variable.values)


def gsat_anomaly(dataset, scenario="ssp245", baseline=(1850, 1900)):
    """Historical + SSP global surface air temperature anomaly.

    Returns ``(years, anomaly_K)``. The two experiments are one continuous
    simulation from the same realisation -- ``stage_cesm.py`` enforces that --
    so they concatenate directly.
    """
    years_h, tas_h = _series(dataset, "historical_tas")
    years_s, tas_s = _series(dataset, f"{scenario}_tas")
    years = np.concatenate([years_h, years_s])
    tas = np.concatenate([tas_h, tas_s])
    if not np.all(np.diff(years) == 1):
        raise ValueError("historical and scenario years do not join up cleanly")
    window = (years >= baseline[0]) & (years <= baseline[1])
    return years, tas - tas[window].mean()


def toa_imbalance(dataset, experiment="historical", drift_correct=True):
    """Net downward TOA flux, ``rsdt - rsut - rlut``, in W m-2.

    With ``drift_correct``, the piControl mean imbalance is removed. CESM2's
    control is not in perfect balance, so this matters for anything that treats
    the flux as an anomaly -- a Gregory regression above all.
    """
    years, rsdt = _series(dataset, f"{experiment}_rsdt")
    _, rsut = _series(dataset, f"{experiment}_rsut")
    _, rlut = _series(dataset, f"{experiment}_rlut")
    net = rsdt - rsut - rlut
    if drift_correct:
        if "piControl_rsdt" not in dataset:
            raise KeyError(
                "piControl radiation is not in this file, so the imbalance "
                "cannot be drift corrected; re-run scripts/stage_cesm.py"
            )
        _, control_rsdt = _series(dataset, "piControl_rsdt")
        _, control_rsut = _series(dataset, "piControl_rsut")
        _, control_rlut = _series(dataset, "piControl_rlut")
        net = net - (control_rsdt - control_rsut - control_rlut).mean()
    return years, net
