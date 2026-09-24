"""Checks on the staged CESM2 targets.

These guard the data file, not the model. Each one corresponds to a mistake that
was actually made while staging it, so they are regression tests in the literal
sense.
"""

import numpy as np
import pytest

from fairjax import cesm

pytestmark = pytest.mark.skipif(
    not cesm.DEFAULT_PATH.exists(),
    reason="run scripts/stage_cesm.py to create data/cesm2_targets.nc",
)


@pytest.fixture(scope="module")
def data():
    return cesm.load()


def test_variables_within_an_experiment_share_a_year_axis(data):
    """Otherwise `rsdt - rsut` broadcasts to 2-D instead of differencing."""
    for experiment in ["piControl", "abrupt-4xCO2", "historical"]:
        difference = (
            data[f"{experiment}_rsdt"]
            - data[f"{experiment}_rsut"]
            - data[f"{experiment}_rlut"]
        )
        assert difference.ndim == 1, f"{experiment} fluxes broadcast to {difference.shape}"


def test_spliced_experiments_share_a_realisation(data):
    """historical and its SSP extension are one continuous simulation."""
    members = dict(part.split("=") for part in data.attrs["members"].split("; "))
    assert members["historical"] == members["ssp245"] == members["ssp585"]


def test_gsat_splice_is_continuous(data):
    years, anomaly = cesm.gsat_anomaly(data, scenario="ssp245")
    assert np.all(np.diff(years) == 1)
    assert years[0] == 1850 and years[-1] == 2100
    # CESM2 is a warm model but not an absurd one
    assert 0.8 < anomaly[years == 2014][0] < 1.5
    assert 2.5 < anomaly[years == 2100][0] < 5.0


def test_gregory_regression_recovers_published_cesm2_sensitivity(data):
    """Drift correction is the difference between ECS 5.7 K and ECS 5.2 K."""
    control_tas = data["piControl_tas"].values
    control_net = (
        data["piControl_rsdt"] - data["piControl_rsut"] - data["piControl_rlut"]
    ).values
    warming = data["abrupt-4xCO2_tas"].values[:150] - control_tas.mean()
    net = (
        data["abrupt-4xCO2_rsdt"] - data["abrupt-4xCO2_rsut"] - data["abrupt-4xCO2_rlut"]
    ).values[:150] - control_net.mean()

    slope, intercept = np.polyfit(warming, net, 1)
    feedback, ecs = -slope, intercept / (-slope) / 2
    assert 0.60 < feedback < 0.70, f"lambda = {feedback:.3f}"
    assert 5.0 < ecs < 5.4, f"ECS = {ecs:.2f} K, published CESM2 is 5.15-5.3 K"


def test_toa_imbalance_refuses_to_guess(data):
    years, net = cesm.toa_imbalance(data, "historical")
    # the observed estimate for the 2000s is about +0.7 W m-2
    assert 0.4 < net[years >= 2000].mean() < 1.0
