"""Pre-stage CESM2 global-mean targets for the hackathon.

Run this BEFORE the session and commit the result. Ten people simultaneously
pulling CMIP6 zarr from cloud storage is the most reliable way to lose the first
hour of a half-day hackathon.

Pulls from the Pangeo CMIP6 catalog on Google Cloud (public, no credentials),
reduces each field to an area-weighted global annual mean, and writes a single
small netCDF to data/cesm2_targets.nc.

    python scripts/stage_cesm.py

Requires: gcsfs, zarr, intake-esm (not core fairjax dependencies).
"""

import pathlib

import numpy as np
import xarray as xr

CATALOG = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
SOURCE = "CESM2"
OUTPUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "cesm2_targets.nc"

# experiment -> variables. tas gives GSAT; the radiation terms give the TOA
# imbalance N = rsdt - rsut - rlut, which constrains ocean heat uptake.
REQUESTS = {
    # piControl needs the radiation terms too: a Gregory regression uses TOA
    # flux *anomalies*, and CESM2's control has a non-zero residual imbalance.
    "piControl": ["tas", "rsdt", "rsut", "rlut"],
    "abrupt-4xCO2": ["tas", "rsdt", "rsut", "rlut"],
    "1pctCO2": ["tas"],
    "historical": ["tas", "rsdt", "rsut", "rlut"],
    "ssp245": ["tas"],
    "ssp585": ["tas"],
}


# Experiments that get spliced together must come from the SAME realisation --
# historical and its SSP extension are one continuous simulation, and mixing
# members would splice two different draws of internal variability.
MEMBER_GROUPS = [
    ["historical", "ssp245", "ssp585"],
    ["piControl", "abrupt-4xCO2", "1pctCO2"],
]


def member_order(member):
    """Sort r1 < r2 < ... < r10 rather than lexicographically ('r10' < 'r1i')."""
    digits = "".join(c if c.isdigit() else " " for c in member).split()
    return tuple(int(d) for d in digits)


def choose_members(catalog):
    """Pick one member per experiment, shared within each splice group."""
    available = {}
    for experiment in REQUESTS:
        found = catalog.search(source_id=SOURCE, experiment_id=experiment,
                               variable_id="tas", table_id="Amon")
        available[experiment] = set(found.df.member_id.unique())

    chosen = {}
    for group in MEMBER_GROUPS:
        present = [e for e in group if available.get(e)]
        if not present:
            continue
        common = set.intersection(*(available[e] for e in present))
        if not common:
            raise SystemExit(f"no member common to {present}; cannot splice them")
        pick = sorted(common, key=member_order)[0]
        for experiment in present:
            chosen[experiment] = pick
    return chosen


def global_annual_mean(data):
    """Area-weighted global mean, then annual mean, as a plain 1-D series."""
    weights = np.cos(np.deg2rad(data.lat))
    spatial = data.weighted(weights).mean(dim=("lat", "lon"))
    return spatial.groupby("time.year").mean("time")


def main():
    import intake

    catalog = intake.open_esm_datastore(CATALOG)
    results = {}
    chosen = choose_members(catalog)
    print("members:", ", ".join(f"{k}={v}" for k, v in sorted(chosen.items())), "\n")

    for experiment, variables in REQUESTS.items():
        for variable in variables:
            query = catalog.search(
                source_id=SOURCE,
                experiment_id=experiment,
                variable_id=variable,
                table_id="Amon",
            )
            if len(query.df) == 0:
                print(f"  ! no data for {experiment}/{variable}")
                continue
            member = chosen[experiment]
            row = query.df[query.df.member_id == member]
            if len(row) == 0:
                print(f"  ! {experiment}/{variable} missing member {member}")
                continue
            store = row.zstore.iloc[0]
            dataset = xr.open_zarr(store, storage_options={"token": "anon"},
                                   consolidated=True)
            series = global_annual_mean(dataset[variable]).squeeze(drop=True).compute()
            results[f"{experiment}_{variable}"] = series.rename(f"{experiment}_{variable}")
            print(f"  {experiment:14s} {variable:5s} {member:10s} "
                  f"{int(series.year[0])}-{int(series.year[-1])}  "
                  f"mean {float(series.mean()):.3f}")

    if not results:
        raise SystemExit("nothing retrieved")

    # One shared year axis PER EXPERIMENT, not per variable. Giving each
    # variable its own year dim means `rsdt - rsut` silently broadcasts to 2-D
    # instead of differencing, which is a trap nobody should have to find.
    merged = xr.Dataset({
        name: series.rename({"year": f"year_{name.split('_')[0]}"})
        for name, series in results.items()
    })
    merged.attrs["source"] = f"{SOURCE} via {CATALOG}"
    merged.attrs["members"] = "; ".join(f"{k}={v}" for k, v in sorted(chosen.items()))
    merged.attrs["note"] = (
        "Area-weighted global annual means. tas in K (absolute, not anomaly): "
        "take anomalies against the relevant baseline before fitting."
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    merged.to_netcdf(OUTPUT)
    print(f"\nwrote {OUTPUT} ({OUTPUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
