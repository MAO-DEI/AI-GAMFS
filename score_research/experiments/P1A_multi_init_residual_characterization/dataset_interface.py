"""Read a selected field across completed single-initialization shards."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def load_fields(run, variable, lead_time, field_type="residual", path="primary",
                init_times=None, start=None, end=None):
    """Return an xarray Dataset with physical fields, masks and UTC provenance."""
    allowed = {"reference", "persistence_residual", "residual", "forecast",
               "initial_state", "persistence_forecast"}
    if field_type not in allowed:
        raise ValueError(f"Unsupported field_type: {field_type}")
    wanted = None if init_times is None else {pd.to_datetime(t, utc=True) for t in init_times}
    start = None if start is None else pd.to_datetime(start, utc=True)
    end = None if end is None else pd.to_datetime(end, utc=True)
    samples = []
    for folder in sorted((Path(run) / "per_init").glob("*")):
        status_file = folder / "status.json"
        residual_path = folder / "residuals.nc"
        if not status_file.is_file() or not residual_path.is_file():
            continue
        status = json.loads(status_file.read_text(encoding="utf-8"))
        if any(status.get(phase, {}).get("state") != "complete" for phase in ("build", "forecast")):
            continue
        with xr.open_dataset(residual_path, decode_timedelta=False) as ds:
            stamp = pd.to_datetime(ds.init_time.values, utc=True)
            if ((wanted is not None and stamp not in wanted) or
                    (start is not None and stamp < start) or (end is not None and stamp > end)):
                continue
            selection = (ds.primary.values == 1) if path == "primary" else (ds.path.values == path)
            records = np.flatnonzero(selection & (ds.lead_time.values == lead_time))
            if len(records) != 1:
                raise ValueError(f"Expected one record for {stamp}, got {len(records)}")
            record = ds.isel(record=int(records[0]))
            model_variable = str(ds.model_variable.sel(variable=variable).item()) if variable in ds.variable.values else variable
            if variable not in ds.variable.values and field_type not in {"forecast", "initial_state", "persistence_forecast"}:
                raise ValueError(f"No reference mapping for {variable}")
            if field_type in {"forecast", "initial_state", "persistence_forecast"}:
                with xr.open_dataset(folder / "data" / "forecasts.nc", decode_timedelta=False) as source:
                    field = (source.forecast.isel(record=int(records[0])) if field_type == "forecast"
                             else source.initial_state.isel(initialization=0))
                    field = field.sel(variable=model_variable, drop=True).load()
                    unit = str(source.unit.sel(variable=model_variable).item())
            else:
                field = (ds.reference.isel(reference_index=int(record.reference_index_for_record.item()))
                         if field_type == "reference" else record[field_type])
                field = field.sel(variable=variable, drop=True).load()
                unit = str(ds.unit.sel(variable=variable).item())
            mask = (record.valid_mask.sel(variable=variable, drop=True).astype(bool).load()
                    if variable in ds.variable.values else np.isfinite(field))
            data = xr.Dataset(
                {field_type: (("latitude", "longitude"), field.values, {"units": unit}),
                 "valid_mask": (("latitude", "longitude"), mask.values)},
                coords={"latitude": ds.latitude.values, "longitude": ds.longitude.values,
                        "valid_time": record.valid_time.values, "path": str(record.path.item()),
                        "reference_file": str(ds.reference_file.isel(
                            reference_index=int(record.reference_index_for_record.item())).item()),
                        "forecast_provenance": ds.attrs["forecast_provenance"],
                        "build_provenance": ds.attrs["build_provenance"]},
            )
            samples.append(data.expand_dims(init_time=[stamp.tz_localize(None).to_datetime64()]))
    if not samples:
        raise ValueError("No matching completed shard")
    combined = xr.concat(samples, dim="init_time", join="exact")
    if wanted is not None and {pd.to_datetime(t, utc=True) for t in combined.init_time.values} != wanted:
        raise ValueError("Some requested initializations have no residual shard")
    combined.attrs.update(time_zone="UTC", variable=variable, lead_hours=lead_time,
                          residual_sign="reference - forecast")
    return combined
