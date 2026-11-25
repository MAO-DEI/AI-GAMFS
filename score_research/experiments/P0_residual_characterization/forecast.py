"""Run original TorchScript weights with explicit paths and reusable prefixes."""
from collections import OrderedDict
from contextlib import contextmanager
import json
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from data_io import DIMS, iso, write_nc


@contextmanager
def project_directory(root):
    # Original stand / anti_stand open ./merra_norm.csv.
    previous = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def preflight(inputs, report):
    report["status"] = "checking"
    report["initializations"] = []
    report["channels"] = [{"name": k, "unit": inputs.model_units[k]} for k in inputs.names]
    report["targets"] = [
        {"alias": k, **mapping, "channel": i, "unit": unit}
        for (k, mapping), i, unit in zip(inputs.mapping.items(), inputs.positions, inputs.units)
    ]
    for when in inputs.initial_times:
        raw, lat, lon = inputs.initial(when)
        entry = {"init_time": iso(when), "shape": list(raw.shape), "dtype": str(raw.dtype),
                 "latitude": lat.tolist(), "longitude": lon.tolist(),
                 "nonfinite": 0, "already_normalized": False,
                 "input_files": {k: str(p) for k, p in inputs.input_files[when].items()},
                 "references": []}
        report["initializations"].append(entry)
        for lead in sorted({sum(path) for path in inputs.plans}):
            valid = when + pd.Timedelta(hours=lead)
            field, source = inputs.reference(valid, lat, lon)
            entry["references"].append({
                "time": iso(valid), "file": str(source), "shape": list(field.shape),
                "invalid_by_variable": (~np.isfinite(field)).sum(axis=(1, 2)).tolist(),
            })
    report["weights"] = {str(step): str(inputs.weight(step))
                         for step in sorted({n for p in inputs.plans for n in p})}
    for path in report["weights"].values():
        if not (inputs.root / path).is_file():
            raise FileNotFoundError(f"Missing TorchScript model: {path}")
    report["status"] = "passed"


def generate(inputs, run, fingerprint, provenance, reset_cache):
    import torch
    from utils.handle import anti_stand, stand

    settings = inputs.config["forecast"]
    precision = settings["initial_precision"]
    if precision not in {"fp16_roundtrip", "fp32_direct"}:
        raise ValueError(f"Unsupported initial_precision {precision}")
    capacity = settings["resident_models"]
    if type(capacity) is not int or capacity < 1:
        raise ValueError("resident_models must be a positive integer")
    torch.manual_seed(settings["seed"])
    device = torch.device(settings["device"])
    models = OrderedDict()
    events = []

    def predict(step, normalized, stamp):
        if step not in models:
            if len(models) == capacity:
                models.popitem(last=False)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            logging.info("Loading %s", inputs.weight(step))
            models[step] = torch.jit.load(str(inputs.weight(step)), map_location=device).eval()
        models.move_to_end(step)
        tag = [stamp.hour / 24, stamp.dayofweek / 7, stamp.dayofyear / 365.25]
        x = torch.from_numpy(normalized).unsqueeze(0).to(device)
        tag_tensor = torch.tensor([tag], dtype=torch.float32, device=device)
        with torch.no_grad():
            y = models[step](x, tag_tensor)
            if not isinstance(y, torch.Tensor) or y.shape != x.shape or y.dtype != torch.float32:
                raise ValueError(f"{step}h model returned unexpected type/shape/dtype")
            return y[0].detach().cpu().numpy().copy()

    forecasts, initials, initial_indices, valid, labels, primary, comparison, leads = (
        [], [], [], [], [], [], [], []
    )
    template = None
    for init_index, stamp in enumerate(inputs.initial_times):
        raw, lat, lon = inputs.initial(stamp)
        if template is None:
            template = lat, lon
        elif not all(np.array_equal(a, b) for a, b in zip(template, (lat, lon), strict=True)):
            raise ValueError("All initializations in a run must share a grid")
        initials.append(raw[inputs.positions])
        normalized = stand(inputs.names, raw.copy())
        if precision == "fp16_roundtrip":
            normalized = normalized.astype(np.float16).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError(f"Nonfinite normalized initial state at {iso(stamp)}")
        prefixes = {(): normalized}
        folder = run / "cache" / fingerprint / stamp.strftime("%Y%m%d_%H%M%S")
        if settings["cache_prefixes"]:
            folder.mkdir(parents=True, exist_ok=True)

        scheduled = sorted(inputs.plans.items())
        for path_index, (path, flags) in enumerate(scheduled):
            for length in range(1, len(path) + 1):
                prefix = path[:length]
                label = "+".join(map(str, prefix))
                source = stamp + pd.Timedelta(hours=sum(prefix[:-1]))
                cache = folder / (label + ".npy")
                if prefix in prefixes:
                    mode = "memory"
                elif settings["cache_prefixes"] and cache.is_file() and not reset_cache:
                    mode = "disk"
                    prefixes[prefix] = np.load(cache, allow_pickle=False)
                else:
                    mode = "forward"
                    logging.info("%s %s from %s", iso(stamp), label, iso(source))
                    prefixes[prefix] = predict(prefix[-1], prefixes[prefix[:-1]], source)
                result = prefixes[prefix]
                if result.shape != normalized.shape or result.dtype != np.float32 or not np.isfinite(result).all():
                    raise ValueError(f"Nonfinite or malformed normalized state {iso(stamp)} {label}")
                if mode == "forward" and settings["cache_prefixes"]:
                    temporary = cache.with_suffix(".tmp")
                    with temporary.open("wb") as file:
                        np.save(file, result, allow_pickle=False)
                    temporary.replace(cache)
                events.append({"init_time": iso(stamp), "path": "+".join(map(str, path)),
                               "prefix": label, "model_hours": prefix[-1], "step": length,
                               "source_time": iso(source), "target_time": iso(source + pd.Timedelta(hours=prefix[-1])),
                               "mode": mode, "time_tag": [source.hour / 24, source.dayofweek / 7,
                                                           source.dayofyear / 365.25]})
            physical = anti_stand(inputs.channels, prefixes[path][inputs.positions].copy())
            if not np.isfinite(physical).all():
                raise ValueError(f"Nonfinite physical forecast at {iso(stamp)} path {path}")
            forecasts.append(physical)
            initial_indices.append(init_index)
            leads.append(sum(path))
            valid.append(stamp + pd.Timedelta(hours=leads[-1]))
            labels.append("+".join(map(str, path)))
            primary.append(flags[0])
            comparison.append(flags[1])
            upcoming = scheduled[path_index + 1][0] if path_index + 1 < len(scheduled) else ()
            prefixes = {p: value for p, value in prefixes.items() if upcoming[:len(p)] == p}

    ds = xr.Dataset(
        {"forecast": (("record", *DIMS), np.stack(forecasts)),
         "initial_state": (("initialization", *DIMS), np.stack(initials))},
        coords={
            "initial_index": ("record", initial_indices),
            "init_time": ("initialization", pd.DatetimeIndex(inputs.initial_times).to_numpy()),
            "valid_time": ("record", pd.DatetimeIndex(valid).to_numpy()),
            "lead_time": ("record", leads), "path": ("record", labels),
            "primary": ("record", np.asarray(primary, dtype="u1")),
            "comparison": ("record", np.asarray(comparison, dtype="u1")),
            "variable": list(inputs.mapping), "latitude": template[0], "longitude": template[1],
            "unit": ("variable", inputs.units),
            "model_variable": ("variable", inputs.channels),
            "reference_variable": ("variable", [v["reference"] for v in inputs.mapping.values()]),
        },
        attrs={"provenance": provenance, "time_zone": "UTC",
               "state_transfer": "full normalized model output without clipping or quantization",
               "initial_precision": precision, "forecast_fingerprint": fingerprint},
    )
    ds.latitude.attrs["units"] = "degrees_north"
    ds.longitude.attrs["units"] = "degrees_east"
    ds.lead_time.attrs["units"] = "h"
    for field in ("forecast", "initial_state"):
        ds[field].attrs["units"] = "see unit coordinate by variable"
    write_nc(ds, run / "data" / "forecasts.nc")
    (run / "model_calls.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
