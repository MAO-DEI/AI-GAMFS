"""Read the original products and construct verified forecast/reference pairs."""
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from utils.datasets import ALIASES, classify, geos_candidates, reference_path

STEPS = (12, 9, 6, 3)
GRID_SHAPE = (361, 576)
LAYERS = {"H", "QV", "RH", "T", "U", "V", "OMEGA"}
DIMS = ("variable", "latitude", "longitude")


def utc(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError(f"Expected explicit UTC timestamp: {value}")
    return stamp.tz_convert("UTC").tz_localize(None)


def iso(value):
    return pd.Timestamp(value).isoformat() + "Z"


def plans(config):
    forecast = config["forecast"]
    leads = forecast["lead_hours"]
    strategy = forecast["primary_strategy"]
    if strategy == "longest_first":
        primary = []
        for hours in leads:
            if type(hours) is not int or hours <= 0 or hours % 3:
                raise ValueError(f"Invalid lead: {hours}")
            steps = []
            for step in STEPS:
                count, hours = divmod(hours, step)
                steps.extend([step] * count)
            primary.append(tuple(steps))
    elif isinstance(strategy, dict):
        primary = [tuple(strategy[h] if h in strategy else strategy[str(h)]) for h in leads]
    else:
        raise ValueError("primary_strategy must be longest_first or a lead-to-path mapping")
    if not leads or len(set(leads)) != len(leads) or [sum(p) for p in primary] != leads:
        raise ValueError("Primary paths must match distinct requested leads")
    comparison = [tuple(p) for p in forecast["comparison_paths"]]
    if comparison and len({sum(p) for p in comparison}) != 1:
        raise ValueError("Comparison paths must share a lead")
    if len(set(comparison)) != len(comparison):
        raise ValueError("Duplicate comparison paths")
    result = {}
    for path in dict.fromkeys(primary + comparison):
        if not path or any(type(step) is not int or step not in STEPS for step in path):
            raise ValueError(f"Unsupported model path: {path}")
        result[path] = (path in primary, path in comparison)
    return result


def unit(value):
    key = str(value).strip().lower().replace(" ", "").replace("**", "").replace("^", "")
    return {"kgkg-1": "1", "dimensionless": "1"}.get(key, key)


def open_source(path):
    return xr.open_dataset(path, engine="netcdf4", decode_coords=False)


def array(field):
    values = np.array(field.values, dtype=np.float64, copy=True)
    for name in ("_FillValue", "missing_value", "fmissing_value"):
        sentinel = field.encoding.get(name, field.attrs.get(name))
        if sentinel is not None:
            scale = field.encoding.get("scale_factor", 1)
            offset = field.encoding.get("add_offset", 0)
            for value in np.atleast_1d(sentinel):
                decoded = np.asarray(value * scale + offset, dtype=field.dtype)
                values[values == decoded] = np.nan
    return values


def reorder(source, target, periodic=False):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if periodic:
        # Center the modular seam between model cells. A tiny negative value
        # at Greenwich must match zero without sorting to the opposite end.
        seam = -180.3125
        source, target = (source - seam) % 360, (target - seam) % 360
    if (source.ndim != 1 or target.ndim != 1 or source.size != target.size
            or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Coordinate shape or finiteness mismatch")
    order = np.argsort(source)
    wanted = np.argsort(target)
    if (np.any(np.diff(source[order]) <= 1e-6)
            or np.any(np.diff(target[wanted]) <= 1e-6)
            or not np.allclose(source[order], target[wanted], rtol=0, atol=1e-6)):
        raise ValueError("Coordinate values differ; this protocol requires exact grid matching")
    indices = np.empty_like(order)
    indices[wanted] = order
    return indices


def choose_time(ds, when, path):
    if "time" not in ds or ds["time"].dims != ("time",):
        raise ValueError(f"Missing decoded time axis: {path}")
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        raise ValueError(f"Expected decoded Gregorian times: {path}")
    indices = np.flatnonzero(ds.time.values == when.to_datetime64())
    if len(indices) != 1:
        raise ValueError(f"{path}: expected one {iso(when)}, found {len(indices)}")
    return ds.isel(time=int(indices[0]))


def resolve_reference_files(root, config):
    data = config["data"]
    directory = root / data["reference_dir"]
    if data.get("reference_files"):
        return [reference_path(directory, name) for name in data["reference_files"]]
    name = data.get("reference_product", "M2T1NXAER")
    product = ALIASES.get(name.casefold())
    if product is None or not product.merra_host:
        raise ValueError(f"Unknown MERRA-2 reference product: {name}")
    days = {(utc(t) + pd.Timedelta(hours=sum(path))).normalize().to_pydatetime()
            for t in config["experiment"]["init_times"] for path in plans(config)}
    found = {day: set() for day in days}
    for folder in (directory, directory / product.directory):
        for path in folder.glob("*.nc4"):
            item = classify(path.name)
            if item and item[0] == product and item[1] in found and not item[2]:
                found[item[1]].add(path.resolve())
    result = []
    for day, paths in sorted(found.items()):
        if len(paths) != 1:
            raise ValueError(f"{name} {day:%Y-%m-%d}: expected one reference file, found {sorted(paths)}")
        result.append(next(iter(paths)))
    return result


class Inputs:
    def __init__(self, root, config, include_initial=True):
        self.root = root
        self.config = config
        self.initial_times = [utc(x) for x in config["experiment"]["init_times"]]
        if not self.initial_times or len(set(self.initial_times)) != len(self.initial_times):
            raise ValueError("Experiment requires distinct init_times")
        self.plans = plans(config)
        self.mapping = config["variables"]
        if not self.mapping:
            raise ValueError("At least one variable is required")
        self.reference_files = resolve_reference_files(root, config)
        self.input_files = {}
        self.reference_times = {}
        with (root / "info.pkl").open("rb") as file:
            info = pickle.load(file)
        norm = pd.read_csv(root / "merra_norm.csv", index_col="index")
        self.names = list(norm.index)
        if (len(info) != 54 or len(norm) != 54 or not norm.index.is_unique
                or not np.isfinite(norm[["mean", "std"]]).to_numpy().all()
                or (norm["std"] <= 0).any()):
            raise ValueError("Invalid model metadata/normalization table")
        self.model_units = dict(zip(self.names, (x["units"] for x in info), strict=True))
        self.channels = [v["model"] for v in self.mapping.values()]
        self.units = [self.model_units[k] for k in self.channels]
        self.positions = [self.names.index(k) for k in self.channels]
        for path in self.reference_files:
            with open_source(path) as ds:
                if "time" not in ds or not np.issubdtype(ds.time.dtype, np.datetime64):
                    raise ValueError(f"Undecoded time in {path}")
                for when in pd.DatetimeIndex(ds.time.values):
                    self.reference_times.setdefault(when, []).append(path)
        self.groups = {}
        if include_initial:
            self._resolve_initial()

    def _resolve_initial(self):
        from utils.init import get_vars

        for group, variables in get_vars().items():
            product = "GEOS.fp.asm." + group
            if group == "tavg1_2d_aer_Nx":
                product = product.replace("tavg1", "tavg3")
            self.groups[product] = (
                [(name, level if name in LAYERS else None)
                 for name in variables["var"]
                 for level in (variables["level"] if name in LAYERS else [None])]
                if isinstance(variables, dict) else [(name, None) for name in variables]
            )
        names = [name if level is None else f"{name}_{level}"
                 for channels in self.groups.values() for name, level in channels]
        if names != self.names:
            raise ValueError("get_vars channel order differs from merra_norm.csv")
        directory = self.root / self.config["data"]["input_dir"]
        missing = []
        for when in self.initial_times:
            self.input_files[when] = {}
            key = when.strftime("%Y%m%d_%H%M")
            for product in self.groups:
                candidates = geos_candidates(directory, ALIASES[product.casefold()], when)
                if len(candidates) != 1:
                    missing.append(f"{product}.{key}: expected one file, found {candidates}")
                else:
                    self.input_files[when][product] = candidates[0]
        if missing:
            raise FileNotFoundError("\n".join(missing))

    def initial(self, when):
        fields = []
        lat = lon = None
        for product, channels in self.groups.items():
            path = self.input_files[when][product]
            with open_source(path) as ds:
                source = choose_time(ds, when, path).isel(lat=slice(None, None, 2), lon=slice(None, None, 2))
                if lat is None:
                    lat, lon = source.lat.values, source.lon.values
                    if (len(lat), len(lon)) != GRID_SHAPE:
                        raise ValueError(f"{path}: model grid must be {GRID_SHAPE}")
                    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
                        raise ValueError(f"{path}: original model grid must be ascending")
                    if not (np.allclose(lat, np.linspace(-90, 90, 361), rtol=0, atol=1e-6)
                            and np.allclose(lon, np.arange(576) * 0.625 - 180, rtol=0, atol=1e-6)):
                        raise ValueError(f"{path}: model grid differs from the original global grid")
                source = source.isel(lat=reorder(source.lat.values, lat),
                                     lon=reorder(source.lon.values, lon, periodic=True))
                for name, level in channels:
                    if name not in source:
                        raise ValueError(f"{path}: missing {name} for {product}")
                    field = source[name]
                    key = name
                    if level is not None:
                        field = field.sel(lev=level)
                        key = f"{name}_{level}"
                    if unit(field.attrs.get("units")) != unit(self.model_units[key]):
                        raise ValueError(f"{path}: {key} units {field.attrs.get('units')} vs {self.model_units[key]}")
                    values = array(field.transpose("lat", "lon")).astype(np.float32)
                    if not np.isfinite(values).all():
                        raise ValueError(f"{path}: {key} has missing/nonfinite input values")
                    fields.append(values)
        return np.stack(fields), lat, lon

    def reference(self, when, lat, lon):
        matches = self.reference_times.get(when, [])
        if len(matches) != 1:
            raise ValueError(f"Reference {iso(when)}: expected one source, found {matches}")
        path = matches[0]
        with open_source(path) as ds:
            ds = choose_time(ds, when, path)
            ds = ds.isel(lat=reorder(ds.lat.values, lat), lon=reorder(ds.lon.values, lon, periodic=True))
            result = []
            for alias, expected in zip(self.mapping, self.units, strict=True):
                name = self.mapping[alias]["reference"]
                if name not in ds or unit(ds[name].attrs.get("units")) != unit(expected):
                    raise ValueError(f"{path}: missing {name} or incompatible units (expected {expected})")
                values = array(ds[name].transpose("lat", "lon"))
                if not np.isfinite(values).any():
                    raise ValueError(f"{path}: {name} has no valid data")
                result.append(values)
        return np.stack(result), path

    def weight(self, step):
        return self.root / self.config["forecast"]["model_dir"] / f"gamfs_{step}h_traced.pt"


def packing(ds):
    return {name: {"zlib": True, "complevel": 2}
            for name, field in ds.data_vars.items() if field.ndim >= 3}


def write_nc(ds, path):
    temporary = path.with_suffix(".nc.tmp")
    ds.to_netcdf(temporary, engine="netcdf4", encoding=packing(ds))
    temporary.replace(path)


def build(config, root, run, provenance):
    inputs = Inputs(root, config, include_initial=False)
    output = run / "data" / "residuals.nc"
    with xr.open_dataset(run / "data" / "forecasts.nc", engine="netcdf4", decode_timedelta=False) as source:
        ds = source.load()
    forecast = ds.forecast.values.astype(np.float64)
    initial = ds.initial_state.values[ds.initial_index.values].astype(np.float64)
    reference = np.empty_like(forecast)
    files = [""] * ds.sizes["record"]
    mask = np.isfinite(forecast) & np.isfinite(initial)
    for index in np.unique(ds.initial_index.values):
        same_init = ds.initial_index.values == index
        for when in np.unique(ds.valid_time.values[same_init]):
            records = np.flatnonzero(same_init & (ds.valid_time.values == when))
            values, path = inputs.reference(pd.Timestamp(when), ds.latitude.values, ds.longitude.values)
            reference[records] = values
            for i in records:
                files[i] = str(path)
            mask[records] = (mask[records].all(axis=0) & np.isfinite(values))
    persistence = np.where(np.isfinite(initial), initial, np.nan)
    dims = ("record", *DIMS)
    for name, field in (
        ("reference", reference), ("residual", np.where(mask, reference - forecast, np.nan)),
        ("persistence_forecast", persistence),
        ("persistence_residual", np.where(mask, reference - persistence, np.nan)),
        ("valid_mask", mask.astype("u1")),
    ):
        ds[name] = (dims, field)
        ds[name].attrs["units"] = "1" if name == "valid_mask" else "see unit coordinate by variable"
    ds["reference_file"] = ("record", files)
    ds.attrs.update(residual_sign="reference - forecast; positive means model underestimation",
                    mask_rule="finite reference, initial and all forecasts at common init/valid time",
                    build_provenance=provenance)
    write_nc(ds, output)
    if ds.comparison.values.any():
        write_nc(ds.isel(record=np.flatnonzero(ds.comparison.values)), run / "data" / "path_comparison.nc")
