"""Inspect products, build residual shards and validate optional P0 imports."""
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from data_io import DIMS, Inputs, iso, plans, utc, write_nc
from forecast import project_directory
from run_experiment import atomic_text, fingerprint, identity, read_json, write_json
from utils.datasets import PRODUCTS, classify, geos_candidates, validate_file


# Reviewed changes from P0 commit 26e3045 to 42392ea. data_io changed file
# discovery and reference indexing only. handle changed the separate original
# CLI reader, while stand and anti_stand stayed identical.
COMPATIBLE_CODE = {
    "data_io.py": ("80b058a119ff85ac3a5731570d160ed728d0f1f200d19230a956de00c091f69b",
                   "3e54bbec2a5c04550815538463b3f810a7bbaf0be4949679cc95e11d59aa820b"),
    "handle.py": ("9e66fc05f3538df2f994e6c8ab5224414a8892779309385cef7aaa3335fda965",
                  "8e660fd96906a04b3ba99e8985fd358fce631a59f4afe34e13ab613bce9db4f9"),
}


def init_folder(run, stamp):
    return run / "per_init" / stamp.strftime("%Y%m%d_%H%M%S")


def init_times(config):
    experiment = config["experiment"]
    start, end = map(utc, (experiment["init_start"], experiment["init_end"]))
    interval = experiment["init_interval_hours"]
    if type(interval) is not int or interval < 1 or start > end:
        raise ValueError("Invalid initialization range")
    times = list(pd.date_range(start, end, freq=pd.Timedelta(hours=interval)))
    if times[-1] != end:
        raise ValueError("Initialization end is off cadence")
    return times


def single_config(config, stamp):
    return {key: config[key] for key in ("data", "forecast", "variables")} | {
        "experiment": {"name": config["experiment"]["name"], "init_times": [iso(stamp)]}}


def write_rows(path, rows):
    atomic_text(path, pd.DataFrame(rows).to_csv(index=False))


def inspect(root, run, config, times):
    manifest, dates = [], []
    inputs_root = root / config["data"]["input_dir"]
    refs_root = root / config["data"]["reference_dir"]
    norm = pd.read_csv(root / "merra_norm.csv", index_col="index")
    required_references = {v["reference"] for v in config["variables"].values()}
    requested = set()
    for stamp in times:
        requested.update((product.name, stamp) for product in PRODUCTS if not product.merra_host)
        requested.update((config["data"]["reference_product"], (stamp + pd.Timedelta(hours=sum(path))).normalize())
                         for path in plans(config))
    checked = {}

    def record(stamp, product, item_time, path, required):
        row = dict(init_time=iso(stamp) if stamp is not None else "", source="MERRA-2" if product.merra_host else "GEOS-FP",
                   product=product.name, file_path=str(path or ""), file_size=None, decoded_time="",
                   status="failed", missing_variables="", validation_message="", required=required)
        if path is None:
            row["validation_message"] = "Missing or ambiguous file"
        else:
            if path not in checked:
                evidence = {}
                try:
                    evidence["file_size"] = path.stat().st_size
                    with netCDF4.Dataset(path) as ds:
                        clock = ds.variables["time"]
                        clock.set_auto_mask(False)
                        times_decoded = netCDF4.num2date(clock[:], clock.units, getattr(clock, "calendar", "standard"))
                        evidence["decoded_time"] = " ".join(t.isoformat() + "Z" for t in times_decoded)
                        evidence["time_increment"] = str(getattr(ds, "time_increment", getattr(clock, "time_increment", "")))
                        evidence["grid"] = "×".join(str(len(ds.dimensions[x])) for x in ("lat", "lon"))
                        names = set(product.markers)
                        if required and product.merra_host:
                            names |= required_references
                        if required and not product.merra_host:
                            ordered = list(norm.index)
                            names |= set(ordered[:12] if "aer" in product.name else ordered[12:17]
                                         if "flx" in product.name else [x.split("_")[0] for x in ordered[17:]])
                        evidence["missing_variables"] = " ".join(sorted(names - ds.variables.keys()))
                        if evidence["missing_variables"]:
                            raise ValueError("Required variables absent")
                    validate_file(path, product, item_time)
                    evidence["status"] = "passed"
                except Exception as error:
                    evidence.update(status="failed", validation_message=str(error))
                checked[path] = evidence
            row.update(checked[path])
        manifest.append(row)

    by_name = {product.name: product for product in PRODUCTS}
    ref_product = by_name[config["data"]["reference_product"]]

    def reference_candidates(product, day):
        return sorted({p.resolve() for folder in (refs_root, refs_root / product.directory)
                       for p in folder.glob("*.nc4") if classify(p.name) == (product, day, False)})

    for stamp in times:
        begin = len(manifest)
        for product in (p for p in PRODUCTS if not p.merra_host):
            files = geos_candidates(inputs_root, product, stamp)
            for path in files or [None]:
                record(stamp, product, stamp, path, True)
            if len(files) > 1:
                for row in manifest[-len(files):]:
                    row.update(status="failed", validation_message="Duplicate candidate")
        leads = sorted({sum(path) for path in plans(config)})
        targets = [stamp + pd.Timedelta(hours=h) for h in leads]
        for day in sorted({t.normalize() for t in targets}):
            files = reference_candidates(ref_product, day)
            for path in files or [None]:
                record(stamp, ref_product, day, path, True)
            if len(files) > 1:
                for row in manifest[-len(files):]:
                    row.update(status="failed", validation_message="Duplicate candidate")
        entry = {"init_time": iso(stamp), "status": "failed", "references": []}
        try:
            if any(row["status"] != "passed" for row in manifest[begin:]):
                raise ValueError("Required file checks failed")
            source = Inputs(root, single_config(config, stamp))
            raw, lat, lon = source.initial(stamp)
            entry.update(shape=list(raw.shape), nonfinite_initial=0, channels=[
                {"channel": i, "name": n, "unit": source.model_units[n]} for i, n in enumerate(source.names)],
                targets=[{"variable": alias, **item, "channel": i, "unit": u}
                         for (alias, item), i, u in zip(source.mapping.items(), source.positions, source.units, strict=True)])
            for target in targets:
                field, path = source.reference(target, lat, lon)
                entry["references"].append(dict(valid_time=iso(target), file=str(path),
                    invalid_by_variable=(~np.isfinite(field)).sum(axis=(1, 2)).tolist(),
                    negative_by_variable=(field < 0).sum(axis=(1, 2)).tolist()))
            entry["status"] = "passed"
        except Exception as error:
            entry["error"] = str(error)
        dates.append(entry)

    for product in PRODUCTS:
        base = refs_root if product.merra_host else inputs_root
        for path in sorted({p.resolve() for folder in (base, base / product.directory) for p in folder.glob("*") if p.is_file()}):
            item = classify(path.name)
            if item and item[0] == product and path not in checked:
                record(None, product, item[1], path, False)
                if item[2]:
                    manifest[-1].update(status="extra_copy", validation_message="Browser copy")
    write_rows(run / "data_manifest.csv", manifest)
    required_files = {r["file_path"]: r["file_size"] for r in manifest
                      if r["required"] and r["file_path"] and r["file_size"] is not None}
    weights = {}
    for step in sorted({h for path in plans(config) for h in path}):
        path = root / config["forecast"]["model_dir"] / f"gamfs_{step}h_traced.pt"
        weights[str(step)] = dict(path=str(path), bytes=path.stat().st_size if path.is_file() else None)
    result = {"status": "passed" if all(d["status"] == "passed" for d in dates) else "failed",
              "initializations": dates, "files_by_product": {
                  name: len({r["file_path"] for r in manifest if r["file_path"] and r["product"] == name})
                  for name in by_name},
              "required_product_dates": len(requested), "required_files_found": len(required_files),
              "required_file_bytes": sum(required_files.values()), "model_weights": weights}
    write_json(run / "data_validation.json", result)
    table = ["| 起报 UTC | 状态 | 参考时次 | 说明 |", "|---|---|---:|---|"]
    for date in dates:
        table.append("| {init_time} | {status} | {count} | {message} |".format(
            **date, count=len(date["references"]), message=date.get("error", "").replace("|", "/").replace("\n", " ")))
    products = [f"- {name}: {count}" for name, count in result["files_by_product"].items()]
    atomic_text(run / "data_validation.md", "\n".join([
        "# P1-A 数据检查", "", "起报字段均按原模型网格读取。额外产品检查解码时轴及基本变量。",
        "", *products, "", f"主实验匹配 {len(required_files)}/{len(requested)} 个产品日期，"
        f"共 {result['required_file_bytes'] / 1e9:.2f} GB。", "", *table, "",
        "变量、单位、参考时次及无效格点数详见 data_validation.json。", ""]))
    return result


def verify_and_seed_p0(archive, source, folder, token, env):
    """Read the P0 archive only after checking all inputs and replaying its stored outputs."""
    from utils.handle import anti_stand

    meta = read_json(archive / "metadata.json")
    old = meta["signature"]
    old_config = old["config"]
    stamp, = source.initial_times
    if old_config["experiment"]["init_times"] != [iso(stamp)]:
        raise ValueError("P0 initialization differs")
    for key in ("initial_precision", "seed", "lead_hours", "primary_strategy", "comparison_paths"):
        if old_config["forecast"][key] != source.config["forecast"][key]:
            raise ValueError(f"P0 forecast setting differs: {key}")
    if old["software"] != env["software"] or old["execution"] != env["execution"]:
        raise ValueError("P0 execution environment differs")
    prefix = "score_research/experiments/P0_residual_characterization/"
    code = {Path(x["path"]).name: x["sha256"] for x in old["code"]}
    required = {"forecast.py": prefix + "forecast.py", "data_io.py": prefix + "data_io.py",
                **{n: "utils/" + n for n in ("init.py", "handle.py", "scheduler.py")},
                "merra_norm.csv": "merra_norm.csv", "info.pkl": "info.pkl"}
    bridge = []
    for name, path in required.items():
        raw = (source.root / path).read_bytes()
        hashes = {hashlib.sha256(raw).hexdigest()}
        if name.endswith((".py", ".csv")):
            hashes.add(hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest())
        if code.get(name) in hashes:
            continue
        compatibility = COMPATIBLE_CODE.get(name)
        if compatibility is None or code.get(name) != compatibility[0] or compatibility[1] not in hashes:
            raise ValueError(f"P0 implementation or normalization differs: {name}")
        bridge.append(name)
    current = [*source.input_files[stamp].values(), *(source.weight(h) for h in {h for p in source.plans for h in p})]
    old_files = {Path(x["path"]).name: x for x in old["files"]}
    for path in current:
        recorded = old_files.get(path.name)
        if recorded is None:
            raise ValueError(f"P0 source filename differs: {path.name}")
        actual = identity(path, content="sha256" in recorded)
        keys = ("bytes", "sha256") if "sha256" in recorded else ("path", "bytes", "mtime_ns")
        if any(recorded.get(k) != actual[k] for k in keys):
            raise ValueError(f"P0 source identity differs: {path.name}")
    old_token = meta["forecast_fingerprint"]
    if old_token != fingerprint(old):
        raise ValueError("P0 provenance fingerprint invalid")
    target = archive / "cache" / old_token / stamp.strftime("%Y%m%d_%H%M%S")
    destination = folder / "cache" / token / target.name
    prefixes = {"+".join(map(str, path[:n])) for path in source.plans for n in range(1, len(path) + 1)}
    raw, lat, lon = source.initial(stamp)
    with xr.open_dataset(archive / "data" / "forecasts.nc", decode_timedelta=False) as ds, project_directory(source.root):
        if (ds.attrs.get("forecast_fingerprint") != old_token
                or ds.attrs.get("state_transfer") != "full normalized model output without clipping or quantization"
                or ds.attrs.get("initial_precision") != source.config["forecast"]["initial_precision"]
                or not np.array_equal(ds.init_time.values, [stamp.to_datetime64()])
                or not np.array_equal(ds.latitude.values, lat) or not np.array_equal(ds.longitude.values, lon)):
            raise ValueError("P0 forecast grid, time or precision differs")
        channels = ds.model_variable.values.tolist()
        positions = [source.names.index(x) for x in channels]
        if len(channels) != 3 or not np.array_equal(raw[positions], ds.initial_state.values[0]):
            raise ValueError("P0 stored initial state differs")
        records = ds.path.values.tolist()
        if sorted(records) != sorted("+".join(map(str, p)) for p in source.plans):
            raise ValueError("P0 path labels differ")
        for label in sorted(prefixes):
            array = np.load(target / (label + ".npy"), mmap_mode="r", allow_pickle=False)
            if array.shape != raw.shape or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Invalid P0 prefix: {label}")
            if label in records:
                i = records.index(label)
                lead = sum(map(int, label.split("+")))
                if (int(ds.lead_time.values[i]) != lead
                        or ds.valid_time.values[i] != (stamp + pd.Timedelta(hours=lead)).to_datetime64()
                        or not np.array_equal(anti_stand(channels, array[positions].copy()), ds.forecast.values[i])):
                    raise ValueError(f"P0 replay failed: {label}")
    # The generate routine reads external caches only after the whole archive is validated.
    destination.mkdir(parents=True, exist_ok=True)
    from shutil import copyfile
    for label in sorted(prefixes):
        name = label + ".npy"
        if not (destination / name).exists():
            temporary = destination / (name + ".import")
            copyfile(target / name, temporary)
            temporary.replace(destination / name)
    return {"archive": str(archive), "forecast_fingerprint": old_token,
            "imported_prefixes": sorted(prefixes), "archive_git_commit": meta["git_commit"],
            "reviewed_code_compatibility": bridge}


def build(root, config, folder, provenance):
    source = Inputs(root, config, include_initial=False)
    stamp, = source.initial_times
    target = folder / "data" / "forecasts.nc"
    with xr.open_dataset(target, decode_timedelta=False) as ds:
        if ds.sizes["initialization"] != 1 or ds.init_time.values[0] != stamp.to_datetime64():
            raise ValueError("Forecast initialization differs from config")
        path_labels = {"+".join(map(str, p)): flags for p, flags in source.plans.items()}
        if sorted(ds.path.values.tolist()) != sorted(path_labels):
            raise ValueError("Forecast paths differ from config")
        for i, label in enumerate(ds.path.values):
            if path_labels[label] != (bool(ds.primary.values[i]), bool(ds.comparison.values[i])):
                raise ValueError(f"Forecast flags differ: {label}")
            if int(ds.lead_time.values[i]) != sum(map(int, label.split("+"))) or ds.initial_index.values[i] != 0:
                raise ValueError(f"Invalid lead or initialization index: {label}")
        expected = stamp.to_datetime64() + ds.lead_time.values.astype("timedelta64[h]")
        if not np.array_equal(expected, ds.valid_time.values):
            raise ValueError("valid_time != init_time + lead_time")
        lat, lon = ds.latitude.values, ds.longitude.values
        channels = source.channels
        f = ds.forecast.sel(variable=channels).values.astype(np.float64)
        initial = ds.initial_state.sel(variable=channels).values[0].astype(np.float64)
        valid_times, lookup = np.unique(ds.valid_time.values, return_inverse=True)
        fields, files = [], []
        for valid in valid_times:
            reference, path = source.reference(pd.Timestamp(valid), lat, lon)
            fields.append(reference.astype(np.float32))
            files.append(str(path))
        ref = np.stack(fields)
        r = ref[lookup].astype(np.float64)
        mask = np.isfinite(f) & np.isfinite(r) & np.isfinite(initial)
        for j in range(len(valid_times)):
            indices = lookup == j
            mask[indices] = mask[indices].all(axis=0)
        residual = np.where(mask, r - f, np.nan).astype(np.float32)
        persist = np.where(mask, r - initial, np.nan).astype(np.float32)
        max_reconstruction = np.zeros(len(channels))
        max_identity = np.zeros(len(channels))
        bound = 8 * np.finfo(np.float32).eps

        def check(delta, scale, valid, recorded):
            if np.any(np.where(valid, abs(delta) > bound * scale, False)):
                raise ValueError("Residual identity exceeds FP32 serialization tolerance")
            recorded[:] = np.maximum(recorded, np.where(valid, abs(delta), 0).max(axis=(-2, -1)))

        for a in range(len(f)):
            check(f[a] + residual[a] - r[a], np.maximum(abs(f[a]), abs(r[a])), mask[a], max_reconstruction)
            for b in np.flatnonzero(lookup[:a] == lookup[a]):
                check(residual[a].astype(float) - residual[b] - (f[b] - f[a]),
                      np.maximum.reduce([abs(f[a]), abs(f[b]), abs(r[a])]), mask[a] & mask[b], max_identity)
        coords = {k: ds[k].load() for k in ("valid_time", "lead_time", "path", "primary", "comparison")}
        coords.update(variable=list(source.mapping), latitude=lat, longitude=lon,
                      init_time=stamp.to_datetime64(), unit=("variable", source.units),
                      model_variable=("variable", channels),
                      reference_variable=("variable", [v["reference"] for v in source.mapping.values()]),
                      reference_time=("reference_index", valid_times))
        out = xr.Dataset(
            {"residual": (("record", *DIMS), residual),
             "persistence_residual": (("record", *DIMS), persist),
             "valid_mask": (("record", *DIMS), mask.astype("u1")),
             "reference": (("reference_index", *DIMS), ref),
             "reference_file": ("reference_index", files),
             "reference_index_for_record": ("record", lookup),
             "max_reconstruction_error": ("variable", max_reconstruction),
             "max_path_identity_error": ("variable", max_identity)}, coords=coords,
            attrs={"time_zone": "UTC", "residual_sign": "reference - forecast",
                   "forecast_provenance": ds.attrs["provenance"],
                   "build_provenance": json.dumps(provenance, ensure_ascii=False),
                   "forecast_artifact": "data/forecasts.nc",
                   "state_transfer": ds.attrs["state_transfer"], "initial_precision": ds.attrs["initial_precision"],
                   "mask_rule": "common finite initial, reference and forecasts at the same init and valid time"})
        for name in ("reference", "residual", "persistence_residual"):
            out[name].attrs["units"] = "see unit coordinate by variable"
        out.latitude.attrs["units"] = "degrees_north"
        out.longitude.attrs["units"] = "degrees_east"
        write_nc(out, folder / "residuals.nc")
