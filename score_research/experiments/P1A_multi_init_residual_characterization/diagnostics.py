"""P1-A statistics, using the P0 diagnostic primitives."""
from collections import defaultdict
from itertools import combinations, combinations_with_replacement

import numpy as np
import pandas as pd
import xarray as xr

from analysis import correlation, negative, spatial_rows, summary, weighted
from storage import write_rows

TAIL_FRACTIONS = (0.001, 0.01, 0.05, 0.1)
CURVE_FRACTIONS = sorted({*TAIL_FRACTIONS, *np.geomspace(0.0001, 1, 21)})


def weights(lat, lon):
    area = np.cos(np.deg2rad(lat.astype(float)))
    area[np.isclose(abs(lat), 90)] = 0
    return np.broadcast_to(area[:, None], (len(lat), len(lon)))


def ratio(numerator, denominator, floor=0):
    return numerator / denominator if denominator > floor else np.nan


def tail(field, mask, area, prediction=None):
    flat = np.flatnonzero(mask)
    x = field.ravel()[flat]
    order = np.argsort(-abs(x), kind="stable")
    ranked = flat[order]
    x = x[order]
    weighted_square = x ** 2 * area.ravel()[ranked]
    cumulative = np.r_[0, np.cumsum(weighted_square, dtype=np.float64)]
    high = np.zeros(field.size, dtype=bool)
    if prediction is not None:
        idx = np.argsort(-prediction.ravel()[flat], kind="stable")
        high[flat[idx[:int(np.ceil(0.01 * len(flat)))]]] = True
    rows = []
    for frac in CURVE_FRACTIONS:
        n = int(np.ceil(frac * len(flat)))
        row = dict(fraction=frac, tail_count=n,
                   contribution=ratio(cumulative[n], cumulative[-1]))
        if frac in TAIL_FRACTIONS:
            row.update(positive_fraction=ratio((x[:n] > 0).sum(), n),
                       negative_fraction=ratio((x[:n] < 0).sum(), n),
                       high_prediction_overlap=ratio(high[ranked[:n]].sum(), n) if prediction is not None else np.nan)
        rows.append(row)
    hotspot = np.zeros(field.size, dtype=bool)
    hotspot[ranked[:int(np.ceil(0.01 * len(flat)))]] = True
    return rows, hotspot.reshape(field.shape)


def analyze(folder, config, floor):
    tables = defaultdict(list)
    with (xr.open_dataset(folder / "residuals.nc", decode_timedelta=False) as ds,
          xr.open_dataset(folder / "data" / "forecasts.nc", decode_timedelta=False) as source):
        lat, lon = ds.latitude.values, ds.longitude.values
        area = weights(lat, lon)
        variables = ds.variable.values.tolist()
        model = ds.model_variable.values.tolist()
        units = dict(zip(variables, ds.unit.values, strict=True))
        stamp = pd.Timestamp(ds.init_time.values).isoformat() + "Z"
        initials = source.initial_state.sel(variable=model).values[0].astype(float)
        main = np.flatnonzero(ds.primary.values)
        paths = np.flatnonzero(ds.comparison.values)
        labels = ds.path.values
        records = []
        for i, label in enumerate(labels):
            row = ds.isel(record=i)
            ref_index = int(row.reference_index_for_record.item())
            key = dict(init_time=stamp, valid_time=pd.Timestamp(row.valid_time.values).isoformat() + "Z",
                       lead_hours=int(row.lead_time.item()), path=str(label), primary=bool(row.primary.item()))
            f = source.forecast.isel(record=i).sel(variable=model).values.astype(float)
            r = ds.reference.isel(reference_index=ref_index).values.astype(float)
            e = row.residual.values.astype(float)
            p = row.persistence_residual.values.astype(float)
            mask = row.valid_mask.values.astype(bool)
            records.append((e, mask, f if i in paths else None))
            for j, variable in enumerate(variables):
                info = dict(**key, variable=variable, unit=units[variable])
                x, ref, err, persist, m = f[j], r[j], e[j], p[j], mask[j]
                metrics = summary(err, m, area)
                persistence = summary(persist, m, area)
                area_sum = area[m].sum()
                direct = np.sqrt(np.dot(area[m], (ref[m] - x[m]) ** 2) / area_sum) if area_sum else np.nan
                if not np.isclose(metrics["rmse"], direct, atol=8 * np.finfo("float32").eps * max(
                        np.max(abs(ref[m]), initial=0), np.max(abs(x[m]), initial=0)), rtol=1e-6, equal_nan=True):
                    raise ValueError(f"Independent RMSE check failed: {info}")
                tables["residual_statistics"].append(dict(
                    **info, **metrics, area_sum=area_sum,
                    squared_error_sum=np.sum(area[m] * err[m] ** 2), independent_rmse=direct))
                skill = 1 - ratio(metrics["rmse"], persistence["rmse"], floor[model[j]])
                tables["persistence_skill"].append(dict(
                    **info, rmse=metrics["rmse"], mae=metrics["mae"], persistence_rmse=persistence["rmse"],
                    persistence_mae=persistence["mae"], skill=skill, area_sum=area_sum,
                    persistence_squared_error_sum=np.sum(area[m] * persist[m] ** 2)))
                neg = negative(x, np.isfinite(x), area, 0)
                for name, values in (("forecast", neg), ("reference", negative(ref, np.isfinite(ref), area, 0))):
                    tables["negative_statistics"].append(dict(**info, field=name, **values))
                if i in paths:
                    tables["path_statistics"].append(dict(**info, **metrics, area_negative_ratio=neg["area_negative_ratio"]))
                clipped = np.maximum(x, 0)
                for subset, valid in (("all", m), ("original_negative", m & (x < 0))):
                    for version, field in (("raw", x), ("clipped", clipped)):
                        tables["clipping_comparison"].append(dict(
                            **info, subset=subset, version=version,
                            **summary(ref - field, valid, area),
                            forecast_mean=weighted(field[valid], area[valid])))
                for label, condition in (("forecast", x), ("initial_state", initials[j])):
                    tables["condition_correlation"].append(dict(
                        **info, condition=label, signed=correlation(condition, err, m, area),
                        absolute=correlation(condition, abs(err), m, area),
                        squared=correlation(condition, err ** 2, m, area)))
                for label, field, stats in (("reference", ref, summary(ref, m, area)),
                                            ("persistence_residual", persist, persistence),
                                            ("residual", err, metrics)):
                    rows, _ = tail(field, m, area, x if label == "residual" else None)
                    tables["tail_statistics"].extend(dict(**info, field=label, **item) for item in rows)
                    tables["parameterization_comparison"].append(dict(
                        **info, field=label, **stats, centered_variance=stats["std"] ** 2,
                        weighted_squared_norm=np.sum(area[m] * field[m] ** 2)))
                    if i in main:
                        tables["spatial_correlation"].extend(dict(**info, field=label, **item)
                            for item in spatial_rows(field, m, area, lat, lon, config))
            if i in main:
                for a, b in combinations_with_replacement(range(len(variables)), 2):
                    pair_mask = mask[a] & mask[b]
                    tables["cross_variable_correlation"].append(dict(
                        **key, a=variables[a], b=variables[b], pairs=pair_mask.sum(),
                        correlation=correlation(e[a], e[b], pair_mask, area)))
        for selection, key in ((main, "cross_lead_correlation"), (paths, "path_discrepancy")):
            for a, b in combinations(selection, 2):
                ea, ma, fa = records[a]
                eb, mb, fb = records[b]
                for j, variable in enumerate(variables):
                    m = ma[j] & mb[j]
                    left, right = (ea[j], eb[j]) if key == "cross_lead_correlation" else (fa[j], fb[j])
                    row = dict(init_time=stamp, variable=variable, unit=units[variable],
                               a=str(labels[a]), b=str(labels[b]),
                               lead_a=int(ds.lead_time.values[a]), lead_b=int(ds.lead_time.values[b]),
                               pairs=m.sum(), correlation=correlation(left, right, m, area))
                    if key == "path_discrepancy":
                        diff = np.sqrt(weighted((left[m] - right[m]) ** 2, area[m]))
                        typical = sum(np.sqrt(weighted(field[j][m] ** 2, area[m])) for field in (ea, eb)) / 2
                        row.update(discrepancy=diff, relative_discrepancy=ratio(diff, typical, floor[model[j]]))
                    tables[key].append(row)
    out = folder / "statistics"
    out.mkdir(exist_ok=True)
    for name, rows in tables.items():
        write_rows(out / f"{name}.csv", rows)
    return sorted(tables)
