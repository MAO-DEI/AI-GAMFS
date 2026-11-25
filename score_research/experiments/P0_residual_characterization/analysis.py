"""Area-weighted diagnostics and offline figures for a P0 residual dataset."""
from collections import defaultdict
from itertools import combinations_with_replacement

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

QUANTILES = [1, 5, 25, 50, 75, 95, 99]
QNAMES = ["p1", "p5", "p25", "median", "p75", "p95", "p99"]


def weighted(v, w):
    return np.average(v, weights=w) if w.sum() > 0 else np.nan


def summary(field, mask, weights):
    x, w = field[mask], weights[mask]
    mean = weighted(x, w)
    quantiles = np.percentile(x, QUANTILES) if x.size else [np.nan] * len(QUANTILES)
    variable = x.size > 1 and np.ptp(x) > 0
    return dict(valid=x.size, invalid=field.size - x.size, mean=mean, bias=mean,
                std=np.sqrt(weighted((x - mean) ** 2, w)), mae=weighted(abs(x), w),
                rmse=np.sqrt(weighted(x ** 2, w)), minimum=x.min() if x.size else np.nan,
                maximum=x.max() if x.size else np.nan,
                extreme_abs=abs(x).max() if x.size else np.nan,
                skewness=stats.skew(x) if variable else np.nan,
                excess_kurtosis=stats.kurtosis(x) if variable else np.nan,
                **dict(zip(QNAMES, quantiles, strict=True)))


def correlation(a, b, mask, weights):
    x, y, w = a[mask], b[mask], weights[mask]
    dx, dy = x - weighted(x, w), y - weighted(y, w)
    sx, sy = weighted(dx ** 2, w), weighted(dy ** 2, w)
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0 or not (sx > 0 and sy > 0):
        return np.nan
    return np.clip(weighted(dx * dy, w) / np.sqrt(sx * sy), -1, 1)


def negative(field, valid, weights, threshold):
    x, w = field[valid], weights[valid]
    magnitudes = -x[x < 0]
    result = dict(valid=x.size, negative_count=magnitudes.size,
                  negative_ratio=magnitudes.size / x.size if x.size else np.nan,
                  area_negative_ratio=weighted(x < 0, w),
                  minimum=x.min() if x.size else np.nan,
                  magnitude_mean=magnitudes.mean() if magnitudes.size else np.nan,
                  magnitude_max=magnitudes.max() if magnitudes.size else np.nan,
                  area_mean_negative_magnitude=weighted(np.maximum(-x, 0), w),
                  threshold=threshold, below_threshold_count=int((x < -threshold).sum()),
                  area_below_threshold_ratio=weighted(x < -threshold, w))
    for p in (5, 25, 50, 75, 95, 99):
        result[f"magnitude_p{p}"] = np.percentile(magnitudes, p) if magnitudes.size else np.nan
    return result


def spatial_rows(field, mask, weights, lat, lon, config):
    dlon = abs(lon[1] - lon[0])
    dlat = abs(lat[1] - lat[0])
    bands = config["latitude_bands"]
    for lag in config["lags"]:
        if type(lag) is not int or not 0 < lag < min(field.shape[0], field.shape[1] // 2 + 1):
            raise ValueError(f"Invalid spatial lag: {lag}")
        shifted = np.roll(field, -lag, axis=1)
        paired = mask & np.roll(mask, -lag, axis=1)
        distances = 2 * 6371.0088 * np.arcsin(
            np.clip(np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lag * dlon) / 2), 0, 1))
        for low, high in zip(bands[:-1], bands[1:], strict=True):
            rows = (lat >= low) & ((lat < high) | ((high == bands[-1]) & (lat == high)))
            m = paired & rows[:, None]
            active = (m & (weights > 0)).any(axis=1)
            d = distances[active]
            yield dict(direction="east-west", lag=lag, band=f"{low} to {high}",
                       distance_min_km=d.min() if d.size else np.nan,
                       distance_max_km=d.max() if d.size else np.nan,
                       pairs=m.sum(), correlation=correlation(field, shifted, m, weights))
        m = mask[:-lag] & mask[lag:]
        w = (weights[:-lag] + weights[lag:]) / 2
        dist = 6371.0088 * np.deg2rad(lag * dlat)
        yield dict(direction="north-south", lag=lag, band="global", distance_min_km=dist,
                   distance_max_km=dist, pairs=m.sum(),
                   correlation=correlation(field[:-lag], field[lag:], m, w))


class Figures:
    def __init__(self, root, dpi, lat, lon):
        self.root, self.dpi = root, dpi
        self.lat, self.lon = lat, lon

    def save(self, fig, group, name):
        folder = self.root / group
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png", dpi=self.dpi)
        plt.close(fig)

    def maps(self, fields, title, unit, group, name):
        fig, axes = plt.subplots(1 if len(fields) == 1 else 2, 1 if len(fields) == 1 else 3,
                                 figsize=(7, 4) if len(fields) == 1 else (15, 7), layout="constrained")
        for ax, (label, array, cmap, low, high) in zip(np.ravel(axes), fields, strict=True):
            img = ax.pcolormesh(self.lon, self.lat, array, shading="auto", cmap=cmap, vmin=low, vmax=high)
            ax.set(title=label, xlabel="Longitude (°E)", ylabel="Latitude (°N)", ylim=(-90, 90))
            fig.colorbar(img, ax=ax, label=unit, shrink=0.8)
        fig.suptitle(title)
        self.save(fig, group, name)

    def matrix(self, values, labels, title, group, name, corr=False):
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        kwargs = dict(vmin=-1, vmax=1, cmap="coolwarm") if corr else {}
        image = ax.imshow(values, **kwargs)
        ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
               xticklabels=labels, yticklabels=labels, title=title)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.colorbar(image, ax=ax)
        self.save(fig, group, name)


def sample_range(fields):
    valid = [x[np.isfinite(x)] for x in fields]
    low = min(x.min() for x in valid if x.size)
    high = max(x.max() for x in valid if x.size)
    delta = max(abs(low), 1e-30) * 0.01
    return (low, high) if high > low else (low - delta, high + delta)


def distributions(plots, items, title, unit, name, detailed=False, group="distribution"):
    samples = [(name, x[np.isfinite(x)]) for name, x in items]
    samples = [(name, x) for name, x in samples if x.size]
    if not samples:
        return
    lo, hi = sample_range([x for _, x in samples])
    bins = np.linspace(lo, hi, 61)
    fig, axes = plt.subplots(1, 3 if detailed else 1, figsize=(14 if detailed else 8, 4.3),
                             layout="constrained")
    hist = np.ravel(axes)[0]
    for label, x in samples:
        hist.hist(x, bins=bins, histtype="step", density=True, label=label)
        if detailed:
            density, edges = np.histogram(x, bins=bins, density=True)
            axes[1].plot((edges[:-1] + edges[1:]) / 2, density, label=label)
            mean, sigma = x.mean(), x.std()
            if sigma > 0:
                axes[1].plot(bins, stats.norm.pdf(bins, mean, sigma), "--", label=f"{label} normal")
                q = np.linspace(0.001, 0.999, 299)
                expected = stats.norm.ppf(q, mean, sigma)
                axes[2].plot(expected, np.quantile(x, q), label=label)
                axes[2].plot(expected, expected, ":", color="grey")
    hist.set(xlabel=unit, ylabel="Grid-cell density", title="Histogram")
    hist.legend(fontsize="small")
    if detailed:
        axes[1].set(xlabel=unit, title="Empirical and fitted density")
        axes[2].set(xlabel=f"Normal quantile ({unit})", ylabel=f"Sample quantile ({unit})", title="Q–Q")
        axes[1].legend(fontsize="small")
        if axes[2].lines:
            axes[2].legend(fontsize="small")
    fig.suptitle(title)
    plots.save(fig, group, name)


def condition_rows(condition, residual, mask, weights, count):
    x = condition[mask]
    if not x.size:
        return
    edges = np.unique(np.quantile(x, np.linspace(0, 1, count + 1)))
    if edges.size == 1:
        edges = np.repeat(edges, 2)
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = mask & (condition >= low)
        selected &= (condition <= high) if i == edges.size - 2 else (condition < high)
        yield dict(bin_index=i, lower=low, upper=high,
                   condition_mean=weighted(condition[selected], weights[selected]),
                   **summary(residual, selected, weights))


def diagnostics(ds, config, plots):
    tables = defaultdict(list)
    lat, lon = ds.latitude.values, ds.longitude.values
    w = np.cos(np.deg2rad(lat))
    w[np.isclose(abs(lat), 90)] = 0
    weights = np.broadcast_to(w[:, None], (len(lat), len(lon)))
    fields = {key: ds[key].values for key in ("forecast", "reference", "residual",
                                              "persistence_residual", "initial_state")}
    masks = ds.valid_mask.values.astype(bool)
    for init, stamp in enumerate(ds.init_time.values):
        records = np.flatnonzero(ds.initial_index.values == init)
        main = sorted((i for i in records if ds.primary.values[i]), key=lambda i: ds.lead_time.values[i])
        paths = [i for i in records if ds.comparison.values[i]]
        for v, variable in enumerate(ds.variable.values):
            unit = str(ds.unit.values[v])
            initial = fields["initial_state"][init, v]
            caption = f"{variable} | init {stamp} UTC"
            prefix = f"i{init:04d}_v{v:02d}"
            if config["spatial"]:
                state_range = sample_range([initial, fields["forecast"][main, v],
                                            fields["reference"][main, v]])
                extremes = np.abs(np.concatenate([fields["residual"][main, v],
                                                  fields["persistence_residual"][main, v]]))
                extent = max(np.nanmax(extremes), 1e-30) if np.isfinite(extremes).any() else 1e-30
            for i in records:
                lead, path = int(ds.lead_time.values[i]), str(ds.path.values[i])
                key = dict(init_time=str(stamp) + "Z", record=int(i), variable=variable,
                           unit=unit, lead_hours=lead, path=path, steps=len(path.split("+")),
                           is_primary=bool(ds.primary.values[i]))
                name, title = f"{prefix}_r{i:04d}", f"{caption} | valid {ds.valid_time.values[i]} UTC | {lead} h [{path}]"
                f, r, e, p = (fields[k][i, v].astype(float) for k in
                              ("forecast", "reference", "residual", "persistence_residual"))
                m = masks[i, v]
                if config["statistics"] or (config["lead_time"] and i in main) or (config["paths"] and i in paths):
                    metrics = summary(e, m, weights)
                    if config["statistics"]:
                        tables["residual_statistics"].append({**key, **metrics})
                        distributions(plots, [("AI-GAMFS", e), ("Persistence", p)], title, unit, name, True)
                    if config["lead_time"] and i in main:
                        tables["lead_time_statistics"].append({**key, **metrics})
                    if config["paths"] and i in paths:
                        negatives = negative(f, m, weights, 0)
                        tables["path_statistics"].append({
                            **key, **metrics, "reference_correlation": correlation(f, r, m, weights),
                            "forecast_negative_ratio": negatives["negative_ratio"],
                            "forecast_area_negative_ratio": negatives["area_negative_ratio"],
                        })
                if config["persistence"]:
                    targets = {"reference": r, "persistence_residual": p, "ai_gamfs_residual": e}
                    for label, x in targets.items():
                        tables["persistence_comparison"].append({
                            **key, "target": label, **summary(x, m, weights),
                            "reference_correlation": correlation(x, r, m, weights),
                        })
                    distributions(plots, [(label, np.where(m, x, np.nan)) for label, x in targets.items()],
                                  title, unit, name, group="targets")
                if config["spatial"] and i in main:
                    low, high = state_range
                    plots.maps([
                        ("Initial", initial, "viridis", low, high),
                        ("Reference", r, "viridis", low, high),
                        ("Forecast", f, "viridis", low, high),
                        ("Residual", e, "RdBu_r", -extent, extent),
                        ("Persistence residual", p, "RdBu_r", -extent, extent),
                        ("Absolute residual", abs(e), "magma", 0, extent),
                    ], title, unit, "spatial", name)
                    targets = {"ai_gamfs_residual": e}
                    if config["persistence"]:
                        targets.update(reference=r, persistence_residual=p)
                    for label, x in targets.items():
                        rows = list(spatial_rows(x, m, weights, lat, lon, config))
                        tables["spatial_correlation"].extend({**key, "target": label, **row} for row in rows)
                if config["negatives"]:
                    threshold = config["negative_thresholds"].get(variable, 0)
                    if not np.isfinite(threshold) or threshold < 0:
                        raise ValueError(f"Invalid negative threshold for {variable}")
                    for label, x, valid in (("forecast", f, np.isfinite(f)),
                                             ("reference", r, np.isfinite(r))):
                        tables["negative_statistics"].append({
                            **key, "field": label, **negative(x, valid, weights, threshold),
                        })
                    # Reference quantile bins indicate whether forecast negatives cluster
                    # in low-reference regions. The reference is a diagnostic only.
                    for row in condition_rows(r, (f < 0).astype(float), m, weights, 5):
                        tables["negative_by_reference"].append({
                            **key, "reference_lower": row["lower"], "reference_upper": row["upper"],
                            "valid": row["valid"], "area_negative_ratio": row["mean"],
                        })
                    clipped = np.maximum(f, 0)
                    for label, subset in (("all", m), ("original_negative", m & (f < 0))):
                        for version, x in (("raw", f), ("clipped", clipped)):
                            fractions = negative(x, subset, weights, threshold)
                            tables["clipping_comparison"].append({
                                **key, "subset": label, "version": version,
                                **summary(r - x, subset, weights),
                                "negative_ratio": fractions["negative_ratio"],
                                "area_negative_ratio": fractions["area_negative_ratio"],
                                "forecast_mean": weighted(x[subset], weights[subset]),
                                "mean_change_from_raw": weighted((x - f)[subset], weights[subset]),
                            })
                    if (f < 0).any():
                        magnitude = np.where(f < 0, -f, np.nan)
                        plots.maps([("Negative forecast magnitude", magnitude, "magma", 0, np.nanmax(magnitude))],
                                   title, unit, "negatives", name)
                if config["conditions"] and m.any():
                    conditional_plots(tables, plots, config, key, title, unit, name,
                                      initial, f, e, m, weights, lat)
            if config["lead_time"]:
                lead_plots(ds, v, main, prefix, caption, unit, fields, masks, weights, tables, plots)
            if config["paths"] and paths:
                path_plots(ds, v, paths, prefix, caption, unit, fields, masks, weights, tables, plots)
    return tables


def conditional_plots(tables, plots, config, key, title, unit, name, initial, forecast, e, m, w, lat):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for column, (label, x, y) in enumerate((("forecast", forecast, e), ("initial", initial, e),
                                            ("forecast_vs_absolute", forecast, abs(e)))):
        corr = correlation(x, y, m, w)
        count = config["condition_bins"]
        if type(count) is not int or count < 1:
            raise ValueError("condition_bins must be a positive integer")
        rows = list(condition_rows(x, y, m, w, count))
        tables["conditional_statistics"].extend({**key, "condition": label, "global_pearson": corr, **row}
                                                 for row in rows)
        ax = axes[0, column]
        mesh = ax.hexbin(x[m], y[m], gridsize=32, mincnt=1, bins="log")
        fig.colorbar(mesh, ax=ax, label="Grid cells")
        response = "|Residual|" if label.endswith("absolute") else "Residual"
        ax.set(xlabel=f"{label.split('_')[0]} ({unit})", ylabel=f"{response} ({unit})")
        ax = axes[1, column]
        for metric in ("mean", "std", "p5", "p95"):
            ax.plot([r["condition_mean"] for r in rows], [r[metric] for r in rows], "o-", label=metric)
        ax.set(xlabel=f"Bin mean ({unit})", ylabel=f"{response} ({unit})")
        ax.legend(fontsize="small")
    bands = config["latitude_bands"]
    for lo, hi in zip(bands[:-1], bands[1:], strict=True):
        region = (lat >= lo) & ((lat < hi) | ((hi == bands[-1]) & (lat == hi)))
        tables["conditional_statistics"].append({
            **key, "condition": "latitude_band", "lower": lo, "upper": hi,
            **summary(e, m & region[:, None], w),
        })
    fig.suptitle(title)
    plots.save(fig, "conditions", name)


def lead_plots(ds, v, records, prefix, caption, unit, fields, masks, w, tables, plots):
    labels = [int(ds.lead_time.values[i]) for i in records]
    summary_rows = [summary(fields["residual"][i, v], masks[i, v], w) for i in records]
    groups = [("bias", "mae", "rmse", "std"), tuple(QNAMES), ("skewness", "excess_kurtosis"), ("extreme_abs",)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, group in zip(axes.flat, groups, strict=True):
        for key in group:
            ax.plot(labels, [row[key] for row in summary_rows], "o-", label=key)
        ax.set(xlabel="Lead (h)", ylabel="Dimensionless" if "skewness" in group else unit)
        ax.legend(fontsize="small", ncol=2)
    fig.suptitle(caption)
    plots.save(fig, "lead_time", prefix)
    distributions(plots, [(f"{lead} h", fields["residual"][i, v]) for i, lead in zip(records, labels)],
                  caption, unit, "distribution_" + prefix, group="lead_time")
    matrix = np.empty((len(records), len(records)))
    for a, i in enumerate(records):
        for b, j in enumerate(records):
            m = masks[i, v] & masks[j, v]
            matrix[a, b] = correlation(fields["residual"][i, v], fields["residual"][j, v], m, w)
            tables["cross_lead_correlation"].append(dict(group=prefix, record_a=int(i), record_b=int(j),
                                                        init_time=str(ds.init_time.values[ds.initial_index.values[i]]) + "Z",
                                                        variable=ds.variable.values[v], lead_a=labels[a], lead_b=labels[b],
                                                        valid=m.sum(), pearson=matrix[a, b]))
    plots.matrix(matrix, labels, caption + " | cross-lead spatial Pearson", "lead_time", prefix + "_corr", True)


def path_plots(ds, v, paths, prefix, caption, unit, fields, masks, w, tables, plots):
    labels = [str(ds.path.values[i]) for i in paths]
    caption += f" | {int(ds.lead_time.values[paths[0]])} h"
    matrices = [np.full((len(paths), len(paths)), np.nan) for _ in range(2)]
    for a, b in combinations_with_replacement(range(len(paths)), 2):
        i, j = paths[a], paths[b]
        m = masks[i, v] & masks[j, v]
        fa, fb = (fields["forecast"][index, v].astype(float) for index in (i, j))
        diff = fa - fb
        identity = fields["residual"][i, v] - fields["residual"][j, v] + diff
        scale = np.maximum.reduce([abs(fa), abs(fb), abs(fields["reference"][i, v])])
        tol = 8 * np.finfo(np.float64).eps * scale
        agrees = bool((abs(identity[m]) <= tol[m]).all()) if m.any() else None
        rmse = np.sqrt(weighted(diff[m] ** 2, w[m]))
        rho = correlation(fa, fb, m, w)
        matrices[0][a, b] = matrices[0][b, a] = rmse
        matrices[1][a, b] = matrices[1][b, a] = rho
        tables["path_discrepancy"].append(dict(group=prefix, record_a=int(i), record_b=int(j),
                                              variable=ds.variable.values[v], path_a=labels[a], path_b=labels[b], unit=unit,
                                              valid=m.sum(), rmse=rmse, forecast_pearson=rho,
                                              identity_max_abs=abs(identity[m]).max() if m.any() else np.nan,
                                              identity_pass=agrees))
        if agrees is False:
            raise ValueError(f"Residual identity failed for records {i}, {j}")
        if a != b and m.any():
            lim = max(abs(diff[m]).max(), 1e-30)
            plots.maps([(f"{labels[a]} minus {labels[b]}", np.where(m, diff, np.nan), "RdBu_r", -lim, lim)],
                       caption, unit, "path_comparison", f"{prefix}_{a}_{b}")
    plots.matrix(matrices[0], labels, f"{caption} | forecast difference RMSE ({unit})",
                 "path_comparison", prefix + "_rmse")
    plots.matrix(matrices[1], labels, caption + " | forecast spatial Pearson",
                 "path_comparison", prefix + "_corr", True)
    distributions(plots, [(label, fields["residual"][i, v]) for label, i in zip(labels, paths)],
                  caption, unit, "distribution_" + prefix, group="path_comparison")


def summarize_spatial(tables, plots):
    if not tables.get("spatial_correlation"):
        return
    frame = pd.DataFrame(tables["spatial_correlation"])
    for record, rows in frame.groupby(["record", "variable"], sort=True):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for (direction, target, band), group in rows.groupby(["direction", "target", "band"]):
            ax = axes[0 if direction == "east-west" else 1]
            low, high = group.distance_min_km.to_numpy(), group.distance_max_km.to_numpy()
            ax.errorbar((low + high) / 2, group.correlation, xerr=(high - low) / 2,
                        marker=".", linewidth=0.7, label=f"{band} {target}")
            ax.set(title=direction, xlabel="Great-circle distance (km)", ylabel="Weighted Pearson")
        for ax in axes:
            ax.legend(fontsize=6, ncol=2)
        fig.suptitle(f"Record {record[0]} | {record[1]}")
        index = list(dict.fromkeys(frame.variable)).index(record[1])
        plots.save(fig, "autocorrelation", f"r{record[0]:04d}_v{index:02d}")


def summarize_negatives(tables, plots):
    if not tables.get("negative_statistics"):
        return
    frame = pd.DataFrame(tables["negative_statistics"])
    frame = frame[frame.is_primary]
    for index, ((stamp, variable), rows) in enumerate(frame.groupby(["init_time", "variable"])):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for field, series in rows.groupby("field"):
            series = series.sort_values("lead_hours")
            axes[0].plot(series.lead_hours, series.area_negative_ratio, "o-", label=field)
            axes[1].plot(series.lead_hours, series.area_mean_negative_magnitude, "o-", label=field)
        axes[0].set(xlabel="Lead (h)", ylabel="Area-weighted negative fraction")
        axes[1].set(xlabel="Lead (h)", ylabel=f"Area mean negative magnitude ({rows.unit.iloc[0]})")
        for ax in axes:
            ax.legend()
        fig.suptitle(f"{variable} | {stamp}")
        plots.save(fig, "negatives", f"lead_summary_{index:03d}")


def analyze(run, config, directory):
    directory.mkdir()
    tables_dir = directory / "statistics"
    tables_dir.mkdir()
    with xr.open_dataset(run / "data" / "residuals.nc", engine="netcdf4", decode_timedelta=False) as ds:
        plots = Figures(directory / "figures", config["output"]["dpi"], ds.latitude.values, ds.longitude.values)
        tables = diagnostics(ds, config["analysis"], plots)
        summarize_spatial(tables, plots)
        summarize_negatives(tables, plots)
        for key, rows in tables.items():
            pd.DataFrame(rows).to_csv(tables_dir / f"{key}.csv", index=False)
