"""Aggregate successful dates and build publication-independent diagnostics."""
from contextlib import ExitStack
from itertools import combinations

import numpy as np
import pandas as pd
import xarray as xr

from analysis import correlation, summary, weighted
from data_io import write_nc
from diagnostics import tail, weights
from storage import write_rows


GROUP = ["variable", "unit", "lead_hours", "path", "primary"]


def collect(folders, out):
    stats = out / "statistics"
    stats.mkdir(exist_ok=True)
    result = {}
    for file in sorted((folders[0] / "statistics").glob("*.csv")):
        result[file.stem] = table = pd.concat(
            [pd.read_csv(d / "statistics" / file.name, dtype={"path": str, "a": str, "b": str})
             for d in folders], ignore_index=True)
        table.to_csv(stats / file.name, index=False)
    metrics = result["residual_statistics"]
    merged = metrics.merge(result["persistence_skill"], on=["init_time", "valid_time", *GROUP],
                           suffixes=("", "_skill"), validate="one_to_one")
    rows = []
    for label, data in (("all_available", merged), ("excluding_p0", merged[merged.init_time != "2024-07-26T22:30:00Z"])):
        for key, g in data.groupby(GROUP, sort=False):
            area = g.area_sum.sum()
            rmse = g.rmse.dropna()
            improved = (g.rmse < g.persistence_rmse) & g.skill.notna()
            mae_valid = g.mae.notna() & g.persistence_mae.notna()
            rows.append(dict(zip(GROUP, key, strict=True), cohort=label, dates=len(g), skill_days=g.skill.count(),
                             improved_days=improved.sum(), mae_comparison_days=mae_valid.sum(),
                             mae_improved_days=((g.mae < g.persistence_mae) & mae_valid).sum(),
                             improved_fraction=improved.sum() / g.skill.count()
                             if g.skill.count() else np.nan, mean_daily_rmse=rmse.mean(), median_daily_rmse=rmse.median(),
                             std_daily_rmse=rmse.std(), min_daily_rmse=rmse.min(), max_daily_rmse=rmse.max(),
                             min_rmse_date=g.loc[rmse.idxmin(), "init_time"] if len(rmse) else "",
                             max_rmse_date=g.loc[rmse.idxmax(), "init_time"] if len(rmse) else "",
                             pooled_rmse=np.sqrt(g.squared_error_sum.sum() / area) if area else np.nan,
                             pooled_persistence_rmse=np.sqrt(g.persistence_squared_error_sum.sum() / area) if area else np.nan,
                             bias_mean=g.bias.mean(), underestimate_days=(g.bias > 0).sum(),
                             overestimate_days=(g.bias < 0).sum(), mean_skill=g.skill.mean(),
                             median_skill=g.skill.median(), min_skill=g.skill.min(), max_skill=g.skill.max()))
    write_rows(stats / "aggregate_statistics.csv", rows)
    pd.DataFrame(rows).query("primary").to_csv(stats / "lead_time_statistics.csv", index=False)
    trends = []
    for (stamp, variable), daily in merged.query("primary").groupby(["init_time", "variable"]):
        ordered = daily.sort_values("lead_hours")
        rmse = ordered.rmse.to_numpy()
        finite = np.isfinite(rmse)
        diff = np.diff(rmse)[finite[1:] & finite[:-1]]
        trends.append(dict(init_time=stamp, variable=variable, valid_leads=finite.sum(),
                           rising_steps=(diff > 0).sum(), falling_steps=(diff < 0).sum(),
                           tied_steps=(diff == 0).sum()))
    write_rows(stats / "lead_time_trends.csv", trends)

    groupings = {
        "spatial_correlation": (GROUP + ["field", "direction", "lag", "band"], ["correlation"]),
        "cross_variable_correlation": (["lead_hours", "a", "b"], ["correlation"]),
        "cross_lead_correlation": (["variable", "unit", "lead_a", "lead_b", "a", "b"], ["correlation"]),
        "tail_statistics": (GROUP + ["field", "fraction"], ["contribution", "high_prediction_overlap",
                                                            "positive_fraction", "negative_fraction"]),
        "path_discrepancy": (["variable", "unit", "a", "b"], ["discrepancy", "relative_discrepancy", "correlation"]),
        "path_statistics": (GROUP, ["rmse", "mae", "bias", "area_negative_ratio"]),
        "negative_statistics": (GROUP + ["field"], ["area_negative_ratio", "magnitude_mean", "magnitude_max"]),
        "parameterization_comparison": (GROUP + ["field"], ["mean", "std", "centered_variance", "weighted_squared_norm"]),
        "condition_correlation": (GROUP + ["condition"], ["signed", "absolute", "squared"]),
    }
    for name, (group, cols) in groupings.items():
        suffix = "tail_contribution" if name == "tail_statistics" else f"aggregate_{name}"
        table = result[name]
        summary_table = table.groupby(group, sort=False)[cols].agg(["count", "mean", "median", "min", "max", "std"])
        summary_table.columns = ["_".join(col) for col in summary_table.columns]
        if name == "negative_statistics":
            summary_table["days_with_negatives"] = table.assign(
                has_negative=table.negative_count > 0).groupby(group, sort=False).has_negative.sum()
        summary_table.reset_index().to_csv(stats / f"{suffix}.csv", index=False)
    clipped = result["clipping_comparison"].pivot(index=["init_time", *GROUP, "subset"], columns="version",
                                                  values=["rmse", "mae", "bias", "forecast_mean"])
    changes = pd.DataFrame({m: clipped[m]["clipped"] - clipped[m]["raw"] for m in ("rmse", "mae", "bias", "forecast_mean")})
    changes.reset_index().groupby(GROUP + ["subset"])[list(changes)].agg(
        ["count", "mean", "min", "max", lambda values: (values.dropna() < 0).sum()]).rename(
        columns={"<lambda_0>": "days_decreased"}).to_csv(stats / "aggregate_clipping_comparison.csv")

    table = result["path_statistics"].copy()
    for metric in ("rmse", "mae"):
        table[f"{metric}_rank"] = table.groupby(["init_time", "variable"])[metric].rank(method="min")
    table.to_csv(stats / "path_rankings_daily.csv", index=False)
    ranking = table.assign(rmse_first=table.rmse_rank == 1, mae_first=table.mae_rank == 1).groupby(
        GROUP)[["rmse_first", "mae_first"]].sum().reset_index()
    ranking.to_csv(stats / "path_ranking.csv", index=False)
    return stats


def pooled_conditions(x, residual, valid, area, bins, key, times):
    values = x[valid]
    if not values.size:
        return []
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) == 1:
        edges = edges.repeat(2)
    result = []
    area_block = np.broadcast_to(area, valid.shape)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        m = valid & (x >= lo) & ((x <= hi) if i == len(edges) - 2 else (x < hi))
        for k, label in [*enumerate(times), (slice(None), "pooled")]:
            use = m[k]
            a = area_block[k]
            result.append(dict(**key, init_time=label, bin_index=i, lower=lo, upper=hi,
                               weight_sum=a[use].sum(), condition_mean=weighted(x[k][use], a[use]),
                               **summary(residual[k], use, a)))
    return result


def aggregate_fields(folders, out, config, stats):
    rows, cross = [], []
    map_folder = out / "fields"
    map_folder.mkdir(exist_ok=True)
    with ExitStack() as scope:
        residuals = [scope.enter_context(xr.open_dataset(path / "residuals.nc", decode_timedelta=False)) for path in folders]
        forecasts = [scope.enter_context(xr.open_dataset(path / "data/forecasts.nc", decode_timedelta=False)) for path in folders]
        ds = residuals[0]
        lat, lon = ds.latitude.values, ds.longitude.values
        times = [pd.Timestamp(x.init_time.values).isoformat() + "Z" for x in residuals]
        for other in residuals[1:]:
            for coord in ("variable", "model_variable", "latitude", "longitude", "path", "lead_time", "unit", "primary"):
                if not np.array_equal(ds[coord].values, other[coord].values):
                    raise ValueError(f"Shards differ in {coord}")
        area = weights(lat, lon)
        for v, variable in enumerate(ds.variable.values):
            model = ds.model_variable.values[v]
            unit = str(ds.unit.values[v])
            initial = np.stack([item.initial_state.isel(initialization=0).sel(variable=model).values for item in forecasts])
            maps = []
            leads = []
            for i, label in enumerate(ds.path.values):
                lead = int(ds.lead_time.values[i])
                key = dict(variable=str(variable), unit=unit, lead_hours=lead, path=str(label))
                err = np.stack([d.residual.isel(record=i, variable=v).values for d in residuals]).astype(float)
                mask = np.stack([d.valid_mask.isel(record=i, variable=v).values for d in residuals]).astype(bool)
                f = np.stack([d.forecast.isel(record=i).sel(variable=model).values for d in forecasts])
                for name, x in (("forecast", f), ("initial_state", initial)):
                    rows.extend(pooled_conditions(x, err, mask, area, config["condition_bins"], dict(**key, condition=name), times))
                if not ds.primary.values[i]:
                    continue
                n = mask.sum(axis=0)
                denom = np.where(n, n, np.nan)
                mean = np.where(mask, err, 0).sum(axis=0) / denom
                sd = np.sqrt(np.where(mask, (err - mean) ** 2, 0).sum(axis=0) / denom)
                absolute = np.where(mask, abs(err), 0).sum(axis=0) / denom
                hotspots = [tail(e, m, area)[1] for e, m in zip(err, mask, strict=True)]
                count = np.sum(hotspots, axis=0)
                if lead in (3, 12, 24):
                    for j, k in combinations(range(len(times)), 2):
                        common = mask[j] & mask[k]
                        cross.append(dict(**key, init_a=times[j], init_b=times[k], pairs=common.sum(),
                                          correlation=correlation(err[j], err[k], common, area)))
                maps.append((mean, sd, absolute, count / denom, n))
                leads.append(lead)
            data = np.stack(maps)
            fields = xr.Dataset({name: (("lead_time", "latitude", "longitude"), data[:, j].astype(dtype))
                                 for j, (name, dtype) in enumerate((("mean_residual", "f4"), ("std_residual", "f4"),
                                   ("mean_absolute_residual", "f4"), ("top_1pct_frequency", "f4"), ("valid_dates", "i4")))},
                                coords=dict(lead_time=leads, latitude=lat, longitude=lon),
                                attrs=dict(variable=str(variable), units=unit, init_times=" ".join(times)))
            position = map_folder / f"variable_{v:02d}.nc"
            write_nc(fields, position)
    write_rows(stats / "conditional_statistics.csv", rows)
    write_rows(stats / "cross_init_spatial_correlation.csv", cross)
