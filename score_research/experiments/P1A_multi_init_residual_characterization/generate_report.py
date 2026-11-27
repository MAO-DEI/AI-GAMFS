"""Render an offline aggregate report from the evaluated shards and tables."""
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from analysis import Figures, plt
from run_experiment import atomic_text
from dataset_interface import load_fields


def figures(run, config):
    stats = run / "statistics"
    destination = run / "figures"
    data = pd.read_csv(stats / "lead_time_statistics.csv")
    spatial = pd.read_csv(stats / "aggregate_spatial_correlation.csv")
    tail = pd.read_csv(stats / "tail_contribution.csv")
    conditions = pd.read_csv(stats / "conditional_statistics.csv")
    cross = pd.read_csv(stats / "aggregate_cross_variable_correlation.csv")
    daily_cross = pd.read_csv(stats / "cross_variable_correlation.csv")
    neg = pd.read_csv(stats / "aggregate_negative_statistics.csv")
    path = pd.read_csv(stats / "aggregate_path_statistics.csv")
    rmse = pd.read_csv(stats / "residual_statistics.csv")
    curves = []
    for v in config["focus_variables"]:
        if v not in data.variable.values:
            continue
        group = data.query("variable == @v and cohort == 'all_available'").sort_values("lead_hours")
        unit = group.unit.iloc[0]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        ax, bx, cx, dx = axes.flat
        for _, each in rmse.query("variable == @v and primary").groupby("init_time"):
            ax.plot(each.lead_hours, each.rmse, color="grey", alpha=.23)
        ax.plot(group.lead_hours, group.mean_daily_rmse, marker="o", label="Mean daily RMSE")
        ax.plot(group.lead_hours, group.median_daily_rmse, label="Median daily RMSE")
        ax.plot(group.lead_hours, group.pooled_persistence_rmse, label="Pooled persistence", linestyle="--")
        ax.fill_between(group.lead_hours, group.min_daily_rmse, group.max_daily_rmse, alpha=.17)
        ax.set(xlabel="Lead (h)", ylabel=f"RMSE ({unit})")
        ax.legend(fontsize="small")
        for lead, part in tail.query("variable == @v and primary and field == 'residual' and lead_hours in [3, 12, 24]").groupby("lead_hours"):
            part = part.sort_values("fraction")
            bx.plot(part.fraction * 100, part.contribution_mean, label=f"{lead} h")
            bx.fill_between(part.fraction * 100, part.contribution_min, part.contribution_max, alpha=.12)
        bx.set(xlabel="Top absolute residual grid cells (%)", ylabel="Squared error share", xscale="log")
        bx.legend()
        for lead, part in spatial.query("variable == @v and primary and field == 'residual' and direction == 'north-south' and lead_hours in [3, 12, 24]").groupby("lead_hours"):
            part = part.sort_values("lag")
            cx.plot(part.lag, part.correlation_mean, marker="o", label=f"{lead} h")
            cx.plot(part.lag, part.correlation_median, linestyle=":")
            cx.fill_between(part.lag, part.correlation_min, part.correlation_max, alpha=.12)
        cx.set(xlabel="Latitude lag (grid cells)", ylabel="Spatial Pearson r")
        cx.legend()
        part = neg.query("variable == @v and primary and field == 'forecast'").sort_values("lead_hours")
        dx.plot(part.lead_hours, part.area_negative_ratio_mean * 100, marker="o", label="Mean")
        dx.plot(part.lead_hours, part.area_negative_ratio_median * 100, label="Median")
        dx.fill_between(part.lead_hours, part.area_negative_ratio_min * 100, part.area_negative_ratio_max * 100, alpha=.15)
        dx.set(xlabel="Lead (h)", ylabel="Negative forecast area (%)")
        dx.legend()
        fig.suptitle(v)
        plots = Figures(destination, config["dpi"], None, None)
        plots.save(fig, "aggregate", v)
        curves.append(destination / "aggregate" / f"{v}.png")

        fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        part = conditions.query("variable == @v and lead_hours == 12 and path == '12' and condition == 'forecast'")
        if not part.empty:
            for _, daily in part.query("init_time != 'pooled'").groupby("init_time"):
                ax.plot(daily.bin_index + 1, daily["std"], color="grey", alpha=.3)
            pooled = part.query("init_time == 'pooled'")
            ax.plot(pooled.bin_index + 1, pooled["std"], marker="o", label="Pooled standard deviation")
            ax.plot(pooled.bin_index + 1, pooled["mean"], marker="o", label="Pooled mean")
            ax.legend(fontsize="small")
        ax.set(xlabel="Common 12 h forecast bin (low to high)", ylabel=f"Residual ({unit})")
        part = path.query("variable == @v")
        if not part.empty:
            labels = part.path.astype(str)
            bx.errorbar(labels, part.rmse_mean, yerr=[part.rmse_mean - part.rmse_min, part.rmse_max - part.rmse_mean],
                        fmt="o", capsize=4, label="Mean and range")
            bx.scatter(labels, part.rmse_median, marker="x", label="Median")
            bx.legend(fontsize="small")
        bx.set(xlabel="12 h path", ylabel=f"RMSE ({unit})")
        fig.suptitle(v)
        plots.save(fig, "aggregate", f"{v}_condition_paths")
        curves.append(destination / "aggregate" / f"{v}_condition_paths.png")

        var_index = list(config["variables"]).index(v)
        with xr.open_dataset(run / "fields" / f"variable_{var_index:02d}.nc", decode_timedelta=False) as map_data:
            if 12 not in map_data.lead_time.values:
                continue
            subset = map_data.sel(lead_time=12)
            plots.lat, plots.lon = subset.latitude.values, subset.longitude.values
            dates = subset.attrs["init_times"].split()
            residual = load_fields(run, v, 12, init_times=dates)
            first = residual.residual.values[0]
            largest = rmse.query("variable == @v and primary and lead_hours == 12").loc[lambda x: x.rmse.idxmax()]
            last = residual.residual.sel(init_time=np.datetime64(largest.init_time.removesuffix("Z"))).values
            mean = subset.mean_residual.values
            limit = max(float(np.nanquantile(abs(np.concatenate([first.ravel(), last.ravel(), mean.ravel()])), .99)), 1e-30)
            maps = [(f"First init ({dates[0][:10]})", first, "RdBu_r", -limit, limit),
                    (f"Largest RMSE ({largest.init_time[:10]})", last, "RdBu_r", -limit, limit),
                    ("Mean", mean, "RdBu_r", -limit, limit),
                    ("Standard deviation", subset.std_residual.values, "magma", 0, None),
                    ("Mean absolute residual", subset.mean_absolute_residual.values, "magma", 0, None),
                    ("Top 1% frequency", subset.top_1pct_frequency.values, "viridis", 0, 1)]
            fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
            for ax, (title, values, cmap, lo, hi) in zip(axes.flat, maps, strict=True):
                im = ax.pcolormesh(plots.lon, plots.lat, values, cmap=cmap, vmin=lo, vmax=hi, shading="auto")
                ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
                fig.colorbar(im, ax=ax, label="Fraction" if title.endswith("frequency") else unit)
            fig.suptitle(f"{v} · 12 h residuals")
            plots.save(fig, "maps", v)
            curves.append(destination / "maps" / f"{v}.png")
    if not cross.empty:
        order = list(config["variables"])
        plots = Figures(destination, config["dpi"], None, None)
        for lead, table in cross.groupby("lead_hours"):
            for metric, symmetric in (("correlation_mean", True), ("correlation_std", False)):
                matrix = pd.DataFrame(index=order, columns=order, dtype=float)
                for row in table.itertuples(index=False):
                    matrix.loc[row.a, row.b] = matrix.loc[row.b, row.a] = getattr(row, metric)
                name = f"cross_variable_{int(lead)}_{metric}"
                plots.matrix(matrix.values, order, f"{int(lead)} h · {metric.replace('_', ' ')}",
                             "matrices", name, corr=symmetric)
                curves.append(destination / "matrices" / f"{name}.png")
        pairs = [("AOD", "DUAOD"), ("DUAOD", "DUSMC"), ("BCEXTTAU", "BCSMASS"),
                 ("DUSMC", "SSSMASS")]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        for ax, (a, b) in zip(axes.flat, pairs, strict=True):
            main = cross.query("(a == @a and b == @b) or (a == @b and b == @a)").sort_values("lead_hours")
            daily = daily_cross.query("(a == @a and b == @b) or (a == @b and b == @a)")
            for _, individual in daily.groupby("init_time"):
                individual = individual.sort_values("lead_hours")
                ax.plot(individual.lead_hours, individual.correlation, color="grey", alpha=.22)
            ax.plot(main.lead_hours, main.correlation_mean, marker="o", label="Mean")
            ax.plot(main.lead_hours, main.correlation_median, linestyle=":", label="Median")
            ax.fill_between(main.lead_hours, main.correlation_min, main.correlation_max, alpha=.15)
            ax.set(title=f"{a} and {b}", xlabel="Lead (h)", ylabel="Spatial Pearson r", ylim=(-1, 1))
            ax.legend(fontsize="small")
        plots.save(fig, "aggregate", "cross_variable_pairs")
        curves.append(destination / "aggregate" / "cross_variable_pairs.png")
    return curves


def findings(run):
    stats = run / "statistics"

    def bounds(file, column, query):
        frame = pd.read_csv(stats / file)
        subset = frame.query(query)
        output = []
        for variable, group in subset.groupby("variable"):
            data = group[column].dropna()
            if len(data):
                output.append(f"{variable} {data.min():.3g}–{data.max():.3g}（{len(data)} 条）")
        return "；".join(output) or "无可计算记录"

    focus = "variable in ['AOD','DUAOD','DUSMC']"
    shared = "init_time != '2024-07-26T22:30:00Z'"
    checks = [
        ("持续性技巧", "persistence_skill.csv", "skill", f"{focus} and primary and {shared}"),
        ("相邻纬向格点相关", "spatial_correlation.csv", "correlation",
         f"{focus} and field == 'residual' and direction == 'north-south' and lag == 1 and {shared}"),
        ("Top 1% 平方误差占比", "tail_statistics.csv", "contribution",
         f"{focus} and primary and fraction == 0.01 and field == 'residual' and {shared}"),
        ("预报与绝对残差相关", "condition_correlation.csv", "absolute",
         f"{focus} and primary and condition == 'forecast' and {shared}"),
        ("3+9 与 9+3 路径差异", "path_discrepancy.csv", "discrepancy",
         f"{focus} and ((a == '3+9' and b == '9+3') or (b == '3+9' and a == '9+3')) and {shared}"),
        ("DUSMC 负值面积比例", "negative_statistics.csv", "area_negative_ratio",
         f"variable == 'DUSMC' and field == 'forecast' and {shared}"),
    ]
    rows = [f"| {title} | {bounds(file, col, query)} | {file} |" for title, file, col, query in checks]
    return "\n".join(["# P1-A 研究证据", "", "下表排除与 P0 重叠的起报。区间为实际观测范围。",
        "", "| P0 现象 | P1-A 统计 | 来源 |", "|---|---|---|", *rows, "",
        "技巧为正表示优于持续性。近邻相关及尾部占比可作为 P2 的空间与极端重构参照。"
        "条件相关可能含代数耦合，路径差衡量确定性预报差异，截断结果适用于终点。"
        "多变量 Pearson 矩阵见统计表和图。七个连续起报只支持本批次的描述性分析。", ""])


def report(run, config, status, included):
    paths = figures(run, dict(config["analysis"], variables=config["variables"]))
    research = findings(run)
    atomic_text(run / "research_findings.md", research)
    links = ["config_snapshot.yaml", "metadata.json", "data_validation.md", "data_validation.json",
             "data_manifest.csv", "status.json", "research_findings.md"]
    links += [str(p.relative_to(run).as_posix()) for pattern in ("statistics/*.csv", "fields/*.nc")
              for p in sorted(run.glob(pattern))]
    date_rows = []
    for stamp, stages in status.items():
        used = pd.Timestamp(stamp).strftime("%Y%m%d_%H%M%S") in included
        cells = [stamp, "是" if used else "否"]
        cells.extend(str(stages.get(p, {}).get("state", "待运行")) for p in ("forecast", "build", "analyze"))
        cells.extend(str(stages.get(p, {}).get("error", "")) for p in ("forecast", "build", "analyze"))
        date_rows.append(cells)
    inventory = pd.DataFrame(date_rows, columns=["起报 UTC", "纳入聚合", "预测", "构建", "分析",
                                                 "预测错误", "构建错误", "分析错误"]).to_html(index=False)
    overview = pd.read_csv(run / "statistics/lead_time_statistics.csv")
    cols = ["variable", "lead_hours", "dates", "skill_days", "improved_days", "mean_daily_rmse", "pooled_rmse",
            "pooled_persistence_rmse"]
    table = overview[overview.cohort == "all_available"][cols].to_html(index=False, float_format=lambda x: f"{x:.5g}")
    parameterization = pd.read_csv(run / "statistics/aggregate_parameterization_comparison.csv")
    param_cols = ["variable", "unit", "field", "mean_mean", "centered_variance_mean", "weighted_squared_norm_mean"]
    param = parameterization.query("primary and lead_hours == 12")[param_cols].to_html(
        index=False, float_format=lambda x: f"{x:.5g}")
    clipping = pd.read_csv(run / "statistics/aggregate_clipping_comparison.csv", header=[0, 1], index_col=list(range(6)))
    clipping_summary = clipping.loc[:, [("rmse", "mean"), ("rmse", "days_decreased")]].to_html(
        float_format=lambda x: f"{x:.5g}")
    validation_path = run / "data_validation.md"
    validation = validation_path.read_text(encoding="utf-8") if validation_path.is_file() else ""
    images = []
    for p in paths:
        url = p.relative_to(run).as_posix()
        images.append(f'<figure><a href="{escape(url)}"><img loading="lazy" src="{escape(url)}" alt="{escape(p.stem)}"></a></figure>')
    assets = "".join(f'<li><a href="{escape(p)}">{escape(p)}</a></li>' for p in links)
    protocol = Path(__file__).with_name("protocol.md").read_text(encoding="utf-8")
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P1-A 多起报残差分析</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}} table{{border-collapse:collapse}}
td,th{{padding:.3rem .5rem;border:1px solid #ccc}} img{{max-width:100%;height:auto}}
pre{{white-space:pre-wrap}} figure{{margin:1rem 0}}</style><h1>P1-A 多起报残差分析</h1>
<h2>运行状态</h2>{inventory}<h2>主路径误差与持续性技巧</h2>{table}
<p>日 RMSE 均值赋予每个起报相同权重。合并 RMSE 先合计各日加权平方误差与有效面积。</p>
<details><summary>数据完整性</summary><pre>{escape(validation)}</pre></details>
<h2>跨日起报图表</h2><p>阴影为实际日期间范围。地图、条件统计、尾部曲线、路径及矩阵可在下方文件中复算。</p>
{"".join(images)}<details><summary>12 h 参数化对照</summary>{param}</details>
<details><summary>终点截断 RMSE 变化（截断后减截断前）</summary>{clipping_summary}</details>
<h2>P0 对照与研究边界</h2><pre>{escape(research)}</pre>
<details><summary>实验协议</summary><pre>{escape(protocol)}</pre></details>
<h2>数据与表格</h2><ul>{assets}</ul></html>"""
    atomic_text(run / "report.html", html)
