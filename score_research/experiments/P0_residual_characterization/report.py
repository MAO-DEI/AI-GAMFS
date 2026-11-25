"""Build one offline HTML index of measured tables, plots and provenance."""
from html import escape
import pandas as pd
import xarray as xr

SECTIONS = [
    ("P0.1 残差统计与分布", ["residual_statistics"], ["distribution"]),
    ("P0.2 空间结构", ["spatial_correlation"], ["spatial", "autocorrelation"]),
    ("P0.3 时效变化", ["lead_time_statistics", "cross_lead_correlation"], ["lead_time"]),
    ("P0.4 接力路径", ["path_statistics", "path_discrepancy"], ["path_comparison"]),
    ("P0.5 非负性与终端截断", ["negative_statistics", "negative_by_reference", "clipping_comparison"], ["negatives"]),
    ("P0.6 持续性与三类预测目标", ["persistence_comparison"], ["targets"]),
    ("P0.7 条件依赖性", ["conditional_statistics"], ["conditions"]),
]


def report(run, analysis, config, source):
    def link(path, name=None):
        return f'<a href="{escape(path.relative_to(run).as_posix(), quote=True)}">{escape(name or path.name)}</a>'

    with xr.open_dataset(run / "data" / "residuals.nc", engine="netcdf4", decode_timedelta=False) as ds:
        structure = str(ds)
        paths = ds[["lead_time", "path", "valid_time", "initial_index"]].coords.to_dataset()
        # Select record coordinates only, avoiding a Cartesian product with grid coordinates.
        schedule = pd.DataFrame({key: paths[key].values for key in
                                 ("lead_time", "path", "valid_time", "initial_index")}).to_html(index=True)
    protocol = (source / "protocol.md").read_text(encoding="utf-8-sig")
    evidence = (source / "claims_and_evidence.md").read_text(encoding="utf-8-sig")
    title = config["experiment"]["name"]
    sections = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        f"<title>{escape(title)}</title>",
        """<style>body{font:16px system-ui,sans-serif;margin:auto;max-width:1250px;padding:24px;color:#192334}
        table{border-collapse:collapse;font-size:13px}td,th{padding:5px;border:1px solid #d7dfe5}
        img{width:min(100%,740px);height:auto} details{margin:10px 0}
        .scroll,pre{overflow:auto;background:#f5f7f9;padding:10px}figure{display:inline-block;margin:10px}</style>""",
        f"<h1>{escape(title)}</h1><h2>实验配置与数据核验</h2>",
    ]
    for name in ("config_snapshot.yaml", "metadata.json", "data_validation.json", "model_calls.json", "run.log"):
        path = run / name
        if path.is_file():
            sections.append(f"<p>{link(path)}</p>")
    sections.append("<h2>模型路径与数据集</h2>" + schedule)
    for path in sorted((run / "data").glob("*.nc")):
        sections.append(f"<p>{link(path)}</p>")
    sections.append(f"<details><summary>NetCDF 结构</summary><pre>{escape(structure)}</pre></details>")
    sections.append(f"<details><summary>实验协议与待检验假设</summary><pre>{escape(protocol)}</pre></details>")
    sections.append("<h2>计算结果</h2><p>表内仅预览前 12 行，完整记录请查看 CSV。图名中的 r 对应记录编号，i 对应起报索引，v 对应变量索引。各分布采用普通网格计数。</p>")
    for name, tables, groups in SECTIONS:
        sections.append(f"<h3>{escape(name)}</h3>")
        produced = False
        for table in tables:
            path = analysis / "statistics" / f"{table}.csv"
            if path.is_file():
                produced = True
                preview = pd.read_csv(path, nrows=12).to_html(index=False, na_rep="—")
                sections.append(f"<details><summary>{escape(table)}</summary><p>{link(path)}</p><div class='scroll'>{preview}</div></details>")
        for group in groups:
            images = sorted((analysis / "figures" / group).glob("*.png"))
            if images:
                produced = True
                sections.append(f"<details><summary>{escape(group)} ({len(images)})</summary>")
                for image in images:
                    rel = escape(image.relative_to(run).as_posix(), quote=True)
                    sections.append(f'<figure><a href="{rel}"><img loading="lazy" src="{rel}" alt="{escape(image.stem)}"></a>'
                                    f'<figcaption>{escape(image.stem)}</figcaption></figure>')
                sections.append("</details>")
        if not produced:
            sections.append("<p>本次运行没有生成此小节的产物。</p>")
    sections.extend([
        "<h2>解释边界与后续验证</h2>",
        "<p>图表记录本次起报的可观测统计。可能的因素包括产品和平均窗差异、气溶胶输送、路径以及数值误差。"
        "本次分析无法归因，也无法估计长期概率分布或校准。下列研究假设和方法选择需要独立样本或建模对照验证。</p>",
        f"<details open><summary>研究证据、方法设计和下一步</summary><pre>{escape(evidence)}</pre></details></html>",
    ])
    target = run / "report.html"
    temp = target.with_suffix(".html.tmp")
    temp.write_text("\n".join(sections), encoding="utf-8")
    temp.replace(target)
