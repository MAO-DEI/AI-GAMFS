# P0 单起报残差实验

[协议](protocol.md) · [本地数据核验](data_validation.md) · [论文主张与证据](claims_and_evidence.md) · [开发报告](development_report.md)

**服务器开跑前请准备 2024-07-24 22:30 UTC 三类正确 GEOS-FP 文件。** NetCDF 解码时间、变量、单位及网格均须通过检查。本地所见 7 月 26 日资料无法用于本配置，其中通量文件内容错误。

## 环境与同步

本实验使用 Python 3.11+、原项目 `requirements.txt` 的库和现有 CUDA PyTorch。使用服务器上已可运行该 TorchScript 的环境。默认一次只驻留一套模型，每套权重约 4.85 GB，设备仍需容纳计算激活。确认 `nvidia-smi`、可用 GPU 显存与结果磁盘空间。

从本地项目根目录执行（把 SSH 目标和远程项目路径改成实际值）。

```bash
ssh USER@HOST 'mkdir -p /3261005002/projects/AI-GAMFS/score_research/experiments'
scp -r score_research/experiments/P0_residual_characterization USER@HOST:/3261005002/projects/AI-GAMFS/score_research/experiments/
```

在服务器执行以下命令。权重和数据沿用已有文件，代码本身不会下载。

```bash
ROOT=/3261005002/projects/AI-GAMFS
source /3261005002/miniforge3/etc/profile.d/conda.sh
conda activate ai-gamfs
cd "$ROOT"
P0=score_research/experiments/P0_residual_characterization
CLI="$P0/run_experiment.py"
CFG="$P0/config.yaml"
RUN=p0_20240724
OUT="results/P0_residual_characterization/$RUN"
ls -lh model/gamfs_{3,6,9,12}h_traced.pt info.pkl merra_norm.csv
find temp_asm -type f \( -name '*20240724_2230*' -o -name 'MERRA2_400.tavg1_2d_aer_Nx.20240725.nc4' \)
python -B "$CLI" inspect --root "$ROOT" --config "$CFG" --run-id "$RUN"
```

应有 `GEOS.fp.asm.tavg3_2d_aer_Nx`、`GEOS.fp.asm.tavg1_2d_flx_Nx` 和 `GEOS.fp.asm.tavg3_3d_asm_Nv` 各一份。文件名格式为 `<product>.20240724_2230.V01.hdf`，也接受 `.nc4` 与 `.nc`。同一产品放在 `temp_asm` 平铺目录或同名子目录。参考文件路径在配置中独立指定。inspect 写出运行目录内的 `data_validation.json` 和 `status.json`，失败返回非零。

## 正式运行、恢复与重分析

```bash
python -B "$CLI" all --root "$ROOT" --config "$CFG" --run-id "$RUN" --resume
```

后台运行改用下列命令，前台与后台示例任选一个。日志重定向文件写在项目目录，运行器同时写入 `$OUT/run.log`。

```bash
nohup python -B "$CLI" all --root "$ROOT" --config "$CFG" --run-id "$RUN" --resume \
  > "${RUN}.nohup.log" 2>&1 &
tail -f "${RUN}.nohup.log"
cat "$OUT/status.json"
```

`forecast`、`build`、`analyze`、`report` 均可用相同参数单独运行。先看 `status.json` 和 `run.log` 定位失败阶段，修正源文件或环境后用 `all --resume` 恢复。全程同一 run id 只允许一个进程操作。若进程被强制终止导致 `.running` 残留，先核对其中 PID、服务器进程及是否存在其它活跃运行，再手动移除该锁。

```bash
# 用已有预报强制重做数据集与分析，并生成报告
python -B "$CLI" build --root "$ROOT" --config "$CFG" --run-id "$RUN" --overwrite
python -B "$CLI" analyze --root "$ROOT" --config "$CFG" --run-id "$RUN" --overwrite
python -B "$CLI" report --root "$ROOT" --config "$CFG" --run-id "$RUN" --overwrite
```

- 默认拒绝重用已有运行目录。`--resume` 复核配置、阶段状态和产物身份，预测可从完整前缀恢复。`--overwrite` 强制重算所选阶段。重算上游会把下游标为 stale，报告以最新完成的分析为准。
- 预测缓存依据配置、起报文件和权重的绝对路径、大小、纳秒修改时间、软件版本、Git commit 和相关代码摘要分区。启用 `--hash-sources` 可增加输入和权重 SHA-256，同一运行的后续恢复应保持该标志一致。重试需要绕开已有前缀时用 `--refresh-cache`。前缀文件存完整归一化状态，约 45 MB/个。默认案例约需 1 GB 量级的结果及缓存空间，图表数量和额外起报会增加占用。
- 执行 `build` 需要已验收的 `forecast` 产物和当前参考文件，读取模型输出无需加载权重。每次 `analyze` 会产生独立子目录，生成报告时按完成标记定位。
- 配置变更使用新运行 ID。加起报时增补 `experiment.init_times` 和参考日文件。可把不同日期拆为多个运行以控制内存。本工具按运行读入分析字段，长期数据建议按起报或短时段分片。加变量时在 `variables` 填模型通道名和参考变量名，当前只接受可对齐坐标与等价物理单位。自定义主路径可将 `primary_strategy` 写成时效到步长列表的映射。所有路径步长必须来自 3/6/9/12，小时时效配置可延长。
- `initial_precision: fp32_direct` 是独立精度敏感性运行，使用新运行 ID。主实验保留原始入口的 `fp16_roundtrip`。
- 时间报错时查看 NetCDF 解码时刻及是否有重复参考文件。通量缺变量时核对产品内容，尤其是已发现的 7 月 26 日误标文件。坐标或单位不匹配时查证上游数据，框架只接受等价网格重排及等价单位。CUDA OOM 时先确保 `resident_models: 1` 并核对可用显存，换设备或驻留策略须用新运行 ID。

## 产物与下载

`data/forecasts.nc` 保存全记录的物理目标预测和起报场，`data/residuals.nc` 另含参考、持续性、两类残差及掩码，`data/path_comparison.nc` 是对照子集。同一起报与有效时间的路径共享掩码。每条预测通过 `initial_index` 关联起报。配置、Git、软件版本、权重标识、源文件、精度、路径与模型调用见运行元数据及 NetCDF 属性。分析子目录包含 CSV、图和完成标记，运行根目录的 `report.html` 通过相对路径引用这些文件。

从本地执行以下命令，保留目录结构后可离线打开 HTML。

```bash
scp -r USER@HOST:/3261005002/projects/AI-GAMFS/results/P0_residual_characterization/p0_20240724 .
```
