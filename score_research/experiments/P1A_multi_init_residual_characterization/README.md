# P1-A 多起报残差实验

[实验协议](protocol.md)说明数据来源、路径、统计口径和证据边界。每个起报单独写入一次 54 通道物理初值和预测。参考和气溶胶残差独立于模型预测，分析可以重跑。

## Linux GPU 服务器

在 Windows 只运行 `inspect` 或小型非模型检查。正式运行使用服务器已配置的环境和权重。下面的三个绝对路径可依实际部署调整。

```bash
PROJECT=/3261005002/projects/AI-GAMFS
CONDA_BASE=/3261005002/miniforge3
ENV_NAME=ai-gamfs
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT"
P1A=score_research/experiments/P1A_multi_init_residual_characterization
RUN=p1a_20240726_20240801
P0=results/P0_residual_characterization/p0_20240724
OUT=results/P1A_multi_init_residual_characterization/$RUN
git branch --show-current
nvidia-smi
free -h
df -h .
python -B "$P1A/run_batch.py" inspect --run-id "$RUN"
```

同步时将本目录、匹配的 P0 模块、`utils`、`info.pkl`、`merra_norm.csv` 和 `temp_asm` 的所需文件同步到上述项目。保留服务器原有已核验权重与 P0 缓存，避免覆盖正在运行的文件。可从具有 rsync 的客户端运行以下命令，先根据站点设置 `GAMFS_SERVER`。

```bash
rsync -avR "$P1A/" score_research/experiments/P0_residual_characterization/ \
  utils/datasets.py utils/init.py utils/handle.py utils/scheduler.py info.pkl merra_norm.csv \
  "$GAMFS_SERVER:$PROJECT/"
```

如果服务器还没有所需 GEOS-FP 和 MERRA-2 文件，按本地 `data_manifest.csv` 中 `required=True` 的路径同步缺件，维持 `temp_asm` 目录结构。已有两个同日 GEOS-FP 扩展名副本时应先检查来源并保留单一可解析输入。权重需放在配置的 `forecast.model_dir`，文件名为 `gamfs_{3,6,9,12}h_traced.pt`。

先验收一个新增日期，然后完整运行并复用已成功日期。

```bash
python -B "$P1A/run_batch.py" all --run-id "$RUN" --resume --init 2024-07-27T22:30:00Z
python -B "$P1A/run_batch.py" all --run-id "$RUN" --resume --p0-cache "$P0"
```

`--p0-cache` 会尝试导入 7 月 26 日的 14 个归一化前缀。实验时间、路径、精度、运行环境、实现哈希、归一化及数据身份需要匹配，且全部终点须逐格复原 P0 保存的三个变量。核对源码后允许本仓库两项特定哈希迁移，分别对应文件发现与参考索引的变更，以及原始 CLI 的独立读取器变更。旧元数据缺少输入和权重 SHA-256 时，必须使用已记录的原绝对路径、字节数和纳秒修改时间核对。因此 Windows 新命名副本或服务器迁移件可能触发重新推理。接受或拒绝的原因见日分片元数据和日志。`--hash-sources` 可给本次输入和权重记录内容指纹。服务器端冒烟需确认模型输出有限、已写入 54 通道、数据集检查和报告均完成。

```bash
# 可独立续跑一个阶段。build 使用已完成的物理预测。
python -B "$P1A/run_batch.py" forecast --run-id "$RUN" --resume --p0-cache "$P0"
python -B "$P1A/run_batch.py" build --run-id "$RUN" --resume
python -B "$P1A/run_batch.py" analyze --run-id "$RUN" --resume
python -B "$P1A/run_batch.py" aggregate --run-id "$RUN" --resume
python -B "$P1A/run_batch.py" report --run-id "$RUN" --resume

# 后台启动、查看日志及同步结果
nohup python -B "$P1A/run_batch.py" all --run-id "$RUN" --resume --p0-cache "$P0" > "$RUN.nohup.log" 2>&1 &
tail -f "$RUN.nohup.log"
# 从接收结果的客户端执行
rsync -av "$GAMFS_SERVER:$PROJECT/$OUT/" "$OUT/"
```

异常退出后查看日分片 `status.json`。确认没有对应进程仍在运行时可移除孤留的 `.running` 文件，再以 `--resume` 重启。`--refresh-forecast` 忽略已有前缀并重跑指定日期预测。手工调整实验日期、映射或配置时使用新运行 ID，额外气象残差需先验证对应产品和映射。分析代码修改可以用原运行 ID 重跑 `analyze → aggregate → report`。

## 容量估算

七天主预测 56 条，另有 28 条路径记录。共享前缀后最多 98 次前向调用。原始必需 GEOS-FP 与 MERRA-2 数据共约 13.7 GB，权重约 19.4 GB。按 207,936 格点、FP32 估算，预测及初值净数组约 4.1 GB，完整前缀缓存约 4.4 GB，参考、两个残差与掩码约 2.45 GB。单个预测分片原子写前的未压缩上限约 0.58 GB。上述为压缩前十进制容量，额外预留 15 GB 供分片、分析产物及临时文件使用。

P0 日志中单日起报的 forecast 阶段约 34 秒，按七个冷启动起报外推约 4 分钟，仅作为 P0 三通道预测的量级参照。P1-A 增加全通道写盘、读取及分析，单日期服务器冒烟用于测量实际耗时及峰值内存。

## 产物与后续读取

- `data_manifest.csv`、`data_validation.*` 和 `status.json` 记录切实检查的文件、时间、通道和运行范围。
- `per_init/*/data/forecasts.nc` 含 54 通道物理初值及每条路径的预测。每日起报的 `model_calls.json` 保留逐模型调用与时间条件。按日拆分便于续跑和 P1-B 扩展。
- `per_init/*/residuals.nc` 含 12 气溶胶残差、共用掩码和去重参考索引。持续性预报使用同片物理初值。
- `statistics/` 含逐日与聚合指标、条件、空间、多变量、路径、尾部和非负对照。 `fields/` 保存多日地图和热点数的统计支持。 `report.html` 和 `research_findings.md` 可离线查看。
- `dataset_interface.load_fields(run, "AOD", 12)` 返回按起报索引的物理场、掩码、有效时间、来源和单位。传入 `field_type` 可取参考、持续性残差、预报或初值，`path="3+9"` 选择路径，`init_times`、`start`、`end` 控制日期选择。
- P1-B 按同一协议扩展季节和起报日期。P2 按独立日期块选择变量、时效和参数化，训练集估计面积加权基底，在独立测试上评价全球及尾部重构。P3 在相同划分构建确定性及简单概率残差基线，供 P4 条件扩散比较。
