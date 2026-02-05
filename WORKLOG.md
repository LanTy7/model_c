# 工作与实验日志（按时间追加）

说明：本文件用于记录每一次“会影响结果复现/对比”的代码改动与实验结果，便于后续写论文、做 ablation、做对比实验时快速回溯。

## 2026-02-05 06:30 多分类 v1.1（数据解析与评估可靠性增强）

- 目标：修复多分类标签解析导致的数据丢失；减少路径踩坑；提升训练/评估的可信度与可复现性。
- 变更：
  - `multi/model_train/train.ipynb`
  - 标签解析：由 `parts[3]` 改为统一取 `parts[-1]`，并增加 `skipped` 计数日志（避免静默丢数据）。
  - 路径：`fasta_file` 改为“向上查找 data/ 目录”的自动解析（避免 cwd 不同导致相对路径错误），并保留强制存在性检查。
  - 模型：将 Global Pooling 改为 Masked Pooling（利用 one-hot 的 PAD 通道 index=20 生成 mask，避免 PAD 污染 max/mean pooling）。
  - 评估：新增 `macro-F1`（`sklearn.metrics.f1_score(average='macro')`）；Early Stopping 与最佳模型保存依据由 `val_acc` 改为 `val_macro_f1`；评估报告中打印 Macro-F1；训练曲线图增加 Val Macro-F1 曲线。
  - 产物保存：图文件名由固定 `training_results.*` 改为以模型文件名为前缀（`{run_tag}_training_results.*`），避免不同实验覆盖。
- 数据：
  - `data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta`
  - 读取结果：`Loaded 17345 sequences (skipped 0 due to header parse)`
- 训练结果（一次完整训练）：
  - `min_samples=50`，合并后类别数 `num_classes=14`
  - `overall accuracy=0.9752`，`macro-f1=0.9514`
  - `Others`：precision=0.70，recall=0.67，f1=0.69
  - 模型：`multi/model_train/well-trained/bilstm_multi_20260205_1316.pth`
  - 图：`multi/model_train/figures/bilstm_multi_20260205_1316_training_results.png` / `.pdf`
- 下一步：
  - 在统一评估协议下扫 `min_samples`（已完成 20/30/40/50），并选择论文写作/对比实验采用的类别体系。

## 2026-02-05 07:10 多分类 `min_samples` 扫参（20/30/40/50）

- 目标：评估“类别粒度（num_classes）与指标”的关系，避免只追求数字导致分类体系过粗。
- 配置：仅修改 `TRAIN_CONFIG['min_samples']`，其余保持一致。
- 结果汇总：
  - `min_samples=20`：num_classes=19；overall acc=0.9694；macro-f1=0.8753；Others P/R/F1=0.64/0.69/0.67
  - `min_samples=30`：num_classes=17；overall acc=0.9746；macro-f1=0.8815；Others P/R/F1=0.83/0.79/0.81
  - `min_samples=40`：num_classes=15；overall acc=0.9729；macro-f1=0.9312；Others P/R/F1=0.69/0.54/0.61
  - `min_samples=50`：num_classes=14；overall acc=0.9752；macro-f1=0.9514；Others P/R/F1=0.70/0.67/0.69
- 备注：
  - 不同 `min_samples` 会改变任务定义（类别体系变化），因此各设置下的指标不能当作“同一任务下”的严格可比结论。
  - `min_samples=30` 的 `Others` 表现相对最好（F1=0.81），适合作为后续继续优化的默认设定候选。
- 下一步：
  - 固化一个论文/对比实验用的类别体系（建议先以 `min_samples=30` 继续迭代），再推进更严格的泛化评估（例如按数据库来源做 holdout）。

