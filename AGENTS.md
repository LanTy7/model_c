# 项目导航（ARG 识别与分类）

本文件定位为“指挥官/地图”：只保留**重要入口、关键约定、当前状态摘要、下一步路线**。细节分别解耦到独立文档中。  
约定：**本仓库内协作文档新增/更新一律使用中文。**

## 目标（一句话版）

对全长氨基酸序列进行 ARG 二分类识别，并对 ARG 进行多分类定类；追求论文级可复现与强泛化表现。

## 快速入口（你要去哪儿看什么）

- 训练代码：
  - 二分类：`binary/model_train/train.ipynb`
  - 多分类：`multi/model_train/train.ipynb`
- 推理代码：
  - 二分类：`binary/model_test/predict.ipynb`
  - 多分类：`multi/model_test/classify.ipynb`
- 主要数据：
  - ARG 数据库：`data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta`
  - 数据说明：`data/database_construct_description.md`
- 工具脚本：
  - 阴性长度匹配采样：`scripts/sample_len_matched_neg.py`
  - 清理 `.faa` 末尾 `*`：`scripts/clean_faa_strip_stop.py`
  - silver standard 评估（B）：`scripts/eval_silver_standard_v0.py`
- 评估方法学（论文可复现协议）：`EVAL_PROTOCOL.md`
- 工作与实验日志（按时间追加）：`WORKLOG.md`

## 关键约定（影响复现与结论）

- 任何影响复现/对比的变更（代码、数据处理、训练配置、评估协议）必须追加记录到 `WORKLOG.md`。
- 真实场景测试：必须区分 `case study` 与“可量化真实性能评估”；细则见 `EVAL_PROTOCOL.md`。
- 多分类允许将极小类合并为 `Others`（`min_samples`），但需要在论文中明确类别体系与阈值。
- 环境：训练在 conda `dl_env` 运行；最终模型确定后再导出环境文件。

## 当前状态摘要（我们已经做到了什么）

- 多分类训练已完成关键可靠性修复与增强（细节与指标见 `WORKLOG.md`）：
  - 修复 label 解析（兼容不同 FASTA header 风格，避免静默丢数据）。
  - 引入 masked pooling（避免 PAD 污染 pooling）。
  - 引入 `Macro-F1` 并用于早停与最佳模型保存。
  - 图/产物按 run 命名保存，避免覆盖。
- 已完成 `min_samples` 扫参（20/30/40/50），用于评估“类别粒度 vs 指标”的关系（详细结果见 `WORKLOG.md`）。

## 下一步路线（优先级从高到低）

1. 固化论文/对比实验采用的类别体系（确定 `min_samples` 与最终 `num_classes`），并在后续迭代中保持一致。
2. 按 `EVAL_PROTOCOL.md` 落地真实性能评估：
  - 给全部 ORF 构建 silver standard 参考标签，统计假阴性/真召回。
  - 做同源泄漏控制（`seen-like` vs `novel-like` 或按来源 holdout）。
3. 二分类阴性集流水线（论文可写）：
  - 阴性去污染 + 长度分布匹配 + hard negatives。
