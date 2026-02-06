# 模型真实场景测试与验证协议（论文可复现版本）

本文档用于定义“如何在真实数据上测试模型”的标准流程，确保结果可复现、可量化、可写入论文方法学部分。

## 约定

- 本文档描述的是“论文方法学级别”的评估协议：阈值、规则与分组一旦确定，后续对比实验应保持一致。
- 如需调整阈值（identity/coverage/e-value 等），必须在 `WORKLOG.md` 记录，并在论文中明确说明。

## 阈值版本（silver standard）

### v0（高置信，默认推荐）

- strict ARG：
  - `evalue <= 1e-10`
  - `pident >= 80`
  - `min(qcov, scov) >= 0.80`
  - 多分类参考：取 strict ARG 命中的 best-hit（推荐 `bitscore` 最大）
- strict non-ARG：
  - relaxed 阈值下完全无命中才标 `non-ARG`
  - relaxed：
    - `evalue <= 1e-5`
    - `pident >= 30`
    - `min(qcov, scov) >= 0.50`
- 其它：
  - `unlabeled`（建议主评估先 exclude；可另做敏感性分析将其当作 non-ARG）

## 1. 测试目标与输入

- 目标：在真实 contig/基因组片段上，经 ORF 预测得到的蛋白序列（`.faa`）中，完成：
  - 二分类：是否为 ARG
  - 多分类：属于哪个 ARG 类别
- 输入：
  - 一批 `.faa` 文件（来自 `contig -> Prodigal -> ORF proteins`）
  - 参考 ARG 数据库（用于“参考标签/验证”）：建议使用与训练标签体系一致的 ARG 库（例如本项目的 `data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta`）。

## 2. 两类评估：应用示例 vs 真实性能评估

必须在论文中区分清楚：

### 2.1 应用示例（case study）

- 流程：`.faa` -> 二分类筛选 ARG -> 多分类定类 -> 对“预测为 ARG 的序列”做同源比对验证。
- 结论类型：更适合展示“模型在真实数据上的输出长什么样”“能发现什么候选 ARG”，偏展示与解释。
- 局限：若只对预测为 ARG 的序列做比对，无法统计假阴性（漏检），因此不能充分反映真实召回率。

### 2.2 真实性能评估（推荐作为主结果）

- 核心要求：必须先给 **全部 ORF** 构建“参考标签（silver standard）”，再与模型输出做完整对比。
- 这样才能同时得到：
  - 真阳性/假阳性/真阴性/假阴性（binary 完整混淆矩阵）
  - 多分类的 macro-F1、每类 F1、混淆矩阵等

## 3. 参考标签（silver standard）构建

由于真实数据通常没有手工真值，建议用“高置信同源比对”作为参考标签，并在论文中明确阈值与规则。

### 3.1 比对工具（任选其一，需在论文写明）

- `DIAMOND blastp`（速度快，适合大规模 ORF）
- `BLASTP`（经典但慢）
- `MMseqs2`（速度与灵活性强）

### 3.2 推荐“高置信”命中阈值（示例，需按数据规模/目标调整）

建议用相对严格阈值来定义参考阳性，避免把远缘同源误当 ARG：

- `e-value <= 1e-5`
- `pident >= 80`（或 70；严格度在论文中说明）
- 覆盖度（至少一个）：
  - `qcov >= 0.8`（alignment_length / query_length）
  - 或 `scov >= 0.8`（alignment_length / subject_length）

说明：

- ORF 可能是截断/不完整蛋白。为兼顾截断情况，可在主结果使用严格阈值（高置信），并在补充材料增加“宽松阈值”的敏感性分析。

### 3.3 参考标签赋值规则（必须写清楚）

对每条 ORF（query）：

1. 若不存在高置信命中：参考标签为 `non-ARG`（或 `unlabeled`；推荐区分 `unlabeled` 以避免把“未知/远缘”硬当阴性）。
2. 若存在高置信命中：
  - 参考二分类标签：`ARG`
  - 参考多分类标签：取最佳命中（best-hit）的 ARG 类别（类别来自参考库序列 header 的类别字段）。
3. 若出现多个高置信命中但类别冲突且无法判定：
  - 标记为 `ambiguous`（从严格多分类评估中剔除或单独统计）

## 4. 防止训练-测试同源泄漏（强烈建议）

如果测试 ORF 与训练集序列高度同源，随机切分会显著高估性能。建议至少做一项：

- “同源去重/去泄漏”：
  - 用同源比对把测试 ORF 与训练集 ARG 库进行比对；
  - 将 `pident` 与覆盖度高于某阈值（例如 `>= 90%` 且覆盖 `>= 90%`）的 ORF 从测试集中剔除或单独报告为 `seen-like` 子集；
  - 主结果在 `novel-like` 子集上报告（更能反映泛化）。
- “按来源数据库 holdout”（若适用）：
  - 按数据库来源（如 sarg/ncbi/megares/card/resf）拆分训练与测试，测试仅使用 holdout 来源的数据（更严格的泛化评估）。

## 5. 指标与报告（建议写入论文）

### 5.1 二分类（ARG vs non-ARG）

- PR-AUC、ROC-AUC、F1、Precision、Recall
- 建议同时报告在不同阈值下的 Precision-Recall 曲线，并说明最终阈值选择策略（例如在验证集上最大化 F1 或在保证 Recall 的前提下最大化 Precision）。

### 5.2 多分类（ARG 类别）

- Macro-F1（核心）
- Weighted-F1（辅助）
- 每类 Precision/Recall/F1 + 混淆矩阵
- 对 `Others`/长尾类单独讨论（避免“垃圾桶类”吞噬长尾后指标虚高）。

### 5.3 端到端（binary -> multi）

建议额外报告端到端效果：

- 端到端正确：ORF 被判为 ARG 且类别也正确
- 端到端错误类型分解：
  - 漏检（参考为 ARG，但二分类预测 non-ARG）
  - 误检（参考为 non-ARG，但二分类预测 ARG）
  - 定类错（二分类对，但多分类类别错）

## 6. 结果呈现建议（论文写作友好）

- 主表：在“固定参考标签定义”下的整体指标（binary + multi）
- 子表：按 ORF 长度分段（例如 <200aa、200-400aa、>400aa）报告，体现对截断 ORF 的鲁棒性
- 泛化表：`seen-like` vs `novel-like`（或 holdout 数据库）对比
- 案例展示：列出若干高置信预测、低置信预测、无同源但高分的候选（作为潜在新 ARG 的讨论点）

## 7. 实施注意事项

- 参考库与类别体系必须与训练体系对齐（否则类别映射会引入额外误差）。
- 所有阈值（identity/coverage/e-value/bitscore/top-hit 规则）必须在论文中明确。
- 若将 `unlabeled` 直接当 `non-ARG`，会低估模型的“发现新 ARG”能力；建议把 `unlabeled` 作为单独集合分析。
