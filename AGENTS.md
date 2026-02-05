# 项目协作备忘（ARG 识别与分类）

本文件用于记录我们双方已对齐的项目背景、当前仓库状态、已发现的问题与修复优先级，以及后续迭代计划（例如 v1.0 -> v1.1 -> v1.2）。  
约定：**后续往本文件新增/更新内容一律使用中文。**

## 目标

- 任务A（二分类）：对全长氨基酸序列进行 ARG 识别（是否为 ARG）。
- 任务B（多分类）：对识别出的 ARG 序列进行类别预测（属于哪个抗性基因类别）。
- 目标：达到科研论文级别的高性能，并能在真实测试/对比实验中站得住脚。

## 当前仓库结构

- `binary/model_train/train.ipynb`：二分类训练（BiLSTM）。
- `binary/model_test/predict.ipynb`：二分类推理（从输入文件夹筛出预测为 ARG 的序列并输出 FASTA）。
- `multi/model_train/train.ipynb`：多分类训练（BiLSTM）。
- `multi/model_test/classify.ipynb`：多分类推理（对预测为 ARG 的序列进行类别分类）。
- `data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta`：整理好的 ARG 数据集（用于多分类；也可作为二分类阳性集）。
- `data/database_construct_description.md`：数据集构建说明（CARD/AMRFinder/SARG/MegaRes/ResFinder；去 SNP；MMseqs2 100% identity 去冗余；最终 30 类）。

## 数据集要点（基于本地快速扫描 `data/*.fasta`）

- 文件：`data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta`
- 序列数：17,345
- 类别数：30（类别名位于 FASTA header 的最后一个 `|...|label` 字段）
- 类别分布严重不均衡（头部大类包括 `beta-lactam`、`multidrug`、`MLS` 等）
- 序列长度分布（约）：p95 ~ 657 aa；最大长度 ~ 1576 aa
- 小样本类别很多：有 17 个类别样本数 <= 50

## 当前模型实现（v1.0）

### 二分类（ARG vs non-ARG）

- 输入：索引编码，PAD=0，X=21，`vocab_size=22`；固定 `max_length=1000`。
- 模型：Embedding -> BiLSTM ->（全局 max pooling + 全局 avg pooling）-> MLP -> 输出 1 个 logit。
- 损失函数：`BCEWithLogitsLoss(pos_weight=neg/pos)`。
- 阴性样本：按 `pos_neg_ratio=3` 抽样（注意：当前阴性 FASTA 尚未放入本仓库）。
- 早停：监控 `val_loss`。
- 学习率调度：`ReduceLROnPlateau`（监控 `val_loss`）。

### 多分类（ARG 类别）

- 输入：one-hot（20 种氨基酸 + PAD）；B/Z/J 做均分；X/未知字符做均匀分布。
- 模型：BiLSTM ->（全局 max pooling + 全局 avg pooling）-> MLP -> logits。
- 损失函数：Focal Loss + Label Smoothing + 类别权重（clamp）。
- 学习率调度：warmup + cosine（按 batch 更新）。
- 早停：监控 `val_acc`。
- 备注：当前代码允许通过 `min_samples` 将极少数类别合并为 `Others`。

## 已知关键问题（在相信指标前必须修复）

1. Notebook 中存在硬编码数据路径，指向其他机器/目录。
  - 风险：可能在“你以为训练A数据，但实际读到B数据”，或者在干净环境下无法复现运行。
2. 多分类标签解析存在缺陷（会静默丢数据）。
  - 当前逻辑对 `record.description` 以 `|` 分割并取 `parts[3]`。
  - 部分 header（例如 SARG 风格）字段数更少，会被直接跳过，导致训练集变小且类别分布变形。
  - 建议修复：类别统一取最后字段 `parts[-1]`。
3. 运行环境不一致（终端与 notebook 环境不一致）。
  - 终端当前 `python` 缺少关键依赖（例如 `numpy`、`biopython`）。
  - 实际训练使用 conda 环境 `dl_env`（暂不导出，等最终模型确定再导出）。

## 已对齐的决策

- 二分类阴性集：
  - 当前阴性集来源于抽取的细菌氨基酸序列（阴性文件尚未纳入本仓库）。
  - 方向可行，但必须“更干净、更难”，否则验证指标可能虚高且泛化不稳。
- 多分类：
  - 允许将极小类合并为 `Others`。
  - 注意：若 `min_samples=50`，此数据上可能会把 30 类中的 17 类合并进 `Others`；可考虑更小阈值（如 10 或 20）以保留细粒度能力。
- 环境导出：
  - 等最终模型训练完成后再导出 `dl_env`（例如 `environment.yml`/`requirements.txt`）。

## v1.1 推荐路线（高收益、低风险）

1. 修复多分类 label 解析，避免静默丢数据。
2. 将硬编码路径替换为仓库相对路径默认值（或清晰的 `PATH_CONFIG` 覆盖方式）。
3. 构建可写进论文的二分类阴性集流水线：
  - 阴性去污染：过滤与已知 ARG 同源的序列（例如 identity >= 30% 且 coverage >= 70%，或更严格）。
  - 阴阳性长度分布匹配：避免模型走“长度捷径”。
  - 加入 hard negatives：选择更容易混淆的非 ARG 蛋白（同样需先去污染）。
4. 固化评估协议（论文可信度）：
  - 分层划分、固定随机种子。
  - 二分类报告 AUC/PR-AUC/F1 等；多分类报告 macro-F1/更平衡的指标，并给出混淆矩阵等分析。

