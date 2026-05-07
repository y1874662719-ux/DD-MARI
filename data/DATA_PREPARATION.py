"""Instructions for preparing ASAP Set 1 data for DD-MARI.
/ ASAP Set 1 数据准备说明

Source / 数据来源
------
Download the ASAP dataset from Kaggle:
  https://www.kaggle.com/c/asap-aes/data

You need: training_set_rel3.tsv  (the main competition file)

Preparation Steps / 准备步骤
-----------------
1. Filter for Set 1 (essay_set == 1) / 筛选 Set 1（essay_set == 1）

2. Split into three Excel files / 拆分为三个 Excel 文件:

   asap_set1_gradient.xlsx   — 10–20 quality-stratified essays
     Select 1–2 essays per score level to cover the full 2–12 range.
     Used for Spearman-guided cold-start initialization.
     每个分段选取 1–2 篇，覆盖 2–12 的完整分数范围。
     用于 Spearman 引导的冷启动初始化。

   asap_set1_training.xlsx   — 40 training samples
     - 10 gold-standard essays (high scores: 10–12)
     - 30 hard-negative essays (score within ±2 of a paired gold essay)
     Add a column 'sample_type': use 'gold' for the 10 high-quality essays,
     'negative' for the 30 hard negatives.
     Alternatively, prefix essay_id with 'orig_' or 'neg_' for auto-detection.
     10 篇金标准样本（高分：10–12）+ 30 篇困难负样本（分差 ≤ 2）。
     添加 'sample_type' 列：金标准填 'gold'，负样本填 'negative'。
     也可在 essay_id 前缀 'orig_' 或 'neg_' 供框架自动识别。

   asap_set1_test.xlsx       — remaining essays (~1,762)
     All essays not in the training or gradient sets.
     不在训练集或校准集中的所有样本（约 1,762 篇）。

3. Place Set1_standard.docx (the ASAP scoring rubric document) in this folder.
   将 Set1_standard.docx（ASAP 评分标准文档）放入本目录。

Required Excel Columns / 必需列
----------------------
  essay_id       (string / 字符串)
  essay          (string, full essay text / 字符串，完整作文文本)
  domain1_score  (integer, ASAP combined score 2–12 / 整数，ASAP 综合分 2–12)

Score Convention / 分数约定
----------------
DD-MARI uses a 2–12 scale matching the ASAP combined score
(sum of two raters, each 1–6).  Store the combined score directly —
NO conversion is needed.  Example: an essay scored 4+5 by two raters
should be stored as domain1_score = 9.

DD-MARI 使用与 ASAP 综合分一致的 2–12 分制
（两位评分者各 1–6 分之和）。直接存储综合分，无需转换。
例如：两位评分者分别给 4 分和 5 分，则存储 domain1_score = 9。

Quick Start / 快速开始
-----------
If your source data is already in the original project folder, run:
如果源数据已在原项目文件夹中，直接运行：

    python prepare_data.py

This script reads the original data files and produces all required
Excel files automatically.
该脚本自动读取原始数据文件并生成所有所需的 Excel 文件。
"""
