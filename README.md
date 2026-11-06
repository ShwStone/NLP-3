# 编程作业三：基于手写解码器语言模型的文本分类

本项目复用编程作业二的手写解码器（因果多头自注意力、LayerNorm、Pre-LN
解码器块、词嵌入与可学习位置编码），在公开英文数据集 AG News 上完成三种
文本分类方案，并给出可复现实验、原始日志、样例、中文报告和自动核验脚本。

- 微调式分类头：最后一个有效词元 → 线性层 → 交叉熵；支持全参数微调、
  冻结主干（线性探针）和随机初始化消融。
- 多项式朴素贝叶斯：同一套小写正则分词，unigram + bigram 词袋，显式实现
  拉普拉斯平滑与对数判别；用 sklearn `MultinomialNB` 交叉核对。
- Prompt 范式（选做）：英文模板 + 类别 verbalizer + 语言模型打分；
  实现零样本、四样本、验证集 verbalizer 搜索和 content-free 校准。

## 主要结果

AG News，官方测试集 7,600 条。所有数字来自 `results/` 中的真实日志。

| 方案 | 测试准确率 | 宏 F1 | 可训练参数 | 训练耗时 | 测试吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 解码器全参数微调 | 93.24% | 93.22% | 10,889,732 | 94.68 s | 11,990 条/s |
| 解码器随机初始化（消融） | 91.86% | 91.83% | 10,889,732 | 94.34 s | 12,191 条/s |
| 多项式朴素贝叶斯 | 90.57% | 90.53% | 800,004 | 6.85 s | 27,102 条/s |
| 解码器冻结主干 | 88.03% | 88.02% | 1,028 | 71.75 s | 12,296 条/s |
| Prompt 零样本（校准） | 73.24% | 72.94% | 0 | 不训练 | 2,468 条/s |
| Prompt 四样本（校准） | 63.57% | 60.20% | 0 | 不训练 | 1,570 条/s |

语言模型预训练：4 层、d=256、4 头、上下文 192、2,000 步；训练交叉熵
10.3450 → 3.9327，验证交叉熵 10.3449 → 4.0634，验证困惑度
31,099.48 → 58.17，耗时 104.97 秒（含评估）。分类头只有 1,028 个参数。

## 数据集与划分

AG News 是公开英文新闻主题分类数据集（Zhang, Zhao & LeCun, 2015），四类：
World、Sports、Business、Sci/Tech。公开 CSV 镜像：

```text
https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv
```

官方训练集 120,000 条，官方测试集 7,600 条。代码把官方训练集按类别分层
90%/10% 划分为训练/验证（108,000 / 12,000），测试集只用于最终评估。
`src/data.py` 首次运行会自动下载两个 CSV；`data/metadata.json` 记录 URL、
SHA-256、类别分布、词表大小和长度统计。

文本由 `title + ". " + description` 组成。分词器是小写正则词级分词器，
只在训练集上统计词频，词表 30,000；分类/语言模型最大长度 192 个词元。
`<pad>/<unk>/<bos>/<eos>` 和模板/verbalizer 词被显式加入词表。

## 目录结构

```text
src/
  model.py            作业二手写解码器，新增无副作用的 encode()
  tokenizer.py        词级分词器与两套 verbalizer
  data.py             下载、解析、分层划分、数据集与语言模型流
  classifier.py       分类头、冻结/微调、checkpoint 读写、参数统计
  pretrain.py         AG News 语言模型预训练
  train_classifier.py full / frozen / scratch 三种分类训练
  naive_bayes.py      手写多项式 NB + sklearn 交叉核对
  prompt.py           Prompt 打分、零样本/四样本、上下文校准
  make_samples.py     导出三组测试样例
  utils.py            随机种子、精度、指标、计时
tests/                24 项单元测试
scripts/              一键实验、verbalizer 调优、汇总、画图、报告、核验、打包
report/               LaTeX 源码、自动生成的表格行、图表与编译 PDF
results/              真实日志、指标、预测、样例、核验结果
data/                 AG News CSV（首次运行下载）与元数据
```

## 环境

实测环境：

- Python 3.12.9
- PyTorch 2.6.0+cu124
- NumPy 2.2.4
- scikit-learn 1.6.1
- scipy 1.17.0
- matplotlib 3.10.1
- NVIDIA RTX 3090 24 GiB

先安装匹配硬件的 PyTorch，再安装其余依赖：

```bash
python -m pip install torch
python -m pip install -r requirements.txt
```

CPU 也可运行，但训练会慢很多。脚本默认使用 CUDA（若可用），可显式
`--device cpu`。为了确定性，`src/utils.py` 会设置
`CUBLAS_WORKSPACE_CONFIG=:4096:8` 并开启确定性算法。

## 快速开始

```bash
# 1. 单元测试
python -m pytest -q

# 2. 下载/解析 AG News，生成 data/metadata.json
python -m src.data

# 3. 语言模型预训练（约 2 分钟，RTX 3090）
python -m src.pretrain

# 4. 全参数微调分类器（约 2 分钟）
python -m src.train_classifier --mode full --epochs 3

# 5. 冻结主干 / 随机初始化消融
python -m src.train_classifier --mode frozen --epochs 6
python -m src.train_classifier --mode scratch --epochs 3

# 6. 朴素贝叶斯
python -m src.naive_bayes

# 7. Prompt（默认已是验证集选出的 verbalizer + 校准）
python -m src.prompt --shots 0
python -m src.prompt --shots 0 --verbalizer-preset default --no-calibrate \
  --output results/prompt_zeroshot_default
python -m src.prompt --shots 1

# 8. 样例、汇总、图表、报告数值
python -m src.make_samples
python scripts/collect_results.py
python scripts/make_figures.py
python scripts/report_values.py

# 9. 核验与报告编译
python scripts/verify_results.py
bash scripts/build_report.sh
python scripts/check_delivery.py
```

完整重跑（会覆盖 `results/` 并重新训练）：

```bash
bash scripts/run_experiment.sh
```

`scripts/run_experiment.sh` 还会先运行
`scripts/tune_prompt_verbalizers.py`，只用验证集枚举 72 组 verbalizer，
生成 `results/prompt_verbalizer_tuning.json`；测试集不参与选择。

## 实现要点

### 复用手写主干

`src/model.py` 保留作业二的显式因果注意力与手写 LayerNorm，不调用
`MultiheadAttention`、`Transformer`、`scaled_dot_product_attention` 或
`LayerNorm`。作业三新增 `DecoderLM.encode(indices)`，返回最后一层 LayerNorm
后的隐状态；`forward` 和分类头都调用它，分类表示与语言模型预训练表示一致。

### 分类头

`ClassifierModel` 取每个样本最后一个真实词元的隐状态，接
`Linear(d_model, num_classes)`。右侧 pad 不会被因果注意力看到，所以
last-token pooling 不受补位影响；代码也实现了带 length mask 的 mean pooling。
`set_backbone_frozen(True)` 冻结主干并让主干保持 eval；全参数模式使用较小的
主干学习率和较大的分类头学习率。

### 朴素贝叶斯

`src/naive_bayes.py` 使用与神经模型相同的分词器，`CountVectorizer` 构造
unigram+bigram 稀疏计数（`min_df=3`，最多 200,000 特征），然后由
`ScratchMultinomialNB` 显式计算：

```text
log P(c)      = log(N_c / N)
log P(w | c)  = log((n_cw + alpha) / (N_c + alpha * V))
```

验证集从 `alpha ∈ {0.1, 0.3, 1.0, 3.0}` 中选 0.1；测试预测与 sklearn
`MultinomialNB` 完全一致（一致率 100%）。

### Prompt

零样本模板：

```text
title: <title> description: <description> topic: <verbalizer><eos>
```

对四类 verbalizer 计算平均对数概率并取最大。默认
`world/sports/business/science` 只有 48.61%；在验证集上枚举 72 组候选词后选
`politics/sports/financial/computers`，再做三个 content-free 模板的上下文校准，
测试准确率达到 73.24%。四样本设置每类拼接一个训练示例，反而降到 63.57%，
报告里分析了原因（上下文被压缩、4 层小模型的 in-context 能力有限）。

## 结果文件

- `results/lm_summary.json`、`results/lm_training_log.csv`、
  `results/lm_eval_log.csv`：语言模型预训练。
- `results/classifier_{full,frozen,scratch}/metrics.json`：分类器指标、逐 epoch
  历史、参数统计、耗时、显存和 checkpoint 哈希。
- `results/naive_bayes/metrics.json`：NB 指标、alpha 搜索、特征数、top 特征、
  sklearn 交叉核对。
- `results/prompt_zeroshot/metrics.json`、`results/prompt_fewshot4/metrics.json`：
  Prompt 指标、verbalizer、校准分数、示例。
- `results/prompt_verbalizer_tuning.json`：仅验证集的 72 组搜索结果。
- `results/comparison.json` / `.csv`：报告主对比表。
- `results/samples.json` / `.txt`：三组固定测试样本和全部模型预测。
- `results/verification.json`：74 项一致性核验。
- `results/tests.xml`、`results/tests_console.log`：24 项单元测试。
- `report/report.pdf`、`实验报告_文本分类.pdf`：中文实验报告。
- `report/figures/`：训练曲线、对比柱状图、混淆矩阵。

## 说明与限制

- 只使用一个随机种子和一次分层划分，没有多种子置信区间；测试集 7,600 条，
  准确率标准误约 0.3 个百分点。
- 语言模型和分类器都是 4 层、d=256 的小模型；Prompt 的 few-shot 能力受规模
  限制，结果不代表大模型的上限。
- AG News CSV 保留 Reuters 模板、URL 片段等原始内容；NB 的 top 特征会利用
  这些伪影，报告中有说明。神经模型与 NB 使用相同文本，比较是公平的。
- Prompt 的 verbalizer 只在验证集选择，测试集没有参与；但验证集搜索仍有
  一定选择偏差，更严谨的做法是嵌套交叉验证。
- `results/*.pt` 和 `data/*.csv` 默认不纳入提交包，运行脚本会自动下载数据并
  重新生成权重；报告和指标文件保留，便于审阅。

## 作业二复用声明

本项目直接复用编程作业二仓库
`https://github.com/xunju/language-model-assignment2` 中本人/本组手写的
`src/model.py`（因果多头自注意力、LayerNorm、Pre-LN 解码器、词嵌入与位置编码），
只新增 `encode()`；没有复制网上完整 GPT 分类实现。分词、分类头、NB、Prompt、
训练与评估代码均为本项目独立实现。

小组均为人智 2401 班：石皓文（2233316022）、张恩瑞（2244413862）、
杜鹏宇（2223210437）、许程（2244313744）。
