# 实验方案设计说明

> Riemannian Domain Adaptation for EEG Vigilance Estimation
> 目标期刊: IEEE Transactions on Biomedical Engineering (TBME)

---

## 1. 研究目标

验证黎曼流形平行传输对齐（Riemannian Parallel Transport Alignment）能否有效消除跨被试 EEG 协方差矩阵的分布偏移，从而提升留一被试（LOSO）跨被试警觉度回归性能。

核心假设：不同被试的 SPD 协方差矩阵因个体生理差异分布在 SPD 流形的不同区域；通过平行传输将各被试的协方差分布对齐到同一参考点，可以消除被试特异偏移，同时保留警觉度相关的 epoch 间变异。

---

## 2. 算法与创新点

### 2.1 算法概述

端到端流程：

```
原始 EEG (被试 s, 17ch × 200Hz)
  │
  ├── Step 1: 频带分解 (δ/θ/α/β/γ Butterworth 带通)
  ├── Step 2: OAS 收缩协方差估计 → SPD 矩阵
  ├── Step 3: 黎曼平行传输对齐 ← 核心创新
  │      G_s = mean_riemann({C_si})
  │      C'_si = G_s^{-1/2} · C_si · G_s^{-1/2}
  ├── Step 4: 切空间投影 (TangentSpace)
  ├── Step 5: 多频段特征拼接 (680 dim)
  ├── Step 6: SelectKBest (f_regression, k=150)
  ├── Step 7: StandardScaler
  ├── Step 8: RidgeCV 回归
  └── Step 9: 时序平滑 (window=3) → COR, RMSE
```

### 2.2 核心创新点

| 创新点 | 说明 |
|--------|------|
| 首次将黎曼平行传输对齐引入 EEG 警觉度估计 | 现有方法均使用欧几里得特征（PSD/微分熵） |
| 分频段对齐策略 | 每个频段独立计算黎曼均值并对齐，保留各频段独立结构 |
| 单被试无监督对齐 | 对齐参数仅基于被试自身 SPD 矩阵，LOSO 下无数据泄漏 |
| 公平 SOTA 对比框架 | 首次在同一 LOSO 框架下对比四种黎曼迁移学习方法 |

### 2.3 对齐方法对比

| 方法 | 变换 | 几何意义 |
|------|------|----------|
| None (基线) | C'_i = C_i | 原始 SPD，被试间偏差保留 |
| Euclidean Alignment | C'_i = M_e^{-1/2} C_i M_e^{-1/2} | 白化到欧几里得均值 |
| RPA | C'_i = λ · G^{-1/2} C_i G^{-1/2} | 平行传输 + 缩放因子 |
| Zanini 2018 | C'_i = G_grand^{-1/2} C_i G_grand^{-1/2} | 平行传输到全数据均值 |
| **本文 RPTA** | C'_i = G^{-1/2} C_i G^{-1/2} | 平行传输到单位矩阵 I |

---

## 3. 实验环境

### 3.1 硬件与依赖

| 项目 | 配置 |
|------|------|
| 工作站 | Linux (Ubuntu)，内存 ≥ 16 GB |
| Python | ≥ 3.13 |
| 核心依赖 | numpy, scipy, scikit-learn, pyriemann, matplotlib |

### 3.2 数据集

| 数据集 | 用途 | 规模 |
|--------|------|------|
| SEED-VIG | 主实验 | 23 被试 × 17ch × ~885 epochs × 200Hz |
| 标签 | PERCLOS | 连续值 [0, 1] |

---

## 4. 实验内容

### 实验 1: LOSO 主实验 (Table I)

**目的**: 验证 Riemannian 对齐在不同通道配置下的跨被试泛化提升。

**设计**: 3 对齐器 (None / Euclidean / Riemannian) × 3 通道 (17ch all / 6ch temporal / 4ch forehead)，23 被试 LOSO。

**统计**: 配对 t 检验 (Riemannian vs None), Cohen's d 效应量, Wilcoxon signed-rank 检验 (4ch), 置换检验 (n=10,000)。

**输出文件**:
- `results/results_table1_all_*.json` — 全量结果（含每被试 COR/RMSE、对齐器对比、配对 t 检验）
- `figures/individual_cor.pdf` — 每被试 COR Bar 图（`run_all_experiments.py --figs-only` 生成）
- `figures/fewshot_curve.pdf` — Few-shot 曲线（`--figs-only` 生成，依赖实验 2 结果）

**运行**: `python run_all_experiments.py --full --only table1` (~60 min)

---

### 实验 2: Few-shot 校准曲线

**目的**: 评估新被试需要多少校准样本才能达到全量对齐性能。

**设计**: n ∈ {0, 1, 5, 10, 50, 100}，每配置重复 3 次取平均。n=0 为 None 基线。

**预期**: n=1 即可达到全量性能的 >95%，n=5~10 接近全量。

**运行**: `python run_all_experiments.py --full --only fewshot` (~30 min)

**输出文件**: `results/results_fewshot_*.json` — 各校准样本量下的平均 COR/RMSE

---

### 实验 3: 消融实验

**目的**: 验证 3 个关键设计维度的合理性。

**3a. 参考点**: identity vs grand_mean — 预期性能一致（切空间自动选切点）

**3b. 协方差估计器**: OAS vs SCM vs LWF — 预期 OAS ≥ LWF > SCM，对齐在各估计器下均有效

**3c. 回归器**: RidgeCV vs SVR — 预期对齐收益不依赖特定回归器

**运行**: `python run_all_experiments.py --full --only ablation` (~20 min, 10 subjects)

**输出文件**: `results/results_ablations_*.json` — 参考点/协方差估计器/回归器各配置对比结果

---

### 实验 4: 频段重要性分析

**目的**: 分析各频段在特征选择中的贡献及对齐方式的影响。

**设计**: 对三种对齐器分别统计 SelectKBest 选中各频段的特征数。10 被试。

**预期**: α (8-14 Hz) 和 β (14-31 Hz) 频段贡献最大；对齐后各频段贡献更均匀。

**运行**: `python run_all_experiments.py --full --only band_importance` (~15 min)

**输出文件**: `results/results_band_importance_*.json` — 各对齐器在各频段的 SelectKBest 选中数

---

### 实验 5: 分频段 vs 全局对齐

**目的**: 验证分频段对齐的必要性。

**设计**: per-band（各频段独立对齐）vs all-band（拼接后统一对齐），Riemannian 对齐，10 被试。

**预期**: 分频段对齐优于全局对齐。

**运行**: `python run_all_experiments.py --full --only band_vs_global` (~20 min)

**输出文件**: `results/results_band_vs_global_*.json` — per-band vs all-band 的 COR/RMSE 对比

---

### 实验 6: SOTA Head-to-Head 对比 (Table II)

**目的**: 在完全相同 LOSO 框架下与现有 SOTA 黎曼迁移学习方法直接对比。

**对比方法**:

| 方法 | 引用 | 核心差异 | 17ch COR |
|------|------|----------|----------|
| None | — | 基线 | 0.585 |
| EA | He 2020, TBME | 欧几里得均值白化 | 0.655 |
| RPA | Rodrigues 2019, TBME | 平行传输 + 缩放因子 λ | 0.672 |
| Zanini 2018 | Zanini 2018, TBME | 参考点为全数据均值 | 0.692 |
| **Ours** | — | 参考点为单位矩阵 I | **0.692** |

**论文对应**: §III-G, Table II

**运行**: `python run_all_experiments.py --full --only sota` (~10 min)

**输出文件**:
- `results/results_sota_head_to_head_*.json` — 5 方法 × 3 通道的 COR/RMSE 表 + 统计检验
- `figures/sota_comparison.pdf` — SOTA 方法柱状对比图

---

### 实验 7: 增强统计分析 (§III-H)

**目的**: 提供比配对 t 检验更丰富的统计推断。

**方法**:

- **Bootstrap CI for Cohen's d**: 2000 次重抽样，不假设正态分布
- **Wilcoxon signed-rank 检验**: 非参数替代
- **Bland-Altman 图**: 展示 None vs Riemannian 的系统偏差和一致性界限
- **Per-subject Waterfall 图**: 展示每被试 ΔCOR 改善幅度

**论文对应**: §III-H (Statistical robustness) — Bootstrap CI 和 Wilcoxon p 写入正文

**运行**: `python run_all_experiments.py --full --only enhanced_stats` (~5 min，仅统计重算)

**输出文件**:
- `results/results_enhanced_stats_*.json` — Bootstrap 95% CI、Wilcoxon p-value
- `figures/paired_scatter.pdf` — None vs Riemannian 配对散点图
- `figures/bland_altman.pdf` — Bland-Altman 一致性图
- `figures/waterfall.pdf` — 每被试 ΔCOR 瀑布图

---

### 实验 8: 非参数配对置换检验 (§III-H)

**目的**: 提供不依赖正态假定的配对显著性检验。

**原理**: 随机翻转 aligner 标签 10000 次，以 mean_diff 为统计量构建零分布，给出经验 p-value。

**论文对应**: §III-H (Statistical robustness) — 置换检验 p 值写入正文; Table I caption 引用 4ch p=0.0001

**运行**: `python run_all_experiments.py --full --only permutation` (~30 min)

**输出文件**: `results/results_permutation_test_*.json` — 各通道配置下的经验 p-value、零分布统计量

---

### 实验 9: 计算成本报表 (Table III)

**目的**: 测量各 pipeline 阶段运行时间，证明 Riemannian 对齐开销可忽略。

**测量**: filter+cov / mean_riemann / tangent space / LOSO full 的 wall-clock 时间。

**实际结果**: Riemannian 对齐额外开销 ~0.13s/被试; LOSO 125s vs 95s (无对齐); 对齐开销 <32% 总时间。

**论文对应**: §III-I (Computational cost), Table III

**运行**: `python run_all_experiments.py --full --only comp_cost` (~5 min)

**输出文件**: `results/results_comp_cost_*.json` — 各 pipeline 阶段平均耗时（秒/被试）

---

### 实验 10: 可视化分析 (辅助)

**目的**: 生成辅助可视化图表，用于分析和调试。

**三组图表**:

| 图表 | 内容 | 论文使用 |
|------|------|----------|
| 箱线图 | None / Euclidean / Riemannian COR 分布 | 辅助分析，未直接入论文 |
| 频段重要性柱状图 | 5 频段 SelectKBest 选中数 | 由 Fig. 4 替代 |
| 对齐散点图 | None vs Riemannian COR | 辅助分析，未直接入论文 |

**运行**: `python run_all_experiments.py --full --only visualization` (~15 min)

**输出文件**:
- `results/results_*.json`（依赖 table1/band_importance 已有结果，不额外保存）
- `figures/subject_performance_boxplot.pdf` — 三种对齐器 COR 分布箱线图
- `figures/band_importance_bar.pdf` — 各频段 SelectKBest 选中数柱状图
- `figures/alignment_scatter.pdf` — None vs Riemannian 每被试 COR 散点图

---

## 5. 参数汇总

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 200 Hz | SEED-VIG 原始采样率 |
| Epoch 长度 | 8 秒 (1600 samples) | 与 PERCLOS 标签对齐 |
| 频段 | δ(1-4), θ(4-8), α(8-14), β(14-31), γ(31-50) Hz | 标准脑电频段 |
| 协方差估计 | OAS | 小样本下更稳定 |
| 黎曼均值迭代 | maxiter=50, tol=1e-6 | pyriemann 默认 |
| 特征选择 | SelectKBest(f_regression, k=150) | 基于方差分析 |
| RidgeCV alphas | [0.01, 0.1, 1.0, 10.0, 100.0] | 交叉验证选择 |
| 时序平滑 | uniform_filter1d, window=3 | 3 点移动平均 |

---

## 6. 运行指南

### 完整实验 (~3 小时)

```bash
cd /home/tanhuang/projects/ric_da

# 主实验
nohup python run_all_experiments.py --full --only table1 &
nohup python run_all_experiments.py --full --only fewshot &

# 消融和分析
nohup python run_all_experiments.py --full --only ablation &
nohup python run_all_experiments.py --full --only band_importance &
nohup python run_all_experiments.py --full --only band_vs_global &
nohup python run_all_experiments.py --full --only visualization &

# 补充实验（重用 table1 缓存，无额外 LOSO 开销）
nohup python run_all_experiments.py --full --only sota &
nohup python run_all_experiments.py --full --only enhanced_stats &
nohup python run_all_experiments.py --full --only permutation &
nohup python run_all_experiments.py --full --only comp_cost &

# 一次性运行全部
nohup python run_all_experiments.py --full &

# 图表
python run_all_experiments.py --figs-only
```

### CLI 参数

```
python run_all_experiments.py [--quick] [--full] [--only TABLE] [--no-figs] [--figs-only] [--no-log]

--quick        5 被试 (开发模式, ~30 min)
--full         23 被试 (提交模式, ~3 h)
--only TABLE   只跑指定实验
--no-figs      跳过图表生成
--figs-only    仅生成图表（基于已有结果）
--no-log       禁用日志保存
```

### --only 选项一览

| 选项 | 实验 | 预计耗时 (--full) |
|------|------|-------------------|
| table1 | LOSO 主实验 | ~60 min |
| fewshot | Few-shot 校准 | ~30 min |
| ablation | 消融实验 | ~20 min |
| band_importance | 频段重要性 | ~15 min |
| band_vs_global | 分频段 vs 全局 | ~20 min |
| visualization | 可视化分析 | ~15 min |
| sota | SOTA Head-to-Head | ~10 min |
| enhanced_stats | 增强统计 | ~5 min |
| permutation | 置换检验 | ~30 min |
| comp_cost | 计算成本 | ~5 min |

---

## 7. 实际结果汇总

| 实验 | 关键结果 | 论文对应 |
|------|---------|----------|
| Table 1 | Riemannian 17ch COR 0.692 (vs None 0.585), p=0.0007, d=0.824, 21/23 improved | Table I, §III-A |
| Few-shot | n=0 即达全量性能 (COR 0.696); n=1 COR 0.681 (98%) | Fig. 2, §III-C |
| 消融 | identity ≈ grand_mean (0.635); OAS 0.635 ≥ LWF 0.628 > SCM 0.581; RidgeCV 0.635 ≈ SVR 0.638 | §III-E |
| 频段重要性 | α 27.2% + β 29.0% = 56%; 对齐方式不影响分布 | Fig. 4, §III-F |
| 分频段 vs 全局 | per-band 0.635 > all-band 0.609 (+0.026) | §III-E |
| SOTA | Ours = Zanini 0.692 > RPA 0.672 > EA 0.655 > None 0.585 | Table II, §III-G |
| 增强统计 | d 95% CI [0.62, 1.21] 不跨零; Wilcoxon p=7.9×10⁻⁶ | §III-H |
| 置换检验 | 17ch p=0; 4ch p=0.0001; 6ch p=0.176 (n.s.) | §III-H |
| 计算成本 | 对齐增量 ~0.13s/被试; LOSO 125s vs 95s (无对齐) | Table III, §III-I |
| 可视化 | 箱线图/散点图/瀑布图（辅助分析，未直接入论文） | — |