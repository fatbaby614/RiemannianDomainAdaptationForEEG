# AGENTS.md — Riemannian Domain Adaptation (ric_da)

> AI agent onboarding document. Read this first before making any changes.

---

## 1. What This Project Is

A research codebase for **Riemannian Domain Adaptation (ric_da)** applied to EEG vigilance estimation cross-subject generalization. Submitted to *IEEE Transactions on Biomedical Engineering* (TBME).

**One-sentence core finding:** Riemannian parallel transport alignment of SPD covariance matrices significantly improves LOSO cross-subject vigilance regression (COR 0.585→0.692, p=0.0007, d=0.824, 21/23 subjects improved) by removing subject-specific biases on the SPD manifold before tangent space projection.

**One dataset:** SEED-VIG (23 subjects × 17ch × 885 epochs).

**Relationship to sibling project (SCAFBTSRegressor):** This is a new method built on top of the FBTS feature pipeline. SCAFBTSRegressor established FBTS features for vigilance regression; ric_da adds Riemannian domain adaptation to solve the cross-subject generalization problem.

---

## 2. Architecture

```
Core Engine              →  data imports         →  experiment scripts
ric_da_core.py ★           SCAFBTSRegressor/       run_all_experiments.py ★
                           config.py                run_ric_da.py
                           data_loader.py           generate_pub_figures.py
                           drozy_loader.py
                           utils.py
                           sca_fbts_fast.py
```

### Core Engine (`ric_da_core.py`)

Three aligner classes sharing the same `BaseAlignment` interface:

| Class | Name | Principle | Equation |
|-------|------|-----------|----------|
| `NoAlignment` | `none` | Identity transform (baseline) | `C' = C` |
| `EuclideanAlignment` | `euclidean` | Whiten by Euclidean mean | `C' = M^{-1/2} C M^{-1/2}` |
| `RiemannianAlignment` | `riemann` | Parallel transport by Riemannian mean | `C' = G^{-1/2} C G^{-1/2}` |

Key API pattern:
```python
from ric_da_core import RiemannianAlignment, precompute_all_subjects, evaluate_loso

# One-time precomputation (cached across all aligners)
covs, labels, _ = precompute_all_subjects(subjects, bands='5band', channels='all')

# Run LOSO with any aligner
aligner = RiemannianAlignment()
results = evaluate_loso(subjects, aligner=aligner, all_covs=covs, all_labels=labels)
```

### Data Layer (imported from `SCAFBTSRegressor/`)

| File | What it provides |
|------|-----------------|
| `config.py` | `SEED_VIG_ROOT`, `DROZY_ROOT`, `BANDS_5`, `BANDS_8` (auto-detects Linux/Windows) |
| `data_loader.py` | `load_raw_eeg()`, `load_perclos()`, `load_eog_features()`, `list_subjects()` |
| `drozy_loader.py` | `load_drozy_subject()` — EDF loading for DROZY |
| `utils.py` | `cor()`, `rmse()`, `mae()` |
| `sca_fbts_fast.py` | `apply_bandpass_filter()` — Butterworth filter bank |

### Config (`config_da.py`)

Centralized config for this project. Key constants:

| Constant | Value | Notes |
|----------|-------|-------|
| `RIC_DA_ROOT` | `ric_da/` directory | All outputs are below this |
| `DA_RESULTS_DIR` | `ric_da/results/` | Experiment result JSONs |
| `DA_FIGURES_DIR` | `ric_da/figures/` | Paper figures (PDF) |
| `DA_PAPER_DIR` | `ric_da/paper/` | LaTeX paper |
| `SEED_VIG_SUBJECTS` | Read from `Raw_Data/` | Dynamic, not hardcoded |
| `QUICK_SUBJECTS` | First 5 subjects | For fast testing |

---

## 3. The Riemannian Alignment Method

### 3.1 Problem

In LOSO cross-subject evaluation, SPD covariance matrices from different subjects occupy different regions of the Riemannian manifold due to subject-specific physiological factors (skull conductivity, electrode impedance, baseline EEG power). This domain shift degrades the tangent space projection and subsequent regression.

### 3.2 Solution: Parallel Transport Alignment

For each subject `s` with SPD matrices `{C_si}`:

1. Compute the Riemannian mean (Frechét mean): `G_s = RiemannianMean({C_si})`
2. Parallel transport to identity: `C'_si = G_s^{-1/2} · C_si · G_s^{-1/2}`

After transport, all subjects' SPD distributions are centered at `I` on the manifold. Subject-specific biases are removed while vigilance-related epoch-to-epoch variations are preserved.

### 3.3 Algorithm Flow

```
Raw EEG
  ↓ (per subject)
Bandpass filter banks (δ/θ/α/β/γ)
  ↓
SPD covariance matrices (OAS shrinkage)
  ↓
┌─────────────────────────────────────────────┐
│  Parallel transport alignment (per subject) │
│  C' = G_s^{-1/2} · C · G_s^{-1/2}          │
│  where G_s = RiemannianMean(subject's SPDs) │
└─────────────────────────────────────────────┘
  ↓
Tangent space projection (per band)
  ↓
Concatenate band features
  ↓
SelectKBest (f_regression, k=150)
  ↓
StandardScaler
  ↓
RidgeCV regression
  ↓
Temporal smoothing (window=3)
  ↓
PERCLOS prediction
```

### 3.4 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Per-band alignment | Different frequency bands exhibit different inter-subject variability patterns |
| Precompute architecture | Filter + covariance is 80% of runtime; compute once, reuse across aligners |
| Fit on train subjects only | No test subject data leaks during alignment parameter estimation |
| Identity reference | Simpler than grand_mean; both tested in ablation, no significant difference |
| RidgeCV regressor | Ablation confirmed RidgeCV outperforms SVR for tangent space features |

---

## 4. Experimental Matrix (10 Core Experiments)

| # | Experiment | --only flag | Purpose | Run time (--full) | Result JSON | Figure(s) |
|---|-----------|-------------|---------|-------------------|-------------|-----------|
| 1 | LOSO Main (Table 1) | `table1` | 4 aligners × 3 channel configs, 23 subjects LOSO | ~60 min | `results/results_table1_all_*.json` | `figures/individual_cor.pdf`, `figures/channel_comparison.pdf` |
| 2 | Few-shot Calibration | `fewshot` | n ∈ {0,1,5,10,50,100}, 3 repeats | ~30 min | `results/results_fewshot_*.json` | `figures/fewshot_curve.pdf` |
| 3 | Ablation | `ablation` | Reference / Cov estimator / Regressor, 10 subjects | ~20 min | `results/results_ablations_*.json` | — (included in paper text) |
| 4 | Band Importance | `band_importance` | SelectKBest feature counts per band, 10 subjects | ~15 min | `results/results_band_importance_*.json` | `figures/band_importance.pdf` |
| 5 | Per-band vs Global | `band_vs_global` | Per-band vs all-band Riemannian alignment, 10 subjects | ~20 min | `results/results_band_vs_global_*.json` | — (included in paper text) |
| 6 | SOTA Head-to-Head | `sota` | None vs EA vs RPA vs Zanini vs Ours, 5×3 table | ~10 min | `results/results_sota_head_to_head_*.json` | — (paper table) |
| 7 | Enhanced Stats | `enhanced_stats` | Bootstrap CI, Bland-Altman, Waterfall plots | ~5 min | `results/results_enhanced_stats_*.json` | `figures/paired_scatter.pdf`, `figures/bland_altman.pdf`, `figures/waterfall.pdf` |
| 8 | Permutation Test | `permutation` | Non-parametric paired test, 10000 permutations | ~30 min | `results/results_permutation_test_*.json` | — (included in paper text) |
| 9 | Computational Cost | `comp_cost` | Pipeline timing breakdown | ~5 min | `results/results_comp_cost_*.json` | — (paper table) |
| 10 | Visualization | `visualization` | Box plots, bar charts, scatter plots | ~15 min | — (reuses exp 1 & 4 data) | `figures/subject_performance_boxplot.pdf` |

**Total full run:** ~3 hours (23 subjects, full pipeline)

**Quick test:** `python run_all_experiments.py --quick` (~30 min, 5 subjects)

**JSON 命名约定:** `results_<实验名>_<时间戳>.json`，例如 `results_table1_all_20260612_120347.json`
**图表格式:** 每张图同时输出 `.pdf` (论文用)、`.svg` (网页用)、`.tiff` (投稿用) 三种格式
**日志文件:** `results/experiment_<时间戳>.log` 记录完整运行输出

---

## 5. File Map

```
ric_da/
├── __init__.py                  # Package marker
├── AGENTS.md                    # ← THIS FILE: agent onboarding
├── config_da.py                 # Config: paths, subjects, parameter grids
├── ric_da_core.py              ★ Core engine: 3 aligners + LOSO + caching
├── generate_pub_figures.py     # Paper figures (individual_cor.pdf, fewshot_curve.pdf)
├── run_all_experiments.py      ★ One-click experiment launcher (TBME submission)
├── run_ric_da.py                # Modular experiment entry point
├── _test_quick.py               # Quick smoke test (3 aligners × 5 subjects)
├── _run_full.py                 # Full 23-subject experiment (used for Windows run)
├── results/                     # ← Experiment result JSONs
│   └── full_results_*.json      # From initial Windows run
├── figures/                     # ← Paper figures (PDF)
│   ├── individual_cor.pdf       # Per-subject COR bar chart
│   └── fewshot_curve.pdf        # Calibration size vs COR curve
└── paper/                       # ← Paper LaTeX source + output
    ├── paper_tbme.tex           # LaTeX source
    └── paper_tbme.pdf           # Compiled PDF (4 pages, 310KB)
```

### Sibling dependency (must be in `../SCAFBTSRegressor/`)

```
SCAFBTSRegressor/
├── config.py                    # Dataset paths (auto-detect Linux/Windows)
├── data_loader.py               # SEED-VIG data loading
├── drozy_loader.py              # DROZY data loading
├── utils.py                     # cor(), rmse(), mae()
└── sca_fbts_fast.py             # apply_bandpass_filter()
```

---

## 6. How to Run

### **On Linux workstation (primary target)**

```bash
# Directory structure must be:
# /home/tanhuang/projects/
# ├── SCAFBTSRegressor/
# └── ric_da/

cd /home/tanhuang/projects/ric_da

# Full experiment (all 10 experiments, ~3 hours)
nohup python run_all_experiments.py --full &

# Individual experiments (recommended for parallel runs)
nohup python run_all_experiments.py --full --only table1 &
nohup python run_all_experiments.py --full --only fewshot &
nohup python run_all_experiments.py --full --only ablation &
nohup python run_all_experiments.py --full --only band_importance &
nohup python run_all_experiments.py --full --only band_vs_global &
nohup python run_all_experiments.py --full --only sota &
nohup python run_all_experiments.py --full --only visualization &

# Quick test (~30 min)
python run_all_experiments.py --quick

# Generate figures only (if experiments already done)
python run_all_experiments.py --figs-only

# Compile paper
cd paper && pdflatex paper_tbme.tex && pdflatex paper_tbme.tex
```

### **On Windows (development)**

```bash
cd D:\THWork\ric_da

# Quick verification
python -m ric_da._test_quick

# Single config
python -m ric_da._run_full

# Via main script
python run_all_experiments.py --quick
```

### CLI Reference

```
python run_all_experiments.py [--quick] [--full] [--only TABLE] [--no-figs] [--figs-only] [--no-log]

--quick        5 subjects (dev mode, ~30 min)
--full         23 subjects (submission mode, ~3 hours)
--only TABLE   One of: table1, fewshot, ablation, band_importance,
               band_vs_global, sota, enhanced_stats, permutation,
               comp_cost, visualization
--no-figs      Skip figure generation after experiments
--figs-only    Generate figures from existing results only
--no-log       Disable log file output

---

## 7. Adding a New Experiment

1. Add the experiment function in `run_all_experiments.py` (follow the pattern of `run_table1`, `run_fewshot`, etc.)
2. Add the CLI option in the `--only` choices list
3. In the `main()` function, add a `if run_all or args.only == 'yourname':` block
4. Add a row to §4 (Experimental Matrix) in this AGENTS.md

---

## 8. Key Metrics

| Metric | Code | Direction |
|--------|------|-----------|
| COR (Pearson r) | `utils.cor()` | Higher = better |
| RMSE | `utils.rmse()` | Lower = better |
| Cohen's d | `ric_da_core.paired_statistics()` | Higher absolute = larger effect |
| Paired t-test p | `scipy.stats.ttest_rel()` | p < 0.05 = significant |

---

## 9. Results Summary (from full 23-subject run)

### Table 1: LOSO Main Results

| Channels | Aligner | COR | RMSE | vs. None |
|----------|---------|-----|------|----------|
| 17ch (all) | None | 0.585 ± 0.230 | 0.278 | — |
| 17ch (all) | Euclidean | 0.655 ± 0.218 | 0.224 | p=0.005, d=0.650 |
| **17ch (all)** | **Riemannian** | **0.692 ± 0.206** | **0.195** | **p=0.0007, d=0.824** |
| 6ch (temporal) | None | 0.659 ± 0.185 | 0.248 | — |
| 6ch (temporal) | Euclidean | 0.622 ± 0.257 | 0.214 | n.s. |
| **6ch (temporal)** | **Riemannian** | **0.689 ± 0.190** | **0.194** | **n.s.** |
| 4ch (forehead) | None | 0.626 ± 0.196 | 0.243 | — |
| 4ch (forehead) | Euclidean | 0.642 ± 0.204 | 0.208 | n.s. |
| **4ch (forehead)** | **Riemannian** | **0.681 ± 0.185** | **0.196** | **p<0.0001†, d=0.850** |

†Wilcoxon signed-rank test; permutation test p=0.0001 (n=10,000)

Key findings:
- Riemannian alignment significantly improves 17ch (p=0.0007, d=0.824) and 4ch (p<0.0001, d=0.850)
- 21/23 subjects individually improved (17ch configuration)
- 4ch forehead retains 98.4% of full 17ch Riemannian performance
- Euclidean alignment provides intermediate improvement on 17ch but underperforms on 6ch

### SOTA Head-to-Head (17ch)

| Method | COR | RMSE |
|--------|-----|------|
| None (baseline) | 0.585 ± 0.230 | 0.278 |
| EA | 0.655 ± 0.218 | 0.224 |
| RPA | 0.672 ± 0.196 | 0.197 |
| Zanini et al. | 0.692 ± 0.206 | 0.195 |
| **Ours** | **0.692 ± 0.206** | **0.195** |

### Enhanced Statistics

| Channel | Wilcoxon p | Bootstrap 95% CI for d | Permutation p (n=10,000) |
|---------|-----------|----------------------|--------------------------|
| 17ch | 7.9×10⁻⁶ | [0.62, 1.21] | 0 |
| 6ch | 0.021 | [-0.10, 1.16] | 0.176 |
| 4ch | 8.8×10⁻⁵ | [0.60, 1.31] | 0.0001 |

### Computational Cost (per subject, 17ch, 885 epochs)

| Stage | None (s) | Riemannian (s) |
|-------|----------|----------------|
| Filter + covariance | 27.9 | 27.9 |
| Riemannian mean | 2.79 | 2.92 |
| Tangent space | 3.02 | 2.93 |
| Full LOSO fold | 95.3 | 125.5 |

---

## 10. Paper Structure (`paper/paper_tbme.tex`)

| Section | Content | Source Data (JSON) | Figure/Table |
|---------|---------|---------------------|-------------|
| Abstract | Problem, method, main results, SOTA comparison, computational cost | `results_table1_all_*.json` | — |
| I. Introduction | Motivation, FBTS background, BCI DA literature, contributions | — | — |
| II. Methods | FBTS pipeline, SPD manifold, parallel transport, protocol | — | — |
| III-A. Table 1 (LOSO Main) | 3 channel configs × 3 aligners, 23 subjects | `results_table1_all_*.json` | Table I |
| III-B. Per-subject analysis | 21/23 improved, individual COR | `results_table1_all_*.json` | Fig. 1 (`individual_cor.pdf`) |
| III-C. Few-shot curve | n=0/1/5/10/50/100 calibration epochs | `results_fewshot_*.json` | Fig. 2 (`fewshot_curve.pdf`) |
| III-D. Channel comparison | 17ch vs 6ch vs 4ch | `results_table1_all_*.json` | Fig. 3 (`channel_comparison.pdf`) |
| III-E. Ablations | Reference / Estimator / Regressor / Per-band vs Global | `results_ablations_*.json` | — |
| III-F. Band importance | δ/θ/α/β/γ feature counts | `results_band_importance_*.json` | Fig. 4 (`band_importance.pdf`) |
| III-G. SOTA comparison | None/EA/RPA/Zanini/Ours head-to-head | `results_sota_head_to_head_*.json` | Table II |
| III-H. Statistical robustness | Wilcoxon, Bootstrap CI, Permutation test | `results_enhanced_stats_*.json`, `results_permutation_test_*.json` | — |
| III-I. Computational cost | Pipeline timing breakdown | `results_comp_cost_*.json` | Table III |
| IV. Discussion | Why it works, SOTA relation, practical implications, limitations | — | — |
| V. Conclusion | Summary, key numbers, future work | — | — |
| References | 18 refs (Zheng 2017, Zanini 2018, Rodrigues 2019, Zhuo 2024, Paillard 2025, Gao 2024, Ju 2025, Cui 2023, etc.) | — | — |

**Target journal:** IEEE Transactions on Biomedical Engineering (TBME)

### 10.1 Paper Figure ↔ Data File Mapping

| Paper Figure | LaTeX Reference | Data Source (JSON) | Generator Function |
|--------------|-----------------|---------------------|---------------------|
| Fig. 1 (individual_cor.pdf) | `\ref{fig:individual}` | `results_table1_all_*.json` | `generate_pub_figures.fig_individual_cor()` |
| Fig. 2 (fewshot_curve.pdf) | `\ref{fig:fewshot}` | `results_fewshot_*.json` | `generate_pub_figures.fig_fewshot()` |
| Fig. 3 (channel_comparison.pdf) | `\ref{fig:channels}` | `results_table1_all_*.json` | `generate_pub_figures.fig_channel_comparison()` |
| Fig. 4 (band_importance.pdf) | `\ref{fig:band}` | `results_band_importance_*.json` | `generate_pub_figures.fig_band_importance()` |

### 10.2 Paper Table ↔ Data File Mapping

| Paper Table | LaTeX Reference | Data Source (JSON) | Content |
|-------------|-----------------|---------------------|---------|
| Table I (Main LOSO) | `\ref{tab:main}` | `results_table1_all_*.json` | 3 channel configs × 3 aligners, mean ± std COR/RMSE, p, d |
| Table II (SOTA) | `\ref{tab:sota}` | `results_sota_head_to_head_*.json` | None/EA/RPA/Zanini/Ours head-to-head (17ch) |
| Table III (Comp Cost) | `\ref{tab:compcost}` | `results_comp_cost_*.json` | Per-stage timing breakdown |

---

## 11. Dependencies

```
numpy, scipy              — Numerical computation
scikit-learn              — RidgeCV, SelectKBest, StandardScaler, SVR
pyriemann                 — Covariances, TangentSpace, mean_riemann, invsqrtm
matplotlib                — Paper figures
```

```bash
pip install numpy scipy scikit-learn pyriemann matplotlib
```

---

## 12. Troubleshooting

| Symptom | Check |
|---------|-------|
| `ModuleNotFoundError: No module named 'data_loader'` | Verify `SCAFBTSRegressor/` is a sibling of `ric_da/` |
| `FileNotFoundError` on SEED-VIG | Check `config.py` paths; Linux paths in `platform.system() == 'Linux'` branch |
| `mean_riemann` converges slowly | Default maxiter=50, tol=1e-6; should converge in 5-15 iterations for EEG data |
| Out of memory on 17ch LOSO | Each fold pools 22×885=19,470 SPD matrices of size 17×17 (~45MB). If issue persists, reduce to `--quick` or 10 subjects |
| Paper compilation: `individual_cor.pdf` not found | Run `generate_pub_figures.py` first, or run `run_all_experiments.py --no-figs` separately |

---

## 13. 实验方案与代码/数据完整追溯

### 13.1 实验方案设计思路

| 阶段 | 实验 | 方案 | 回答的科学问题 |
|------|------|------|----------------|
| **核心论证** | Exp 1 (Table 1) | LOSO 跨被试主实验，3 通道配置 × 4 对齐器 | Riemannian 对齐是否真的有效？在哪种通道配置下最有效？ |
| **实用价值** | Exp 2 (Few-shot) | 测试被试校准样本量 n=0/1/5/10/50/100 时的性能曲线 | 实际部署时需要多少校准数据？ |
| **机制解释** | Exp 3 (Ablation) | 消融：参考点(identity vs grand_mean) / 估计器(OAS/SCM/LWF) / 回归器(RidgeCV/SVR) | 性能提升的来源是哪个组件？ |
| **可解释性** | Exp 4 (Band Importance) | SelectKBest 选出的 150 特征在 5 个频段的分布 | 对齐后哪些频段对预测贡献最大？ |
| **方法变体** | Exp 5 (Per-band vs Global) | 分频带对齐 vs 全局（跨频带拼接后）对齐 | 是否需要为每个频段单独对齐？ |
| **方法定位** | Exp 6 (SOTA) | 与 None / Euclidean / RPA / Zanini 2018 直接对比 | 相比现有 BCI 域适应方法的优势？ |
| **统计严谨** | Exp 7 (Enhanced Stats) | Bootstrap CI、Wilcoxon 检验、Bland-Altman、瀑布图 | 配对 t 检验是否被非正态性违反？效应量有多大？ |
| **统计严谨** | Exp 8 (Permutation) | 10000 次置换检验，不依赖分布假设 | 在小样本下结果是否仍然显著？ |
| **部署可行** | Exp 9 (Comp Cost) | 各 pipeline 阶段（滤波/协方差/对齐/切空间/LOSO）的耗时 | 实时部署的计算开销是否可接受？ |
| **论文呈现** | Exp 10 (Visualization) | 箱线图、柱状图、散点图等论文级图表 | 图表是否符合 TBME 期刊要求？ |

### 13.2 代码函数与输出文件追溯

| 实验 | 代码函数 (run_all_experiments.py) | 输入依赖 | JSON 输出 | 图表输出 |
|------|----------------------------------|----------|-----------|----------|
| Exp 1 | `run_table1()` (line ~150) | `evaluate_loso()` | `results_table1_all_*.json` | `individual_cor.pdf`, `channel_comparison.pdf` |
| Exp 2 | `run_fewshot()` → `evaluate_loso_fewshot()` | `evaluate_loso()` 变体 | `results_fewshot_*.json` | `fewshot_curve.pdf` |
| Exp 3 | `run_ablations()` | `evaluate_loso()` 3 变体 | `results_ablations_*.json` | — (paper text) |
| Exp 4 | `run_band_importance()` | `mean_riemann` + `invsqrtm` + `SelectKBest` | `results_band_importance_*.json` | `band_importance.pdf` |
| Exp 5 | `run_band_vs_global()` | `evaluate_loso()` + `evaluate_loso_global_band()` | `results_band_vs_global_*.json` | — (paper text) |
| Exp 6 | `run_sota_head_to_head()` | 4 个 aligner 类 | `results_sota_head_to_head_*.json` | — (paper table) |
| Exp 7 | `run_enhanced_stats()` | JSON 加载 + scipy 统计 | `results_enhanced_stats_*.json` | `paired_scatter.pdf`, `bland_altman.pdf`, `waterfall.pdf` |
| Exp 8 | `run_permutation_test()` | JSON 加载 + numpy 置换 | `results_permutation_test_*.json` | — (paper text) |
| Exp 9 | `run_computational_cost()` | `time.time()` 计时 | `results_comp_cost_*.json` | — (paper table) |
| Exp 10 | `run_visualization_analysis()` | Exp 1 & 4 数据 | — (无 JSON) | `subject_performance_boxplot.pdf` |

### 13.3 核心代码文件清单

| 文件 | 行数（约） | 作用 | 关键函数 |
|------|-----------|------|---------|
| `ric_da_core.py` | ~500 | 核心引擎：3 个对齐器类 + LOSO 评估 | `NoAlignment`, `EuclideanAlignment`, `RiemannianAlignment`, `precompute_all_subjects()`, `evaluate_loso()` |
| `run_all_experiments.py` | ~1500 | 一键实验启动器（10 个实验） | `run_table1`, `run_fewshot`, `run_ablations`, `run_band_importance`, `run_band_vs_global`, `run_sota_head_to_head`, `run_enhanced_stats`, `run_permutation_test`, `run_computational_cost`, `run_visualization_analysis` |
| `generate_pub_figures.py` | ~300 | 论文图表生成（PDF/TIFF） | `fig_individual_cor()`, `fig_fewshot()`, `fig_channel_comparison()`, `fig_band_importance()` |
| `config_da.py` | ~100 | 路径、被试列表、参数网格 | `RIC_DA_ROOT`, `DA_RESULTS_DIR`, `DA_FIGURES_DIR` |
| `paper/paper_tbme.tex` | ~750 | LaTeX 论文源文件 | 4 个 `\includegraphics{}` 引用 |

### 13.4 完整数据流图

```
SEED-VIG Raw EEG (23 subjects × 17ch × 885 epochs)
    ↓ [precompute_all_subjects] 一次性预计算（缓存）
SPD covariance matrices (per band: δ/θ/α/β/γ)
    ↓ [evaluate_loso, 22 train + 1 test, repeat for each aligner]
LOSO predictions (per subject × per aligner × per channel config)
    ↓ [save to JSON]
results/results_table1_all_*.json
    ↓ [generate_pub_figures]
figures/individual_cor.pdf + channel_comparison.pdf
    ↓ [LaTeX \includegraphics]
paper/paper_tbme.pdf (Fig.1, Fig.3)

Similar flow for: fewshot → Fig.2
                  band_importance → Fig.4
                  enhanced_stats → paired_scatter/bland_altman/waterfall
```

### 13.5 重新生成特定图表的命令

```bash
# 重新生成所有图表（不重跑实验）
python run_all_experiments.py --figs-only

# 只重跑某个实验
python run_all_experiments.py --full --only table1
python run_all_experiments.py --full --only fewshot
python run_all_experiments.py --full --only ablation
python run_all_experiments.py --full --only band_importance
python run_all_experiments.py --full --only band_vs_global
python run_all_experiments.py --full --only sota
python run_all_experiments.py --full --only enhanced_stats
python run_all_experiments.py --full --only permutation
python run_all_experiments.py --full --only comp_cost
python run_all_experiments.py --full --only visualization

# 重新编译论文
cd paper && pdflatex paper_tbme.tex && pdflatex paper_tbme.tex
```

### 13.6 数据版本管理建议

- **实验结果** (`results/*.json`)：按时间戳命名，保留所有历史版本以便追溯
- **图表文件** (`figures/*.pdf/svg/tiff`)：每次重新生成会覆盖，建议在论文投稿前锁定一组
- **日志文件** (`results/experiment_*.log`)：记录每次运行的完整 stdout，便于回溯错误
- **论文 PDF** (`paper/paper_tbme.pdf`)：随 LaTeX 源码同步编译，附 Git commit hash
