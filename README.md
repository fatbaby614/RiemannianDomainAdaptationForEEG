# ric_da — Riemannian Domain Adaptation for Cross-Subject EEG Vigilance Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![IEEE TBME](https://img.shields.io/badge/submitted%20to-IEEE%20TBME-blueviolet)](https://www.embs.org/tbm/)

A research codebase for **Riemannian parallel transport alignment** applied to cross-subject EEG vigilance regression. This work is submitted to *IEEE Transactions on Biomedical Engineering*.

**Core finding:** Riemannian parallel transport of SPD covariance matrices to a common reference (the identity) significantly improves leave-one-subject-out (LOSO) vigilance regression, raising the mean Pearson correlation from **0.585 → 0.692** (p = 0.0007, Cohen's d = 0.824, 21/23 subjects improved).

## Overview

Cross-subject generalization is a critical bottleneck for EEG-based vigilance estimation. Subject-specific physiological factors displace SPD covariance matrices to different regions of the Riemannian manifold, degrading tangent-space regression. Our method aligns each subject's covariance matrices to the identity via parallel transport, removing subject-specific bias while preserving vigilance-related variability.

### Pipeline

```
Raw EEG (subject s, 17ch × 200 Hz)
  │
  ├── Step 1: Filter-bank (δ/θ/α/β/γ Butterworth band-pass)
  ├── Step 2: OAS shrinkage covariance → SPD matrices  
  ├── Step 3: Riemannian parallel transport alignment ★
  │      G_s = Riemannian mean({C_si})
  │      C'_si = G_s^{-1/2} · C_si · G_s^{-1/2}
  ├── Step 4: Tangent space projection (TangentSpace)
  ├── Step 5: Multi-band feature concatenation (680 dim)
  ├── Step 6: SelectKBest (f_regression, k=150)
  ├── Step 7: StandardScaler
  ├── Step 8: RidgeCV regression → PERCLOS prediction
  └── Step 9: Temporal smoothing (window=3) → COR, RMSE
```

### Alignment methods implemented

| Class | Key | Description |
|-------|-----|-------------|
| `NoAlignment` | `none` | Identity baseline |
| `EuclideanAlignment` | `euclidean` | Whiten by Euclidean mean |
| `RiemannianAlignment` | `riemann` | Parallel transport via Riemannian mean (proposed) |

## Dataset

- **SEED-VIG**: 23 subjects, 17 EEG channels, 200 Hz sampling rate, ~885 epochs per subject
- **Target**: PERCLOS (percentage of eye closure) regression
- **Protocol**: Leave-one-subject-out (LOSO) cross-validation

## Key Results

| Metric | Baseline (No Alignment) | Ours (Riemannian) | Gain |
|--------|:----------------------:|:-----------------:|:----:|
| Pearson COR | 0.585 ± 0.230 | **0.692 ± 0.206** | +0.107 |
| RMSE | 0.278 | **0.195** | −0.083 |
| Subjects improved | — | **21/23** | — |
| Forehead (4ch) COR | 0.626 | **0.681** (98.3%) | +0.055 |

### Few-shot calibration
Even **one calibration epoch** recovers 98.4% of the full-alignment performance (COR 0.681), and **five epochs** fully match it (COR 0.693).

## Quick Start

### Dependencies

- Python 3.10+
- `pyriemann`, `scikit-learn`, `numpy`, `scipy`, `matplotlib`
- [SCAFBTSRegressor](https://github.com/your-org/SCAFBTSRegressor) (sibling project for filter-bank tangent-space feature extraction)

### Usage

```python
from ric_da_core import RiemannianAlignment, precompute_all_subjects, evaluate_loso

# Precompute covariance matrices (cached across all aligners)
covs, labels, _ = precompute_all_subjects(subjects, bands='5band', channels='all')

# Run LOSO with Riemannian alignment
aligner = RiemannianAlignment(metric='riemann')
results = evaluate_loso(subjects, aligner=aligner, all_covs=covs, all_labels=labels)

print(f"COR: {results['cor_mean']:.3f} ± {results['cor_std']:.3f}")
```

### Command-line entry points

```bash
# Full experiment suite (Table 1, few-shot, ablation, SOTA, band importance, etc.)
python run_all_experiments.py --full

# Quick verification (5 subjects)
python run_ric_da.py --quick

# Reproduce publication figures
python generate_pub_figures.py
```

## Project Structure

```
├── ric_da_core.py           # Core alignment algorithms + LOSO evaluation engine
├── run_ric_da.py             # Main experiment entry point
├── run_all_experiments.py    # Full experiment suite runner
├── generate_pub_figures.py   # Publication figure generation
├── config_da.py              # Experiment configuration
├── AGENTS.md                 # Developer onboarding & architecture docs
├── EXPERIMENT_DESIGN.md      # Detailed experimental design (Chinese)
├── paper/
│   ├── paper_tbme.tex        # Manuscript source
│   └── figures/              # Generated publication figures
└── results/                  # Experiment results (JSON)
    └── all_experiments_*.json  # Combined results
```

## Citation

```bibtex
@article{tan2026riemannian,
  title={Validating Riemannian Parallel Transport Alignment for Cross-Subject EEG Vigilance Estimation},
  author={Tan, Huang and Li, Xiangzhu and Zhang, Li and Yin, Guangqiang},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2026}
}
```

## License

This project is licensed under the MIT License.