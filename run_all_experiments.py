# -*- coding: utf-8 -*-
"""
run_all_experiments.py — 一键启动所有 TBME 实验

用法:
    python run_all_experiments.py [--quick] [--full] [--only TABLE]
                                   [--no-figs] [--figs-only] [--no-log]

选项:
    --quick             5 subjects, quick mode
    --full              23 subjects, full mode
    --only TABLE        Run specific experiment only
    --no-figs           Skip figure generation
    --figs-only         Generate figures only (from cached results)
    --no-log            Disable log file output
"""

import sys
import os
import json
import time
import argparse
import warnings
import logging
import traceback
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon, pearsonr
from scipy.ndimage import uniform_filter1d
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.distance import distance_riemann

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# PATH SETUP — 确保 SCAFBTSRegressor 在 sys.path 中
# ════════════════════════════════════════════════════════════════
_project_root = os.path.dirname(os.path.abspath(__file__))
_sibling = os.path.join(os.path.dirname(_project_root), 'SCAFBTSRegressor')
if os.path.isdir(_sibling) and _sibling not in sys.path:
    sys.path.insert(0, _sibling)

from sca_fbts_fast import apply_bandpass_filter

from ric_da_core import (
    NoAlignment, EuclideanAlignment, RiemannianAlignment,
    RiemannianProcrustesAlignment, get_aligner,
    precompute_all_subjects, precompute_subject_covs,
    evaluate_loso, evaluate_loso_global_band,
    paired_statistics, format_results_table,
)

from config_da import (
    SEED_VIG_SUBJECTS, QUICK_SUBJECTS,
    DA_RESULTS_DIR, DA_FIGURES_DIR, SEED_VIG_ROOT,
)

from data_loader import load_perclos, list_subjects
from utils import cor, rmse


# ════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════

class TeeStream:
    """双输出流：同时输出到控制台和文件。"""
    def __init__(self, file, first_line='', stream=sys.stdout):
        self.file = file
        self.stream = stream
        if first_line:
            self.stream.write(first_line)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
        self.stream.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


_log_handle = None
_original_stdout = None


def setup_logging(log_dir, label='experiment'):
    """设置日志输出：同时输出到控制台和日志文件。

    Returns:
        log_path (str): 日志文件路径
    """
    global _log_handle, _original_stdout
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'{label}_{ts}.log')
    _original_stdout = sys.stdout
    f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = TeeStream(f, stream=sys.stdout)
    _log_handle = f
    return log_path


def teardown_logging():
    """关闭日志输出，恢复原始 stdout。"""
    global _log_handle, _original_stdout
    if _log_handle is not None:
        _log_handle.close()
        _log_handle = None
    if _original_stdout is not None:
        sys.stdout = _original_stdout
        _original_stdout = None


# ════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════

def timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def save_partial(data, name):
    """保存部分结果。"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(DA_RESULTS_DIR, f'results_{name}_{ts}.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [Saved] {path}")
    return path


# ════════════════════════════════════════════════════════════════
# EXPERIMENT FUNCTIONS
# ════════════════════════════════════════════════════════════════

def run_table1(subjects, quick=False):
    """三种对齐器 × 三种通道配置。"""
    print("\n" + "=" * 70)
    print("TABLE 1: LOSO Main Experiment")
    print(f"  Subjects: {len(subjects)}, 3 aligners × 3 channels")
    print("=" * 70)

    channels_map = {
        'all': list(range(17)),
        'temporal': list(range(0, 6)),
        'forehead': list(range(0, 4)),
    }
    band = '5band'
    aligners = ['none', 'euclidean', 'riemann', 'rpa']
    results = {}

    for ch_name, ch_idx in channels_map.items():
        print(f"\n  {'─' * 30}")
        print(f"  Channels: {ch_name} ({len(ch_idx)}ch)")
        print(f"  {'─' * 30}")

        covs, labels, _ = precompute_all_subjects(
            subjects, bands=band, channels=ch_idx
        )
        ch_results = {}
        for a_name in aligners:
            t0 = time.time()
            aligner = get_aligner(a_name)
            _res = evaluate_loso(
                subjects, aligner=aligner, bands=band, channels=ch_idx,
                regressor='ridgecv', n_features=150, use_eog=False,
                verbose=False, all_covs=covs, all_labels=labels,
            )
            cor_scores = _res['cor_all']
            rmse_scores = _res['rmse_all']
            elapsed = time.time() - t0
            cor_arr = np.array(cor_scores)
            rmse_arr = np.array(rmse_scores)
            ch_results[a_name] = {
                'cor': cor_scores,
                'rmse': rmse_scores,
                'cor_mean': float(np.mean(cor_arr)),
                'cor_std': float(np.std(cor_arr)),
                'rmse_mean': float(np.mean(rmse_arr)),
                'rmse_std': float(np.std(rmse_arr)),
                'time_s': elapsed,
            }
            print(f"    {a_name:>10}: COR={np.mean(cor_arr):.4f}±{np.std(cor_arr):.4f}, "
                  f"RMSE={np.mean(rmse_arr):.4f} [{elapsed:.0f}s]")

        # 统计检验 (riemann vs none)
        if 'riemann' in ch_results and 'none' in ch_results:
            a = np.array(ch_results['riemann']['cor'])
            b = np.array(ch_results['none']['cor'])
            t_stat, p_val = ttest_rel(a, b)
            d = (np.mean(a) - np.mean(b)) / np.std(a - b, ddof=1)
            n_better = int(np.sum(np.array(a) > np.array(b)))
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            print(f"    {'─' * 25}")
            print(f"    Riemannian vs None: t={t_stat:.3f}, p={p_val:.4f} {sig}, "
                  f"d={d:.3f}, better={n_better}/{len(a)}")
            ch_results['stats'] = {
                't_stat': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(d),
                'n_better': n_better,
                'n_subjects': len(a),
            }
        results[ch_name] = ch_results
        print(f"  Channel config done. ({elapsed:.0f}s)")

    # 保存
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    final = {
        'experiment': 'table1',
        'timestamp': ts,
        'subjects': subjects,
        'config': {'bands': band, 'n_features': 150,
                   'regressor': 'ridgecv', 'use_eog': False},
        'results': results,
    }
    path = os.path.join(DA_RESULTS_DIR, f'results_table1_all_{ts}.json')
    with open(path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n  [Saved] {path}")
    return results


def evaluate_loso_fewshot(subjects, calib_sizes=None, n_repeats=3,
                          bands='5band', channels=None):
    """Few-shot LOSO: 用 n 个校准 epoch 估算测试被试的黎曼均值。

    流程:
        1. 预计算所有被试的 SPD 协方差（同 evaluate_loso）
        2. 对每个测试被试 s:
           a. 训练被试使用全量对齐
           b. 从测试被试随机选 n 个 epoch 做校准集
           c. 用校准集计算黎曼均值 G_calib
           d. 对齐测试被试全部 epoch
           e. 剩余 epoch 用于测试
        3. 每个 calib_size 重复 n_repeats 次取平均
    """
    if calib_sizes is None:
        calib_sizes = [0, 1, 5, 10, 50, 100]
    if channels is None:
        channels = list(range(17))

    print(f"\n{'=' * 70}")
    print(f"FEW-SHOT: Calibration Curve")
    print(f"  Calibration sizes: {calib_sizes}")
    print(f"  Repeats per size: {n_repeats}")
    print(f"{'=' * 70}")

    covs, labels, _ = precompute_all_subjects(subjects, bands=bands, channels=channels)

    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
    results = {size: [] for size in calib_sizes}

    for calib_size in calib_sizes:
        print(f"\n  Calibration size: n={calib_size}")
        repeat_cors = []
        for rep in range(n_repeats):
            subj_cors = []
            for test_subj in subjects:
                train_subjs = [s for s in subjects if s != test_subj]

                # 训练被试对齐 (全量)
                train_pooled_per_band = []
                for band in freq_bands:
                    aligned = []
                    for s in train_subjs:
                        G = mean_riemann(covs[s][band])
                        G_inv_sqrt = invsqrtm(G)
                        aligned.append(np.array([G_inv_sqrt @ c @ G_inv_sqrt
                                                  for c in covs[s][band]]))
                    train_pooled_per_band.append(np.concatenate(aligned, axis=0))

                # 测试被试: 用 calib_size 个 epoch 做校准
                test_covs_orig = covs[test_subj]
                n_epochs = labels[test_subj].shape[0]

                if calib_size == 0:
                    # 零样本 = 用训练被试的 grand Riemannian mean 对齐测试被试
                    # 论文: "the test subject is aligned using the grand mean
                    # of the training subjects' Riemannian means"
                    test_aligned = {}
                    for band_key in covs[test_subj]:
                        # 计算训练被试的 grand Riemannian mean
                        train_covs_band = np.concatenate(
                            [covs[s][band_key] for s in train_subjs], axis=0
                        )
                        G_grand = mean_riemann(train_covs_band)
                        G_grand_inv_sqrt = invsqrtm(G_grand)
                        test_aligned[band_key] = np.array([
                            G_grand_inv_sqrt @ c @ G_grand_inv_sqrt
                            for c in covs[test_subj][band_key]
                        ])
                else:
                    calib_size_actual = min(calib_size, n_epochs)
                    rng = np.random.default_rng(rep + 42)
                    calib_idx = rng.choice(n_epochs, size=calib_size_actual, replace=False)
                    test_aligned = {}
                    for band_key in covs[test_subj]:
                        band_covs = covs[test_subj][band_key]
                        G_calib = mean_riemann(band_covs[calib_idx])
                        G_inv_sqrt = invsqrtm(G_calib)
                        test_aligned[band_key] = np.array([
                            G_inv_sqrt @ c @ G_inv_sqrt for c in band_covs
                        ])

                # 切空间投影 + 回归
                train_feats_list = []
                test_feats_list = []
                for b_idx, b_key in enumerate(freq_bands):
                    ts = TangentSpace(metric='riemann')
                    train_feats_list.append(ts.fit_transform(train_pooled_per_band[b_idx]))
                    test_feats_list.append(ts.transform(test_aligned[b_key]))
                train_feats = np.concatenate(train_feats_list, axis=1)
                test_feats = np.concatenate(test_feats_list, axis=1)

                # 获取训练被试的标签（与 train_feats 对应）
                train_labels = np.concatenate([labels[s] for s in train_subjs], axis=0)

                # 特征选择 + 标准化 + RidgeCV
                selector = SelectKBest(f_regression, k=min(150, train_feats.shape[1]))
                train_selected = selector.fit_transform(train_feats, train_labels)
                test_selected = selector.transform(test_feats)

                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(train_selected)
                test_scaled = scaler.transform(test_selected)

                reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                reg.fit(train_scaled, train_labels)
                y_pred = reg.predict(test_scaled)
                y_pred = uniform_filter1d(y_pred, size=3)

                cor_val = cor(labels[test_subj], y_pred)
                subj_cors.append(cor_val)

            mean_cor = float(np.mean(subj_cors))
            std_cor = float(np.std(subj_cors))
            repeat_cors.append(mean_cor)
            print(f"    rep={rep+1}: COR={mean_cor:.4f}±{std_cor:.4f}")

        results[calib_size] = {
            'mean': float(np.mean(repeat_cors)),
            'std': float(np.std(repeat_cors)),
            'per_repeat': repeat_cors,
        }
        print(f"  n={calib_size:>3}: COR={results[calib_size]['mean']:.4f}")

    return results


def run_fewshot(subjects, quick=False):
    """Few-shot 实验的包装函数。"""
    n_repeats = 3 if not quick else 2
    calib_sizes = [0, 1, 5, 10, 50, 100] if not quick else [0, 1, 5, 50]
    results = evaluate_loso_fewshot(subjects, calib_sizes=calib_sizes,
                                     n_repeats=n_repeats)
    save_partial(results, 'fewshot')
    return results


def run_ablations(subjects, quick=False):
    """消融实验：参考点 / 协方差估计器 / 回归器。"""
    print("\n" + "=" * 70)
    print("ABLATION: Reference / Estimator / Regressor")
    print("=" * 70)

    n_subj = min(10, len(subjects)) if not quick else min(5, len(subjects))
    subjs = subjects[:n_subj]
    band = '5band'
    ch = list(range(17))
    results = {}
    refs = ['identity', 'grand_mean']
    estimators = ['oas', 'scm', 'lwf']
    regressors = ['ridgecv', 'svr']
    n_features = 150

    # 参考点消融
    print("\n  Reference point ablation:")
    covs, labels, _ = precompute_all_subjects(subjs, bands=band, channels=ch)
    ref_results = {}
    for ref in refs:
        if ref == 'identity':
            aligner = RiemannianAlignment()
        else:
            aligner = RiemannianAlignment(reference=ref)
        _res = evaluate_loso(
            subjs, aligner=aligner, bands=band, channels=ch,
            regressor='ridgecv', n_features=n_features,
            verbose=False, all_covs=covs, all_labels=labels,
        )
        cor_scores = _res['cor_all']
        rmse_scores = _res['rmse_all']
        cor_arr = np.array(cor_scores)
        ref_results[ref] = {
            'cor_mean': float(np.mean(cor_arr)),
            'cor_std': float(np.std(cor_arr)),
        }
        print(f"    {ref:<15}: COR={np.mean(cor_arr):.4f}±{np.std(cor_arr):.4f}")
    results['reference'] = ref_results

    # 估计器消融
    print("\n  Estimator ablation:")
    est_results = {}
    for est in estimators:
        covs_e, labels_e, _ = precompute_all_subjects(
            subjs, bands=band, channels=ch, estimator=est
        )
        for a_name in ['none', 'riemann']:
            aligner = get_aligner(a_name)
            _res = evaluate_loso(
                subjs, aligner=aligner, bands=band, channels=ch,
                regressor='ridgecv', n_features=n_features,
                verbose=False, all_covs=covs_e, all_labels=labels_e,
            )
            cor_scores = _res['cor_all']
            rmse_scores = _res['rmse_all']
            cor_arr = np.array(cor_scores)
            key = f"{est}+{a_name}"
            est_results[key] = {
                'cor_mean': float(np.mean(cor_arr)),
                'cor_std': float(np.std(cor_arr)),
            }
            print(f"    {est:<5} + {a_name:<10}: COR={np.mean(cor_arr):.4f}±{np.std(cor_arr):.4f}")
    results['estimator'] = est_results

    # 回归器消融
    print("\n  Regressor ablation:")
    reg_results = {}
    for reg in regressors:
        for a_name in ['none', 'riemann']:
            aligner = get_aligner(a_name)
            _res = evaluate_loso(
                subjs, aligner=aligner, bands=band, channels=ch,
                regressor=reg, n_features=n_features,
                verbose=False, all_covs=covs, all_labels=labels,
            )
            cor_scores = _res['cor_all']
            rmse_scores = _res['rmse_all']
            cor_arr = np.array(cor_scores)
            key = f"{reg}+{a_name}"
            reg_results[key] = {
                'cor_mean': float(np.mean(cor_arr)),
                'cor_std': float(np.std(cor_arr)),
            }
            print(f"    {reg:<10} + {a_name:<10}: COR={np.mean(cor_arr):.4f}±{np.std(cor_arr):.4f}")
    results['regressor'] = reg_results

    save_partial(results, 'ablations')
    return results


def run_band_importance(subjects, quick=False):
    """频段重要性分析。"""
    print("\n" + "=" * 70)
    print("BAND IMPORTANCE: feature selection per band")
    print("=" * 70)

    n_subj = min(10, len(subjects)) if not quick else min(5, len(subjects))
    subjs = subjects[:n_subj]
    band = '5band'
    ch = list(range(17))
    results = {}

    covs, labels, _ = precompute_all_subjects(subjs, bands=band, channels=ch)
    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    for a_name in ['none', 'euclidean', 'riemann']:
        print(f"\n  Aligner: {a_name}")
        aligner = get_aligner(a_name)
        band_counts = {b: 0 for b in band_names}

        for test_subj in subjs:
            train_subjs = [s for s in subjs if s != test_subj]
            train_pooled = []
            for b_idx, band_key in enumerate(freq_bands):
                # 使用 aligner 对齐训练被试（与 evaluate_loso 一致）
                band_covs_train = {s: covs[s][band_key] for s in train_subjs}
                band_covs_aligned = aligner.fit_transform(band_covs_train)
                train_pooled.append(np.concatenate(
                    [band_covs_aligned[s] for s in train_subjs], axis=0
                ))

            train_feats_list = []
            for band_data in train_pooled:
                ts = TangentSpace(metric='riemann')
                train_feats_list.append(ts.fit_transform(band_data))
            train_feats = np.concatenate(train_feats_list, axis=1)
            train_labels = np.concatenate([labels[s] for s in train_subjs], axis=0)

            selector = SelectKBest(f_regression, k=min(150, train_feats.shape[1]))
            selector.fit(train_feats, train_labels)

            # 每个频段的特征数: 总维度 153 (17*18/2), 5 bands = 765
            dim_per_band = 153
            for b_idx, b_name in enumerate(band_names):
                start = b_idx * dim_per_band
                end = start + dim_per_band
                mask = selector.get_support()[start:end]
                band_counts[b_name] += int(np.sum(mask))

        results[a_name] = band_counts
        total = sum(band_counts.values())
        print(f"    Total features selected: {total}")
        for b_name in band_names:
            pct = band_counts[b_name] / total * 100
            print(f"      {b_name:<8}: {band_counts[b_name]:>4} ({pct:.1f}%)")

    save_partial(results, 'band_importance')
    return results


def run_band_vs_global(subjects, quick=False):
    """分频段对齐 vs 全局对齐对比。"""
    print("\n" + "=" * 70)
    print("BAND vs GLOBAL: per-band vs all-band alignment")
    print("=" * 70)

    n_subj = min(10, len(subjects)) if not quick else min(5, len(subjects))
    subjs = subjects[:n_subj]
    band = '5band'
    ch = list(range(17))
    results = {}

    covs, labels, _ = precompute_all_subjects(subjs, bands=band, channels=ch)

    # 分频段对齐 (per-band, 默认)
    print("\n  Per-band alignment:")
    aligner = RiemannianAlignment()
    _res = evaluate_loso(
        subjs, aligner=aligner, bands=band, channels=ch,
        regressor='ridgecv', n_features=150,
        verbose=False, all_covs=covs, all_labels=labels,
    )
    cor_scores = _res['cor_all']
    rmse_scores = _res['rmse_all']
    cor_arr = np.array(cor_scores)
    results['per_band'] = {
        'cor_mean': float(np.mean(cor_arr)),
        'cor_std': float(np.std(cor_arr)),
    }
    print(f"    COR={np.mean(cor_arr):.4f}±{np.std(cor_arr):.4f}")

    # 全局对齐 (all-band)
    print("\n  Global alignment:")
    _res_g = evaluate_loso_global_band(
        subjs, aligner=aligner, bands=band, channels=ch,
        regressor='ridgecv', n_features=150,
        verbose=False, all_covs=covs, all_labels=labels,
    )
    cor_scores_g = _res_g['cor_all']
    rmse_scores_g = _res_g['rmse_all']
    cor_arr_g = np.array(cor_scores_g)
    results['global'] = {
        'cor_mean': float(np.mean(cor_arr_g)),
        'cor_std': float(np.std(cor_arr_g)),
    }
    print(f"    COR={np.mean(cor_arr_g):.4f}±{np.std(cor_arr_g):.4f}")

    save_partial(results, 'band_vs_global')
    return results


def run_visualization_analysis(subjects, quick=False):
    """可视化分析：箱线图、频段重要性柱状图、对齐散点图。"""
    print("\n" + "=" * 70)
    print("VISUALIZATION ANALYSIS: boxplot, band importance, scatter")
    print("=" * 70)

    n_subj = min(10, len(subjects)) if not quick else min(5, len(subjects))
    subjs = subjects[:n_subj]
    band = '5band'
    ch = list(range(17))
    freq_bands_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]

    # ── 箱线图：三种对齐器 COR 分布 ──
    print("\n  Generating boxplot...")
    covs, labels, _ = precompute_all_subjects(subjs, bands=band, channels=ch)
    cor_data = {}
    for a_name in ['none', 'euclidean', 'riemann']:
        aligner = get_aligner(a_name)
        _res = evaluate_loso(
            subjs, aligner=aligner, bands=band, channels=ch,
            regressor='ridgecv', n_features=150,
            verbose=False, all_covs=covs, all_labels=labels,
        )
        cor_data[a_name] = _res['cor_all']

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    positions = [1, 2, 3]
    labels_bp = ['None', 'Euclidean', 'Riemannian']
    bp_data = [cor_data[a] for a in ['none', 'euclidean', 'riemann']]
    bp = ax.boxplot(bp_data, positions=positions, labels=labels_bp,
                    patch_artist=True, widths=0.5)
    colors_box = ['#ff9999', '#99ccff', '#99ff99']
    for patch, c in zip(bp['boxes'], colors_box):
        patch.set_facecolor(c)
    ax.set_ylabel('COR', fontsize=12)
    ax.set_title('LOSO COR Distribution by Aligner', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    fig_path = os.path.join(DA_FIGURES_DIR, 'subject_performance_boxplot.pdf')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {fig_path}")

    # ── 频段重要性柱状图 ──
    print("\n  Generating band importance bar chart...")
    band_counts_all = {b: 0 for b in freq_bands_names}
    dim_per_band = 153
    aligner_riem = get_aligner('riemann')
    for test_subj in subjs:
        train_subjs = [s for s in subjs if s != test_subj]
        train_pooled = []
        for b_idx, band_key in enumerate(freq_bands):
            band_covs_train = {s: covs[s][band_key] for s in train_subjs}
            band_covs_aligned = aligner_riem.fit_transform(band_covs_train)
            train_pooled.append(np.concatenate(
                [band_covs_aligned[s] for s in train_subjs], axis=0
            ))
        train_feats_list = []
        for band_data in train_pooled:
            ts = TangentSpace(metric='riemann')
            train_feats_list.append(ts.fit_transform(band_data))
        train_feats = np.concatenate(train_feats_list, axis=1)
        train_labels = np.concatenate([labels[s] for s in train_subjs], axis=0)
        selector = SelectKBest(f_regression, k=min(150, train_feats.shape[1]))
        selector.fit(train_feats, train_labels)
        for b_idx, b_name in enumerate(freq_bands_names):
            start = b_idx * dim_per_band
            end = start + dim_per_band
            mask = selector.get_support()[start:end]
            band_counts_all[b_name] += int(np.sum(mask))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors_band = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ax.bar(freq_bands_names, [band_counts_all[b] for b in freq_bands_names],
           color=colors_band)
    ax.set_ylabel('Features Selected', fontsize=12)
    ax.set_title('Band Importance (SelectKBest, k=150)', fontsize=14)
    fig_path = os.path.join(DA_FIGURES_DIR, 'band_importance_bar.pdf')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {fig_path}")

    # ── 对齐散点图 (None vs Riemannian) ──
    print("\n  Generating alignment scatter plot...")
    cor_none = np.array(cor_data['none'])
    cor_riem = np.array(cor_data['riemann'])
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.scatter(cor_none, cor_riem, c='steelblue', s=60, alpha=0.7, edgecolors='k')
    lims = [min(cor_none.min(), cor_riem.min()) - 0.05,
            max(cor_none.max(), cor_riem.max()) + 0.05]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
    ax.plot(lims, [l + 0.05 for l in lims], 'k:', alpha=0.3)
    ax.plot(lims, [l - 0.05 for l in lims], 'k:', alpha=0.3)
    n_better = int(np.sum(cor_riem > cor_none))
    ax.set_xlabel('None COR', fontsize=12)
    ax.set_ylabel('Riemannian COR', fontsize=12)
    ax.set_title(f'Alignment Scatter ({n_better}/{len(cor_none)} improved)',
                 fontsize=14)
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig_path = os.path.join(DA_FIGURES_DIR, 'alignment_scatter.pdf')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {fig_path}")

    return {
        'cor_none': cor_none.tolist(),
        'cor_riemann': cor_riem.tolist(),
        'n_better': n_better,
        'n_subjects': len(cor_none),
    }


# ════════════════════════════════════════════════════════════════
# SOTA HEAD-TO-HEAD
# ════════════════════════════════════════════════════════════════

def _paired_cohens_d(a, b):
    """配对 Cohen's d: mean(a-b) / std(a-b, ddof=1).
    负值表示 a > b.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.mean(diff) / np.std(diff, ddof=1))


def _bootstrap_ci_diff(a, b, n_boot=2000, ci=95):
    """Bootstrap 95% CI for paired Cohen's d.
    Returns (ci_low, ci_high).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(2024)
    d_boot = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d_boot[i] = _paired_cohens_d(a[idx], b[idx])
    alpha = (100 - ci) / 2
    return float(np.percentile(d_boot, alpha)), float(np.percentile(d_boot, 100 - alpha))


def _find_latest_table1_json(channel='all'):
    """Find the latest table1 result JSON.

    run_table1 saves all channels in a single file named
    results_table1_all_{ts}.json, so we always search for that pattern
    regardless of the channel argument.
    """
    pattern = 'results_table1_all_'
    candidates = [f for f in os.listdir(DA_RESULTS_DIR)
                  if f.startswith(pattern) and f.endswith('.json')]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return os.path.join(DA_RESULTS_DIR, candidates[0])


def _load_table1_or_run(subjects, channel='all', quick=False):
    """Load cached table1 results or run if not available."""
    fpath = _find_latest_table1_json(channel)
    if fpath and os.path.getsize(fpath) > 0:
        try:
            with open(fpath) as f:
                data = json.load(f)
            return data['results']
        except (json.JSONDecodeError, KeyError):
            pass
    print(f"  [Cache miss] Running table1 ({channel})...")
    # 使用低精度快速模式
    n_subj = min(10, len(subjects)) if not quick else min(5, len(subjects))
    return run_table1(subjects[:n_subj], quick=quick)


def run_sota_head_to_head(subjects, quick=False):
    """P2-9: 与 SOTA 直接 head-to-head.

    SOTA baselines:
      - He 2020: Euclidean Alignment (Euclidean mean whitening)
      - Rodrigues 2019: Riemannian Procrustes Analysis (RPA)
      - Zanini 2018: Riemannian TL = RiemannianAlignment(reference='grand_mean')

    我们 (this work): Riemannian parallel transport (reference='identity')

    报告:
      - 4×3 表格 (aligner × channel)
      - 配对 t 检验: 我们 vs 每个 SOTA, 含 Cohen's d 与 bootstrap CI
      - 4-aligner bar chart 对比
    """
    print("\n" + "=" * 70)
    print("SOTA HEAD-TO-HEAD: EA vs RPA vs Zanini vs Ours")
    print("=" * 70)

    channels_map = {
        'all': list(range(17)),
        'temporal': list(range(0, 6)),
        'forehead': list(range(0, 4)),
    }
    band = '5band'
    results = {}

    for ch_name, ch_idx in channels_map.items():
        print(f"\n  {'─' * 30}")
        print(f"  Channels: {ch_name} ({len(ch_idx)}ch)")
        print(f"  {'─' * 30}")

        ch_results = {}
        covs, labels, _ = precompute_all_subjects(subjects, bands=band, channels=ch_idx)

        # 定义要比较的对齐器
        aligners = {
            'none': NoAlignment(),
            'ea': EuclideanAlignment(),
            'rpa': RiemannianProcrustesAlignment(),
            'zanini': RiemannianAlignment(reference='grand_mean'),
            'ours': RiemannianAlignment(reference='identity'),
        }

        for a_name, aligner in aligners.items():
            _res = evaluate_loso(
                subjects, aligner=aligner, bands=band, channels=ch_idx,
                regressor='ridgecv', n_features=150, use_eog=False,
                verbose=False, all_covs=covs, all_labels=labels,
            )
            cor_scores = _res['cor_all']
            rmse_scores = _res['rmse_all']
            cor_arr = np.array(cor_scores)
            ch_results[a_name] = {
                'cor': cor_scores,
                'rmse': rmse_scores,
                'cor_mean': float(np.mean(cor_arr)),
                'cor_std': float(np.std(cor_arr)),
            }
            print(f"    {a_name:>8}: COR={np.mean(cor_arr):.4f}±{np.std(cor_arr):.4f}")

        # 统计检验: 我们 vs 每个 SOTA
        stats = {}
        ours_cor = np.array(ch_results['ours']['cor'])
        for other in ['none', 'ea', 'rpa', 'zanini']:
            other_cor = np.array(ch_results[other]['cor'])
            t_stat, p_val = ttest_rel(ours_cor, other_cor)
            d = _paired_cohens_d(ours_cor, other_cor)
            ci_low, ci_high = _bootstrap_ci_diff(ours_cor, other_cor)
            n_better = int(np.sum(ours_cor > other_cor))
            stats[other] = {
                't_stat': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': d,
                'ci95_d': [ci_low, ci_high],
                'n_better': n_better,
                'n_subjects': len(ours_cor),
            }
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            print(f"      vs {other:<8}: t={t_stat:.3f}, p={p_val:.4f} {sig}, "
                  f"d={d:.3f} [{ci_low:.3f}, {ci_high:.3f}], "
                  f"better={n_better}/{len(ours_cor)}")

        ch_results['stats'] = stats
        results[ch_name] = ch_results

    # 保存
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    final = {
        'experiment': 'sota_head_to_head',
        'timestamp': ts,
        'subjects': subjects,
        'config': {'bands': band, 'n_features': 150, 'regressor': 'ridgecv'},
        'results': results,
    }
    path = os.path.join(DA_RESULTS_DIR, f'results_sota_head_to_head_{ts}.json')
    with open(path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n  [Saved] {path}")

    # 生成 SOTA 对比柱状图
    _plot_sota_comparison(results)

    return results


def _plot_sota_comparison(results):
    """Plot SOTA head-to-head comparison bar chart."""
    ch_names = ['all', 'temporal', 'forehead']
    aligner_labels = ['None', 'EA', 'RPA', 'Zanini', 'Ours']
    aligner_keys = ['none', 'ea', 'rpa', 'zanini', 'ours']
    colors = ['#ff9999', '#99ccff', '#ffcc99', '#cc99ff', '#99ff99']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, ch_name in zip(axes, ch_names):
        if ch_name not in results:
            continue
        ch_data = results[ch_name]
        means = [ch_data[k]['cor_mean'] for k in aligner_keys]
        stds = [ch_data[k]['cor_std'] for k in aligner_keys]
        bars = ax.bar(aligner_labels, means, yerr=stds, capsize=5,
                      color=colors, edgecolor='gray', linewidth=1)
        # Highlight Ours
        bars[4].set_edgecolor('red')
        bars[4].set_linewidth(3)
        ax.set_title(f'{ch_name.upper()} Channels', fontsize=13)
        ax.set_ylabel('COR', fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        # Stats text
        ours_mean = ch_data['ours']['cor_mean']
        for i, (k, m) in enumerate(zip(aligner_keys, means)):
            if k != 'ours':
                d = ch_data.get('stats', {}).get(k, {}).get('cohens_d', 0)
                sig = ch_data.get('stats', {}).get(k, {}).get('p_value', 1)
                star = '***' if sig < 0.001 else '**' if sig < 0.01 else '*' if sig < 0.05 else 'ns'
                ax.annotate(f'd={d:.2f} {star}',
                            xy=(i, m + stds[i] + 0.02),
                            fontsize=7, ha='center', rotation=45)

    fig.suptitle('SOTA Comparison (LOSO COR)', fontsize=15, y=1.02)
    fig_path = os.path.join(DA_FIGURES_DIR, 'sota_comparison.pdf')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {fig_path}")


# ════════════════════════════════════════════════════════════════
# ENHANCED STATS: Bootstrap CI + Bland-Altman + Waterfall
# ════════════════════════════════════════════════════════════════

def run_enhanced_stats(subjects, quick=False):
    """P0-3: 增强统计 — Bootstrap CI, Bland-Altman, Waterfall.

    重用 table1 缓存（只需 per-subject COR 数组）。
    不跑 LOSO，纯统计重算 + 图表。
    """
    print("\n" + "=" * 70)
    print("ENHANCED STATS: Bootstrap CI + Bland-Altman + Waterfall")
    print("=" * 70)

    channels = ['all', 'temporal', 'forehead']
    results = {}

    for ch_name in channels:
        # 加载 table1 缓存
        fpath = _find_latest_table1_json(ch_name)
        if fpath is None:
            print(f"  [Skip] No table1 cache for {ch_name}")
            continue
        with open(fpath) as f:
            table1_data = json.load(f)

        ch_results = table1_data['results'].get(ch_name, {})
        none_cor_raw = ch_results.get('none', {}).get('cor', None)
        riemann_cor_raw = ch_results.get('riemann', {}).get('cor', None)

        if none_cor_raw is None or riemann_cor_raw is None:
            print(f"  [Skip] Missing data for {ch_name}")
            continue

        a, b = np.asarray(none_cor_raw, dtype=float), np.asarray(riemann_cor_raw, dtype=float)
        diff = b - a
        n = len(a)

        # 配对 t 检验
        t_stat, p_val = ttest_rel(b, a)
        d = _paired_cohens_d(b, a)

        # Bootstrap CI for Cohen's d
        ci_low_d, ci_high_d = _bootstrap_ci_diff(b, a)

        # Bootstrap CI for mean diff
        rng = np.random.default_rng(2024)
        boot_diffs = np.zeros(2000)
        for i in range(2000):
            idx = rng.integers(0, n, size=n)
            boot_diffs[i] = np.mean(b[idx] - a[idx])
        ci_low_diff = float(np.percentile(boot_diffs, 2.5))
        ci_high_diff = float(np.percentile(boot_diffs, 97.5))

        # Wilcoxon
        w_stat, wilcoxon_p = wilcoxon(b, a, alternative='two-sided')

        n_better = int(np.sum(b > a))
        n_worse = int(np.sum(b < a))
        n_tied = n - n_better - n_worse

        ch_stats = {
            'mean_cor_none': float(np.mean(a)),
            'mean_cor_riemann': float(np.mean(b)),
            'mean_diff': float(np.mean(diff)),
            'ci95_diff': [ci_low_diff, ci_high_diff],
            'cohens_d': d,
            'ci95_d': [ci_low_d, ci_high_d],
            'paired_t_p': float(p_val),
            'wilcoxon_p': float(wilcoxon_p),
            'n_better': n_better,
            'n_worse': n_worse,
            'n_tied': n_tied,
        }
        results[ch_name] = ch_stats
        print(f"\n  {ch_name}: mean_diff={np.mean(diff):.4f} "
              f"[{ci_low_diff:.4f}, {ci_high_diff:.4f}], "
              f"d={d:.3f} [{ci_low_d:.3f}, {ci_high_d:.3f}], "
              f"p={p_val:.4f}, wilcoxon_p={wilcoxon_p:.4f}, "
              f"better={n_better}/{n}")

    # 图表: 配对散点图 (三通道)
    n_ch = sum(1 for ch in channels if ch in results)
    if n_ch > 0:
        fig, axes = plt.subplots(1, n_ch, figsize=(6 * n_ch, 5))
        if n_ch == 1:
            axes = [axes]
        for ax, ch_name in zip(axes, channels):
            if ch_name not in results:
                continue
            fpath = _find_latest_table1_json(ch_name)
            with open(fpath) as f:
                table1_data = json.load(f)
            ch_data = table1_data['results'][ch_name]
            a_ch = np.array(ch_data['none']['cor'])
            b_ch = np.array(ch_data['riemann']['cor'])
            d_val = _paired_cohens_d(b_ch, a_ch)
            ax.scatter(a_ch, b_ch, c='steelblue', s=40, alpha=0.6, edgecolors='k')
            lims = [min(a_ch.min(), b_ch.min()) - 0.05,
                    max(a_ch.max(), b_ch.max()) + 0.05]
            ax.plot(lims, lims, 'k--', alpha=0.5)
            ax.set_xlabel('None COR', fontsize=11)
            ax.set_ylabel('Riemannian COR', fontsize=11)
            ax.set_title(f'{ch_name} (d={d_val:.3f})', fontsize=13)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')
        fig_path = os.path.join(DA_FIGURES_DIR, 'paired_scatter.pdf')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  [Saved] {fig_path}")

        # Bland-Altman (仅 17ch)
        if 'all' in results:
            fpath = _find_latest_table1_json('all')
            with open(fpath) as f:
                table1_data = json.load(f)
            ch_data = table1_data['results']['all']
            a_all = np.array(ch_data['none']['cor'])
            b_all = np.array(ch_data['riemann']['cor'])

            means = (a_all + b_all) / 2
            diffs = b_all - a_all
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.scatter(means, diffs, c='steelblue', s=40, alpha=0.6, edgecolors='k')
            ax.axhline(mean_diff, color='red', linestyle='-', linewidth=1.5, label=f'Mean diff={mean_diff:.4f}')
            ax.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--', linewidth=1, label=f'+1.96SD={mean_diff + 1.96 * std_diff:.4f}')
            ax.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--', linewidth=1, label=f'-1.96SD={mean_diff - 1.96 * std_diff:.4f}')
            ax.set_xlabel('Mean of None and Riemannian COR', fontsize=11)
            ax.set_ylabel('Difference (Riemannian - None)', fontsize=11)
            ax.set_title('Bland-Altman Plot (17ch)', fontsize=13)
            ax.legend(fontsize=9)
            fig_path = os.path.join(DA_FIGURES_DIR, 'bland_altman.pdf')
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  [Saved] {fig_path}")

        # 瀑布图 (仅 17ch)
        if 'all' in results:
            fpath = _find_latest_table1_json('all')
            with open(fpath) as f:
                table1_data = json.load(f)
            ch_data = table1_data['results']['all']
            a_all = np.array(ch_data['none']['cor'])
            b_all = np.array(ch_data['riemann']['cor'])
            diff_all = b_all - a_all
            sort_idx = np.argsort(diff_all)[::-1]

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            colors_wf = ['green' if d > 0 else 'red' for d in diff_all[sort_idx]]
            ax.bar(range(len(diff_all)), diff_all[sort_idx], color=colors_wf, edgecolor='gray', linewidth=0.5)
            ax.axhline(y=0, color='black', linewidth=0.8)
            ax.set_xlabel('Subject', fontsize=11)
            ax.set_ylabel('ΔCOR (Riemannian - None)', fontsize=11)
            ax.set_title(f'Waterfall Plot (17ch, {int(np.sum(diff_all > 0))}/{len(diff_all)} improved)', fontsize=13)
            fig_path = os.path.join(DA_FIGURES_DIR, 'waterfall.pdf')
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  [Saved] {fig_path}")

    # 保存
    save_partial(results, 'enhanced_stats')
    return results


# ════════════════════════════════════════════════════════════════
# PERMUTATION TEST: 非参数配对置换检验
# ════════════════════════════════════════════════════════════════

def run_permutation_test(subjects, quick=False):
    """P0-4: 置换检验——非参数配对检验.

    原理：
      随机翻转 aligner 标签 n_perm 次，以 mean_diff 为统计量，
      给出经验 p-value.

      不假设正态性, 对 LOSO 的依赖结构更稳健.

    重用 table1 缓存（只需 per-subject COR 数组），额外 n_perm=10000
    """
    print("\n" + "=" * 70)
    print("PERMUTATION TEST: non-parametric paired permutation test")
    print("=" * 70)

    channels = ['all', 'temporal', 'forehead']
    n_perm = 10000 if not quick else 999
    results = {}

    for ch_name in channels:
        fpath = _find_latest_table1_json(ch_name)
        if fpath is None:
            print(f"  [Skip] No table1 cache for {ch_name}")
            continue
        with open(fpath) as f:
            table1_data = json.load(f)

        ch_results = table1_data['results'].get(ch_name, {})
        none_cor = ch_results.get('none', {}).get('cor', None)
        riemann_cor = ch_results.get('riemann', {}).get('cor', None)
        if none_cor is None or riemann_cor is None:
            print(f"  [Skip] Missing data for {ch_name}")
            continue

        a = np.asarray(none_cor, dtype=float)
        b = np.asarray(riemann_cor, dtype=float)
        observed_diff = np.mean(b - a)
        n = len(a)

        rng = np.random.default_rng(2024)
        perm_diffs = np.zeros(n_perm)
        for i in range(n_perm):
            flip = rng.integers(0, 2, size=n).astype(float) * 2 - 1
            perm_diffs[i] = np.mean(flip * (b - a))

        p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))

        ch_perm = {
            'observed_mean_diff': float(observed_diff),
            'p_value': p_value,
            'n_perm': n_perm,
            'n_subjects': n,
        }
        results[ch_name] = ch_perm

        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
        print(f"\n  {ch_name}: mean_diff={observed_diff:.4f}, "
              f"p={p_value:.4f} {sig} ({n_perm} permutations)")

    # 保存
    save_partial(results, 'permutation_test')
    return results


# ════════════════════════════════════════════════════════════════
# INTERACTION ANALYSIS: 协方差估计器 × 对齐方法
# ════════════════════════════════════════════════════════════════

def run_computational_cost(subjects, quick=False):
    """P1-5: 计算成本报表, 各 pipeline 阶段用时.

    报告: filter+cov / mean_riemann / tangent space / fit+pred
    (None vs Riemannian) 的秒数 (per subject × 885 epochs).
    """
    print("\n" + "=" * 70)
    print("COMPUTATIONAL COST: pipeline timing")
    print("=" * 70)

    n_subj = min(5, len(subjects)) if not quick else min(3, len(subjects))
    subjs = subjects[:n_subj]
    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]

    # ⭐ FIX: 一次性预计算所有被试的协方差矩阵
    print(f"  [Precompute] {len(subjs)} subjects (5band, all)...")
    t0_pre = time.time()
    cache = precompute_all_subjects(subjs, bands='5band', channels='all')
    all_covs, all_labels = cache[0], cache[1]
    pre_time = time.time() - t0_pre
    filter_cov_per_subject = pre_time / len(subjs)
    print(f"  Done. ({pre_time:.1f}s, ~{filter_cov_per_subject:.1f}s/subject)")

    n_epochs = all_labels[subjs[0]].shape[0]

    timings = {
        'none': defaultdict(list),
        'riemann': defaultdict(list),
    }

    for s in subjs:
        print(f"  Subject: {s}")

        for AlignerName, AlignerCls in [('none', NoAlignment),
                                         ('riemann', RiemannianAlignment)]:
            # ── mean_riemann 计时 ──
            t0 = time.time()
            _ = [mean_riemann(all_covs[s][b_key]) for b_key in freq_bands]
            t_mr = time.time() - t0
            timings[AlignerName]['mean_riemann'].append(t_mr)

            # ── 切空间投影 计时 ──
            t0 = time.time()
            for b_key in freq_bands:
                covs_band = all_covs[s][b_key]
                ts = TangentSpace(metric='riemann')
                _ = ts.fit_transform(covs_band)
            t_ts = time.time() - t0
            timings[AlignerName]['tangent_space'].append(t_ts)

    # ── LOSO 全流程计时（每个 aligner 只跑一次）──
    for AlignerName, AlignerCls in [('none', NoAlignment),
                                     ('riemann', RiemannianAlignment)]:
        t0 = time.time()
        res = evaluate_loso(
            subjs, aligner=AlignerCls(), bands='5band',
            channels='all', regressor='ridgecv',
            n_features=150, verbose=False,
            all_covs=all_covs, all_labels=all_labels,
        )
        t_full = time.time() - t0
        timings[AlignerName]['loso_full'].append(t_full)

    # 汇总报表
    print(f"\n  {'─' * 60}")
    print(f"  {'Stage':<25} {'None':>10} {'Riemannian':>12} {'Delta':>10}")
    print(f"  {'─' * 60}")
    for stage in ['mean_riemann', 'tangent_space', 'loso_full']:
        none_mean = np.mean(timings['none'][stage])
        riem_mean = np.mean(timings['riemann'][stage])
        delta = riem_mean - none_mean if stage == 'loso_full' else riem_mean - none_mean
        print(f"  {stage:<25} {none_mean:>8.2f}s {riem_mean:>10.2f}s {delta:>+8.2f}s")
    print(f"  {'─' * 60}")
    print(f"  {f'filter+cov (per subj)':<25} {filter_cov_per_subject:>8.2f}s")
    print(f"  {'Epochs/subject':<25} {n_epochs:>10d}")
    print(f"  {'Subjects timed':<25} {len(subjs):>10d}")

    results = {
        'filter_cov_per_subject_s': filter_cov_per_subject,
        'n_epochs': n_epochs,
        'n_subjects': len(subjs),
        'timings': {
            stage: {
                'none_mean_s': float(np.mean(timings['none'][stage])),
                'riemann_mean_s': float(np.mean(timings['riemann'][stage])),
                'none_std_s': float(np.std(timings['none'][stage])),
                'riemann_std_s': float(np.std(timings['riemann'][stage])),
            }
            for stage in ['mean_riemann', 'tangent_space', 'loso_full']
        },
    }

    save_partial(results, 'comp_cost')
    return results


# ════════════════════════════════════════════════════════════════
# GENERATE ALL FIGURES (from cache)
# ════════════════════════════════════════════════════════════════

def generate_all_figures(all_data):
    """Generate all paper figures from cached experiment data."""
    print("\nGenerating figures from cached results...")

    # ── Figure 1: Individual COR bar chart ──
    table1_data = all_data.get('table1')
    if table1_data and 'all' in table1_data:
        _plot_individual_cor(table1_data['all'])

    # ── Figure 2: Few-shot curve ──
    fewshot_data = all_data.get('fewshot')
    if fewshot_data:
        _plot_fewshot_curve(fewshot_data)



    print("  Done.")


def _plot_individual_cor(ch_data):
    """Plot per-subject COR comparison: None vs Euclidean vs Riemannian."""
    none_cor = np.array(ch_data.get('none', {}).get('cor', []))
    euclidean_cor = np.array(ch_data.get('euclidean', {}).get('cor', []))
    riemann_cor = np.array(ch_data.get('riemann', {}).get('cor', []))

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    x = np.arange(len(none_cor))
    w = 0.25
    ax.bar(x - w, none_cor, w, label='None', color='#ff9999', edgecolor='gray')
    ax.bar(x, euclidean_cor, w, label='Euclidean', color='#99ccff', edgecolor='gray')
    ax.bar(x + w, riemann_cor, w, label='Riemannian', color='#99ff99', edgecolor='gray')
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('COR', fontsize=12)
    ax.set_title('Per-Subject COR: None vs Euclidean vs Riemannian', fontsize=14)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    fig_path = os.path.join(DA_FIGURES_DIR, 'individual_cor.pdf')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {fig_path}")


def _plot_fewshot_curve(fewshot_data):
    """Plot few-shot calibration curve."""
    calib_sizes = sorted([int(k) for k in fewshot_data.keys()])
    means = [fewshot_data[k] if isinstance(k, int) else fewshot_data[str(k)] for k in calib_sizes]
    means = [m['mean'] for m in means]
    stds = [fewshot_data[k] if isinstance(k, int) else fewshot_data[str(k)] for k in calib_sizes]
    stds = [s['std'] for s in stds]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.errorbar(calib_sizes, means, yerr=stds, fmt='o-', capsize=5,
                color='steelblue', linewidth=2, markersize=8)
    ax.set_xlabel('Calibration Epochs', fontsize=12)
    ax.set_ylabel('COR', fontsize=12)
    ax.set_title('Few-Shot Calibration Curve', fontsize=14)
    ax.set_xscale('log')
    ax.set_xticks(calib_sizes)
    ax.set_xticklabels([str(k) for k in calib_sizes])
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(DA_FIGURES_DIR, 'fewshot_curve.pdf')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description='ric_da — Riemannian Domain Adaptation Experiments'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 5 subjects, 3 repeats')
    parser.add_argument('--full', action='store_true',
                        help='Full mode: 23 subjects')
    parser.add_argument('--only', type=str, default=None,
                        choices=['table1', 'fewshot',
                                 'ablation', 'band_importance', 'band_vs_global',
                                 'visualization', 'sota', 'enhanced_stats',
                                 'permutation', 'comp_cost'],
                        help='Run a specific experiment only')
    parser.add_argument('--no-figs', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--figs-only', action='store_true',
                        help='Generate figures only from cached results')
    parser.add_argument('--no-log', action='store_true',
                        help='Disable log file output')

    args = parser.parse_args()

    # 确定模式
    use_full = args.full or not args.quick
    if args.quick:
        use_full = False
    subjects = SEED_VIG_SUBJECTS if use_full else QUICK_SUBJECTS
    n_subj = len(subjects)

    # 日志
    if not args.no_log:
        log_path = setup_logging(DA_RESULTS_DIR)
        print(f"Log file saved to: {log_path}")

    print("=" * 70)
    print("ric_da — Riemannian Domain Adaptation Experiments")
    print(f"  Mode: {'FULL' if use_full else 'QUICK'} ({n_subj} subjects)")
    print(f"  Timestamp: {timestamp()}")
    print(f"  Data root: {SEED_VIG_ROOT if 'SEED_VIG_ROOT' in dir() else 'N/A'}")
    if not args.no_log:
        print(f"  Log file: {log_path}")
    print("=" * 70)

    if args.figs_only:
        # Load cached results and generate figures
        all_data = {}
        for exp_name in ['table1', 'fewshot']:
            candidates = [f for f in os.listdir(DA_RESULTS_DIR)
                          if f.startswith(f'results_{exp_name}') and f.endswith('.json')]
            if candidates:
                candidates.sort(reverse=True)
                fpath = os.path.join(DA_RESULTS_DIR, candidates[0])
                try:
                    with open(fpath, 'r') as f:
                        content = f.read()
                    if not content.strip():
                        print(f"  [Skip] Empty file: {candidates[0]}")
                        continue
                    loaded = json.loads(content)
                    if exp_name == 'table1' and 'results' in loaded:
                        all_data['table1'] = loaded['results']
                    elif exp_name == 'table1':
                        all_data['table1'] = loaded
                    elif exp_name == 'fewshot' and 'results' in loaded:
                        all_data['fewshot'] = loaded['results']
                    elif exp_name == 'fewshot':
                        all_data['fewshot'] = loaded
                except (json.JSONDecodeError, Exception) as e:
                    print(f"  [Skip] Error reading {candidates[0]}: {e}")
        generate_all_figures(all_data)
        if not args.no_log:
            teardown_logging()
        return

    if args.only:
        experiments = [args.only]
    else:
        experiments = ['table1', 'fewshot',
                       'ablation', 'band_importance', 'band_vs_global',
                       'visualization', 'sota', 'enhanced_stats',
                       'permutation', 'comp_cost']

    all_data = {}
    for exp_name in experiments:
        print(f"\n{'=' * 70}")
        print(f"Running: {exp_name}")
        print(f"{'=' * 70}")
        t0 = time.time()

        try:
            if exp_name == 'table1':
                all_data['table1'] = run_table1(subjects, quick=not use_full)
            elif exp_name == 'fewshot':
                all_data['fewshot'] = run_fewshot(subjects, quick=not use_full)
            elif exp_name == 'ablation':
                all_data['ablation'] = run_ablations(subjects, quick=not use_full)
            elif exp_name == 'band_importance':
                all_data['band_importance'] = run_band_importance(subjects, quick=not use_full)
            elif exp_name == 'band_vs_global':
                all_data['band_vs_global'] = run_band_vs_global(subjects, quick=not use_full)
            elif exp_name == 'visualization':
                all_data['visualization'] = run_visualization_analysis(subjects, quick=not use_full)
            elif exp_name == 'sota':
                all_data['sota'] = run_sota_head_to_head(subjects, quick=not use_full)
            elif exp_name == 'enhanced_stats':
                all_data['enhanced_stats'] = run_enhanced_stats(subjects, quick=not use_full)
            elif exp_name == 'permutation':
                all_data['permutation'] = run_permutation_test(subjects, quick=not use_full)
            elif exp_name == 'comp_cost':
                all_data['comp_cost'] = run_computational_cost(subjects, quick=not use_full)
            else:
                print(f"  [Skip] Unknown experiment: {exp_name}")

            elapsed = time.time() - t0
            print(f"\n  [{exp_name}] Total: {elapsed:.0f}s")

        except Exception as e:
            print(f"\n  [ERROR] {exp_name} failed: {e}")
            traceback.print_exc()

    # 保存全量结果
    if all_data:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        final = {
            'experiments': list(all_data.keys()),
            'timestamp': ts,
            'subjects': subjects,
            'data': all_data,
        }
        fpath = os.path.join(DA_RESULTS_DIR, f'all_experiments_{ts}.json')
        with open(fpath, 'w') as f:
            json.dump(final, f, indent=2)
        print(f"\n{'=' * 70}")
        print(f"All results saved to: {fpath}")
        print(f"{'=' * 70}")

    # 图表
    if not args.no_figs and all_data:
        generate_all_figures(all_data)

    print("\nDONE.")
    if not args.no_log:
        teardown_logging()


if __name__ == '__main__':
    main()