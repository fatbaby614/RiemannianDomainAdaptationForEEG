#!/usr/bin/env python3
"""
run_ric_da.py — 黎曼域自适应主实验
====================================

投稿目标: IEEE TBME

实验矩阵:

    Table 1: LOSO 对比 — 三种对齐策略 × 三种通道配置 × 两种频段
    Table 2: Few-shot 校准 — 校准量对 COR 的影响
    Table 3: 消融实验 — 参考点、对齐位置、协方差估计器
    Table 4: 跨数据集泛化 — SEED-VIG → DROZY

用法:
    # 完整实验
    python run_ric_da.py --n-jobs 4

    # 快速验证 (5 被试)
    python run_ric_da.py --quick

    # 只跑主实验结果 (Table 1)
    python run_ric_da.py --table 1

    # 只跑频段消融
    python run_ric_da.py --band-abl
"""

import sys, os, json, time, argparse
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SCAFBTSRegressor'))

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from ric_da_core import (
    NoAlignment, EuclideanAlignment, RiemannianAlignment,
    evaluate_loso, paired_statistics, precompute_subject_covs
)
from config_da import (
    SEED_VIG_SUBJECTS, QUICK_SUBJECTS, DA_RESULTS_DIR,
    ALIGNERS, BAND_OPTIONS, CHANNEL_OPTIONS, REGRESSOR_OPTIONS,
    FEWSHOT_SIZES, REFERENCE_OPTIONS
)

from utils import cor, rmse


def timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def run_experiment_table1(subjects, quick=False, n_jobs=1):
    """Table 1: LOSO 对齐对比 — 三种对齐 × 三种通道 × 两种频段。

    主实验: 229 subjects × 3 aligners × 3 channels × 2 bands = 18 组
    """
    print("\n" + "="*70)
    print("TABLE 1: LOSO with Riemannian Domain Adaptation")
    print("="*70)

    results = {}
    aligner_map = {
        'none': NoAlignment,
        'euclidean': EuclideanAlignment,
        'riemann': RiemannianAlignment,
    }

    for bands in BAND_OPTIONS:
        for ch in CHANNEL_OPTIONS:
            for aligner_name in ALIGNERS:
                key = f"{aligner_name}_{bands}_{ch}"
                print(f"\n--- {key} ---")

                aligner = aligner_map[aligner_name]()
                res = evaluate_loso(
                    subjects=subjects,
                    aligner=aligner,
                    bands=bands,
                    channels=ch,
                    regressor='ridgecv',
                    n_features=150,
                    use_eog=False,
                )
                results[key] = res
                results[key]['aligner'] = aligner_name
                results[key]['bands'] = bands
                results[key]['channels'] = ch

    # 汇总表格
    print("\n" + "="*70)
    print("TABLE 1 SUMMARY")
    print("="*70)
    header = f"{'Aligner':<12} {'Bands':<8} {'Channels':<10} {'COR':>8} {'±':>4} {'RMSE':>8}"
    print(header)
    print("-" * 54)
    for key, res in results.items():
        print(f"{res['aligner']:<12} {res['bands']:<8} {res['channels']:<10} "
              f"{res['cor_mean']:>8.4f} {'±':>4} {res['rmse_mean']:>8.4f}")

    # 统计检验
    print("\n\nStatistical tests (paired t-test):")
    for ch in CHANNEL_OPTIONS:
        print(f"\n  Channels = {ch}:")
        cor_dict = {}
        for aligner_name in ALIGNERS:
            key = f"{aligner_name}_5band_{ch}"
            if key in results:
                cor_dict[aligner_name] = results[key]['cor_all']
        stats = paired_statistics(cor_dict)
        for pair, s in stats.items():
            print(f"    {pair}: t={s['t_stat']:.3f}, p={s['p_value']:.4f}, "
                  f"d={s['cohens_d']:.3f}, better={s['better']}")

    return results


def run_experiment_table2_fewshot(subjects, quick=False):
    """Table 2: Few-shot 校准实验 — 不同校准量下的 LOSO 对齐效果。

    流程:
        1. 对每个训练被试，用全部 885 epochs 计算黎曼均值
        2. 对测试被试，随机选 n 个校准 epoch → 计算黎曼均值 → 对齐
        3. 用其余 epoch 测试
        4. 重复 5 次取平均
    """
    print("\n" + "="*70)
    print("TABLE 2: FEW-SHOT CALIBRATION")
    print("="*70)

    calib_sizes = [0, 1, 3, 5, 10, 20, 50, 100, 200] if not quick else [0, 5, 50]
    n_repeats = 5 if not quick else 2

    results = {}

    for calib_size in calib_sizes:
        print(f"\n--- Calibration size = {calib_size} ---")

        if calib_size == 0:
            # 零样本: 仅对齐训练集，不对齐测试被试
            aligner = RiemannianAlignment(reference='identity')
        else:
            aligner = RiemannianAlignment(reference='identity')

        # TODO: Implement few-shot evaluation
        # For now, call standard LOSO with the aligner
        # (a proper few-shot implementation would compute the target
        #  subject's Riemannian mean from calib_size calibration epochs)
        res = evaluate_loso(
            subjects=subjects,
            aligner=aligner,
            bands='5band',
            channels='all',
            regressor='ridgecv',
            n_features=150,
        )
        key = f"calib_{calib_size}"
        results[key] = res
        results[key]['calibration_size'] = calib_size

    return results


def run_experiment_table3_ablations(subjects, quick=False):
    """Table 3: 消融实验 — 参考点选择、对齐位置、协方差估计器。"""
    print("\n" + "="*70)
    print("TABLE 3: ABLATION EXPERIMENTS")
    print("="*70)

    results = {}

    # A: 参考点消融
    print("\n--- Ablation A: Reference point ---")
    for ref in REFERENCE_OPTIONS:
        aligner = RiemannianAlignment(reference=ref)
        res = evaluate_loso(
            subjects=subjects,
            aligner=aligner,
            bands='5band',
            channels='all',
            regressor='ridgecv',
            n_features=150,
        )
        results[f'riemann_ref_{ref}'] = res

    # B: 协方差估计器消融
    print("\n--- Ablation B: Covariance estimator ---")
    for est in ['oas', 'lwf', 'scm']:
        aligner = RiemannianAlignment(reference='identity')
        res = evaluate_loso(
            subjects=subjects,
            aligner=aligner,
            bands='5band',
            channels='all',
            estimator=est,
            regressor='ridgecv',
            n_features=150,
        )
        results[f'riemann_est_{est}'] = res

    # C: 回归器消融
    print("\n--- Ablation C: Regressor ---")
    for reg in ['svr', 'ridgecv']:
        for aligner_name in ['none', 'riemann']:
            if aligner_name == 'none':
                aligner = NoAlignment()
            else:
                aligner = RiemannianAlignment()
            res = evaluate_loso(
                subjects=subjects,
                aligner=aligner,
                bands='5band',
                channels='all',
                regressor=reg,
                n_features=150,
            )
            results[f'{aligner_name}_reg_{reg}'] = res

    return results


def run_experiment_table4_cross_dataset(quick=False):
    """Table 4: 跨数据集泛化 — SEED-VIG → DROZY。

    用全部 SEED-VIG 训练（+对齐），在 DROZY 上零样本测试。
    """
    print("\n" + "="*70)
    print("TABLE 4: CROSS-DATASET (SEED-VIG → DROZY)")
    print("="*70)

    # 这个实验需要加载 DROZY 数据并适配其通道数和 epoch 结构
    # 部分实现，此处用占位
    print("\n[待实现: DROZY 数据加载 + 跨数据集评估]")
    return {}


def save_results(all_results, label):
    """保存实验结果。"""
    ts = timestamp()
    path = os.path.join(DA_RESULTS_DIR, f'da_results_{label}_{ts}.json')
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description='Riemannian Domain Adaptation for Vigilance Estimation')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式 (5 被试)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='并行被试数 (当前串行)')
    parser.add_argument('--table', type=int, default=None,
                        help='只跑指定表 (1/2/3/4)')
    parser.add_argument('--band-abl', action='store_true',
                        help='额外频段消融')
    parser.add_argument('--save', action='store_true', default=True,
                        help='保存结果')
    args = parser.parse_args()

    subjects = QUICK_SUBJECTS if args.quick else SEED_VIG_SUBJECTS
    print(f"Subjects: {len(subjects)} ({'quick' if args.quick else 'full'})")

    all_results = {
        'metadata': {
            'timestamp': timestamp(),
            'n_subjects': len(subjects),
            'quick': args.quick,
        }
    }

    # Table 1: Main LOSO comparison
    if args.table is None or args.table == 1:
        t1 = run_experiment_table1(subjects, quick=args.quick, n_jobs=args.n_jobs)
        all_results['table1'] = t1

    # Table 2: Few-shot
    if args.table is None or args.table == 2:
        t2 = run_experiment_table2_fewshot(subjects, quick=args.quick)
        all_results['table2'] = t2

    # Table 3: Ablations
    if args.table is None or args.table == 3:
        t3 = run_experiment_table3_ablations(subjects, quick=args.quick)
        all_results['table3'] = t3

    # Table 4: Cross-dataset
    if args.table is None or args.table == 4:
        t4 = run_experiment_table4_cross_dataset(quick=args.quick)
        all_results['table4'] = t4

    # Save
    if args.save:
        save_results(all_results, 'all' if args.table is None else f'table{args.table}')

    print("\nDone.")


if __name__ == '__main__':
    main()
