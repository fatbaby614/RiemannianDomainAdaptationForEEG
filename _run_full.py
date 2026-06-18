"""
全量实验 23 人：三种对齐器 × 三种通道配置
结果存入 ric_da/results/
"""
import sys, os, time, json, warnings
from datetime import datetime
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'SCAFBTSRegressor'))

from ric_da_core import (
    NoAlignment, EuclideanAlignment, RiemannianAlignment,
    precompute_all_subjects, evaluate_loso,
    format_results_table, paired_statistics
)
from config_da import (
    SEED_VIG_SUBJECTS, QUICK_SUBJECTS,
    DA_RESULTS_DIR, CHANNEL_OPTIONS
)

def run_one_config(subjects, channels, band='5band', use_cache=True):
    """跑一个通道配置的全部对齐器。"""
    print("\n" + "="*70)
    print(f"CONFIG: channels={channels}, band={band}")
    print("="*70)

    results = {}
    cache = None

    for name, Aligner in [
        ('none', NoAlignment),
        ('euclidean', EuclideanAlignment),
        ('riemann', RiemannianAlignment),
    ]:
        aligner = Aligner()
        if use_cache and cache is None:
            # 第一次运行时预计算
            cache = precompute_all_subjects(
                subjects, bands=band, channels=channels
            )

        t0 = time.time()
        res = evaluate_loso(
            subjects=subjects,
            aligner=aligner,
            bands=band,
            channels=channels,
            regressor='ridgecv',
            n_features=150,
            use_eog=False,
            verbose=False,
            all_covs=cache[0] if use_cache else None,
            all_labels=cache[1] if use_cache else None,
        )
        elapsed = time.time() - t0
        results[name] = res
        print(f"  {name.upper():>10}: COR={res['cor_mean']:.4f} ± "
              f"{res['cor_std']:.4f}, RMSE={res['rmse_mean']:.4f} "
              f"[{elapsed:.0f}s]")

    # 配对统计检验
    cor_dict = {name: results[name]['cor_all'] for name in ['none', 'euclidean', 'riemann']}
    stats = paired_statistics(cor_dict)
    print(f"\n  Statistical tests:")
    for pair, s in stats.items():
        sig = '***' if s['p_value'] < 0.001 else \
              '**'  if s['p_value'] < 0.01  else \
              '*'   if s['p_value'] < 0.05  else 'n.s.'
        print(f"    {pair}: t={s['t_stat']:.3f}, p={s['p_value']:.4f} {sig}, "
              f"d={s['cohens_d']:.3f}, better={s['better']}")

    return results, stats


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    subjects = SEED_VIG_SUBJECTS
    print(f"Full experiment: {len(subjects)} subjects")
    print(f"Timestamp: {ts}")

    all_output = {'metadata': {
        'timestamp': ts, 'n_subjects': len(subjects),
        'calibrator': 'n_jobs=1',
    }}

    for ch in CHANNEL_OPTIONS:
        t0 = time.time()
        results, stats = run_one_config(subjects, ch, band='5band')
        all_output[f'ch_{ch}'] = {
            'results': {k: {kk: vv for kk, vv in v.items()
                           if kk in ['cor_mean','cor_std','cor_all',
                                     'rmse_mean','rmse_std','rmse_all']}
                       for k, v in results.items()},
            'statistics': stats,
        }
        all_output['metadata'][f'time_{ch}'] = round(time.time() - t0)

    # 保存
    path = os.path.join(DA_RESULTS_DIR, f'full_results_{ts}.json')
    with open(path, 'w') as f:
        json.dump(all_output, f, indent=2)
    print(f"\nResults saved to: {path}")
    print("\nDONE")


if __name__ == '__main__':
    main()
