"""
快速验证脚本：一次预计算，三个对齐器复用（Step 1 优化验证）。
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'SCAFBTSRegressor'))

from ric_da_core import (
    NoAlignment, EuclideanAlignment, RiemannianAlignment,
    precompute_all_subjects, evaluate_loso,
    format_results_table, paired_statistics
)
from config_da import QUICK_SUBJECTS

print("=" * 60)
print(f"Quick Test: 3 aligners × {len(QUICK_SUBJECTS)} subjects (cached)")
print(f"Subjects: {QUICK_SUBJECTS}")
print("=" * 60)

# 一次预计算
print("\n--- Precompute all subjects (once) ---")
t0 = time.time()
all_covs, all_labels, _ = precompute_all_subjects(
    QUICK_SUBJECTS, bands='5band', channels='all'
)
print(f"  Precompute total: {time.time()-t0:.0f}s")

# 三个对齐器共享缓存
all_results = {}
for name, Aligner in [
    ('none', NoAlignment),
    ('euclidean', EuclideanAlignment),
    ('riemann', RiemannianAlignment)
]:
    aligner = Aligner()
    print(f"\n>>> {name.upper()} aligner (cached)")
    t0 = time.time()
    res = evaluate_loso(
        subjects=QUICK_SUBJECTS,
        aligner=aligner,
        bands='5band',
        channels='all',
        regressor='ridgecv',
        n_features=100,
        use_eog=False,
        verbose=True,
        all_covs=all_covs,
        all_labels=all_labels,
    )
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.0f}s")
    all_results[name] = res

# 汇总
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(format_results_table(all_results))

cor_dict = {name: res['cor_all'] for name, res in all_results.items()}
stats = paired_statistics(cor_dict)
print("\n\nPaired t-tests:")
for pair, s in stats.items():
    sig = '***' if s['p_value'] < 0.001 else \
          '**'  if s['p_value'] < 0.01  else \
          '*'   if s['p_value'] < 0.05  else 'n.s.'
    print(f"  {pair}: t={s['t_stat']:.3f}, p={s['p_value']:.4f} {sig}, d={s['cohens_d']:.3f}")
print("\nDONE")
