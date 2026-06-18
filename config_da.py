"""
config_da.py — Domain Adaptation 实验配置
=========================================
本模块所有生成物（结果JSON、图表、论文）均存放在 `ric_da/` 内部。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SCAFBTSRegressor'))

from config import SEED_VIG_ROOT, DROZY_ROOT
from data_loader import list_subjects

# ── ric_da 根目录（本文件所在目录） ──
RIC_DA_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 生成物子目录 ──
DA_RESULTS_DIR = os.path.join(RIC_DA_ROOT, 'results')    # 实验结果 JSON
DA_FIGURES_DIR = os.path.join(RIC_DA_ROOT, 'figures')    # 论文图表
DA_PAPER_DIR   = os.path.join(RIC_DA_ROOT, 'paper')       # 论文 LaTeX

for d in [DA_RESULTS_DIR, DA_FIGURES_DIR, DA_PAPER_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 被试列表 ──
SEED_VIG_SUBJECTS = list_subjects(SEED_VIG_ROOT)
QUICK_SUBJECTS = SEED_VIG_SUBJECTS[:5]

# ── 实验参数 ──
ALIGNERS = ['none', 'euclidean', 'riemann']
BAND_OPTIONS = ['5band', '8band']
CHANNEL_OPTIONS = ['all', 'temporal', 'forehead']
REGRESSOR_OPTIONS = ['svr', 'ridgecv']
FEATURE_RANGES = [50, 100, 150, 200]

# Few-shot 校准数量
FEWSHOT_SIZES = [0, 1, 3, 5, 10, 20, 50, 100, 200]

# 消融: 对齐参考点
REFERENCE_OPTIONS = ['identity', 'grand_mean']
