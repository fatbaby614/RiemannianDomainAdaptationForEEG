"""
ric_da_core.py — 核心对齐算法 + 跨被试评测引擎
===============================================

实现了三种对齐策略，用于消除 EEG 协方差矩阵在跨被试场景下的分布偏移：

1. NoAlignment       — 原始 LOSO（作为基线）
2. EuclideanAlignment — 欧氏空间白化对齐（基于样本协方差均值）
3. RiemannianAlignment — 黎曼流形平行传输对齐（基于黎曼均值）

参考文献:
    [1] Zanini et al., "Transfer Learning: A Riemannian Geometry Framework
        With Applications to BCI," IEEE TBME, 2018.
    [2] Rodrigues et al., "Riemannian Procrustes Analysis: Transfer Learning
        for Brain-Computer Interfaces," IEEE TBME, 2019.

用法:
    from ric_da_core import RiemannianAlignment, evaluate_loso
    aligner = RiemannianAlignment(metric='riemann')
    results = evaluate_loso(subjects, aligner=aligner, bands='5band')
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SCAFBTSRegressor'))

import numpy as np
import time
from datetime import datetime
from collections import defaultdict

from scipy.ndimage import uniform_filter1d
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann, mean_euclid
from pyriemann.utils.base import sqrtm, invsqrtm
from pyriemann.utils.distance import distance_riemann

from config import SEED_VIG_ROOT, RESULTS_DIR, BANDS_5, BANDS_8
from utils import cor, rmse


# ════════════════════════════════════════════════════════════════
# 对齐器基类 + 三种实现
# ════════════════════════════════════════════════════════════════

class BaseAlignment:
    """对齐器基类。所有对齐器实现相同的接口。"""
    name = 'base'

    def fit(self, covs_per_subject):
        """拟合对齐参数。

        Args:
            covs_per_subject: dict, {subject_label: (n_epochs, C, C) SPD array}
        """
        raise NotImplementedError

    def transform(self, covs, subject_label=None):
        """将协方差矩阵对齐到参考点。

        Args:
            covs: (n_epochs, C, C) SPD 矩阵
            subject_label: 被试标识（用于被试特定变换）

        Returns:
            covs_aligned: (n_epochs, C, C) 对齐后的 SPD 矩阵
        """
        raise NotImplementedError

    def fit_transform(self, covs_per_subject):
        """拟合后立即变换。"""
        self.fit(covs_per_subject)
        return {s: self.transform(c, s) for s, c in covs_per_subject.items()}

    def align_subject(self, covs):
        """对单个被试的协方差矩阵独立做对齐（无监督，不使用标签）。

        在 LOSO 中用于测试被试：用自己的协方差矩阵计算均值并做对齐，
        使其分布也以参考点为中心。这是无监督操作，不构成数据泄漏。

        Args:
            covs: (n_epochs, C, C) 单个被试的 SPD 矩阵

        Returns:
            covs_aligned: (n_epochs, C, C) 对齐后的 SPD 矩阵
        """
        raise NotImplementedError


class NoAlignment(BaseAlignment):
    """无对齐基线 — 保持原始协方差矩阵不变。"""
    name = 'none'

    def fit(self, covs_per_subject):
        pass

    def transform(self, covs, subject_label=None):
        return covs

    def align_subject(self, covs):
        return covs


class EuclideanAlignment(BaseAlignment):
    """欧氏空间白化对齐。

    对每个被试，计算其样本协方差矩阵的欧氏均值 M，
    然后用 M^{-1/2} * C * M^{-1/2} 做白化变换。
    """
    name = 'euclidean'

    def __init__(self, reference='identity'):
        self.reference = reference          # 'identity' 或 'grand_mean'
        self.transforms_ = {}               # {subject: inv_sqrt_mean}

    def fit(self, covs_per_subject):
        self.transforms_ = {}
        all_covs_list = []

        for subject, covs in covs_per_subject.items():
            # 计算该被试所有协方差矩阵的欧氏均值
            eucl_mean = np.mean(covs, axis=0)   # (C, C)
            # 确保对称正定
            eucl_mean = (eucl_mean + eucl_mean.T) / 2
            # M^{-1/2}
            inv_sqrt = invsqrtm(eucl_mean)
            self.transforms_[subject] = inv_sqrt
            all_covs_list.append(covs)

        if self.reference == 'grand_mean':
            # 计算全局参考点（所有被试池化后的均值）
            all_covs = np.concatenate(all_covs_list, axis=0)
            grand_mean = np.mean(all_covs, axis=0)
            grand_mean = (grand_mean + grand_mean.T) / 2
            # Zanini re-centering: W = G_grand^{1/2} · G_s^{-1/2}
            # Step 2 needs G_grand^{1/2} (not G_grand^{-1/2})
            self.grand_sqrt_ = sqrtm(grand_mean)
        else:
            self.grand_sqrt_ = np.eye(covs.shape[-1])

    def transform(self, covs, subject_label=None):
        """白化变换: C' = T^{-1/2} * C * T^{-1/2}

        For reference='grand_mean', the full transform is:
            C'' = G_grand^{1/2} * M_s^{-1/2} * C * M_s^{-1/2} * G_grand^{1/2}
        """
        if subject_label is not None and subject_label in self.transforms_:
            T = self.transforms_[subject_label]
        else:
            # fallback for unknown subjects: use grand_sqrt only
            G = self.grand_sqrt_
            return G @ covs @ G

        # Step 1: align to identity via M_s^{-1/2}
        covs_id = T @ covs @ T

        # Step 2: if grand_mean reference, transport identity → grand mean
        if self.reference == 'grand_mean':
            G = self.grand_sqrt_
            return G @ covs_id @ G
        else:
            return covs_id

    def align_subject(self, covs):
        """独立对齐单个被试：计算欧氏均值并白化。

        For reference='grand_mean', after aligning to identity,
        transport to the grand mean reference point.
        """
        eucl_mean = np.mean(covs, axis=0)
        eucl_mean = (eucl_mean + eucl_mean.T) / 2
        T = invsqrtm(eucl_mean)
        covs_id = T @ covs @ T

        if self.reference == 'grand_mean' and hasattr(self, 'grand_sqrt_'):
            G = self.grand_sqrt_
            return G @ covs_id @ G
        else:
            return covs_id


class RiemannianAlignment(BaseAlignment):
    """黎曼流形平行传输对齐 (Zanini et al. TBME 2018)。

    对每个被试，计算其 SPD 协方差矩阵的黎曼均值 G，
    然后用 G^{-1/2} * C * G^{-1/2} 将协方差矩阵平行传输到参考点。

    数学原理:
        在 SPD 流形上，G^{-1/2} * C * G^{-1/2} 将 C 从它在流形上的
        位置"移动"到以 G 为中心的对数映射像，实际效果是将所有被试的
        协方差分布对齐到同一个参考区域。
    """
    name = 'riemann'

    def __init__(self, reference='identity', max_iter=50, tol=1e-6):
        self.reference = reference          # 'identity' 或 'grand_mean'
        self.max_iter = max_iter
        self.tol = tol
        self.transforms_ = {}               # {subject: G^{-1/2}}

    def fit(self, covs_per_subject):
        self.transforms_ = {}
        all_covs_list = []

        for subject, covs in covs_per_subject.items():
            # 计算该被试的黎曼均值 G
            riem_mean = mean_riemann(covs, maxiter=self.max_iter, tol=self.tol)
            # G^{-1/2}
            inv_sqrt = invsqrtm(riem_mean)
            self.transforms_[subject] = inv_sqrt
            all_covs_list.append(covs)

        if self.reference == 'grand_mean':
            # 计算全数据集的黎曼均值作为参考点
            all_covs = np.concatenate(all_covs_list, axis=0)
            grand_riem = mean_riemann(all_covs, maxiter=self.max_iter, tol=self.tol)
            # Zanini re-centering: W = G_grand^{1/2} · G_s^{-1/2}
            # Step 2 needs G_grand^{1/2} (not G_grand^{-1/2})
            self.grand_sqrt_ = sqrtm(grand_riem)
        else:
            self.grand_sqrt_ = np.eye(covs.shape[-1])

    def transform(self, covs, subject_label=None):
        """平行传输: C' = G_s^{-1/2} * C * G_s^{-1/2}

        For reference='grand_mean', the full transform is:
            C'' = G_grand^{1/2} * G_s^{-1/2} * C * G_s^{-1/2} * G_grand^{1/2}
        i.e., first align to identity, then transport identity to grand mean.
        """
        if subject_label is not None and subject_label in self.transforms_:
            T = self.transforms_[subject_label]
        else:
            # fallback for unknown subjects: use grand_sqrt only
            G = self.grand_sqrt_
            return G @ covs @ G

        # Step 1: align to identity via G_s^{-1/2}
        covs_id = T @ covs @ T

        # Step 2: if grand_mean reference, transport identity → grand mean
        if self.reference == 'grand_mean':
            G = self.grand_sqrt_
            return G @ covs_id @ G
        else:
            return covs_id

    def align_subject(self, covs):
        """独立对齐单个被试：计算黎曼均值并做平行传输。

        For reference='grand_mean', after aligning to identity,
        transport to the grand mean reference point.
        """
        riem_mean = mean_riemann(covs, maxiter=self.max_iter, tol=self.tol)
        T = invsqrtm(riem_mean)
        covs_id = T @ covs @ T

        if self.reference == 'grand_mean' and hasattr(self, 'grand_sqrt_'):
            G = self.grand_sqrt_
            return G @ covs_id @ G
        else:
            return covs_id


class RiemannianProcrustesAlignment(BaseAlignment):
    """Riemannian Procrustes Analysis (RPA) — Rodrigues et al. TBME 2019.

    RPA 在平行传输的基础上增加了缩放因子，使得对齐后的矩阵迹等于维度。
    
    数学原理:
        C' = λ * G^{-1/2} * C * G^{-1/2}
        其中 λ = d / trace(G^{-1/2} * C * G^{-1/2})
        d 是矩阵维度（通道数）

    与标准黎曼对齐的区别:
        - 标准黎曼对齐: C' = G^{-1/2} * C * G^{-1/2}
        - RPA: C' = λ * G^{-1/2} * C * G^{-1/2}，增加了缩放因子

    参考文献:
        Rodrigues et al., "Riemannian Procrustes Analysis: Transfer Learning
        for Brain-Computer Interfaces," IEEE TBME, 2019.
    """
    name = 'rpa'

    def __init__(self, reference='identity', max_iter=50, tol=1e-6):
        self.reference = reference          # 'identity' 或 'grand_mean'
        self.max_iter = max_iter
        self.tol = tol
        self.transforms_ = {}               # {subject: G^{-1/2}}

    def fit(self, covs_per_subject):
        self.transforms_ = {}
        all_covs_list = []

        for subject, covs in covs_per_subject.items():
            # 计算该被试的黎曼均值 G
            riem_mean = mean_riemann(covs, maxiter=self.max_iter, tol=self.tol)
            # G^{-1/2}
            inv_sqrt = invsqrtm(riem_mean)
            self.transforms_[subject] = inv_sqrt
            all_covs_list.append(covs)

        if self.reference == 'grand_mean':
            # 计算全数据集的黎曼均值作为参考点
            all_covs = np.concatenate(all_covs_list, axis=0)
            grand_riem = mean_riemann(all_covs, maxiter=self.max_iter, tol=self.tol)
            # Zanini re-centering: W = G_grand^{1/2} · G_s^{-1/2}
            # Step 2 needs G_grand^{1/2} (not G_grand^{-1/2})
            self.grand_sqrt_ = sqrtm(grand_riem)
        else:
            self.grand_sqrt_ = np.eye(covs.shape[-1])

    def transform(self, covs, subject_label=None):
        """RPA变换: C' = λ * G_s^{-1/2} * C * G_s^{-1/2}, 其中 λ = d / trace(C_proj)

        For reference='grand_mean', the parallel transport step is:
            C'' = G_grand^{1/2} * G_s^{-1/2} * C * G_s^{-1/2} * G_grand^{1/2}
        then RPA scaling is applied.
        """
        if subject_label is not None and subject_label in self.transforms_:
            T = self.transforms_[subject_label]
        else:
            G = self.grand_sqrt_
            covs_proj = G @ covs @ G
            d = covs.shape[-1]
            traces = np.trace(covs_proj, axis1=-2, axis2=-1)
            lambdas = d / traces
            return covs_proj * lambdas[:, np.newaxis, np.newaxis]

        # Step 1: align to identity via G_s^{-1/2}
        covs_id = T @ covs @ T

        # Step 2: if grand_mean reference, transport identity → grand mean
        if self.reference == 'grand_mean':
            G = self.grand_sqrt_
            covs_proj = G @ covs_id @ G
        else:
            covs_proj = covs_id

        # Step 3: RPA scaling λ = d / trace(C_proj)
        d = covs.shape[-1]
        traces = np.trace(covs_proj, axis1=-2, axis2=-1)
        lambdas = d / traces
        covs_aligned = covs_proj * lambdas[:, np.newaxis, np.newaxis]

        return covs_aligned

    def align_subject(self, covs):
        """独立对齐单个被试：计算黎曼均值、平行传输 + RPA 缩放。

        For reference='grand_mean', after aligning to identity,
        transport to the grand mean reference point.
        """
        riem_mean = mean_riemann(covs, maxiter=self.max_iter, tol=self.tol)
        T = invsqrtm(riem_mean)
        covs_id = T @ covs @ T

        if self.reference == 'grand_mean' and hasattr(self, 'grand_sqrt_'):
            G = self.grand_sqrt_
            covs_proj = G @ covs_id @ G
        else:
            covs_proj = covs_id

        d = covs.shape[-1]
        traces = np.trace(covs_proj, axis1=-2, axis2=-1)
        lambdas = d / traces
        return covs_proj * lambdas[:, np.newaxis, np.newaxis]


def get_aligner(name, **kwargs):
    """对齐器工厂函数。"""
    registry = {
        'none': NoAlignment,
        'euclidean': EuclideanAlignment,
        'riemann': RiemannianAlignment,
        'rpa': RiemannianProcrustesAlignment,
    }
    if name not in registry:
        raise ValueError(f"未知对齐器: {name}，可选: {list(registry.keys())}")
    return registry[name](**kwargs)


# ════════════════════════════════════════════════════════════════
# 核心评测引擎 — 带对齐的 LOSO 回归
# ════════════════════════════════════════════════════════════════

def precompute_subject_covs(subject, bands='5band', channels='all',
                            estimator='oas'):
    """预计算单个被试所有频段的 SPD 协方差矩阵。

    Args:
        subject: 被试 ID（如 "1_20151124_noon"）
        bands: '5band' | '8band'
        channels: 'all' | 'temporal' | 'forehead' | list of indices
        estimator: 协方差估计器 'oas' | 'lwf' | 'scm' | 'cov' | 'corr'

    Returns:
        cov_dict: {(low, high): (n_epochs, C, C)} SPD 矩阵
        y: (n_epochs,) PERCLOS 标签
        n_channels: 使用的通道数
    """
    from data_loader import load_raw_eeg, load_perclos

    # 频段设置
    if bands == '5band':
        freq_bands = BANDS_5
    elif bands == '8band':
        freq_bands = BANDS_8
    else:
        raise ValueError(f"不支持的频段配置: {bands}")

    # 通道选择
    if channels == 'all':
        ch_idx = None
    elif channels == 'temporal':
        ch_idx = [0, 1, 2, 3, 4, 5]
    elif channels == 'forehead':
        ch_idx = [0, 1, 2, 3]
    elif isinstance(channels, (list, np.ndarray)):
        ch_idx = list(channels)
    else:
        raise ValueError(f"不支持的通道配置: {channels}")

    # 加载数据
    raw, sr = load_raw_eeg(SEED_VIG_ROOT, subject)
    y = load_perclos(SEED_VIG_ROOT, subject)

    if ch_idx is not None:
        raw = raw[:, ch_idx]

    raw_T = raw.T.astype(np.float64)          # (C, n_times)
    epoch_len = int(sr * 8)                    # 1600 points
    n_epochs = raw.shape[0] // epoch_len
    n_ch = raw.shape[1]

    # 逐频段滤波 + 协方差估计
    cov_dict = {}
    cov_est = Covariances(estimator=estimator)

    from sca_fbts_fast import apply_bandpass_filter

    for low, high in freq_bands:
        filtered = apply_bandpass_filter(raw_T, low, high, sr)
        epochs = np.array([
            filtered[:, i * epoch_len:(i + 1) * epoch_len]
            for i in range(n_epochs)
        ], dtype=np.float64)                     # (n_epochs, C, T)
        cov_dict[(low, high)] = cov_est.transform(epochs)

    return cov_dict, y, n_ch


def precompute_all_subjects(subjects, bands='5band', channels='all',
                            estimator='oas', use_eog=False, verbose=True):
    """全局预计算：一次滤波+协方差，供多个对齐器复用。

    Args:
        subjects: 被试 ID 列表
        bands: 频段配置
        channels: 通道配置
        estimator: 协方差估计器
        use_eog: 是否加载 EOG 特征
        verbose: 是否打印进度

    Returns:
        (all_covs, all_labels, all_eog) 三元组
    """
    from data_loader import load_eog_features

    print(f"\n[Precompute] {len(subjects)} subjects ({bands}, {channels})...")
    t0 = time.time()

    all_covs = {}
    all_labels = {}
    all_eog = {}

    for subj in subjects:
        cov_dict, y, n_ch = precompute_subject_covs(
            subj, bands=bands, channels=channels, estimator=estimator
        )
        all_covs[subj] = cov_dict
        all_labels[subj] = y

        if use_eog:
            eog = load_eog_features(SEED_VIG_ROOT, subj, method='features_table_ica')
            if eog.shape[0] != y.shape[0]:
                eog = eog[:y.shape[0]]
            all_eog[subj] = eog

    print(f"  Done. ({time.time() - t0:.1f}s)")
    return all_covs, all_labels, all_eog


def evaluate_loso(subjects, aligner=None, bands='5band', channels='all',
                  estimator='oas', regressor='svr', n_features=100,
                  n_select_ts=150, temporal_smoothing=True,
                  smoothing_window=3, use_eog=False, verbose=True,
                  all_covs=None, all_labels=None, all_eog=None):
    """带对齐的 LOSO 跨被试评测（单模态或融合 EOG）。

    流程:
        1. 预计算所有被试的 SPD 协方差矩阵（每个频段）
        2. 对每个频段: 拟合并应用对齐器，池化训练被试的数据
        3. 切空间投影（训练集拟合，测试集映射）
        4. 特征选择 + 标准化 + SVR/Ridge 回归

    支持传入预计算数据以跨对齐器复用:

        covs, labels, eog = precompute_all_subjects(subjects)
        for a in [NoAlignment(), EuclideanAlignment(), RiemannianAlignment()]:
            res = evaluate_loso(subjects, aligner=a, all_covs=covs,
                                all_labels=labels, all_eog=eog)

    Args:
        subjects: 被试 ID 列表
        aligner: BaseAlignment 子类实例（None 等价于 NoAlignment）
        bands: 频段配置
        channels: 通道配置
        estimator: 协方差估计器
        regressor: 'svr' | 'ridge' | 'ridgecv' | 'rfr'
        n_features: 特征选择数
        n_select_ts: 切空间特征选择数（融合模式）
        temporal_smoothing: 是否时序平滑
        smoothing_window: 平滑窗口
        use_eog: 是否融合 EOG
        verbose: 是否打印进度
        all_covs: 可选，预计算数据 {subject: {band: (n,C,C)}}
        all_labels: 可选，预计算标签 {subject: (n,)}
        all_eog: 可选，预计算 EOG 特征 {subject: (n,d)}

    Returns:
        dict: {cor_mean, cor_std, cor_all, rmse_mean, ...}
    """
    if aligner is None:
        aligner = NoAlignment()

    if bands == '5band':
        freq_bands = BANDS_5
    else:
        freq_bands = BANDS_8

    print(f"\n{'='*60}")
    print(f"LOSO + {aligner.name.upper()} Aligner")
    print(f"  Bands: {bands}, Channels: {channels}, Regressor: {regressor}")
    print(f"  EOG fusion: {use_eog}, n_features: {n_features}")
    print(f"{'='*60}")

    # ── Step 1: 预计算（或复用传入的数据） ──
    if all_covs is None:
        print(f"\n[1/4] Precomputing {len(subjects)} subjects...")
        all_covs = {}
        all_labels = {}
        all_eog = {} if use_eog else None
        t0 = time.time()
        for subj in subjects:
            cov_dict, y, n_ch = precompute_subject_covs(
                subj, bands=bands, channels=channels, estimator=estimator
            )
            all_covs[subj] = cov_dict
            all_labels[subj] = y
            if use_eog:
                from data_loader import load_eog_features
                eog = load_eog_features(SEED_VIG_ROOT, subj, method='features_table_ica')
                if eog.shape[0] != y.shape[0]:
                    eog = eog[:y.shape[0]]
                all_eog[subj] = eog
        print(f"  Done. ({time.time() - t0:.1f}s)")
    else:
        print(f"  [Cache] Using precomputed data for {len(subjects)} subjects.")
        if all_labels is None:
            raise ValueError("all_covs provided but all_labels missing")

    # ── Step 2: LOSO 迭代 ──
    print(f"\n[2/4] Running LOSO with {aligner.name.upper()} alignment...")
    cor_list, rmse_list, mae_list = [], [], []
    band_importance_list = []  # 记录每个 fold 的频段重要性

    for i, test_subj in enumerate(subjects):
        t_fold = time.time()

        # 训练被试
        train_subjs = [s for s in subjects if s != test_subj]

        # ── 每频段独立: 对齐 → 池化 → 切空间 ──
        # (按频段对齐是因为不同频段的 SPD 分布可能不同)
        features_list = []
        test_features_list = []

        for band in freq_bands:
            # 收集该频段所有训练被试的协方差矩阵
            band_covs_train = {}
            for s in train_subjs:
                band_covs_train[s] = all_covs[s][band]

            # 对齐（仅在训练集上拟合对齐器）
            band_covs_aligned = aligner.fit_transform(band_covs_train)

            # 池化对齐后的训练协方差
            train_covs_pooled = np.concatenate(
                [band_covs_aligned[s] for s in train_subjs], axis=0
            )
            train_labels_pooled = np.concatenate(
                [all_labels[s] for s in train_subjs]
            )

            # 切空间投影
            ts = TangentSpace(metric='riemann')
            ts_feats = ts.fit_transform(train_covs_pooled, train_labels_pooled)
            features_list.append(ts_feats)

            # ── 测试被试 ──
            test_covs = all_covs[test_subj][band]
            # 测试被试的对齐：用自己的协方差矩阵计算均值并做对齐
            # 这是无监督的（不需要标签），不构成数据泄漏
            # 参考 Zanini et al. 2018: 每个被试独立对齐到参考点
            test_covs_aligned = aligner.align_subject(test_covs)
            test_feats = ts.transform(test_covs_aligned)
            test_features_list.append(test_feats)

        # 拼接特征
        X_train = np.hstack(features_list)
        X_test = np.hstack(test_features_list)
        y_train = np.concatenate([all_labels[s] for s in train_subjs])
        y_test = all_labels[test_subj]

        # ── 融合 EOG 特征 ──
        if use_eog:
            eog_train = np.concatenate([all_eog[s] for s in train_subjs])
            eog_test = all_eog[test_subj]
            X_train = np.hstack([X_train, eog_train])
            X_test = np.hstack([X_test, eog_test])
            n_select = n_select_ts
        else:
            n_select = n_features

        # ── Step 3: 特征选择 ──
        band_importance = []  # 记录每个频段被选中的特征数
        # 保存每个频段原始切空间特征数（用于频段重要性统计）
        n_feats_per_band_orig = None
        if features_list:
            n_feats_per_band_orig = features_list[0].shape[1]
            
        if X_train.shape[1] > n_select:
            sel = SelectKBest(f_regression, k=n_select)
            X_train = sel.fit_transform(X_train, y_train)
            X_test = sel.transform(X_test)
            
            # 统计频段重要性
            if len(freq_bands) > 1 and n_feats_per_band_orig is not None:
                selected_mask = sel.get_support()
                for b_loop_i, band in enumerate(freq_bands):
                    start = b_loop_i * n_feats_per_band_orig
                    end = start + n_feats_per_band_orig
                    band_selected = selected_mask[start:end].sum()
                    band_importance.append((band, band_selected))

        # ── Step 4: 标准化 + 回归 ──
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if regressor == 'svr':
            # 切空间特征已投影到欧氏空间，线性 kernel 更适合
            # RBF kernel 在高维切空间特征上容易过拟合
            clf = SVR(kernel='linear', C=1.0)
        elif regressor == 'ridge':
            clf = Ridge(alpha=1.0)
        elif regressor == 'ridgecv':
            clf = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        else:
            raise ValueError(f"不支持的回归器: {regressor}")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if temporal_smoothing and len(y_pred) > 1:
            y_pred = uniform_filter1d(y_pred, size=smoothing_window)

        c = cor(y_test, y_pred)
        r = rmse(y_test, y_pred)
        cor_list.append(c)
        rmse_list.append(r)
        band_importance_list.append(band_importance)

        if verbose:
            print(f"  [{i+1}/{len(subjects)}] {test_subj}: "
                  f"COR={c:.4f}, RMSE={r:.4f} "
                  f"({time.time()-t_fold:.1f}s)")

    # ── 汇总频段重要性 ──
    band_importance_summary = {}
    if band_importance_list and band_importance_list[0]:
        for band, _ in band_importance_list[0]:
            band_key = f"{band[0]}-{band[1]}Hz"
            # band_importance_list 中的每个元素是列表 [(band, count), ...]
            counts = []
            for fold_imp in band_importance_list:
                for b, cnt in fold_imp:
                    if b == band:
                        counts.append(cnt)
                        break
                else:
                    counts.append(0)
            band_importance_summary[band_key] = {
                'mean': float(np.mean(counts)),
                'std': float(np.std(counts)),
                'all': [float(c) for c in counts]
            }

    # ── 汇总 ──
    results = {
        'cor_mean': float(np.mean(cor_list)),
        'cor_std': float(np.std(cor_list)),
        'cor_all': [float(x) for x in cor_list],
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std': float(np.std(rmse_list)),
        'rmse_all': [float(x) for x in rmse_list],
        'n_subjects': len(subjects),
        'aligner': aligner.name,
        'bands': bands,
        'channels': channels,
        'regressor': regressor,
        'estimator': estimator,
        'use_eog': use_eog,
        'n_features': n_features,
        'band_importance': band_importance_summary,
    }

    print(f"\n  === {aligner.name.upper()}: COR={results['cor_mean']:.4f}±"
          f"{results['cor_std']:.4f}, RMSE={results['rmse_mean']:.4f}±"
          f"{results['rmse_std']:.4f}")

    return results


def evaluate_loso_global_band(subjects, aligner=None, bands='5band', channels='all',
                              estimator='oas', regressor='ridgecv', n_features=150,
                              temporal_smoothing=True, smoothing_window=3,
                              use_eog=False, verbose=True,
                              all_covs=None, all_labels=None, all_eog=None):
    """全局频带对齐的 LOSO 评估 — 用于消融实验对比。

    与 evaluate_loso 的区别：
        - evaluate_loso: 对每个频段分别计算均值并对齐（分频带对齐）
        - evaluate_loso_global_band: 将所有频段特征拼接后计算一个全局均值并对齐

    用于验证"分频带对齐"设计决策的有效性。
    """
    if aligner is None:
        aligner = NoAlignment()

    if bands == '5band':
        freq_bands = BANDS_5
    else:
        freq_bands = BANDS_8

    print(f"\n{'='*60}")
    print(f"LOSO + {aligner.name.upper()} Aligner (GLOBAL BAND)")
    print(f"  Bands: {bands}, Channels: {channels}, Regressor: {regressor}")
    print(f"  EOG fusion: {use_eog}, n_features: {n_features}")
    print(f"{'='*60}")

    # ── Step 1: 预计算（或复用传入的数据） ──
    if all_covs is None:
        print(f"\n[1/4] Precomputing {len(subjects)} subjects...")
        all_covs = {}
        all_labels = {}
        all_eog = {} if use_eog else None
        t0 = time.time()
        for subj in subjects:
            cov_dict, y, n_ch = precompute_subject_covs(
                subj, bands=bands, channels=channels, estimator=estimator
            )
            all_covs[subj] = cov_dict
            all_labels[subj] = y
            if use_eog:
                from data_loader import load_eog_features
                eog = load_eog_features(SEED_VIG_ROOT, subj, method='features_table_ica')
                if eog.shape[0] != y.shape[0]:
                    eog = eog[:y.shape[0]]
                all_eog[subj] = eog
        print(f"  Done. ({time.time() - t0:.1f}s)")
    else:
        print(f"  [Cache] Using precomputed data for {len(subjects)} subjects.")
        if all_labels is None:
            raise ValueError("all_covs provided but all_labels missing")

    # ── Step 2: LOSO 迭代 ──
    print(f"\n[2/4] Running LOSO with {aligner.name.upper()} alignment (global band)...")
    cor_list, rmse_list = [], []

    for i, test_subj in enumerate(subjects):
        t_fold = time.time()

        # 训练被试
        train_subjs = [s for s in subjects if s != test_subj]

        # ── 全局频带对齐：先拼接所有频段，再计算全局均值对齐 ──
        # 收集所有训练被试的所有频段协方差矩阵
        all_train_covs = {}
        train_n_epochs = {}  # 记录每个被试的 epoch 数
        for s in train_subjs:
            # 将该被试所有频段的协方差矩阵沿 epoch 维度拼接
            covs_by_band = [all_covs[s][band] for band in freq_bands]
            n_epochs = len(all_labels[s])
            train_n_epochs[s] = n_epochs
            # 每个频段的协方差矩阵是 (n_epochs, C, C)
            # 我们需要将它们沿 epoch 维度堆叠，形成 (n_epochs * n_bands, C, C)
            all_train_covs[s] = np.concatenate(covs_by_band, axis=0)

        # 对齐（用全局拼接后的协方差计算均值）
        all_train_aligned = aligner.fit_transform(all_train_covs)

        # 切空间投影（对每个频段分别投影，保持特征结构）
        features_list = []
        test_features_list = []

        # 训练集：需要将对齐后的全局矩阵拆分成各个频段
        n_bands = len(freq_bands)

        for band_idx, band in enumerate(freq_bands):
            # 从全局对齐的训练数据中提取该频段
            # 注意：每个被试的 epoch 数不同，需要用各自的 epoch 数计算切分位置
            band_train_covs = []
            for s in train_subjs:
                n_epochs_s = train_n_epochs[s]
                start = band_idx * n_epochs_s
                end = (band_idx + 1) * n_epochs_s
                band_train_covs.append(all_train_aligned[s][start:end])
            
            train_covs_pooled = np.concatenate(band_train_covs, axis=0)
            train_labels_pooled = np.concatenate([all_labels[s] for s in train_subjs])

            # 切空间投影
            ts = TangentSpace(metric='riemann')
            ts_feats = ts.fit_transform(train_covs_pooled, train_labels_pooled)
            features_list.append(ts_feats)

            # ── 测试被试 ──
            # 测试被试也需要用全局对齐的方式处理
            # 先收集测试被试所有频段的协方差
            n_test_epochs = len(all_labels[test_subj])
            all_test_covs = np.concatenate([all_covs[test_subj][b] for b in freq_bands], axis=0)
            # 对齐：用测试被试自己的协方差独立计算均值并做对齐
            all_test_aligned = aligner.align_subject(all_test_covs)
            # 提取该频段（使用测试被试实际的 epoch 数）
            test_covs_aligned = all_test_aligned[band_idx * n_test_epochs:(band_idx + 1) * n_test_epochs]
            test_feats = ts.transform(test_covs_aligned)
            test_features_list.append(test_feats)

        # 拼接特征
        X_train = np.hstack(features_list)
        X_test = np.hstack(test_features_list)
        y_train = np.concatenate([all_labels[s] for s in train_subjs])
        y_test = all_labels[test_subj]

        # ── 融合 EOG 特征 ──
        if use_eog:
            eog_train = np.concatenate([all_eog[s] for s in train_subjs])
            eog_test = all_eog[test_subj]
            X_train = np.hstack([X_train, eog_train])
            X_test = np.hstack([X_test, eog_test])
            n_select = n_features  # 使用相同的特征数
        else:
            n_select = n_features

        # ── Step 3: 特征选择 ──
        if X_train.shape[1] > n_select:
            sel = SelectKBest(f_regression, k=n_select)
            X_train = sel.fit_transform(X_train, y_train)
            X_test = sel.transform(X_test)

        # ── Step 4: 标准化 + 回归 ──
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if regressor == 'svr':
            # 切空间特征已投影到欧氏空间，线性 kernel 更适合
            clf = SVR(kernel='linear', C=1.0)
        elif regressor == 'ridge':
            clf = Ridge(alpha=1.0)
        elif regressor == 'ridgecv':
            clf = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        else:
            raise ValueError(f"不支持的回归器: {regressor}")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if temporal_smoothing and len(y_pred) > 1:
            y_pred = uniform_filter1d(y_pred, size=smoothing_window)

        c = cor(y_test, y_pred)
        r = rmse(y_test, y_pred)
        cor_list.append(c)
        rmse_list.append(r)

        if verbose:
            print(f"  [{i+1}/{len(subjects)}] {test_subj}: "
                  f"COR={c:.4f}, RMSE={r:.4f} "
                  f"({time.time()-t_fold:.1f}s)")

    # ── 汇总 ──
    results = {
        'cor_mean': float(np.mean(cor_list)),
        'cor_std': float(np.std(cor_list)),
        'cor_all': [float(x) for x in cor_list],
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std': float(np.std(rmse_list)),
        'rmse_all': [float(x) for x in rmse_list],
        'n_subjects': len(subjects),
        'aligner': aligner.name,
        'bands': bands,
        'channels': channels,
        'regressor': regressor,
        'estimator': estimator,
        'use_eog': use_eog,
        'n_features': n_features,
        'alignment_mode': 'global_band',  # 标记为全局频带对齐
    }

    print(f"\n  === {aligner.name.upper()} (GLOBAL): COR={results['cor_mean']:.4f}±"
          f"{results['cor_std']:.4f}, RMSE={results['rmse_mean']:.4f}±"
          f"{results['rmse_std']:.4f}")

    return results


# ════════════════════════════════════════════════════════════════
# 统计检验
# ════════════════════════════════════════════════════════════════

def paired_statistics(cor_dict):
    """对多个对齐器的 LOSO 结果做配对统计检验。

    Args:
        cor_dict: {aligner_name: [cor_per_subject]}

    Returns:
        dict: {pair: {t_stat, p_value, cohens_d, wilcoxon_p}}
    """
    from scipy.stats import ttest_rel, wilcoxon

    names = list(cor_dict.keys())
    stats = {}
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            a, b = np.array(cor_dict[n1]), np.array(cor_dict[n2])
            t_stat, p_val = ttest_rel(a, b)
            # Cohen's d
            diff = a - b
            d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
            # Wilcoxon
            try:
                _, wp = wilcoxon(a, b)
            except ValueError:
                wp = 1.0

            stats[f'{n1}_vs_{n2}'] = {
                't_stat': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(d),
                'wilcoxon_p': float(wp),
                'mean_diff': float(np.mean(diff)),
                'better': n1 if np.mean(diff) > 0 else n2,
            }
    return stats


def format_results_table(all_results):
    """格式化结果用于打印。"""
    lines = []
    lines.append(f"{'Aligner':<15} {'COR':>8} {'±':>2} {'COR_std':<8} "
                 f"{'RMSE':>8} {'±':>2} {'RMSE_std':<8}")
    lines.append("-" * 65)
    for name, res in all_results.items():
        lines.append(f"{name:<15} {res['cor_mean']:>8.4f} {'±':>2} "
                     f"{res['cor_std']:<8.4f} {res['rmse_mean']:>8.4f} "
                     f"{'±':>2} {res['rmse_std']:<8.4f}")
    return "\n".join(lines)
