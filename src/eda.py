"""Class separability statistics and unsupervised analysis."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from .sar import FEATURE_NAMES


def eta_squared(values, labels):
    grand = values.mean()
    ss_between = sum((labels == c).sum() * (values[labels == c].mean() - grand) ** 2 for c in np.unique(labels))
    ss_total = ((values - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def run_eda(F, y, class_names, fig_dir, seed=0):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    res = {"anova": {}}
    for name in ("vv_mean", "vh_mean", "ratio_mean", "rvi_mean"):
        v = F[:, FEATURE_NAMES.index(name)]
        groups = [v[y == c] for c in np.unique(y)]
        f, p = stats.f_oneway(*groups)
        h, pk = stats.kruskal(*groups)
        res["anova"][name] = {"F": float(f), "p": float(p), "kruskal_H": float(h),
                              "kruskal_p": float(pk), "eta_squared": eta_squared(v, y)}
    Z = StandardScaler().fit_transform(F)
    pca = PCA(n_components=2, random_state=seed).fit(Z)
    P = pca.transform(Z)
    km = KMeans(n_clusters=len(np.unique(y)), n_init=10, random_state=seed).fit(Z)
    res["pca_explained_variance"] = [float(v) for v in pca.explained_variance_ratio_]
    res["kmeans_ari"] = float(adjusted_rand_score(y, km.labels_))
    res["kmeans_nmi"] = float(normalized_mutual_info_score(y, km.labels_))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, name in enumerate(("vv_mean", "vh_mean")):
        v = F[:, FEATURE_NAMES.index(name)]
        ax[j].boxplot([v[y == c] for c in np.unique(y)], showfliers=False)
        ax[j].set_xticks(range(1, len(np.unique(y)) + 1))
        ax[j].set_xticklabels([class_names[c] for c in np.unique(y)], rotation=45)
        ax[j].set_title(f"{name} (dB) by class")
    fig.tight_layout(); fig.savefig(fig_dir / "01_backscatter_by_class.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for c in np.unique(y):
        ax.scatter(P[y == c, 0], P[y == c, 1], s=8, alpha=0.6, label=class_names[c])
    ax.set_title("PCA of SAR features"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "02_pca_scatter.png", dpi=130); plt.close(fig)
    return res
