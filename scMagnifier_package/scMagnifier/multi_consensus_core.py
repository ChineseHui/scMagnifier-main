#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from typing import List
import umap.umap_ as umap
import scipy.sparse as sp
import seaborn as sns
import matplotlib
plt.switch_backend('Agg')

# -----------------------------
# Helper functions
# -----------------------------
def collect_csv_files(csv_dirs: List[str]) -> List[str]:
    """仅收集以 cluster 开头的 CSV 文件"""
    files = []
    for d in csv_dirs:
        if os.path.isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.lower().startswith("cluster") and fname.lower().endswith(".csv"):
                    files.append(os.path.join(d, fname))
        elif os.path.isfile(d) and os.path.basename(d).lower().startswith("cluster") and d.lower().endswith(".csv"):
            files.append(d)
    return sorted(files)

def read_cluster_csv_as_series(path: str) -> pd.Series:
    """读取单个 cluster CSV 并返回 Series"""
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        df = pd.read_csv(path, header=None)
        df = df.set_index(0)
    if df.shape[1] > 1:
        ser = df.iloc[:, -1].astype(str)
    else:
        ser = df.iloc[:, 0].astype(str)
    ser.index = ser.index.astype(str)
    ser.name = os.path.splitext(os.path.basename(path))[0]
    return ser

# -----------------------------
# Distance-related
# -----------------------------
def get_pca_representation(adata, cell_order, n_pcs=20, method: str = "harmony"):
    temp = adata[cell_order, :].copy()

    # === 根据method选择优先的embedding ===
    if method == "harmony":
        if "X_pca_harmony" in temp.obsm and temp.obsm["X_pca_harmony"].shape[1] >= n_pcs:
            print("[INFO] Using Harmony-corrected PCA (X_pca_harmony)")
            return np.asarray(temp.obsm["X_pca_harmony"][:, :n_pcs])
    elif method == "scanorama":
        if "X_scanorama" in temp.obsm:
            print("[INFO] Using Scanorama embedding (X_scanorama, full dimensionality)")
            return np.asarray(temp.obsm["X_scanorama"])
    elif method == "scvi":
        if "X_scVI" in temp.obsm and temp.obsm["X_scVI"].shape[1] >= n_pcs:
            print("[INFO] Using scVI latent space (X_scVI)")
            return np.asarray(temp.obsm["X_scVI"][:, :n_pcs])

def minmax_scale_matrix(M):
    mn, mx = M.min(), M.max()
    if mx <= mn:
        return np.zeros_like(M)
    return (M - mn) / (mx - mn)

def compute_combined_distance(X_pca, X_onehot, alpha=0.5, normalize_each=True, onehot_metric="cosine"):
    """计算融合距离矩阵（原代码完全保留）"""
    print("[INFO] Computing embedding (Euclidean) distance matrix ...")
    D_pca = pairwise_distances(X_pca, metric="euclidean")

    print(f"[INFO] Computing One-hot ({onehot_metric}) distance matrix ...")
    D_onehot = pairwise_distances(X_onehot, metric=onehot_metric)

    if normalize_each:
        D_pca = minmax_scale_matrix(D_pca)
        D_onehot = minmax_scale_matrix(D_onehot)

    D_combined = alpha * D_pca + (1 - alpha) * D_onehot
    return D_combined

# -----------------------------
# Core Function（新增method参数，整合所有分支）
# -----------------------------
def multi_consensus(
    csv_dirs: List[str] = ["perturb_results/0p1", "perturb_results/neg0p1"],  
    h5ad_path: str = os.path.join("preprocessed_result", "preprocessed.h5ad"), 
    resolutions: List[float] = [1.5],
    seed: int = 42,
    outdir: str = "consensus_result",
    cluster_method: str = "leiden",
    alpha: float = 0.8,
    umap_n_neighbors: int = 10,
    n_pcs_for_dist: int = 20,
    onehot_metric: str = "cosine",
    label_key: str = "assigned_cluster",
    method: str = "harmony"  
) -> None:
    """
    多算法整合的单细胞共识聚类函数（支持Harmony/Scanorama/scVI embedding优先级）
    核心逻辑与原三个独立代码完全一致，仅通过method参数切换embedding优先级。

    参数说明：
    ----------
    csv_dirs : List[str], optional
        包含cluster*.csv文件的目录列表，默认值：["perturb_results/0p1", "perturb_results/neg0p1"]
    h5ad_path : str, optional
        输入预处理后的h5ad文件路径，默认值："preprocessed_result/preprocessed.h5ad"
    resolutions : List[float], optional
        聚类分辨率列表，默认值：[1.5]
    seed : int, optional
        随机种子，默认值：42
    outdir : str, optional
        输出目录，默认值："consensus_result"
    cluster_method : str, optional
        聚类方法（leiden/louvain），默认值："leiden"
    alpha : float, optional
        融合距离中PCA距离的权重（1-alpha为onehot距离权重），默认值：0.5
    umap_n_neighbors : int, optional
        UMAP和kNN的近邻数，默认值：15
    n_pcs_for_dist : int, optional
        PCA维度（仅对Harmony/scVI生效），默认值：20
    onehot_metric : str, optional
        onehot距离计算的度量方式，默认值："cosine"
    label_key : str, optional
        adata.obs中真实标签列名，默认值："assigned_cluster"
    method : str, optional
        embedding优先级算法，可选值：harmony/scanorama/scvi，默认值："harmony"
    """
    # 参数预处理与校验
    method = method.lower()
    valid_methods = ["harmony", "scanorama", "scvi"]
    if method not in valid_methods:
        raise ValueError(f"method必须是{valid_methods}之一，当前输入：{method}")
    
    
    os.makedirs(outdir, exist_ok=True)

    # 1. 加载数据
    csv_files = collect_csv_files(csv_dirs)
    if len(csv_files) == 0:
        raise ValueError("No cluster*.csv files found.")
    print(f"[INFO] Found {len(csv_files)} cluster CSV files (method={method}).")
    adata = sc.read_h5ad(h5ad_path)
    canonical_cells = adata.obs_names.astype(str).tolist()
    print(f"[INFO] Loaded h5ad with {len(canonical_cells)} cells.")

    # 2. 对齐细胞
    series_list = [read_cluster_csv_as_series(p) for p in csv_files]
    intersect_cells = set(canonical_cells)
    for s in series_list:
        intersect_cells &= set(s.index)
    cell_order = [c for c in canonical_cells if c in intersect_cells]
    print(f"[INFO] {len(cell_order)} cells after alignment.")

    # 3. 构建 One-hot
    onehot_parts = []
    for ser in series_list:
        ser2 = ser.reindex(cell_order).fillna("NA")
        onehot = pd.get_dummies(ser2, prefix=ser.name, dtype=np.uint8)
        onehot_parts.append(onehot)
    X_onehot = pd.concat(onehot_parts, axis=1).values
    print(f"[INFO] One-hot matrix shape: {X_onehot.shape}")

    # 4. 获取embedding
    X_pca = get_pca_representation(adata, cell_order, n_pcs=n_pcs_for_dist, method=method)
    print(f"[INFO] Embedding matrix shape: {X_pca.shape}")

    # 5. 计算融合距离并生成 UMAP
    D_combined = compute_combined_distance(
        X_pca, X_onehot, alpha=alpha, normalize_each=True, onehot_metric=onehot_metric
    )
    print("[INFO] Running new UMAP (precomputed distances) ...")
    reducer = umap.UMAP(n_neighbors=umap_n_neighbors, metric="precomputed", random_state=seed)
    embedding = reducer.fit_transform(D_combined)
    print(f"[INFO] New UMAP embedding shape: {embedding.shape}")

    # 创建 adata_custom
    adata_custom = adata[cell_order, :].copy()
    adata_custom.obsm["X_umap_custom"] = embedding
    adata_custom.obsp["combined_distance"] = D_combined

    # 保存 h5ad
    out_h5_path = os.path.join(outdir, "adata_with_rpcumap.h5ad")
    adata_custom.write_h5ad(out_h5_path)
    print(f"[SAVED] Custom UMAP + combined_distance saved to {out_h5_path}")

    # -----------------------------
    # 6. 基于 D_combined 进行 kNN + 聚类
    # -----------------------------
    print("\n[INFO] Building kNN graph from the combined distance matrix (precomputed)...")
    try:
        distances_graph = kneighbors_graph(D_combined, n_neighbors=umap_n_neighbors,
                                           mode="distance", metric="precomputed", include_self=False)
        connectivity_graph = kneighbors_graph(D_combined, n_neighbors=umap_n_neighbors,
                                              mode="connectivity", metric="precomputed", include_self=False)
    except Exception as e:
        print(f"[WARN] kneighbors_graph failed with error: {e}. Attempting fallback to single connectivity graph.")
        connectivity_graph = kneighbors_graph(D_combined, n_neighbors=umap_n_neighbors,
                                              mode="connectivity", metric="precomputed", include_self=False)
        distances_graph = connectivity_graph.copy()

    connectivity_graph = connectivity_graph.maximum(connectivity_graph.T)
    distances_graph = distances_graph.maximum(distances_graph.T)

    adata_cluster = adata_custom.copy()
    adata_cluster.obsp["connectivities"] = connectivity_graph.tocsr()
    adata_cluster.obsp["distances"] = distances_graph.tocsr()
    print(f"[INFO] kNN graph placed into adata_cluster.obsp")

    # -----------------------------
    # 聚类并绘图
    # -----------------------------
    def plot_umap(adata_tmp, color_key, png_path):
        fig = sc.pl.umap(adata_tmp, color=color_key, size=8, alpha=0.8, show=False, return_fig=True)
        fig.axes[0].set_title(fig.axes[0].get_title(), fontsize=10)
        legend = fig.axes[0].get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(10)
        save_kwargs = {"bbox_inches": 'tight'}
        fig.savefig(png_path, **save_kwargs)
        plt.close(fig)

    for res in resolutions:
        if cluster_method.lower() == "louvain":
            print(f"\n[INFO] Running louvain clustering at resolution={res}")
            sc.tl.louvain(adata_cluster, resolution=res, key_added=f"louvain_{res}")
            cluster_key = f"louvain_{res}"
            out_csv = os.path.join(outdir, f"louvain_res{res}.csv")
        elif cluster_method.lower() == "leiden":
            print(f"\n[INFO] Running leiden clustering at resolution={res}")
            sc.tl.leiden(adata_cluster, resolution=res, key_added=f"leiden_{res}")
            cluster_key = f"leiden_{res}"
            out_csv = os.path.join(outdir, f"leiden_res{res}.csv")
        else:
            raise ValueError(f"Unsupported cluster_method={cluster_method}")

        pd.DataFrame({
            "cell_id": cell_order,
            "cluster": adata_cluster.obs[cluster_key].values
        }).to_csv(out_csv, index=False)
        print(f"[SAVED] {out_csv}")

        # 原始 UMAP
        if "X_umap" in adata.obsm:
            temp_old = adata[cell_order, :].copy()
            temp_old.obs["cluster"] = adata_cluster.obs[cluster_key].astype(str).values
            plot_umap(temp_old, color_key="cluster",
                      png_path=os.path.join(outdir, f"{cluster_method}_UMAP_res{res}.png"))

        # 新 UMAP
        temp_new = adata_cluster.copy()
        temp_new.obs["cluster"] = adata_cluster.obs[cluster_key].astype(str).values
        temp_new.obsm["X_umap"] = temp_new.obsm["X_umap_custom"]
        plot_umap(temp_new, color_key="cluster",
                  png_path=os.path.join(outdir, f"{cluster_method}_rpcUMAP_res{res}.png"))

    # 真实标签
    if label_key in adata_custom.obs:
        temp_label = adata_custom.copy()
        temp_label.obsm["X_umap"] = temp_label.obsm["X_umap_custom"]
        plot_umap(temp_label, color_key=label_key,
                  png_path=os.path.join(outdir, f"rpcumap_realLabel_{label_key}.png"))

    print("\n[DONE] All resolutions processed (method={}).".format(method))
