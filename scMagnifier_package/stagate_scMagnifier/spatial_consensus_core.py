#!/usr/bin/env python3
"""
spatial_consensus.py

SC3S consensus clustering with STAGATE embedding (spatial-focused version)
- Aligns cells across cluster CSV files
- Computes combined distance (STAGATE + one-hot cluster)
- Generates custom UMAP from precomputed distance
- Runs Louvain/Leiden clustering at multiple resolutions
- Plots UMAP (old/new) with clusters and real labels
- Supports spatial plot with configurable spot size for spatial transcriptomics data

Package usage:
    from scMagnifier import spatial_consensus
    
    # 基础使用
    spatial_consensus()
    
    # 自定义参数使用（含空间点大小控制）
    spatial_consensus(
        h5ad_path="/path/to/preprocessed.h5ad",
        csv_dirs=["/path/to/perturb/0p1", "/path/to/perturb/neg0p1"],
        resolutions=[0.3],
        cluster_method="leiden",
        spot_size=30  # 自定义空间作图的点大小
    )
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from typing import List, Optional  # 新增Optional用于spot_size参数
import umap.umap_ as umap
import scipy.sparse as sp

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
# STAGATE-related (核心保留)
# -----------------------------
def get_stagate_representation(adata, cell_order, n_dims=20):
    """获取 STAGATE 表征（带严格异常校验）"""
    temp = adata[cell_order, :]

    if "STAGATE" not in temp.obsm:
        raise KeyError(
            "[ERROR] adata.obsm['STAGATE'] not found. "
            "Please ensure STAGATE has been run and embedding stored."
        )

    X = np.asarray(temp.obsm["STAGATE"])
    if X.shape[1] < n_dims:
        raise ValueError(
            f"[ERROR] STAGATE embedding has only {X.shape[1]} dims < requested {n_dims}"
        )

    print(f"[INFO] Using STAGATE embedding with first {n_dims} dimensions.")
    return X[:, :n_dims]

def minmax_scale_matrix(M):
    """Min-Max归一化矩阵"""
    mn, mx = M.min(), M.max()
    if mx <= mn:
        return np.zeros_like(M)
    return (M - mn) / (mx - mn)

def compute_combined_distance(X_embed, X_onehot, alpha=0.8, normalize_each=True, onehot_metric="cosine"):
    """计算融合距离矩阵（STAGATE + one-hot）"""
    print(f"[INFO] Computing STAGATE (Euclidean) distance matrix ...")
    D_embed = pairwise_distances(X_embed, metric="euclidean")

    print(f"[INFO] Computing One-hot ({onehot_metric}) distance matrix ...")
    D_onehot = pairwise_distances(X_onehot, metric=onehot_metric)

    if normalize_each:
        D_embed = minmax_scale_matrix(D_embed)
        D_onehot = minmax_scale_matrix(D_onehot)

    D_combined = alpha * D_embed + (1 - alpha) * D_onehot
    return D_combined

class ClusterColorMap:
    """聚类标签颜色映射（保留）"""
    def __init__(self, labels):
        self.labels_str = labels.astype(str)
        self.unique = sorted(np.unique(self.labels_str), key=lambda x: int(x))
        cmap = plt.get_cmap("tab20")
        self.label2color = {lab: cmap(i % 20) for i, lab in enumerate(self.unique)}

    def get_colors(self, labels=None):
        if labels is None:
            labels = self.labels_str
        labels = labels.astype(str)
        return np.array([self.label2color[l] for l in labels])

# -----------------------------
# Core Function (spatial_consensus 主函数)
# -----------------------------
def spatial_consensus(
    csv_dirs: List[str] = ["perturb_results/0p1", "perturb_results/neg0p1"], 
    h5ad_path: str = os.path.join("preprocessed_result", "preprocessed.h5ad"),  
    resolutions: List[float] = [0.3],
    seed: int = 42,
    outdir: str = "consensus_result",  # 调整为指定默认值
    cluster_method: str = "leiden",
    alpha: float = 0.8,
    umap_n_neighbors: int = 10,
    n_dims_for_dist: int = 20,
    onehot_metric: str = "cosine",
    label_key: str = "predictions",
    spot_size: Optional[int] = None,  # 改为：默认None，不传则用scanpy默认值
) -> None:
    """
    SC3S共识聚类核心函数（空间聚焦版，STAGATE嵌入）
    
    适配preprocess/perturb输出，强化空间作图的可配置性，保留所有原有核心功能。
    
    参数说明：
    ----------
    csv_dirs : List[str], optional
        包含cluster*.csv文件的目录列表（默认：["perturb_results/0p1", "perturb_results/neg0p1"]）
    h5ad_path : str, optional
        预处理后的h5ad文件路径（需包含STAGATE嵌入，默认：preprocessed_result/preprocessed.h5ad）
    resolutions : List[float], optional
        聚类分辨率列表（默认：[0.3]）
    seed : int, optional
        随机种子（保证结果可复现，默认：42）
    outdir : str, optional
        共识聚类结果输出目录（默认：consensus_result）
    cluster_method : str, optional
        聚类方法，支持"louvain"或"leiden"（默认：leiden）
    alpha : float, optional
        融合距离权重（STAGATE距离占比，默认：0.8）
    umap_n_neighbors : int, optional
        自定义UMAP的近邻数（默认：10）
    n_dims_for_dist : int, optional
        计算距离时使用的STAGATE维度数（默认：20）
    onehot_metric : str, optional
        one-hot聚类矩阵的距离度量（默认：cosine）
    label_key : str, optional
        真实细胞标签列名（默认：predictions），用于绘制真实标签UMAP
    spot_size : Optional[int], optional
        空间作图的点大小（默认：None，不传则使用scanpy默认值；传入数值则覆盖默认）
    
    返回值：
    ----------
    None
        结果保存到指定输出目录，包含：
        - 自定义UMAP的h5ad文件
        - 各分辨率聚类结果CSV
        - UMAP图（原始/自定义）
        - 空间分布图（可配置点大小）
        - 真实标签UMAP图
    """
    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)

    # 1. 加载并校验数据
    csv_files = collect_csv_files(csv_dirs)
    if len(csv_files) == 0:
        raise ValueError("No cluster*.csv files found.")
    print(f"[INFO] Found {len(csv_files)} cluster CSV files.")
    
    # 校验h5ad文件是否存在
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(
            f"[ERROR] h5ad file not found: {h5ad_path}\n"
            "Hint: Check if preprocess function has run successfully and path is correct."
        )
    adata = sc.read_h5ad(h5ad_path)
    canonical_cells = adata.obs_names.astype(str).tolist()
    print(f"[INFO] Loaded h5ad with {len(canonical_cells)} cells.")

    # 2. 对齐细胞
    series_list = [read_cluster_csv_as_series(p) for p in csv_files]
    intersect_cells = set(canonical_cells)
    for s in series_list:
        intersect_cells &= set(s.index)
    cell_order = [c for c in canonical_cells if c in intersect_cells]
    print(f"[INFO] {len(cell_order)} cells after alignment (intersection across all CSV files).")

    # 3. 构建 One-hot 矩阵
    onehot_parts = []
    for ser in series_list:
        ser2 = ser.reindex(cell_order).fillna("NA")
        onehot = pd.get_dummies(ser2, prefix=ser.name, dtype=np.uint8)
        onehot_parts.append(onehot)
    X_onehot = pd.concat(onehot_parts, axis=1).values
    print(f"[INFO] One-hot matrix shape: {X_onehot.shape}")

    # 4. 获取STAGATE表征（核心保留）
    X_stagate = get_stagate_representation(adata, cell_order, n_dims=n_dims_for_dist)
    print(f"[INFO] STAGATE matrix shape: {X_stagate.shape}")

    # 5. 计算融合距离并生成 UMAP
    D_combined = compute_combined_distance(
        X_stagate, X_onehot, alpha=alpha, normalize_each=True, onehot_metric=onehot_metric
    )
    print("[INFO] Running new UMAP (precomputed distances) ...")
    
    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        metric="precomputed",
        random_state=seed,
    )
    embedding = reducer.fit_transform(D_combined)
    print(f"[INFO] New UMAP embedding shape: {embedding.shape}")

    # 创建 adata_custom 并保存
    adata_custom = adata[cell_order, :].copy()
    adata_custom.obsm["X_umap_custom"] = embedding
    adata_custom.obsp["combined_distance"] = D_combined

    # 保存h5ad文件
    out_h5_path = os.path.join(outdir, "adata_with_rpcumap.h5ad")
    adata_custom.write_h5ad(out_h5_path)
    print(f"[SAVED] Custom UMAP + combined_distance saved to {out_h5_path}")

    # -----------------------------
    # 6. 基于 D_combined 构建鲁棒的kNN图
    # -----------------------------
    print("\n[INFO] Building kNN graph from the combined distance matrix (precomputed)...")
    try:
        distances_graph = kneighbors_graph(
            D_combined, 
            n_neighbors=umap_n_neighbors,
            mode="distance", 
            metric="precomputed", 
            include_self=False
        )
        connectivity_graph = kneighbors_graph(
            D_combined, 
            n_neighbors=umap_n_neighbors,
            mode="connectivity", 
            metric="precomputed", 
            include_self=False
        )
    except Exception as e:
        print(f"[WARN] kneighbors_graph failed with error: {e}. Attempting fallback to single connectivity graph.")
        connectivity_graph = kneighbors_graph(
            D_combined, 
            n_neighbors=umap_n_neighbors,
            mode="connectivity", 
            metric="precomputed", 
            include_self=False
        )
        distances_graph = connectivity_graph.copy()

    # 转为对称矩阵
    connectivity_graph = connectivity_graph.maximum(connectivity_graph.T)
    distances_graph = distances_graph.maximum(distances_graph.T)

    adata_cluster = adata_custom.copy()
    adata_cluster.obsp["connectivities"] = connectivity_graph.tocsr()
    adata_cluster.obsp["distances"] = distances_graph.tocsr()
    print(f"[INFO] kNN graph (connectivities + distances) placed into adata_cluster.obsp")

    # -----------------------------
    # 绘图函数（保留原样式）
    # -----------------------------
    def plot_umap(adata_tmp, color_key, png_path):
        fig = sc.pl.umap(
            adata_tmp,
            color=color_key,
            size=8,
            alpha=0.8,
            show=False,
            return_fig=True,
        )
        ax = fig.axes[0]
        # 保留原有的自定义样式
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_title(ax.get_title(), fontsize=10)
        # 调整图例字体
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(10)
        # 保存图片
        fig.savefig(png_path, bbox_inches="tight", dpi=600)
        plt.close(fig)

    # -----------------------------
    # 多分辨率聚类 + 输出（修改spot_size逻辑）
    # -----------------------------
    # 日志提示spot_size使用策略
    if spot_size is not None:
        print(f"[INFO] Will use custom spot size ({spot_size}) for spatial plots (if available).")
    else:
        print(f"[INFO] Will use scanpy default spot size for spatial plots (if available).")

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

        # 保存聚类结果CSV
        pd.DataFrame({
            "cell_id": cell_order,
            "cluster": adata_cluster.obs[cluster_key].values
        }).to_csv(out_csv, index=False)
        print(f"[SAVED] {out_csv}")

        # 原 UMAP
        if "X_umap" in adata.obsm:
            temp_old = adata[cell_order, :].copy()
            temp_old.obs["cluster"] = adata_cluster.obs[cluster_key].astype(str).values
            plot_umap(
                temp_old,
                "cluster",
                os.path.join(outdir, f"{cluster_method}_UMAP_res{res}.png")
            )

        # 新 UMAP
        temp_new = adata_cluster.copy()
        temp_new.obs["cluster"] = adata_cluster.obs[cluster_key].astype(str).values
        temp_new.obsm["X_umap"] = temp_new.obsm["X_umap_custom"]
        plot_umap(
            temp_new,
            "cluster",
            os.path.join(outdir, f"{cluster_method}_rpcUMAP_res{res}.png")
        )

        # 空间图（动态传递spot_size：输入则传，未输入则不传）
        if "spatial" in adata_custom.obsm:
            # 构建基础参数
            spatial_kwargs = {
                "adata": adata_cluster,
                "color": cluster_key,
                "show": False
            }
            # 仅当spot_size不为None时添加该参数
            if spot_size is not None:
                spatial_kwargs["spot_size"] = spot_size
                print(f"[INFO] Plotting spatial graph for {cluster_method} resolution={res} (custom spot_size={spot_size})")
            else:
                print(f"[INFO] Plotting spatial graph for {cluster_method} resolution={res} (using scanpy default spot_size)")
            
            # 调用绘图函数
            sc.pl.spatial(** spatial_kwargs)
            plt.savefig(
                os.path.join(outdir, f"spatial_{cluster_method}_res{res}.png"),
                dpi=600,
                bbox_inches="tight",
            )
            plt.close()

    # 真实标签UMAP
    if label_key in adata_custom.obs:
        print(f"\n[INFO] Plotting UMAP with real labels ({label_key})")
        temp_label = adata_custom.copy()
        temp_label.obsm["X_umap"] = temp_label.obsm["X_umap_custom"]
        plot_umap(
            temp_label,
            label_key,
            os.path.join(outdir, f"rpcumap_realLabel_{label_key}.png")
        )

    print("\n[DONE] All resolutions processed.")
